"""Small automatic-differentiation helpers for pointwise PINN fields."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Literal

import torch
from torch import Tensor


@dataclass(frozen=True)
class _StructuredDerivativeContext:
    outputs: Tensor
    inputs: Tensor
    gradient: Callable[[Tensor], Tensor]
    second: Callable[[Tensor, int, int], Tensor]


_STRUCTURED_CONTEXT: ContextVar[_StructuredDerivativeContext | None] = ContextVar(
    "openpinn_structured_derivative_context", default=None
)
_ACTIVE_DERIVATIVE_BACKEND: ContextVar[Any | None] = ContextVar(
    "openpinn_active_derivative_backend", default=None
)
_INSIDE_DERIVATIVE_BACKEND: ContextVar[bool] = ContextVar(
    "openpinn_inside_derivative_backend", default=False
)


@contextmanager
def use_derivative_backend(backend: Any | None) -> Iterator[None]:
    """Route coordinate derivative calls through one registered backend."""

    token = _ACTIVE_DERIVATIVE_BACKEND.set(backend)
    try:
        yield
    finally:
        _ACTIVE_DERIVATIVE_BACKEND.reset(token)


def _structured_derivatives_active(outputs: Tensor, inputs: Tensor) -> bool:
    context = _STRUCTURED_CONTEXT.get()
    return bool(
        context is not None
        and inputs is context.inputs
        and outputs.shape == context.outputs.shape
    )


@contextmanager
def structured_derivatives(
    outputs: Tensor,
    inputs: Tensor,
    *,
    gradient: Callable[[Tensor], Tensor],
    second: Callable[[Tensor, int, int], Tensor],
) -> Iterator[None]:
    """Route ``grad``/``hessian`` for one flattened structured-grid field.

    This narrow bridge lets a coordinate-form equation callable be evaluated by
    a grid operator without pretending that grid values are coordinate-autodiff
    values.  The supplied callbacks must use a real ``GridDerivative`` backend.
    Calls for every other tensor keep the ordinary autograd behavior.
    """

    token = _STRUCTURED_CONTEXT.set(
        _StructuredDerivativeContext(outputs, inputs, gradient, second)
    )
    try:
        yield
    finally:
        _STRUCTURED_CONTEXT.reset(token)


def _scalar_component(outputs: Tensor, component: int | None) -> Tensor:
    if outputs.ndim == 1:
        if component not in (None, 0):
            raise IndexError("A one-dimensional output has only component 0.")
        return outputs.reshape(-1, 1)
    if outputs.ndim != 2:
        raise ValueError("outputs must have shape (N,) or (N, C).")
    if component is None:
        if outputs.shape[1] != 1:
            raise ValueError("component is required for multi-output fields.")
        component = 0
    if not 0 <= component < outputs.shape[1]:
        raise IndexError(f"component {component} is out of range for {outputs.shape[1]} outputs.")
    return outputs[:, component : component + 1]


def grad(
    outputs: Tensor,
    inputs: Tensor,
    component: int | None = None,
    *,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> Tensor:
    """Return pointwise gradients of one output component with respect to inputs.

    For the usual PINN batch, each output row depends only on the corresponding
    input row.  Differentiating the component sum therefore produces an
    ``(N, D)`` tensor of per-point coordinate derivatives.
    """

    backend = _ACTIVE_DERIVATIVE_BACKEND.get()
    if backend is not None and not _INSIDE_DERIVATIVE_BACKEND.get():
        method = getattr(backend, "grad", None)
        if not callable(method):
            raise TypeError("The selected derivative backend has no coordinate grad().")
        token = _INSIDE_DERIVATIVE_BACKEND.set(True)
        try:
            return method(
                outputs,
                inputs,
                component,
                create_graph=create_graph,
                retain_graph=retain_graph,
            )
        finally:
            _INSIDE_DERIVATIVE_BACKEND.reset(token)
    context = _STRUCTURED_CONTEXT.get()
    if _structured_derivatives_active(outputs, inputs):
        assert context is not None
        if component not in (None, 0):
            raise IndexError("Structured scalar fields only have component 0.")
        return context.gradient(outputs)
    if not inputs.requires_grad:
        raise ValueError("inputs.requires_grad must be True before calling grad().")
    scalar = _scalar_component(outputs, component)
    if not scalar.requires_grad:
        return inputs * 0.0
    derivatives = torch.autograd.grad(
        scalar,
        inputs,
        grad_outputs=torch.ones_like(scalar),
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True,
    )[0]
    if derivatives is None:
        return inputs * 0.0
    return derivatives


def hessian(
    outputs: Tensor,
    inputs: Tensor,
    component: int | None = None,
    i: int | None = 0,
    j: int | None = None,
    *,
    create_graph: bool = True,
) -> Tensor:
    """Return selected second derivatives or a full pointwise Hessian.

    ``hessian(y, x, i=0, j=0)`` returns shape ``(N, 1)``.  Passing
    ``i=None, j=None`` returns all coordinates with shape ``(N, D, D)``.
    If ``j`` is omitted for a selected ``i``, the diagonal derivative ``j=i``
    is returned.
    """

    backend = _ACTIVE_DERIVATIVE_BACKEND.get()
    if backend is not None and not _INSIDE_DERIVATIVE_BACKEND.get():
        method = getattr(backend, "hessian", None)
        if not callable(method):
            raise TypeError("The selected derivative backend has no coordinate hessian().")
        token = _INSIDE_DERIVATIVE_BACKEND.set(True)
        try:
            return method(
                outputs,
                inputs,
                component,
                i=i,
                j=j,
                create_graph=create_graph,
            )
        finally:
            _INSIDE_DERIVATIVE_BACKEND.reset(token)

    context = _STRUCTURED_CONTEXT.get()
    if _structured_derivatives_active(outputs, inputs):
        assert context is not None
        if component not in (None, 0):
            raise IndexError("Structured scalar fields only have component 0.")
        dimension = inputs.shape[1]
        if i is None:
            if j is not None:
                raise ValueError("j must also be None when requesting the full Hessian.")
            rows = [
                torch.cat([context.second(outputs, row, col) for col in range(dimension)], dim=1)
                for row in range(dimension)
            ]
            return torch.stack(rows, dim=1)
        return context.second(outputs, i, i if j is None else j)

    first = grad(outputs, inputs, component, create_graph=True, retain_graph=True)
    dimension = inputs.shape[1]

    def second(first_index: int, second_index: int) -> Tensor:
        if not 0 <= first_index < dimension or not 0 <= second_index < dimension:
            raise IndexError("Hessian coordinate index is out of range.")
        selected = first[:, first_index : first_index + 1]
        if not selected.requires_grad:
            # A constant first derivative has an exact zero second derivative.
            return inputs[:, second_index : second_index + 1] * 0.0
        values = torch.autograd.grad(
            selected,
            inputs,
            grad_outputs=torch.ones_like(selected),
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if values is None:
            return inputs[:, second_index : second_index + 1] * 0.0
        return values[:, second_index : second_index + 1]

    if i is None:
        if j is not None:
            raise ValueError("j must also be None when requesting the full Hessian.")
        rows = [torch.cat([second(row, col) for col in range(dimension)], dim=1) for row in range(dimension)]
        return torch.stack(rows, dim=1)

    return second(i, i if j is None else j)


class AutodiffDerivative:
    """Coordinate autodiff backend wrapping :func:`grad` and :func:`hessian`.

    This is the first-class ``derivative`` registry object for PINN
    collocation.  Grid operators use :class:`~openpinn.GridDerivative`
    instead (``partial`` on structured-grid tensors).
    """

    def mode(self) -> Literal["autodiff"]:
        """Return ``"autodiff"``."""

        return "autodiff"

    def grad(
        self,
        outputs: Tensor,
        inputs: Tensor,
        component: int | None = None,
        *,
        create_graph: bool = True,
        retain_graph: bool = True,
    ) -> Tensor:
        """Differentiate one output component with respect to coordinates."""

        return grad(
            outputs,
            inputs,
            component,
            create_graph=create_graph,
            retain_graph=retain_graph,
        )

    def hessian(
        self,
        outputs: Tensor,
        inputs: Tensor,
        component: int | None = None,
        i: int | None = 0,
        j: int | None = None,
        *,
        create_graph: bool = True,
    ) -> Tensor:
        """Return selected second derivatives or a full pointwise Hessian."""

        return hessian(
            outputs,
            inputs,
            component,
            i=i,
            j=j,
            create_graph=create_graph,
        )
