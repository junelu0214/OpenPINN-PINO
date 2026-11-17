"""Boundary and initial conditions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Protocol, Union

import torch
from torch import Tensor, nn

ValueLike = Union[float, int, Tensor, Callable[[Tensor], Tensor]]
Selector = Callable[[Tensor], Tensor]
NormalLike = Union[Tensor, Sequence[float], Callable[[Tensor], Tensor]]


class ModelLike(Protocol):
    """Minimal model protocol used by conditions."""

    def __call__(self, points: Tensor) -> Tensor: ...


def evaluate_value(value: ValueLike, points: Tensor, *, like: Tensor | None = None) -> Tensor:
    """Evaluate a scalar/tensor/callable and normalize its batch shape."""

    result = value(points) if callable(value) else value
    reference = points if like is None else like
    if not isinstance(result, Tensor):
        result = reference.new_tensor(float(result))
    else:
        result = result.to(device=reference.device, dtype=reference.dtype)
    if result.ndim == 0:
        result = result.expand(reference.shape[0], 1)
    elif result.ndim == 1:
        result = result.reshape(-1, 1)
    if result.shape[0] not in (1, reference.shape[0]):
        raise ValueError("Condition value has an incompatible batch dimension.")
    if result.shape[0] == 1 and reference.shape[0] != 1:
        result = result.expand(reference.shape[0], *result.shape[1:])
    return result


@dataclass
class Condition:
    """Base class for pointwise conditions."""

    value: ValueLike
    name: str
    weight: float = 1.0
    selector: Selector | None = None

    sample_kind: str = "boundary"

    def select(self, points: Tensor) -> Tensor:
        """Optionally restrict a shared sample to this condition's region."""

        if self.selector is None:
            return points
        mask = self.selector(points)
        if not isinstance(mask, Tensor):
            mask = torch.as_tensor(mask, device=points.device)
        mask = mask.to(device=points.device, dtype=torch.bool).reshape(-1)
        if mask.numel() != points.shape[0]:
            raise ValueError("Condition selector must return one Boolean per point.")
        selected = points[mask]
        if selected.shape[0] == 0:
            raise ValueError(f"Condition {self.name!r} selected no sample points.")
        return selected

    def error(self, model: nn.Module, points: Tensor) -> Tensor:
        """Return prediction minus target at selected points."""

        selected = self.select(points)
        prediction = model(selected)
        target = evaluate_value(self.value, selected, like=prediction)
        if target.shape[1:] != prediction.shape[1:]:
            try:
                target = target.expand_as(prediction)
            except RuntimeError as exc:
                raise ValueError("Condition target and model output shapes are incompatible.") from exc
        return prediction - target


class DirichletBC(Condition):
    """Pointwise prescribed-value condition on the spatial boundary."""

    def __init__(
        self,
        value: ValueLike,
        boundary: Selector | None = None,
        *,
        name: str = "boundary",
        weight: float = 1.0,
    ) -> None:
        super().__init__(value=value, name=name, weight=weight, selector=boundary)
        self.sample_kind = "boundary"


def _evaluate_normal(normal: NormalLike, points: Tensor) -> Tensor:
    """Evaluate and normalize outward normals for derivative conditions."""

    result = normal(points) if callable(normal) else normal
    if not isinstance(result, Tensor):
        result = points.new_tensor(tuple(float(value) for value in result))
    else:
        result = result.to(device=points.device, dtype=points.dtype)
    if result.ndim == 1:
        result = result.reshape(1, -1).expand(points.shape[0], -1)
    if result.shape != points.shape:
        raise ValueError(
            "Boundary normals must have shape (coordinate_dim,) or (N, coordinate_dim)."
        )
    lengths = torch.linalg.vector_norm(result, dim=1, keepdim=True)
    if torch.any(lengths <= 0):
        raise ValueError("Boundary normals must be non-zero.")
    return result / lengths


class NeumannBC(Condition):
    """Prescribe the outward normal derivative ``du/dn`` on a boundary part."""

    def __init__(
        self,
        value: ValueLike,
        normal: NormalLike,
        boundary: Selector | None = None,
        *,
        name: str = "boundary",
        weight: float = 1.0,
    ) -> None:
        super().__init__(value=value, name=name, weight=weight, selector=boundary)
        self.normal = normal
        self.sample_kind = "boundary"

    def error(self, model: nn.Module, points: Tensor) -> Tensor:
        """Return outward normal derivative minus its prescribed value."""

        selected = self.select(points)
        if not selected.requires_grad:
            selected = selected.detach().requires_grad_(True)
        prediction = model(selected)
        if prediction.ndim != 2 or prediction.shape[1] != 1:
            raise ValueError("NeumannBC currently supports scalar model outputs only.")
        if prediction.requires_grad:
            derivatives = torch.autograd.grad(
                prediction,
                selected,
                grad_outputs=torch.ones_like(prediction),
                create_graph=True,
                retain_graph=True,
            )[0]
        else:
            derivatives = torch.zeros_like(selected)
        normal = _evaluate_normal(self.normal, selected)
        normal_derivative = (derivatives * normal).sum(dim=1, keepdim=True)
        target = evaluate_value(self.value, selected, like=normal_derivative)
        return normal_derivative - target


class RobinBC(NeumannBC):
    """Prescribe ``alpha*u + beta*du/dn = value`` on a boundary part."""

    def __init__(
        self,
        value: ValueLike,
        normal: NormalLike,
        boundary: Selector | None = None,
        *,
        alpha: ValueLike = 1.0,
        beta: ValueLike = 1.0,
        name: str = "boundary",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            value=value,
            normal=normal,
            boundary=boundary,
            name=name,
            weight=weight,
        )
        self.alpha = alpha
        self.beta = beta

    def error(self, model: nn.Module, points: Tensor) -> Tensor:
        """Return the Robin expression minus its prescribed value."""

        selected = self.select(points)
        if not selected.requires_grad:
            selected = selected.detach().requires_grad_(True)
        prediction = model(selected)
        if prediction.ndim != 2 or prediction.shape[1] != 1:
            raise ValueError("RobinBC currently supports scalar model outputs only.")
        if prediction.requires_grad:
            derivatives = torch.autograd.grad(
                prediction,
                selected,
                grad_outputs=torch.ones_like(prediction),
                create_graph=True,
                retain_graph=True,
            )[0]
        else:
            derivatives = torch.zeros_like(selected)
        normal = _evaluate_normal(self.normal, selected)
        normal_derivative = (derivatives * normal).sum(dim=1, keepdim=True)
        alpha = evaluate_value(self.alpha, selected, like=prediction)
        beta = evaluate_value(self.beta, selected, like=prediction)
        target = evaluate_value(self.value, selected, like=prediction)
        return alpha * prediction + beta * normal_derivative - target


class InitialCondition(Condition):
    """Pointwise prescribed value at the initial time."""

    def __init__(
        self,
        value: ValueLike,
        *,
        name: str = "initial",
        weight: float = 1.0,
        selector: Selector | None = None,
    ) -> None:
        super().__init__(value=value, name=name, weight=weight, selector=selector)
        self.sample_kind = "initial"
