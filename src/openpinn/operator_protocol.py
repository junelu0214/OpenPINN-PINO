"""Operator evaluation hooks with model-native derivative paths.

This module is the product-side extension point for operator-learning
families. It records which contract a family satisfies and how the
documentation configurator can discover that family. DeepONet is implemented
in :mod:`openpinn.operators.deeponet`, FNO in :mod:`openpinn.operators.fno`,
and the M10 physics trainer in :mod:`openpinn.operators.pino`. DeepONet and
FNO are selectable model backbones through ``build_operator_config`` /
``train_operator``; training mode is a separate axis. ``solve`` remains the
PINN entry point.

Each selectable model uses its natural derivative. DeepONet implements
:class:`CoordinateOperator` and differentiates predictions with respect to
query coordinates. FNO implements :class:`GridOperator` and uses discrete
grid derivatives. Cross-resampling adapters are not part of the formal
training path.

Classic PINN training stays on ``PINNModel`` plus ``openpinn.autodiff``.
Those models are not reclassified as operator hooks in this milestone.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

from torch import Tensor

from .registry import normalize_component_name


DerivativeMode = Literal["autodiff", "grid"]
OperatorKind = Literal["coordinate", "grid"]

_DERIVATIVE_MODE_FOR_KIND: dict[OperatorKind, DerivativeMode] = {
    "coordinate": "autodiff",
    "grid": "grid",
}


@runtime_checkable
class CoordinateOperator(Protocol):
    """Coordinate-conditioned operator evaluated at continuous query points.

    Implementations do not need to inherit a base class. Satisfying this
    protocol provides the formal coordinate-operator evaluation contract.

    ``predict`` arguments:

    * ``coordinates`` has shape ``(n_points, n_dims)``. Callers that need a
      PDE residual set ``requires_grad=True`` and differentiate the output
      with :func:`openpinn.grad` or :func:`openpinn.hessian`.
    * ``condition`` is an optional encoding of an input function. It is ``None``
      for a plain coordinate network, shape ``(n_branch,)`` when one code is
      shared by every query, or shape ``(n_points, n_branch)`` when each
      query has its own code. This protocol does not build that code.

    The return value has shape ``(n_points, n_fields)``.
    """

    def predict(
        self, coordinates: Tensor, condition: Tensor | None = None
    ) -> Tensor:
        """Evaluate the field at ``coordinates``."""

        ...


# Deprecated public spelling retained for one minor version. Protocol aliases
# are intentionally warning-free because structural ``isinstance`` checks do
# not provide a useful call site for a deprecation warning.
CoordinateModel = CoordinateOperator


@runtime_checkable
class GridDerivative(Protocol):
    """Discrete derivative of a structured-grid field.

    ``field`` has shape ``(batch, channels, *spatial)``. ``axis`` indexes a
    spatial axis (``0`` is the first spatial dimension, not the batch or the
    channel). ``order`` is ``1`` for a first derivative and ``2`` for a
    second derivative. The result has the same shape as ``field``.

    Finite-difference and spectral stencils live in
    :mod:`openpinn.operators.grid_derivative`. PINO minimizes a grid residual
    built from this protocol.
    """

    def partial(self, field: Tensor, axis: int, order: int = 1) -> Tensor:
        """Differentiate ``field`` along one spatial axis."""

        ...


@runtime_checkable
class GridOperator(Protocol):
    """Map a structured-grid input field to an output field on that grid.

    ``inputs`` has shape ``(batch, in_channels, *spatial)`` with at least
    one spatial axis. ``predict(inputs, *, grid=...)`` returns
    ``(batch, out_channels, *spatial)`` and keeps the spatial shape.

    PDE residuals for this contract use a :class:`GridDerivative` on the
    output tensor. They do not call :func:`openpinn.grad` on coordinates,
    because the network input is a discretized field rather than a coordinate
    graph.
    """

    def predict(self, inputs: Tensor, *, grid: object) -> Tensor:
        """Evaluate the operator on one batched grid tensor."""

        ...


class OperatorHookRecord(TypedDict):
    """JSON record stored on the documentation configurator payload."""

    name: str
    kind: OperatorKind
    derivative_mode: DerivativeMode
    title: str
    summary: str
    implemented: bool
    selectable: bool
    milestone: str
    aliases: list[str]
    parameters: dict[str, Any]


@dataclass(frozen=True)
class OperatorHookSpec:
    """Metadata for one operator family.

    A spec is a registration record. ``implemented=False`` means no network
    module is wired up. ``selectable=True`` requires ``implemented=True`` and
    means ``build_operator_config`` / ``train_operator`` accept the name.
    Selectable operators stay out of the PINN method catalog and out of
    ``solve``. ``parameters`` is the configurator knob schema for that family.
    """

    name: str
    kind: OperatorKind
    derivative_mode: DerivativeMode
    title: str
    summary: str
    implemented: bool = False
    selectable: bool = False
    milestone: str = ""
    aliases: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in _DERIVATIVE_MODE_FOR_KIND:
            choices = ", ".join(_DERIVATIVE_MODE_FOR_KIND)
            raise ValueError(f"Operator kind must be one of: {choices}.")
        expected = _DERIVATIVE_MODE_FOR_KIND[self.kind]
        if self.derivative_mode != expected:
            raise ValueError(
                f"Operator kind {self.kind!r} requires derivative mode "
                f"{expected!r}, got {self.derivative_mode!r}."
            )
        if self.selectable and not self.implemented:
            raise ValueError(
                f"Operator hook {self.name!r} cannot be selectable before it is implemented."
            )
        if not str(self.milestone).strip() and self.milestone != "":
            raise ValueError("Operator milestone must be empty or a non-blank label.")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("Operator parameters must be a mapping.")


def _record(spec: OperatorHookSpec) -> OperatorHookRecord:
    return {
        "name": spec.name,
        "kind": spec.kind,
        "derivative_mode": spec.derivative_mode,
        "title": spec.title,
        "summary": spec.summary,
        "implemented": spec.implemented,
        "selectable": spec.selectable,
        "milestone": spec.milestone,
        "aliases": list(spec.aliases),
        "parameters": copy.deepcopy(dict(spec.parameters)),
    }


class OperatorRegistry:
    """Name-to-spec registry for operator protocol hooks.

    Registries are explicit objects. :func:`default_operator_registry` returns
    a new registry of the built-in reserved slots. Registering a project stub
    on that object does not change the next default registry.
    """

    def __init__(self) -> None:
        self._specs: dict[str, OperatorHookSpec] = {}
        self._aliases: dict[str, str] = {}
        self._order: list[str] = []

    def register(self, spec: OperatorHookSpec, *, replace: bool = False) -> None:
        """Register one hook.

        Duplicate names and aliases are rejected unless ``replace=True``.
        Replacing a canonical entry drops the previous aliases.
        """

        if not isinstance(spec, OperatorHookSpec):
            raise TypeError("Operator hooks must be OperatorHookSpec instances.")
        canonical = normalize_component_name(spec.name)
        alias_names = {canonical}
        for alias in spec.aliases:
            alias_names.add(normalize_component_name(alias))
        existing = self._specs.get(canonical)
        if existing is not None and not replace:
            raise ValueError(f"Operator hook {spec.name!r} is already registered.")
        conflicts = sorted(
            alias
            for alias in alias_names
            if alias in self._aliases and self._aliases[alias] != canonical
        )
        if conflicts:
            raise ValueError(
                "Operator hook alias conflict: " + ", ".join(conflicts) + "."
            )
        if existing is not None:
            for alias, target in tuple(self._aliases.items()):
                if target == canonical:
                    del self._aliases[alias]
        else:
            self._order.append(canonical)
        self._specs[canonical] = spec
        for alias in alias_names:
            self._aliases[alias] = canonical

    def contains(self, name: str) -> bool:
        """Return whether ``name`` is a registered hook or alias."""

        try:
            normalized = normalize_component_name(name)
        except ValueError:
            return False
        return normalized in self._aliases

    def resolve(self, name: str) -> OperatorHookSpec:
        """Return the spec for a hook name or alias."""

        normalized = normalize_component_name(name)
        try:
            canonical = self._aliases[normalized]
        except KeyError as exc:
            choices = ", ".join(self.available()) or "<none>"
            raise ValueError(
                f"Unknown operator hook {name!r}; available hooks: {choices}."
            ) from exc
        return self._specs[canonical]

    def available(self) -> tuple[str, ...]:
        """Return canonical hook names in registration order."""

        return tuple(spec.name for spec in self.specs())

    def specs(self) -> tuple[OperatorHookSpec, ...]:
        """Return registered hooks in registration order."""

        return tuple(self._specs[name] for name in self._order)

    def records(self) -> tuple[OperatorHookRecord, ...]:
        """Return JSON-ready copies of the registered hooks."""

        return tuple(_record(spec) for spec in self.specs())


def _reserved_hooks() -> tuple[OperatorHookSpec, ...]:
    """Built-in operator families and the deprecated PINO selection record."""

    from .operators.deeponet import DEEPONET_PARAMETER_SCHEMA
    from .operators.fno import FNO_PARAMETER_SCHEMA
    from .operators.pino import PINO_PARAMETER_SCHEMA

    return (
        OperatorHookSpec(
            name="deeponet",
            kind="coordinate",
            derivative_mode="autodiff",
            title="DeepONet",
            summary=(
                "Branch/trunk coordinate operator with query-point autodiff physics. Select it with "
                "build_operator_config('deeponet') and train_operator. "
                "solve() remains the PINN entry point."
            ),
            implemented=True,
            selectable=True,
            milestone="M1c",
            aliases=("deep_operator_network",),
            parameters=copy.deepcopy(DEEPONET_PARAMETER_SCHEMA),
        ),
        OperatorHookSpec(
            name="fno",
            kind="grid",
            derivative_mode="grid",
            title="Fourier neural operator",
            summary=(
                "Fourier-layer grid operator. Select it with "
                "build_operator_config('fno') and train_operator. "
                "PDE-side derivatives use GridDerivative, not coordinate autodiff. "
                "solve() remains the PINN entry point."
            ),
            implemented=True,
            selectable=True,
            milestone="M1d",
            aliases=("fourier_neural_operator",),
            parameters=copy.deepcopy(FNO_PARAMETER_SCHEMA),
        ),
        OperatorHookSpec(
            name="pino",
            kind="grid",
            derivative_mode="grid",
            title="Physics-informed neural operator",
            summary=(
                "Deprecated selection shim. Choose the FNO model and a hybrid or "
                "physics_only training mode instead. PINO remains an implementation "
                "helper, not a configurator model."
            ),
            implemented=True,
            selectable=False,
            milestone="M1e",
            aliases=("physics_informed_neural_operator",),
            parameters=copy.deepcopy(PINO_PARAMETER_SCHEMA),
        ),
    )


def default_operator_registry() -> OperatorRegistry:
    """Return a new registry of the built-in operator hooks.

    DeepONet and FNO are implemented and selectable. PINO is retained only as
    a deprecated API shim and implementation record.
    """

    registry = OperatorRegistry()
    for spec in _reserved_hooks():
        registry.register(spec)
    return registry


def list_operator_hooks() -> tuple[OperatorHookSpec, ...]:
    """List the built-in operator hooks for help pages and the configurator."""

    return default_operator_registry().specs()


def list_selectable_operators() -> tuple[OperatorHookSpec, ...]:
    """Return hooks that ``build_operator_config`` and ``train_operator`` accept."""

    return tuple(spec for spec in list_operator_hooks() if spec.selectable)


def operator_hook_records() -> tuple[OperatorHookRecord, ...]:
    """Return the built-in hooks as JSON-ready records."""

    return default_operator_registry().records()


def attach_operator_hooks(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``payload`` plus an ``operator_hooks`` list.

    The documentation configurator stores this list beside the existing
    PINN method families. PINN checkboxes read ``families`` only. A selectable
    hook is trained with ``build_operator_config`` and ``train_operator``,
    not with ``solve``. The input mapping is not mutated.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("Configurator payload must be a mapping.")
    updated = dict(payload)
    updated["operator_hooks"] = [dict(record) for record in operator_hook_records()]
    return updated
