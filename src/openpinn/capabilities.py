"""Unified model, differentiation, and backend capability resolution.

The numerical implementations remain in their focused registries: pointwise
models are constructed by :class:`ComponentRegistry`, while DeepONet and FNO
retain their operator builders.  This module provides the shared architectural
view described by the OpenPINN/PINO specifications: every model belongs to the
model axis, every derivative belongs to the differentiation axis, and a
validated execution plan selects the backend without pretending that all
models consume the same tensor layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .registry import normalize_component_name


BackendKind = Literal["pinn", "operator", "custom"]
EvaluationKind = Literal["pointwise", "coordinate_operator", "grid_operator"]
TrainingMode = Literal["data_only", "physics_only", "hybrid"]
DerivativeDomain = Literal["coordinate", "structured_grid"]


@dataclass(frozen=True)
class ModelCapability:
    """Runtime-independent capability record for one model family."""

    name: str
    backend: BackendKind
    evaluation_kind: EvaluationKind
    derivative_names: frozenset[str]
    training_modes: frozenset[str]
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class DerivativeCapability:
    """Capability record for a registered differentiation path."""

    name: str
    domain: DerivativeDomain
    maximum_order: int
    aliases: tuple[str, ...] = ()
    requires_periodic_grid: bool = False


@dataclass(frozen=True)
class ExecutionPlan:
    """Validated model × derivative × backend selection."""

    backend: BackendKind
    model: str
    evaluation_kind: EvaluationKind
    training_mode: str
    differentiation: str | None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "backend": self.backend,
            "model": self.model,
            "evaluation_kind": self.evaluation_kind,
            "training_mode": self.training_mode,
            "differentiation": self.differentiation,
        }


class CapabilityRegistry:
    """Shared metadata view over pointwise and operator model families.

    This registry intentionally stores capabilities rather than factories.
    Construction remains delegated to the existing component/operator
    registries, so adding the architectural layer cannot route an FNO through
    the pointwise ``Trainer`` by accident.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelCapability] = {}
        self._model_aliases: dict[str, str] = {}
        self._derivatives: dict[str, DerivativeCapability] = {}
        self._derivative_aliases: dict[str, str] = {}

    @staticmethod
    def _register_aliases(
        alias_map: dict[str, str], name: str, aliases: tuple[str, ...]
    ) -> str:
        canonical = normalize_component_name(name)
        for value in (name, *aliases):
            alias = normalize_component_name(value)
            previous = alias_map.get(alias)
            if previous is not None and previous != canonical:
                raise ValueError(f"Capability alias {value!r} is already registered.")
            alias_map[alias] = canonical
        return canonical

    def register_model(self, capability: ModelCapability) -> None:
        if not isinstance(capability, ModelCapability):
            raise TypeError("model capability must be a ModelCapability")
        canonical = self._register_aliases(
            self._model_aliases, capability.name, capability.aliases
        )
        if canonical in self._models:
            raise ValueError(
                f"Model capability {capability.name!r} is already registered."
            )
        self._models[canonical] = capability

    def register_derivative(self, capability: DerivativeCapability) -> None:
        if not isinstance(capability, DerivativeCapability):
            raise TypeError("derivative capability must be a DerivativeCapability")
        canonical = self._register_aliases(
            self._derivative_aliases, capability.name, capability.aliases
        )
        if canonical in self._derivatives:
            raise ValueError(
                f"Derivative capability {capability.name!r} is already registered."
            )
        self._derivatives[canonical] = capability

    def model(self, name: str) -> ModelCapability:
        normalized = normalize_component_name(name)
        try:
            return self._models[self._model_aliases[normalized]]
        except KeyError as exc:
            choices = ", ".join(item.name for item in self._models.values())
            raise ValueError(
                f"Unknown model capability {name!r}; available models: {choices}."
            ) from exc

    def derivative(self, name: str) -> DerivativeCapability:
        normalized = normalize_component_name(name)
        try:
            return self._derivatives[self._derivative_aliases[normalized]]
        except KeyError as exc:
            choices = ", ".join(item.name for item in self._derivatives.values())
            raise ValueError(
                f"Unknown differentiation capability {name!r}; "
                f"available paths: {choices}."
            ) from exc

    def models(self) -> tuple[ModelCapability, ...]:
        return tuple(self._models.values())

    def derivatives(self) -> tuple[DerivativeCapability, ...]:
        return tuple(self._derivatives.values())

    def resolve(
        self,
        model: str,
        *,
        backend: str | None = None,
        training_mode: str = "physics_only",
        differentiation: str | None = None,
    ) -> ExecutionPlan:
        """Validate a selection and return its canonical execution plan."""

        model_capability = self.model(model)
        selected_backend = (
            model_capability.backend if backend in {None, "auto"} else backend
        )
        if selected_backend not in {"pinn", "operator", "custom"}:
            raise ValueError("backend must be 'auto', 'pinn', 'operator', or 'custom'.")
        if selected_backend != model_capability.backend:
            raise ValueError(
                f"Model {model_capability.name!r} requires backend "
                f"{model_capability.backend!r}, got {selected_backend!r}."
            )
        if training_mode not in model_capability.training_modes:
            choices = ", ".join(sorted(model_capability.training_modes))
            raise ValueError(
                f"Model {model_capability.name!r} does not support training mode "
                f"{training_mode!r}; supported modes: {choices}."
            )

        selected_derivative: str | None
        if training_mode == "data_only":
            if differentiation not in {None, "none"}:
                raise ValueError(
                    "data_only training does not consume a differentiation path; "
                    "omit differentiation."
                )
            selected_derivative = None
        else:
            if differentiation in {None, "auto"}:
                differentiation = sorted(model_capability.derivative_names)[0]
            derivative_capability = self.derivative(str(differentiation))
            selected_derivative = derivative_capability.name
            allowed = {
                normalize_component_name(item)
                for item in model_capability.derivative_names
            }
            if normalize_component_name(selected_derivative) not in allowed:
                choices = ", ".join(sorted(model_capability.derivative_names))
                raise ValueError(
                    f"Model {model_capability.name!r} is incompatible with "
                    f"differentiation {selected_derivative!r}; supported paths: "
                    f"{choices}."
                )
            expected_domain = (
                "structured_grid"
                if model_capability.evaluation_kind == "grid_operator"
                else "coordinate"
            )
            if derivative_capability.domain != expected_domain:
                raise ValueError(
                    f"Model {model_capability.name!r} produces "
                    f"{expected_domain} fields, but differentiation "
                    f"{selected_derivative!r} consumes "
                    f"{derivative_capability.domain} fields."
                )

        return ExecutionPlan(
            backend=selected_backend,  # type: ignore[arg-type]
            model=model_capability.name,
            evaluation_kind=model_capability.evaluation_kind,
            training_mode=training_mode,
            differentiation=selected_derivative,
        )


def default_capability_registry() -> CapabilityRegistry:
    """Build the shared model/differentiation view from live registries."""

    from .experiment import default_component_registry
    from .operator_protocol import default_operator_registry

    registry = CapabilityRegistry()
    components = default_component_registry()
    for name in components.available("model"):
        registry.register_model(
            ModelCapability(
                name=name,
                backend="pinn",
                evaluation_kind="pointwise",
                derivative_names=frozenset(
                    {"autodiff", "weak_variational", "hybrid_ad_numerical"}
                ),
                training_modes=frozenset({"physics_only", "hybrid"}),
            )
        )

    for hook in default_operator_registry().specs():
        if not hook.implemented or not hook.selectable:
            continue
        derivatives = (
            frozenset({"autodiff"})
            if hook.kind == "coordinate"
            else frozenset({"spectral", "finite_difference"})
        )
        registry.register_model(
            ModelCapability(
                name=hook.name,
                backend="operator",
                evaluation_kind=(
                    "coordinate_operator"
                    if hook.kind == "coordinate"
                    else "grid_operator"
                ),
                derivative_names=derivatives,
                training_modes=frozenset({"data_only", "physics_only", "hybrid"}),
                aliases=tuple(hook.aliases),
            )
        )

    registry.register_derivative(
        DerivativeCapability(
            "autodiff", "coordinate", 2, aliases=("coordinate_ad",)
        )
    )
    registry.register_derivative(
        DerivativeCapability(
            "weak_variational", "coordinate", 2, aliases=("weak",)
        )
    )
    registry.register_derivative(
        DerivativeCapability(
            "hybrid_ad_numerical", "coordinate", 2, aliases=("hybrid",)
        )
    )
    registry.register_derivative(
        DerivativeCapability(
            "spectral",
            "structured_grid",
            2,
            aliases=("grid", "fourier"),
            requires_periodic_grid=True,
        )
    )
    registry.register_derivative(
        DerivativeCapability(
            "finite_difference",
            "structured_grid",
            2,
            aliases=("fd", "finite_volume"),
        )
    )
    return registry


def resolve_execution_plan(
    model: str,
    *,
    backend: str | None = None,
    training_mode: str = "physics_only",
    differentiation: str | None = None,
) -> ExecutionPlan:
    """Resolve a selection through a fresh default capability registry."""

    return default_capability_registry().resolve(
        model,
        backend=backend,
        training_mode=training_mode,
        differentiation=differentiation,
    )


__all__ = [
    "BackendKind",
    "CapabilityRegistry",
    "DerivativeCapability",
    "DerivativeDomain",
    "EvaluationKind",
    "ExecutionPlan",
    "ModelCapability",
    "TrainingMode",
    "default_capability_registry",
    "resolve_execution_plan",
]
