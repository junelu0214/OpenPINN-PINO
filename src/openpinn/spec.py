"""Machine-readable ProblemSpec and MethodSpec for OpenPINN (M2).

ProblemSpec describes a physics problem (geometry, constitutive law,
boundary conditions, field outputs).  MethodSpec selects a solver
configuration from the existing method registry.  Both are validated
with pydantic so that invalid specs are rejected with clear errors
before any training or codegen step (M3).
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .method_registry import default_method_registry, normalize_method_name
from .operator_protocol import default_operator_registry
from .registry import ComponentKind


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PhysicsType(str, Enum):
    linear_elasticity = "linear_elasticity"


class GeometryType(str, Enum):
    rectangle = "rectangle"
    circle = "circle"


class BCType(str, Enum):
    dirichlet = "dirichlet"
    neumann = "neumann"


class FieldKind(str, Enum):
    displacement = "displacement"


class ConstitutiveModel(str, Enum):
    isotropic_linear = "isotropic_linear"


class StructuralAssumption(str, Enum):
    plane_stress = "plane_stress"
    plane_strain = "plane_strain"


class DifferentiatorMode(str, Enum):
    autodiff = "autodiff"
    grid = "grid"


# ---------------------------------------------------------------------------
# ProblemSpec sub-models
# ---------------------------------------------------------------------------

class RectangleGeometry(BaseModel):
    """Outer rectangular domain."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["rectangle"]
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @model_validator(mode="after")
    def _bounds_ordered(self) -> "RectangleGeometry":
        if self.x_max <= self.x_min:
            raise ValueError(
                f"x_max ({self.x_max}) must be greater than x_min ({self.x_min})"
            )
        if self.y_max <= self.y_min:
            raise ValueError(
                f"y_max ({self.y_max}) must be greater than y_min ({self.y_min})"
            )
        return self


class CircleHole(BaseModel):
    """A circular hole cut from the domain."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["circle"]
    center: tuple[float, float]
    radius: float = Field(gt=0.0)


class Geometry(BaseModel):
    """Problem geometry: outer rectangle with optional circular holes."""
    model_config = ConfigDict(extra="forbid")

    outer: RectangleGeometry
    holes: list[CircleHole] = Field(default_factory=list)

    @model_validator(mode="after")
    def _holes_inside_domain(self) -> "Geometry":
        o = self.outer
        for i, h in enumerate(self.holes):
            cx, cy = h.center
            r = h.radius
            if not (
                o.x_min < cx - r
                and cx + r < o.x_max
                and o.y_min < cy - r
                and cy + r < o.y_max
            ):
                raise ValueError(
                    f"Hole {i} (center={h.center}, radius={r}) does not lie "
                    f"strictly inside the outer rectangle "
                    f"[{o.x_min}, {o.x_max}] x [{o.y_min}, {o.y_max}]"
                )
        for i in range(len(self.holes)):
            for j in range(i + 1, len(self.holes)):
                hi, hj = self.holes[i], self.holes[j]
                dist = math.hypot(
                    hi.center[0] - hj.center[0],
                    hi.center[1] - hj.center[1],
                )
                if dist <= hi.radius + hj.radius:
                    raise ValueError(
                        f"Holes {i} and {j} overlap or touch "
                        f"(distance={dist:.6f}, sum of radii={hi.radius + hj.radius:.6f})"
                    )
        return self


class FieldSpec(BaseModel):
    """Physical field definition."""
    model_config = ConfigDict(extra="forbid")

    kind: FieldKind
    components: list[str]

    @field_validator("components")
    @classmethod
    def _at_least_one(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("components must not be empty")
        if len(v) != len(set(v)):
            raise ValueError("component names must be unique")
        return v


class ConstitutiveSpec(BaseModel):
    """Constitutive (material) law."""
    model_config = ConfigDict(extra="forbid")

    model: ConstitutiveModel
    assumption: StructuralAssumption = Field(
        default=StructuralAssumption.plane_stress,
        description="Explicit two-dimensional structural assumption.",
    )
    E: float = Field(gt=0.0, description="Young's modulus, must be positive")
    nu: float = Field(
        ge=-1.0, lt=0.5,
        description="Poisson's ratio; must be in [-1, 0.5) for isotropic 2D",
    )
    stress_voigt: bool = True


class BoundaryCondition(BaseModel):
    """One boundary condition entry."""
    model_config = ConfigDict(extra="forbid")

    id: str
    region: str = Field(
        description="Edge identifier, e.g. 'left', 'right', 'top', 'bottom', 'hole_0', ..."
    )
    type: BCType
    dofs: list[str] = Field(
        description="Degrees of freedom this BC constrains, e.g. ['ux'] or ['ux','uy']"
    )
    value: Union[float, list[float]] = Field(
        description="Prescribed value(s). Scalar applied to all listed dofs; "
        "list must match len(dofs)."
    )

    @field_validator("dofs")
    @classmethod
    def _nonempty_dofs(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("dofs must not be empty")
        if len(v) != len(set(v)):
            raise ValueError("dofs must be unique within one BC entry")
        return v

    @model_validator(mode="after")
    def _value_length(self) -> "BoundaryCondition":
        if isinstance(self.value, list) and len(self.value) != len(self.dofs):
            raise ValueError(
                f"value list length ({len(self.value)}) must match "
                f"dofs length ({len(self.dofs)})"
            )
        return self


class Units(BaseModel):
    """Declared unit system (informational, checked for presence)."""
    model_config = ConfigDict(extra="forbid")

    length: str = "m"
    force: str = "N"
    stress: str = "Pa"


# ---------------------------------------------------------------------------
# ProblemSpec top-level
# ---------------------------------------------------------------------------

class ProblemSpec(BaseModel):
    """Complete machine-readable problem specification for OpenPINN.

    Validated to reject physically invalid or incomplete descriptions before
    any codegen or training step.
    """
    model_config = ConfigDict(extra="forbid")

    dim: Literal[2]
    physics: PhysicsType
    geometry: Geometry
    field: FieldSpec
    constitutive: ConstitutiveSpec
    boundaries: list[BoundaryCondition]
    units: Units = Field(default_factory=Units)

    @model_validator(mode="after")
    def _validate_boundaries(self) -> "ProblemSpec":
        if not self.boundaries:
            raise ValueError("At least one boundary condition is required")

        valid_components = set(self.field.components)
        for bc in self.boundaries:
            invalid = set(bc.dofs) - valid_components
            if invalid:
                raise ValueError(
                    f"Boundary '{bc.id}' references unknown dofs {sorted(invalid)}; "
                    f"valid components: {sorted(valid_components)}"
                )

        dof_map: dict[tuple[str, str], list[BCType]] = {}
        for bc in self.boundaries:
            for dof in bc.dofs:
                key = (bc.region, dof)
                dof_map.setdefault(key, []).append(bc.type)

        for (region, dof), types in dof_map.items():
            if BCType.dirichlet in types and BCType.neumann in types:
                raise ValueError(
                    f"Conflicting BC on region '{region}', dof '{dof}': "
                    f"cannot apply both Dirichlet and Neumann to the same "
                    f"degree of freedom on the same edge"
                )
        return self

    @model_validator(mode="after")
    def _validate_required_edges(self) -> "ProblemSpec":
        """All outer edges and hole boundaries should have at least one BC."""
        required_edges = {"left", "right", "top", "bottom"}
        for i, _ in enumerate(self.geometry.holes):
            required_edges.add(f"hole_{i}")

        declared_regions = {bc.region for bc in self.boundaries}
        missing = required_edges - declared_regions
        if missing:
            raise ValueError(
                f"Missing boundary conditions for required edge(s): "
                f"{sorted(missing)}. Every outer edge and hole boundary "
                f"must have at least one BC."
            )
        return self


def _require_registered_component(
    kind: ComponentKind, name: str, label: str
) -> str:
    """Resolve a YAML name against the live component registry.

    Unknown names raise ``ValueError`` listing registered choices.  There is
    no silent fallback to a baseline component.
    """

    from .experiment import default_component_registry

    registry = default_component_registry()
    if not registry.contains(kind, name):
        choices = ", ".join(registry.available(kind)) or "<none>"
        raise ValueError(
            f"Unknown {label} {name!r}; available {kind}s: {choices}."
        )
    return name


# ---------------------------------------------------------------------------
# MethodSpec sub-models
# ---------------------------------------------------------------------------

class SamplingConfig(BaseModel):
    """Sampling configuration subset."""
    model_config = ConfigDict(extra="forbid")

    name: str = "uniform"
    interior: int = Field(default=256, ge=1)
    boundary: int = Field(default=64, ge=1)
    initial: Optional[int] = Field(default=None, ge=1)


class LossConfig(BaseModel):
    """Loss configuration subset."""
    model_config = ConfigDict(extra="forbid")

    formulation: str = "strong"
    weighting: str = "fixed"
    weights: Optional[dict[str, float]] = None


class OptimizerConfig(BaseModel):
    """Optimizer configuration subset."""
    model_config = ConfigDict(extra="forbid")

    name: str = "adam"
    lr: float = Field(default=1e-3, gt=0.0)


class ModelConfig(BaseModel):
    """Network architecture configuration."""
    model_config = ConfigDict(extra="forbid")

    name: str = "mlp"
    hidden_layers: list[int] = Field(default_factory=lambda: [64, 64, 64])
    activation: str = "tanh"
    options: dict[str, Any] = Field(default_factory=dict)


class DifferentiatorConfig(BaseModel):
    """Derivative computation mode, resolved against the derivative registry."""
    model_config = ConfigDict(extra="forbid")

    mode: str = DifferentiatorMode.autodiff.value
    options: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# MethodSpec top-level
# ---------------------------------------------------------------------------

class MethodSpec(BaseModel):
    """Solver / method configuration for OpenPINN.

    Method and variant names are validated against the live method registry
    and operator registry.  Unknown names are rejected.
    """
    model_config = ConfigDict(extra="forbid")

    method: str
    variant: Optional[str] = None
    path: Literal["pinn", "operator"] = "pinn"
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    network: ModelConfig = Field(default_factory=ModelConfig)
    differentiator: DifferentiatorConfig = Field(
        default_factory=DifferentiatorConfig
    )
    training: dict[str, Any] = Field(
        default_factory=lambda: {"steps": 1000}
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_canonical_axis_names(cls, value: Any) -> Any:
        """Accept the paper's canonical axis names without breaking v2 YAML.

        The v2 schema used ``path``, ``network``, ``optimizer`` and
        ``differentiator``.  The framework paper names those concepts
        ``backend``, ``model``, ``optimization`` and ``differentiation``.
        Internally we retain the v2 fields for one compatibility cycle while
        accepting both spellings and rejecting ambiguous duplicate input.
        """

        if not isinstance(value, dict):
            return value
        data = dict(value)
        aliases = {
            "backend": "path",
            "model": "network",
            "optimization": "optimizer",
            "differentiation": "differentiator",
        }
        for canonical, legacy in aliases.items():
            if canonical in data and legacy in data:
                raise ValueError(
                    f"Pass method.{canonical} or method.{legacy}, not both."
                )
            if canonical in data:
                data[legacy] = data.pop(canonical)

        differentiator = data.get("differentiator")
        if isinstance(differentiator, dict) and "name" in differentiator:
            differentiator = dict(differentiator)
            if "mode" in differentiator:
                raise ValueError(
                    "Pass differentiation.name or differentiator.mode, not both."
                )
            differentiator["mode"] = differentiator.pop("name")
            data["differentiator"] = differentiator

        path = str(data.get("path", "pinn"))
        network = data.get("network")
        if "method" not in data:
            if path == "operator" and isinstance(network, dict) and network.get("name"):
                data["method"] = network["name"]
            else:
                data["method"] = "baseline"
        return data

    def canonical_method(self) -> dict[str, Any]:
        """Return the paper-aligned five-axis method representation."""

        operator_registry = default_operator_registry()
        model_name = self.network.name
        if self.path == "operator":
            if not operator_registry.contains(model_name):
                # Legacy operator specs stored the operator in ``method`` and a
                # branch/trunk subnetwork placeholder in ``network``.
                model_name = self.method
            model_name = operator_registry.resolve(model_name).name
        else:
            from .experiment import default_component_registry

            model_name = default_component_registry().canonical_name(
                "model", model_name
            )
        return {
            "backend": self.path,
            "model": {
                "name": model_name,
                "hidden_layers": list(self.network.hidden_layers),
                "activation": self.network.activation,
                "options": dict(self.network.options),
            },
            "sampling": self.sampling.model_dump(mode="python"),
            "loss": self.loss.model_dump(mode="python"),
            "optimization": self.optimizer.model_dump(mode="python"),
            "differentiation": {
                "name": self.differentiator.mode,
                "options": dict(self.differentiator.options),
            },
            "training": dict(self.training),
        }

    def execution_plan(self) -> Any:
        """Resolve model/backend/differentiation compatibility."""

        from .capabilities import resolve_execution_plan

        canonical = self.canonical_method()
        mode = str(self.training.get("mode", "physics_only"))
        differentiation = canonical["differentiation"]["name"]
        if mode == "data_only" and "differentiator" not in self.model_fields_set:
            differentiation = None
        return resolve_execution_plan(
            canonical["model"]["name"],
            backend=canonical["backend"],
            training_mode=mode,
            differentiation=differentiation,
        )

    @model_validator(mode="after")
    def _validate_method_in_registry(self) -> "MethodSpec":
        method_reg = default_method_registry()
        operator_reg = default_operator_registry()

        if self.path == "pinn":
            try:
                method_reg.resolve(self.method, variant=self.variant)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
        elif self.path == "operator":
            if not operator_reg.contains(self.method):
                available = ", ".join(operator_reg.available())
                raise ValueError(
                    f"Unknown operator '{self.method}'; "
                    f"available operators: {available}"
                )
            if self.variant is not None:
                raise ValueError(
                    "Operator-path methods do not support the 'variant' field; "
                    "remove it or switch to path='pinn'"
                )

        _require_registered_component("sampler", self.sampling.name, "sampler")
        if self.path == "pinn":
            _require_registered_component("model", self.network.name, "model")
        elif operator_reg.contains(self.network.name):
            # Canonical specs put DeepONet/FNO directly on the model axis.
            selected = operator_reg.resolve(self.network.name)
            if selected.name != operator_reg.resolve(self.method).name:
                raise ValueError(
                    "Operator method and model names disagree; use one canonical "
                    "model selection."
                )
        _require_registered_component("optimizer", self.optimizer.name, "optimizer")
        _require_registered_component(
            "formulation", self.loss.formulation, "formulation"
        )
        _require_registered_component("loss", self.loss.weighting, "loss")
        _require_registered_component(
            "derivative", self.differentiator.mode, "derivative"
        )
        return self

    @model_validator(mode="after")
    def _validate_training(self) -> "MethodSpec":
        if "steps" in self.training:
            steps = self.training["steps"]
            if not isinstance(steps, int) or steps <= 0:
                raise ValueError(
                    f"training.steps must be a positive integer, got {steps!r}"
                )
        return self


# ---------------------------------------------------------------------------
# Loader / validator
# ---------------------------------------------------------------------------

def load_problem_spec(source: str | dict[str, Any]) -> ProblemSpec:
    """Load and validate a ProblemSpec from a YAML file path or dict.

    Raises ``pydantic.ValidationError`` with detailed field-level messages
    on any schema or cross-field violation.
    """
    if isinstance(source, str):
        with open(source, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a YAML mapping at top level, got {type(data).__name__}")
    else:
        data = source
    return ProblemSpec.model_validate(data)


def load_method_spec(source: str | dict[str, Any]) -> MethodSpec:
    """Load and validate a MethodSpec from a YAML file path or dict.

    Raises ``pydantic.ValidationError`` with detailed field-level messages
    on any schema or cross-field violation.
    """
    if isinstance(source, str):
        with open(source, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a YAML mapping at top level, got {type(data).__name__}")
    else:
        data = source
    return MethodSpec.model_validate(data)


def validate_problem_spec(source: str | dict[str, Any]) -> list[str]:
    """Return a list of validation error messages (empty = valid)."""
    try:
        load_problem_spec(source)
        return []
    except Exception as exc:
        return [str(exc)]


def validate_method_spec(source: str | dict[str, Any]) -> list[str]:
    """Return a list of validation error messages (empty = valid)."""
    try:
        load_method_spec(source)
        return []
    except Exception as exc:
        return [str(exc)]
