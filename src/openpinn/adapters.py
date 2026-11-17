"""Structural adapters for the shared PINN engine."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor

from .artifacts import ArtifactWriter
from .conditions import Condition
from .engine import EngineRunResult, problem_from_compiled_expressions, run_pinn
from .expressions import CompiledProblemExpressions, ExpressionCompileError, compile_problem_expressions
from .geometry import Interval, RectangleWithCircularHoles, SpaceTimeDomain
from .models import MLP
from .problems import PDEProblem
from .sampling import UniformSampler


RUNNABLE_GENERATOR_ID = "openpinn.shared-pinn-engine.v1"
BackendId = Literal["poisson_1d", "heat_1d_unsteady", "scalar_strong_form_v1"]
CapabilityStatus = Literal["supported", "unsupported", "incomplete"]


@dataclass(frozen=True)
class TrainingCapability:
    """Result of structural executable-backend matching."""

    status: CapabilityStatus
    backend_id: BackendId | None
    reason: str
    smoke_requires_python: bool = True
    training_complete: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class _CompiledCondition(Condition):
    def __init__(self, evaluate: Any, *, name: str, sample_kind: str, selector: Any = None) -> None:
        super().__init__(value=0.0, name=name, selector=selector)
        self.evaluate = evaluate
        self.sample_kind = sample_kind

    def error(self, model: Any, points: Tensor) -> Tensor:
        selected = self.select(points)
        return self.evaluate(selected, model)


def _text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def _named(entries: Any) -> list[Mapping[str, Any]]:
    return [entry for entry in entries if isinstance(entry, Mapping)] if isinstance(entries, list) else []


def _coordinate(problem: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    return next((item for item in _named(problem.get("coordinates")) if item.get("name") == name), None)


def _bounds(coordinate: Mapping[str, Any] | None) -> tuple[float, float] | None:
    values = coordinate and coordinate.get("bounds")
    if not isinstance(values, Mapping):
        return None
    lower, upper = values.get("min"), values.get("max")
    if isinstance(lower, (int, float)) and isinstance(upper, (int, float)) and lower < upper:
        return float(lower), float(upper)
    return None


def _method_supported(method: Mapping[str, Any]) -> bool:
    loss = method.get("loss") if isinstance(method.get("loss"), Mapping) else {}
    differentiator = method.get("differentiator") if isinstance(method.get("differentiator"), Mapping) else {}
    return (
        method.get("path") == "pinn"
        and loss.get("formulation", "strong") == "strong"
        and differentiator.get("mode", "autodiff") == "autodiff"
    )


def _scalar_field(problem: Mapping[str, Any]) -> str | None:
    fields = _named(problem.get("fields"))
    if len(fields) != 1:
        return None
    components = fields[0].get("components")
    if not isinstance(components, list) or len(components) != 1 or not isinstance(components[0], str):
        return None
    return components[0]


def _locations(boundaries: Sequence[Mapping[str, Any]], kind: str) -> set[str]:
    return {
        _text(item.get("location"))
        for item in boundaries
        if _text(item.get("type")) == kind and item.get("value") == 0
    }


def _boundary_location_supported(problem: Mapping[str, Any], location: Any) -> bool:
    if not isinstance(location, str):
        return False
    match = re.fullmatch(
        r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*",
        location,
    )
    if match is None:
        return False
    coordinate = _coordinate(problem, match.group(1))
    if coordinate is None or coordinate.get("role") != "space":
        return False
    bounds = _bounds(coordinate)
    return bounds is not None and float(match.group(2)) in bounds


def match_training_backend(problem: Mapping[str, Any], method: Mapping[str, Any]) -> TrainingCapability:
    """Match registered training backends using declared structure."""

    contract = problem.get("trainable_contract")
    if not isinstance(contract, Mapping) or contract.get("mapping_complete") is not True:
        return TrainingCapability("incomplete", None, "Stage A mapping_complete is required.")
    if not _method_supported(method):
        return TrainingCapability("unsupported", None, "Only strong-form PINN with autodiff is registered.")
    field = _scalar_field(problem)
    if field is None:
        return TrainingCapability("unsupported", None, "A single scalar field is required.")
    coordinates = _named(problem.get("coordinates"))
    equations = _named(problem.get("equations"))
    boundaries = _named(problem.get("boundaries"))
    initials = _named(problem.get("initial_conditions"))
    expression = _text(equations[0].get("expression", equations[0].get("residual"))) if len(equations) == 1 else ""
    x_bounds = _bounds(_coordinate(problem, "x"))
    coordinate_names = [item.get("name") for item in coordinates]
    coordinate_roles = [item.get("role") for item in coordinates]
    geometry = problem.get("geometry") if isinstance(problem.get("geometry"), Mapping) else {}
    dirichlet = _locations(boundaries, "dirichlet")
    neumann = _locations(boundaries, "neumann")

    poisson_operator = f"dx(dx({field.lower()}))" in expression and "dt(" not in expression
    poisson_rhs = "pi*pi*sin(pi*x)" in expression
    if (
        coordinate_names == ["x"]
        and field == "u"
        and coordinate_roles == ["space"]
        and _text(geometry.get("type")) == "interval"
        and x_bounds == (0.0, 1.0)
        and poisson_operator
        and poisson_rhs
        and dirichlet == {"x=0", "x=1"}
        and not neumann
        and not initials
    ):
        return TrainingCapability("supported", "poisson_1d", "Registered 1D interval/Dirichlet strong-form structure.")

    parameters = {item.get("name"): item.get("value") for item in _named(problem.get("parameters"))}
    time_bounds = _bounds(_coordinate(problem, "t"))
    initial_expression = _text(initials[0].get("expression")) if len(initials) == 1 else ""
    heat_operator = (
        f"dt({field.lower()})" in expression
        and f"dx(dx({field.lower()}))" in expression
        and "alpha" in expression
    )
    if (
        coordinate_names == ["x", "t"]
        and field == "u"
        and coordinate_roles == ["space", "time"]
        and _text(geometry.get("type")) == "interval"
        and x_bounds == (0.0, 1.0)
        and time_bounds == (0.0, 1.0)
        and heat_operator
        and isinstance(parameters.get("alpha"), (int, float))
        and float(parameters["alpha"]) == 0.01
        and len(parameters) == 1
        and dirichlet == {"x=0"}
        and neumann == {"x=1"}
        and initial_expression == "sin(pi*x/2)"
    ):
        return TrainingCapability("supported", "heat_1d_unsteady", "Registered 1D interval+time mixed-BC strong-form structure.")
    spatial = [item for item in coordinates if item.get("role") == "space"]
    temporal = [item for item in coordinates if item.get("role") == "time"]
    supported_coordinates = (
        [item.get("name") for item in spatial] in (["x"], ["x", "y"])
        and len(temporal) <= 1
        and len(spatial) + len(temporal) == len(coordinates)
        and (not temporal or temporal[0].get("name") == "t")
        and all(_bounds(item) is not None for item in coordinates)
    )
    geometry_type = _text(geometry.get("type"))
    supported_geometry = (
        len(spatial) == 1 and geometry_type == "interval"
    ) or (
        len(spatial) == 2 and geometry_type in {"rectangle", "rectangular"}
    )
    supported_boundaries = bool(boundaries) and all(
        _text(item.get("type")) in {"dirichlet", "neumann", "robin"}
        and _boundary_location_supported(problem, item.get("location"))
        and ("value" in item or "expression" in item or "value_expression" in item)
        and (_text(item.get("type")) != "robin" or ("alpha" in item and "beta" in item))
        for item in boundaries
    )
    supported_initials = (not temporal and not initials) or (bool(temporal) and bool(initials))
    if supported_coordinates and supported_geometry and equations and supported_boundaries and supported_initials:
        try:
            compile_problem_expressions(problem)
        except ExpressionCompileError as error:
            return TrainingCapability("unsupported", None, f"Restricted expression compilation failed: {error}")
        return TrainingCapability(
            "supported", "scalar_strong_form_v1",
            "Registered scalar strong-form PINN capabilities: 1D/2D, optional time, compiled equations and conditions.",
        )
    return TrainingCapability("unsupported", None, "No registered executable backend matches the declared operators and conditions.")


def _location_selector(location: str, coordinate_names: Sequence[str]):
    match = re.fullmatch(
        r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*",
        location,
    )
    if match is None or match.group(1) not in coordinate_names:
        raise ValueError(f"Unsupported boundary location {location!r}.")
    index, value = coordinate_names.index(match.group(1)), float(match.group(2))

    def select(points: Tensor) -> Tensor:
        return torch.isclose(points[:, index], points.new_tensor(value))

    return select


def _compiled_conditions(snapshot: Mapping[str, Any], compiled: CompiledProblemExpressions) -> list[Condition]:
    conditions: list[Condition] = []
    for index, boundary in enumerate(_named(snapshot.get("boundaries"))):
        name = str(boundary.get("id") or f"boundary_{index}")
        conditions.append(
            _CompiledCondition(
                compiled.boundaries[name], name=name, sample_kind="boundary",
                selector=_location_selector(str(boundary.get("location", "")), list(compiled.coordinate_names)),
            )
        )
    for index, initial in enumerate(_named(snapshot.get("initial_conditions"))):
        name = str(initial.get("id") or f"initial_{index}")
        conditions.append(_CompiledCondition(compiled.initial_conditions[name], name=name, sample_kind="initial"))
    return conditions


def build_registered_problem(
    snapshot: Mapping[str, Any], method: Mapping[str, Any]
) -> tuple[PDEProblem, TrainingCapability]:
    """Compile a supported snapshot into a problem for ``run_pinn``."""

    capability = match_training_backend(snapshot, method)
    if capability.status != "supported" or capability.backend_id is None:
        raise ValueError(capability.reason)
    compiled = compile_problem_expressions(snapshot)
    x_min, x_max = _bounds(_coordinate(snapshot, "x")) or (0.0, 1.0)
    spatial_names = [item.get("name") for item in _named(snapshot.get("coordinates")) if item.get("role") == "space"]
    if spatial_names == ["x", "y"]:
        y_min, y_max = _bounds(_coordinate(snapshot, "y")) or (0.0, 1.0)
        space: Any = RectangleWithCircularHoles((x_min, y_min), (x_max, y_max), ())
    else:
        space = Interval(x_min, x_max)
    if _coordinate(snapshot, "t") is not None:
        t_min, t_max = _bounds(_coordinate(snapshot, "t")) or (0.0, 1.0)
        domain: Any = SpaceTimeDomain(space, t_min, t_max)
    else:
        domain = space
    problem = problem_from_compiled_expressions(
        compiled, domain, _compiled_conditions(snapshot, compiled), name=capability.backend_id
    )
    return problem, capability


def _reference_errors(backend_id: BackendId, model: MLP, alpha: float) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        if backend_id == "poisson_1d":
            points = torch.linspace(0.0, 1.0, 101, device=device).reshape(-1, 1)
            exact = torch.sin(math.pi * points)
        else:
            x = torch.linspace(0.0, 1.0, 51, device=device)
            t = torch.linspace(0.0, 1.0, 51, device=device)
            xx, tt = torch.meshgrid(x, t, indexing="ij")
            points = torch.stack((xx.reshape(-1), tt.reshape(-1)), dim=1)
            exact = torch.sin(math.pi * points[:, :1] / 2) * torch.exp(
                -alpha * math.pi**2 * points[:, 1:2] / 4
            )
        difference = model(points) - exact
        return {
            "max_abs": float(difference.abs().max()),
            "rel_l2": float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(exact)),
        }


def _loss_history(output_dir: Path, result: EngineRunResult) -> list[float]:
    with (output_dir / result.history_path).open(encoding="utf-8-sig", newline="") as handle:
        return [float(row["total"]) for row in csv.DictReader(handle) if row.get("total")]


def compiled_alpha(snapshot: Mapping[str, Any]) -> float:
    for parameter in _named(snapshot.get("parameters")):
        if parameter.get("name") == "alpha" and isinstance(parameter.get("value"), (int, float)):
            return float(parameter["value"])
    return 0.01


def run_registered_pinn(
    snapshot: Mapping[str, Any], method: Mapping[str, Any], *, output_dir: str | Path,
    pair_fingerprint: str, max_steps: int, max_wall_seconds: float, device: str = "cpu",
    smoke: bool = False, seed: int = 20261110,
    semantic_concerns: Sequence[Mapping[str, Any]] = (), generator_id: str = RUNNABLE_GENERATOR_ID,
) -> tuple[EngineRunResult, dict[str, Any]]:
    """Run a registered adapter and write its same-run product RunArtifact."""

    if not isinstance(pair_fingerprint, str) or not pair_fingerprint:
        raise ValueError("pair_fingerprint is required")
    problem, capability = build_registered_problem(snapshot, method)
    assert capability.backend_id is not None
    torch.manual_seed(seed)
    model = MLP(problem.domain.dim, problem.output_dim, hidden_layers=(32, 32, 32))
    sampler = UniformSampler(
        problem, n_interior=64, n_boundary=16,
        n_initial=16 if _named(snapshot.get("initial_conditions")) else 0, seed=seed,
    )
    run_steps, target = (1 if smoke else max_steps), Path(output_dir)
    result = run_pinn(
        problem, model=model, sampler=sampler, output_dir=target,
        max_steps=run_steps, max_wall_seconds=max_wall_seconds, device=device,
    )
    losses = _loss_history(target, result)
    errors: dict[str, float] = {}
    if capability.backend_id in {"poisson_1d", "heat_1d_unsteady"} and result.status in {"budget_exhausted", "not_converged", "completed"}:
        errors = _reference_errors(capability.backend_id, model, compiled_alpha(snapshot))
    failure, stack = result.failure or {}, (result.failure or {}).get("traceback")
    artifact = {
        "schema_version": 1, "kind": "RunArtifact", "pair_fingerprint": pair_fingerprint,
        "generator_id": generator_id, "backend_runnable": True,
        "backend_id": capability.backend_id, "artifact_origin": "local_generated_run",
        "example": False, "fixture": False, "smoke_only": bool(smoke), "status": result.status,
        "recorded_at": datetime.now(timezone.utc).isoformat(), "device": result.device,
        "budgets": {"max_steps": run_steps, "max_wall_seconds": result.max_wall_seconds},
        "steps_completed": result.steps_completed, "engine": result.as_dict(),
        "metrics": {"loss_history": losses, "final_loss": losses[-1] if losses else None, "error_metrics": errors},
        "failure": {"type": failure.get("type"), "message": failure.get("message"),
                    "stack": stack, "truncated": bool(stack and len(stack) >= 4000)},
        "semantic_concerns": [dict(item) for item in semantic_concerns],
        "accuracy_threshold": None, "accuracy_verified": False,
    }
    writer = ArtifactWriter(target)
    writer.write_json("run_artifact.json", artifact)
    writer.write_manifest(metadata={
        "kind": "RegisteredPINNRun",
        "status": result.status,
        "backend_id": capability.backend_id,
        "pair_fingerprint": pair_fingerprint,
    })
    return result, artifact
