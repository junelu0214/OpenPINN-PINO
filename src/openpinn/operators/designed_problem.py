"""Train selectable operators on a user-designed coordinate PDE.

Unlike the catalog smoke trainers, this module consumes a real
:class:`openpinn.PDEProblem` and optional CSV observations.  DeepONet keeps
coordinate autodiff.  FNO evaluates the same equation on a structured grid,
with ``grad`` and ``hessian`` scoped to a real ``GridDerivative`` backend.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..autodiff import structured_derivatives
from ..conditions import InitialCondition, NeumannBC, RobinBC, evaluate_value
from ..metrics import mean_absolute_error, relative_l2_error, rmse
from ..problems import PDEProblem
from ..reference_data import OperatorObservation, load_operator_observation_csv
from .grid_derivative import make_grid_derivative
from .selection import OperatorTrainResult, TRAINING_MODES, _as_operator_config
from .uniform_grid import UniformGrid


def _mean_square(values: Tensor) -> Tensor:
    return torch.mean(values.square())


def _problem_bounds(problem: PDEProblem) -> tuple[tuple[float, float], ...]:
    domain = problem.domain
    if hasattr(domain, "time") and hasattr(domain, "space"):
        spatial = _geometry_bounds(getattr(domain, "space"))
        time = getattr(domain, "time")
        return (*spatial, (float(time.lower), float(time.upper)))
    return _geometry_bounds(domain)


def _geometry_bounds(domain: object) -> tuple[tuple[float, float], ...]:
    bounds = getattr(domain, "bounds", None)
    if bounds is None:
        raise TypeError("Designed grid training requires geometry bounds.")
    if len(bounds) == 2 and all(isinstance(value, (int, float)) for value in bounds):
        return ((float(bounds[0]), float(bounds[1])),)
    normalized = tuple((float(pair[0]), float(pair[1])) for pair in bounds)
    if any(not upper > lower for lower, upper in normalized):
        raise ValueError("Designed problem bounds must be finite increasing intervals.")
    return normalized


def _as_observation(
    observation: OperatorObservation | str | Path | None,
    *,
    coordinate_columns: Sequence[str | int],
    value_column: str | int,
    coordinate_scales: Sequence[float] | None,
    value_scale: float,
    fixed_time: float | None,
) -> OperatorObservation | None:
    if observation is None or isinstance(observation, OperatorObservation):
        return observation
    return load_operator_observation_csv(
        observation,
        coordinate_columns=coordinate_columns,
        value_column=value_column,
        coordinate_scales=coordinate_scales,
        value_scale=value_scale,
        fixed_time=fixed_time,
    )


def _validate_observation(problem: PDEProblem, observation: OperatorObservation) -> None:
    if observation.coordinates.ndim != 2 or observation.coordinates.shape[1] != problem.domain.dim:
        raise ValueError(
            f"Observation coordinates must have shape (N, {problem.domain.dim})."
        )
    if observation.values.ndim != 2 or observation.values.shape != (observation.coordinates.shape[0], 1):
        raise ValueError("Operator observations must contain one scalar value per coordinate.")
    inside = problem.domain.contains(observation.coordinates, atol=1e-6)
    if not bool(torch.all(inside)):
        raise ValueError(
            f"Observation CSV has {int((~inside).sum())} coordinates outside the designed problem."
        )


def _coordinate_condition_loss(problem: PDEProblem, model: torch.nn.Module, count: int, *, device: str) -> Tensor:
    losses: list[Tensor] = []
    for condition in problem.conditions:
        if isinstance(condition, InitialCondition):
            sampler = getattr(problem.domain, "sample_initial", None)
            if sampler is None:
                raise ValueError("InitialCondition requires a space-time domain.")
        else:
            sampler = problem.domain.sample_boundary
        points = sampler(count, device=device)
        losses.append(float(condition.weight) * _mean_square(condition.error(model, points)))
    if not losses:
        parameter = next(model.parameters())
        return parameter.sum() * 0.0
    return torch.stack(losses).sum()


def _train_coordinate(
    config: Mapping[str, Any],
    problem: PDEProblem,
    observation: OperatorObservation | None,
    *,
    device: str,
) -> OperatorTrainResult:
    from .deeponet import build_deeponet

    if int(config["n_dims"]) != problem.domain.dim or int(config["n_fields"]) != problem.output_dim:
        raise ValueError(
            "DeepONet n_dims/n_fields must match the designed problem dimension/output_dim."
        )
    torch.manual_seed(int(config["seed"]))
    model = build_deeponet(config).to(device)
    mode = str(config["training_mode"])
    query_count = max(4, int(config["n_query"]))
    obs_coordinates = None if observation is None else observation.coordinates.to(device)
    obs_values = None if observation is None else observation.values.to(device)

    def losses() -> tuple[Tensor, Tensor, Tensor]:
        data = next(model.parameters()).sum() * 0.0
        if obs_coordinates is not None and obs_values is not None:
            data = _mean_square(model.predict(obs_coordinates) - obs_values)
        physics = next(model.parameters()).sum() * 0.0
        if mode != "data_only":
            coordinates = problem.domain.sample_interior(query_count, device=device)
            coordinates.requires_grad_(True)
            prediction = model.predict(coordinates)
            residuals = problem.evaluate_residual(coordinates, prediction)
            physics = sum(_mean_square(value) for value in residuals.values())
            physics = physics + _coordinate_condition_loss(
                problem, model, max(4, query_count // 2), device=device
            )
        total = (
            float(config["data_weight"]) * data
            + float(config["physics_weight"]) * physics
        )
        if mode == "data_only":
            total = data
        return total, data, physics

    model.train()
    initial_total, initial_data, initial_physics = losses()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    history: list[float] = []
    for _ in range(int(config["steps"])):
        optimizer.zero_grad(set_to_none=True)
        total, _, _ = losses()
        total.backward()
        optimizer.step()
        history.append(float(total.detach()))
    model.eval()
    final_total, final_data, final_physics = losses()

    def evaluator(coordinates: Tensor) -> Tensor:
        parameter = next(model.parameters())
        return model.predict(coordinates.to(device=parameter.device, dtype=parameter.dtype))

    return OperatorTrainResult(
        operator="deeponet",
        steps=int(config["steps"]),
        initial_loss=float(initial_total.detach()),
        final_loss=float(final_total.detach()),
        history=tuple(history),
        model=model,
        config=dict(config),
        metrics={
            "initial_data_loss": float(initial_data.detach()),
            "final_data_loss": float(final_data.detach()),
            "initial_physics_loss": float(initial_physics.detach()),
            "final_physics_loss": float(final_physics.detach()),
        },
        evaluator=evaluator,
    )


def _grid_from_observation(
    observation: OperatorObservation,
) -> tuple[UniformGrid, Tensor]:
    coordinates = observation.coordinates
    axes = tuple(torch.unique(coordinates[:, axis], sorted=True) for axis in range(coordinates.shape[1]))
    shape = tuple(int(axis.numel()) for axis in axes)
    if math.prod(shape) != coordinates.shape[0]:
        raise ValueError(
            "FNO observation CSV must contain one complete Cartesian grid; missing points are not interpolated."
        )
    spacing: list[float] = []
    for axis in axes:
        if axis.numel() < 2:
            raise ValueError("FNO observation axes need at least two distinct coordinates.")
        differences = torch.diff(axis)
        if not torch.allclose(differences, differences[0], rtol=1e-5, atol=1e-7):
            raise ValueError("FNO observation coordinates must form a uniform grid.")
        spacing.append(float(differences[0]))
    lookup = {
        tuple(float(value) for value in point): observation.values[index, 0]
        for index, point in enumerate(coordinates)
    }
    mesh = torch.meshgrid(*axes, indexing="ij")
    ordered_points = torch.stack([item.reshape(-1) for item in mesh], dim=1)
    try:
        ordered_values = torch.stack(
            [lookup[tuple(float(value) for value in point)] for point in ordered_points]
        )
    except KeyError as exc:
        raise ValueError("FNO observation Cartesian grid has a missing coordinate.") from exc
    return UniformGrid(axes, spacing=spacing, periodic=False), ordered_values.reshape(1, 1, *shape)


def _default_grid(problem: PDEProblem, size: int) -> UniformGrid:
    bounds = _problem_bounds(problem)
    axes = tuple(torch.linspace(lower, upper, size) for lower, upper in bounds)
    spacing = tuple(float(axis[1] - axis[0]) for axis in axes)
    return UniformGrid(axes, spacing=spacing, periodic=False)


def _grid_coordinates(grid: UniformGrid, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    axes = tuple(axis.to(device=device, dtype=dtype) for axis in grid.nodes)
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([axis.reshape(-1) for axis in mesh], dim=1)


def _flatten_field(field: Tensor) -> Tensor:
    return field[0].movedim(0, -1).reshape(-1, field.shape[1])


def _field_from_points(values: Tensor, shape: Sequence[int]) -> Tensor:
    return values.reshape(*shape, values.shape[1]).movedim(-1, 0).unsqueeze(0)


class CoordinateProblemOnGrid:
    """Evaluate one scalar coordinate ``PDEProblem`` with grid derivatives."""

    def __init__(self, problem: PDEProblem) -> None:
        if problem.output_dim != 1:
            raise ValueError("Designed FNO training currently supports one scalar output field.")
        self.problem = problem

    def residual(
        self,
        solution: Tensor,
        inputs: Tensor,
        *,
        grid: UniformGrid,
        derivative: str = "finite_difference",
    ) -> dict[str, Tensor]:
        del inputs
        coordinates = _grid_coordinates(grid, device=solution.device, dtype=solution.dtype)
        flattened = _flatten_field(solution)
        stencil = make_grid_derivative(derivative, spacing=grid.spacing)

        def partial(values: Tensor, first: int, second: int | None = None) -> Tensor:
            field = _field_from_points(values, grid.shape)
            field = stencil.partial(field, axis=first, order=1)
            if second is not None:
                field = stencil.partial(field, axis=second, order=1)
            return _flatten_field(field)

        def gradient(values: Tensor) -> Tensor:
            return torch.cat([partial(values, axis) for axis in range(grid.n_dims)], dim=1)

        with structured_derivatives(
            flattened,
            coordinates,
            gradient=gradient,
            second=lambda values, first, second: partial(values, first, second),
        ):
            return self.problem.evaluate_residual(coordinates, flattened)


def _grid_condition_loss(
    problem: PDEProblem,
    adapter: CoordinateProblemOnGrid,
    solution: Tensor,
    grid: UniformGrid,
    *,
    derivative: str,
) -> Tensor:
    del adapter
    if not problem.conditions:
        return solution.sum() * 0.0
    coordinates = _grid_coordinates(grid, device=solution.device, dtype=solution.dtype)
    prediction = _flatten_field(solution)
    stencil = make_grid_derivative(derivative, spacing=grid.spacing)
    gradient = torch.cat(
        [_flatten_field(stencil.partial(solution, axis=axis, order=1)) for axis in range(grid.n_dims)],
        dim=1,
    )
    boundary_mask = torch.zeros(coordinates.shape[0], dtype=torch.bool, device=coordinates.device)
    for axis, nodes in enumerate(grid.nodes):
        lower = coordinates.new_tensor(float(nodes[0]))
        upper = coordinates.new_tensor(float(nodes[-1]))
        boundary_mask |= torch.isclose(coordinates[:, axis], lower, atol=1e-6, rtol=0.0)
        boundary_mask |= torch.isclose(coordinates[:, axis], upper, atol=1e-6, rtol=0.0)
    losses: list[Tensor] = []
    for condition in problem.conditions:
        if isinstance(condition, InitialCondition):
            mask = torch.isclose(
                coordinates[:, -1], coordinates.new_tensor(float(grid.nodes[-1][0])),
                atol=1e-6, rtol=0.0,
            )
        elif condition.selector is not None:
            mask = condition.selector(coordinates).to(dtype=torch.bool, device=coordinates.device)
        else:
            mask = boundary_mask
        if not bool(torch.any(mask)):
            raise ValueError(f"Condition {condition.name!r} selected no structured-grid points.")
        points = coordinates[mask]
        values = prediction[mask]
        if isinstance(condition, (NeumannBC, RobinBC)):
            normal = condition.normal(points) if callable(condition.normal) else condition.normal
            normal_tensor = torch.as_tensor(normal, device=points.device, dtype=points.dtype)
            if normal_tensor.ndim == 1:
                normal_tensor = normal_tensor.reshape(1, -1).expand(points.shape[0], -1)
            normal_tensor = normal_tensor / torch.linalg.vector_norm(normal_tensor, dim=1, keepdim=True)
            normal_derivative = (gradient[mask] * normal_tensor).sum(dim=1, keepdim=True)
            if isinstance(condition, RobinBC):
                error = (
                    evaluate_value(condition.alpha, points, like=values) * values
                    + evaluate_value(condition.beta, points, like=values) * normal_derivative
                    - evaluate_value(condition.value, points, like=values)
                )
            else:
                error = normal_derivative - evaluate_value(condition.value, points, like=normal_derivative)
        else:
            error = values - evaluate_value(condition.value, points, like=values)
        losses.append(float(condition.weight) * _mean_square(error))
    return torch.stack(losses).sum()


def _train_grid(
    config: Mapping[str, Any],
    problem: PDEProblem,
    observation: OperatorObservation | None,
    *,
    device: str,
    input_function: Callable[[Tensor], Tensor] | None,
) -> OperatorTrainResult:
    from .fno import build_fno

    if problem.domain.dim not in (1, 2):
        raise ValueError("Designed FNO training supports one or two grid coordinates; 3D is not enabled.")
    if int(config["n_dims"]) != problem.domain.dim or int(config["out_channels"]) != 1:
        raise ValueError("FNO n_dims/out_channels must match the designed scalar problem.")
    if int(config["in_channels"]) != 1:
        raise ValueError("Designed FNO training currently uses one conditioning input channel.")
    torch.manual_seed(int(config["seed"]))
    model = build_fno(config).to(device)
    mode = str(config["training_mode"])
    if observation is not None:
        grid, target = _grid_from_observation(observation)
    else:
        grid = _default_grid(problem, max(4, int(config["grid_size"])))
        target = None
    grid = UniformGrid(
        tuple(axis.to(device) for axis in grid.nodes),
        spacing=grid.spacing,
        periodic=grid.periodic,
    )
    coordinates = _grid_coordinates(grid, device=torch.device(device), dtype=torch.float32)
    if input_function is None:
        bounds = _problem_bounds(problem)
        input_values = sum(
            (coordinates[:, axis : axis + 1] - lower) / (upper - lower)
            for axis, (lower, upper) in enumerate(bounds)
        ) / len(bounds)
    else:
        input_values = input_function(coordinates)
    if input_values.ndim == 1:
        input_values = input_values.reshape(-1, 1)
    if input_values.shape != (coordinates.shape[0], 1):
        raise ValueError("input_function must return one scalar per grid coordinate.")
    inputs = _field_from_points(input_values, grid.shape)
    target = None if target is None else target.to(device)
    adapter = CoordinateProblemOnGrid(problem)
    derivative = str(config["derivative"])

    def losses() -> tuple[Tensor, Tensor, Tensor]:
        solution = model.predict(inputs, grid=grid)
        data = solution.sum() * 0.0 if target is None else _mean_square(solution - target)
        physics = solution.sum() * 0.0
        if mode != "data_only":
            residuals = adapter.residual(solution, inputs, grid=grid, derivative=derivative)
            physics = sum(_mean_square(value) for value in residuals.values())
            physics = physics + _grid_condition_loss(
                problem, adapter, solution, grid, derivative=derivative
            )
        total = float(config["data_weight"]) * data + float(config["physics_weight"]) * physics
        if mode == "data_only":
            total = data
        return total, data, physics

    model.train()
    initial_total, initial_data, initial_physics = losses()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    history: list[float] = []
    for _ in range(int(config["steps"])):
        optimizer.zero_grad(set_to_none=True)
        total, _, _ = losses()
        total.backward()
        optimizer.step()
        history.append(float(total.detach()))
    model.eval()
    final_total, final_data, final_physics = losses()

    def evaluator(eval_coordinates: Tensor) -> Tensor:
        temporary = OperatorObservation(
            coordinates=eval_coordinates.detach().cpu(),
            values=torch.zeros((eval_coordinates.shape[0], 1)),
            source=Path("<evaluation>"),
        )
        eval_grid, _ = _grid_from_observation(temporary)
        eval_grid = UniformGrid(
            tuple(axis.to(coordinates.device) for axis in eval_grid.nodes),
            spacing=eval_grid.spacing,
            periodic=eval_grid.periodic,
        )
        ordered = _grid_coordinates(
            eval_grid, device=coordinates.device, dtype=coordinates.dtype
        )
        if input_function is None:
            bounds = _problem_bounds(problem)
            eval_values = sum(
                (ordered[:, axis : axis + 1] - lower) / (upper - lower)
                for axis, (lower, upper) in enumerate(bounds)
            ) / len(bounds)
        else:
            eval_values = input_function(ordered)
        if eval_values.ndim == 1:
            eval_values = eval_values.reshape(-1, 1)
        eval_inputs = _field_from_points(eval_values, eval_grid.shape)
        ordered_prediction = _flatten_field(model.predict(eval_inputs, grid=eval_grid))
        positions = {
            tuple(float(value) for value in point): index
            for index, point in enumerate(ordered.detach().cpu())
        }
        indices = [
            positions[tuple(float(value) for value in point)]
            for point in eval_coordinates.detach().cpu()
        ]
        return ordered_prediction[torch.as_tensor(indices, device=ordered_prediction.device)]

    return OperatorTrainResult(
        operator="fno",
        steps=int(config["steps"]),
        initial_loss=float(initial_total.detach()),
        final_loss=float(final_total.detach()),
        history=tuple(history),
        model=model,
        config=dict(config),
        metrics={
            "initial_data_loss": float(initial_data.detach()),
            "final_data_loss": float(final_data.detach()),
            "initial_physics_loss": float(initial_physics.detach()),
            "final_physics_loss": float(final_physics.detach()),
        },
        evaluator=evaluator,
    )


def train_operator_problem(
    source: str | Mapping[str, Any],
    problem: PDEProblem,
    *,
    observation: OperatorObservation | str | Path | None = None,
    coordinate_columns: Sequence[str | int] = ("x",),
    value_column: str | int = "u",
    coordinate_scales: Sequence[float] | None = None,
    value_scale: float = 1.0,
    fixed_time: float | None = None,
    input_function: Callable[[Tensor], Tensor] | None = None,
    device: str = "cpu",
) -> OperatorTrainResult:
    """Train DeepONet or FNO on a designed PDE and local observations."""

    if not isinstance(problem, PDEProblem):
        raise TypeError("problem must be an openpinn.PDEProblem.")
    config = _as_operator_config(source, {})
    mode = str(config["training_mode"])
    if mode not in TRAINING_MODES:
        raise ValueError(f"training_mode must be one of {list(TRAINING_MODES)!r}.")
    loaded = _as_observation(
        observation,
        coordinate_columns=coordinate_columns,
        value_column=value_column,
        coordinate_scales=coordinate_scales,
        value_scale=value_scale,
        fixed_time=fixed_time,
    )
    if mode in ("data_only", "hybrid") and loaded is None:
        raise ValueError(f"training_mode={mode!r} requires an observation CSV.")
    if loaded is not None:
        _validate_observation(problem, loaded)
    if config["operator"] == "deeponet":
        return _train_coordinate(config, problem, loaded, device=device)
    if config["operator"] == "fno":
        return _train_grid(
            config,
            problem,
            None if mode == "physics_only" else loaded,
            device=device,
            input_function=input_function,
        )
    raise ValueError("Designed-problem training supports DeepONet and FNO only.")


def evaluate_operator_reference_csv(
    result: OperatorTrainResult,
    path: str | Path,
    *,
    coordinate_columns: Sequence[str | int] = ("x",),
    value_column: str | int = "u",
    coordinate_scales: Sequence[float] | None = None,
    value_scale: float = 1.0,
    prediction_scale: float = 1.0,
    field_index: int = 0,
    fixed_time: float | None = None,
) -> dict[str, float]:
    """Evaluate an operator result on CSV points without using them for training."""

    if result.evaluator is None:
        raise TypeError("This operator result does not expose designed-problem evaluation.")
    if field_index != 0:
        raise ValueError("Designed operator reference evaluation currently has scalar field_index=0.")
    if not math.isfinite(prediction_scale) or prediction_scale == 0:
        raise ValueError("prediction_scale must be finite and non-zero.")
    observation = load_operator_observation_csv(
        path,
        coordinate_columns=coordinate_columns,
        value_column=value_column,
        coordinate_scales=coordinate_scales,
        value_scale=value_scale,
        fixed_time=fixed_time,
    )
    with torch.no_grad():
        prediction = (
            result.evaluator(observation.coordinates).detach().cpu().numpy().reshape(-1)
            * prediction_scale
        )
    reference = observation.values.numpy().reshape(-1)
    metrics = {
        "relative_l2": relative_l2_error(prediction, reference),
        "rmse": rmse(prediction, reference),
        "mae": mean_absolute_error(prediction, reference),
    }
    result.metrics.update({f"external_reference_{key}": value for key, value in metrics.items()})
    return metrics


__all__ = [
    "CoordinateProblemOnGrid",
    "evaluate_operator_reference_csv",
    "train_operator_problem",
]
