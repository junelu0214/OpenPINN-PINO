"""Unstacked DeepONet coordinate operator.

The branch network encodes an input function sampled at sensors. The trunk
network encodes query coordinates. Their latent dot product, plus an optional
bias, is the output field. This is an independent PyTorch implementation of
the architecture in Lu, Jin, Pang, Zhang, and Karniadakis, Nature Machine
Intelligence (2021), DOI ``10.1038/s42256-021-00302-5``. It does not copy a
reference repository.

``predict`` follows :class:`openpinn.CoordinateOperator`: coordinates have shape
``(n_points, n_dims)`` and the optional ``condition`` tensor is one shared
sensor vector ``(n_sensors,)`` or a per-query sensor matrix
``(n_points, n_sensors)``. ``condition=None`` uses a learned default sensor
vector so the module can still be evaluated as a coordinate network.
``forward(..., branch=...)`` remains a compatibility alias for one minor
version.

``solve`` stays on the PINN catalog. Select this operator with
:func:`build_operator_config` and :func:`train_operator`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor, nn

from ..models import MLP
from .selection import OperatorTrainResult


_ACTIVATIONS = ("tanh", "relu", "gelu", "silu", "sigmoid")
_TRAINING_MODES = ("data_only", "physics_only", "hybrid")
_METADATA_KEYS = frozenset({"operator", "kind", "derivative_mode", "title"})


def _parameter(
    label: str,
    kind: str,
    default: Any,
    *,
    description: str,
    effect: str,
    minimum: float | None = None,
    choices: Sequence[Any] = (),
    ui_hidden: bool = False,
) -> dict[str, Any]:
    record = {
        "label": label,
        "kind": kind,
        "default": default,
        "description": description,
        "effect": effect,
        "minimum": minimum,
        "choices": list(choices),
    }
    if ui_hidden:
        record["ui_hidden"] = True
    return record


DEEPONET_PARAMETER_SCHEMA: dict[str, dict[str, Any]] = {
    "training_mode": _parameter(
        "Training mode",
        "choice",
        "data_only",
        description="Choose supervised data, PDE residual only, or their hybrid.",
        effect="Physics modes differentiate predictions at query coordinates with autodiff.",
        choices=_TRAINING_MODES,
    ),
    "n_sensors": _parameter(
        "Sensor count",
        "integer",
        16,
        description="Number of points where each input function is sampled.",
        effect="More sensors resolve the input function more finely and widen the branch network.",
        minimum=2,
    ),
    "n_dims": _parameter(
        "Query dimension",
        "integer",
        1,
        description="Coordinate width of each trunk query.",
        effect="Must match the query tensor's last dimension. The built-in trainer uses 1.",
        minimum=1,
    ),
    "n_fields": _parameter(
        "Output fields",
        "integer",
        1,
        description="Number of scalar fields returned at each query.",
        effect="The branch width grows by this factor. The built-in trainer uses 1.",
        minimum=1,
    ),
    "latent_dim": _parameter(
        "Latent width",
        "integer",
        16,
        description="Width of the branch/trunk dot product.",
        effect="A wider latent space can represent richer operators and costs more parameters.",
        minimum=1,
    ),
    "branch_hidden": _parameter(
        "Branch hidden widths",
        "integer_list",
        [32, 32],
        description="Hidden-layer widths of the sensor encoder. An empty list is one linear map.",
        effect="Deeper or wider layers increase the branch capacity and the training cost.",
    ),
    "trunk_hidden": _parameter(
        "Trunk hidden widths",
        "integer_list",
        [32, 32],
        description="Hidden-layer widths of the coordinate encoder. An empty list is one linear map.",
        effect="Deeper or wider layers increase the trunk capacity and the training cost.",
    ),
    "activation": _parameter(
        "Activation",
        "choice",
        "tanh",
        description="Hidden-layer activation shared by the branch and the trunk.",
        effect="tanh is the usual smooth choice. relu, gelu, silu, and sigmoid are also accepted.",
        choices=_ACTIVATIONS,
    ),
    "bias": _parameter(
        "Output bias",
        "boolean",
        True,
        description="Add a learned bias after the branch/trunk dot product.",
        effect="Turning it off forces the latent product alone to match the output scale.",
    ),
    "lr": _parameter(
        "Learning rate",
        "number",
        1e-3,
        description="Adam step size used by train_operator.",
        effect="Larger values move faster on the synthetic batch and can overshoot.",
        minimum=1e-12,
    ),
    "data_weight": _parameter(
        "Data weight",
        "number",
        1.0,
        description="Multiplier on labeled MSE in hybrid mode.",
        effect="Ignored by physics_only; data_only uses its existing supervised smoke.",
        minimum=0.0,
    ),
    "physics_weight": _parameter(
        "Physics weight",
        "number",
        1.0,
        description="Multiplier on the coordinate-point PDE residual objective.",
        effect="Must be positive in physics_only mode.",
        minimum=0.0,
    ),
    "data_pretrain_steps": _parameter(
        "Data pretrain steps",
        "integer",
        0,
        description="Hybrid-only supervised updates before joint training.",
        effect="Clamped so at least one physics update remains.",
        minimum=0,
    ),
    "steps": _parameter(
        "Training steps",
        "integer",
        20,
        description="Number of Adam updates on one fixed synthetic batch.",
        effect="More steps fit that batch more closely. This is a smoke trainer, not a benchmark.",
        minimum=1,
    ),
    "batch_size": _parameter(
        "Batch size",
        "integer",
        8,
        description="Number of input functions in the fixed synthetic batch.",
        effect="A larger batch averages more input functions into each step.",
        minimum=1,
    ),
    "n_query": _parameter(
        "Queries per function",
        "integer",
        16,
        description="Number of output locations supervised for each input function.",
        effect="More queries supervise the trunk at more coordinates.",
        minimum=1,
    ),
    "seed": _parameter(
        "Seed",
        "integer",
        0,
        description="Seed for network initialization and the synthetic batch.",
        effect="Changing it draws a different initialization and a different batch.",
        minimum=0,
    ),
    "n_modes": _parameter(
        "Fourier modes",
        "integer",
        3,
        description="Number of sine and cosine modes in each synthetic input function.",
        effect="More modes make the antiderivative operator less smooth.",
        minimum=1,
        ui_hidden=True,
    ),
}


def _require_int(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return int(value)


def _require_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _widths(name: str, value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        parsed: list[Any] = []
        for part in parts:
            try:
                parsed.append(int(part))
            except ValueError as exc:
                raise ValueError(f"{name} entries must be positive integers.") from exc
        value = parsed
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list of positive integers.")
    widths: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or int(item) <= 0:
            raise ValueError(f"{name} entries must be positive integers.")
        widths.append(int(item))
    return tuple(widths)


@dataclass(frozen=True)
class DeepONetConfig:
    """Network widths for one unstacked DeepONet."""

    n_sensors: int = 16
    n_dims: int = 1
    n_fields: int = 1
    latent_dim: int = 16
    branch_hidden: tuple[int, ...] = (32, 32)
    trunk_hidden: tuple[int, ...] = (32, 32)
    activation: str = "tanh"
    bias: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "n_sensors", _require_int("n_sensors", self.n_sensors, minimum=2)
        )
        object.__setattr__(self, "n_dims", _require_int("n_dims", self.n_dims, minimum=1))
        object.__setattr__(
            self, "n_fields", _require_int("n_fields", self.n_fields, minimum=1)
        )
        object.__setattr__(
            self, "latent_dim", _require_int("latent_dim", self.latent_dim, minimum=1)
        )
        object.__setattr__(self, "branch_hidden", _widths("branch_hidden", self.branch_hidden))
        object.__setattr__(self, "trunk_hidden", _widths("trunk_hidden", self.trunk_hidden))
        object.__setattr__(self, "bias", _require_bool("bias", self.bias))
        activation = str(self.activation).lower()
        if activation not in _ACTIVATIONS:
            choices = ", ".join(_ACTIVATIONS)
            raise ValueError(f"activation must be one of: {choices}.")
        object.__setattr__(self, "activation", activation)


class DeepONet(nn.Module):
    """Branch/trunk map from sensors and coordinates to field values."""

    def __init__(self, config: DeepONetConfig | None = None, /, **overrides: Any) -> None:
        super().__init__()
        selected = config if config is not None else DeepONetConfig()
        if not isinstance(selected, DeepONetConfig):
            raise TypeError("DeepONet config must be a DeepONetConfig.")
        if overrides:
            selected = replace(selected, **overrides)
        self.config = selected
        self.n_sensors = selected.n_sensors
        self.n_dims = selected.n_dims
        self.n_fields = selected.n_fields
        self.latent_dim = selected.latent_dim
        self.branch_net = MLP(
            selected.n_sensors,
            selected.latent_dim * selected.n_fields,
            hidden_layers=selected.branch_hidden,
            activation=selected.activation,
        )
        self.trunk_net = MLP(
            selected.n_dims,
            selected.latent_dim,
            hidden_layers=selected.trunk_hidden,
            activation=selected.activation,
        )
        if selected.bias:
            self.output_bias = nn.Parameter(torch.zeros(selected.n_fields))
        else:
            self.output_bias = None
        self.default_sensors = nn.Parameter(torch.zeros(selected.n_sensors))

    def derivative_mode(self) -> str:
        """Return ``autodiff``. Trunk outputs depend on the coordinate tensor."""

        return "autodiff"

    def predict(self, coordinates: Tensor, condition: Tensor | None = None) -> Tensor:
        """Evaluate the field at ``coordinates``.

        Returns shape ``(n_points, n_fields)``.
        """

        if coordinates.ndim != 2 or coordinates.shape[1] != self.n_dims:
            raise ValueError(
                f"DeepONet coordinates must have shape (n_points, {self.n_dims}), "
                f"got {tuple(coordinates.shape)}."
            )
        n_points = int(coordinates.shape[0])
        sensors = self._branch_sensors(condition, n_points)
        if sensors.shape[0] == 1:
            values = self.evaluate(sensors, coordinates)
            return values[0]
        branch_code = self.branch_net(sensors).reshape(
            n_points, self.n_fields, self.latent_dim
        )
        trunk = self.trunk_net(coordinates)
        prediction = torch.einsum("pfk,pk->pf", branch_code, trunk)
        return self._add_bias(prediction)

    def forward(self, coordinates: Tensor, branch: Tensor | None = None) -> Tensor:
        """Compatibility alias for :meth:`predict`; use ``condition`` there."""

        return self.predict(coordinates, condition=branch)

    def evaluate(self, sensors: Tensor, coordinates: Tensor) -> Tensor:
        """Evaluate a batch of input functions.

        ``sensors`` has shape ``(batch, n_sensors)``. ``coordinates`` is shared
        across the batch with shape ``(n_query, n_dims)``, or per example with
        shape ``(batch, n_query, n_dims)``. The result has shape
        ``(batch, n_query, n_fields)``.
        """

        if sensors.ndim != 2 or sensors.shape[1] != self.n_sensors:
            raise ValueError(
                f"DeepONet sensors must have shape (batch, {self.n_sensors}), "
                f"got {tuple(sensors.shape)}."
            )
        batch = int(sensors.shape[0])
        branch_code = self.branch_net(sensors).reshape(batch, self.n_fields, self.latent_dim)
        if coordinates.ndim == 2:
            if coordinates.shape[1] != self.n_dims:
                raise ValueError(
                    f"DeepONet coordinates must have shape (n_query, {self.n_dims}), "
                    f"got {tuple(coordinates.shape)}."
                )
            trunk = self.trunk_net(coordinates)
            prediction = torch.einsum("bfk,qk->bqf", branch_code, trunk)
        elif coordinates.ndim == 3:
            if coordinates.shape[0] != batch or coordinates.shape[2] != self.n_dims:
                raise ValueError(
                    "Per-example coordinates must have shape "
                    f"({batch}, n_query, {self.n_dims}), got {tuple(coordinates.shape)}."
                )
            n_query = int(coordinates.shape[1])
            trunk = self.trunk_net(coordinates.reshape(batch * n_query, self.n_dims))
            trunk = trunk.reshape(batch, n_query, self.latent_dim)
            prediction = torch.einsum("bfk,bqk->bqf", branch_code, trunk)
        else:
            raise ValueError(
                "DeepONet coordinates must have shape (n_query, n_dims) or "
                f"(batch, n_query, n_dims), got {tuple(coordinates.shape)}."
            )
        return self._add_bias(prediction)

    def _branch_sensors(self, condition: Tensor | None, n_points: int) -> Tensor:
        if condition is None:
            return self.default_sensors.reshape(1, self.n_sensors)
        if condition.ndim == 1:
            if condition.shape[0] != self.n_sensors:
                raise ValueError(
                    f"Shared condition/branch must have shape ({self.n_sensors},), "
                    f"got {tuple(condition.shape)}."
                )
            return condition.reshape(1, self.n_sensors)
        if condition.ndim == 2 and condition.shape == (n_points, self.n_sensors):
            return condition
        raise ValueError(
            "condition/branch must be None, a shared sensor vector of shape "
            f"({self.n_sensors},), or a per-query matrix of shape "
            f"({n_points}, {self.n_sensors}); got {tuple(condition.shape)}."
        )

    def _add_bias(self, prediction: Tensor) -> Tensor:
        if self.output_bias is None:
            return prediction
        bias_shape = (1,) * (prediction.ndim - 1) + (self.n_fields,)
        return prediction + self.output_bias.reshape(bias_shape)


def _network_config(options: Mapping[str, Any]) -> DeepONetConfig:
    return DeepONetConfig(
        n_sensors=int(options["n_sensors"]),
        n_dims=int(options["n_dims"]),
        n_fields=int(options["n_fields"]),
        latent_dim=int(options["latent_dim"]),
        branch_hidden=tuple(int(width) for width in options["branch_hidden"]),
        trunk_hidden=tuple(int(width) for width in options["trunk_hidden"]),
        activation=str(options["activation"]),
        bias=bool(options["bias"]),
    )


def build_deeponet(config: Mapping[str, Any] | DeepONetConfig | None = None, /, **overrides: Any) -> DeepONet:
    """Construct a DeepONet from a config mapping, a :class:`DeepONetConfig`, or keywords."""

    from .selection import build_operator_config

    if isinstance(config, DeepONetConfig):
        return DeepONet(config, **overrides)
    if config is None:
        source: dict[str, Any] = {}
    elif isinstance(config, Mapping):
        source = {
            key: value
            for key, value in config.items()
            if key not in _METADATA_KEYS
        }
    else:
        raise TypeError("build_deeponet expects a mapping, DeepONetConfig, or keyword options.")
    source.update(overrides)
    resolved = build_operator_config("deeponet", **source)
    return DeepONet(_network_config(resolved))


def antiderivative_batch(
    batch_size: int,
    n_sensors: int,
    n_query: int,
    *,
    generator: torch.Generator,
    n_modes: int = 3,
    query_x: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Draw one batch of the 1D antiderivative operator.

    Each input function is a random Fourier series on ``[0, 1]``. Sensor
    values are that series on a uniform grid. Targets are the exact integral
    from 0 to each query coordinate, with shape ``(batch, n_query, 1)``.
    Shared query coordinates have shape ``(n_query, 1)``. Pass ``query_x``
    with shape ``(n_query,)`` to supervise specific locations instead of
    drawing them.
    """

    batch = _require_int("batch_size", batch_size, minimum=1)
    sensors = _require_int("n_sensors", n_sensors, minimum=2)
    modes = _require_int("n_modes", n_modes, minimum=1)
    frequencies = torch.arange(1, modes + 1, dtype=torch.float32)
    sine_amplitude = torch.randn(batch, modes, generator=generator)
    cosine_amplitude = torch.randn(batch, modes, generator=generator)
    sensor_x = torch.linspace(0.0, 1.0, sensors)
    if query_x is None:
        queries = _require_int("n_query", n_query, minimum=1)
        query_positions = torch.rand(queries, generator=generator)
    else:
        if query_x.ndim != 1 or query_x.shape[0] < 1:
            raise ValueError("query_x must have shape (n_query,).")
        query_positions = query_x.detach().to(dtype=torch.float32).reshape(-1)
        queries = _require_int("n_query", n_query, minimum=1)
        if int(query_positions.shape[0]) != queries:
            raise ValueError("n_query must match query_x.shape[0].")
    scale = 1.0 / (2.0 * math.pi * frequencies)

    def _angles(positions: Tensor) -> Tensor:
        return (2.0 * math.pi) * frequencies.view(1, -1) * positions.view(-1, 1)

    sensor_angle = _angles(sensor_x)
    sensor_values = sine_amplitude @ torch.sin(sensor_angle).transpose(0, 1)
    sensor_values = sensor_values + cosine_amplitude @ torch.cos(sensor_angle).transpose(0, 1)

    query_angle = _angles(query_positions)
    sine = torch.sin(query_angle)
    one_minus_cosine = 1.0 - torch.cos(query_angle)
    integral = (sine_amplitude * scale.view(1, -1)) @ one_minus_cosine.transpose(0, 1)
    integral = integral + (cosine_amplitude * scale.view(1, -1)) @ sine.transpose(0, 1)
    return sensor_values, query_positions.unsqueeze(-1), integral.unsqueeze(-1)


def train_deeponet(config: Mapping[str, Any], *, device: str = "cpu") -> OperatorTrainResult:
    """Train DeepONet for a few Adam steps on a fixed antiderivative batch.

    The batch is drawn once and reused, so the loss is the mean squared error
    of that smoke sample. ``n_dims`` and ``n_fields`` must stay at 1 because
    this dataset is the scalar 1D antiderivative. Other shapes can still be
    constructed with :func:`build_deeponet` and trained by the caller.
    """

    if int(config["n_dims"]) != 1 or int(config["n_fields"]) != 1:
        raise ValueError(
            "train_operator supports the 1D scalar antiderivative "
            "(n_dims=1, n_fields=1). Use build_deeponet for other shapes."
        )
    selected_device = str(device)
    torch.manual_seed(int(config["seed"]))
    model = build_deeponet(config).to(selected_device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config["seed"]))
    sensors, coordinates, target = antiderivative_batch(
        int(config["batch_size"]),
        int(config["n_sensors"]),
        int(config["n_query"]),
        generator=generator,
        n_modes=int(config["n_modes"]),
    )
    sensors = sensors.to(selected_device)
    coordinates = coordinates.to(selected_device)
    target = target.to(selected_device)

    def _loss() -> Tensor:
        return torch.mean((model.evaluate(sensors, coordinates) - target) ** 2)

    model.train()
    with torch.no_grad():
        initial_loss = float(_loss())
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    history: list[float] = []
    for _ in range(int(config["steps"])):
        optimizer.zero_grad(set_to_none=True)
        loss = _loss()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        final_loss = float(_loss())
    return OperatorTrainResult(
        operator="deeponet",
        steps=int(config["steps"]),
        initial_loss=initial_loss,
        final_loss=final_loss,
        history=tuple(history),
        model=model,
        config=dict(config),
    )


def train_deeponet_physics(
    config: Mapping[str, Any], *, device: str = "cpu"
) -> OperatorTrainResult:
    """Train DeepONet with a coordinate-autodiff Poisson residual.

    This fixed 1D manufactured batch is a short execution smoke, not an
    accuracy benchmark. ``physics_only`` never places its labeled solution in
    the objective; the target is retained only for offline diagnostics.
    """

    from .coordinate_pde import (
        periodic_poisson_coordinate_batch,
        periodic_poisson_coordinate_residual,
    )

    if int(config["n_dims"]) != 1 or int(config["n_fields"]) != 1:
        raise ValueError(
            "DeepONet coordinate physics supports n_dims=1 and n_fields=1 "
            "for the built-in smoke."
        )
    if int(config["n_sensors"]) < 2 * int(config["n_modes"]):
        raise ValueError(
            "n_sensors must be at least 2 * n_modes for the DeepONet "
            "coordinate-Poisson smoke."
        )
    mode = str(config["training_mode"])
    if mode not in ("physics_only", "hybrid"):
        raise ValueError("coordinate physics requires training_mode='physics_only' or 'hybrid'.")
    data_weight = float(config["data_weight"])
    physics_weight = float(config["physics_weight"])
    if mode == "hybrid" and data_weight == 0.0 and physics_weight == 0.0:
        raise ValueError("hybrid mode needs a positive data_weight or physics_weight.")
    if mode == "physics_only" and physics_weight == 0.0:
        raise ValueError("physics_only mode needs a positive physics_weight.")

    selected_device = str(device)
    torch.manual_seed(int(config["seed"]))
    model = build_deeponet(config).to(selected_device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config["seed"]))
    sensors, coordinates, forcing, target = periodic_poisson_coordinate_batch(
        int(config["batch_size"]),
        int(config["n_sensors"]),
        int(config["n_query"]),
        generator=generator,
        n_harmonics=int(config["n_modes"]),
    )
    sensors = sensors.to(selected_device)
    coordinates = coordinates.to(selected_device)
    forcing = forcing.to(selected_device)
    target = target.to(selected_device)
    steps = int(config["steps"])
    requested_pretrain = int(config["data_pretrain_steps"])
    pretrain = (
        min(max(requested_pretrain, 0), max(steps - 1, 0))
        if mode == "hybrid"
        else 0
    )

    def _terms(*, create_graph: bool) -> tuple[Tensor, Tensor, Tensor]:
        query = coordinates.detach().clone().requires_grad_(True)
        predictions: list[Tensor] = []
        residuals: list[Tensor] = []
        for condition, sample_forcing in zip(sensors, forcing, strict=True):
            prediction = model.predict(query, condition=condition)
            residual = periodic_poisson_coordinate_residual(
                prediction,
                query,
                sample_forcing,
                create_graph=create_graph,
            )
            predictions.append(prediction)
            residuals.append(residual)
        stacked_prediction = torch.stack(predictions, dim=0)
        stacked_residual = torch.stack(residuals, dim=0)
        data = torch.mean((stacked_prediction - target) ** 2)
        physics = torch.mean(stacked_residual**2)
        centered_prediction = stacked_prediction - stacked_prediction.mean(dim=1, keepdim=True)
        centered_target = target - target.mean(dim=1, keepdim=True)
        centered = torch.mean((centered_prediction - centered_target) ** 2)
        return data, physics, centered

    def _objective(step_index: int, *, create_graph: bool) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        data, physics, centered = _terms(create_graph=create_graph)
        if mode == "physics_only":
            objective = physics_weight * physics
        elif step_index < pretrain:
            objective = data_weight * data
        else:
            objective = data_weight * data + physics_weight * physics
        return objective, data, physics, centered

    model.train()
    initial_objective, initial_data, initial_physics, initial_centered = _objective(
        0, create_graph=False
    )
    initial_loss = float(initial_objective.detach())
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    history: list[float] = []
    for step_index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _data, _physics, _centered = _objective(step_index, create_graph=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))

    model.eval()
    final_objective, final_data, final_physics, final_centered = _objective(
        steps - 1, create_graph=False
    )
    return OperatorTrainResult(
        operator="deeponet",
        steps=steps,
        initial_loss=initial_loss,
        final_loss=float(final_objective.detach()),
        history=tuple(history),
        model=model,
        config=dict(config),
        metrics={
            "data_pretrain_steps": float(pretrain),
            "initial_data_loss": float(initial_data.detach()),
            "final_data_loss": float(final_data.detach()),
            "initial_physics_loss": float(initial_physics.detach()),
            "final_physics_loss": float(final_physics.detach()),
            "initial_boundary_loss": 0.0,
            "final_boundary_loss": 0.0,
            "initial_centered_data_loss": float(initial_centered.detach()),
            "final_centered_data_loss": float(final_centered.detach()),
        },
    )
