"""Extensible physics-informed neural-operator training.

PINO is an FNO whose training objective can include a structured-grid PDE
residual. ``predict(inputs, *, grid=...)`` is the same
:class:`openpinn.GridOperator` map as FNO; ``forward`` remains a compatibility
alias for one minor version. The residual uses :class:`openpinn.GridDerivative`
(spectral or periodic finite difference), not coordinate autodiff.

The implementation helper learns the 1D periodic Poisson solution operator
``f ↦ u`` for ``-u_xx = f`` on ``[0, 2π)``. It supports two physics modes:

* ``hybrid`` (default): optional data-only pretrain, then a data term plus
  the grid residual. This is the practical path.
* ``physics_only``: no labeled solution enters the objective. For the
  spectral 1D periodic Poisson trainer, the residual is weighted by the
  inverse Laplacian (``1/k²``, with the zero mode removed). This fixes the
  constant gauge and makes the loss equivalent to centered solution error.

This is an independent PyTorch implementation of the hybrid data-plus-
physics idea in Li, Zheng, Kovachki, Jin, Chen, Liu, Azizzadenesheli, and
Anandkumar, arXiv ``2111.03794``. Its public evaluation signature is aligned
with the work repository; that repository's implementation is not copied.

The training loop also accepts a structural :class:`GridPDEProblem` and an
injected trainable :class:`openpinn.GridOperator`. ``solve`` and the coordinate
PINN ``PDEProblem`` stay unchanged. The public configurator selects an FNO or
DeepONet model plus ``training_mode``; PINO is not a model option. This loop is
used by FNO physics, while DeepONet physics stays coordinate-native. Call
:func:`train_pino` directly for the advanced grid injection points.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor, nn

from .fno import FNO, FNOConfig
from ..operator_protocol import GridOperator
from .selection import OperatorTrainResult, build_operator_config
from .pde_problem import GridPDEProblem, PeriodicPoisson1D
from .uniform_grid import UniformGrid


_ACTIVATIONS = ("gelu", "tanh", "relu", "silu")
_MODES = ("hybrid", "physics_only")
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
) -> dict[str, Any]:
    return {
        "label": label,
        "kind": kind,
        "default": default,
        "description": description,
        "effect": effect,
        "minimum": minimum,
        "choices": list(choices),
    }


PINO_PARAMETER_SCHEMA: dict[str, dict[str, Any]] = {
    "n_dims": _parameter(
        "Spatial dimensions",
        "integer",
        1,
        description="Number of structured-grid axes. 1 and 2 are implemented.",
        effect="Must match the grid tensor's spatial rank. The built-in trainer uses 1.",
        minimum=1,
    ),
    "in_channels": _parameter(
        "Input channels",
        "integer",
        1,
        description="Channel count of the input forcing field.",
        effect="Must match grid_field.shape[1]. The built-in trainer uses 1.",
        minimum=1,
    ),
    "out_channels": _parameter(
        "Output channels",
        "integer",
        1,
        description="Channel count of the predicted solution field.",
        effect="The FNO projection writes this many channels. The built-in trainer uses 1.",
        minimum=1,
    ),
    "width": _parameter(
        "Lifted width",
        "integer",
        8,
        description="Channel width inside the Fourier layers of the FNO backbone.",
        effect="A wider latent grid increases capacity and cost.",
        minimum=1,
    ),
    "n_layers": _parameter(
        "Fourier layers",
        "integer",
        2,
        description="Number of spectral-convolution blocks in the FNO backbone.",
        effect="More layers compose more Fourier multipliers.",
        minimum=1,
    ),
    "n_modes": _parameter(
        "Fourier modes",
        "integer",
        4,
        description="Number of retained low-frequency modes on each spatial axis.",
        effect="Must fit the grid: each axis needs at least 2 * n_modes points.",
        minimum=1,
    ),
    "project_width": _parameter(
        "Projection width",
        "integer",
        16,
        description="Hidden width of the two-layer projection back to output channels.",
        effect="A wider projection can mix lifted channels more expressively.",
        minimum=1,
    ),
    "activation": _parameter(
        "Activation",
        "choice",
        "gelu",
        description="Nonlinearity after each Fourier block and the first projection layer.",
        effect="gelu is the usual FNO choice. tanh, relu, and silu are also accepted.",
        choices=_ACTIVATIONS,
    ),
    "mode": _parameter(
        "Training mode",
        "choice",
        "hybrid",
        description=(
            "hybrid uses a data term plus the grid PDE residual. physics_only "
            "drops the data term and minimizes the residual only."
        ),
        effect=(
            "hybrid is the practical default. For spectral 1D periodic Poisson, "
            "physics_only uses an inverse-Laplacian residual norm and recovers "
            "the solution up to its constant gauge."
        ),
        choices=_MODES,
    ),
    "data_weight": _parameter(
        "Data weight",
        "number",
        1.0,
        description="Multiplier on the supervised MSE. Ignored in physics_only mode.",
        effect="Larger values keep the hybrid trainer closer to the labeled batch.",
        minimum=0.0,
    ),
    "physics_weight": _parameter(
        "Physics weight",
        "number",
        1.0,
        description=(
            "Multiplier on the Poisson physics objective: raw residual MSE in "
            "hybrid, or the inverse-Laplacian norm in spectral physics_only."
        ),
        effect=(
            "Hybrid scales the raw residual MSE. Spectral physics_only scales the "
            "inverse-Laplacian residual norm, removing k² mode imbalance."
        ),
        minimum=0.0,
    ),
    "data_pretrain_steps": _parameter(
        "Data pretrain steps",
        "integer",
        0,
        description=(
            "Hybrid only: data-only Adam steps before the data-plus-physics fine-tune. "
            "Clamped so at least one physics step remains when steps > 0."
        ),
        effect="Set above 0 for a data-then-physics fine-tune. 0 is simultaneous hybrid.",
        minimum=0,
    ),
    "lr": _parameter(
        "Learning rate",
        "number",
        1e-3,
        description="Adam step size used by train_operator.",
        effect="Larger values move faster on the synthetic batch and can overshoot.",
        minimum=1e-12,
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
        4,
        description="Number of forcing functions in the fixed synthetic batch.",
        effect="A larger batch averages more input functions into each step.",
        minimum=1,
    ),
    "grid_size": _parameter(
        "Grid size",
        "integer",
        32,
        description="Number of periodic samples on each trained axis.",
        effect="Must be at least 2 * n_modes. The built-in trainer is 1D.",
        minimum=4,
    ),
    "n_harmonics": _parameter(
        "Harmonics",
        "integer",
        2,
        description="Highest integer wave number in each manufactured solution.",
        effect="More harmonics add nontrivial solution and forcing modes to the smoke batch.",
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
    "derivative": _parameter(
        "Grid derivative",
        "choice",
        "spectral",
        description="Discrete derivative used to form the Poisson residual of the output grid.",
        effect="spectral uses a Fourier multiplier. finite_difference uses periodic central differences.",
        choices=("spectral", "finite_difference"),
    ),
}


def _require_int(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return int(value)


@dataclass(frozen=True)
class PINOConfig:
    """FNO backbone widths for one physics-informed grid operator."""

    n_dims: int = 1
    in_channels: int = 1
    out_channels: int = 1
    width: int = 8
    n_layers: int = 2
    n_modes: int = 4
    project_width: int = 16
    activation: str = "gelu"

    def __post_init__(self) -> None:
        backbone = self.backbone_config()
        object.__setattr__(self, "n_dims", backbone.n_dims)
        object.__setattr__(self, "in_channels", backbone.in_channels)
        object.__setattr__(self, "out_channels", backbone.out_channels)
        object.__setattr__(self, "width", backbone.width)
        object.__setattr__(self, "n_layers", backbone.n_layers)
        object.__setattr__(self, "n_modes", backbone.n_modes)
        object.__setattr__(self, "project_width", backbone.project_width)
        object.__setattr__(self, "activation", backbone.activation)

    def backbone_config(self) -> FNOConfig:
        """Return the FNO widths used by this operator."""

        return FNOConfig(
            n_dims=self.n_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            width=self.width,
            n_layers=self.n_layers,
            n_modes=self.n_modes,
            project_width=self.project_width,
            activation=self.activation,
        )


class PINO(nn.Module):
    """FNO grid operator plus a GridDerivative Poisson residual."""

    def __init__(self, config: PINOConfig | FNOConfig | None = None, /, **overrides: Any) -> None:
        super().__init__()
        if isinstance(config, FNOConfig):
            selected = PINOConfig(
                n_dims=config.n_dims,
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                width=config.width,
                n_layers=config.n_layers,
                n_modes=config.n_modes,
                project_width=config.project_width,
                activation=config.activation,
            )
        else:
            selected = config if config is not None else PINOConfig()
        if not isinstance(selected, PINOConfig):
            raise TypeError("PINO config must be a PINOConfig or FNOConfig.")
        if overrides:
            selected = replace(selected, **overrides)
        self.config = selected
        self.backbone = FNO(selected.backbone_config())
        self.n_dims = selected.n_dims
        self.in_channels = selected.in_channels
        self.out_channels = selected.out_channels
        self.width = selected.width
        self.n_modes = selected.n_modes

    def derivative_mode(self) -> str:
        """Return ``grid``. PDE residuals use :class:`~openpinn.GridDerivative`."""

        return "grid"

    def grid_derivative(self, kind: str = "spectral", *, spacing: float | Sequence[float] = 1.0):
        """Return a discrete derivative for this operator's output grid."""

        return self.backbone.grid_derivative(kind, spacing=spacing)

    def pde_residual(
        self,
        solution: Tensor,
        forcing: Tensor,
        *,
        spacing: float | Sequence[float],
        derivative: str = "spectral",
    ) -> Tensor:
        """Return the periodic Poisson residual ``-Δu - f``.

        ``solution`` and ``forcing`` have shape ``(batch, channels, *spatial)``.
        The Laplacian is the sum of second derivatives along every spatial axis.
        """

        if tuple(solution.shape) != tuple(forcing.shape):
            raise ValueError(
                "PINO residual fields must share a shape, got "
                f"{tuple(solution.shape)} and {tuple(forcing.shape)}."
            )
        grid = UniformGrid.from_shape(
            solution.shape[2:],
            spacing=spacing,
            periodic=True,
            device=solution.device,
            dtype=solution.dtype,
        )
        return PeriodicPoisson1D().residual(
            solution, forcing, grid=grid, derivative=derivative
        )

    def predict(self, inputs: Tensor, *, grid: object) -> Tensor:
        """Evaluate the FNO backbone using keyword-only grid metadata."""

        return self.backbone.predict(inputs, grid=grid)

    def forward(self, grid_field: Tensor) -> Tensor:
        """Compatibility alias for the historical grid-free call."""

        return self.backbone(grid_field)


def _network_config(options: Mapping[str, Any]) -> PINOConfig:
    return PINOConfig(
        n_dims=int(options["n_dims"]),
        in_channels=int(options["in_channels"]),
        out_channels=int(options["out_channels"]),
        width=int(options["width"]),
        n_layers=int(options["n_layers"]),
        n_modes=int(options["n_modes"]),
        project_width=int(options["project_width"]),
        activation=str(options["activation"]),
    )


def build_pino(config: Mapping[str, Any] | PINOConfig | FNOConfig | None = None, /, **overrides: Any) -> PINO:
    """Construct a PINO from a config mapping, a :class:`PINOConfig`, or keywords."""

    if isinstance(config, (PINOConfig, FNOConfig)):
        return PINO(config, **overrides)
    if config is None:
        source: dict[str, Any] = {}
    elif isinstance(config, Mapping):
        source = {key: value for key, value in config.items() if key not in _METADATA_KEYS}
    else:
        raise TypeError("build_pino expects a mapping, PINOConfig, FNOConfig, or keyword options.")
    source.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        resolved = build_operator_config("pino", **source)
    return PINO(_network_config(resolved))


def poisson_operator_batch(
    batch_size: int,
    grid_size: int,
    *,
    generator: torch.Generator,
    n_harmonics: int = 2,
) -> tuple[Tensor, Tensor, Tensor]:
    """Draw one batch of the 1D periodic Poisson operator ``-u_xx = f``.

    Each solution is a zero-mean random Fourier series on ``[0, 2π)``. The
    forcing is the exact ``-u_xx``. Returned fields have shape
    ``(batch, 1, grid_size)``. The sample locations have shape ``(grid_size,)``.
    """

    batch = _require_int("batch_size", batch_size, minimum=1)
    size = _require_int("grid_size", grid_size, minimum=4)
    modes = _require_int("n_harmonics", n_harmonics, minimum=1)
    grid_x = torch.linspace(0.0, 2.0 * math.pi, size + 1, dtype=torch.float32)[:-1]
    frequencies = torch.arange(1, modes + 1, dtype=torch.float32)
    sine_amplitude = torch.randn(batch, modes, generator=generator)
    cosine_amplitude = torch.randn(batch, modes, generator=generator)
    angles = frequencies.view(1, -1, 1) * grid_x.view(1, 1, -1)
    solution = sine_amplitude.unsqueeze(-1) * torch.sin(angles)
    solution = solution + cosine_amplitude.unsqueeze(-1) * torch.cos(angles)
    solution = solution.sum(dim=1)
    scale = (frequencies * frequencies).view(1, -1, 1)
    forcing = scale * sine_amplitude.unsqueeze(-1) * torch.sin(angles)
    forcing = forcing + scale * cosine_amplitude.unsqueeze(-1) * torch.cos(angles)
    forcing = forcing.sum(dim=1)
    return grid_x, forcing.unsqueeze(1), solution.unsqueeze(1)


def _mean_square(field: Tensor) -> Tensor:
    return torch.mean(field * field)


def _centered_mean_square(prediction: Tensor, target: Tensor) -> Tensor:
    prediction = prediction - prediction.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    return _mean_square(prediction - target)


def _pretrain_steps(mode: str, steps: int, requested: int) -> int:
    if mode != "hybrid":
        return 0
    return min(max(requested, 0), max(steps - 1, 0))


def train_pino(
    config: Mapping[str, Any],
    *,
    device: str = "cpu",
    problem: GridPDEProblem | None = None,
    model: nn.Module | None = None,
) -> OperatorTrainResult:
    """Train a grid operator for a few Adam steps on a fixed 1D batch.

    The batch is drawn once and reused. ``n_dims``, ``in_channels``, and
    ``out_channels`` must stay at 1 because this dataset is a scalar periodic
    Poisson problem. Other shapes can still be constructed with
    :func:`build_pino` and trained by the caller.

    By default this builds an FNO-backed :class:`PINO` and uses
    :class:`PeriodicPoisson1D`. ``problem`` may replace the PDE residual, its
    optional boundary residual, and its scalar physics reduction. ``model``
    may replace the network with any trainable ``nn.Module`` satisfying
    ``GridOperator.predict(inputs, *, grid=...)``.

    ``physics_only`` never uses the labeled solution in the objective. Both
    modes retain labeled data diagnostics for evaluation. The fixed batch
    remains the built-in smoke data source; custom production data loops can
    reuse the same problem and model contracts.
    """

    if int(config["n_dims"]) != 1 or int(config["in_channels"]) != 1 or int(config["out_channels"]) != 1:
        raise ValueError(
            "train_operator supports the 1D scalar periodic Poisson operator "
            "(n_dims=1, in_channels=1, out_channels=1). Use build_pino for other shapes."
        )
    grid_size = _require_int("grid_size", int(config["grid_size"]), minimum=4)
    n_modes = _require_int("n_modes", int(config["n_modes"]), minimum=1)
    if grid_size < 2 * n_modes:
        raise ValueError("grid_size must be at least 2 * n_modes for the 1D trainer.")

    mode = str(config.get("mode", config.get("training_mode", "hybrid")))
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {list(_MODES)!r}.")
    data_weight = float(config["data_weight"])
    physics_weight = float(config["physics_weight"])
    if mode == "hybrid" and data_weight == 0.0 and physics_weight == 0.0:
        raise ValueError("hybrid mode needs a positive data_weight or physics_weight.")
    if mode == "physics_only" and physics_weight == 0.0:
        raise ValueError("physics_only mode needs a positive physics_weight.")

    selected_problem = PeriodicPoisson1D() if problem is None else problem
    if not isinstance(selected_problem, GridPDEProblem):
        raise TypeError(
            "problem must implement GridPDEProblem residual, boundary_residual, "
            "and physics_objective methods."
        )
    torch.manual_seed(int(config["seed"]))
    selected_model = build_pino(config) if model is None else model
    if not isinstance(selected_model, nn.Module):
        raise TypeError("model must be a torch.nn.Module with trainable parameters.")
    if not isinstance(selected_model, GridOperator):
        raise TypeError("model must implement predict(inputs, *, grid=...).")
    selected_model = selected_model.to(device)
    trainable_parameters = [
        parameter for parameter in selected_model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("model must expose at least one trainable parameter.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config["seed"]))
    grid_x, forcing, target = poisson_operator_batch(
        int(config["batch_size"]),
        grid_size,
        generator=generator,
        n_harmonics=int(config["n_harmonics"]),
    )
    forcing = forcing.to(device)
    target = target.to(device)
    spacing = float(grid_x[1] - grid_x[0])
    grid = UniformGrid(grid_x.to(device), spacing=spacing, periodic=True)
    derivative = str(config["derivative"])
    steps = int(config["steps"])
    pretrain = _pretrain_steps(mode, steps, int(config["data_pretrain_steps"]))

    def _terms() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        prediction = selected_model.predict(forcing, grid=grid)
        data = _mean_square(prediction - target)
        residual = selected_problem.residual(
            prediction, forcing, grid=grid, derivative=derivative
        )
        if not isinstance(residual, Tensor):
            raise TypeError("problem.residual must return a Tensor.")
        if tuple(residual.shape) != tuple(prediction.shape):
            raise ValueError(
                "problem.residual must return the solution shape, got "
                f"{tuple(residual.shape)} instead of {tuple(prediction.shape)}."
            )
        boundary_residual = selected_problem.boundary_residual(
            prediction, forcing, grid=grid
        )
        if boundary_residual is not None and not isinstance(boundary_residual, Tensor):
            raise TypeError("problem.boundary_residual must return a Tensor or None.")
        boundary = (
            prediction.new_zeros(())
            if boundary_residual is None
            else _mean_square(boundary_residual)
        )
        physics = _mean_square(residual) + boundary
        physics_objective = selected_problem.physics_objective(
            residual, grid=grid, mode=mode, derivative=derivative
        )
        if not isinstance(physics_objective, Tensor):
            raise TypeError("problem.physics_objective must return a Tensor.")
        if physics_objective.ndim != 0:
            raise ValueError("problem.physics_objective must return a scalar Tensor.")
        physics_objective = physics_objective + boundary
        centered = _centered_mean_square(prediction, target)
        return data, physics, centered, physics_objective, boundary

    def _objective(step_index: int) -> Tensor:
        data, _physics, _centered, physics_objective, _boundary = _terms()
        if mode == "physics_only":
            return physics_weight * physics_objective
        if step_index < pretrain:
            return data_weight * data
        return data_weight * data + physics_weight * physics_objective

    selected_model.train()
    with torch.no_grad():
        (
            initial_data,
            initial_physics,
            initial_centered,
            _initial_physics_objective,
            initial_boundary,
        ) = (
            float(item) for item in _terms()
        )
        initial_loss = float(_objective(0))
    optimizer = torch.optim.Adam(trainable_parameters, lr=float(config["lr"]))
    history: list[float] = []
    for step_index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _objective(step_index)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    selected_model.eval()
    with torch.no_grad():
        (
            final_data,
            final_physics,
            final_centered,
            _final_physics_objective,
            final_boundary,
        ) = (
            float(item) for item in _terms()
        )
        final_loss = float(_objective(steps - 1))
    return OperatorTrainResult(
        operator="pino",
        steps=steps,
        initial_loss=initial_loss,
        final_loss=final_loss,
        history=tuple(history),
        model=selected_model,
        config=dict(config),
        metrics={
            "data_pretrain_steps": float(pretrain),
            "initial_data_loss": initial_data,
            "final_data_loss": final_data,
            "initial_physics_loss": initial_physics,
            "final_physics_loss": final_physics,
            "initial_boundary_loss": initial_boundary,
            "final_boundary_loss": final_boundary,
            "initial_centered_data_loss": initial_centered,
            "final_centered_data_loss": final_centered,
        },
    )
