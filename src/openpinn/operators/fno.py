"""Fourier neural operator on structured grids.

The network lifts a grid field, applies Fourier-layer blocks (a truncated
spectral convolution plus a local channel mix), and projects back to the
output channels. ``predict(inputs, *, grid=...)`` follows
:class:`openpinn.GridOperator`: input shape
``(batch, in_channels, *spatial)`` and output shape
``(batch, out_channels, *spatial)`` with the spatial size unchanged.
``forward(grid_field)`` remains a compatibility alias for one minor version.

PDE-side derivatives of that output use :class:`openpinn.GridDerivative`
(see :mod:`openpinn.operators.grid_derivative`), not coordinate autodiff.

This is an independent PyTorch implementation of the architecture in Li,
Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, and Anandkumar,
ICLR (2021), arXiv ``2010.08895``. It does not copy a reference repository.

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

from .grid_derivative import make_grid_derivative
from .selection import OperatorTrainResult, build_operator_config
from .uniform_grid import UniformGrid, validate_grid


_ACTIVATIONS = ("gelu", "tanh", "relu", "silu")
_TRAINING_MODES = ("data_only", "physics_only", "hybrid")
_METADATA_KEYS = frozenset({"operator", "kind", "derivative_mode", "title"})
_CONV_CLASSES = {1: nn.Conv1d, 2: nn.Conv2d}


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


FNO_PARAMETER_SCHEMA: dict[str, dict[str, Any]] = {
    "training_mode": _parameter(
        "Training mode",
        "choice",
        "data_only",
        description="Choose supervised data, PDE residual only, or their hybrid.",
        effect="Physics modes reuse the GridPDEProblem trainer with this FNO model.",
        choices=_TRAINING_MODES,
    ),
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
        description="Channel count of the input grid field.",
        effect="Must match grid_field.shape[1]. The built-in trainer uses 1.",
        minimum=1,
    ),
    "out_channels": _parameter(
        "Output channels",
        "integer",
        1,
        description="Channel count of the output grid field.",
        effect="The projection layer writes this many channels. The built-in trainer uses 1.",
        minimum=1,
    ),
    "width": _parameter(
        "Lifted width",
        "integer",
        8,
        description="Channel width inside the Fourier layers.",
        effect="A wider latent grid increases capacity and cost.",
        minimum=1,
    ),
    "n_layers": _parameter(
        "Fourier layers",
        "integer",
        2,
        description="Number of spectral-convolution blocks.",
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
        effect="Ignored by physics_only; data_only keeps the derivative smoke.",
        minimum=0.0,
    ),
    "physics_weight": _parameter(
        "Physics weight",
        "number",
        1.0,
        description="Multiplier on the GridPDEProblem objective.",
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
        4,
        description="Number of input functions in the fixed synthetic batch.",
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
        description="Number of sine and cosine terms in each synthetic input function.",
        effect="More harmonics make the derivative operator less smooth.",
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
        description="Discrete derivative used to inspect PDE-side residuals of the output grid.",
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


def _activation_module(name: str) -> nn.Module:
    selected = str(name).lower()
    activations: dict[str, type[nn.Module]] = {
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
    }
    if selected not in activations:
        choices = ", ".join(_ACTIVATIONS)
        raise ValueError(f"activation must be one of: {choices}.")
    return activations[selected]()


@dataclass(frozen=True)
class FNOConfig:
    """Widths and Fourier-mode count for one grid FNO."""

    n_dims: int = 1
    in_channels: int = 1
    out_channels: int = 1
    width: int = 8
    n_layers: int = 2
    n_modes: int = 4
    project_width: int = 16
    activation: str = "gelu"

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_dims", _require_int("n_dims", self.n_dims, minimum=1))
        if self.n_dims not in (1, 2):
            raise ValueError("n_dims must be 1 or 2.")
        object.__setattr__(
            self, "in_channels", _require_int("in_channels", self.in_channels, minimum=1)
        )
        object.__setattr__(
            self, "out_channels", _require_int("out_channels", self.out_channels, minimum=1)
        )
        object.__setattr__(self, "width", _require_int("width", self.width, minimum=1))
        object.__setattr__(self, "n_layers", _require_int("n_layers", self.n_layers, minimum=1))
        object.__setattr__(self, "n_modes", _require_int("n_modes", self.n_modes, minimum=1))
        object.__setattr__(
            self, "project_width", _require_int("project_width", self.project_width, minimum=1)
        )
        activation = str(self.activation).lower()
        if activation not in _ACTIVATIONS:
            choices = ", ".join(_ACTIVATIONS)
            raise ValueError(f"activation must be one of: {choices}.")
        object.__setattr__(self, "activation", activation)

    @property
    def modes(self) -> tuple[int, ...]:
        """Retained Fourier modes on each spatial axis."""

        return (self.n_modes,) * self.n_dims


class SpectralConv(nn.Module):
    """Truncated Fourier multiplier on a 1D or 2D grid."""

    def __init__(self, in_channels: int, out_channels: int, n_modes: Sequence[int]) -> None:
        super().__init__()
        modes = tuple(int(item) for item in n_modes)
        if not modes or any(item < 1 for item in modes):
            raise ValueError("n_modes entries must be positive integers.")
        if len(modes) not in (1, 2):
            raise ValueError("SpectralConv supports 1D or 2D grids.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_modes = modes
        self.n_dims = len(modes)
        scale = 1.0 / (self.in_channels * self.out_channels)
        n_blocks = 1 if self.n_dims == 1 else 2
        self.weight_real = nn.ParameterList()
        self.weight_imag = nn.ParameterList()
        for _ in range(n_blocks):
            self.weight_real.append(
                nn.Parameter(scale * torch.randn(self.in_channels, self.out_channels, *modes))
            )
            self.weight_imag.append(
                nn.Parameter(scale * torch.randn(self.in_channels, self.out_channels, *modes))
            )

    def _weights(self, index: int) -> Tensor:
        return torch.complex(self.weight_real[index], self.weight_imag[index])

    def _used_modes(self, spatial: tuple[int, ...]) -> tuple[int, ...]:
        used: list[int] = []
        for axis, (mode_count, size) in enumerate(zip(self.n_modes, spatial)):
            last_axis = axis == self.n_dims - 1
            limit = size // 2 + 1 if last_axis else size
            if last_axis:
                available = limit
            else:
                available = size // 2
            if available < 1:
                raise ValueError(
                    f"Spatial axis {axis} of size {size} is too small for a Fourier layer."
                )
            used.append(min(mode_count, available))
        if self.n_dims == 2 and spatial[0] < 2 * used[0]:
            raise ValueError(
                "2D Fourier layers need the first spatial axis to have at least "
                f"{2 * used[0]} points to keep positive and negative modes disjoint, "
                f"got {spatial[0]}."
            )
        return tuple(used)

    def forward(self, grid_field: Tensor) -> Tensor:
        if grid_field.ndim != 2 + self.n_dims:
            raise ValueError(
                f"SpectralConv expects shape (batch, channels, *spatial) with "
                f"{self.n_dims} spatial axes, got {tuple(grid_field.shape)}."
            )
        if grid_field.shape[1] != self.in_channels:
            raise ValueError(
                f"SpectralConv expected {self.in_channels} channels, "
                f"got {grid_field.shape[1]}."
            )
        spatial = tuple(int(size) for size in grid_field.shape[2:])
        used = self._used_modes(spatial)
        fft_dims = tuple(range(2, grid_field.ndim))
        transformed = torch.fft.rfftn(grid_field, dim=fft_dims)
        output = torch.zeros(
            grid_field.shape[0],
            self.out_channels,
            *transformed.shape[2:],
            dtype=transformed.dtype,
            device=grid_field.device,
        )
        if self.n_dims == 1:
            modes = used[0]
            weights = self._weights(0)[:, :, :modes]
            output[:, :, :modes] = torch.einsum(
                "bix,iox->box", transformed[:, :, :modes], weights
            )
        else:
            kx, ky = used
            low = self._weights(0)[:, :, :kx, :ky]
            high = self._weights(1)[:, :, :kx, :ky]
            output[:, :, :kx, :ky] = torch.einsum(
                "bixy,ioxy->boxy", transformed[:, :, :kx, :ky], low
            )
            output[:, :, -kx:, :ky] = torch.einsum(
                "bixy,ioxy->boxy", transformed[:, :, -kx:, :ky], high
            )
        return torch.fft.irfftn(output, s=spatial, dim=fft_dims)


class FNO(nn.Module):
    """Lifted Fourier-layer map from one structured grid to another."""

    def __init__(self, config: FNOConfig | None = None, /, **overrides: Any) -> None:
        super().__init__()
        selected = config if config is not None else FNOConfig()
        if not isinstance(selected, FNOConfig):
            raise TypeError("FNO config must be an FNOConfig.")
        if overrides:
            selected = replace(selected, **overrides)
        self.config = selected
        self.n_dims = selected.n_dims
        self.in_channels = selected.in_channels
        self.out_channels = selected.out_channels
        self.width = selected.width
        self.n_modes = selected.n_modes
        conv = _CONV_CLASSES[selected.n_dims]
        self.lift = conv(selected.in_channels, selected.width, kernel_size=1)
        self.spectral = nn.ModuleList(
            SpectralConv(selected.width, selected.width, selected.modes)
            for _ in range(selected.n_layers)
        )
        self.local = nn.ModuleList(
            conv(selected.width, selected.width, kernel_size=1)
            for _ in range(selected.n_layers)
        )
        self.project_hidden = conv(selected.width, selected.project_width, kernel_size=1)
        self.project_out = conv(selected.project_width, selected.out_channels, kernel_size=1)
        self.activation = _activation_module(selected.activation)

    def derivative_mode(self) -> str:
        """Return ``grid``. PDE residuals use :class:`~openpinn.GridDerivative`."""

        return "grid"

    def grid_derivative(self, kind: str = "spectral", *, spacing: float | Sequence[float] = 1.0):
        """Return a discrete derivative for this operator's output grid."""

        return make_grid_derivative(kind, spacing=spacing)

    def predict(self, inputs: Tensor, *, grid: object) -> Tensor:
        """Evaluate the operator on ``inputs`` described by ``grid``.

        Returns shape ``(batch, out_channels, *spatial)``. ``grid`` is
        keyword-only and must expose the :class:`UniformGrid` metadata.
        """

        validate_grid(grid, inputs.shape[2:], n_dims=self.n_dims)
        return self._evaluate(inputs)

    def forward(self, grid_field: Tensor) -> Tensor:
        """Compatibility alias using the historical grid-free call.

        Returns shape ``(batch, out_channels, *spatial)``.
        """

        return self._evaluate(grid_field)

    def _evaluate(self, grid_field: Tensor) -> Tensor:
        """Apply the unchanged FNO tensor computation."""

        if grid_field.ndim != 2 + self.n_dims:
            raise ValueError(
                f"FNO grid fields must have shape (batch, {self.in_channels}, "
                f"*spatial) with {self.n_dims} spatial axes, got {tuple(grid_field.shape)}."
            )
        if grid_field.shape[1] != self.in_channels:
            raise ValueError(
                f"FNO expected {self.in_channels} input channels, got {grid_field.shape[1]}."
            )
        hidden = self.lift(grid_field)
        for spectral, local in zip(self.spectral, self.local):
            hidden = self.activation(spectral(hidden) + local(hidden))
        hidden = self.activation(self.project_hidden(hidden))
        return self.project_out(hidden)


def _network_config(options: Mapping[str, Any]) -> FNOConfig:
    return FNOConfig(
        n_dims=int(options["n_dims"]),
        in_channels=int(options["in_channels"]),
        out_channels=int(options["out_channels"]),
        width=int(options["width"]),
        n_layers=int(options["n_layers"]),
        n_modes=int(options["n_modes"]),
        project_width=int(options["project_width"]),
        activation=str(options["activation"]),
    )


def build_fno(config: Mapping[str, Any] | FNOConfig | None = None, /, **overrides: Any) -> FNO:
    """Construct an FNO from a config mapping, an :class:`FNOConfig`, or keywords."""

    if isinstance(config, FNOConfig):
        return FNO(config, **overrides)
    if config is None:
        source: dict[str, Any] = {}
    elif isinstance(config, Mapping):
        source = {key: value for key, value in config.items() if key not in _METADATA_KEYS}
    else:
        raise TypeError("build_fno expects a mapping, FNOConfig, or keyword options.")
    source.update(overrides)
    resolved = build_operator_config("fno", **source)
    return FNO(_network_config(resolved))


def fourier_derivative_batch(
    batch_size: int,
    grid_size: int,
    *,
    generator: torch.Generator,
    n_harmonics: int = 2,
) -> tuple[Tensor, Tensor, Tensor]:
    """Draw one batch of the 1D periodic derivative operator.

    Each input is a random Fourier series on ``[0, 2π)``. The target is the
    exact spatial derivative. Returned fields have shape
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
    field = sine_amplitude.unsqueeze(-1) * torch.sin(angles)
    field = field + cosine_amplitude.unsqueeze(-1) * torch.cos(angles)
    field = field.sum(dim=1)
    derivative = (sine_amplitude * frequencies).unsqueeze(-1) * torch.cos(angles)
    derivative = derivative - (cosine_amplitude * frequencies).unsqueeze(-1) * torch.sin(angles)
    derivative = derivative.sum(dim=1)
    return grid_x, field.unsqueeze(1), derivative.unsqueeze(1)


def train_fno(config: Mapping[str, Any], *, device: str = "cpu") -> OperatorTrainResult:
    """Train FNO for a few Adam steps on a fixed 1D derivative batch.

    The batch is drawn once and reused. ``n_dims``, ``in_channels``, and
    ``out_channels`` must stay at 1 because this dataset is a scalar periodic
    derivative. Other shapes can still be constructed with :func:`build_fno`
    and trained by the caller. The trainer does not minimize a PDE residual;
    :meth:`FNO.grid_derivative` is the PDE-side derivative for later use.
    """

    if int(config["n_dims"]) != 1 or int(config["in_channels"]) != 1 or int(config["out_channels"]) != 1:
        raise ValueError(
            "train_operator supports the 1D scalar periodic derivative "
            "(n_dims=1, in_channels=1, out_channels=1). Use build_fno for other shapes."
        )
    grid_size = _require_int("grid_size", int(config["grid_size"]), minimum=4)
    n_modes = _require_int("n_modes", int(config["n_modes"]), minimum=1)
    if grid_size < 2 * n_modes:
        raise ValueError("grid_size must be at least 2 * n_modes for the 1D trainer.")

    torch.manual_seed(int(config["seed"]))
    model = build_fno(config).to(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config["seed"]))
    grid_x, field, target = fourier_derivative_batch(
        int(config["batch_size"]),
        grid_size,
        generator=generator,
        n_harmonics=int(config["n_harmonics"]),
    )
    field = field.to(device)
    target = target.to(device)
    grid = UniformGrid(
        grid_x,
        spacing=float(grid_x[1] - grid_x[0]),
        periodic=True,
    )

    def _loss() -> Tensor:
        return torch.mean((model.predict(field, grid=grid) - target) ** 2)

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
        operator="fno",
        steps=int(config["steps"]),
        initial_loss=initial_loss,
        final_loss=final_loss,
        history=tuple(history),
        model=model,
        config=dict(config),
    )
