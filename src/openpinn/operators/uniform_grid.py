"""Lightweight structured-grid metadata for public operator evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import torch
from torch import Tensor


def _tuple_value(value: object, n_dims: int, *, name: str) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(value)
        if len(values) != n_dims:
            raise ValueError(f"{name} must have one entry per grid axis ({n_dims}).")
        return values
    return (value,) * n_dims


@dataclass(frozen=True, init=False, eq=False)
class UniformGrid:
    """Uniform Cartesian grid metadata used by ``GridOperator.predict``.

    ``nodes`` is one one-dimensional coordinate tensor per spatial axis.
    ``spacing`` and ``periodic`` accept either one scalar shared by every axis
    or one value per axis. The derived ``shape``, ``n_points``, and ``n_dims``
    attributes mirror the reference work-repository contract.
    """

    nodes: tuple[Tensor, ...]
    spacing: tuple[float, ...]
    periodic: tuple[bool, ...]

    def __init__(
        self,
        nodes: Tensor | Sequence[Tensor],
        *,
        spacing: float | Sequence[float],
        periodic: bool | Sequence[bool] = True,
    ) -> None:
        axes = (nodes,) if isinstance(nodes, Tensor) else tuple(nodes)
        if not axes:
            raise ValueError("UniformGrid needs at least one coordinate axis.")
        for axis, values in enumerate(axes):
            if not isinstance(values, Tensor) or values.ndim != 1 or values.numel() < 1:
                raise ValueError(
                    f"UniformGrid nodes[{axis}] must be a nonempty one-dimensional tensor."
                )
            if not torch.isfinite(values).all():
                raise ValueError(f"UniformGrid nodes[{axis}] must be finite.")
        spacings = tuple(float(item) for item in _tuple_value(spacing, len(axes), name="spacing"))
        if any(item <= 0.0 or not math.isfinite(item) for item in spacings):
            raise ValueError("UniformGrid spacing entries must be finite and positive.")
        periodic_values = _tuple_value(periodic, len(axes), name="periodic")
        if any(not isinstance(item, bool) for item in periodic_values):
            raise TypeError("UniformGrid periodic entries must be booleans.")
        object.__setattr__(self, "nodes", axes)
        object.__setattr__(self, "spacing", spacings)
        object.__setattr__(self, "periodic", tuple(periodic_values))

    @classmethod
    def from_shape(
        cls,
        shape: int | Sequence[int],
        *,
        spacing: float | Sequence[float] = 1.0,
        periodic: bool | Sequence[bool] = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "UniformGrid":
        """Construct grid axes from a spatial shape and uniform spacing."""

        sizes = (shape,) if isinstance(shape, Integral) else tuple(shape)
        if not sizes or any(
            isinstance(size, bool) or not isinstance(size, Integral) or int(size) < 1
            for size in sizes
        ):
            raise ValueError("UniformGrid shape entries must be positive integers.")
        normalized_sizes = tuple(int(size) for size in sizes)
        spacings = tuple(
            float(item) for item in _tuple_value(spacing, len(normalized_sizes), name="spacing")
        )
        nodes = tuple(
            torch.arange(size, device=device, dtype=dtype) * axis_spacing
            for size, axis_spacing in zip(normalized_sizes, spacings)
        )
        return cls(nodes, spacing=spacings, periodic=periodic)

    @property
    def shape(self) -> tuple[int, ...]:
        """Number of samples on each spatial axis."""

        return tuple(int(axis.numel()) for axis in self.nodes)

    @property
    def n_points(self) -> int:
        """Total number of Cartesian grid points."""

        return math.prod(self.shape)

    @property
    def n_dims(self) -> int:
        """Number of spatial axes."""

        return len(self.nodes)


def validate_grid(grid: object, spatial_shape: Sequence[int], *, n_dims: int) -> None:
    """Validate the public grid metadata needed by FNO/PINO prediction."""

    required = ("shape", "spacing", "periodic", "n_points", "n_dims")
    missing = [name for name in required if not hasattr(grid, name)]
    if missing:
        raise TypeError(
            "grid must provide UniformGrid-compatible metadata: " + ", ".join(required) + "."
        )
    shape = tuple(int(size) for size in getattr(grid, "shape"))
    expected = tuple(int(size) for size in spatial_shape)
    if int(getattr(grid, "n_dims")) != n_dims or len(shape) != n_dims:
        raise ValueError(f"grid must describe {n_dims} spatial dimensions, got shape {shape}.")
    if shape != expected:
        raise ValueError(f"grid shape {shape} does not match input spatial shape {expected}.")
    if int(getattr(grid, "n_points")) != math.prod(expected):
        raise ValueError("grid n_points does not match its spatial shape.")
    spacings = tuple(
        float(item)
        for item in _tuple_value(getattr(grid, "spacing"), n_dims, name="grid spacing")
    )
    if any(item <= 0.0 or not math.isfinite(item) for item in spacings):
        raise ValueError("grid spacing entries must be finite and positive.")
    periodic = _tuple_value(getattr(grid, "periodic"), n_dims, name="grid periodic")
    if any(not isinstance(item, bool) for item in periodic):
        raise TypeError("grid periodic entries must be booleans.")
