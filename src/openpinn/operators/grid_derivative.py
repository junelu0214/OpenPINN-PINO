"""Discrete derivatives of structured-grid fields.

:class:`GridDerivative` is the PDE-side derivative for a :class:`GridOperator`.
These stencils differentiate a tensor of shape ``(batch, channels, *spatial)``
along a spatial axis. They do not call coordinate autodiff.

Fourier neural operators typically live on periodic grids, so both the
spectral multiplier and the finite-difference stencil are periodic. PINO
minimizes a residual built from these stencils.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor


def _as_spacing(spacing: float | Sequence[float], n_spatial: int) -> tuple[float, ...]:
    if isinstance(spacing, Sequence) and not isinstance(spacing, (str, bytes)):
        values = tuple(float(item) for item in spacing)
        if len(values) != n_spatial:
            raise ValueError(
                f"spacing must have one value per spatial axis ({n_spatial}), "
                f"got {len(values)}."
            )
    else:
        values = (float(spacing),) * n_spatial
    if any(item <= 0.0 or not math.isfinite(item) for item in values):
        raise ValueError("spacing entries must be finite and positive.")
    return values


def _spatial_axis(field: Tensor, axis: int) -> int:
    if field.ndim < 3:
        raise ValueError(
            "Grid fields must have shape (batch, channels, *spatial) with at "
            f"least one spatial axis, got {tuple(field.shape)}."
        )
    n_spatial = field.ndim - 2
    if axis < 0 or axis >= n_spatial:
        raise ValueError(
            f"spatial axis must be in [0, {n_spatial}), got {axis}."
        )
    return 2 + axis


def _require_order(order: int) -> int:
    if isinstance(order, bool) or not isinstance(order, int) or order not in (1, 2):
        raise ValueError("order must be 1 or 2.")
    return order


class SpectralGridDerivative:
    """Periodic Fourier derivative along one structured-grid axis.

    ``partial`` uses a full FFT along the selected spatial axis. The
    multiplier is ``(2π i ξ)^order`` with ``ξ`` from :func:`torch.fft.fftfreq`.
    Odd-order Nyquist modes on even-length axes are zeroed so a real field
    stays real.
    """

    def __init__(self, spacing: float | Sequence[float] = 1.0) -> None:
        self.spacing: float | tuple[float, ...] = (
            tuple(float(item) for item in spacing)
            if isinstance(spacing, Sequence) and not isinstance(spacing, (str, bytes))
            else float(spacing)
        )

    def partial(self, field: Tensor, axis: int, order: int = 1) -> Tensor:
        """Differentiate ``field`` along one spatial axis."""

        selected_order = _require_order(order)
        dim = _spatial_axis(field, axis)
        n_points = int(field.shape[dim])
        dx = _as_spacing(self.spacing, field.ndim - 2)[axis]
        frequencies = torch.fft.fftfreq(n_points, d=dx, device=field.device, dtype=torch.float64)
        if selected_order % 2 == 1 and n_points % 2 == 0:
            frequencies[n_points // 2] = 0.0
        factor = (2j * math.pi * frequencies) ** selected_order
        view = [1] * field.ndim
        view[dim] = n_points
        transformed = torch.fft.fft(field.to(dtype=torch.float64), dim=dim)
        differentiated = torch.fft.ifft(transformed * factor.view(view), dim=dim)
        return differentiated.real.to(dtype=field.dtype)

    def derivative(self, field: Tensor, axis: int) -> Tensor:
        """Reference-style first derivative adapter for one spatial axis."""

        return self.partial(field, axis=axis, order=1)

    def second_derivative(self, field: Tensor, axis: int) -> Tensor:
        """Reference-style second derivative adapter for one spatial axis."""

        return self.partial(field, axis=axis, order=2)


class FiniteDifferenceGridDerivative:
    """Periodic central differences along one structured-grid axis.

    First derivatives use ``(u[i+1] - u[i-1]) / (2 Δx)``. Second derivatives
    use ``(u[i+1] - 2 u[i] + u[i-1]) / Δx²``. Endpoints wrap, matching the
    periodic grids used by Fourier layers.
    """

    def __init__(self, spacing: float | Sequence[float] = 1.0) -> None:
        self.spacing: float | tuple[float, ...] = (
            tuple(float(item) for item in spacing)
            if isinstance(spacing, Sequence) and not isinstance(spacing, (str, bytes))
            else float(spacing)
        )

    def partial(self, field: Tensor, axis: int, order: int = 1) -> Tensor:
        """Differentiate ``field`` along one spatial axis."""

        selected_order = _require_order(order)
        dim = _spatial_axis(field, axis)
        dx = _as_spacing(self.spacing, field.ndim - 2)[axis]
        left = torch.roll(field, shifts=1, dims=dim)
        right = torch.roll(field, shifts=-1, dims=dim)
        if selected_order == 1:
            return (right - left) / (2.0 * dx)
        return (right - 2.0 * field + left) / (dx * dx)

    def derivative(self, field: Tensor, axis: int) -> Tensor:
        """Reference-style first derivative adapter for one spatial axis."""

        return self.partial(field, axis=axis, order=1)

    def second_derivative(self, field: Tensor, axis: int) -> Tensor:
        """Reference-style second derivative adapter for one spatial axis."""

        return self.partial(field, axis=axis, order=2)


def make_grid_derivative(
    kind: str = "spectral",
    *,
    spacing: float | Sequence[float] = 1.0,
) -> Any:
    """Return a :class:`~openpinn.GridDerivative` of the requested kind.

    ``kind`` is ``"spectral"`` (Fourier multiplier) or ``"finite_difference"``
    (periodic central differences; aliases ``"fd"`` and ``"fourier"``).
    """

    selected = str(kind).lower().replace("-", "_")
    if selected in {"spectral", "fourier"}:
        return SpectralGridDerivative(spacing=spacing)
    if selected in {"finite_difference", "fd"}:
        return FiniteDifferenceGridDerivative(spacing=spacing)
    raise ValueError(
        "Grid derivative kind must be 'spectral' or 'finite_difference', "
        f"got {kind!r}."
    )
