"""Structured-grid PDE problems for physics-informed operator training.

This module is deliberately separate from :mod:`openpinn.problems`, whose
``PDEProblem`` contract describes coordinate PINNs.  A :class:`GridPDEProblem`
receives complete batched fields with shape ``(batch, channels, *spatial)``
and grid metadata suitable for :class:`~openpinn.GridOperator` evaluation.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from .grid_derivative import make_grid_derivative


def _mean_square(field: Tensor) -> Tensor:
    return torch.mean(field * field)


def _grid_spacing(grid: object) -> tuple[float, ...]:
    if not hasattr(grid, "spacing"):
        raise TypeError("grid must provide UniformGrid-compatible spacing metadata.")
    spacing = getattr(grid, "spacing")
    if isinstance(spacing, (int, float)):
        values = (float(spacing),)
    else:
        values = tuple(float(value) for value in spacing)
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("grid spacing entries must be finite and positive.")
    return values


@runtime_checkable
class GridPDEProblem(Protocol):
    """PDE residual and reduction hooks for structured-grid operators.

    This is a structural protocol: callers may supply any object implementing
    these three methods without inheriting an OpenPINN base class.  It is not
    the coordinate-PINN :class:`openpinn.PDEProblem` used by ``solve``.
    """

    def residual(
        self,
        solution: Tensor,
        inputs: Tensor,
        *,
        grid: object,
        derivative: str = "spectral",
    ) -> Tensor:
        """Return the pointwise PDE residual on the structured grid."""

        ...

    def boundary_residual(
        self,
        solution: Tensor,
        inputs: Tensor,
        *,
        grid: object,
    ) -> Tensor | None:
        """Return an optional boundary residual, or ``None`` when unused."""

        ...

    def physics_objective(
        self,
        residual: Tensor,
        *,
        grid: object,
        mode: str,
        derivative: str,
    ) -> Tensor:
        """Reduce the pointwise PDE residual to a scalar training term."""

        ...


def _spectral_inverse_laplacian_mean_square(
    residual: Tensor, *, spacing: float
) -> Tensor:
    """Return a 1D periodic residual in the inverse-Laplacian norm."""

    n_points = int(residual.shape[-1])
    frequencies = torch.fft.fftfreq(
        n_points,
        d=spacing,
        device=residual.device,
        dtype=residual.dtype,
    )
    wave_numbers = 2.0 * math.pi * frequencies
    inverse_k_squared = torch.zeros_like(wave_numbers)
    nonzero = wave_numbers != 0.0
    inverse_k_squared[nonzero] = wave_numbers[nonzero].square().reciprocal()
    spectrum = torch.fft.fft(residual, dim=-1)
    filtered = torch.fft.ifft(spectrum * inverse_k_squared, dim=-1).real
    return _mean_square(filtered)


class PeriodicPoisson1D:
    """Built-in periodic Poisson problem used by the PINO smoke trainer.

    ``residual`` computes ``-Δu - f``.  The public trainer uses one spatial
    dimension, while the residual retains the historical all-axis Laplacian
    behavior needed by :meth:`openpinn.PINO.pde_residual` compatibility calls.
    """

    def residual(
        self,
        solution: Tensor,
        inputs: Tensor,
        *,
        grid: object,
        derivative: str = "spectral",
    ) -> Tensor:
        if tuple(solution.shape) != tuple(inputs.shape):
            raise ValueError(
                "Periodic Poisson fields must share a shape, got "
                f"{tuple(solution.shape)} and {tuple(inputs.shape)}."
            )
        n_dims = solution.ndim - 2
        spacing = _grid_spacing(grid)
        if len(spacing) != n_dims:
            raise ValueError(
                f"grid spacing must have one entry per spatial axis ({n_dims})."
            )
        stencil = make_grid_derivative(derivative, spacing=spacing)
        residual = -inputs
        for axis in range(n_dims):
            residual = residual - stencil.partial(solution, axis=axis, order=2)
        return residual

    def boundary_residual(
        self,
        solution: Tensor,
        inputs: Tensor,
        *,
        grid: object,
    ) -> None:
        """Return no explicit BC term because the built-in grid is periodic."""

        del solution, inputs, grid
        return None

    def physics_objective(
        self,
        residual: Tensor,
        *,
        grid: object,
        mode: str,
        derivative: str,
    ) -> Tensor:
        """Use raw MSE, or the M9 1/k² norm for spectral physics-only."""

        if mode == "physics_only" and derivative == "spectral":
            if residual.ndim != 3:
                raise ValueError(
                    "PeriodicPoisson1D spectral physics_only expects one spatial axis."
                )
            return _spectral_inverse_laplacian_mean_square(
                residual, spacing=_grid_spacing(grid)[0]
            )
        return _mean_square(residual)


__all__ = ["GridPDEProblem", "PeriodicPoisson1D"]
