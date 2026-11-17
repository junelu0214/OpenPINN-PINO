"""Coordinate-point PDE helpers for the built-in DeepONet physics smoke.

These helpers are deliberately separate from :mod:`openpinn.operators.pde_problem`.
DeepONet differentiates predictions with respect to query coordinates through
autodiff; it does not resample onto a structured grid or use ``GridPDEProblem``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..autodiff import hessian


def periodic_poisson_coordinate_batch(
    batch_size: int,
    n_sensors: int,
    n_query: int,
    *,
    generator: torch.Generator,
    n_harmonics: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return forcing sensors, query points, forcing, and exact solution.

    The manufactured zero-mean solution is a random Fourier series on
    ``[0, 2π)`` and the forcing is ``-u_xx``. Shapes are respectively
    ``(batch, n_sensors)``, ``(n_query, 1)``, and two tensors of shape
    ``(batch, n_query, 1)``.
    """

    frequencies = torch.arange(1, n_harmonics + 1, dtype=torch.float32)
    sine_amplitude = torch.randn(batch_size, n_harmonics, generator=generator)
    cosine_amplitude = torch.randn(batch_size, n_harmonics, generator=generator)
    sensor_x = torch.arange(n_sensors, dtype=torch.float32) * (2.0 * math.pi / n_sensors)
    query_x = torch.rand(n_query, generator=generator) * (2.0 * math.pi)

    def _fields(points: Tensor) -> tuple[Tensor, Tensor]:
        angles = frequencies.view(1, -1, 1) * points.view(1, 1, -1)
        sine = sine_amplitude.unsqueeze(-1) * torch.sin(angles)
        cosine = cosine_amplitude.unsqueeze(-1) * torch.cos(angles)
        solution = (sine + cosine).sum(dim=1)
        scale = frequencies.square().view(1, -1, 1)
        forcing = (scale * (sine + cosine)).sum(dim=1)
        return forcing, solution

    sensor_forcing, _ = _fields(sensor_x)
    query_forcing, target = _fields(query_x)
    return (
        sensor_forcing,
        query_x.unsqueeze(-1),
        query_forcing.unsqueeze(-1),
        target.unsqueeze(-1),
    )


def periodic_poisson_coordinate_residual(
    solution: Tensor,
    coordinates: Tensor,
    forcing: Tensor,
    *,
    create_graph: bool = True,
) -> Tensor:
    """Return the pointwise ``-u_xx - f`` residual at query coordinates."""

    if solution.ndim != 2 or solution.shape[1] != 1:
        raise ValueError("solution must have shape (n_points, 1).")
    if coordinates.ndim != 2 or coordinates.shape[1] != 1:
        raise ValueError("coordinates must have shape (n_points, 1).")
    if tuple(forcing.shape) != tuple(solution.shape):
        raise ValueError("forcing must have the same shape as solution.")
    return -hessian(solution, coordinates, i=0, j=0, create_graph=create_graph) - forcing


__all__ = [
    "periodic_poisson_coordinate_batch",
    "periodic_poisson_coordinate_residual",
]
