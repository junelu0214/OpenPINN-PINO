from __future__ import annotations

import math

import pytest
import torch

import openpinn as op


def test_burgers_reference_satisfies_initial_and_boundary_data() -> None:
    reference = op.BurgersColeHopfReference(0.01 / math.pi)
    initial = torch.tensor(
        [[-1.0, 0.0], [-0.5, 0.0], [0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
        dtype=torch.float64,
    )
    expected = -torch.sin(math.pi * initial[:, :1])
    torch.testing.assert_close(reference(initial), expected, atol=1e-14, rtol=0.0)

    boundary = torch.tensor(
        [[-1.0, 0.01], [-1.0, 0.5], [1.0, 0.01], [1.0, 1.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        reference(boundary), torch.zeros((4, 1), dtype=torch.float64), atol=0.0, rtol=0.0
    )


def test_burgers_reference_converges_when_quadrature_is_doubled() -> None:
    coarse = op.BurgersColeHopfReference(
        0.01 / math.pi, quadrature_points=512
    )
    fine = op.BurgersColeHopfReference(
        0.01 / math.pi, quadrature_points=1024
    )
    x = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64)
    times = torch.tensor([0.01, 0.33, 0.67, 1.0], dtype=torch.float64)
    coordinates = torch.cartesian_prod(x, times)
    torch.testing.assert_close(
        coarse(coordinates), fine(coordinates), atol=2e-10, rtol=2e-10
    )


def test_burgers_reference_satisfies_the_pde_by_finite_differences() -> None:
    viscosity = 0.01 / math.pi
    reference = op.BurgersColeHopfReference(viscosity)
    coordinates = torch.tensor(
        [
            [-0.8, 0.10],
            [-0.3, 0.25],
            [0.2, 0.50],
            [0.7, 0.90],
        ],
        dtype=torch.float64,
    )
    spatial_step = 5e-4
    time_step = 1e-4
    x_shift = torch.tensor([spatial_step, 0.0], dtype=torch.float64)
    t_shift = torch.tensor([0.0, time_step], dtype=torch.float64)
    value = reference(coordinates)
    u_x = (
        reference(coordinates + x_shift) - reference(coordinates - x_shift)
    ) / (2.0 * spatial_step)
    u_xx = (
        reference(coordinates + x_shift)
        - 2.0 * value
        + reference(coordinates - x_shift)
    ) / (spatial_step**2)
    u_t = (
        reference(coordinates + t_shift) - reference(coordinates - t_shift)
    ) / (2.0 * time_step)
    residual = u_t + value * u_x - viscosity * u_xx
    assert float(residual.abs().max()) < 1e-4


def test_burgers_reference_rejects_unvalidated_early_positive_time() -> None:
    reference = op.BurgersColeHopfReference(0.01 / math.pi)
    with pytest.raises(ValueError, match="validated quadrature range"):
        reference(torch.tensor([[0.0, 0.001]], dtype=torch.float64))
