from __future__ import annotations

import pytest
import torch

import openpinn as op


def test_grad_and_selected_hessian_match_polynomial() -> None:
    coordinates = torch.tensor(
        [[-0.5, 0.25], [0.2, -0.3], [0.8, 0.6]],
        dtype=torch.float64,
        requires_grad=True,
    )
    values = coordinates[:, :1] ** 3 + coordinates[:, 1:2] ** 2

    gradient = op.grad(values, coordinates)
    expected_gradient = torch.cat(
        (3.0 * coordinates[:, :1] ** 2, 2.0 * coordinates[:, 1:2]), dim=1
    )
    torch.testing.assert_close(gradient, expected_gradient)

    xx = op.hessian(values, coordinates, i=0, j=0)
    tt = op.hessian(values, coordinates, i=1, j=1)
    torch.testing.assert_close(xx, 6.0 * coordinates[:, :1])
    torch.testing.assert_close(tt, torch.full_like(tt, 2.0))


def test_full_hessian_has_pointwise_shape_and_zero_cross_terms() -> None:
    coordinates = torch.tensor(
        [[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64, requires_grad=True
    )
    values = coordinates[:, :1] ** 2 + coordinates[:, 1:2] ** 2

    hessian = op.hessian(values, coordinates, i=None, j=None)

    assert hessian.shape == (2, 2, 2)
    expected = torch.eye(2, dtype=torch.float64).mul(2.0).expand(2, -1, -1)
    torch.testing.assert_close(hessian, expected)


def test_grad_rejects_coordinates_without_autograd_tracking() -> None:
    coordinates = torch.ones((2, 1))
    with pytest.raises(ValueError, match="requires_grad"):
        op.grad(coordinates**2, coordinates)


def test_grad_returns_exact_zero_for_constant_output() -> None:
    coordinates = torch.tensor([[0.2], [0.7]], requires_grad=True)
    constant = torch.ones((2, 1))

    derivative = op.grad(constant, coordinates)

    torch.testing.assert_close(derivative, torch.zeros_like(coordinates))
