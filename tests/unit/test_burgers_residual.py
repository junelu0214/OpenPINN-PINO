from __future__ import annotations

import math

import pytest
import torch

import openpinn as op


def test_burgers_residual_matches_a_manufactured_field() -> None:
    viscosity = 0.01 / math.pi
    problem = op.make_burgers_problem(viscosity=viscosity)
    coordinates = torch.tensor(
        [[-0.4, 0.2], [0.3, 0.7]], dtype=torch.float64, requires_grad=True
    )
    x = coordinates[:, :1]
    t = coordinates[:, 1:2]
    prediction = x.square() + 3.0 * t

    residual = problem.evaluate_residual(coordinates, prediction)["pde"]
    expected = 3.0 + prediction * (2.0 * x) - viscosity * 2.0

    torch.testing.assert_close(residual, expected)


def test_burgers_rejects_a_domain_with_incompatible_corner_data() -> None:
    domain = op.SpaceTimeDomain(op.Interval(-0.5, 0.5), op.Interval(0.0, 1.0))

    with pytest.raises(ValueError, match="compatible"):
        op.make_burgers_problem(domain)
