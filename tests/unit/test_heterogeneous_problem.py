from __future__ import annotations

import torch

import openpinn as op


def test_piecewise_poisson_exact_field_has_zero_residual_away_from_interface() -> None:
    problem = op.make_piecewise_poisson_problem(
        interface=0.0, left_diffusivity=1.0, right_diffusivity=4.0
    )
    coordinates = torch.tensor(
        [[-0.8], [-0.2], [0.2], [0.8]],
        dtype=torch.float64,
        requires_grad=True,
    )
    exact = problem.exact(coordinates)
    residual = problem.evaluate_residual(coordinates, exact)["pde"]

    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-12, rtol=0.0)


def test_piecewise_poisson_exact_field_is_continuous_with_continuous_flux() -> None:
    problem = op.make_piecewise_poisson_problem(
        interface=0.0, left_diffusivity=1.0, right_diffusivity=4.0
    )
    epsilon = 1e-7
    left = torch.tensor([[-epsilon]], dtype=torch.float64, requires_grad=True)
    right = torch.tensor([[epsilon]], dtype=torch.float64, requires_grad=True)
    left_value = problem.exact(left)
    right_value = problem.exact(right)
    left_flux = op.grad(left_value, left) * 1.0
    right_flux = op.grad(right_value, right) * 4.0

    torch.testing.assert_close(left_value, right_value, atol=2e-7, rtol=0.0)
    torch.testing.assert_close(left_flux, right_flux, atol=1e-12, rtol=0.0)


def test_piecewise_poisson_residual_supports_model_backward() -> None:
    torch.manual_seed(53)
    problem = op.make_piecewise_poisson_problem()
    model = op.PINNModel(op.MLP(1, 1, hidden_layers=(8, 8))).to(torch.float64)
    sampler = op.UniformSampler(
        problem, interior=12, boundary=4, seed=59, dtype=torch.float64
    )
    trainer = op.Trainer(model, problem, sampler, optimizer="adam", lr=1e-3)

    history = trainer.fit(steps=1)

    assert len(history) == 1
    assert torch.isfinite(torch.tensor(history.last["total"]))
