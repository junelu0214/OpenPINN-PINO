from __future__ import annotations

import pytest
import torch

import openpinn as op


def test_exact_two_field_solution_satisfies_all_named_residuals() -> None:
    problem = op.make_coupled_manufactured_problem(coupling_strength=1.7)
    coordinates = torch.linspace(-1.0, 1.0, 19, dtype=torch.float64).reshape(-1, 1)
    coordinates.requires_grad_(True)
    exact = problem.exact(coordinates)

    residuals = problem.evaluate_residual(coordinates, exact)

    assert problem.field_names == ("u", "v")
    assert problem.output_dim == 2
    assert exact.shape == (19, 2)
    assert tuple(residuals) == ("equation_u", "equation_v", "coupling")
    for residual in residuals.values():
        assert residual.shape == (19, 1)
        torch.testing.assert_close(
            residual, torch.zeros_like(residual), atol=1e-11, rtol=1e-11
        )


def test_coupled_residuals_use_the_intended_output_components() -> None:
    kappa = 1.25
    epsilon = 0.2
    problem = op.make_coupled_manufactured_problem(coupling_strength=kappa)
    coordinates = torch.linspace(-0.8, 0.8, 11, dtype=torch.float64).reshape(-1, 1)
    coordinates.requires_grad_(True)
    x = coordinates[:, :1]
    perturbed = torch.cat((x.square(), x.pow(3) + epsilon * x), dim=1)

    residuals = problem.evaluate_residual(coordinates, perturbed)

    torch.testing.assert_close(
        residuals["equation_u"], -kappa * epsilon * x, atol=1e-11, rtol=1e-11
    )
    torch.testing.assert_close(
        residuals["equation_v"], kappa * epsilon * x, atol=1e-11, rtol=1e-11
    )
    torch.testing.assert_close(
        residuals["coupling"], torch.full_like(x, epsilon), atol=1e-11, rtol=1e-11
    )


def test_vector_boundary_data_is_independent_of_internal_reference() -> None:
    problem = op.make_coupled_manufactured_problem()
    boundary = problem.domain.sample_boundary(4, dtype=torch.float64).requires_grad_(True)

    error = problem.conditions[0].error(lambda points: problem.exact(points), boundary)

    assert error.shape == (4, 2)
    torch.testing.assert_close(error, torch.zeros_like(error), atol=0.0, rtol=0.0)


def test_two_output_model_backpropagates_all_coupled_loss_terms() -> None:
    problem = op.make_coupled_manufactured_problem()
    model = op.PINNModel(op.MLP(1, 2, hidden_layers=(12, 12))).to(torch.float64)
    sampler = op.UniformSampler(problem, 16, 4, dtype=torch.float64, seed=5)
    formulation = op.StrongFormulation(problem)
    batch = sampler.sample(requires_grad=True)

    terms = formulation.loss_terms(model, batch)
    sum(terms.values()).backward()

    assert {"equation_u", "equation_v", "coupling", "boundary"} == set(terms)
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() for gradient in gradients if gradient is not None)


@pytest.mark.parametrize("value", (0.0, -1.0, float("nan"), float("inf")))
def test_invalid_coupling_strength_fails_early(value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        op.make_coupled_manufactured_problem(coupling_strength=value)
