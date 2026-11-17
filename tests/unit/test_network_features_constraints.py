from __future__ import annotations

import pytest
import torch

import openpinn as op


def test_identity_and_fourier_features_have_explicit_dimensions() -> None:
    coordinates = torch.tensor(
        [[-0.5], [0.0], [0.75]], dtype=torch.float64, requires_grad=True
    )
    identity = op.IdentityFeature(input_dim=1)
    torch.testing.assert_close(identity(coordinates), coordinates)
    assert identity.output_dimension() == 1

    first = op.FourierFeatures(1, num_frequencies=4, scale=2.0, seed=41).to(
        dtype=torch.float64
    )
    second = op.FourierFeatures(1, num_frequencies=4, scale=2.0, seed=41).to(
        dtype=torch.float64
    )
    first_values = first(coordinates)
    second_values = second(coordinates)

    assert first.output_dimension() == 8
    assert first_values.shape == (3, 8)
    torch.testing.assert_close(first_values, second_values)
    feature_gradient = torch.autograd.grad(first_values.sum(), coordinates)[0]
    assert torch.isfinite(feature_gradient).all()


def test_mlp_and_pinn_model_preserve_batch_contract() -> None:
    torch.manual_seed(43)
    feature = op.FourierFeatures(1, num_frequencies=3, seed=5)
    network = op.MLP(
        input_dim=feature.output_dimension(),
        output_dim=1,
        hidden_layers=(8, 8),
        activation="tanh",
    )
    model = op.PINNModel(network, feature_transform=feature)
    coordinates = torch.linspace(-1.0, 1.0, 7).reshape(-1, 1).requires_grad_(True)

    prediction = model(coordinates)
    derivative = op.grad(prediction, coordinates)

    assert prediction.shape == (7, 1)
    assert derivative.shape == (7, 1)
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(derivative).all()


def test_modified_mlp_has_finite_output_and_requires_equal_hidden_widths() -> None:
    torch.manual_seed(45)
    model = op.ModifiedMLP(1, 1, hidden_layers=(8, 8, 8), activation="tanh")
    coordinates = torch.linspace(-1.0, 1.0, 7).reshape(-1, 1).requires_grad_(True)

    prediction = model(coordinates)
    derivative = op.grad(prediction, coordinates)

    assert prediction.shape == (7, 1)
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(derivative).all()
    with pytest.raises(ValueError, match="equal hidden widths"):
        op.ModifiedMLP(1, 1, hidden_layers=(8, 16))


def test_adaptive_tanh_is_trainable_and_differentiable() -> None:
    activation = op.AdaptiveTanh(initial_slope=1.0)
    inputs = torch.linspace(-1.0, 1.0, 7).reshape(-1, 1).requires_grad_(True)

    output = activation(inputs)
    output.square().mean().backward()

    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
    assert activation.slope.requires_grad
    assert activation.slope.grad is not None
    assert torch.isfinite(activation.slope.grad)
    with pytest.raises(ValueError, match="positive"):
        op.AdaptiveTanh(initial_slope=0.0)


def test_partial_transfer_freezes_everything_except_last_linear_layer() -> None:
    model = op.PINNModel(op.MLP(2, 1, hidden_layers=(8, 8)))

    op.freeze_all_but_last_linear_layers(model, last_n=1)

    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable == {"network.linear_layers.2.weight", "network.linear_layers.2.bias"}
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 9
    with pytest.raises(ValueError, match="positive integer"):
        op.freeze_all_but_last_linear_layers(model, last_n=0)


def test_two_subdomain_model_produces_finite_interface_losses() -> None:
    torch.manual_seed(46)
    problem = op.make_piecewise_poisson_problem(interface=0.0)
    model = op.DomainDecompositionPINN(
        1, 1, hidden_layers=(8, 8), interface=0.0
    )
    formulation = op.InterfaceStrongFormulation(problem)
    batch = op.UniformSampler(
        problem, interior=16, boundary=2, initial=0, seed=48
    ).sample()

    terms = formulation.loss_terms(model, batch)
    prediction = model(torch.tensor([[-0.5], [0.0], [0.5]]))

    assert prediction.shape == (3, 1)
    assert {"pde", "boundary", "interface_solution", "interface_flux"} == set(terms)
    assert all(value.ndim == 0 and torch.isfinite(value) for value in terms.values())
    assert sum(parameter.numel() for parameter in model.left_network.parameters()) == sum(
        parameter.numel() for parameter in model.right_network.parameters()
    )


def test_analytic_hard_constraint_is_exact_at_interval_endpoints() -> None:
    torch.manual_seed(47)
    constraint = op.HardConstraint(
        distance=lambda x: 1.0 - x[:, :1].square(),
        boundary_value=lambda x: 2.0 + x[:, :1],
    )
    model = op.PINNModel(
        op.MLP(1, 1, hidden_layers=(8,)), output_transform=constraint
    )
    boundary = torch.tensor([[-1.0], [1.0]])
    interior = torch.tensor([[0.0], [0.25]])

    boundary_prediction = model(boundary)
    interior_prediction = model(interior)

    torch.testing.assert_close(boundary_prediction, 2.0 + boundary, atol=0.0, rtol=0.0)
    assert torch.isfinite(interior_prediction).all()
