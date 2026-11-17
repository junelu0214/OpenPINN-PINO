"""DeepONet construction, registry selection, and a tiny CPU training smoke."""

from __future__ import annotations

import inspect
import math

import pytest
import torch

import openpinn as op
from openpinn.operators.coordinate_pde import periodic_poisson_coordinate_residual


def _tiny() -> op.DeepONet:
    return op.DeepONet(
        n_sensors=6,
        n_dims=1,
        n_fields=1,
        latent_dim=4,
        branch_hidden=(8,),
        trunk_hidden=(8,),
        activation="tanh",
        bias=True,
    )


def test_deeponet_satisfies_coordinate_operator_and_forward_shapes() -> None:
    model = _tiny()
    assert isinstance(model, op.CoordinateOperator)
    assert not hasattr(op, "GridOperatorFromCoordinateModel")
    assert model.derivative_mode() == "autodiff"

    coordinates = torch.linspace(0.0, 1.0, 5).unsqueeze(1)
    shared = model.predict(coordinates)
    assert tuple(shared.shape) == (5, 1)
    condition = torch.linspace(-1.0, 1.0, model.n_sensors)
    conditioned = model.predict(coordinates, condition=condition)
    branched = model(coordinates, branch=condition)
    torch.testing.assert_close(conditioned, branched)
    assert tuple(branched.shape) == (5, 1)
    per_query = model(coordinates, branch=torch.randn(5, model.n_sensors))
    assert tuple(per_query.shape) == (5, 1)

    batch = model.evaluate(torch.randn(3, model.n_sensors), coordinates)
    assert tuple(batch.shape) == (3, 5, 1)
    per_example = model.evaluate(
        torch.randn(3, model.n_sensors),
        torch.rand(3, 4, 1),
    )
    assert tuple(per_example.shape) == (3, 4, 1)


def test_deeponet_coordinate_autodiff_is_finite() -> None:
    model = _tiny()
    coordinates = torch.linspace(0.0, 1.0, 5).unsqueeze(1).requires_grad_(True)
    values = model.predict(coordinates, condition=torch.ones(model.n_sensors))
    derivatives = op.grad(values, coordinates)
    assert tuple(derivatives.shape) == tuple(coordinates.shape)
    assert torch.isfinite(derivatives).all()
    assert float(derivatives.detach().abs().sum()) > 0.0


def test_coordinate_poisson_residual_vanishes_on_manufactured_solution() -> None:
    coordinates = torch.linspace(0.0, 2.0 * math.pi, 17).unsqueeze(1)
    coordinates.requires_grad_(True)
    solution = torch.sin(2.0 * coordinates)
    forcing = 4.0 * torch.sin(2.0 * coordinates)
    residual = periodic_poisson_coordinate_residual(
        solution, coordinates, forcing, create_graph=False
    )
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-6, rtol=1e-6)


def test_vector_output_and_rejected_branch_shapes() -> None:
    model = op.DeepONet(
        n_sensors=4,
        n_dims=2,
        n_fields=2,
        latent_dim=3,
        branch_hidden=(),
        trunk_hidden=(),
        bias=False,
    )
    coordinates = torch.zeros(4, 2)
    values = model(coordinates, branch=torch.ones(4))
    assert tuple(values.shape) == (4, 2)
    assert model.output_bias is None

    with pytest.raises(ValueError, match="branch"):
        model(coordinates, branch=torch.ones(3))
    with pytest.raises(ValueError, match="coordinates"):
        model(torch.zeros(4, 1))


def test_antiderivative_targets_match_cumulative_trapezoid() -> None:
    positions = torch.linspace(0.0, 1.0, 257)
    generator = torch.Generator().manual_seed(2)
    sensors, coordinates, target = op.antiderivative_batch(
        2,
        257,
        257,
        generator=generator,
        n_modes=3,
        query_x=positions,
    )
    cumulative = torch.cumulative_trapezoid(sensors, positions, dim=-1)
    cumulative = torch.cat([torch.zeros(2, 1), cumulative], dim=-1)
    assert tuple(coordinates.shape) == (257, 1)
    assert torch.allclose(cumulative, target[..., 0], atol=1e-4, rtol=1e-4)
    assert torch.allclose(target[:, 0, 0], torch.zeros(2), atol=1e-6)


def test_operator_config_rejects_unknown_options_and_unimplemented_hooks() -> None:
    config = op.build_operator_config("deep_operator_network", latent_dim=5, bias=False)
    assert config["operator"] == "deeponet"
    assert config["kind"] == "coordinate"
    assert config["derivative_mode"] == "autodiff"
    assert config["latent_dim"] == 5
    assert config["bias"] is False
    assert config["n_modes"] == 3
    model = op.build_deeponet(config)
    assert model.latent_dim == 5
    assert model.output_bias is None

    with pytest.raises(ValueError, match="Unknown option"):
        op.build_operator_config("deeponet", width=4)
    with pytest.warns(DeprecationWarning, match="coordinate autodiff"):
        legacy = op.build_operator_config("deeponet", derivative="spectral")
    assert "derivative" not in legacy
    with pytest.raises(ValueError, match="n_dims=1"):
        op.train_operator("deeponet", n_dims=2, steps=1, n_sensors=4, latent_dim=2)
    with pytest.raises(ValueError, match="n_sensors must be at least"):
        op.train_operator(
            "deeponet",
            training_mode="physics_only",
            steps=1,
            n_sensors=3,
            n_modes=2,
        )


def test_train_operator_smoke_uses_default_n_modes() -> None:
    result = op.train_operator(
        "deeponet",
        steps=1,
        seed=0,
        device="cpu",
        n_sensors=8,
        latent_dim=4,
        branch_hidden=(8,),
        trunk_hidden=(8,),
        batch_size=2,
        n_query=4,
    )
    assert result.config["n_modes"] == 3
    assert math.isfinite(result.final_loss)


def test_train_operator_smoke_on_cpu() -> None:
    result = op.train_operator(
        "deeponet",
        steps=12,
        seed=0,
        device="cpu",
        n_sensors=8,
        latent_dim=8,
        branch_hidden=(16,),
        trunk_hidden=(16,),
        batch_size=4,
        n_query=8,
        n_modes=2,
        lr=1e-2,
    )
    assert result.operator == "deeponet"
    assert result.steps == 12
    assert len(result.history) == 12
    assert math.isfinite(result.initial_loss)
    assert math.isfinite(result.final_loss)
    assert all(math.isfinite(value) for value in result.history)
    assert result.final_loss < result.initial_loss

    coordinates = torch.linspace(0.0, 1.0, 4).unsqueeze(1)
    prediction = result.model(coordinates, branch=torch.zeros(result.model.n_sensors))
    assert tuple(prediction.shape) == (4, 1)
    assert torch.isfinite(prediction).all()


@pytest.mark.parametrize("training_mode", ["physics_only", "hybrid"])
def test_deeponet_coordinate_autodiff_physics_modes(training_mode: str) -> None:
    result = op.train_operator(
        "deeponet",
        training_mode=training_mode,
        steps=3,
        seed=0,
        device="cpu",
        n_sensors=8,
        latent_dim=4,
        branch_hidden=(8,),
        trunk_hidden=(8,),
        batch_size=2,
        n_query=8,
        n_modes=2,
        lr=1e-2,
    )
    assert result.operator == "deeponet"
    assert result.config["training_mode"] == training_mode
    assert isinstance(result.model, op.DeepONet)
    assert isinstance(result.model, op.CoordinateOperator)
    assert "grid" not in inspect.signature(result.model.predict).parameters
    assert math.isfinite(result.initial_loss)
    assert math.isfinite(result.final_loss)
    assert math.isfinite(result.metrics["initial_physics_loss"])
    assert math.isfinite(result.metrics["final_physics_loss"])
    assert result.metrics["data_pretrain_steps"] == 0.0

    coordinates = torch.linspace(0.0, 1.0, 4).unsqueeze(1).requires_grad_(True)
    prediction = result.model.predict(
        coordinates, condition=torch.zeros(result.model.n_sensors)
    )
    assert tuple(op.hessian(prediction, coordinates).shape) == (4, 1)
    assert torch.isfinite(prediction).all()
