"""FNO construction, GridDerivative PDE-side use, and a tiny CPU training smoke."""

from __future__ import annotations

import math

import pytest
import torch

import openpinn as op


def _tiny(**overrides: object) -> op.FNO:
    options = dict(
        n_dims=1,
        in_channels=1,
        out_channels=1,
        width=4,
        n_layers=2,
        n_modes=2,
        project_width=8,
        activation="gelu",
    )
    options.update(overrides)
    return op.FNO(**options)


def test_fno_satisfies_grid_operator_and_forward_shapes() -> None:
    model = _tiny()
    assert isinstance(model, op.GridOperator)
    assert model.derivative_mode() == "grid"

    field = torch.randn(3, 1, 16)
    grid = op.UniformGrid.from_shape(16, spacing=(2.0 * math.pi) / 16.0, periodic=True)
    values = model.predict(field, grid=grid)
    torch.testing.assert_close(values, model(field))
    assert tuple(values.shape) == (3, 1, 16)
    assert torch.isfinite(values).all()

    with pytest.raises(TypeError, match="positional"):
        model.predict(field, grid)  # type: ignore[misc]
    wrong_grid = op.UniformGrid.from_shape(8, spacing=1.0, periodic=True)
    with pytest.raises(ValueError, match="does not match"):
        model.predict(field, grid=wrong_grid)

    with pytest.raises(ValueError, match="spatial"):
        model(torch.randn(3, 1, 16, 16))
    with pytest.raises(ValueError, match="channels"):
        model(torch.randn(3, 2, 16))


def test_fno_two_dimensional_forward_keeps_grid() -> None:
    model = _tiny(n_dims=2, n_modes=2, width=4, n_layers=1, project_width=4)
    field = torch.randn(2, 1, 8, 8)
    grid = op.UniformGrid.from_shape((8, 8), spacing=(0.25, 0.5), periodic=True)
    values = model.predict(field, grid=grid)
    assert tuple(values.shape) == (2, 1, 8, 8)
    assert torch.isfinite(values).all()


def test_grid_derivatives_match_sine_on_periodic_interval() -> None:
    n_points = 64
    spacing = (2.0 * math.pi) / n_points
    grid = torch.linspace(0.0, 2.0 * math.pi, n_points + 1, dtype=torch.float64)[:-1]
    field = torch.sin(3.0 * grid).view(1, 1, n_points)
    first = 3.0 * torch.cos(3.0 * grid).view(1, 1, n_points)
    second = -9.0 * torch.sin(3.0 * grid).view(1, 1, n_points)

    spectral = op.make_grid_derivative("spectral", spacing=spacing)
    finite = op.make_grid_derivative("fd", spacing=spacing)
    assert isinstance(spectral, op.GridDerivative)
    assert isinstance(finite, op.GridDerivative)
    assert torch.allclose(spectral.partial(field, 0, 1), first, atol=1e-8, rtol=1e-8)
    assert torch.allclose(spectral.partial(field, 0, 2), second, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(spectral.derivative(field, axis=0), first)
    torch.testing.assert_close(spectral.second_derivative(field, axis=0), second)
    field32 = field.to(dtype=torch.float32)
    first32 = first.to(dtype=torch.float32)
    second32 = second.to(dtype=torch.float32)
    assert torch.allclose(finite.partial(field32, 0, 1), first32, atol=2e-2, rtol=2e-2)
    assert torch.allclose(finite.partial(field32, 0, 2), second32, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(finite.derivative(field32, axis=0), finite.partial(field32, 0, 1))
    torch.testing.assert_close(
        finite.second_derivative(field32, axis=0), finite.partial(field32, 0, 2)
    )


def test_grid_derivatives_on_two_dimensional_product_mode() -> None:
    n_points = 32
    spacing = (2.0 * math.pi) / n_points
    x = torch.linspace(0.0, 2.0 * math.pi, n_points + 1)[:-1]
    y = torch.linspace(0.0, 2.0 * math.pi, n_points + 1)[:-1]
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    field = (torch.sin(2.0 * xx) * torch.cos(yy)).view(1, 1, n_points, n_points)
    d_dx = (2.0 * torch.cos(2.0 * xx) * torch.cos(yy)).view(1, 1, n_points, n_points)
    d2_dy2 = (-torch.sin(2.0 * xx) * torch.cos(yy)).view(1, 1, n_points, n_points)

    spectral = op.SpectralGridDerivative(spacing=spacing)
    assert torch.allclose(spectral.partial(field, 0, 1), d_dx, atol=1e-5, rtol=1e-5)
    assert torch.allclose(spectral.partial(field, 1, 2), d2_dy2, atol=1e-4, rtol=1e-4)
    finite = op.FiniteDifferenceGridDerivative(spacing=spacing)
    assert torch.allclose(finite.partial(field, 0, 1), d_dx, atol=5e-2, rtol=5e-2)


def test_fno_output_uses_grid_derivative_not_coordinate_autodiff() -> None:
    model = _tiny()
    field = torch.randn(2, 1, 16)
    prediction = model(field)
    spacing = (2.0 * math.pi) / 16.0
    residual = model.grid_derivative("spectral", spacing=spacing).partial(
        prediction, axis=0, order=2
    )
    assert tuple(residual.shape) == tuple(prediction.shape)
    assert torch.isfinite(residual).all()
    finite = model.grid_derivative("finite_difference", spacing=spacing).partial(
        prediction, axis=0, order=1
    )
    assert tuple(finite.shape) == tuple(prediction.shape)
    assert torch.isfinite(finite).all()


def test_fourier_derivative_batch_matches_spectral_grid_derivative() -> None:
    generator = torch.Generator().manual_seed(4)
    grid_x, field, target = op.fourier_derivative_batch(
        3, 48, generator=generator, n_harmonics=2
    )
    spacing = float(grid_x[1] - grid_x[0])
    estimated = op.make_grid_derivative("spectral", spacing=spacing).partial(field, 0, 1)
    assert torch.allclose(estimated, target, atol=1e-4, rtol=1e-4)


def test_operator_config_selects_fno_and_rejects_unknown_options() -> None:
    config = op.build_operator_config("fourier_neural_operator", width=6, n_modes=3)
    assert config["operator"] == "fno"
    assert config["kind"] == "grid"
    assert config["derivative_mode"] == "grid"
    assert config["width"] == 6
    assert config["n_modes"] == 3
    model = op.build_fno(config)
    assert model.width == 6
    assert model.n_modes == 3

    with pytest.raises(ValueError, match="Unknown option"):
        op.build_operator_config("fno", latent_dim=4)
    with pytest.raises(ValueError, match="n_dims=1"):
        op.train_operator("fno", n_dims=2, steps=1, width=4, n_modes=2, grid_size=8)


def test_fno_is_selectable_and_absent_from_pinn_catalog() -> None:
    names = [spec.name for spec in op.list_selectable_operators()]
    assert names == ["deeponet", "fno"]
    assert "fno" not in {spec.name for spec in op.list_methods()}
    with pytest.raises(ValueError, match="build_operator_config"):
        op.build_config("poisson", method="fno", steps=1)


def test_train_operator_fno_smoke_on_cpu() -> None:
    result = op.train_operator(
        "fno",
        steps=16,
        seed=0,
        device="cpu",
        width=8,
        n_layers=2,
        n_modes=4,
        project_width=16,
        grid_size=32,
        batch_size=4,
        n_harmonics=2,
        lr=1e-2,
    )
    assert result.operator == "fno"
    assert result.steps == 16
    assert len(result.history) == 16
    assert math.isfinite(result.initial_loss)
    assert math.isfinite(result.final_loss)
    assert all(math.isfinite(value) for value in result.history)
    assert result.final_loss < result.initial_loss
    assert isinstance(result.model, op.FNO)

    field = torch.randn(2, 1, 32)
    prediction = result.model(field)
    assert tuple(prediction.shape) == (2, 1, 32)
    residual = result.model.grid_derivative(
        result.config["derivative"],
        spacing=(2.0 * math.pi) / 32.0,
    ).partial(prediction, axis=0, order=2)
    assert tuple(residual.shape) == (2, 1, 32)
    assert torch.isfinite(residual).all()


def test_train_operator_fno_physics_only_keeps_m9_recovery() -> None:
    result = op.train_operator(
        "fno",
        training_mode="physics_only",
        steps=64,
        seed=0,
        device="cpu",
        width=8,
        n_layers=2,
        n_modes=4,
        project_width=16,
        grid_size=32,
        batch_size=4,
        n_harmonics=2,
        lr=1e-2,
        physics_weight=1.0,
    )
    assert result.operator == "fno"
    assert result.config["training_mode"] == "physics_only"
    assert isinstance(result.model, op.FNO)
    assert result.metrics["data_pretrain_steps"] == 0.0
    assert result.metrics["final_centered_data_loss"] < 0.2
    assert result.metrics["final_physics_loss"] < result.metrics["initial_physics_loss"]
