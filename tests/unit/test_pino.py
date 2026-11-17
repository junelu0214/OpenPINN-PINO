"""PINO construction, GridDerivative residual, and hybrid / physics-only smokes."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

import openpinn as op


def _tiny(**overrides: object) -> op.PINO:
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
    return op.PINO(**options)


def test_pino_satisfies_grid_operator_and_reuses_fno_backbone() -> None:
    model = _tiny()
    assert isinstance(model, op.GridOperator)
    assert isinstance(model.backbone, op.FNO)
    assert model.derivative_mode() == "grid"

    field = torch.randn(3, 1, 16)
    grid = op.UniformGrid.from_shape(16, spacing=(2.0 * math.pi) / 16.0, periodic=True)
    values = model.predict(field, grid=grid)
    torch.testing.assert_close(values, model(field))
    assert tuple(values.shape) == (3, 1, 16)
    assert torch.isfinite(values).all()
    assert tuple(model.backbone(field).shape) == (3, 1, 16)

    with pytest.raises(ValueError, match="spatial"):
        model(torch.randn(3, 1, 16, 16))
    with pytest.raises(ValueError, match="channels"):
        model(torch.randn(3, 2, 16))


def test_pino_two_dimensional_forward_keeps_grid() -> None:
    model = _tiny(n_dims=2, n_modes=2, width=4, n_layers=1, project_width=4)
    field = torch.randn(2, 1, 8, 8)
    values = model(field)
    assert tuple(values.shape) == (2, 1, 8, 8)
    residual = model.pde_residual(
        values, field, spacing=(2.0 * math.pi) / 8.0, derivative="spectral"
    )
    assert tuple(residual.shape) == (2, 1, 8, 8)
    assert torch.isfinite(residual).all()


def test_poisson_batch_residual_vanishes_on_the_manufactured_solution() -> None:
    generator = torch.Generator().manual_seed(4)
    grid_x, forcing, solution = op.poisson_operator_batch(
        3, 48, generator=generator, n_harmonics=2
    )
    spacing = float(grid_x[1] - grid_x[0])
    model = _tiny()
    residual = model.pde_residual(solution, forcing, spacing=spacing, derivative="spectral")
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-3, rtol=1e-3)
    finite = model.pde_residual(
        solution, forcing, spacing=spacing, derivative="finite_difference"
    )
    assert torch.isfinite(finite).all()
    assert float(finite.abs().mean()) < 0.2


def test_operator_config_pino_name_is_a_deprecated_fno_mode_shim() -> None:
    with pytest.warns(DeprecationWarning, match="deprecated"):
        config = op.build_operator_config(
            "physics_informed_neural_operator",
            width=6,
            n_modes=3,
            mode="physics_only",
            data_pretrain_steps=2,
        )
    assert config["operator"] == "fno"
    assert config["kind"] == "grid"
    assert config["derivative_mode"] == "grid"
    assert config["width"] == 6
    assert config["n_modes"] == 3
    assert config["training_mode"] == "physics_only"
    assert config["data_pretrain_steps"] == 2
    model = op.build_pino(config)
    assert model.width == 6
    assert model.n_modes == 3
    assert isinstance(model, op.PINO)

    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="Unknown option"):
        op.build_operator_config("pino", latent_dim=4)
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="n_dims=1"):
        op.train_operator("pino", n_dims=2, steps=1, width=4, n_modes=2, grid_size=8)


def test_pino_is_not_selectable_and_remains_absent_from_pinn_catalog() -> None:
    names = [spec.name for spec in op.list_selectable_operators()]
    assert names == ["deeponet", "fno"]
    assert "pino" not in {spec.name for spec in op.list_methods()}
    with pytest.raises(ValueError, match="Unknown OpenPINN method"):
        op.build_config("poisson", method="pino", steps=1)


def _smoke_kwargs() -> dict[str, object]:
    return dict(
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


def test_train_operator_pino_shim_routes_to_fno_hybrid() -> None:
    with pytest.warns(DeprecationWarning, match="deprecated"):
        result = op.train_operator(
            "pino",
            steps=16,
            mode="hybrid",
            data_pretrain_steps=6,
            data_weight=1.0,
            physics_weight=0.1,
            **_smoke_kwargs(),
        )
    assert result.operator == "fno"
    assert result.steps == 16
    assert len(result.history) == 16
    assert result.config["training_mode"] == "hybrid"
    assert math.isfinite(result.initial_loss)
    assert math.isfinite(result.final_loss)
    assert all(math.isfinite(value) for value in result.history)
    assert result.final_loss < result.initial_loss
    assert math.isfinite(result.metrics["final_data_loss"])
    assert math.isfinite(result.metrics["final_physics_loss"])
    assert result.metrics["final_data_loss"] < result.metrics["initial_data_loss"]
    assert result.metrics["data_pretrain_steps"] == 6.0
    assert isinstance(result.model, op.FNO)

    forcing = torch.randn(2, 1, 32)
    grid = op.UniformGrid.from_shape(32, spacing=(2.0 * math.pi) / 32.0)
    prediction = result.model.predict(forcing, grid=grid)
    assert tuple(prediction.shape) == (2, 1, 32)
    residual = op.PeriodicPoisson1D().residual(
        prediction, forcing, grid=grid, derivative=result.config["derivative"]
    )
    assert tuple(residual.shape) == (2, 1, 32)
    assert torch.isfinite(residual).all()


def test_train_operator_fno_physics_only_recovers_centered_poisson_solution() -> None:
    result = op.train_operator(
        "fno",
        steps=64,
        training_mode="physics_only",
        data_pretrain_steps=4,
        data_weight=1.0,
        physics_weight=1.0,
        **_smoke_kwargs(),
    )
    assert result.operator == "fno"
    assert result.config["training_mode"] == "physics_only"
    assert result.metrics["data_pretrain_steps"] == 0.0
    assert math.isfinite(result.initial_loss)
    assert math.isfinite(result.final_loss)
    assert math.isfinite(result.metrics["initial_physics_loss"])
    assert math.isfinite(result.metrics["final_physics_loss"])
    assert math.isfinite(result.metrics["final_data_loss"])
    assert math.isfinite(result.metrics["final_centered_data_loss"])
    assert all(math.isfinite(value) for value in result.history)
    assert result.metrics["final_centered_data_loss"] < 0.2
    assert result.metrics["final_physics_loss"] < result.metrics["initial_physics_loss"]


class _IdentityGridProblem:
    """Non-Poisson test problem with an exercised boundary hook."""

    def __init__(self) -> None:
        self.residual_calls = 0
        self.boundary_calls = 0
        self.objective_calls = 0

    def residual(
        self,
        solution: Tensor,
        inputs: Tensor,
        *,
        grid: object,
        derivative: str = "spectral",
    ) -> Tensor:
        del grid, derivative
        self.residual_calls += 1
        return solution - inputs

    def boundary_residual(
        self,
        solution: Tensor,
        inputs: Tensor,
        *,
        grid: object,
    ) -> Tensor:
        del grid
        self.boundary_calls += 1
        return solution[..., :1] - inputs[..., :1]

    def physics_objective(
        self,
        residual: Tensor,
        *,
        grid: object,
        mode: str,
        derivative: str,
    ) -> Tensor:
        del grid, mode, derivative
        self.objective_calls += 1
        return torch.mean(residual.square())


class _ScaledIdentityGridOperator(nn.Module):
    """Small trainable GridOperator with no FNO or PINO internals."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.0))

    def predict(self, inputs: Tensor, *, grid: object) -> Tensor:
        assert getattr(grid, "n_dims") == 1
        return self.scale * inputs


def _extensible_config(*, steps: int, lr: float = 1e-2) -> dict[str, object]:
    return op.build_operator_config(
        "fno",
        training_mode="physics_only",
        steps=steps,
        seed=0,
        width=4,
        n_layers=1,
        n_modes=2,
        project_width=4,
        grid_size=8,
        batch_size=2,
        n_harmonics=1,
        lr=lr,
    )


def test_train_pino_uses_non_poisson_grid_problem_hooks() -> None:
    problem = _IdentityGridProblem()
    assert isinstance(problem, op.GridPDEProblem)

    result = op.train_pino(_extensible_config(steps=3), problem=problem, device="cpu")

    assert isinstance(result.model, op.PINO)
    assert math.isfinite(result.final_loss)
    assert math.isfinite(result.metrics["final_physics_loss"])
    assert math.isfinite(result.metrics["final_boundary_loss"])
    assert problem.residual_calls > 0
    assert problem.boundary_calls == problem.residual_calls
    assert problem.objective_calls == problem.residual_calls


def test_train_pino_accepts_non_fno_grid_operator_backbone() -> None:
    problem = _IdentityGridProblem()
    model = _ScaledIdentityGridOperator()
    assert isinstance(model, op.GridOperator)
    assert not isinstance(model, (op.FNO, op.PINO))

    result = op.train_pino(
        _extensible_config(steps=8, lr=0.1),
        problem=problem,
        model=model,
        device="cpu",
    )

    assert result.model is model
    assert float(model.scale.detach()) != 0.0
    assert result.final_loss < result.initial_loss
    assert result.metrics["final_physics_loss"] < result.metrics["initial_physics_loss"]
    inputs = torch.randn(2, 1, 8)
    grid = op.UniformGrid.from_shape(8, spacing=1.0, periodic=True)
    assert tuple(model.predict(inputs, grid=grid).shape) == tuple(inputs.shape)
