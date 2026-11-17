from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import pytest

import openpinn as op
from openpinn.ecosystem_components import DiagonalCurvatureAdapter, GradientStatisticLoss


def test_method_ecosystem_has_paper_five_axes_and_live_statuses() -> None:
    payload = op.method_ecosystem_payload()
    axes = payload["axes"]
    assert [axis["kind"] for axis in axes] == [
        "model", "sampler", "loss", "optimizer", "derivative"
    ]
    assert all(axis["families"] for axis in axes)
    model = next(axis for axis in axes if axis["kind"] == "model")
    fno = next(item for item in model["families"] if item["slug"] == "spectral_operators")
    graph = next(item for item in model["families"] if item["slug"] == "graph_operators")
    assert fno["status"] == "built_in"
    assert fno["implementations"] == ["operator:fno", "model:lowrankoperator"]
    assert graph["status"] == "built_in"
    assert graph["implementations"] == ["model:gno", "model:multipolegno"]
    assert "特征值" not in fno["setting"]
    assert all(family["boundary"] and family["status"] == "built_in" for axis in axes for family in axis["families"])
    second = next(item for axis in axes if axis["kind"] == "optimizer" for item in axis["families"] if item["slug"] == "second_order")
    ntk = next(item for axis in axes if axis["kind"] == "loss" for item in axis["families"] if item["slug"] == "ntk_balanced")
    discrete = next(item for axis in axes if axis["kind"] == "derivative" for item in axis["families"] if item["slug"] == "discrete_grid")
    assert "parameter -= lr * grad / (|grad| + damping)" in second["boundary"]
    assert "sqrt(v)" in second["boundary"]
    assert "平方和" in ntk["boundary"]
    assert "缺少坐标 grad/hessian 后拒绝训练" in discrete["boundary"]
    docs = Path("docs/methods.rst").read_text(encoding="utf-8")
    assert "parameter -= lr * grad / (|grad| + damping)" in docs
    assert "(u[i+1] - u[i-1]) / (2 * spacing)" in docs
    assert "预印本 4.4 节一致" in docs
    extension = Path("docs/_ext/openpinn_method_catalog.py").read_text(encoding="utf-8")
    assert "ECOSYSTEM_REFERENCES" in extension
    assert "简单比较示例" in extension
    for ecosystem_axis in axes:
        for ecosystem_family in ecosystem_axis["families"]:
            assert f'"{ecosystem_family["slug"]}"' in extension
    weak = next(
        item
        for axis in axes
        if axis["kind"] == "derivative"
        for item in axis["families"]
        if item["slug"] == "weak_variational"
    )
    assert "solve()" in weak["setting"]
    assert all(
        family["status"] == "built_in"
        for axis in axes
        for family in axis["families"]
    )


def test_solve_accepts_custom_five_axis_registry(tmp_path) -> None:
    registry = op.default_component_registry()
    problems_before = registry.available("problem")

    class TwiceLoss:
        def compose(self, terms: Mapping[str, torch.Tensor]) -> torch.Tensor:
            return 2.0 * sum(terms.values())

    def triple_loss(terms: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return 3.0 * sum(terms.values())

    registry.register("loss", "twice", lambda **kwargs: TwiceLoss())
    registry.register("loss", "triple", lambda **kwargs: triple_loss)
    problem = op.make_poisson_problem()
    composed = op.solve(
        problem,
        steps=1,
        plot=False,
        output_dir=tmp_path / "compose-loss",
        overrides={"loss": {"weighting": "twice"}, "model": {"hidden_layers": [8, 8]}},
        registry=registry,
    )
    called = op.solve(
        "poisson",
        steps=1,
        plot=False,
        output_dir=tmp_path / "callable-loss",
        overrides={"loss": {"weighting": "triple"}, "model": {"hidden_layers": [8, 8]}},
        registry=registry,
    )
    assert registry.available("problem") == problems_before
    for result, factor in ((composed, 2.0), (called, 3.0)):
        last = result.history[-1]
        expected = factor * (last["pde"] + last["boundary"])
        assert last["total"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "name", ["gno", "multipole_gno", "low_rank_operator", "physics_informed_kan"]
)
def test_new_model_families_are_executable(name: str, tmp_path) -> None:
    registry = op.default_component_registry()
    problem = op.make_poisson_problem()
    model = registry.create(
        "model",
        name,
        config={"name": name, "hidden_layers": [8, 8], "activation": "tanh"},
        problem=problem,
        registry=registry,
    )
    points = torch.linspace(-1.0, 1.0, 8).reshape(-1, 1).requires_grad_(True)
    prediction = model(points)
    assert prediction.shape == (8, 1)
    first = op.grad(prediction, points)
    second = op.hessian(prediction, points)
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()
    result = op.solve(
        "poisson",
        steps=1,
        plot=False,
        output_dir=tmp_path / name,
        overrides={"model": {"name": name, "hidden_layers": [8, 8]}},
    )
    assert torch.isfinite(torch.tensor(result.history[-1]["total"]))


@pytest.mark.parametrize("name", ["grid", "lhs", "qmc", "rar", "uncertainty", "das"])
def test_new_sampling_families_sample_and_update(name: str) -> None:
    registry = op.default_component_registry()
    problem = op.make_poisson_problem()
    config = {"name": name, "interior": 8, "boundary": 4}
    if name in {"rar", "uncertainty", "das"}:
        config.update({"candidate_pool": 16, "enrich": 2, "density_steps": 2})
    sampler = registry.create(
        "sampler",
        name,
        config=config,
        problem=problem,
        seed=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    batch = sampler.sample(problem)
    assert batch.interior.shape == (8, 1)
    if name in {"rar", "uncertainty", "das"}:
        model = op.PINNModel(op.MLP(1, 1, hidden_layers=(8,)))
        sampler.update(model)
        batch = sampler.sample(problem)
        assert torch.isfinite(batch.interior).all()


@pytest.mark.parametrize(
    "name", ["gradient_balanced", "ntk_balanced", "inverse_dirichlet"]
)
def test_new_loss_families_complete_one_training_step(name: str, tmp_path) -> None:
    result = op.solve(
        "poisson",
        steps=1,
        plot=False,
        output_dir=tmp_path / name,
        overrides={"loss": {"weighting": name}},
    )
    assert torch.isfinite(torch.tensor(result.history[-1]["total"]))


@pytest.mark.parametrize("name", ["gauss_newton", "natural_gradient"])
def test_curvature_optimizers_complete_one_training_step(name: str, tmp_path) -> None:
    result = op.solve(
        "poisson",
        steps=1,
        plot=False,
        output_dir=tmp_path / name,
        overrides={"optimizer": {"name": name, "lr": 1e-3}},
    )
    assert torch.isfinite(torch.tensor(result.history[-1]["total"]))


def test_new_derivative_families_execute_their_native_contracts() -> None:
    registry = op.default_component_registry()
    field = torch.sin(torch.linspace(0, 2 * torch.pi, 16)).reshape(1, 1, 16)
    for name in ("finite_volume", "hybrid_ad_numerical"):
        derivative = registry.create("derivative", name, config={"spacing": 0.1})
        assert torch.isfinite(derivative.partial(field, axis=0)).all()
    weak = registry.create("derivative", "weak_variational")
    points = torch.linspace(0, 1, 8).reshape(-1, 1).requires_grad_(True)
    assert torch.isfinite(weak.grad(points.square(), points)).all()
    integral = weak.integrate_by_parts(
        torch.ones_like(points), torch.ones_like(points), torch.full_like(points, 1 / 8)
    )
    assert torch.isfinite(integral).all()
    finite_volume = registry.create("derivative", "finite_volume", config={"spacing": 0.2})
    finite_difference = registry.create(
        "derivative", "finitedifference", config={"spacing": 0.2}
    )
    torch.testing.assert_close(
        finite_volume.partial(field, axis=0, order=1),
        finite_difference.partial(field, axis=0, order=1),
    )


def test_graph_model_respects_hard_interval_constraint() -> None:
    registry = op.default_component_registry()
    problem = op.make_poisson_problem()
    model = registry.create(
        "model",
        "gno",
        config={
            "name": "gno",
            "hidden_layers": [8],
            "output_transform": {"name": "hard"},
        },
        problem=problem,
        registry=registry,
    )
    endpoints = torch.tensor([[problem.domain.lower], [problem.domain.upper]])
    torch.testing.assert_close(model(endpoints), problem.exact(endpoints), atol=1e-5, rtol=1e-5)


def test_rar_keeps_previously_enriched_points() -> None:
    registry = op.default_component_registry()
    problem = op.make_poisson_problem()
    sampler = registry.create(
        "sampler",
        "rar",
        config={"interior": 4, "boundary": 2, "options": {"candidate_pool": 8, "enrich": 2}},
        problem=problem,
        seed=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    model = op.PINNModel(op.MLP(1, 1, hidden_layers=(8,)))
    sampler.sample(problem)
    sampler.update(model)
    first = sampler.state_dict()["selected"].clone()
    sampler.update(model)
    second = sampler.state_dict()["selected"]
    retained = first[-2:]
    matches = (
        (second[:, None, :] - retained[None, :, :]).abs().sum(dim=-1) < 1e-6
    ).any(dim=0)
    assert bool(matches.all())
    assert second.shape == (4, 1)


@pytest.mark.parametrize("name", ["rar", "uncertainty", "das"])
def test_adaptive_samplers_survive_a_second_training_step(name: str, tmp_path) -> None:
    options = {"candidate_pool": 12, "enrich": 2, "ensemble_size": 2, "density_steps": 1}
    result = op.solve(
        "poisson",
        steps=2,
        plot=False,
        output_dir=tmp_path / name,
        overrides={
            "model": {"hidden_layers": [8, 8]},
            "sampling": {
                "name": name,
                "interior": 8,
                "boundary": 4,
                "options": options,
            },
        },
    )
    assert torch.isfinite(torch.tensor(result.history[-1]["total"]))


def test_causal_loss_sorts_space_time_and_rejects_static(tmp_path) -> None:
    with pytest.raises(ValueError, match="space-time"):
        op.solve(
            "poisson",
            steps=1,
            plot=False,
            output_dir=tmp_path / "static-causal",
            overrides={"loss": {"weighting": "causal"}, "model": {"hidden_layers": [8, 8]}},
        )
    result = op.solve(
        "burgers",
        steps=1,
        plot=False,
        output_dir=tmp_path / "burgers-causal",
        overrides={
            "loss": {"weighting": "causal"},
            "model": {"hidden_layers": [8, 8]},
            "sampling": {"interior": 12, "boundary": 4, "initial": 4},
        },
    )
    assert torch.isfinite(torch.tensor(result.history[-1]["total"]))

    composer = op.default_component_registry().create(
        "loss", "causal", config={"name": "causal", "options": {"epsilon": 5.0}}
    )
    problem = op.make_burgers_problem()
    interior = torch.tensor([[0.0, 0.9], [0.0, 0.1], [0.0, 0.5]], requires_grad=True)
    batch = op.SampleBatch(
        interior=interior,
        boundary=torch.tensor([[-1.0, 0.2], [1.0, 0.8]]),
        initial=torch.tensor([[0.2, 0.0], [-0.4, 0.0]]),
    )
    composer.bind_batch(batch, problem)
    total = composer.compose_pointwise(
        {
            "pde": torch.tensor([0.0, 4.0, 0.0]),
            "boundary": torch.tensor([1.0, 1.0]),
            "initial": torch.tensor([2.0, 2.0]),
        }
    )
    ordered = torch.tensor([4.0, 0.0, 0.0])
    prefix = torch.cumsum(ordered, dim=0) - ordered
    causal = torch.exp(-5.0 * prefix / 3.0)
    boundary = torch.tensor([1.0, 1.0])
    boundary_prefix = torch.cumsum(boundary, dim=0) - boundary
    boundary_causal = torch.exp(-5.0 * boundary_prefix / 2.0)
    expected = (causal * ordered).mean() + (boundary_causal * boundary).mean() + torch.tensor(2.0)
    torch.testing.assert_close(total, expected)


def test_checkpoint_restores_dynamic_loss_weights_and_curvature(tmp_path) -> None:
    def build() -> op.Trainer:
        registry = op.default_component_registry()
        problem = op.make_poisson_problem()
        model = registry.create(
            "model",
            "mlp",
            config={"name": "mlp", "hidden_layers": [8, 8]},
            problem=problem,
            registry=registry,
        )
        sampler = registry.create(
            "sampler",
            "uniform",
            config={"interior": 16, "boundary": 8},
            problem=problem,
            seed=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        loss = registry.create(
            "loss",
            "gradient_balanced",
            config={"name": "gradient_balanced"},
            weights={"pde": 1.0, "boundary": 1.0},
        )
        optimizer = registry.create(
            "optimizer",
            "natural_gradient",
            config={"name": "natural_gradient", "lr": 1e-3, "options": {"beta": 0.5}},
            parameters=[parameter for parameter in model.parameters() if parameter.requires_grad],
        )
        return op.Trainer(model, problem, sampler, optimizer=optimizer, loss_composer=loss)

    original = build()
    original.fit(steps=3, resample_every=100)
    assert original.loss_composer._dynamic
    assert any(float(item.abs().sum()) > 0.0 for item in original.optimizer._curvature)
    checkpoint = tmp_path / "checkpoint.pt"
    original.save(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["loss_composer"]["dynamic_weights"]
    assert state["optimizer"]["natural"] is True
    assert state["optimizer"]["curvature"]

    restored = build()
    restored.load(checkpoint)
    assert restored.loss_composer._dynamic == original.loss_composer._dynamic
    for left, right in zip(
        original.optimizer._curvature, restored.optimizer._curvature, strict=True
    ):
        torch.testing.assert_close(left, right)
    blank = build()
    blank.load(checkpoint)
    for item in blank.optimizer._curvature:
        item.zero_()
    saved_rng = state["torch_rng_state"].cpu().clone()
    torch.set_rng_state(saved_rng.clone())
    restored.fit(steps=1, resample_every=100)
    torch.set_rng_state(saved_rng.clone())
    blank.fit(steps=1, resample_every=100)
    assert any(
        not torch.allclose(left, right)
        for left, right in zip(
            restored.model.parameters(), blank.model.parameters(), strict=True
        )
    )


def test_gauss_newton_step_matches_normalized_gradient() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = DiagonalCurvatureAdapter([parameter], lr=0.1, damping=0.1, natural=False)

    def closure() -> torch.Tensor:
        return parameter.square().sum()

    optimizer.step(closure)
    gradient = 2.0
    expected = 1.0 - 0.1 * gradient / (abs(gradient) + 0.1)
    torch.testing.assert_close(parameter.detach(), torch.tensor([expected]))


def test_ntk_balanced_statistic_is_squared_parameter_gradient() -> None:
    layer = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        layer.weight.fill_(0.5)
    composer = GradientStatisticLoss(rule="ntk", momentum=0.0)
    composer.bind_model(layer)
    weight = layer.weight
    composer.weighted_terms({"a": weight.square().sum(), "b": (weight * 2).square().sum()})
    assert composer._dynamic["a"] == pytest.approx(8.5)
    assert composer._dynamic["b"] == pytest.approx(8.5 / 16.0)


def test_solve_rejects_grid_only_differentiator_but_keeps_option_checks(tmp_path) -> None:
    with pytest.raises(ValueError, match="grid-only"):
        op.solve(
            "poisson",
            steps=1,
            plot=False,
            output_dir=tmp_path / "no-differentiator",
            overrides={"differentiator": {"mode": "finite_volume"}},
        )
    with pytest.raises(ValueError, match="unsupported option"):
        op.solve(
            "poisson",
            steps=1,
            plot=False,
            output_dir=tmp_path / "bad-option",
            overrides={"differentiator": {"mode": "finite_volume", "options": {"stride": 2}}},
        )


def test_ecosystem_overrides_win_over_legacy_method_patches() -> None:
    config, _ = op.build_config(
        "poisson",
        methods=[op.method("fourier", "random_fourier")],
        steps=1,
        overrides={
            "model": {"name": "gno", "feature": {"name": "identity"}},
            "differentiator": {"mode": "weak_variational"},
        },
    )
    assert config["model"]["name"] == "gno"
    assert config["model"]["feature"]["name"] == "identity"
    assert config["differentiator"]["mode"] == "weak_variational"
    assert payload_has_selector()


def payload_has_selector() -> bool:
    payload = op.method_ecosystem_payload()
    contract = payload["configurator"]
    assert [control["axis"] for control in contract["controls"]] == [
        "model", "model", "model", "sampler", "loss", "optimizer", "derivative"
    ]
    assert all(family["documentation"].endswith(".html") for axis in payload["axes"] for family in axis["families"])
    derivative = next(control for control in contract["controls"] if control["key"] == "derivative")
    assert [choice["value"] for choice in derivative["choices"]] == [
        "autodiff", "weakvariational", "hybridadnumerical",
        "finitedifference", "grid", "finitevolume",
    ]
    assert all("paths" in choice for control in contract["controls"] for choice in control["choices"])
    return True


def test_selected_weak_backend_trains_without_lowering_derivative_order(tmp_path) -> None:
    result = op.solve(
        "poisson",
        steps=1,
        plot=False,
        output_dir=tmp_path / "weak",
        overrides={
            "differentiator": {"mode": "weak_variational"},
            "model": {"hidden_layers": [8, 8]},
        },
    )
    assert torch.isfinite(torch.tensor(result.history[-1]["total"]))


def test_selected_hybrid_backend_keeps_autodiff_strong_form(tmp_path) -> None:
    result = op.solve(
        "poisson",
        steps=1,
        plot=False,
        output_dir=tmp_path / "hybrid",
        overrides={
            "differentiator": {"mode": "hybrid_ad_numerical"},
            "model": {"hidden_layers": [8, 8]},
        },
    )
    assert torch.isfinite(torch.tensor(result.history[-1]["total"]))
