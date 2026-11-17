from __future__ import annotations

import pytest
import torch

import openpinn as op


EXPECTED_METHODS = {
    "baseline",
    "modified_mlp",
    "adaptive_activation",
    "fourier",
    "hard_boundary",
    "adaptive_weights",
    "adaptive_sampling",
    "adam_lbfgs",
    "time_window",
    "transfer",
    "domain_decomposition",
}


def test_registry_lists_baseline_and_ten_first_round_methods() -> None:
    specs = op.list_methods()
    assert {spec.name for spec in specs} == EXPECTED_METHODS
    assert all(spec.component_file.endswith(".py") for spec in specs)
    assert all(spec.default_variant == spec.variant for spec in specs)
    assert all(spec.variant_file.startswith("methods/") for spec in specs)


def test_adaptive_sampling_family_lists_and_resolves_concrete_variant() -> None:
    variants = op.list_method_variants("adaptive_sampling")
    assert [variant.name for variant in variants] == [
        "residual_adaptive",
        "residual_movement",
        "gradient_movement",
        "importance_sampling",
    ]
    config, spec = op.build_config(
        "burgers",
        method="adaptive_sampling",
        variant="residual_adaptive",
        steps=2,
    )
    assert spec.name == "adaptive_sampling"
    assert spec.variant == "residual_adaptive"
    assert config["method"]["variant"] == "residual_adaptive"
    assert config["method"]["variant_file"] == (
        "methods/adaptive_sampling/residual_adaptive.py"
    )


def test_old_family_only_call_uses_default_variant() -> None:
    config, _ = op.build_config("burgers", method="adaptive_sampling", steps=2)
    assert config["method"]["variant"] == "residual_adaptive"


def test_adaptive_weight_family_lists_all_concrete_variants() -> None:
    assert [
        variant.name for variant in op.list_method_variants("adaptive_weights")
    ] == [
        "self_adaptive",
        "uncertainty_weighting",
        "relobralo",
        "magnitude_power",
        "pointwise_self_adaptive",
    ]


def test_unknown_variant_is_rejected_with_family_choices() -> None:
    with pytest.raises(ValueError, match="available variants: residual_adaptive"):
        op.build_config(
            "burgers",
            method="adaptive_sampling",
            variant="not_registered",
            steps=2,
        )


def test_fourier_alias_resolves_to_feature_patch() -> None:
    config, spec = op.build_config("poisson", method="fourier_features", steps=2)
    assert spec.name == "fourier"
    assert spec.variant == "random_fourier"
    assert config["model"]["feature"]["name"] == "fourier"
    assert config["method"]["component_file"] == "features.py"


def test_problem_method_compatibility_is_explicit() -> None:
    with pytest.raises(ValueError, match="not compatible"):
        op.build_config("poisson", method="time_window", steps=2)
    with pytest.raises(ValueError, match="not compatible"):
        op.build_config("burgers", method="domain_decomposition", steps=2)


def test_adam_lbfgs_stages_match_requested_steps() -> None:
    config, _ = op.build_config("poisson", method="adam_lbfgs", steps=10)
    stages = config["optimizer"]["stages"]
    assert sum(stage["steps"] for stage in stages) == 10
    assert [stage["name"] for stage in stages] == ["adam", "lbfgs"]


def test_transfer_requires_a_checkpoint() -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        op.build_config("burgers", method="transfer", steps=2)


def test_custom_static_problem_reports_methods_without_training() -> None:
    def equation(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return -op.hessian(u, x, i=0, j=0)

    problem = op.PDEProblem(
        op.Interval(-1.0, 1.0),
        equation,
        [op.DirichletBC(0.0)],
        name="report_only",
        metadata={"kind": "custom_static"},
    )
    report = op.check_method_compatibility(problem)

    assert report.problem == "report_only"
    assert report.family == "static"
    assert ("time_window", "causal_window") in {
        (item.method, item.variant) for item in report.incompatible
    }
    assert ("transfer", "checkpoint_transfer") in {
        (item.method, item.variant) for item in report.conditional
    }
    assert ("adaptive_weights", "pointwise_self_adaptive") in {
        (item.method, item.variant) for item in report.applicable
    }
    assert "不是精度保证" in report.to_text()


def test_custom_static_family_is_used_by_actual_method_resolution() -> None:
    problem = op.PDEProblem(
        op.Interval(-1.0, 1.0),
        lambda x, u: -op.hessian(u, x, i=0, j=0),
        [op.DirichletBC(0.0)],
        metadata={"kind": "custom_static"},
    )
    config, spec = op.build_config(
        problem,
        method="adaptive_weights",
        variant="pointwise_self_adaptive",
        steps=2,
    )
    assert spec.variant == "pointwise_self_adaptive"
    assert config["loss"]["weighting"]["name"] == "pointwise_self_adaptive"


def test_dynamic_and_heterogeneous_reports_use_distinct_rules() -> None:
    dynamic = op.PDEProblem(
        op.SpaceTimeDomain(op.Interval(-1.0, 1.0), 0.0, 1.0),
        lambda x, u: op.grad(u, x)[:, 1:2],
        [op.DirichletBC(0.0), op.InitialCondition(0.0)],
        name="dynamic_report",
    )
    dynamic_report = op.check_method_compatibility(dynamic)
    assert ("time_window", "causal_window") in {
        (item.method, item.variant) for item in dynamic_report.applicable
    }

    heterogeneous = op.PDEProblem(
        op.Interval(-1.0, 1.0),
        lambda x, u: -op.hessian(u, x, i=0, j=0),
        [op.DirichletBC(0.0)],
        name="heterogeneous_report",
        metadata={
            "kind": "heterogeneous_static",
            "interface": 0.0,
            "left_diffusivity": 1.0,
            "right_diffusivity": 4.0,
        },
    )
    heterogeneous_report = op.check_method_compatibility(heterogeneous)
    domain_result = next(
        item
        for item in heterogeneous_report.methods
        if item.method == "domain_decomposition"
    )
    assert domain_result.status == "conditional"
