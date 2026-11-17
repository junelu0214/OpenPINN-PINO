from __future__ import annotations

import pytest

import openpinn as op
from openpinn.method_options import METHOD_PARAMETER_SCHEMAS


def test_every_registered_variant_has_one_parameter_schema() -> None:
    registered = {
        (family.name, variant.name)
        for family in op.list_methods()
        for variant in op.list_method_variants(family.name)
    }
    assert set(METHOD_PARAMETER_SCHEMAS) == registered
    for parameters in METHOD_PARAMETER_SCHEMAS.values():
        for schema in parameters.values():
            assert schema["description"]
            assert schema["effect"]


def test_every_documented_default_parameter_set_is_accepted() -> None:
    for (method_name, variant), parameters in METHOD_PARAMETER_SCHEMAS.items():
        options = {
            name: (
                schema["placeholder"]
                if schema["default"] is None and schema["required"]
                else schema["default"]
            )
            for name, schema in parameters.items()
            if schema["default"] is not None or schema["required"]
        }
        selected = op.method(method_name, variant, **options)
        assert selected.method == method_name
        assert selected.variant == variant


def test_method_helper_normalizes_aliases_and_validates_options() -> None:
    selected = op.method("rff", num_frequencies=8, include_input=False)
    assert selected.method == "fourier"
    assert selected.variant == "random_fourier"
    assert selected.options == {"num_frequencies": 8, "include_input": False}

    with pytest.raises(ValueError, match="Unknown option"):
        op.method("fourier", misspelled_scale=2.0)


def test_independent_methods_compose_into_fixed_component_slots() -> None:
    config, spec = op.build_config(
        "poisson",
        methods=[
            op.method("adaptive_weights", "relobralo", temperature=0.2),
            op.method("fourier", "random_fourier", num_frequencies=8),
            op.method("modified_mlp", "gated_mlp"),
        ],
        steps=3,
    )
    assert spec.name == "modified_mlp+fourier+adaptive_weights"
    assert [item["name"] for item in config["methods"]] == [
        "modified_mlp",
        "fourier",
        "adaptive_weights",
    ]
    assert config["model"]["name"] == "modified_mlp"
    assert config["model"]["feature"]["num_frequencies"] == 8
    assert config["loss"]["weighting"]["temperature"] == 0.2


def test_method_stack_rejects_same_family_and_same_slot() -> None:
    with pytest.raises(ValueError, match="Only one variant"):
        op.build_config(
            "burgers",
            methods=[
                op.method("adaptive_sampling", "residual_adaptive"),
                op.method("adaptive_sampling", "importance_sampling"),
            ],
            steps=2,
        )

    with pytest.raises(ValueError, match="both control 'sampling'"):
        op.build_config(
            "burgers",
            methods=[
                op.method("adaptive_sampling", "residual_adaptive"),
                op.method("time_window", "causal_window"),
            ],
            steps=2,
        )


def test_baseline_cannot_be_combined_with_improvement() -> None:
    with pytest.raises(ValueError, match="baseline/plain_pinn"):
        op.build_config(
            "poisson",
            methods=[
                op.method("baseline"),
                op.method("fourier"),
            ],
            steps=1,
        )


def test_adaptive_activation_and_adam_lbfgs_options_are_exposed() -> None:
    activation_config, _ = op.build_config(
        "burgers",
        methods=[op.method("adaptive_activation", initial_slope=1.5)],
        steps=1,
    )
    assert activation_config["model"]["activation_options"] == {
        "initial_slope": 1.5
    }

    optimizer_config, _ = op.build_config(
        "poisson",
        methods=[
            op.method(
                "adam_lbfgs",
                adam_lr=2e-3,
                lbfgs_lr=0.5,
                lbfgs_fraction=0.25,
                lbfgs_max_steps=3,
                lbfgs_max_iter=4,
                history_size=7,
            )
        ],
        steps=8,
    )
    stages = optimizer_config["optimizer"]["stages"]
    assert stages == [
        {"name": "adam", "steps": 6, "lr": 2e-3},
        {
            "name": "lbfgs",
            "steps": 2,
            "lr": 0.5,
            "max_iter": 4,
            "history_size": 7,
            "line_search_fn": "strong_wolfe",
        },
    ]


def test_legacy_single_method_arguments_remain_available() -> None:
    config, spec = op.build_config(
        "poisson",
        method="fourier",
        variant="random_fourier",
        method_options={"num_frequencies": 4},
        steps=1,
    )
    assert spec.name == "fourier"
    assert config["model"]["feature"]["num_frequencies"] == 4

    with pytest.raises(ValueError, match="not both"):
        op.build_config(
            "poisson",
            method="fourier",
            methods=[op.method("adaptive_weights")],
            steps=1,
        )


def test_log_output_interval_can_be_overridden_by_short_api() -> None:
    config, _ = op.build_config(
        "poisson",
        methods=[op.method("baseline")],
        steps=1000,
        training={"log_every": 100},
    )
    assert config["training"]["steps"] == 1000
    assert config["training"]["log_every"] == 100
