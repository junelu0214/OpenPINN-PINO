"""Tests for M2 ProblemSpec and MethodSpec YAML schemas + validation."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from pydantic import ValidationError

from openpinn.spec import (
    ProblemSpec,
    MethodSpec,
    load_problem_spec,
    load_method_spec,
    validate_problem_spec,
    validate_method_spec,
)

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "specs"

# ---- Fixtures: minimal valid dicts ----------------------------------------

def _valid_problem() -> dict:
    return {
        "dim": 2,
        "physics": "linear_elasticity",
        "geometry": {
            "outer": {
                "type": "rectangle",
                "x_min": 0.0,
                "x_max": 2.0,
                "y_min": 0.0,
                "y_max": 2.0,
            },
            "holes": [
                {"type": "circle", "center": [1.0, 1.0], "radius": 0.25},
            ],
        },
        "field": {"kind": "displacement", "components": ["ux", "uy"]},
        "constitutive": {
            "model": "isotropic_linear",
            "E": 210e9,
            "nu": 0.3,
            "stress_voigt": True,
        },
        "boundaries": [
            {"id": "fix_left", "region": "left", "type": "dirichlet",
             "dofs": ["ux", "uy"], "value": [0.0, 0.0]},
            {"id": "traction_right", "region": "right", "type": "neumann",
             "dofs": ["ux"], "value": 1e6},
            {"id": "shear_right", "region": "right", "type": "neumann",
             "dofs": ["uy"], "value": 0.0},
            {"id": "free_top", "region": "top", "type": "neumann",
             "dofs": ["ux", "uy"], "value": [0.0, 0.0]},
            {"id": "free_bottom", "region": "bottom", "type": "neumann",
             "dofs": ["ux", "uy"], "value": [0.0, 0.0]},
            {"id": "free_hole", "region": "hole_0", "type": "neumann",
             "dofs": ["ux", "uy"], "value": [0.0, 0.0]},
        ],
        "units": {"length": "m", "force": "N", "stress": "Pa"},
    }


def _valid_method_pinn() -> dict:
    return {
        "method": "baseline",
        "variant": "plain_pinn",
        "path": "pinn",
        "sampling": {"name": "uniform", "interior": 256, "boundary": 64},
        "loss": {"formulation": "strong"},
        "optimizer": {"name": "adam", "lr": 1e-3},
        "network": {"name": "mlp", "hidden_layers": [64, 64], "activation": "tanh"},
        "differentiator": {"mode": "autodiff"},
        "training": {"steps": 1000},
    }


def _valid_method_operator() -> dict:
    return {
        "method": "deeponet",
        "path": "operator",
        "sampling": {"name": "uniform", "interior": 512, "boundary": 128},
        "loss": {"formulation": "strong"},
        "optimizer": {"name": "adam", "lr": 5e-4},
        "network": {"name": "mlp", "hidden_layers": [128, 128], "activation": "tanh"},
        "differentiator": {"mode": "autodiff"},
        "training": {"steps": 5000},
    }


# ===========================================================================
# ProblemSpec: positive tests
# ===========================================================================

class TestProblemSpecValid:
    def test_load_from_dict(self):
        spec = load_problem_spec(_valid_problem())
        assert spec.dim == 2
        assert spec.physics.value == "linear_elasticity"
        assert len(spec.geometry.holes) == 1
        assert spec.field.components == ["ux", "uy"]
        assert spec.constitutive.E == 210e9
        assert spec.constitutive.nu == 0.3

    def test_load_from_yaml_file(self):
        path = EXAMPLES / "plate_with_hole_problem.yaml"
        if not path.exists():
            pytest.skip("Example YAML not found")
        spec = load_problem_spec(str(path))
        assert spec.dim == 2
        assert len(spec.boundaries) >= 6

    def test_validate_returns_empty(self):
        assert validate_problem_spec(_valid_problem()) == []

    def test_no_holes_if_not_needed(self):
        d = _valid_problem()
        d["geometry"]["holes"] = []
        del d["boundaries"][5]  # remove hole_0 BC
        spec = load_problem_spec(d)
        assert len(spec.geometry.holes) == 0

    def test_default_units(self):
        d = _valid_problem()
        del d["units"]
        spec = load_problem_spec(d)
        assert spec.units.length == "m"

    def test_scalar_bc_value(self):
        d = _valid_problem()
        d["boundaries"][0]["value"] = 0.0
        d["boundaries"][0]["dofs"] = ["ux"]
        spec = load_problem_spec(d)
        assert spec.boundaries[0].value == 0.0


# ===========================================================================
# ProblemSpec: negative tests
# ===========================================================================

class TestProblemSpecInvalid:

    def test_missing_required_field_dim(self):
        d = _valid_problem()
        del d["dim"]
        with pytest.raises(ValidationError, match="dim"):
            load_problem_spec(d)

    def test_missing_required_field_physics(self):
        d = _valid_problem()
        del d["physics"]
        with pytest.raises(ValidationError, match="physics"):
            load_problem_spec(d)

    def test_missing_required_field_geometry(self):
        d = _valid_problem()
        del d["geometry"]
        with pytest.raises(ValidationError, match="geometry"):
            load_problem_spec(d)

    def test_missing_required_field_constitutive(self):
        d = _valid_problem()
        del d["constitutive"]
        with pytest.raises(ValidationError, match="constitutive"):
            load_problem_spec(d)

    def test_wrong_dim(self):
        d = _valid_problem()
        d["dim"] = 3
        with pytest.raises(ValidationError, match="dim"):
            load_problem_spec(d)

    def test_unknown_physics(self):
        d = _valid_problem()
        d["physics"] = "navier_stokes"
        with pytest.raises(ValidationError, match="physics"):
            load_problem_spec(d)

    def test_unknown_top_level_key(self):
        d = _valid_problem()
        d["foobar"] = 42
        with pytest.raises(ValidationError, match="foobar"):
            load_problem_spec(d)

    def test_unknown_geometry_key(self):
        d = _valid_problem()
        d["geometry"]["extra_stuff"] = True
        with pytest.raises(ValidationError, match="extra_stuff"):
            load_problem_spec(d)

    # --- Geometry: hole outside domain ---

    def test_hole_outside_domain_right(self):
        d = _valid_problem()
        d["geometry"]["holes"][0]["center"] = [1.9, 1.0]
        with pytest.raises(ValidationError, match="(?i)hole.*inside"):
            load_problem_spec(d)

    def test_hole_outside_domain_top(self):
        d = _valid_problem()
        d["geometry"]["holes"][0]["center"] = [1.0, 1.9]
        with pytest.raises(ValidationError, match="(?i)hole.*inside"):
            load_problem_spec(d)

    def test_hole_outside_domain_left(self):
        d = _valid_problem()
        d["geometry"]["holes"][0]["center"] = [0.1, 1.0]
        with pytest.raises(ValidationError, match="(?i)hole.*inside"):
            load_problem_spec(d)

    def test_hole_zero_radius(self):
        d = _valid_problem()
        d["geometry"]["holes"][0]["radius"] = 0.0
        with pytest.raises(ValidationError, match="radius"):
            load_problem_spec(d)

    def test_hole_negative_radius(self):
        d = _valid_problem()
        d["geometry"]["holes"][0]["radius"] = -0.5
        with pytest.raises(ValidationError, match="radius"):
            load_problem_spec(d)

    def test_overlapping_holes(self):
        d = _valid_problem()
        d["geometry"]["holes"].append(
            {"type": "circle", "center": [1.2, 1.0], "radius": 0.25}
        )
        d["boundaries"].append(
            {"id": "free_hole1", "region": "hole_1", "type": "neumann",
             "dofs": ["ux", "uy"], "value": [0.0, 0.0]}
        )
        with pytest.raises(ValidationError, match="(?i)overlap|touch"):
            load_problem_spec(d)

    def test_rectangle_x_bounds_inverted(self):
        d = _valid_problem()
        d["geometry"]["outer"]["x_min"] = 5.0
        d["geometry"]["outer"]["x_max"] = 1.0
        with pytest.raises(ValidationError, match="x_max"):
            load_problem_spec(d)

    # --- Constitutive: E <= 0 ---

    def test_e_zero(self):
        d = _valid_problem()
        d["constitutive"]["E"] = 0.0
        with pytest.raises(ValidationError, match="E"):
            load_problem_spec(d)

    def test_e_negative(self):
        d = _valid_problem()
        d["constitutive"]["E"] = -100.0
        with pytest.raises(ValidationError, match="E"):
            load_problem_spec(d)

    def test_nu_too_high(self):
        d = _valid_problem()
        d["constitutive"]["nu"] = 0.5
        with pytest.raises(ValidationError, match="nu"):
            load_problem_spec(d)

    def test_nu_at_minus_one_allowed(self):
        d = _valid_problem()
        d["constitutive"]["nu"] = -1.0
        spec = load_problem_spec(d)
        assert spec.constitutive.nu == -1.0

    # --- Boundaries: Dirichlet/Neumann conflict ---

    def test_dirichlet_neumann_conflict_same_dof(self):
        d = _valid_problem()
        d["boundaries"].append(
            {"id": "extra", "region": "left", "type": "neumann",
             "dofs": ["ux"], "value": 0.0}
        )
        with pytest.raises(ValidationError, match="(?i)conflict.*dirichlet.*neumann|conflict.*BC"):
            load_problem_spec(d)

    # --- Boundaries: missing required edge ---

    def test_missing_required_edge_top(self):
        d = _valid_problem()
        d["boundaries"] = [
            bc for bc in d["boundaries"] if bc["region"] != "top"
        ]
        with pytest.raises(ValidationError, match="(?i)missing.*top"):
            load_problem_spec(d)

    def test_missing_required_edge_hole(self):
        d = _valid_problem()
        d["boundaries"] = [
            bc for bc in d["boundaries"] if bc["region"] != "hole_0"
        ]
        with pytest.raises(ValidationError, match="(?i)missing.*hole_0"):
            load_problem_spec(d)

    def test_missing_all_boundaries(self):
        d = _valid_problem()
        d["boundaries"] = []
        with pytest.raises(ValidationError, match="(?i)at least one"):
            load_problem_spec(d)

    # --- Field: unknown dofs in BC ---

    def test_bc_references_unknown_dof(self):
        d = _valid_problem()
        d["boundaries"][0]["dofs"] = ["ux", "uz"]
        with pytest.raises(ValidationError, match="(?i)unknown dofs"):
            load_problem_spec(d)

    def test_empty_components(self):
        d = _valid_problem()
        d["field"]["components"] = []
        with pytest.raises(ValidationError, match="(?i)components.*empty"):
            load_problem_spec(d)

    def test_duplicate_components(self):
        d = _valid_problem()
        d["field"]["components"] = ["ux", "ux"]
        with pytest.raises(ValidationError, match="(?i)unique"):
            load_problem_spec(d)

    # --- BC: value list length mismatch ---

    def test_bc_value_length_mismatch(self):
        d = _valid_problem()
        d["boundaries"][0]["value"] = [0.0, 0.0, 0.0]
        with pytest.raises(ValidationError, match="(?i)value.*length"):
            load_problem_spec(d)

    # --- validate_problem_spec helper ---

    def test_validate_returns_errors(self):
        d = _valid_problem()
        del d["dim"]
        errors = validate_problem_spec(d)
        assert len(errors) > 0
        assert "dim" in errors[0].lower() or "field required" in errors[0].lower()


# ===========================================================================
# MethodSpec: positive tests
# ===========================================================================

class TestMethodSpecValid:
    def test_load_pinn_baseline(self):
        spec = load_method_spec(_valid_method_pinn())
        assert spec.method == "baseline"
        assert spec.path == "pinn"

    def test_load_operator_deeponet(self):
        spec = load_method_spec(_valid_method_operator())
        assert spec.method == "deeponet"
        assert spec.path == "operator"

    def test_load_from_yaml_file_pinn(self):
        path = EXAMPLES / "pinn_method.yaml"
        if not path.exists():
            pytest.skip("Example YAML not found")
        spec = load_method_spec(str(path))
        assert spec.method == "baseline"

    def test_load_from_yaml_file_deeponet(self):
        path = EXAMPLES / "deeponet_method.yaml"
        if not path.exists():
            pytest.skip("Example YAML not found")
        spec = load_method_spec(str(path))
        assert spec.method == "deeponet"

    def test_validate_returns_empty(self):
        assert validate_method_spec(_valid_method_pinn()) == []

    def test_all_pinn_families(self):
        families = [
            ("baseline", "plain_pinn"),
            ("fourier", "random_fourier"),
            ("adaptive_weights", "self_adaptive"),
            ("adaptive_sampling", "residual_adaptive"),
            ("modified_mlp", "gated_mlp"),
            ("hard_boundary", "exact_output_transform"),
            ("adam_lbfgs", "adam_then_lbfgs"),
            ("adaptive_activation", "adaptive_tanh"),
            ("time_window", "causal_window"),
            ("transfer", "checkpoint_transfer"),
            ("domain_decomposition", "two_subdomain"),
        ]
        for method, variant in families:
            d = _valid_method_pinn()
            d["method"] = method
            d["variant"] = variant
            spec = load_method_spec(d)
            assert spec.method == method

    def test_operator_fno(self):
        d = _valid_method_operator()
        d["method"] = "fno"
        d["differentiator"]["mode"] = "grid"
        spec = load_method_spec(d)
        assert spec.method == "fno"

    def test_defaults_filled(self):
        d = {"method": "baseline"}
        spec = load_method_spec(d)
        assert spec.path == "pinn"
        assert spec.sampling.interior == 256
        assert spec.optimizer.lr == 1e-3


# ===========================================================================
# MethodSpec: negative tests
# ===========================================================================

class TestMethodSpecInvalid:

    def test_unknown_method_name(self):
        d = _valid_method_pinn()
        d["method"] = "magic_solver"
        with pytest.raises(ValidationError, match="(?i)unknown.*method"):
            load_method_spec(d)

    def test_unknown_variant(self):
        d = _valid_method_pinn()
        d["variant"] = "nonexistent_variant"
        with pytest.raises(ValidationError, match="(?i)unknown.*variant"):
            load_method_spec(d)

    def test_unknown_operator(self):
        d = _valid_method_operator()
        d["method"] = "transformer_operator"
        with pytest.raises(ValidationError, match="(?i)unknown.*operator"):
            load_method_spec(d)

    def test_operator_with_variant_rejected(self):
        d = _valid_method_operator()
        d["variant"] = "some_variant"
        with pytest.raises(ValidationError, match="(?i)variant"):
            load_method_spec(d)

    def test_unknown_top_level_key(self):
        d = _valid_method_pinn()
        d["unknown_field"] = True
        with pytest.raises(ValidationError, match="unknown_field"):
            load_method_spec(d)

    def test_unknown_optimizer(self):
        d = _valid_method_pinn()
        d["optimizer"]["name"] = "rmsprop"
        with pytest.raises(ValidationError, match="(?i)unknown.*optimizer"):
            load_method_spec(d)

    def test_unknown_loss_formulation(self):
        d = _valid_method_pinn()
        d["loss"]["formulation"] = "collocation_magic"
        with pytest.raises(ValidationError, match="(?i)unknown.*formulation"):
            load_method_spec(d)

    def test_negative_lr(self):
        d = _valid_method_pinn()
        d["optimizer"]["lr"] = -0.01
        with pytest.raises(ValidationError, match="lr"):
            load_method_spec(d)

    def test_zero_interior_samples(self):
        d = _valid_method_pinn()
        d["sampling"]["interior"] = 0
        with pytest.raises(ValidationError, match="interior"):
            load_method_spec(d)

    def test_negative_training_steps(self):
        d = _valid_method_pinn()
        d["training"]["steps"] = -10
        with pytest.raises(ValidationError, match="(?i)steps.*positive"):
            load_method_spec(d)

    def test_extra_sampling_key(self):
        d = _valid_method_pinn()
        d["sampling"]["magic_key"] = 42
        with pytest.raises(ValidationError, match="magic_key"):
            load_method_spec(d)

    def test_validate_returns_errors(self):
        d = _valid_method_pinn()
        d["method"] = "magic_solver"
        errors = validate_method_spec(d)
        assert len(errors) > 0


# ===========================================================================
# Cross-spec integration: load both YAML examples
# ===========================================================================

class TestExampleYAMLFiles:
    def test_plate_with_hole_example_roundtrip(self):
        path = EXAMPLES / "plate_with_hole_problem.yaml"
        if not path.exists():
            pytest.skip("Example YAML not found")
        spec = load_problem_spec(str(path))
        reloaded = load_problem_spec(spec.model_dump())
        assert reloaded.dim == spec.dim
        assert len(reloaded.boundaries) == len(spec.boundaries)

    def test_pinn_method_example_roundtrip(self):
        path = EXAMPLES / "pinn_method.yaml"
        if not path.exists():
            pytest.skip("Example YAML not found")
        spec = load_method_spec(str(path))
        reloaded = load_method_spec(spec.model_dump())
        assert reloaded.method == spec.method
