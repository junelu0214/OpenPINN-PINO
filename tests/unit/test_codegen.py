"""Tests for M3 codegen: ProblemSpec + MethodSpec -> runnable config."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch

from openpinn.codegen import (
    TractionBC,
    build_problem,
    build_solve_kwargs,
    from_specs,
)
from openpinn.spec import load_problem_spec, load_method_spec

SPEC_DIR = Path(__file__).resolve().parents[2] / "examples" / "specs"


def _valid_problem_dict():
    return {
        "dim": 2,
        "physics": "linear_elasticity",
        "geometry": {
            "outer": {
                "type": "rectangle",
                "x_min": 0.0, "x_max": 2.0,
                "y_min": 0.0, "y_max": 2.0,
            },
            "holes": [
                {"type": "circle", "center": [1.0, 1.0], "radius": 0.25},
            ],
        },
        "field": {"kind": "displacement", "components": ["ux", "uy"]},
        "constitutive": {
            "model": "isotropic_linear", "E": 210e9, "nu": 0.3,
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
    }


def _valid_method_dict():
    return {
        "method": "baseline",
        "variant": "plain_pinn",
        "path": "pinn",
        "sampling": {"name": "uniform", "interior": 64, "boundary": 32},
        "loss": {"formulation": "strong"},
        "optimizer": {"name": "adam", "lr": 1e-3},
        "network": {"name": "mlp", "hidden_layers": [32, 32], "activation": "tanh"},
        "differentiator": {"mode": "autodiff"},
        "training": {"steps": 5},
    }


# ============================================================================
# Positive: valid YAML -> codegen succeeds
# ============================================================================

class TestCodegenValid:

    def test_from_dicts(self):
        problem, kwargs = from_specs(_valid_problem_dict(), _valid_method_dict())
        assert problem.name == "plate_with_hole"
        assert problem.output_dim == 2
        assert problem.domain.dim == 2
        assert kwargs["method"] == "baseline"
        assert kwargs["steps"] == 5

    def test_from_yaml_files(self):
        if not SPEC_DIR.exists():
            pytest.skip("Example specs not found")
        problem, kwargs = from_specs(
            str(SPEC_DIR / "plate_with_hole_problem.yaml"),
            str(SPEC_DIR / "pinn_method.yaml"),
            steps=3,
        )
        assert problem.output_dim == 2
        assert kwargs["steps"] == 3

    def test_build_problem_conditions(self):
        spec = load_problem_spec(_valid_problem_dict())
        problem = build_problem(spec)
        names = [c.name for c in problem.conditions]
        assert "dirichlet_left" in names
        assert any("traction_" in n for n in names)

    def test_build_problem_field_names(self):
        spec = load_problem_spec(_valid_problem_dict())
        problem = build_problem(spec)
        assert problem.field_names == ("ux", "uy")

    def test_residual_evaluates(self):
        spec = load_problem_spec(_valid_problem_dict())
        problem = build_problem(spec)
        pts = problem.domain.sample_interior(16, dtype=torch.float32)
        pts = pts.requires_grad_(True)
        from openpinn import MLP, PINNModel
        model = PINNModel(MLP(2, 2, hidden_layers=[16, 16]))
        pred = model(pts)
        res = problem.evaluate_residual(pts, pred)
        assert "momentum_x" in res
        assert "momentum_y" in res
        assert all(torch.isfinite(v).all() for v in res.values())

    def test_conditions_evaluate(self):
        spec = load_problem_spec(_valid_problem_dict())
        problem = build_problem(spec)
        from openpinn import MLP, PINNModel
        model = PINNModel(MLP(2, 2, hidden_layers=[16, 16]))
        bpts = problem.domain.sample_boundary(32, dtype=torch.float32)
        bpts = bpts.requires_grad_(True)
        for c in problem.conditions:
            err = c.error(model, bpts)
            assert torch.isfinite(err).all(), f"Condition {c.name} produced non-finite error"

    def test_loss_weights_mapped_correctly(self):
        prob_spec = load_problem_spec(_valid_problem_dict())
        meth_dict = _valid_method_dict()
        meth_dict["loss"]["weights"] = {"pde": 1.0, "boundary": 10.0}
        meth_spec = load_method_spec(meth_dict)
        kwargs, problem = build_solve_kwargs(prob_spec, meth_spec, steps=3)
        weights = kwargs["overrides"]["loss"]["weights"]
        assert "momentum_x" in weights
        assert "momentum_y" in weights
        assert weights["momentum_x"] == 1.0
        for cn in [c.name for c in problem.conditions]:
            assert cn in weights
            assert weights[cn] == 10.0

    def test_steps_override(self):
        problem, kwargs = from_specs(
            _valid_problem_dict(), _valid_method_dict(), steps=42
        )
        assert kwargs["steps"] == 42


# ============================================================================
# Negative: invalid specs fail before codegen
# ============================================================================

class TestCodegenNegative:

    def test_missing_dim_rejected(self):
        d = _valid_problem_dict()
        del d["dim"]
        with pytest.raises(Exception, match="dim"):
            from_specs(d, _valid_method_dict())

    def test_missing_physics_rejected(self):
        d = _valid_problem_dict()
        del d["physics"]
        with pytest.raises(Exception, match="physics"):
            from_specs(d, _valid_method_dict())

    def test_unknown_physics_rejected(self):
        d = _valid_problem_dict()
        d["physics"] = "navier_stokes"
        with pytest.raises(Exception, match="physics"):
            from_specs(d, _valid_method_dict())

    def test_negative_E_rejected(self):
        d = _valid_problem_dict()
        d["constitutive"]["E"] = -100.0
        with pytest.raises(Exception, match="E"):
            from_specs(d, _valid_method_dict())

    def test_nu_at_half_rejected(self):
        d = _valid_problem_dict()
        d["constitutive"]["nu"] = 0.5
        with pytest.raises(Exception, match="nu"):
            from_specs(d, _valid_method_dict())

    def test_missing_boundary_rejected(self):
        d = _valid_problem_dict()
        d["boundaries"] = [b for b in d["boundaries"] if b["region"] != "top"]
        with pytest.raises(Exception, match="(?i)missing.*top"):
            from_specs(d, _valid_method_dict())

    def test_unknown_method_rejected(self):
        m = _valid_method_dict()
        m["method"] = "magic_solver"
        with pytest.raises(Exception, match="(?i)unknown.*method"):
            from_specs(_valid_problem_dict(), m)

    def test_conflicting_bc_rejected(self):
        d = _valid_problem_dict()
        d["boundaries"].append(
            {"id": "extra", "region": "left", "type": "neumann",
             "dofs": ["ux"], "value": 0.0}
        )
        with pytest.raises(Exception, match="(?i)conflict"):
            from_specs(d, _valid_method_dict())

    def test_bc_unknown_dof_rejected(self):
        d = _valid_problem_dict()
        d["boundaries"][0]["dofs"] = ["ux", "uz"]
        with pytest.raises(Exception, match="(?i)unknown dofs"):
            from_specs(d, _valid_method_dict())

    def test_hole_outside_domain_rejected(self):
        d = _valid_problem_dict()
        d["geometry"]["holes"][0]["center"] = [1.9, 1.0]
        with pytest.raises(Exception, match="(?i)hole.*inside"):
            from_specs(d, _valid_method_dict())
