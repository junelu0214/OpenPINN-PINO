"""Tests for Agent A: openpinn.agents.validator.

Covers:
- Valid plate-with-hole specs -> ok
- Negative cases (missing field, BC conflict, unknown method) -> ok=False
- Optional code compliance checks
- Core modules do NOT import openpinn.agents
- CLI exit codes
"""

from __future__ import annotations

import ast
import copy
import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from openpinn.agents.validator import (
    Severity,
    ValidationError,
    ValidationReport,
    validate,
)

SPEC_DIR = Path(__file__).resolve().parents[2] / "examples" / "specs"


# ---------------------------------------------------------------------------
# Helpers: minimal valid dicts
# ---------------------------------------------------------------------------

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
    }


def _valid_method() -> dict:
    return {
        "method": "baseline",
        "variant": "plain_pinn",
        "path": "pinn",
        "sampling": {"name": "uniform", "interior": 512, "boundary": 128},
        "loss": {"formulation": "strong", "weights": {"pde": 1.0, "boundary": 10.0}},
        "optimizer": {"name": "adam", "lr": 0.001},
        "network": {"name": "mlp", "hidden_layers": [64, 64, 64, 64], "activation": "tanh"},
        "differentiator": {"mode": "autodiff"},
        "training": {"steps": 5000},
    }


# ---------------------------------------------------------------------------
# Positive: valid plate-with-hole specs
# ---------------------------------------------------------------------------

class TestValidSpecs:
    def test_valid_dicts(self):
        report = validate(_valid_problem(), _valid_method())
        assert report.ok is True
        assert report.errors == []
        assert report.problem_spec is not None
        assert report.method_spec is not None

    def test_valid_yaml_files(self):
        report = validate(
            str(SPEC_DIR / "plate_with_hole_problem.yaml"),
            str(SPEC_DIR / "pinn_method.yaml"),
        )
        assert report.ok is True
        assert report.errors == []

    def test_problem_only(self):
        report = validate(_valid_problem())
        assert report.ok is True
        assert report.problem_spec is not None
        assert report.method_spec is None

    def test_report_as_dict(self):
        report = validate(_valid_problem(), _valid_method())
        d = report.as_dict()
        assert d["ok"] is True
        assert isinstance(d["errors"], list)
        assert isinstance(d["warnings"], list)

    def test_accepts_preloaded_spec_objects(self):
        from openpinn.spec import load_problem_spec, load_method_spec

        prob = load_problem_spec(_valid_problem())
        meth = load_method_spec(_valid_method())
        report = validate(prob, meth)
        assert report.ok is True


# ---------------------------------------------------------------------------
# Negative: M2 schema failures
# ---------------------------------------------------------------------------

class TestSchemaFailures:
    def test_missing_field(self):
        prob = _valid_problem()
        del prob["field"]
        report = validate(prob, _valid_method())
        assert report.ok is False
        assert len(report.errors) >= 1
        codes = [e.code for e in report.errors]
        assert any("schema" in c for c in codes)

    def test_negative_youngs_modulus(self):
        prob = _valid_problem()
        prob["constitutive"]["E"] = -100.0
        report = validate(prob, _valid_method())
        assert report.ok is False
        assert any("E" in e.path or "constitutive" in e.path for e in report.errors)

    def test_bc_conflict(self):
        """Dirichlet and Neumann on the same dof/region should fail."""
        prob = _valid_problem()
        prob["boundaries"].append({
            "id": "conflict_left",
            "region": "left",
            "type": "neumann",
            "dofs": ["ux"],
            "value": 1.0,
        })
        report = validate(prob, _valid_method())
        assert report.ok is False
        assert any("conflict" in e.message.lower() or "Conflict" in e.message
                    for e in report.errors)

    def test_unknown_method(self):
        meth = _valid_method()
        meth["method"] = "nonexistent_solver_xyz"
        report = validate(_valid_problem(), meth)
        assert report.ok is False
        assert len(report.errors) >= 1


# ---------------------------------------------------------------------------
# Negative: codegen readiness
# ---------------------------------------------------------------------------

class TestCodegenReadiness:
    def test_rejects_grid_differentiator_pinn(self):
        meth = _valid_method()
        meth["differentiator"] = {"mode": "grid"}
        report = validate(_valid_problem(), meth)
        assert report.ok is False
        assert any(e.code == "codegen.unsupported_differentiator" for e in report.errors)

    def test_differentiator_mode_is_plain_str(self):
        from openpinn.spec import load_method_spec

        spec = load_method_spec(_valid_method())
        assert type(spec.differentiator.mode) is str
        report = validate(_valid_problem(), spec)
        assert report.ok is True

    def test_differentiator_mode_enum_still_compares(self):
        from openpinn.agents.validator import _registry_name
        from openpinn.spec import DifferentiatorMode

        assert _registry_name("grid") == "grid"
        assert _registry_name(DifferentiatorMode.grid) == "grid"
        assert _registry_name(DifferentiatorMode.autodiff) == "autodiff"


# ---------------------------------------------------------------------------
# Code compliance checks
# ---------------------------------------------------------------------------

class TestCodeCompliance:
    def test_clean_code_passes(self):
        code = textwrap.dedent("""\
            import torch
            import numpy as np
            from openpinn import MLP

            class MyModel(torch.nn.Module):
                def forward(self, x):
                    return self.net(x)
        """)
        report = validate(_valid_problem(), code=code)
        assert report.ok is True

    def test_forbidden_import_fails(self):
        code = textwrap.dedent("""\
            import requests
            import torch
        """)
        report = validate(_valid_problem(), code=code)
        assert report.ok is False
        assert any(e.code == "code.forbidden_import" for e in report.errors)

    def test_pde_in_forward_fails(self):
        code = textwrap.dedent("""\
            import torch
            from openpinn import grad

            class Net(torch.nn.Module):
                def forward(self, x):
                    out = self.net(x)
                    residual = grad(out, x)
                    return residual
        """)
        report = validate(_valid_problem(), code=code)
        assert report.ok is False
        assert any(e.code == "code.pde_in_forward" for e in report.errors)

    def test_syntax_error_in_code(self):
        code = "def foo(\n"
        report = validate(_valid_problem(), code=code)
        assert report.ok is False
        assert any(e.code == "code.syntax_error" for e in report.errors)

    def test_grad_outside_forward_is_ok(self):
        code = textwrap.dedent("""\
            import torch
            from openpinn import grad

            def residual(coords, pred):
                return grad(pred, coords)

            class Net(torch.nn.Module):
                def forward(self, x):
                    return self.net(x)
        """)
        report = validate(_valid_problem(), code=code)
        code_errors = [e for e in report.errors if e.code.startswith("code.")]
        assert len(code_errors) == 0


# ---------------------------------------------------------------------------
# Boundary consistency (agent-level checks beyond M2)
# ---------------------------------------------------------------------------

class TestBoundaryConsistency:
    def test_duplicate_bc_id(self):
        prob = _valid_problem()
        prob["boundaries"].append({
            "id": "fix_left",
            "region": "hole_0",
            "type": "neumann",
            "dofs": ["ux", "uy"],
            "value": [0.0, 0.0],
        })
        prob["boundaries"] = [
            bc for bc in prob["boundaries"]
            if bc["id"] != "free_hole"
        ] + [prob["boundaries"][-1]]
        report = validate(prob)
        assert any(e.code == "boundary.duplicate_id" for e in report.errors)


# ---------------------------------------------------------------------------
# Core modules must NOT import openpinn.agents
# ---------------------------------------------------------------------------

class TestNoAgentsImportFromCore:
    """Verify that openpinn core/training modules never import openpinn.agents."""

    def test_no_agents_import_in_core(self):
        src_dir = Path(__file__).resolve().parents[2] / "src" / "openpinn"
        core_files = [
            p for p in src_dir.glob("*.py")
            if p.name != "__init__.py"
        ]
        for pyfile in core_files:
            source = pyfile.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(pyfile))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("openpinn.agents"), (
                            f"{pyfile.name} has top-level 'import {alias.name}'"
                        )
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.level == 0:
                        assert not node.module.startswith("openpinn.agents"), (
                            f"{pyfile.name} has top-level "
                            f"'from {node.module} import ...'"
                        )

    def test_no_agents_import_in_init(self):
        """__init__.py re-exports must not pull in agents at import time."""
        init_path = Path(__file__).resolve().parents[2] / "src" / "openpinn" / "__init__.py"
        source = init_path.read_text(encoding="utf-8")
        assert "openpinn.agents" not in source
        assert "from .agents" not in source


# ---------------------------------------------------------------------------
# Pre-codegen gate via from_specs(pre_validate=True)
# ---------------------------------------------------------------------------

class TestCodegenPreGate:
    def test_valid_specs_with_pre_validate(self):
        from openpinn.codegen import from_specs

        problem, kwargs = from_specs(
            _valid_problem(), _valid_method(),
            steps=2, pre_validate=True,
        )
        assert problem is not None
        assert "method" in kwargs

    def test_operator_path_is_rejected_for_plate_codegen(self):
        method = _valid_method()
        method["path"] = "operator"
        method["method"] = "deeponet"
        method.pop("variant", None)
        report = validate(_valid_problem(), method)
        assert report.ok is False
        assert any(item.code == "codegen.operator_path" for item in report.errors)

    def test_invalid_spec_blocked_by_pre_validate(self):
        """M2 schema errors are caught by load_method_spec before the gate,
        but agent-level checks (e.g. duplicate BC ids) are caught by the
        pre_validate gate itself."""
        from openpinn.codegen import from_specs

        prob = _valid_problem()
        prob["boundaries"].append({
            "id": "fix_left",
            "region": "hole_0",
            "type": "neumann",
            "dofs": ["ux", "uy"],
            "value": [0.0, 0.0],
        })
        prob["boundaries"] = [
            bc for bc in prob["boundaries"]
            if not (bc["id"] == "free_hole" and bc["region"] == "hole_0")
        ] + [prob["boundaries"][-1]]
        with pytest.raises(ValueError, match="validation failed"):
            from_specs(
                prob, _valid_method(),
                steps=2, pre_validate=True,
            )


# ---------------------------------------------------------------------------
# CLI exit codes
# ---------------------------------------------------------------------------

class TestCLI:
    def test_valid_specs_exit_zero(self):
        result = subprocess.run(
            [
                sys.executable, "-m", "openpinn.agents",
                str(SPEC_DIR / "plate_with_hole_problem.yaml"),
                str(SPEC_DIR / "pinn_method.yaml"),
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "PASS" in result.stdout

    def test_invalid_problem_exit_nonzero(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("dim: 99\n")
        result = subprocess.run(
            [sys.executable, "-m", "openpinn.agents", str(bad_yaml)],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "FAIL" in result.stdout or "error" in result.stdout.lower()

    def test_json_output(self):
        import json

        result = subprocess.run(
            [
                sys.executable, "-m", "openpinn.agents",
                str(SPEC_DIR / "plate_with_hole_problem.yaml"),
                str(SPEC_DIR / "pinn_method.yaml"),
                "--json",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["ok"] is True
