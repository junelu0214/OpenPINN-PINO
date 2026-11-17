from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import re

import pytest
import torch

import openpinn as op


def heat_snapshot() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "GenericProblemDescription",
        "physics": "heat_equation",
        "coordinates": [
            {"name": "x", "role": "space", "unit": "m", "bounds": {"min": 0, "max": 1}},
            {"name": "t", "role": "time", "unit": "s", "bounds": {"min": 0, "max": 1}},
        ],
        "fields": [{"name": "u", "kind": "scalar", "components": ["u"], "shape": [1], "unit": "K"}],
        "parameters": [{"name": "alpha", "value": 0.01, "unit": "m^2/s"}],
        "equations": [{"id": "heat", "form": "expression", "expression": "dt(u)-alpha*dx(dx(u))=0"}],
        "geometry": {"type": "interval", "x_min": 0, "x_max": 1},
        "boundaries": [
            {"id": "left", "location": "x=0", "type": "dirichlet", "value": 0, "unit": "K"},
            {"id": "right", "location": "x=1", "type": "neumann", "value": 0, "unit": "K/m"},
        ],
        "initial_conditions": [{"id": "initial", "field": "u", "expression": "sin(pi*x/2)", "unit": "K"}],
        "trainable_contract": {"mapping_complete": True},
    }


def analytic_heat(coordinates: torch.Tensor) -> torch.Tensor:
    x = coordinates[:, :1]
    t = coordinates[:, 1:2]
    alpha = 0.01
    return torch.sin(math.pi * x / 2) * torch.exp(-alpha * math.pi**2 * t / 4)


def test_heat_snapshot_compiles_named_finite_residuals() -> None:
    compiled = op.compile_problem_expressions(heat_snapshot())
    coordinates = torch.tensor(
        [[0.2, 0.1], [0.5, 0.4], [0.8, 0.9]],
        dtype=torch.float64,
        requires_grad=True,
    )

    residual = compiled.equations["heat"](coordinates, analytic_heat)

    assert compiled.parameters == {"alpha": 0.01}
    assert compiled.coordinate_names == ("x", "t")
    assert compiled.field_names == ("u",)
    assert residual.shape == (3, 1)
    assert torch.isfinite(residual).all()


def test_heat_boundary_and_initial_residuals_are_compiled() -> None:
    compiled = op.compile_problem_expressions(heat_snapshot())
    left = torch.tensor([[0.0, 0.2], [0.0, 0.8]], dtype=torch.float64, requires_grad=True)
    right = torch.tensor([[1.0, 0.2], [1.0, 0.8]], dtype=torch.float64, requires_grad=True)
    initial = torch.tensor([[0.2, 0.0], [0.8, 0.0]], dtype=torch.float64, requires_grad=True)

    assert compiled.boundaries["left"](left, analytic_heat).shape == (2, 1)
    assert compiled.boundaries["right"](right, analytic_heat).shape == (2, 1)
    assert compiled.initial_conditions["initial"](initial, analytic_heat).shape == (2, 1)
    torch.testing.assert_close(compiled.boundaries["left"](left, analytic_heat), torch.zeros((2, 1), dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(compiled.boundaries["right"](right, analytic_heat), torch.zeros((2, 1), dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(compiled.initial_conditions["initial"](initial, analytic_heat), torch.zeros((2, 1), dtype=torch.float64), atol=1e-12, rtol=0)


def test_analytic_heat_pde_and_conditions_are_near_zero() -> None:
    compiled = op.compile_problem_expressions(heat_snapshot())
    interior = torch.tensor(
        [[0.1, 0.2], [0.4, 0.5], [0.9, 0.7]], dtype=torch.float64, requires_grad=True
    )
    left = torch.tensor([[0.0, 0.3]], dtype=torch.float64, requires_grad=True)
    right = torch.tensor([[1.0, 0.3]], dtype=torch.float64, requires_grad=True)
    initial = torch.tensor([[0.3, 0.0]], dtype=torch.float64, requires_grad=True)
    residuals = [
        compiled.equations["heat"](interior, analytic_heat),
        compiled.boundaries["left"](left, analytic_heat),
        compiled.boundaries["right"](right, analytic_heat),
        compiled.initial_conditions["initial"](initial, analytic_heat),
    ]

    maximum = max(float(value.detach().abs().max()) for value in residuals)
    assert maximum < 1e-10


def test_derivatives_match_manual_autodiff_oracle() -> None:
    snapshot = heat_snapshot()
    snapshot["equations"] = [
        {"id": "time", "expression": "dt(u)"},
        {"id": "space", "expression": "dx(u)"},
        {"id": "space2", "expression": "dx(dx(u))"},
    ]
    compiled = op.compile_problem_expressions(snapshot)
    coordinates = torch.tensor(
        [[0.2, -0.3], [0.7, 0.4]], dtype=torch.float64, requires_grad=True
    )
    field = coordinates[:, :1] ** 3 * coordinates[:, 1:2] ** 2 + torch.sin(coordinates[:, :1]) * torch.exp(coordinates[:, 1:2])
    manual_gradient = op.grad(field, coordinates)
    manual_xx = op.hessian(field, coordinates, i=0, j=0)

    torch.testing.assert_close(compiled.equations["time"](coordinates, field), manual_gradient[:, 1:2])
    torch.testing.assert_close(compiled.equations["space"](coordinates, field), manual_gradient[:, :1])
    torch.testing.assert_close(compiled.equations["space2"](coordinates, field), manual_xx)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("laplace(u)", "unregistered function"),
        ("foo(u)", "unregistered function"),
        ("dz(u)", "undeclared coordinate"),
        ("u+intruder", "unbound symbol"),
        ("__import__('os')", "unsafe function"),
    ],
)
def test_unsafe_or_unbound_expressions_fail_closed(source: str, message: str) -> None:
    snapshot = heat_snapshot()
    snapshot["equations"] = [{"id": "unsafe", "expression": source}]

    with pytest.raises(op.ExpressionCompileError, match=message):
        op.compile_problem_expressions(snapshot)


def test_mapping_incomplete_snapshot_fails_closed() -> None:
    snapshot = heat_snapshot()
    snapshot["trainable_contract"] = {"mapping_complete": False}

    with pytest.raises(op.ExpressionCompileError, match="mapping_complete"):
        op.compile_problem_expressions(snapshot)


def test_dy_requires_a_declared_y_coordinate() -> None:
    one_dimensional = heat_snapshot()
    one_dimensional["equations"] = [{"id": "yy", "expression": "dy(dy(u))"}]
    with pytest.raises(op.ExpressionCompileError, match="undeclared coordinate"):
        op.compile_problem_expressions(one_dimensional)

    two_dimensional = deepcopy(one_dimensional)
    two_dimensional["coordinates"] = [
        {"name": "x", "role": "space", "unit": "m", "bounds": {"min": 0, "max": 1}},
        {"name": "y", "role": "space", "unit": "m", "bounds": {"min": 0, "max": 1}},
    ]
    compiled = op.compile_problem_expressions(two_dimensional)
    coordinates = torch.tensor([[0.2, 0.3], [0.7, 0.8]], dtype=torch.float64, requires_grad=True)
    values = coordinates[:, 1:2] ** 3
    torch.testing.assert_close(compiled.equations["yy"](coordinates, values), 6 * coordinates[:, 1:2])


def test_compiler_has_no_dynamic_execution_calls_and_backend_remains_honest() -> None:
    root = Path(__file__).resolve().parents[2]
    compiler = (root / "src/openpinn/expressions.py").read_text(encoding="utf-8")
    codebuilder = (root / "docs/_static/codebuilder_agent.js").read_text(encoding="utf-8")

    assert re.search(r"\beval\s*\(|\bexec\s*\(|new\s+Function", compiler) is None
    assert 'var GENERATOR_ID = "openpinn.generic-training-skeleton.v1"' in codebuilder
    assert 'var RUNNABLE_GENERATOR_ID = "openpinn.shared-pinn-engine.v1"' in codebuilder
    assert "backend_runnable: runnable" in codebuilder
    assert "NotImplementedError" in codebuilder
