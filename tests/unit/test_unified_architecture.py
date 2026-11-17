"""Contracts for the paper-aligned model/differentiation architecture."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import openpinn as op
from openpinn.spec import load_method_spec


def test_shared_registry_contains_pointwise_and_operator_models() -> None:
    names = {item.name for item in op.default_capability_registry().models()}
    assert {"mlp", "deeponet", "fno"} <= names


def test_execution_plan_selects_model_native_derivative() -> None:
    assert op.resolve_execution_plan("mlp").as_dict() == {
        "backend": "pinn",
        "model": "mlp",
        "evaluation_kind": "pointwise",
        "training_mode": "physics_only",
        "differentiation": "autodiff",
    }
    fno = op.resolve_execution_plan(
        "fno", backend="operator", differentiation="grid"
    )
    assert fno.evaluation_kind == "grid_operator"
    assert fno.differentiation == "spectral"


def test_execution_plan_rejects_invalid_cross_product() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        op.resolve_execution_plan(
            "deeponet", backend="operator", differentiation="spectral"
        )
    with pytest.raises(ValueError, match="requires backend"):
        op.resolve_execution_plan("fno", backend="pinn", differentiation="spectral")


def test_method_spec_accepts_paper_axis_names_and_keeps_legacy_properties() -> None:
    spec = load_method_spec(
        {
            "backend": "operator",
            "model": {"name": "fno", "options": {"modes": 8}},
            "sampling": {"name": "uniform", "interior": 16, "boundary": 4},
            "loss": {"formulation": "strong"},
            "optimization": {"name": "adam", "lr": 1e-3},
            "differentiation": {"name": "spectral"},
            "training": {"steps": 2, "mode": "physics_only"},
        }
    )
    assert spec.path == "operator"
    assert spec.network.name == "fno"
    assert spec.method == "fno"
    assert spec.canonical_method()["model"]["name"] == "fno"
    assert spec.execution_plan().differentiation == "spectral"


def test_data_only_plan_does_not_require_derivatives() -> None:
    plan = op.resolve_execution_plan(
        "deeponet", backend="operator", training_mode="data_only"
    )
    assert plan.differentiation is None


def test_operator_result_has_backend_neutral_view() -> None:
    raw = SimpleNamespace(
        operator="fno",
        steps=2,
        initial_loss=2.0,
        final_loss=1.0,
        history=(1.5, 1.0),
        metrics={"physics_loss": 0.5},
    )
    result = op.as_unified_result(raw)
    assert result.backend == "operator"
    assert result.model == "fno"
    assert result.status == "completed"
    assert result.history[-1]["total"] == 1.0
