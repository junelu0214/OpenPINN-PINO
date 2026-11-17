from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess

import pytest

import openpinn as op


METHOD = {
    "path": "pinn",
    "method": "baseline",
    "variant": "plain_pinn",
    "sampling": {"name": "uniform"},
    "loss": {"formulation": "strong", "weighting": "fixed"},
    "optimizer": {"name": "adam"},
    "network": {"name": "mlp"},
    "differentiator": {"mode": "autodiff"},
    "training": {"steps": 20},
}


def poisson_snapshot() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "GenericProblemDescription",
        "physics": "poisson",
        "coordinates": [
            {"name": "x", "role": "space", "unit": "m", "bounds": {"min": 0, "max": 1}}
        ],
        "fields": [{"name": "u", "components": ["u"], "unit": "1"}],
        "parameters": [],
        "equations": [
            {"id": "poisson", "form": "expression", "expression": "-dx(dx(u))=pi*pi*sin(pi*x)"}
        ],
        "geometry": {"type": "interval", "x_min": 0, "x_max": 1},
        "boundaries": [
            {"id": "left", "location": "x=0", "type": "dirichlet", "value": 0},
            {"id": "right", "location": "x=1", "type": "dirichlet", "value": 0},
        ],
        "initial_conditions": [],
        "trainable_contract": {"mapping_complete": True},
    }


def heat_snapshot() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "GenericProblemDescription",
        "physics": "heat_equation",
        "coordinates": [
            {"name": "x", "role": "space", "unit": "m", "bounds": {"min": 0, "max": 1}},
            {"name": "t", "role": "time", "unit": "s", "bounds": {"min": 0, "max": 1}},
        ],
        "fields": [{"name": "u", "components": ["u"], "unit": "K"}],
        "parameters": [{"name": "alpha", "value": 0.01, "unit": "m^2/s"}],
        "equations": [
            {"id": "heat", "form": "expression", "expression": "dt(u)-alpha*dx(dx(u))=0"}
        ],
        "geometry": {"type": "interval", "x_min": 0, "x_max": 1},
        "boundaries": [
            {"id": "left", "location": "x=0", "type": "dirichlet", "value": 0},
            {"id": "right", "location": "x=1", "type": "neumann", "value": 0},
        ],
        "initial_conditions": [
            {"id": "initial", "field": "u", "expression": "sin(pi*x/2)"}
        ],
        "trainable_contract": {"mapping_complete": True},
    }


@pytest.mark.parametrize(
    ("snapshot_factory", "backend_id"),
    [(poisson_snapshot, "poisson_1d"), (heat_snapshot, "heat_1d_unsteady")],
)
def test_registered_cards_use_shared_engine_and_persist_measured_errors(
    tmp_path: Path, snapshot_factory, backend_id: str
) -> None:
    result, artifact = op.run_registered_pinn(
        snapshot_factory(),
        METHOD,
        output_dir=tmp_path,
        pair_fingerprint=f"test:{backend_id}",
        max_steps=3,
        max_wall_seconds=30.0,
        seed=20261110,
    )

    assert result.status == "budget_exhausted"
    assert result.steps_completed == 3
    assert artifact["backend_id"] == backend_id
    assert artifact["status"] == "budget_exhausted"
    assert artifact["backend_runnable"] is True
    assert artifact["accuracy_threshold"] is None
    assert artifact["accuracy_verified"] is False
    assert artifact["engine"]["kind"] == "PINNEngineRun"
    assert len(artifact["metrics"]["loss_history"]) == 3
    assert all(math.isfinite(value) for value in artifact["metrics"]["loss_history"])
    assert set(artifact["metrics"]["error_metrics"]) == {"max_abs", "rel_l2"}
    assert all(math.isfinite(value) for value in artifact["metrics"]["error_metrics"].values())
    persisted = json.loads((tmp_path / "run_artifact.json").read_text(encoding="utf-8"))
    assert persisted == artifact
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert {"engine_run.json", "run_artifact.json"} <= set(manifest["files"])


def test_registry_matches_structure_and_fails_closed() -> None:
    poisson = op.match_training_backend(poisson_snapshot(), METHOD)
    heat = op.match_training_backend(heat_snapshot(), METHOD)
    assert (poisson.status, poisson.backend_id) == ("supported", "poisson_1d")
    assert (heat.status, heat.backend_id) == ("supported", "heat_1d_unsteady")

    incomplete = heat_snapshot()
    incomplete["trainable_contract"] = {"mapping_complete": False}
    assert op.match_training_backend(incomplete, METHOD).status == "incomplete"

    unsupported = heat_snapshot()
    unsupported["boundaries"][1]["type"] = "periodic"
    match = op.match_training_backend(unsupported, METHOD)
    assert match.status == "unsupported"
    assert match.backend_id is None

    renamed = poisson_snapshot()
    renamed["fields"] = [{"name": "v", "components": ["v"], "unit": "1"}]
    renamed["equations"][0]["expression"] = "-dx(dx(v))=pi*pi*sin(pi*x)"
    renamed_match = op.match_training_backend(renamed, METHOD)
    assert (renamed_match.status, renamed_match.backend_id) == ("supported", "scalar_strong_form_v1")

    time_boundary = heat_snapshot()
    time_boundary["boundaries"][0]["location"] = "t=0"
    assert op.match_training_backend(time_boundary, METHOD).status == "unsupported"


def test_generic_scalar_2d_multiple_residuals_and_robin_use_shared_engine(tmp_path: Path) -> None:
    snapshot = {
        "schema_version": 1, "kind": "GenericProblemDescription", "physics": "scalar_elliptic",
        "coordinates": [
            {"name": "x", "role": "space", "unit": "m", "bounds": {"min": 0, "max": 1}},
            {"name": "y", "role": "space", "unit": "m", "bounds": {"min": 0, "max": 1}},
        ],
        "fields": [{"name": "v", "components": ["v"], "unit": "1"}], "parameters": [],
        "equations": [
            {"id": "pde", "form": "expression", "expression": "dx(dx(v))+dy(dy(v))=0"},
            {"id": "aux", "form": "expression", "expression": "dx(v)-dx(v)=0"},
        ],
        "geometry": {"type": "rectangle", "x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1},
        "boundaries": [
            {"id": "left", "location": "x=0", "type": "dirichlet", "value": 0},
            {"id": "right", "location": "x=1", "type": "robin", "alpha": 1, "beta": 1, "value": 0},
            {"id": "bottom", "location": "y=0", "type": "dirichlet", "value": 0},
            {"id": "top", "location": "y=1", "type": "neumann", "value": 0},
        ],
        "initial_conditions": [], "trainable_contract": {"mapping_complete": True},
    }
    capability = op.match_training_backend(snapshot, METHOD)
    assert (capability.status, capability.backend_id) == ("supported", "scalar_strong_form_v1"), capability.reason
    result, artifact = op.run_registered_pinn(
        snapshot, METHOD, output_dir=tmp_path, pair_fingerprint="generic:2d",
        max_steps=2, max_wall_seconds=30, seed=20261110,
    )
    assert result.status == "budget_exhausted"
    assert result.steps_completed == 2
    assert artifact["backend_id"] == "scalar_strong_form_v1"
    assert artifact["metrics"]["error_metrics"] == {}


def test_generic_boundary_value_expression_alias_compiles() -> None:
    snapshot = poisson_snapshot()
    snapshot["fields"] = [{"name": "w", "components": ["w"], "unit": "1"}]
    snapshot["equations"] = [{"id": "pde", "form": "expression", "expression": "-dx(dx(w))-pi*pi*sin(pi*x)=0"}]
    snapshot["boundaries"] = [
        {"location": "x=0", "type": "dirichlet", "value": 0},
        {"location": "x=1", "type": "robin", "alpha": 1, "beta": 1,
         "expression": "alpha*w+beta*dx(w)=value", "value_expression": "-pi"},
    ]
    capability = op.match_training_backend(snapshot, METHOD)
    assert (capability.status, capability.backend_id) == ("supported", "scalar_strong_form_v1")
    problem, _ = op.build_registered_problem(snapshot, METHOD)
    assert len(problem.conditions) == 2


def test_browser_and_python_capability_registries_have_card_parity() -> None:
    root = Path(__file__).resolve().parents[2]
    incomplete = heat_snapshot()
    incomplete["trainable_contract"] = {"mapping_complete": False}
    drifted = heat_snapshot()
    drifted["initial_conditions"][0]["expression"] = "sin(pi*x)"
    snapshots = [poisson_snapshot(), heat_snapshot(), incomplete, drifted]
    payload = json.dumps({"snapshots": snapshots, "method": METHOD})
    script = """
const fs = require('node:fs');
const api = require('./docs/_static/generic_validation.js');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(input.snapshots.map(problem => api.executionCapability(problem, input.method))));
"""
    completed = subprocess.run(
        ["node", "-e", script], cwd=root, input=payload, text=True,
        capture_output=True, check=True,
    )
    browser = json.loads(completed.stdout)
    python = [op.match_training_backend(snapshot, METHOD).as_dict() for snapshot in snapshots]
    assert [(item["status"], item["backend_id"]) for item in browser] == [
        (item["status"], item["backend_id"]) for item in python
    ]


def test_smoke_is_one_real_step_and_never_completed(tmp_path: Path) -> None:
    result, artifact = op.run_registered_pinn(
        poisson_snapshot(), METHOD, output_dir=tmp_path,
        pair_fingerprint="test:smoke", max_steps=20, max_wall_seconds=30.0, smoke=True,
    )
    assert result.steps_completed == 1
    assert result.status == "budget_exhausted"
    assert artifact["smoke_only"] is True
    assert artifact["status"] != "completed"
