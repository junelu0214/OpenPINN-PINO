from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import openpinn as op
from tests._assertions import assert_finite_tree


REQUIRED_FILES = {
    "config.yaml",
    "status.json",
    "metrics.json",
    "history.csv",
    "predictions.npz",
    "compute_ledger.json",
    "checkpoint.pt",
    "manifest.json",
}


def _assert_contract(result: Any) -> None:
    output_dir = Path(result.output_dir)
    present = {path.name for path in output_dir.iterdir() if path.is_file()}
    assert REQUIRED_FILES <= present
    for filename in REQUIRED_FILES:
        assert (output_dir / filename).stat().st_size > 0

    persisted_config = op.load_config(output_dir / "config.yaml")
    assert persisted_config["schema_version"] == 1

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert isinstance(metrics, dict) and metrics
    assert_finite_tree(metrics)
    assert result["metrics"] == result.metrics

    status = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["finite_outputs"] is True
    assert status["accuracy_claim"] is False

    with (output_dir / "history.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames is not None
        assert {"step", "total"} <= set(reader.fieldnames)
    assert rows
    for row in rows:
        for value in row.values():
            if value not in (None, ""):
                assert math.isfinite(float(value))

    with np.load(output_dir / "predictions.npz") as predictions:
        assert {"coordinates", "prediction"} <= set(predictions.files)
        for name in predictions.files:
            if np.issubdtype(predictions[name].dtype, np.number):
                assert np.isfinite(predictions[name]).all(), name

    ledger = json.loads(
        (output_dir / "compute_ledger.json").read_text(encoding="utf-8")
    )
    assert {
        "device",
        "dtype",
        "steps",
        "wall_time_seconds",
        "trainable_parameters",
    } <= set(ledger)
    assert ledger["device"] == "cpu"
    assert ledger["steps"] > 0
    assert ledger["wall_time_seconds"] >= 0.0
    assert ledger["trainable_parameters"] > 0

    checkpoint = torch.load(
        output_dir / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    assert {"model", "optimizer", "steps_completed", "history"} <= set(checkpoint)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert REQUIRED_FILES.difference({"manifest.json"}) <= set(manifest["files"])
    for metadata in manifest["files"].values():
        assert metadata["bytes"] > 0
        assert len(metadata["sha256"]) == 64


@pytest.mark.parametrize(
    "fixture_name",
    (
        "poisson_smoke_result",
        "heterogeneous_poisson_smoke_result",
        "burgers_smoke_result",
        "coupled_manufactured_smoke_result",
    ),
)
def test_smoke_runs_follow_the_standard_output_contract(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    _assert_contract(request.getfixturevalue(fixture_name))
