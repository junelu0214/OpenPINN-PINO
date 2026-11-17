from __future__ import annotations

import json
from types import SimpleNamespace

import openpinn as op


def test_writer_records_real_result_without_accuracy_claim(tmp_path) -> None:
    result = SimpleNamespace(
        history=[{"step": 1, "total": 4.0}, {"step": 2, "total": 2.5}],
        metrics={"rmse": 0.3, "problem": "test"},
    )
    script = tmp_path / "edited.py"
    script.write_text("# user edit\n", encoding="utf-8")
    path = op.write_diagnostic_run_artifact(
        tmp_path,
        result=result,
        status="budget_exhausted",
        exported_pair_snapshot={"problem": {"name": "test"}, "method": {"name": "pinn"}},
        script_path=script,
        code_origin="user-modified",
        source="user-script",
        budgets={"max_steps": 2, "max_wall_seconds": None},
        steps_completed=2,
        device="cpu",
    )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["kind"] == "RunArtifact"
    assert artifact["status"] == "budget_exhausted"
    assert artifact["metrics"]["loss_history"] == [4.0, 2.5]
    assert artifact["metrics"]["final_loss"] == 2.5
    assert artifact["script_hash_status"] == "available"
    assert artifact["code_origin"] == "user-modified"
    assert artifact["accuracy_threshold"] is None
    assert artifact["accuracy_verified"] is False


def test_writer_accepts_failed_run_without_loss(tmp_path) -> None:
    error = RuntimeError("training failed before the first loss")
    path = op.write_diagnostic_run_artifact(
        tmp_path,
        status="failed",
        failure=error,
        exported_pair_snapshot={"problem": {}, "method": {}},
        source="user-script",
    )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["status"] == "failed"
    assert artifact["metrics"]["loss_history"] == []
    assert artifact["metrics"]["final_loss"] is None
    assert artifact["failure"]["type"] == "RuntimeError"
    assert "first loss" in artifact["failure"]["message"]
