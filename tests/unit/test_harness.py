from __future__ import annotations

import json
from pathlib import Path

import yaml

from openpinn import cli as openpinn_cli
from openpinn.harness import (
    check_task,
    initialize_task,
    inspect_run,
    run_task,
    workflow_reference,
)
from openpinn.harness.cli import main as harness_main


def _write_runner(task_root: Path, body: str) -> None:
    (task_root / "solution" / "run.py").write_text(body, encoding="utf-8")


def _result_writer(extra: str = "") -> str:
    return f'''from pathlib import Path
import json, os
output = Path(os.environ["OPENPINN_OUTPUT_DIR"])
output.mkdir(parents=True, exist_ok=True)
{extra}
(output / "run_result.json").write_text(json.dumps({{
  "schema_version": 1,
  "status": "completed",
  "summary": "agent solved an unregistered problem",
  "metrics": {{"custom_residual": 0.125}},
  "claims": {{"accuracy_verified": False}},
  "artifacts": ["field.json"]
}}), encoding="utf-8")
(output / "field.json").write_text("[]", encoding="utf-8")
'''


def test_workflow_has_no_registered_capability_gate() -> None:
    result = workflow_reference()
    assert result["ok"] is True
    assert result["data"]["mode"] == "agent_workspace"
    assert result["data"]["capability_gate"] is False
    assert "builtin_problems" not in result["data"]
    assert "methods" not in result["data"]
    assert result["data"]["contract"]["properties"]["mode"]


def test_init_creates_open_workspace_and_check_is_process_only(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = initialize_task("tasks/arbitrary_multiphysics")
    assert result["ok"] is True
    root = tmp_path / "tasks" / "arbitrary_multiphysics"
    assert (root / "problem.md").is_file()
    assert (root / "plan.md").is_file()
    assert yaml.safe_load((root / "task.yaml").read_text(encoding="utf-8"))["mode"] == "agent_workspace"

    checked = check_task(root)
    assert checked["ok"] is True
    assert checked["data"]["physics_capability_checked"] is False
    assert not any("unsupported" in item["message"].lower() for item in checked["warnings"])


def test_missing_workflow_documents_are_advisory_but_syntax_is_blocking(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "task"
    initialize_task(root)
    (root / "problem.md").unlink()
    checked = check_task(root)
    assert checked["ok"] is True
    assert any(item["code"] == "workflow.document_missing" for item in checked["warnings"])

    _write_runner(root, "def broken(:\n")
    checked = check_task(root)
    assert checked["ok"] is False
    assert checked["errors"][0]["code"] == "runner.syntax_error"


def test_agent_authored_unregistered_problem_runs_and_keeps_dual_evidence(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "task"
    initialize_task(root)
    _write_runner(root, _result_writer())

    result = run_task(root)
    assert result["ok"] is True, result
    assert result["status"] == "completed"
    artifact_path = Path(result["files"][0])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["kind"] == "OpenPINNAgentRun"
    assert artifact["code_origin"] == "agent_authored"
    assert artifact["harness_observations"]["physics_capability_checked"] is False
    assert artifact["harness_observations"]["exit_code"] == 0
    assert artifact["harness_observations"]["declared_files"] == [
        {"path": "field.json", "exists": True, "inside_output": True}
    ]
    assert artifact["agent_claims"] == {"accuracy_verified": False}
    assert artifact["source_hash"].startswith("sha256:")

    inspected = inspect_run(root)
    assert inspected["ok"] is True
    assert inspected["data"]["agent_result"]["metrics"]["custom_residual"] == 0.125
    assert inspected["data"]["harness_observations"]["result_values_finite"] is True
    assert any(item["code"] == "evidence.agent_claims_unverified" for item in inspected["warnings"])


def test_timeout_terminates_agent_runner_and_records_observation(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "task"
    initialize_task(root)
    task = yaml.safe_load((root / "task.yaml").read_text(encoding="utf-8"))
    task["execution"]["max_wall_seconds"] = 0.05
    (root / "task.yaml").write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")
    _write_runner(root, "import time\ntime.sleep(60)\n")

    result = run_task(root)
    assert result["ok"] is False
    assert result["status"] == "interrupted"
    artifact = json.loads(Path(result["files"][0]).read_text(encoding="utf-8"))
    assert artifact["harness_observations"]["timed_out"] is True
    assert artifact["harness_observations"]["result_file_present"] is False


def test_nonfinite_agent_result_is_preserved_as_invalid_evidence(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "task"
    initialize_task(root)
    _write_runner(
        root,
        '''from pathlib import Path
import json, os
output = Path(os.environ["OPENPINN_OUTPUT_DIR"]); output.mkdir(parents=True, exist_ok=True)
(output / "run_result.json").write_text(json.dumps({"status":"completed","metrics":{"loss":float("nan")}}), encoding="utf-8")
''',
    )
    result = run_task(root)
    assert result["ok"] is False
    assert result["status"] == "invalid_result"
    artifact = json.loads(Path(result["files"][0]).read_text(encoding="utf-8"))
    assert artifact["harness_observations"]["result_json_valid"] is True
    assert artifact["harness_observations"]["result_values_finite"] is False
    assert artifact["agent_result"]["invalid_result"]


def test_paths_cannot_escape_workspace(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "task"
    initialize_task(root)
    task = yaml.safe_load((root / "task.yaml").read_text(encoding="utf-8"))
    task["runner"]["entrypoint"] = "../outside.py"
    (root / "task.yaml").write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")
    checked = check_task(root)
    assert checked["ok"] is False
    assert checked["errors"][0]["code"] == "harness.schema_validation"


def test_cli_emits_json_for_workflow_and_init(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert harness_main(["workflow"]) == 0
    assert json.loads(capsys.readouterr().out)["stage"] == "workflow"

    assert openpinn_cli.main(["harness", "init", "tasks/case"]) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["stage"] == "init"
    assert Path(created["data"]["task_root"]).is_dir()

    assert harness_main(["init", "tasks/case"]) == 2
    rejected = json.loads(capsys.readouterr().out)
    assert rejected["errors"][0]["code"] == "harness.task_exists"
