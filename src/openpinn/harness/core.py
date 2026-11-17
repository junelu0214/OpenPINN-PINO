"""Workflow, execution, and evidence services for agent-authored OpenPINN tasks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import ValidationError

from .contracts import HARNESS_SCHEMA_VERSION, HarnessTask


_SUCCESS_STATUSES = {"completed", "budget_exhausted", "not_converged"}
_AGENT_STATUSES = _SUCCESS_STATUSES | {"failed", "interrupted", "unknown"}
_SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".md", ".toml"}
_SECRET_PATTERN = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{16,}|api[_-]?key\s*[=:]\s*['\"][^'\"]{8,})",
    re.IGNORECASE,
)


def _result(
    *,
    ok: bool,
    stage: str,
    status: str,
    summary: str,
    errors: list[dict[str, Any]] | None = None,
    warnings: list[dict[str, Any]] | None = None,
    files: list[str] | None = None,
    next_actions: list[dict[str, Any]] | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": HARNESS_SCHEMA_VERSION,
        "ok": ok,
        "stage": stage,
        "status": status,
        "summary": summary,
        "errors": errors or [],
        "warnings": warnings or [],
        "files": files or [],
        "next_actions": next_actions or [],
        "data": data or {},
    }


def _error(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message}


def _advisory(code: str, path: str, message: str) -> dict[str, Any]:
    return {"code": code, "path": path, "message": message, "blocking": False}


def _inside(root: Path, relative: str, label: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"{label} resolves outside the task workspace")
    return candidate


def _workspace_path(source: str | Path) -> tuple[Path, Path]:
    path = Path(source)
    task_file = path / "task.yaml" if path.is_dir() else path
    task_file = task_file.resolve()
    task_root = task_file.parent
    current = Path.cwd().resolve()
    if task_root != current and current not in task_root.parents:
        raise ValueError("task workspace must stay inside the current working directory")
    return task_file, task_root


def load_task(source: str | Path | Mapping[str, Any]) -> tuple[HarnessTask, Path]:
    """Load one workspace task and return its resolved root."""

    if isinstance(source, Mapping):
        values = dict(source)
        root = Path.cwd().resolve()
    else:
        task_file, root = _workspace_path(source)
        values = yaml.safe_load(task_file.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("task.yaml must contain one YAML object")
    return HarnessTask.model_validate(values), root


def workflow_reference() -> dict[str, Any]:
    """Describe the OpenPINN-inspired process, not a capability registry."""

    return _result(
        ok=True,
        stage="workflow",
        status="complete",
        summary="The Harness manages workflow and evidence; the general-purpose agent owns the scientific solution.",
        data={
            "contains_llm": False,
            "capability_gate": False,
            "mode": "agent_workspace",
            "stages": [
                "problem_understanding",
                "method_planning",
                "code_construction",
                "smoke_execution",
                "bounded_execution",
                "evidence_inspection",
                "diagnosis_and_revision",
            ],
            "planning_axes": [
                "physics_and_units",
                "geometry_and_conditions",
                "representation",
                "sampling",
                "loss",
                "optimization",
                "differentiation",
                "verification",
            ],
            "contract": HarnessTask.model_json_schema(),
            "boundaries": [
                "The Harness does not decide whether a physical problem or method is supported.",
                "Agent-authored Python is executed as a bounded subprocess without a shell.",
                "Agent claims and Harness observations remain separate in the evidence artifact.",
                "The subprocess is not an operating-system security sandbox and inherits the current environment.",
            ],
        },
    )


def task_template(task_id: str) -> dict[str, Any]:
    """Return the single open-workspace task contract."""

    return {
        "schema_version": 1,
        "mode": "agent_workspace",
        "task": {
            "id": task_id,
            "title": "Replace with the user's physical problem",
            "description_file": "problem.md",
            "plan_file": "plan.md",
        },
        "runner": {"entrypoint": "solution/run.py", "args": []},
        "execution": {
            "device": "cpu",
            "max_wall_seconds": 300,
            "max_output_bytes": 1_073_741_824,
            "output_dir": "outputs/run_001",
            "budget": {"phase": "smoke"},
        },
        "evidence": {"result_file": "run_result.json"},
    }


def initialize_task(target: str | Path) -> dict[str, Any]:
    """Create a workflow workspace without selecting a physical capability."""

    root = Path(target)
    if root.is_absolute():
        resolved = root.resolve()
    else:
        resolved = (Path.cwd() / root).resolve()
    current = Path.cwd().resolve()
    if resolved != current and current not in resolved.parents:
        return _result(
            ok=False,
            stage="init",
            status="rejected",
            summary="Task initialization was rejected.",
            errors=[_error("harness.path_outside_workspace", "target", "target must stay inside the current working directory")],
        )
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        return _result(
            ok=False,
            stage="init",
            status="rejected",
            summary="Refusing to overwrite a non-empty task directory.",
            errors=[_error("harness.task_exists", "target", str(resolved))],
        )
    resolved.mkdir(parents=True, exist_ok=True)
    (resolved / "solution").mkdir()
    (resolved / "tests").mkdir()
    (resolved / "references").mkdir()
    task_id = resolved.name or "openpinn_task"
    (resolved / "task.yaml").write_text(
        yaml.safe_dump(task_template(task_id), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (resolved / "problem.md").write_text(
        "# Physical problem\n\nDescribe fields, equations, geometry, initial/boundary conditions, parameters, units, desired outputs, and unresolved user choices.\n",
        encoding="utf-8",
    )
    (resolved / "plan.md").write_text(
        "# Computational plan\n\nRecord representation, sampling, losses, optimization, differentiation, smoke budget, reference evidence, metrics, and known limitations.\n",
        encoding="utf-8",
    )
    (resolved / "solution" / "run.py").write_text(
        "\"\"\"Agent-owned entrypoint. Replace this skeleton with the scientific solution.\"\"\"\n\nraise NotImplementedError(\"The general-purpose agent must implement this task\")\n",
        encoding="utf-8",
    )
    files = [
        resolved / "task.yaml",
        resolved / "problem.md",
        resolved / "plan.md",
        resolved / "solution" / "run.py",
    ]
    return _result(
        ok=True,
        stage="init",
        status="complete",
        summary="An open agent workspace was created without selecting or restricting the physics.",
        files=[str(path) for path in files],
        next_actions=[{"action": "agent_implement", "path": str(resolved), "reason": "Understand the problem, write the plan, implement the runner, and then check the workspace."}],
        data={"task_root": str(resolved)},
    )


def _schema_errors(exc: ValidationError) -> list[dict[str, str]]:
    return [
        _error(
            "harness.schema_validation",
            ".".join(str(part) for part in item["loc"]),
            item["msg"],
        )
        for item in exc.errors()
    ]


def check_task(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Check workflow structure and executable syntax, never physical capability."""

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    try:
        task, root = load_task(source)
        entrypoint = _inside(root, task.runner.entrypoint, "runner.entrypoint")
        output_dir = _inside(root, task.execution.output_dir, "execution.output_dir")
        result_file = _inside(output_dir, task.evidence.result_file, "evidence.result_file")
        if not entrypoint.is_file():
            errors.append(_error("runner.entrypoint_missing", "runner.entrypoint", str(entrypoint)))
        else:
            source_text = entrypoint.read_text(encoding="utf-8")
            try:
                compile(source_text, str(entrypoint), "exec")
            except SyntaxError as exc:
                errors.append(_error("runner.syntax_error", "runner.entrypoint", f"line {exc.lineno}: {exc.msg}"))
            if _SECRET_PATTERN.search(source_text):
                warnings.append(_advisory("security.possible_secret", "runner.entrypoint", "The entrypoint appears to contain a credential; use the process environment instead."))
        for field, relative in (
            ("task.description_file", task.task.description_file),
            ("task.plan_file", task.task.plan_file),
        ):
            document = _inside(root, relative, field)
            if not document.is_file():
                warnings.append(_advisory("workflow.document_missing", field, f"Recommended workflow document is missing: {relative}"))
            elif not document.read_text(encoding="utf-8").strip():
                warnings.append(_advisory("workflow.document_empty", field, f"Recommended workflow document is empty: {relative}"))
        plan = _inside(root, task.task.plan_file, "task.plan_file")
        if plan.is_file():
            text = plan.read_text(encoding="utf-8").lower()
            if not any(term in text for term in ("verif", "valid", "reference", "验证", "参考", "制造解")):
                warnings.append(_advisory("workflow.verification_unplanned", "task.plan_file", "No verification or independent reference plan was found; execution may continue, but accuracy cannot be established."))
        if output_dir.exists() and not output_dir.is_dir():
            errors.append(_error("harness.output_not_directory", "execution.output_dir", str(output_dir)))
        return _result(
            ok=not errors,
            stage="check",
            status="ready" if not errors else "blocked",
            summary="Workspace checks completed without judging the physical problem or method.",
            errors=errors,
            warnings=warnings,
            next_actions=[{"action": "run", "path": str(root), "reason": "Execute the agent-authored entrypoint with the declared budget."}] if not errors else [],
            data={
                "task_root": str(root),
                "entrypoint": str(entrypoint),
                "output_dir": str(output_dir),
                "result_file": str(result_file),
                "physics_capability_checked": False,
            },
        )
    except ValidationError as exc:
        return _result(ok=False, stage="check", status="blocked", summary="task.yaml schema validation failed.", errors=_schema_errors(exc))
    except Exception as exc:
        return _result(ok=False, stage="check", status="blocked", summary="Workspace checks failed.", errors=[_error(f"harness.{type(exc).__name__}", "task", str(exc))])


def _hash_sources(root: Path, output_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _SOURCE_SUFFIXES:
            continue
        resolved = path.resolve()
        if resolved == output_dir or output_dir in resolved.parents:
            continue
        if "__pycache__" in path.parts or ".git" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _finite_json(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_json(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite_json(item) for key, item in value.items())
    return False


def _output_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=10)


def _declared_files(agent_result: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    values = agent_result.get("artifacts", [])
    if not isinstance(values, list):
        return [{"path": "<invalid artifacts field>", "exists": False, "inside_output": False}]
    for value in values:
        if not isinstance(value, str):
            records.append({"path": repr(value), "exists": False, "inside_output": False})
            continue
        try:
            candidate = _inside(output_dir, value, "agent artifact")
            records.append({"path": value, "exists": candidate.is_file(), "inside_output": True})
        except ValueError:
            records.append({"path": value, "exists": False, "inside_output": False})
    return records


def run_task(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Execute an agent-authored Python entrypoint without a capability gate."""

    checked = check_task(source)
    if not checked["ok"]:
        return checked
    task, root = load_task(source)
    entrypoint = _inside(root, task.runner.entrypoint, "runner.entrypoint")
    output_dir = _inside(root, task.execution.output_dir, "execution.output_dir")
    result_file = _inside(output_dir, task.evidence.result_file, "evidence.result_file")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        return _result(
            ok=False,
            stage="execution",
            status="rejected",
            summary="Refusing to overwrite an existing run.",
            errors=[_error("harness.output_not_empty", "execution.output_dir", str(output_dir))],
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "harness.log"
    artifact_path = output_dir / "harness_artifact.json"
    source_hash = _hash_sources(root, output_dir)
    task_hash = "sha256:" + hashlib.sha256(
        json.dumps(task.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    run_id = str(uuid.uuid4())
    environment = os.environ.copy()
    environment.update(
        {
            "OPENPINN_TASK_DIR": str(root),
            "OPENPINN_OUTPUT_DIR": str(output_dir),
            "OPENPINN_DEVICE": task.execution.device,
            "OPENPINN_MAX_WALL_SECONDS": str(task.execution.max_wall_seconds),
            "OPENPINN_RUN_ID": run_id,
            "OPENPINN_BUDGET_JSON": json.dumps(task.execution.budget, ensure_ascii=False),
        }
    )
    command = [sys.executable, str(entrypoint), *task.runner.args]
    started = time.monotonic()
    timed_out = False
    exit_code: int | None = None
    launch_error: str | None = None
    with log_path.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=True,
            )
            try:
                exit_code = process.wait(timeout=task.execution.max_wall_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate(process)
                exit_code = process.returncode
        except Exception as exc:
            launch_error = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - started

    agent_result: dict[str, Any] = {}
    result_error: str | None = None
    if result_file.is_file():
        try:
            loaded = json.loads(result_file.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("run_result.json must contain one JSON object")
            agent_result = loaded
        except Exception as exc:
            result_error = str(exc)
    else:
        result_error = "Agent result file was not created"

    result_json_valid = result_file.is_file() and result_error is None
    metrics = agent_result.get("metrics", {})
    result_finite = _finite_json(agent_result)
    declared_files = _declared_files(agent_result, output_dir)
    declared_files_valid = all(
        item.get("inside_output") and item.get("exists") for item in declared_files
    )
    output_bytes = _output_size(output_dir)
    agent_status = agent_result.get("status")
    if launch_error is not None:
        observed_status = "failed"
        result_error = launch_error
    elif timed_out:
        observed_status = "interrupted"
    elif exit_code != 0:
        observed_status = "failed"
    elif result_error is not None:
        observed_status = "incomplete"
    elif not result_finite:
        observed_status = "invalid_result"
        result_error = "Agent result contains non-finite or non-JSON values"
    elif not isinstance(metrics, dict):
        observed_status = "invalid_result"
        result_error = "Agent result metrics must be a JSON object"
    elif not declared_files_valid:
        observed_status = "invalid_result"
        result_error = "One or more declared artifacts are missing or outside the output directory"
    elif output_bytes > task.execution.max_output_bytes:
        observed_status = "resource_exceeded"
        result_error = "Run output exceeded execution.max_output_bytes"
    elif agent_status not in _AGENT_STATUSES:
        observed_status = "invalid_result"
        result_error = f"Unsupported agent status: {agent_status!r}"
    else:
        observed_status = str(agent_status)

    recorded_agent_result = agent_result if result_finite else {
        "invalid_result": "The original result file contains a non-finite or non-JSON value."
    }
    agent_claims = agent_result.get("claims", {})
    if not isinstance(agent_claims, dict) or not _finite_json(agent_claims):
        agent_claims = {}

    artifact = {
        "schema_version": 1,
        "kind": "OpenPINNAgentRun",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": observed_status,
        "code_origin": "agent_authored",
        "task_hash": task_hash,
        "source_hash": source_hash,
        "entrypoint": task.runner.entrypoint,
        "budget": task.execution.budget,
        "agent_result": recorded_agent_result,
        "agent_claims": agent_claims,
        "harness_observations": {
            "physics_capability_checked": False,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "elapsed_seconds": elapsed,
            "device": task.execution.device,
            "result_file_present": result_file.is_file(),
            "result_json_valid": result_json_valid,
            "result_values_finite": result_finite,
            "metrics_object": isinstance(metrics, dict),
            "declared_files_valid": declared_files_valid,
            "output_bytes": output_bytes,
            "declared_files": declared_files,
        },
        "failure": {"message": result_error},
        "files": {
            "log": log_path.name,
            "agent_result": str(result_file.relative_to(output_dir)) if result_file.is_file() else None,
        },
    }
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    ok = observed_status in _SUCCESS_STATUSES
    return _result(
        ok=ok,
        stage="execution",
        status=observed_status,
        summary="The agent-authored runner finished; inspect Harness observations separately from Agent claims.",
        errors=[] if ok else [_error("harness.run_unsuccessful", "execution", result_error or f"runner status is {observed_status}")],
        warnings=checked["warnings"],
        files=[str(artifact_path), str(log_path)] + ([str(result_file)] if result_file.is_file() else []),
        next_actions=[{"action": "inspect", "path": str(artifact_path), "reason": "Review observed evidence before accepting scientific claims."}],
        data={"run_id": run_id, "agent_status": agent_status, "harness_status": observed_status},
    )


def _artifact_path(source: str | Path) -> Path:
    path = Path(source)
    if path.is_file():
        return path
    if path.is_dir() and (path / "harness_artifact.json").is_file():
        return path / "harness_artifact.json"
    task, root = load_task(path)
    return _inside(root, task.execution.output_dir, "execution.output_dir") / "harness_artifact.json"


def inspect_run(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Read dual-track evidence without endorsing the agent's physics claims."""

    try:
        if isinstance(source, Mapping):
            artifact = dict(source)
            path: Path | None = None
        else:
            path = _artifact_path(source)
            artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact.get("kind") != "OpenPINNAgentRun" or artifact.get("schema_version") != 1:
            raise ValueError("Expected a schema_version 1 OpenPINNAgentRun artifact")
        status = str(artifact.get("status", "unknown"))
        claims = artifact.get("agent_claims", {})
        observations = artifact.get("harness_observations", {})
        warnings: list[dict[str, Any]] = []
        if claims:
            warnings.append(_advisory("evidence.agent_claims_unverified", "agent_claims", "Claims were authored by the general-purpose agent; the Harness records but does not independently prove them."))
        if not isinstance(claims, dict) or not claims.get("accuracy_verified", False):
            warnings.append(_advisory("evidence.accuracy_unverified", "agent_claims.accuracy_verified", "No agent-reported accuracy verification is present."))
        if not observations.get("result_values_finite", False):
            warnings.append(_advisory("evidence.nonfinite_or_invalid", "harness_observations.result_values_finite", "The result is invalid or contains non-finite values."))
        return _result(
            ok=status in _SUCCESS_STATUSES,
            stage="inspection",
            status=status,
            summary="Agent claims and Harness-observed execution evidence were returned separately.",
            warnings=warnings,
            files=[str(path)] if path else [],
            data={
                "run_id": artifact.get("run_id"),
                "agent_result": artifact.get("agent_result", {}),
                "agent_claims": claims,
                "harness_observations": observations,
                "source_hash": artifact.get("source_hash"),
                "task_hash": artifact.get("task_hash"),
                "failure": artifact.get("failure", {}),
            },
        )
    except Exception as exc:
        return _result(
            ok=False,
            stage="inspection",
            status="rejected",
            summary="Harness artifact inspection failed.",
            errors=[_error(f"harness.{type(exc).__name__}", "artifact", str(exc))],
        )
