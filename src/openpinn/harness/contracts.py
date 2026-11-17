"""Stable contracts for the OpenPINN workflow harness."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


HARNESS_SCHEMA_VERSION = 1
MAX_WALL_SECONDS = 86_400.0
MAX_OUTPUT_BYTES = 10 * 1024 * 1024 * 1024


def _relative_path(value: str, label: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{label} must be a non-empty relative path without '..'")
    return path.as_posix()


class TaskDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=300)
    description_file: str = "problem.md"
    plan_file: str = "plan.md"

    @field_validator("description_file", "plan_file")
    @classmethod
    def _safe_document_path(cls, value: str) -> str:
        return _relative_path(value, "task document path")


class RunnerDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entrypoint: str = "solution/run.py"
    args: list[str] = Field(default_factory=list, max_length=64)

    @field_validator("entrypoint")
    @classmethod
    def _safe_entrypoint(cls, value: str) -> str:
        path = _relative_path(value, "runner.entrypoint")
        if Path(path).suffix != ".py":
            raise ValueError("runner.entrypoint must be a Python file")
        return path

    @field_validator("args")
    @classmethod
    def _safe_args(cls, value: list[str]) -> list[str]:
        for item in value:
            if "\x00" in item or len(item) > 4096:
                raise ValueError("runner arguments must be bounded strings without NUL")
        return value


class ExecutionDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: str = "cpu"
    max_wall_seconds: float = Field(default=300.0, gt=0.0, le=MAX_WALL_SECONDS)
    max_output_bytes: int = Field(default=1_073_741_824, ge=1, le=MAX_OUTPUT_BYTES)
    output_dir: str = "outputs/run"
    budget: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @field_validator("device")
    @classmethod
    def _safe_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized in {"cpu", "mps", "cuda"}:
            return normalized
        if normalized.startswith("cuda:") and normalized[5:].isdigit():
            return normalized
        raise ValueError("device must be cpu, mps, cuda, or cuda:<index>")

    @field_validator("output_dir")
    @classmethod
    def _safe_output_path(cls, value: str) -> str:
        return _relative_path(value, "execution.output_dir")


class EvidenceDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_file: str = "run_result.json"

    @field_validator("result_file")
    @classmethod
    def _safe_result_path(cls, value: str) -> str:
        return _relative_path(value, "evidence.result_file")


class HarnessTask(BaseModel):
    """One agent-authored scientific-computing workspace."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = HARNESS_SCHEMA_VERSION
    mode: Literal["agent_workspace"] = "agent_workspace"
    task: TaskDescription
    runner: RunnerDescription = Field(default_factory=RunnerDescription)
    execution: ExecutionDescription = Field(default_factory=ExecutionDescription)
    evidence: EvidenceDescription = Field(default_factory=EvidenceDescription)
