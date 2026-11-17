"""Shared command-line and validation helpers for the starter examples."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from numbers import Number
from pathlib import Path
from typing import Any

import numpy as np
import torch

import openpinn as op


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "reference_runs"
REQUIRED_ARTIFACTS = (
    "config.yaml",
    "status.json",
    "metrics.json",
    "history.csv",
    "predictions.npz",
    "compute_ledger.json",
    "manifest.json",
)


def _numeric_values(value: Any) -> list[float]:
    """Flatten finite-checkable values while ignoring labels and metadata."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).to(torch.float64).tolist()
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            return value.astype(float, copy=False).reshape(-1).tolist()
        return []
    if isinstance(value, Number) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, Mapping):
        values: list[float] = []
        for item in value.values():
            values.extend(_numeric_values(item))
        return values
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = []
        for item in value:
            values.extend(_numeric_values(item))
        return values
    return []


def assert_finite_result(result: Any) -> None:
    """Validate execution health without making an accuracy claim."""

    values = _numeric_values(result.metrics) + _numeric_values(result.predictions)
    if not values:
        raise RuntimeError("Experiment returned no numeric metrics or predictions.")
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("Experiment returned NaN or infinite values.")


def assert_output_contract(output_dir: Path, *, checkpoint_expected: bool) -> None:
    missing = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).is_file()]
    if checkpoint_expected and not (output_dir / "checkpoint.pt").is_file():
        missing.append("checkpoint.pt")
    if missing:
        raise RuntimeError(f"Experiment output contract is incomplete: {missing}")


def run_config(config_path: Path, output_dir: Path | None = None) -> Any:
    """Run one public-API experiment and enforce the starter output contract."""

    config = deepcopy(op.load_config(config_path))
    if output_dir is not None:
        config.setdefault("output", {})["root"] = str(output_dir)

    result = op.Experiment.from_config(config).run(output_dir=output_dir)
    assert_finite_result(result)

    actual_output_dir = Path(result.output_dir)
    checkpoint_expected = bool(config.get("output", {}).get("save_checkpoint", False))
    assert_output_contract(actual_output_dir, checkpoint_expected=checkpoint_expected)
    return result


def build_parser(problem: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Run the OpenPINN {problem.title()} baseline. "
            "Use --smoke for a fast CPU integration check, not an accuracy benchmark."
        )
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use the tiny deterministic CPU configuration.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Use a custom YAML configuration instead of the bundled baseline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output.root from the selected configuration.",
    )
    return parser


def select_config(problem: str, *, smoke: bool, custom: Path | None) -> Path:
    if custom is not None:
        return custom.expanduser().resolve()
    suffix = "smoke" if smoke else "baseline"
    return CONFIG_DIR / f"{problem}_{suffix}.yaml"


def print_summary(problem: str, config_path: Path, result: Any) -> None:
    summary = {
        "problem": problem,
        "config": str(config_path),
        "output_dir": str(result.output_dir),
        "status": "completed_with_finite_outputs",
        "accuracy_claim": False,
        "metrics": result.metrics,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
