"""Fair Burgers comparison of fixed tanh and layer-wise adaptive tanh."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import openpinn as op

from ._comparison import (
    failure_summary,
    result_summary,
    write_comparison_files,
    write_dynamic_1d_figure,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "reference_runs" / "burgers_adaptive_activation.yaml"


def _variant(config: Mapping[str, Any], activation: str) -> dict[str, Any]:
    selected = deepcopy(dict(config))
    selected.setdefault("model", {})["activation"] = activation
    return selected


def run_comparison(
    config_source: str | Path | Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Run both activations and keep artifacts even if one run fails."""

    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Comparison directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    method_config = op.load_config(config_source)
    configured = str(method_config.get("model", {}).get("activation", "")).lower()
    if configured.replace("_", "") != "adaptivetanh":
        raise ValueError("The method config must select model.activation: adaptive_tanh.")
    variants = {
        "plain_pinn": _variant(method_config, "tanh"),
        "adaptive_tanh": _variant(method_config, "adaptive_tanh"),
    }

    results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for method, config in variants.items():
        variant_dir = root / method
        try:
            result = op.Experiment.from_config(config).run(output_dir=variant_dir)
        except Exception as error:
            rows.append(failure_summary(method, variant_dir, error))
        else:
            results[method] = result
            rows.append(result_summary(method, result))

    completed = len(results) == 2
    differences: dict[str, float] = {}
    if completed:
        plain, adaptive = results["plain_pinn"], results["adaptive_tanh"]
        for key in (
            "relative_l2_error",
            "max_absolute_error",
            "rmse",
            "pde_residual_rmse",
            "boundary_rmse",
            "initial_rmse",
        ):
            differences[f"adaptive_minus_plain_{key}"] = float(adaptive.metrics[key]) - float(
                plain.metrics[key]
            )
        differences["adaptive_minus_plain_elapsed_seconds"] = float(
            adaptive.compute_ledger["elapsed_seconds"]
        ) - float(plain.compute_ledger["elapsed_seconds"])
        differences["adaptive_minus_plain_parameter_count"] = int(
            adaptive.compute_ledger["parameter_count"]
        ) - int(plain.compute_ledger["parameter_count"])
        write_dynamic_1d_figure(
            root / "comparison.svg",
            plain,
            adaptive,
            title="Burgers: fixed tanh vs adaptive tanh",
            improved_label="Adaptive tanh",
        )

    comparison = {
        "schema_version": 1,
        "status": "completed" if completed else "failed",
        "accuracy_claim": False,
        "changed_component": "model.activation: tanh -> adaptive_tanh",
        "fairness": {
            "same_problem": True,
            "same_sampling": True,
            "same_hidden_width_and_depth": True,
            "same_optimizer": True,
            "same_steps": True,
            "same_experiment_seed": True,
            "parameter_count_is_expected_to_differ": True,
        },
        "variants": rows,
        "differences": differences,
    }
    write_comparison_files(
        root,
        comparison,
        rows,
        source_note=(
            "This independent PyTorch implementation uses the paper's trainable-slope "
            "idea; no TensorFlow reference code was copied. The reference repository is MIT licensed."
        ),
        limitations=(
            "The adaptive variant adds one scalar parameter per hidden layer.",
            "The Cole-Hopf reference is used only for post-training evaluation.",
            "This is one fixed-seed comparison, not a statistical superiority claim.",
        ),
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "burgers_adaptive_activation_comparison",
    )
    args = parser.parse_args()
    comparison = run_comparison(args.config, args.output_dir)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    if comparison["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
