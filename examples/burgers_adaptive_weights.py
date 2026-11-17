"""Fair Burgers comparison of fixed and trainable term-level loss weights."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "reference_runs" / "burgers_adaptive_weights.yaml"


def _variant(config: Mapping[str, Any], weighting_name: str) -> dict[str, Any]:
    selected = deepcopy(dict(config))
    if weighting_name == "fixed":
        selected.setdefault("loss", {})["weighting"] = {"name": "fixed"}
    else:
        selected.setdefault("loss", {}).setdefault("weighting", {})["name"] = weighting_name
    return selected


def run_comparison(
    config_source: str | Path | Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Run fixed and adaptive term weights and preserve all outcomes."""

    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Comparison directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    method_config = op.load_config(config_source)
    weighting = method_config.get("loss", {}).get("weighting", {})
    configured = weighting.get("name", "") if isinstance(weighting, Mapping) else weighting
    if str(configured).lower().replace("_", "") != "selfadaptive":
        raise ValueError("The method config must select loss.weighting.name: self_adaptive.")
    variants = {
        "plain_pinn": _variant(method_config, "fixed"),
        "adaptive_weights": _variant(method_config, "self_adaptive"),
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
    learned_weights: dict[str, float] = {}
    if completed:
        plain, adaptive = results["plain_pinn"], results["adaptive_weights"]
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
        learned_weights = {
            key.removeprefix("final_weight_"): float(value)
            for key, value in adaptive.metrics.items()
            if key.startswith("final_weight_")
        }
        write_dynamic_1d_figure(
            root / "comparison.svg",
            plain,
            adaptive,
            title="Burgers: fixed vs adaptive term-level loss weights",
            improved_label="Adaptive weights",
        )

    comparison = {
        "schema_version": 1,
        "status": "completed" if completed else "failed",
        "accuracy_claim": False,
        "changed_component": "loss.weighting: fixed -> self_adaptive",
        "method_scope": "term-level saddle-point variant; not per-collocation-point SA-PINN",
        "fairness": {
            "same_problem": True,
            "same_sampling": True,
            "same_network": True,
            "same_initial_loss_weights": True,
            "same_network_optimizer": True,
            "same_steps": True,
            "same_experiment_seed": True,
            "adaptive_weight_parameter_count_is_expected_to_differ": True,
        },
        "variants": rows,
        "learned_weights": learned_weights,
        "differences": differences,
    }
    write_comparison_files(
        root,
        comparison,
        rows,
        source_note=(
            "The term-level saddle-point update was independently written from the paper idea; "
            "no code from the license-unclear SA-PINNs repository was copied or adapted."
        ),
        limitations=(
            "This minimal method learns one positive scalar per loss term, whereas the paper learns one mask per collocation point.",
            "The adaptive scalars use SGD ascent while the network uses Adam descent.",
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
        default=PROJECT_ROOT / "runs" / "burgers_adaptive_weights_comparison",
    )
    args = parser.parse_args()
    comparison = run_comparison(args.config, args.output_dir)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    if comparison["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
