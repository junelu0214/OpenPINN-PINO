"""Fair Burgers comparison of full-time and expanding-window training."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "reference_runs" / "burgers_time_window.yaml"


def _variant(config: Mapping[str, Any], sampler_name: str) -> dict[str, Any]:
    selected = deepcopy(dict(config))
    selected.setdefault("sampling", {})["name"] = sampler_name
    if sampler_name == "uniform":
        selected["sampling"].pop("options", None)
    return selected


def run_comparison(
    config_source: str | Path | Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Run full-time and time-window samplers with complete artifacts."""

    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Comparison directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    method_config = op.load_config(config_source)
    configured = str(method_config.get("sampling", {}).get("name", ""))
    if configured.lower().replace("_", "") != "timewindow":
        raise ValueError("The method config must select sampling.name: time_window.")
    variants = {
        "plain_pinn": _variant(method_config, "uniform"),
        "time_window": _variant(method_config, "time_window"),
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
    diagnostics: dict[str, Any] = {}
    if completed:
        plain, window = results["plain_pinn"], results["time_window"]
        for key in (
            "relative_l2_error",
            "max_absolute_error",
            "rmse",
            "pde_residual_rmse",
            "boundary_rmse",
            "initial_rmse",
        ):
            differences[f"window_minus_plain_{key}"] = float(window.metrics[key]) - float(
                plain.metrics[key]
            )
        differences["window_minus_plain_elapsed_seconds"] = float(
            window.compute_ledger["elapsed_seconds"]
        ) - float(plain.compute_ledger["elapsed_seconds"])
        differences["window_minus_plain_parameter_count"] = int(
            window.compute_ledger["parameter_count"]
        ) - int(plain.compute_ledger["parameter_count"])
        diagnostics = dict(window.compute_ledger.get("sampler_diagnostics", {}))
        write_dynamic_1d_figure(
            root / "comparison.svg",
            plain,
            window,
            title="Burgers: full-time vs expanding time-window training",
            improved_label="Time window",
        )

    comparison = {
        "schema_version": 1,
        "status": "completed" if completed else "failed",
        "accuracy_claim": False,
        "changed_component": "sampling.name: uniform -> time_window",
        "method_scope": "generic expanding time-window curriculum; not patented CausalPINNs loss weighting",
        "fairness": {
            "same_problem": True,
            "same_network": True,
            "same_loss": True,
            "same_optimizer": True,
            "same_steps": True,
            "same_point_counts": True,
            "same_resample_interval": True,
            "same_experiment_seed": True,
            "same_parameter_count": True,
        },
        "variants": rows,
        "time_window_diagnostics": diagnostics,
        "differences": differences,
    }
    write_comparison_files(
        root,
        comparison,
        rows,
        source_note=(
            "No CausalPINNs code or patented causal-loss implementation was copied, adapted, "
            "or enabled; this is an independently written generic time-domain curriculum."
        ),
        limitations=(
            "This result must not be described as a reproduction of the paper's causal weighting algorithm.",
            "The Cole-Hopf reference is used only for post-training evaluation.",
            "The schedule uses four equal time prefixes and one seed without hyperparameter search.",
        ),
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "burgers_time_window_comparison",
    )
    args = parser.parse_args()
    comparison = run_comparison(args.config, args.output_dir)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    if comparison["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
