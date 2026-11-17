"""Fair Burgers comparison of uniform and residual-adaptive sampling."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "reference_runs" / "burgers_adaptive_sampling.yaml"


def _variant(config: Mapping[str, Any], sampler_name: str) -> dict[str, Any]:
    selected = deepcopy(dict(config))
    selected.setdefault("sampling", {})["name"] = sampler_name
    if sampler_name == "uniform":
        selected["sampling"].pop("options", None)
    return selected


def run_comparison(
    config_source: str | Path | Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Run uniform and residual-adaptive Burgers samplers."""

    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Comparison directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    method_config = op.load_config(config_source)
    configured = str(method_config.get("sampling", {}).get("name", ""))
    if configured.lower().replace("_", "") != "residualadaptive":
        raise ValueError("The method config must select sampling.name: residual_adaptive.")
    variants = {
        "plain_pinn": _variant(method_config, "uniform"),
        "adaptive_sampling": _variant(method_config, "residual_adaptive"),
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
        plain, adaptive = results["plain_pinn"], results["adaptive_sampling"]
        for key in (
            "relative_l2_error",
            "max_absolute_error",
            "rmse",
            "pde_residual_rmse",
            "boundary_rmse",
            "initial_rmse",
        ):
            differences[f"adaptive_minus_plain_{key}"] = float(
                adaptive.metrics[key]
            ) - float(plain.metrics[key])
        differences["adaptive_minus_plain_elapsed_seconds"] = float(
            adaptive.compute_ledger["elapsed_seconds"]
        ) - float(plain.compute_ledger["elapsed_seconds"])
        differences["adaptive_minus_plain_parameter_count"] = int(
            adaptive.compute_ledger["parameter_count"]
        ) - int(plain.compute_ledger["parameter_count"])
        diagnostics = dict(adaptive.compute_ledger.get("sampler_diagnostics", {}))
        write_dynamic_1d_figure(
            root / "comparison.svg",
            plain,
            adaptive,
            title="Burgers: uniform vs residual-adaptive sampling",
            improved_label="Adaptive sampling",
        )

    comparison = {
        "schema_version": 1,
        "status": "completed" if completed else "failed",
        "accuracy_claim": False,
        "changed_component": "sampling.name: uniform -> residual_adaptive",
        "fairness": {
            "same_problem": True,
            "same_network": True,
            "same_loss": True,
            "same_optimizer": True,
            "same_steps": True,
            "same_interior_point_count": True,
            "same_boundary_and_initial_point_counts": True,
            "same_resample_interval": True,
            "same_boundary_and_initial_random_stream": True,
            "same_experiment_seed": True,
            "same_parameter_count": True,
        },
        "variants": rows,
        "sampling_diagnostics": diagnostics,
        "differences": differences,
    }
    write_comparison_files(
        root,
        comparison,
        rows,
        source_note=(
            "The RAD-style probability sampler was independently written from the "
            "paper description; no code from the Apache-2.0 repository was copied."
        ),
        limitations=(
            "This is one residual-weighted distribution variant, not every method in the paper.",
            "Candidate-pool residual evaluation adds compute and is included in elapsed time.",
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
        default=PROJECT_ROOT / "runs" / "burgers_adaptive_sampling_comparison",
    )
    args = parser.parse_args()
    comparison = run_comparison(args.config, args.output_dir)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    if comparison["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
