"""Three-way Poisson comparison of Adam, L-BFGS, and their sequence."""

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
    write_static_1d_three_way_figure,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "reference_runs" / "poisson_adam_lbfgs.yaml"


def _variants(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    stages = list(config["optimizer"]["stages"])
    adam_stage = dict(stages[0])
    lbfgs_stage = dict(stages[1])
    adam_steps = int(adam_stage.pop("steps"))
    lbfgs_steps = int(lbfgs_stage.pop("steps"))

    adam = deepcopy(dict(config))
    adam["optimizer"] = adam_stage
    adam["training"]["steps"] = adam_steps + lbfgs_steps

    lbfgs = deepcopy(dict(config))
    lbfgs["optimizer"] = lbfgs_stage
    lbfgs["training"]["steps"] = lbfgs_steps

    staged = deepcopy(dict(config))
    return {"adam_only": adam, "lbfgs_only": lbfgs, "adam_lbfgs": staged}


def run_comparison(
    config_source: str | Path | Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Run the three optimizer regimes and retain every artifact/failure."""

    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Comparison directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    method_config = op.load_config(config_source)
    configured = str(method_config.get("optimizer", {}).get("name", ""))
    if configured.lower().replace("_", "") != "adamlbfgs":
        raise ValueError("The method config must select optimizer.name: adam_lbfgs.")
    variants = _variants(method_config)

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

    completed = len(results) == 3
    differences: dict[str, float] = {}
    optimizer_diagnostics: dict[str, Any] = {}
    if completed:
        adam = results["adam_only"]
        lbfgs = results["lbfgs_only"]
        staged = results["adam_lbfgs"]
        for comparison_name, other in (("lbfgs", lbfgs), ("staged", staged)):
            for key in (
                "relative_l2_error",
                "pde_residual_rmse",
                "boundary_rmse",
            ):
                differences[f"{comparison_name}_minus_adam_{key}"] = float(
                    other.metrics[key]
                ) - float(adam.metrics[key])
            differences[f"{comparison_name}_minus_adam_elapsed_seconds"] = float(
                other.compute_ledger["elapsed_seconds"]
            ) - float(adam.compute_ledger["elapsed_seconds"])
        optimizer_diagnostics = dict(
            staged.compute_ledger.get("optimizer_diagnostics", {})
        )
        write_static_1d_three_way_figure(
            root / "comparison.svg",
            adam,
            lbfgs,
            staged,
            title="Poisson optimizer comparison",
        )

    comparison = {
        "schema_version": 1,
        "status": "completed" if completed else "failed",
        "accuracy_claim": False,
        "changed_component": "optimizer regime: Adam / L-BFGS / Adam -> L-BFGS",
        "fairness": {
            "same_problem": True,
            "same_fixed_collocation_batch": True,
            "same_network": True,
            "same_loss": True,
            "same_experiment_seed": True,
            "same_parameter_count": True,
            "outer_steps_are_not_equivalent_compute_units": True,
            "wall_time_and_closure_calls_are_reported": True,
        },
        "variants": rows,
        "staged_optimizer_diagnostics": optimizer_diagnostics,
        "differences": differences,
    }
    write_comparison_files(
        root,
        comparison,
        rows,
        source_note=(
            "The sequence uses OpenPINN's existing PyTorch optimizer adapters; no code "
            "from the license-unclear opt_for_pinns repository was copied or adapted."
        ),
        limitations=(
            "Adam uses 1020 outer steps, L-BFGS uses 20 outer steps, and the sequence uses 1000 + 20; compare closure calls and time, not outer steps alone.",
            "The three regimes were not tuned to equal wall-clock budgets.",
            "The comparison uses one 1D Poisson configuration and one seed.",
        ),
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "poisson_adam_lbfgs_comparison",
    )
    args = parser.parse_args()
    comparison = run_comparison(args.config, args.output_dir)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    if comparison["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
