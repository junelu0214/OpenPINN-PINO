"""Fair Poisson comparison of the baseline MLP and Modified MLP."""

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
    write_static_1d_figure,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "reference_runs" / "poisson_modified_mlp.yaml"


def _variant(config: Mapping[str, Any], model_name: str) -> dict[str, Any]:
    selected = deepcopy(dict(config))
    selected.setdefault("model", {})["name"] = model_name
    return selected


def run_comparison(
    config_source: str | Path | Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Run both architectures and keep artifacts even if one run fails."""

    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Comparison directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    method_config = op.load_config(config_source)
    if str(method_config.get("model", {}).get("name", "")).lower() != "modified_mlp":
        raise ValueError("The method config must select model.name: modified_mlp.")
    variants = {
        "plain_pinn": _variant(method_config, "mlp"),
        "modified_mlp": _variant(method_config, "modified_mlp"),
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
        plain, modified = results["plain_pinn"], results["modified_mlp"]
        for key in (
            "relative_l2_error",
            "max_absolute_error",
            "rmse",
            "pde_residual_rmse",
        ):
            differences[f"modified_minus_plain_{key}"] = float(modified.metrics[key]) - float(
                plain.metrics[key]
            )
        differences["modified_minus_plain_elapsed_seconds"] = float(
            modified.compute_ledger["elapsed_seconds"]
        ) - float(plain.compute_ledger["elapsed_seconds"])
        differences["modified_minus_plain_parameter_count"] = int(
            modified.compute_ledger["parameter_count"]
        ) - int(plain.compute_ledger["parameter_count"])
        write_static_1d_figure(
            root / "comparison.svg",
            plain,
            modified,
            title="Poisson: plain MLP vs Modified MLP",
            improved_label="Modified MLP",
        )

    comparison = {
        "schema_version": 1,
        "status": "completed" if completed else "failed",
        "accuracy_claim": False,
        "changed_component": "model.name: mlp -> modified_mlp",
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
            "The implementation was written from the paper description; no code from "
            "the license-unclear reference repository was copied or adapted."
        ),
        limitations=(
            "Modified MLP has additional encoder parameters, so parameter counts differ.",
            "The comparison uses one high-frequency Poisson configuration without a hyperparameter search.",
        ),
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "poisson_modified_mlp_comparison",
    )
    args = parser.parse_args()
    comparison = run_comparison(args.config, args.output_dir)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    if comparison["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
