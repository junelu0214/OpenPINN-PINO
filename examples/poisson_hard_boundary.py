"""Fair Poisson comparison of soft and exact hard Dirichlet conditions."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "reference_runs" / "poisson_hard_boundary.yaml"


def _variant(config: Mapping[str, Any], constraint_name: str) -> dict[str, Any]:
    selected = deepcopy(dict(config))
    selected.setdefault("model", {})["output_transform"] = {"name": constraint_name}
    return selected


def run_comparison(
    config_source: str | Path | Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Run soft and hard conditions while retaining failures and artifacts."""

    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Comparison directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    method_config = op.load_config(config_source)
    output_transform = method_config.get("model", {}).get("output_transform", {})
    configured = (
        output_transform.get("name", "")
        if isinstance(output_transform, Mapping)
        else str(output_transform)
    )
    if str(configured).lower() != "hard":
        raise ValueError("The method config must select model.output_transform.name: hard.")
    variants = {
        "plain_pinn": _variant(method_config, "soft"),
        "hard_boundary": _variant(method_config, "hard"),
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
        plain, hard = results["plain_pinn"], results["hard_boundary"]
        for key in (
            "relative_l2_error",
            "max_absolute_error",
            "rmse",
            "pde_residual_rmse",
            "boundary_rmse",
        ):
            differences[f"hard_minus_plain_{key}"] = float(hard.metrics[key]) - float(
                plain.metrics[key]
            )
        differences["hard_minus_plain_elapsed_seconds"] = float(
            hard.compute_ledger["elapsed_seconds"]
        ) - float(plain.compute_ledger["elapsed_seconds"])
        differences["hard_minus_plain_parameter_count"] = int(
            hard.compute_ledger["parameter_count"]
        ) - int(plain.compute_ledger["parameter_count"])
        write_static_1d_figure(
            root / "comparison.svg",
            plain,
            hard,
            title="Poisson: soft vs exact hard boundary conditions",
            improved_label="Hard boundary",
        )

    comparison = {
        "schema_version": 1,
        "status": "completed" if completed else "failed",
        "accuracy_claim": False,
        "changed_component": "model.output_transform: soft -> hard",
        "fairness": {
            "same_problem": True,
            "same_sampling": True,
            "same_network": True,
            "same_loss_weights": True,
            "same_optimizer": True,
            "same_steps": True,
            "same_experiment_seed": True,
            "same_parameter_count": True,
        },
        "variants": rows,
        "differences": differences,
    }
    write_comparison_files(
        root,
        comparison,
        rows,
        source_note=(
            "The distance-times-network transform was already independently implemented "
            "in the handover starter; no DeepXDE source code was copied or adapted."
        ),
        limitations=(
            "The exact transform here covers compatible 1D Dirichlet interval data, not arbitrary geometries.",
            "The comparison uses one high-frequency Poisson configuration and one seed.",
        ),
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "poisson_hard_boundary_comparison",
    )
    args = parser.parse_args()
    comparison = run_comparison(args.config, args.output_dir)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    if comparison["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
