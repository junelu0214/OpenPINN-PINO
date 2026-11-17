"""Compare one MLP with two interface-coupled Poisson subnetworks."""

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
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "reference_runs" / "heterogeneous_poisson_domain_decomposition.yaml"
)


def _variants(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    plain = deepcopy(dict(config))
    plain["model"]["name"] = "mlp"
    plain["model"].pop("options", None)
    plain["loss"]["formulation"] = "strong"
    plain["loss"]["weights"].pop("interface_solution", None)
    plain["loss"]["weights"].pop("interface_flux", None)
    decomposed = deepcopy(dict(config))
    return {"plain_pinn": plain, "domain_decomposition": decomposed}


def run_comparison(
    config_source: str | Path | Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Run single- and two-network variants and retain every artifact."""

    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Comparison directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    method_config = op.load_config(config_source)
    configured_model = str(method_config.get("model", {}).get("name", ""))
    configured_formulation = str(method_config.get("loss", {}).get("formulation", ""))
    if configured_model.lower().replace("_", "") != "domaindecomposition":
        raise ValueError("Method config must select model.name: domain_decomposition.")
    if configured_formulation.lower().replace("_", "") != "interfacestrong":
        raise ValueError("Method config must select loss.formulation: interface_strong.")
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

    completed = len(results) == 2
    differences: dict[str, float] = {}
    if completed:
        plain = results["plain_pinn"]
        decomposed = results["domain_decomposition"]
        for key in (
            "relative_l2_error",
            "max_absolute_error",
            "rmse",
            "pde_residual_rmse",
            "boundary_rmse",
            "interface_solution_jump",
            "interface_flux_jump",
        ):
            differences[f"decomposed_minus_plain_{key}"] = float(
                decomposed.metrics[key]
            ) - float(plain.metrics[key])
        differences["decomposed_minus_plain_elapsed_seconds"] = float(
            decomposed.compute_ledger["elapsed_seconds"]
        ) - float(plain.compute_ledger["elapsed_seconds"])
        differences["decomposed_minus_plain_parameter_count"] = int(
            decomposed.compute_ledger["parameter_count"]
        ) - int(plain.compute_ledger["parameter_count"])
        write_static_1d_figure(
            root / "comparison.svg",
            plain,
            decomposed,
            title="Heterogeneous Poisson: one MLP vs two subnetworks",
            improved_label="Two subnetworks",
        )

    comparison = {
        "schema_version": 1,
        "status": "completed" if completed else "failed",
        "accuracy_claim": False,
        "changed_component": "single MLP/strong form -> two subnetworks/interface strong form",
        "method_scope": "minimal 1D two-subdomain interface method; not full parallel XPINNs",
        "fairness": {
            "same_problem": True,
            "same_sampling": True,
            "same_subnetwork_width_and_depth_as_plain_network": True,
            "same_optimizer": True,
            "same_steps": True,
            "same_experiment_seed": True,
            "parameter_count_is_expected_to_differ": True,
            "interface_terms_are_intrinsic_to_the_method": True,
        },
        "variants": rows,
        "differences": differences,
    }
    write_comparison_files(
        root,
        comparison,
        rows,
        source_note=(
            "The two-network dispatch and interface losses were independently written; "
            "no code from the MIT reference repository was copied or adapted."
        ),
        limitations=(
            "This starter supports exactly two 1D subdomains and one scalar field.",
            "It is a minimal interface-coupled method, without parallel execution or general XPINN partitions.",
            "The decomposed model has two subnetworks and therefore more parameters than the plain model.",
        ),
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "heterogeneous_poisson_domain_decomposition_comparison",
    )
    args = parser.parse_args()
    comparison = run_comparison(args.config, args.output_dir)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    if comparison["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
