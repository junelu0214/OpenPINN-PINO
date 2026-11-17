from __future__ import annotations

from pathlib import Path

import openpinn as op

from examples.heterogeneous_poisson_domain_decomposition import run_comparison


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "reference_runs"


def test_domain_decomposition_uses_model_and_formulation_switches(tmp_path: Path) -> None:
    config = op.load_config(
        CONFIG_DIR / "heterogeneous_poisson_domain_decomposition.yaml"
    )
    config["training"] = {"steps": 2, "log_every": 0}
    config["sampling"]["interior"] = 16
    config["sampling"]["boundary"] = 2
    config["evaluation"]["points"] = 33

    comparison = run_comparison(config, tmp_path / "domain_decomposition_comparison")

    assert comparison["status"] == "completed"
    assert comparison["changed_component"].startswith("single MLP")
    assert comparison["fairness"]["interface_terms_are_intrinsic_to_the_method"] is True
    method_metrics = op.load_config(
        tmp_path / "domain_decomposition_comparison" / "domain_decomposition" / "metrics.json"
    )
    method_config = op.load_config(
        tmp_path / "domain_decomposition_comparison" / "domain_decomposition" / "config.yaml"
    )
    assert method_config["model"]["name"] == "domain_decomposition"
    assert method_config["loss"]["formulation"] == "interface_strong"
    assert "train_final_interface_solution_loss" in method_metrics
    assert "train_final_interface_flux_loss" in method_metrics
    assert (tmp_path / "domain_decomposition_comparison" / "comparison.svg").stat().st_size > 0
    assert (tmp_path / "domain_decomposition_comparison" / "failure_notes.md").is_file()
