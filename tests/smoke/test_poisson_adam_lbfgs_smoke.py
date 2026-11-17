from __future__ import annotations

from pathlib import Path

import openpinn as op

from examples.poisson_adam_lbfgs import run_comparison


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "reference_runs"


def test_adam_lbfgs_uses_explicit_ordered_stages(tmp_path: Path) -> None:
    config = op.load_config(CONFIG_DIR / "poisson_adam_lbfgs.yaml")
    config["optimizer"]["stages"][0]["steps"] = 2
    config["optimizer"]["stages"][1]["steps"] = 1
    config["optimizer"]["stages"][1]["max_iter"] = 2
    config["training"] = {"steps": 3, "resample_every": 10, "log_every": 0}
    config["sampling"]["interior"] = 16
    config["sampling"]["boundary"] = 2
    config["evaluation"]["points"] = 33

    comparison = run_comparison(config, tmp_path / "adam_lbfgs_comparison")

    assert comparison["status"] == "completed"
    assert comparison["changed_component"].startswith("optimizer regime")
    diagnostics = comparison["staged_optimizer_diagnostics"]
    assert diagnostics["completed_outer_steps"] == 3
    assert [stage["name"] for stage in diagnostics["stages"]] == ["adam", "lbfgs"]
    assert diagnostics["stages"][0]["closure_calls"] == 2
    assert diagnostics["stages"][1]["closure_calls"] >= 1
    assert (
        tmp_path
        / "adam_lbfgs_comparison"
        / "adam_lbfgs"
        / "optimizer_diagnostics.json"
    ).is_file()
    assert (tmp_path / "adam_lbfgs_comparison" / "comparison.svg").stat().st_size > 0
    assert (tmp_path / "adam_lbfgs_comparison" / "failure_notes.md").is_file()
