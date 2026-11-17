from __future__ import annotations

from pathlib import Path

import openpinn as op

from examples.poisson_modified_mlp import run_comparison


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "reference_runs"


def test_modified_mlp_uses_the_unified_model_switch(tmp_path: Path) -> None:
    config = op.load_config(CONFIG_DIR / "poisson_modified_mlp.yaml")
    config["training"] = {"steps": 2, "log_every": 0}
    config["sampling"]["interior"] = 16
    config["sampling"]["boundary"] = 2
    config["evaluation"]["points"] = 33

    comparison = run_comparison(config, tmp_path / "modified_mlp_comparison")

    assert comparison["status"] == "completed"
    assert comparison["changed_component"] == "model.name: mlp -> modified_mlp"
    assert comparison["fairness"]["same_steps"] is True
    modified_config = op.load_config(
        tmp_path / "modified_mlp_comparison" / "modified_mlp" / "config.yaml"
    )
    assert modified_config["model"]["name"] == "modified_mlp"
    assert (tmp_path / "modified_mlp_comparison" / "comparison.svg").stat().st_size > 0
    assert (tmp_path / "modified_mlp_comparison" / "failure_notes.md").is_file()
