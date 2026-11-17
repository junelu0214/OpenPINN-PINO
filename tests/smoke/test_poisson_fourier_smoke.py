from __future__ import annotations

import json
from pathlib import Path

import openpinn as op

from examples.poisson_fourier import run_comparison


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "reference_runs"


def test_fourier_poisson_uses_the_unified_feature_switch(tmp_path: Path) -> None:
    config = op.load_config(CONFIG_DIR / "poisson_fourier.yaml")
    config["training"] = {"steps": 2, "log_every": 0}
    config["sampling"]["interior"] = 16
    config["sampling"]["boundary"] = 2
    config["evaluation"]["points"] = 33

    comparison = run_comparison(config, tmp_path / "fourier_comparison")

    assert comparison["status"] == "completed"
    assert comparison["changed_component"] == "model.feature: identity -> fourier"
    assert {variant["method"] for variant in comparison["variants"]} == {
        "plain_pinn",
        "fourier_pinn",
    }
    persisted = op.load_config(
        tmp_path / "fourier_comparison" / "fourier_pinn" / "config.yaml"
    )
    assert persisted["model"]["feature"]["name"] == "fourier"
    assert (tmp_path / "fourier_comparison" / "comparison.svg").stat().st_size > 0
    notes = (tmp_path / "fourier_comparison" / "failure_notes.md").read_text(
        encoding="utf-8"
    )
    assert "None. Both runs completed" in notes
    summary = json.loads(
        (tmp_path / "fourier_comparison" / "comparison.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["fairness"]["same_steps"] is True
