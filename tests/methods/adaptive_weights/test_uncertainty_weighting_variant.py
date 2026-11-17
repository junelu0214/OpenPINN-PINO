from __future__ import annotations

from pathlib import Path

import torch

import openpinn as op


def test_uncertainty_weights_move_in_expected_directions() -> None:
    composer = op.UncertaintyLossComposer({"large": 1.0, "small": 1.0}, lr=0.05)
    before = composer.current_weights()
    updated = composer.adapt(
        {"large": torch.tensor(4.0), "small": torch.tensor(0.25)}
    )
    assert updated["large"] < float(before["large"])
    assert updated["small"] > float(before["small"])
    total = composer({"large": torch.tensor(4.0), "small": torch.tensor(0.25)})
    assert torch.isfinite(total)


def test_uncertainty_weighting_trains_and_saves_weights(tmp_path: Path) -> None:
    result = op.solve(
        "poisson",
        method="adaptive_weights",
        variant="uncertainty_weighting",
        steps=2,
        model={"hidden_layers": [8, 8]},
        sampling={"interior": 8, "boundary": 2},
        method_options={"lr": 0.01},
        overrides={"evaluation": {"points": 9}},
        output_dir=tmp_path / "uncertainty",
    )
    assert result.metrics["variant"] == "uncertainty_weighting"
    assert result.compute_ledger["adaptive_weight_parameter_count"] == 2
    assert "final_weight_pde" in result.metrics
    assert "final_weight_boundary" in result.metrics
