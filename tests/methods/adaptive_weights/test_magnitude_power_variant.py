from __future__ import annotations

from pathlib import Path

import torch

import openpinn as op


def test_magnitude_power_matches_fixed_fractional_power_formula() -> None:
    composer = op.MagnitudePowerLossComposer(
        {"pde": 2.0, "boundary": 1.0},
        roots={"pde": 2.0, "boundary": 1.0},
    )
    weighted = composer.weighted_terms(
        {"pde": torch.tensor(9.0), "boundary": torch.tensor(4.0)}
    )
    assert torch.isclose(weighted["pde"], torch.tensor(6.0))
    assert torch.isclose(weighted["boundary"], torch.tensor(4.0))
    assert torch.isclose(composer.compose({"pde": torch.tensor(9.0), "boundary": torch.tensor(4.0)}), torch.tensor(10.0))


def test_magnitude_power_trains_as_zero_parameter_loss_transform(tmp_path: Path) -> None:
    result = op.solve(
        "poisson",
        method="adaptive_weights",
        variant="magnitude_power",
        steps=2,
        model={"hidden_layers": [8, 8]},
        sampling={"interior": 8, "boundary": 2},
        method_options={"roots": {"pde": 2.0, "boundary": 2.0}},
        overrides={"evaluation": {"points": 9}},
        output_dir=tmp_path / "magnitude_power",
    )
    assert result.metrics["variant"] == "magnitude_power"
    assert result.compute_ledger["adaptive_weight_parameter_count"] == 0
    assert (result.output_dir / "loss.svg").is_file()
