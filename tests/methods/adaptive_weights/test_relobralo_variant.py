from __future__ import annotations

from pathlib import Path

import torch

import openpinn as op


def test_relobralo_emphasizes_term_with_larger_relative_change() -> None:
    composer = op.ReLoBRaLoLossComposer(
        {"a": 1.0, "b": 1.0},
        alpha=0.0,
        temperature=0.1,
        expected_lookback=1.0,
        seed=3,
    )
    composer.adapt({"a": torch.tensor(1.0), "b": torch.tensor(1.0)})
    updated = composer.adapt({"a": torch.tensor(4.0), "b": torch.tensor(1.0)})
    assert updated["a"] > updated["b"]
    assert abs(updated["a"] + updated["b"] - 2.0) < 1e-6


def test_relobralo_state_round_trip_preserves_history_and_random_stream() -> None:
    original = op.ReLoBRaLoLossComposer(
        {"a": 1.0, "b": 1.0}, expected_lookback=0.5, seed=17
    )
    original.adapt({"a": torch.tensor(1.0), "b": torch.tensor(2.0)})
    original.adapt({"a": torch.tensor(2.0), "b": torch.tensor(1.0)})
    restored = op.ReLoBRaLoLossComposer(
        {"a": 1.0, "b": 1.0}, expected_lookback=0.5, seed=17
    )
    restored.load_training_state_dict(original.training_state_dict())
    next_terms = {"a": torch.tensor(3.0), "b": torch.tensor(0.5)}
    assert original.adapt(next_terms) == restored.adapt(next_terms)


def test_relobralo_trains_without_extra_parameters(tmp_path: Path) -> None:
    result = op.solve(
        "poisson",
        method="adaptive_weights",
        variant="relobralo",
        steps=3,
        model={"hidden_layers": [8, 8]},
        sampling={"interior": 8, "boundary": 2},
        method_options={"alpha": 0.9, "temperature": 0.2, "seed": 5},
        overrides={"evaluation": {"points": 9}},
        output_dir=tmp_path / "relobralo",
    )
    assert result.metrics["variant"] == "relobralo"
    assert result.compute_ledger["adaptive_weight_parameter_count"] == 0
    assert "weight_pde" in result.history[-1]
    assert "final_weight_boundary" in result.metrics
