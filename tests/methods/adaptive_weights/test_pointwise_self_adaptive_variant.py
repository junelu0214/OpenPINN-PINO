from __future__ import annotations

from pathlib import Path

import torch

import openpinn as op


def test_pointwise_masks_match_points_and_harder_point_grows_more() -> None:
    composer = op.PointwiseAdaptiveLossComposer({"pde": 1.0, "boundary": 1.0}, lr=0.05)
    terms = {
        "pde": torch.tensor([4.0, 0.0]),
        "boundary": torch.tensor([1.0]),
    }
    total = composer.compose_pointwise(terms)
    assert torch.isfinite(total)
    before = {name: value.clone() for name, value in composer.point_weights().items()}
    composer.adapt(terms)
    after = composer.point_weights()
    assert sum(value.numel() for value in after.values()) == 3
    assert after["pde"][0] > before["pde"][0]
    assert torch.isclose(after["pde"][1], before["pde"][1])


def test_pointwise_masks_and_optimizer_state_round_trip() -> None:
    terms = {
        "pde": torch.tensor([2.0, 0.0]),
        "boundary": torch.tensor([1.0]),
    }
    original = op.PointwiseAdaptiveLossComposer(
        {"pde": 1.0, "boundary": 1.0}, lr=0.01
    )
    original.compose_pointwise(terms)
    original.adapt(terms)
    restored = op.PointwiseAdaptiveLossComposer(
        {"pde": 1.0, "boundary": 1.0}, lr=0.01
    )
    restored.load_training_state_dict(original.training_state_dict())
    for name in terms:
        assert torch.equal(
            original.point_weights()[name], restored.point_weights()[name]
        )
    original_next = original.adapt(terms)
    restored_next = restored.adapt(terms)
    for name in terms:
        assert abs(original_next[name] - restored_next[name]) < 1e-6


def test_pointwise_sa_pinn_trains_with_one_mask_per_training_point(tmp_path: Path) -> None:
    result = op.solve(
        "burgers",
        method="adaptive_weights",
        variant="pointwise_self_adaptive",
        steps=2,
        model={"hidden_layers": [8, 8]},
        sampling={"interior": 8, "boundary": 2, "initial": 2},
        method_options={"lr": 0.01},
        overrides={"evaluation": {"points": {"space": 5, "time": 4}}},
        output_dir=tmp_path / "pointwise_sa",
    )
    assert result.metrics["variant"] == "pointwise_self_adaptive"
    assert result.compute_ledger["adaptive_weight_parameter_count"] == 12
    assert set(
        key for key in result.metrics if key.startswith("final_weight_")
    ) == {"final_weight_pde", "final_weight_boundary", "final_weight_initial"}
