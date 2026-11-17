from __future__ import annotations

from pathlib import Path

import torch

import openpinn as op


def _trainer(seed: int) -> op.Trainer:
    torch.manual_seed(seed)
    problem = op.make_poisson_problem()
    model = op.PINNModel(op.MLP(1, 1, hidden_layers=(8, 8)))
    sampler = op.UniformSampler(problem, interior=12, boundary=4, seed=seed)
    return op.Trainer(model, problem, sampler, optimizer="adam", lr=1e-3)


def test_checkpoint_restores_model_optimizer_history_and_step(tmp_path: Path) -> None:
    source = _trainer(61)
    source.fit(steps=2)
    checkpoint = source.save(tmp_path / "checkpoint.pt")

    restored = _trainer(67)
    restored.load(checkpoint, map_location="cpu")
    points = torch.linspace(-1.0, 1.0, 9).reshape(-1, 1)

    torch.testing.assert_close(source.predict(points), restored.predict(points))
    assert restored.steps_completed == source.steps_completed == 2
    assert restored.history.as_list() == source.history.as_list()
    assert restored.optimizer.state_dict().keys() == source.optimizer.state_dict().keys()


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    uninterrupted = _trainer(71)
    uninterrupted.fit(steps=4)

    first_stage = _trainer(71)
    first_stage.fit(steps=2)
    checkpoint = first_stage.save(tmp_path / "resume.pt")
    resumed = _trainer(999)
    resumed.load(checkpoint, map_location="cpu")
    resumed.fit(steps=2)

    points = torch.linspace(-1.0, 1.0, 17).reshape(-1, 1)
    torch.testing.assert_close(
        uninterrupted.predict(points), resumed.predict(points), atol=0.0, rtol=0.0
    )
    assert resumed.steps_completed == uninterrupted.steps_completed == 4
    for uninterrupted_record, resumed_record in zip(
        uninterrupted.history, resumed.history, strict=True
    ):
        comparable = set(uninterrupted_record).difference({"elapsed_seconds"})
        for key in comparable:
            assert uninterrupted_record[key] == resumed_record[key]


def test_checkpoint_resume_preserves_resampling_phase_and_active_batch(
    tmp_path: Path,
) -> None:
    uninterrupted = _trainer(79)
    uninterrupted.fit(steps=6, resample_every=3)

    first_stage = _trainer(79)
    first_stage.fit(steps=2, resample_every=3)
    checkpoint = first_stage.save(tmp_path / "resample_resume.pt")
    resumed = _trainer(997)
    resumed.load(checkpoint, map_location="cpu")
    resumed.fit(steps=4, resample_every=3)

    points = torch.linspace(-1.0, 1.0, 17).reshape(-1, 1)
    torch.testing.assert_close(
        uninterrupted.predict(points), resumed.predict(points), atol=0.0, rtol=0.0
    )
