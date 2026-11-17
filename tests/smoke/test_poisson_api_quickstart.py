from __future__ import annotations

import math

import openpinn as op

from examples.poisson_api_quickstart import run


def test_public_api_quickstart_runs_a_short_poisson_training() -> None:
    trainer, history = run(steps=2, seed=211)

    assert isinstance(trainer, op.Trainer)
    assert isinstance(history, op.TrainingHistory)
    assert len(history) == 2
    assert math.isfinite(float(history.last["total"]))
    assert trainer.steps_completed == 2
