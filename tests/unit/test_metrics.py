from __future__ import annotations

import math

import openpinn as op


def test_error_metrics_have_expected_values() -> None:
    prediction = [1.0, 3.0, 5.0]
    reference = [1.0, 2.0, 4.0]

    assert math.isclose(op.mean_absolute_error(prediction, reference), 2.0 / 3.0)
    assert math.isclose(op.rmse(prediction, reference), math.sqrt(2.0 / 3.0))
    assert math.isclose(op.r2_score(prediction, reference), 4.0 / 7.0)


def test_r2_score_is_finite_for_constant_targets() -> None:
    assert op.r2_score([2.0, 2.0], [2.0, 2.0]) == 1.0
    assert op.r2_score([2.0, 3.0], [2.0, 2.0]) == 0.0
