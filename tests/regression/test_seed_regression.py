from __future__ import annotations

from pathlib import Path

from examples.poisson import run
from tests._assertions import assert_nested_allclose


def test_same_seed_repeats_the_smoke_trajectory(tmp_path: Path) -> None:
    first = run(smoke=True, output_dir=tmp_path / "first")
    second = run(smoke=True, output_dir=tmp_path / "second")

    assert_nested_allclose(first.metrics, second.metrics)
    assert_nested_allclose(first.predictions, second.predictions)
