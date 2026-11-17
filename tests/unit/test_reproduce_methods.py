from __future__ import annotations

from examples.reproduce_methods import METHODS, primary_pairs, summarize_pairs, summarize_rows


def _row(seed: int, variant: str, error: float, seconds: float) -> dict[str, object]:
    return {
        "method_slug": "fourier",
        "variant": variant,
        "seed": seed,
        "status": "completed",
        "relative_l2_error": error,
        "elapsed_seconds": seconds,
    }


def test_reproducibility_summary_uses_sample_standard_deviation() -> None:
    rows = [
        _row(1, "plain_pinn", 2.0, 1.0),
        _row(2, "plain_pinn", 4.0, 1.2),
    ]

    summary = summarize_rows(rows)
    error = next(row for row in summary if row["metric"] == "relative_l2_error")

    assert error["count"] == 2
    assert error["mean"] == 3.0
    assert error["sample_std"] == 2.0**0.5


def test_primary_pairs_are_matched_by_seed() -> None:
    rows = [
        _row(1, "plain_pinn", 2.0, 1.0),
        _row(1, "fourier_pinn", 1.0, 2.0),
        _row(2, "plain_pinn", 4.0, 1.0),
        _row(2, "fourier_pinn", 1.0, 3.0),
    ]
    spec = (METHODS[0],)

    pairs = primary_pairs(rows, spec)
    summary = summarize_pairs(pairs)

    assert [pair["method_over_baseline"] for pair in pairs] == [0.5, 0.25]
    assert summary[0]["count"] == 2
    assert summary[0]["improved_seed_count"] == 2
    assert summary[0]["mean_method_over_baseline_time"] == 2.5
