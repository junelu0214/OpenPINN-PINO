from pathlib import Path

from examples.compare_method_variant import run_comparison


def test_generic_variant_comparison_keeps_test_outputs_in_tmp_path(tmp_path: Path) -> None:
    result = run_comparison(
        problem="poisson",
        family="adaptive_weights",
        variant="uncertainty_weighting",
        output_dir=tmp_path / "comparison",
        steps=2,
        method_options={"lr": 0.01},
    )
    assert result["status"] == "completed"
    assert result["variant"] == "uncertainty_weighting"
    root = tmp_path / "comparison"
    assert (root / "comparison.json").is_file()
    assert (root / "comparison.csv").is_file()
    assert (root / "comparison.svg").is_file()
    assert (root / "failure_notes.md").is_file()
