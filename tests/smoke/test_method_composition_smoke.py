from __future__ import annotations

from pathlib import Path

import openpinn as op


def test_generated_style_two_method_code_runs(tmp_path: Path) -> None:
    methods = [
        op.method(
            "fourier",
            "random_fourier",
            num_frequencies=4,
            scale=1.0,
            seed=29,
            include_input=True,
            trainable=False,
        ),
        op.method(
            "adaptive_weights",
            "relobralo",
            alpha=0.999,
            temperature=0.1,
            expected_lookback=0.999,
            epsilon=1e-12,
            seed=0,
        ),
    ]
    result = op.solve(
        "poisson",
        methods=methods,
        steps=1,
        seed=0,
        model={"hidden_layers": [8, 8]},
        sampling={"interior": 8, "boundary": 4},
        output_root=tmp_path,
        plot=False,
    )
    assert result.metrics["method"] == "fourier+adaptive_weights"
    assert (result.output_dir / "config.yaml").is_file()
    assert (result.output_dir / "metrics.json").is_file()
    assert (result.output_dir / "failure_notes.md").is_file()
