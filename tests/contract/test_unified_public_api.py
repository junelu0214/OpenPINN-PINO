from __future__ import annotations

import openpinn as op
from openpinn import cli


def test_unified_entry_points_are_public() -> None:
    assert callable(op.solve)
    assert callable(op.solve_config)
    assert callable(op.build_config)
    assert callable(op.list_methods)
    assert callable(op.list_method_variants)


def test_all_method_names_are_unique_and_documentable() -> None:
    specs = op.list_methods()
    names = [spec.name for spec in specs]
    assert len(names) == len(set(names))
    assert all(spec.title and spec.description and spec.component_file for spec in specs)


def test_cli_steps_are_optional_and_can_override_compact_yaml(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_solve_config(source, *, output_dir=None, steps=None):
        captured.update(source=source, output_dir=output_dir, steps=steps)
        return object()

    monkeypatch.setattr(cli, "solve_config", fake_solve_config)
    monkeypatch.setattr(cli, "result_summary", lambda result: "{}")
    assert cli.main(["run", "--config", "example.yaml", "--steps", "7"]) == 0
    assert captured["steps"] == 7


def test_cli_accepts_method_family_and_variant(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_solve(problem, **kwargs):
        captured.update(problem=problem, **kwargs)
        return object()

    monkeypatch.setattr(cli, "solve", fake_solve)
    monkeypatch.setattr(cli, "result_summary", lambda result: "{}")
    assert (
        cli.main(
            [
                "run",
                "--problem",
                "burgers",
                "--method",
                "adaptive_sampling",
                "--variant",
                "residual_adaptive",
                "--steps",
                "3",
            ]
        )
        == 0
    )
    assert captured["method"] == "adaptive_sampling"
    assert captured["variant"] == "residual_adaptive"
