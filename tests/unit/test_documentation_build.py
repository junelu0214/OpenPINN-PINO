"""Ensure the delivery entry point cannot silently publish stale manuals."""
import importlib.util
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_builder(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("openpinn_docs_builder", ROOT / "scripts/build_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module


def test_build_order_and_current_archive(monkeypatch, tmp_path):
    module = load_builder(monkeypatch, tmp_path)
    public = tmp_path / "docs/manuals/public"
    public.mkdir(parents=True)
    contents = {
        "user_manual.html": "current user",
        "developer_manual.html": "current developer",
        "manual_build_report.json": json.dumps({"date": "2026-10-12"}),
        "manual_validation.json": json.dumps({"status": "passed"}),
    }
    for name, value in contents.items():
        (public / name).write_text(value, encoding="utf-8")
    commands = []

    def run(command, **kwargs):
        assert kwargs == {"cwd": tmp_path, "check": True}
        commands.append(command)

    monkeypatch.setattr(module.subprocess, "run", run)
    module.main()
    assert commands[0][1:] == ["scripts/build_manuals.py"]
    assert commands[1][1:] == ["scripts/validate_manuals.py"]
    assert commands[2][1:6] == ["-m", "sphinx", "-E", "-a", "-W"]
    assert commands[2][-1] == "docs/_build/html"
    assert commands[3][1:6] == ["-m", "sphinx", "-E", "-a", "-W"]
    assert "language=en" in commands[3]
    assert commands[3][-1] == "docs/_build/html/en"
    archive = tmp_path / "docs/manuals/OpenPINN_双说明书_2026-10-12.zip"
    with zipfile.ZipFile(archive) as package:
        assert set(package.namelist()) == set(contents)
        for name, value in contents.items():
            assert package.read(name).decode("utf-8") == value


@pytest.mark.parametrize("failed_stage", [0, 1, 2, 3])
def test_failed_stage_stops_delivery(monkeypatch, tmp_path, failed_stage):
    module = load_builder(monkeypatch, tmp_path)
    archive = tmp_path / "docs/manuals/OpenPINN_双说明书_2026-10-12.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"previous delivery")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) - 1 == failed_stage:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        module.main()
    assert len(calls) == failed_stage + 1
    assert archive.read_bytes() == b"previous delivery"
