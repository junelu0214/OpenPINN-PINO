"""Build current manuals, validate examples, build Sphinx, then package manuals.

Cross-platform canonical documentation entry point. Does not run the full test
suite or claim visual browser validation. Any failed command stops publication
of the archive; older output files may remain and must not be treated as current.
"""
from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sphinx_commands() -> list[list[str]]:
    """Return strict Chinese-root and English-subtree Sphinx builds."""
    common = [
        sys.executable, "-m", "sphinx", "-E", "-a", "-W", "--keep-going",
        "-b", "html",
    ]
    return [
        [*common, "-d", "docs/_build/doctrees/zh_CN", "docs", "docs/_build/html"],
        [*common, "-D", "language=en", "-D", "html_title=OpenPINN Documentation",
         "-d", "docs/_build/doctrees/en",
         "docs", "docs/_build/html/en"],
    ]


def main() -> None:
    commands = [
        [sys.executable, "scripts/build_manuals.py"],
        [sys.executable, "scripts/validate_manuals.py"],
        *sphinx_commands(),
    ]
    for command in commands:
        print("Running: " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    public = ROOT / "docs/manuals/public"
    date = json.loads((public / "manual_build_report.json").read_text(encoding="utf-8"))["date"]
    archive = ROOT / "docs/manuals" / f"OpenPINN_双说明书_{date}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for name in ("user_manual.html", "developer_manual.html",
                     "manual_build_report.json", "manual_validation.json"):
            package.write(public / name, arcname=name)
    print(f"Documentation built: {ROOT / 'docs/_build/html/index.html'}")
    print(f"Manual archive: {archive}")


if __name__ == "__main__":
    main()
