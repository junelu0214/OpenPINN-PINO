#!/usr/bin/env python3
"""Copy the Cloudflare invite gate into a Sphinx HTML directory.

The Pages deploy uploads ``docs/_build/html``. Advanced mode only runs a
``_worker.js`` that is inside that directory, so the worker is staged after
the Sphinx build and is not part of the documentation source.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "deploy" / "cloudflare" / "invite_gate.mjs"
FORBIDDEN = (
    "DOCS_DEPLOY_KEY",
    "CLOUDFLARE_API_TOKEN=",
    "INVITE_PASSPHRASE=",
    "BEGIN OPENSSH",
    "BEGIN PRIVATE KEY",
    "xiwang0706/openpinn-docs",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--html-dir",
        type=Path,
        default=ROOT / "docs" / "_build" / "html",
        help="Sphinx HTML output that wrangler pages deploy will upload",
    )
    args = parser.parse_args()
    if not SOURCE.is_file():
        print(f"Missing gate worker: {SOURCE}", file=sys.stderr)
        return 1
    text = SOURCE.read_text(encoding="utf-8")
    for token in FORBIDDEN:
        if token in text:
            print(f"Refusing to stage the gate worker because it contains {token!r}.", file=sys.stderr)
            return 1
    if "env.ASSETS.fetch" not in text or "INVITE_PASSPHRASE" not in text:
        print("Gate worker is missing the passphrase check or the asset fetch.", file=sys.stderr)
        return 1
    if "node:" in text or "preview_server" in text:
        print("Refusing to stage a worker that imports the local preview server.", file=sys.stderr)
        return 1
    html_dir = args.html_dir
    if not html_dir.is_dir():
        print(f"Sphinx HTML directory does not exist: {html_dir}", file=sys.stderr)
        return 1
    destination = html_dir / "_worker.js"
    shutil.copyfile(SOURCE, destination)
    print(f"Staged invite gate: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
