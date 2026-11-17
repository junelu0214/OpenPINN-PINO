#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON:-python}"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
"$python_bin" scripts/build_docs.py
