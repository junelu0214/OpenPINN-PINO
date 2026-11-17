#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON:-python}"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
run_stamp="$(date +%Y%m%d_%H%M%S)"
run_root="$repo_root/runs/check_$run_stamp"

cd "$repo_root"
PYTHONPATH=src "$python_bin" -m pytest -q
PYTHONPATH=src "$python_bin" -m openpinn run --problem poisson --method baseline --steps 5 --output-dir "$run_root/poisson"
PYTHONPATH=src "$python_bin" -m openpinn run --problem burgers --method baseline --steps 5 --output-dir "$run_root/burgers"
PYTHONPATH=src "$python_bin" -m openpinn run --problem burgers --method adaptive_sampling --variant residual_adaptive --steps 2 --output-dir "$run_root/adaptive_sampling_variant"
PYTHONPATH=src "$python_bin" scripts/validate_method_configurator.py
"$python_bin" scripts/build_docs.py
printf 'Checks passed. Smoke artifacts: %s\n' "$run_root"
