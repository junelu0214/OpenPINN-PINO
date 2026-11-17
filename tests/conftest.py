from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "configs" / "reference_runs"

# Tests still use ``import openpinn as op``.  This path setup merely permits a
# source-tree test run before an editable install is made.
for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture(scope="session")
def poisson_smoke_result(tmp_path_factory: pytest.TempPathFactory) -> Any:
    from examples.poisson import run

    output_dir = tmp_path_factory.mktemp("openpinn") / "poisson"
    return run(smoke=True, output_dir=output_dir)


@pytest.fixture(scope="session")
def burgers_smoke_result(tmp_path_factory: pytest.TempPathFactory) -> Any:
    from examples.burgers import run

    output_dir = tmp_path_factory.mktemp("openpinn") / "burgers"
    return run(smoke=True, output_dir=output_dir)


@pytest.fixture(scope="session")
def heterogeneous_poisson_smoke_result(
    tmp_path_factory: pytest.TempPathFactory,
) -> Any:
    from examples.heterogeneous_poisson import run

    output_dir = tmp_path_factory.mktemp("openpinn") / "heterogeneous_poisson"
    return run(smoke=True, output_dir=output_dir)


@pytest.fixture(scope="session")
def coupled_manufactured_smoke_result(
    tmp_path_factory: pytest.TempPathFactory,
) -> Any:
    from examples.coupled_manufactured import run

    output_dir = tmp_path_factory.mktemp("openpinn") / "coupled_manufactured"
    return run(smoke=True, output_dir=output_dir)
