from pathlib import Path

import pytest

from pg_factory.deps import FactoryDeps
from pg_factory.state import FactoryState


@pytest.fixture
def factory_state() -> FactoryState:
    return FactoryState()


@pytest.fixture
def factory_deps(tmp_path: Path) -> FactoryDeps:
    return FactoryDeps(
        project_path=tmp_path,
        dry_run=True,
    )
