"""Shared test fixtures for component registry validation tests."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
_SCHEMAS_DIR = Path(__file__).resolve().parents[1] / "schemas"

# Ensure scripts/ is on sys.path so `import validate` and `from _utils...` work
# at module-import time (before any fixtures run).
_scripts_str = str(_SCRIPTS_DIR)
if _scripts_str not in sys.path:
    sys.path.insert(0, _scripts_str)


def _write_json(path: Path, obj: object) -> None:
    """Write a JSON object to `path`, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _minimal_valid_component(**overrides: object) -> dict:
    """Return a minimal valid component dict that passes schema validation."""
    base: dict = {
        "schemaVersion": 1,
        "title": "Test Component",
        "author": {"github": "testuser"},
        "links": {"github": "https://github.com/testuser/test-repo"},
        "governance": {"enabled": True},
        "categories": ["Widgets"],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def repo_root(tmp_path: Path) -> Path:
    """Create a minimal repo layout with schemas and an empty components dir.

    Returns the repo root.  Tests can drop ``*.json`` files into
    ``components/registry/components/`` to trigger specific validation errors.
    """
    registry = tmp_path / "components" / "registry"
    schemas_dir = registry / "schemas"
    components_dir = registry / "components"
    compiled_dir = registry / "compiled"

    # Copy real schema files.
    schemas_dir.mkdir(parents=True)
    for schema_file in _SCHEMAS_DIR.glob("*.json"):
        shutil.copy2(schema_file, schemas_dir / schema_file.name)

    components_dir.mkdir(parents=True)
    compiled_dir.mkdir(parents=True)

    return tmp_path


@pytest.fixture()
def write_component(repo_root: Path):
    """Helper to write a component JSON into the components dir.

    Returns a callable: ``write_component(name, data)`` where ``name`` is
    the filename stem (``.json`` is appended) and ``data`` is a dict.
    """
    components_dir = repo_root / "components" / "registry" / "components"

    def _write(name: str, data: dict) -> Path:
        path = components_dir / f"{name}.json"
        _write_json(path, data)
        return path

    return _write


@pytest.fixture()
def write_raw_file(repo_root: Path):
    """Write arbitrary bytes to a component file (for non-JSON / size tests)."""
    components_dir = repo_root / "components" / "registry" / "components"

    def _write(name: str, content: str | bytes) -> Path:
        path = components_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            path.write_text(content, encoding="utf-8")
        else:
            path.write_bytes(content)
        return path

    return _write


@pytest.fixture()
def scripts_dir() -> Path:
    """Return the scripts directory so tests can adjust ``sys.path``."""
    return _SCRIPTS_DIR


# Re-export for tests that want to build components directly.
__all__ = ["_minimal_valid_component", "_write_json"]
