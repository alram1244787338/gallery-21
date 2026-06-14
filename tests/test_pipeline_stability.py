"""Stability tests for the build_catalog → compute_ranking pipeline.

These tests verify two key properties:

1. **Idempotency** — re-running the scripts with unchanged source data produces
   byte-identical output (no spurious timestamp diffs).
2. **Responsiveness** — when source data or metrics *do* change, the relevant
   timestamps and values are correctly refreshed.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Component Gallery submission (source-of-truth)",
    "type": "object",
    "additionalProperties": False,
    "required": ["schemaVersion", "title", "author", "links", "governance", "categories"],
    "properties": {
        "schemaVersion": {"type": "integer", "const": 1},
        "title": {"type": "string", "minLength": 1, "maxLength": 80},
        "author": {
            "type": "object",
            "additionalProperties": False,
            "required": ["github"],
            "properties": {
                "github": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 39,
                    "pattern": "^[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*$",
                },
                "displayName": {"type": "string", "minLength": 1, "maxLength": 80},
            },
        },
        "links": {
            "type": "object",
            "additionalProperties": False,
            "required": ["github"],
            "properties": {
                "github": {
                    "type": "string",
                    "format": "uri",
                    "pattern": r"^https://github\.com/[^/]+/[^/]+/?$",
                },
                "pypi": {"type": ["string", "null"], "pattern": "^[A-Za-z0-9_.-]+$"},
                "demo": {"type": ["string", "null"], "format": "uri"},
                "docs": {"type": ["string", "null"], "format": "uri"},
            },
        },
        "media": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"image": {"type": ["string", "null"], "format": "uri"}},
        },
        "install": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"pip": {"type": ["string", "null"], "minLength": 1, "maxLength": 200}},
        },
        "governance": {
            "type": "object",
            "additionalProperties": False,
            "required": ["enabled"],
            "properties": {
                "enabled": {"type": "boolean"},
                "notes": {"type": ["string", "null"], "maxLength": 500},
            },
        },
        "categories": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "enum": ["Widgets", "Charts", "Dataframes"]},
        },
    },
}

SAMPLE_COMPONENT: dict[str, Any] = {
    "schemaVersion": 1,
    "title": "TestWidget",
    "author": {"github": "testuser"},
    "links": {
        "github": "https://github.com/testuser/test-widget",
        "pypi": "test-widget",
    },
    "governance": {"enabled": True, "notes": None},
    "categories": ["Widgets"],
}

RANKING_CONFIG: dict[str, Any] = {
    "halfLifeDays": 90.0,
    "weights": {"stars": 1.0, "recency": 2.0, "contributors": 0.5, "downloads": 0.35},
}


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


@pytest.fixture()
def registry(tmp_path: Path) -> dict[str, Path]:
    """Build a minimal registry directory tree for testing."""
    repo_root = tmp_path / "repo"
    registry_root = repo_root / "components" / "registry"
    schemas_dir = registry_root / "schemas"
    components_dir = registry_root / "components"
    compiled_dir = registry_root / "compiled"

    schemas_dir.mkdir(parents=True)
    components_dir.mkdir(parents=True)
    compiled_dir.mkdir(parents=True)

    _write_json(schemas_dir / "component.schema.json", SAMPLE_SCHEMA)
    _write_json(components_dir / "test-widget.json", SAMPLE_COMPONENT)
    _write_json(registry_root / "ranking_config.json", RANKING_CONFIG)

    return {
        "repo_root": repo_root,
        "registry_root": registry_root,
        "schemas_dir": schemas_dir,
        "components_dir": components_dir,
        "compiled_dir": compiled_dir,
        "compiled_file": compiled_dir / "components.json",
        "ranking_config": registry_root / "ranking_config.json",
        "component_file": components_dir / "test-widget.json",
    }


def _make_previous_compiled(
    components: list[dict[str, Any]],
    generated_at: str = "2025-01-01T00:00:00Z",
) -> dict[str, Any]:
    return {
        "generatedAt": generated_at,
        "schemaVersion": 1,
        "categories": ["All", "Widgets", "Charts", "Dataframes"],
        "components": components,
    }


def _sample_previous_component() -> dict[str, Any]:
    """A compiled component as it would exist in a previous artifact."""
    return {
        "title": "TestWidget",
        "author": "testuser",
        "pipLink": "pip install test-widget",
        "pypi": "test-widget",
        "categories": ["Widgets"],
        "image": None,
        "gitHubUrl": "https://github.com/testuser/test-widget",
        "enabled": True,
        "appUrl": None,
        "socialUrl": "https://github.com/testuser",
        "metrics": {
            "github": {
                "stars": 100,
                "forks": 10,
                "openIssues": 5,
                "contributorsCount": 3,
                "lastPushAt": "2025-06-01T12:00:00Z",
                "fetchedAt": "2025-06-10T04:00:00Z",
                "isStale": False,
            },
            "pypi": {
                "latestVersion": "1.0.0",
                "latestReleaseAt": "2025-05-01T10:00:00Z",
                "fetchedAt": "2025-06-10T04:00:00Z",
                "isStale": False,
            },
            "pypistats": {
                "lastDay": 50,
                "lastWeek": 300,
                "lastMonth": 1200,
                "fetchedAt": "2025-06-10T04:00:00Z",
                "isStale": False,
            },
        },
        "ranking": {
            "score": 5.123,
            "signals": {
                "starsScore": 2.004,
                "recencyScore": 0.5,
                "contributorsScore": 0.602,
                "daysSinceUpdate": 30,
                "daysSinceGithubPush": 30,
                "daysSincePypiRelease": 40,
                "downloadsScore": 3.079,
            },
            "computedAt": "2025-06-10T04:00:00Z",
        },
    }


# ---------------------------------------------------------------------------
# build_catalog stability
# ---------------------------------------------------------------------------


class TestBuildCatalogStability:
    def test_generated_at_preserved_on_rerun(self, registry: dict[str, Path]) -> None:
        """When source data is unchanged, generatedAt must not change."""
        from build_catalog import build_catalog

        prev_comp = _sample_previous_component()
        previous = _make_previous_compiled([prev_comp], generated_at="2025-06-10T04:00:00Z")
        _write_json(registry["compiled_file"], previous)

        compiled, errors = build_catalog(
            repo_root=registry["repo_root"],
            registry_root=registry["registry_root"],
            out_path=registry["compiled_file"],
            components_dir=registry["components_dir"],
            previous_path=registry["compiled_file"],
            skip_invalid=False,
        )
        assert not errors
        assert compiled["generatedAt"] == "2025-06-10T04:00:00Z"

    def test_generated_at_refreshed_on_source_change(self, registry: dict[str, Path]) -> None:
        """When source data changes, generatedAt must be updated."""
        from build_catalog import build_catalog

        prev_comp = _sample_previous_component()
        previous = _make_previous_compiled([prev_comp], generated_at="2025-06-10T04:00:00Z")
        _write_json(registry["compiled_file"], previous)

        # Modify the source component (change title)
        modified = {**SAMPLE_COMPONENT, "title": "TestWidget V2"}
        _write_json(registry["component_file"], modified)

        compiled, errors = build_catalog(
            repo_root=registry["repo_root"],
            registry_root=registry["registry_root"],
            out_path=registry["compiled_file"],
            components_dir=registry["components_dir"],
            previous_path=registry["compiled_file"],
            skip_invalid=False,
        )
        assert not errors
        assert compiled["generatedAt"] != "2025-06-10T04:00:00Z"

    def test_generated_at_set_on_first_build(self, registry: dict[str, Path]) -> None:
        """First build (no previous artifact) must set generatedAt."""
        from build_catalog import build_catalog

        compiled, errors = build_catalog(
            repo_root=registry["repo_root"],
            registry_root=registry["registry_root"],
            out_path=registry["compiled_file"],
            components_dir=registry["components_dir"],
            previous_path=None,
            skip_invalid=False,
        )
        assert not errors
        assert isinstance(compiled["generatedAt"], str)
        assert compiled["generatedAt"].endswith("Z")

    def test_byte_identical_output_on_rerun(self, registry: dict[str, Path]) -> None:
        """Full output JSON must be byte-identical on a second run with no changes."""
        from _utils.io import dump_json_atomic
        from build_catalog import build_catalog

        prev_comp = _sample_previous_component()
        previous = _make_previous_compiled([prev_comp], generated_at="2025-06-10T04:00:00Z")
        _write_json(registry["compiled_file"], previous)

        # First run
        compiled1, _ = build_catalog(
            repo_root=registry["repo_root"],
            registry_root=registry["registry_root"],
            out_path=registry["compiled_file"],
            components_dir=registry["components_dir"],
            previous_path=registry["compiled_file"],
            skip_invalid=False,
        )
        dump_json_atomic(registry["compiled_file"], compiled1)
        first_bytes = registry["compiled_file"].read_bytes()

        # Second run (same inputs)
        compiled2, _ = build_catalog(
            repo_root=registry["repo_root"],
            registry_root=registry["registry_root"],
            out_path=registry["compiled_file"],
            components_dir=registry["components_dir"],
            previous_path=registry["compiled_file"],
            skip_invalid=False,
        )
        dump_json_atomic(registry["compiled_file"], compiled2)
        second_bytes = registry["compiled_file"].read_bytes()

        assert first_bytes == second_bytes, "Rerun produced different bytes — spurious diff!"


# ---------------------------------------------------------------------------
# compute_ranking stability
# ---------------------------------------------------------------------------


class TestComputeRankingStability:
    def test_days_since_values_are_integers(self, registry: dict[str, Path]) -> None:
        """daysSince* signals must be whole-day integers for diff stability."""
        from compute_ranking import _compute_ranking, _load_ranking_config

        cfg = _load_ranking_config(registry["ranking_config"])
        comp = _sample_previous_component()
        # Use a now that doesn't land on exact day boundaries
        now = datetime(2025, 7, 15, 13, 37, 42, tzinfo=timezone.utc)  # noqa: UP017

        ranking = _compute_ranking(comp, cfg=cfg, now=now)

        for key in ("daysSinceUpdate", "daysSinceGithubPush", "daysSincePypiRelease"):
            val = ranking["signals"][key]
            if val is not None:
                assert isinstance(val, int), f"{key} should be int, got {type(val).__name__}: {val}"

    def test_computed_at_preserved_when_ranking_unchanged(self, registry: dict[str, Path]) -> None:
        """When ranking signals are identical, computedAt must not change."""
        from compute_ranking import _ranking_changed

        comp = _sample_previous_component()
        old_ranking = comp["ranking"]

        # Simulate: compute ranking with metrics that produce the same signals
        # Use a fixed "now" that would produce the same day counts as old_ranking
        # Old ranking has daysSinceGithubPush=30, daysSincePypiRelease=40
        # lastPushAt=2025-06-01, so now=2025-07-01 gives 30 days
        # latestReleaseAt=2025-05-01, so now=2025-06-10 gives 40 days
        # But we need a single "now" — so we just verify the _ranking_changed logic directly.
        new_ranking = {
            "score": old_ranking["score"],
            "signals": {**old_ranking["signals"]},
            "computedAt": "2025-07-01T00:00:00Z",
        }

        assert not _ranking_changed(new_ranking, old_ranking)

    def test_computed_at_refreshed_when_score_changes(self, registry: dict[str, Path]) -> None:
        """When ranking score changes, computedAt must be updated."""
        from compute_ranking import _ranking_changed

        old_ranking = _sample_previous_component()["ranking"]
        new_ranking = {
            "score": old_ranking["score"] + 1.0,  # Different score
            "signals": {**old_ranking["signals"]},
            "computedAt": "2025-07-01T00:00:00Z",
        }

        assert _ranking_changed(new_ranking, old_ranking)

    def test_computed_at_refreshed_when_signal_changes(self, registry: dict[str, Path]) -> None:
        """When any signal changes, computedAt must be updated."""
        from compute_ranking import _ranking_changed

        old_ranking = _sample_previous_component()["ranking"]
        new_ranking = {
            "score": old_ranking["score"],
            "signals": {**old_ranking["signals"], "starsScore": 99.0},
            "computedAt": "2025-07-01T00:00:00Z",
        }

        assert _ranking_changed(new_ranking, old_ranking)

    def test_stable_ranking_across_time_drift(self, registry: dict[str, Path]) -> None:
        """Two ranking computations seconds apart must produce identical signals."""
        from compute_ranking import _compute_ranking, _load_ranking_config

        cfg = _load_ranking_config(registry["ranking_config"])
        comp = _sample_previous_component()

        now1 = datetime(2025, 7, 15, 13, 37, 0, tzinfo=timezone.utc)  # noqa: UP017
        now2 = datetime(2025, 7, 15, 13, 37, 30, tzinfo=timezone.utc)  # noqa: UP017

        r1 = _compute_ranking(comp, cfg=cfg, now=now1)
        r2 = _compute_ranking(comp, cfg=cfg, now=now2)

        # Signals (excluding computedAt) must be identical
        assert r1["signals"] == r2["signals"]
        assert r1["score"] == r2["score"]

    def test_full_compute_ranking_idempotent(self, registry: dict[str, Path]) -> None:
        """compute_rankings() called twice with same input must produce identical output."""
        from compute_ranking import compute_rankings

        # Create a compiled catalog with a component that has metrics
        comp = _sample_previous_component()
        # Remove old ranking so it gets recomputed
        comp["ranking"] = None
        catalog = _make_previous_compiled([comp], generated_at="2025-06-10T04:00:00Z")
        _write_json(registry["compiled_file"], catalog)

        # First run
        compute_rankings(
            compiled_in=registry["compiled_file"],
            compiled_out=registry["compiled_file"],
            config_path=registry["ranking_config"],
            limit=None,
        )
        first_bytes = registry["compiled_file"].read_bytes()

        # Second run (same input — the ranking values are now in the file)
        # Wait a moment to ensure wall-clock time differs
        time.sleep(1.1)
        compute_rankings(
            compiled_in=registry["compiled_file"],
            compiled_out=registry["compiled_file"],
            config_path=registry["ranking_config"],
            limit=None,
        )
        second_bytes = registry["compiled_file"].read_bytes()

        assert first_bytes == second_bytes, (
            "Second compute_ranking run produced different bytes — spurious diff!"
        )

    def test_computed_at_refreshed_for_new_component(self, registry: dict[str, Path]) -> None:
        """A component with no previous ranking must get a fresh computedAt."""
        from compute_ranking import _ranking_changed

        assert _ranking_changed({"score": 1.0, "signals": {}, "computedAt": "now"}, None)


# ---------------------------------------------------------------------------
# End-to-end pipeline stability
# ---------------------------------------------------------------------------


class TestEndToEndStability:
    def test_build_then_ranking_idempotent(self, registry: dict[str, Path]) -> None:
        """build_catalog → compute_ranking must be idempotent when run twice."""
        from _utils.io import dump_json_atomic
        from build_catalog import build_catalog
        from compute_ranking import compute_rankings

        # Create a previous compiled file
        prev_comp = _sample_previous_component()
        previous = _make_previous_compiled([prev_comp], generated_at="2025-06-10T04:00:00Z")
        _write_json(registry["compiled_file"], previous)

        def run_pipeline() -> bytes:
            compiled, errors = build_catalog(
                repo_root=registry["repo_root"],
                registry_root=registry["registry_root"],
                out_path=registry["compiled_file"],
                components_dir=registry["components_dir"],
                previous_path=registry["compiled_file"],
                skip_invalid=False,
            )
            assert not errors
            dump_json_atomic(registry["compiled_file"], compiled)
            compute_rankings(
                compiled_in=registry["compiled_file"],
                compiled_out=registry["compiled_file"],
                config_path=registry["ranking_config"],
                limit=None,
            )
            return registry["compiled_file"].read_bytes()

        first = run_pipeline()
        time.sleep(1.1)
        second = run_pipeline()

        assert first == second, "Pipeline is not idempotent — rerun produces different output!"

    def test_build_then_ranking_responds_to_metric_change(self, registry: dict[str, Path]) -> None:
        """When metrics change (via previous artifact), ranking must update."""
        from _utils.io import dump_json_atomic
        from build_catalog import build_catalog
        from compute_ranking import compute_rankings

        prev_comp = _sample_previous_component()
        previous = _make_previous_compiled([prev_comp], generated_at="2025-06-10T04:00:00Z")
        _write_json(registry["compiled_file"], previous)

        # Run once to get baseline
        compiled, _ = build_catalog(
            repo_root=registry["repo_root"],
            registry_root=registry["registry_root"],
            out_path=registry["compiled_file"],
            components_dir=registry["components_dir"],
            previous_path=registry["compiled_file"],
            skip_invalid=False,
        )
        dump_json_atomic(registry["compiled_file"], compiled)
        compute_rankings(
            compiled_in=registry["compiled_file"],
            compiled_out=registry["compiled_file"],
            config_path=registry["ranking_config"],
            limit=None,
        )
        baseline = _read_json(registry["compiled_file"])

        # Now simulate a metrics change: update stars in the previous artifact
        prev_comp_v2 = _sample_previous_component()
        prev_comp_v2["metrics"]["github"]["stars"] = 5000  # Massive star increase
        previous_v2 = _make_previous_compiled([prev_comp_v2], generated_at="2025-06-10T04:00:00Z")
        _write_json(registry["compiled_file"], previous_v2)

        # Run pipeline again
        compiled2, _ = build_catalog(
            repo_root=registry["repo_root"],
            registry_root=registry["registry_root"],
            out_path=registry["compiled_file"],
            components_dir=registry["components_dir"],
            previous_path=registry["compiled_file"],
            skip_invalid=False,
        )
        dump_json_atomic(registry["compiled_file"], compiled2)
        compute_rankings(
            compiled_in=registry["compiled_file"],
            compiled_out=registry["compiled_file"],
            config_path=registry["ranking_config"],
            limit=None,
        )
        updated = _read_json(registry["compiled_file"])

        # Stars and ranking score must have changed
        old_stars = baseline["components"][0]["metrics"]["github"]["stars"]
        new_stars = updated["components"][0]["metrics"]["github"]["stars"]
        assert new_stars == 5000
        assert new_stars != old_stars
        assert (
            updated["components"][0]["ranking"]["score"]
            != baseline["components"][0]["ranking"]["score"]
        )
