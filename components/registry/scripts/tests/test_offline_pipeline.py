"""Tests for offline pipeline (--no-enrich) behavior and freshness handling.

These tests verify:
1. _utils/freshness.py correctly detects stale metrics
2. compute_ranking.py handles stale data properly (isStale, computedAt)
3. build_catalog.py outputs correct pipelineMode and carry-forward metadata
4. compiled.schema.json validates both full and offline artifacts
5. Integration: the full vs offline pipeline paths produce explainably different outputs
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Adjust sys.path so imports work when running from the tests/ directory.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# These tests require Python 3.11+ for datetime.UTC.
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason=f"Requires Python 3.11+ (current: {sys.version_info.major}.{sys.version_info.minor})",
)

# Conditional imports — only evaluated when tests actually run (Python 3.11+).
if sys.version_info >= (3, 11):
    from datetime import UTC, datetime, timedelta

    from _utils.freshness import detect_stale_metrics, oldest_metric_fetched_at
    from _utils.io import dump_json, load_json
    from _utils.time import utc_now_iso
else:
    # Dummy values so the module can be collected by pytest on older Python.
    UTC = None  # type: ignore[assignment]
    datetime = None  # type: ignore[assignment, misc]
    timedelta = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _iso(dt: datetime) -> str:
    """Format a datetime as ISO8601 with Z suffix."""
    return dt.isoformat().replace("+00:00", "Z")


def _make_component(
    *,
    title: str = "Test Component",
    github_url: str = "https://github.com/test/repo",
    pypi: str | None = "test-pkg",
    pip_link: str | None = "pip install test-pkg",
    gh_stars: int = 100,
    gh_fetched_at: str | None = None,
    gh_is_stale: bool | None = None,
    gh_last_push_at: str | None = None,
    pypi_fetched_at: str | None = None,
    pypi_is_stale: bool | None = None,
    pypistats_fetched_at: str | None = None,
    pypistats_is_stale: bool | None = None,
    pypistats_last_month: int | None = 1000,
) -> dict:
    """Create a minimal compiled component for testing."""
    comp: dict = {
        "title": title,
        "author": "test",
        "pipLink": pip_link,
        "pypi": pypi,
        "categories": ["Widgets"],
        "image": None,
        "gitHubUrl": github_url,
        "enabled": True,
        "appUrl": None,
        "socialUrl": "https://github.com/test",
        "metrics": {},
        "ranking": None,
    }

    gh_metrics: dict = {
        "stars": gh_stars,
        "forks": 10,
        "contributorsCount": 5,
        "openIssues": 3,
        "lastPushAt": gh_last_push_at or _iso(datetime.now(UTC) - timedelta(days=30)),
    }
    if gh_fetched_at is not None:
        gh_metrics["fetchedAt"] = gh_fetched_at
    if gh_is_stale is not None:
        gh_metrics["isStale"] = gh_is_stale
    comp["metrics"]["github"] = gh_metrics

    if pypi:
        pypi_metrics: dict = {
            "latestVersion": "1.0.0",
            "latestReleaseAt": _iso(datetime.now(UTC) - timedelta(days=60)),
        }
        if pypi_fetched_at is not None:
            pypi_metrics["fetchedAt"] = pypi_fetched_at
        if pypi_is_stale is not None:
            pypi_metrics["isStale"] = pypi_is_stale
        comp["metrics"]["pypi"] = pypi_metrics

    if pypi:
        ps_metrics: dict = {}
        if pypistats_last_month is not None:
            ps_metrics["lastMonth"] = pypistats_last_month
            ps_metrics["lastWeek"] = 250
            ps_metrics["lastDay"] = 30
        if pypistats_fetched_at is not None:
            ps_metrics["fetchedAt"] = pypistats_fetched_at
        if pypistats_is_stale is not None:
            ps_metrics["isStale"] = pypistats_is_stale
        comp["metrics"]["pypistats"] = ps_metrics

    return comp


def _make_ranking_config() -> dict:
    return {
        "schemaVersion": 1,
        "halfLifeDays": 90.0,
        "weights": {
            "stars": 1.0,
            "recency": 2.0,
            "contributors": 0.5,
            "downloads": 0.35,
        },
    }


# ---------------------------------------------------------------------------
# Tests: _utils/freshness.py — detect_stale_metrics
# ---------------------------------------------------------------------------


class TestDetectStaleMetrics:
    """Tests for detect_stale_metrics()."""

    def test_fresh_metrics_not_stale(self):
        """Metrics fetched recently should not be flagged as stale."""
        now = datetime.now(UTC)
        comp = _make_component(
            gh_fetched_at=_iso(now - timedelta(hours=1)),
            gh_is_stale=False,
            pypi_fetched_at=_iso(now - timedelta(hours=1)),
            pypi_is_stale=False,
            pypistats_fetched_at=_iso(now - timedelta(hours=1)),
            pypistats_is_stale=False,
        )
        result = detect_stale_metrics(comp, stale_threshold_hours=48.0)
        assert result["github"] is False
        assert result["pypi"] is False
        assert result["pypistats"] is False

    def test_old_metrics_are_stale(self):
        """Metrics fetched long ago should be flagged as stale."""
        now = datetime.now(UTC)
        comp = _make_component(
            gh_fetched_at=_iso(now - timedelta(hours=100)),
            gh_is_stale=False,
            pypi_fetched_at=_iso(now - timedelta(hours=100)),
            pypi_is_stale=False,
            pypistats_fetched_at=_iso(now - timedelta(hours=100)),
            pypistats_is_stale=False,
        )
        result = detect_stale_metrics(comp, stale_threshold_hours=48.0)
        assert result["github"] is True
        assert result["pypi"] is True
        assert result["pypistats"] is True

    def test_explicit_is_stale_flag(self):
        """isStale=true should always be detected regardless of fetchedAt age."""
        now = datetime.now(UTC)
        comp = _make_component(
            gh_fetched_at=_iso(now - timedelta(hours=1)),
            gh_is_stale=True,  # explicitly stale
            pypi_fetched_at=_iso(now - timedelta(hours=1)),
            pypi_is_stale=False,
            pypistats_fetched_at=_iso(now - timedelta(hours=1)),
            pypistats_is_stale=False,
        )
        result = detect_stale_metrics(comp, stale_threshold_hours=48.0)
        assert result["github"] is True
        assert result["pypi"] is False
        assert result["pypistats"] is False

    def test_missing_fetched_at_is_stale(self):
        """Metrics without fetchedAt should be treated as stale."""
        comp = _make_component(
            gh_fetched_at=None,  # no fetchedAt
            pypi_fetched_at=None,
            pypistats_fetched_at=None,
        )
        result = detect_stale_metrics(comp, stale_threshold_hours=48.0)
        assert result["github"] is True
        assert result["pypi"] is True
        assert result["pypistats"] is True

    def test_no_metrics_bucket_with_expected_data(self):
        """Component with gitHubUrl but no github metrics → stale for github."""
        comp = _make_component()
        comp["metrics"] = {}  # no buckets at all
        result = detect_stale_metrics(comp, stale_threshold_hours=48.0)
        # Has gitHubUrl → expected github data → stale
        assert result["github"] is True
        # Has pypi/pipLink → expected pypi data → stale
        assert result["pypi"] is True
        assert result["pypistats"] is True

    def test_no_metrics_bucket_without_expected_data(self):
        """Component without gitHubUrl and no github metrics → not stale."""
        comp = _make_component(github_url="", pypi=None, pip_link=None)
        comp["metrics"] = {}
        result = detect_stale_metrics(comp, stale_threshold_hours=48.0)
        # No gitHubUrl → no github data expected → not stale
        assert result["github"] is False
        # No pypi/pipLink → no pypi data expected → not stale
        assert result["pypi"] is False
        assert result["pypistats"] is False

    def test_zero_threshold_disables_age_check(self):
        """stale_threshold_hours=0 should only check isStale flags."""
        now = datetime.now(UTC)
        comp = _make_component(
            gh_fetched_at=_iso(now - timedelta(hours=1000)),
            gh_is_stale=False,
            pypi_fetched_at=_iso(now - timedelta(hours=1000)),
            pypi_is_stale=False,
            pypistats_fetched_at=_iso(now - timedelta(hours=1000)),
            pypistats_is_stale=False,
        )
        result = detect_stale_metrics(comp, stale_threshold_hours=0)
        # Age check disabled; isStale=false → not stale
        assert result["github"] is False
        assert result["pypi"] is False
        assert result["pypistats"] is False


# ---------------------------------------------------------------------------
# Tests: _utils/freshness.py — oldest_metric_fetched_at
# ---------------------------------------------------------------------------


class TestOldestMetricFetchedAt:
    """Tests for oldest_metric_fetched_at()."""

    def test_returns_oldest_timestamp(self):
        now = datetime.now(UTC)
        old_ts = _iso(now - timedelta(days=10))
        new_ts = _iso(now - timedelta(hours=1))
        comp = _make_component(
            gh_fetched_at=old_ts,
            pypi_fetched_at=new_ts,
            pypistats_fetched_at=new_ts,
        )
        result = oldest_metric_fetched_at(comp)
        assert result == old_ts

    def test_returns_none_when_no_metrics(self):
        comp = _make_component()
        comp["metrics"] = {}
        result = oldest_metric_fetched_at(comp)
        assert result is None

    def test_ignores_buckets_without_fetched_at(self):
        now = datetime.now(UTC)
        ts = _iso(now - timedelta(hours=5))
        comp = _make_component(
            gh_fetched_at=None,
            pypi_fetched_at=ts,
            pypistats_fetched_at=None,
        )
        result = oldest_metric_fetched_at(comp)
        assert result == ts


# ---------------------------------------------------------------------------
# Tests: compute_ranking.py — stale ranking behavior
# ---------------------------------------------------------------------------


class TestComputeRanking:
    """Tests for compute_ranking stale-aware behavior."""

    def test_fresh_metrics_ranking_not_stale(self, tmp_path):
        """Ranking on fresh metrics should have isStale=false."""
        from compute_ranking import compute_rankings

        now = datetime.now(UTC)
        comp = _make_component(
            gh_fetched_at=_iso(now - timedelta(hours=1)),
            gh_is_stale=False,
            pypi_fetched_at=_iso(now - timedelta(hours=1)),
            pypi_is_stale=False,
            pypistats_fetched_at=_iso(now - timedelta(hours=1)),
            pypistats_is_stale=False,
        )

        catalog = {
            "generatedAt": _iso(now),
            "schemaVersion": 1,
            "categories": ["All", "Widgets"],
            "components": [comp],
        }
        in_path = tmp_path / "components.json"
        cfg_path = tmp_path / "ranking_config.json"
        dump_json(in_path, catalog)
        dump_json(cfg_path, _make_ranking_config())

        rc = compute_rankings(
            compiled_in=in_path,
            compiled_out=in_path,
            config_path=cfg_path,
            limit=None,
            stale_threshold_hours=48.0,
        )
        assert rc == 0

        result = load_json(in_path)
        ranking = result["components"][0]["ranking"]
        assert ranking["isStale"] is False
        assert ranking["staleBuckets"] == []
        # computedAt should be recent (within last minute)
        computed_at = datetime.fromisoformat(ranking["computedAt"].replace("Z", "+00:00"))
        assert (datetime.now(UTC) - computed_at).total_seconds() < 60

    def test_stale_metrics_ranking_is_stale(self, tmp_path):
        """Ranking on stale metrics should have isStale=true and correct staleBuckets."""
        from compute_ranking import compute_rankings

        now = datetime.now(UTC)
        old_ts = _iso(now - timedelta(hours=100))
        comp = _make_component(
            gh_fetched_at=old_ts,
            gh_is_stale=False,
            pypi_fetched_at=old_ts,
            pypi_is_stale=False,
            pypistats_fetched_at=old_ts,
            pypistats_is_stale=False,
        )

        catalog = {
            "generatedAt": _iso(now),
            "schemaVersion": 1,
            "categories": ["All", "Widgets"],
            "components": [comp],
        }
        in_path = tmp_path / "components.json"
        cfg_path = tmp_path / "ranking_config.json"
        dump_json(in_path, catalog)
        dump_json(cfg_path, _make_ranking_config())

        rc = compute_rankings(
            compiled_in=in_path,
            compiled_out=in_path,
            config_path=cfg_path,
            limit=None,
            stale_threshold_hours=48.0,
        )
        assert rc == 0

        result = load_json(in_path)
        ranking = result["components"][0]["ranking"]
        assert ranking["isStale"] is True
        assert "github" in ranking["staleBuckets"]
        assert "pypi" in ranking["staleBuckets"]
        assert "pypistats" in ranking["staleBuckets"]

        # computedAt should reflect old data age (oldest fetchedAt), not now
        computed_at = ranking["computedAt"]
        assert computed_at == old_ts

    def test_stale_computed_at_reflects_oldest_metric(self, tmp_path):
        """When metrics are stale, computedAt should use the oldest fetchedAt."""
        from compute_ranking import compute_rankings

        now = datetime.now(UTC)
        oldest = _iso(now - timedelta(hours=200))
        middle = _iso(now - timedelta(hours=100))
        newest = _iso(now - timedelta(hours=50))

        comp = _make_component(
            gh_fetched_at=oldest,
            gh_is_stale=False,
            pypi_fetched_at=middle,
            pypi_is_stale=False,
            pypistats_fetched_at=newest,
            pypistats_is_stale=False,
        )

        catalog = {
            "generatedAt": _iso(now),
            "schemaVersion": 1,
            "categories": ["All", "Widgets"],
            "components": [comp],
        }
        in_path = tmp_path / "components.json"
        cfg_path = tmp_path / "ranking_config.json"
        dump_json(in_path, catalog)
        dump_json(cfg_path, _make_ranking_config())

        rc = compute_rankings(
            compiled_in=in_path,
            compiled_out=in_path,
            config_path=cfg_path,
            limit=None,
            stale_threshold_hours=48.0,
        )
        assert rc == 0

        result = load_json(in_path)
        ranking = result["components"][0]["ranking"]
        assert ranking["isStale"] is True
        # computedAt should be the oldest fetchedAt
        assert ranking["computedAt"] == oldest

    def test_force_stale_uses_current_time(self, tmp_path):
        """--force-stale should set computedAt to now even on stale data."""
        from compute_ranking import compute_rankings

        now = datetime.now(UTC)
        old_ts = _iso(now - timedelta(hours=200))
        comp = _make_component(
            gh_fetched_at=old_ts,
            gh_is_stale=False,
            pypi_fetched_at=old_ts,
            pypi_is_stale=False,
            pypistats_fetched_at=old_ts,
            pypistats_is_stale=False,
        )

        catalog = {
            "generatedAt": _iso(now),
            "schemaVersion": 1,
            "categories": ["All", "Widgets"],
            "components": [comp],
        }
        in_path = tmp_path / "components.json"
        cfg_path = tmp_path / "ranking_config.json"
        dump_json(in_path, catalog)
        dump_json(cfg_path, _make_ranking_config())

        rc = compute_rankings(
            compiled_in=in_path,
            compiled_out=in_path,
            config_path=cfg_path,
            limit=None,
            stale_threshold_hours=48.0,
            force_stale=True,
        )
        assert rc == 0

        result = load_json(in_path)
        ranking = result["components"][0]["ranking"]
        assert ranking["isStale"] is True  # still flagged as stale
        # But computedAt should be close to now
        computed_at = datetime.fromisoformat(ranking["computedAt"].replace("Z", "+00:00"))
        assert (datetime.now(UTC) - computed_at).total_seconds() < 60

    def test_partial_stale_only_stale_buckets_listed(self, tmp_path):
        """When only some buckets are stale, staleBuckets should list only those."""
        from compute_ranking import compute_rankings

        now = datetime.now(UTC)
        old_ts = _iso(now - timedelta(hours=200))
        new_ts = _iso(now - timedelta(hours=1))

        comp = _make_component(
            gh_fetched_at=old_ts,  # stale
            gh_is_stale=False,
            pypi_fetched_at=new_ts,  # fresh
            pypi_is_stale=False,
            pypistats_fetched_at=new_ts,  # fresh
            pypistats_is_stale=False,
        )

        catalog = {
            "generatedAt": _iso(now),
            "schemaVersion": 1,
            "categories": ["All", "Widgets"],
            "components": [comp],
        }
        in_path = tmp_path / "components.json"
        cfg_path = tmp_path / "ranking_config.json"
        dump_json(in_path, catalog)
        dump_json(cfg_path, _make_ranking_config())

        rc = compute_rankings(
            compiled_in=in_path,
            compiled_out=in_path,
            config_path=cfg_path,
            limit=None,
            stale_threshold_hours=48.0,
        )
        assert rc == 0

        result = load_json(in_path)
        ranking = result["components"][0]["ranking"]
        assert ranking["isStale"] is True
        assert ranking["staleBuckets"] == ["github"]

    def test_score_computed_regardless_of_staleness(self, tmp_path):
        """Scores should be computed even on stale data (just flagged)."""
        from compute_ranking import compute_rankings

        now = datetime.now(UTC)
        old_ts = _iso(now - timedelta(hours=200))
        comp = _make_component(
            gh_stars=1000,
            gh_fetched_at=old_ts,
            gh_is_stale=False,
            pypi_fetched_at=old_ts,
            pypi_is_stale=False,
            pypistats_fetched_at=old_ts,
            pypistats_is_stale=False,
        )

        catalog = {
            "generatedAt": _iso(now),
            "schemaVersion": 1,
            "categories": ["All", "Widgets"],
            "components": [comp],
        }
        in_path = tmp_path / "components.json"
        cfg_path = tmp_path / "ranking_config.json"
        dump_json(in_path, catalog)
        dump_json(cfg_path, _make_ranking_config())

        rc = compute_rankings(
            compiled_in=in_path,
            compiled_out=in_path,
            config_path=cfg_path,
            limit=None,
            stale_threshold_hours=48.0,
        )
        assert rc == 0

        result = load_json(in_path)
        ranking = result["components"][0]["ranking"]
        assert ranking["score"] > 0
        assert ranking["signals"]["starsScore"] > 0


# ---------------------------------------------------------------------------
# Tests: build_catalog.py — pipeline mode metadata
# ---------------------------------------------------------------------------


class TestBuildCatalogPipelineMode:
    """Tests for build_catalog pipeline mode handling."""

    def _make_source_component(self, tmp_path: Path, name: str = "test-comp") -> Path:
        """Create a minimal source component JSON file."""
        comp = {
            "schemaVersion": 1,
            "title": "Test Component",
            "author": {"github": "testuser"},
            "links": {"github": "https://github.com/testuser/test-repo"},
            "governance": {"enabled": True},
            "categories": ["Widgets"],
        }
        comp_dir = tmp_path / "components"
        comp_dir.mkdir(exist_ok=True)
        path = comp_dir / f"{name}.json"
        dump_json(path, comp)
        return path

    def _make_schema(self, registry_root: Path) -> None:
        """Create a minimal component schema for build_catalog."""
        schema_dir = registry_root / "schemas"
        schema_dir.mkdir(exist_ok=True)
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": ["schemaVersion", "title", "author", "links", "governance", "categories"],
            "properties": {
                "schemaVersion": {"type": "integer", "const": 1},
                "title": {"type": "string", "minLength": 1, "maxLength": 80},
                "author": {
                    "type": "object",
                    "required": ["github"],
                    "properties": {
                        "github": {"type": "string", "pattern": "^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?$"},
                        "displayName": {"type": "string"},
                    },
                },
                "links": {
                    "type": "object",
                    "required": ["github"],
                    "properties": {
                        "github": {"type": "string"},
                        "pypi": {"type": ["string", "null"]},
                        "demo": {"type": ["string", "null"]},
                        "docs": {"type": ["string", "null"]},
                    },
                },
                "media": {
                    "type": "object",
                    "properties": {
                        "image": {"type": ["string", "null"]},
                    },
                },
                "install": {
                    "type": "object",
                    "properties": {
                        "pip": {"type": ["string", "null"]},
                    },
                },
                "governance": {
                    "type": "object",
                    "required": ["enabled"],
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "notes": {"type": ["string", "null"]},
                    },
                },
                "categories": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "enum": ["Widgets", "Charts", "LLMs"]},
                },
            },
        }
        dump_json(schema_dir / "component.schema.json", schema)

    def test_full_mode_outputs_pipeline_mode_full(self, tmp_path):
        """Full mode should output pipelineMode='full' and no enrichmentSkipped."""
        from build_catalog import build_catalog

        registry_root = tmp_path / "components" / "registry"
        registry_root.mkdir(parents=True)
        self._make_schema(registry_root)

        components_dir = registry_root / "components"
        self._make_source_component(tmp_path / "components" / "registry")

        compiled, errors = build_catalog(
            repo_root=tmp_path,
            registry_root=registry_root,
            out_path=registry_root / "compiled" / "components.json",
            components_dir=components_dir,
            previous_path=None,
            skip_invalid=False,
            pipeline_mode="full",
        )

        assert not errors
        assert compiled["pipelineMode"] == "full"
        assert "enrichmentSkipped" not in compiled

    def test_offline_mode_outputs_pipeline_mode_offline(self, tmp_path):
        """Offline mode should output pipelineMode='offline' and enrichmentSkipped=true."""
        from build_catalog import build_catalog

        registry_root = tmp_path / "components" / "registry"
        registry_root.mkdir(parents=True)
        self._make_schema(registry_root)

        components_dir = registry_root / "components"
        self._make_source_component(tmp_path / "components" / "registry")

        compiled, errors = build_catalog(
            repo_root=tmp_path,
            registry_root=registry_root,
            out_path=registry_root / "compiled" / "components.json",
            components_dir=components_dir,
            previous_path=None,
            skip_invalid=False,
            pipeline_mode="offline",
        )

        assert not errors
        assert compiled["pipelineMode"] == "offline"
        assert compiled["enrichmentSkipped"] is True

    def test_offline_mode_carries_forward_generated_at(self, tmp_path):
        """Offline mode with carry_forward should preserve the previous generatedAt."""
        from build_catalog import build_catalog

        registry_root = tmp_path / "components" / "registry"
        registry_root.mkdir(parents=True)
        self._make_schema(registry_root)

        components_dir = registry_root / "components"
        self._make_source_component(tmp_path / "components" / "registry")

        # Create a "previous" compiled artifact with a known generatedAt.
        old_generated_at = "2026-01-15T12:00:00Z"
        previous = {
            "generatedAt": old_generated_at,
            "schemaVersion": 1,
            "categories": ["All", "Widgets"],
            "components": [],
        }
        prev_path = registry_root / "compiled" / "components.json"
        prev_path.parent.mkdir(parents=True, exist_ok=True)
        dump_json(prev_path, previous)

        compiled, errors = build_catalog(
            repo_root=tmp_path,
            registry_root=registry_root,
            out_path=registry_root / "compiled" / "components.json",
            components_dir=components_dir,
            previous_path=prev_path,
            skip_invalid=False,
            pipeline_mode="offline",
            carry_forward_generated_at=True,
        )

        assert not errors
        assert compiled["generatedAt"] == old_generated_at

    def test_full_mode_updates_generated_at(self, tmp_path):
        """Full mode (carry_forward=False) should set generatedAt to current time."""
        from build_catalog import build_catalog

        registry_root = tmp_path / "components" / "registry"
        registry_root.mkdir(parents=True)
        self._make_schema(registry_root)

        components_dir = registry_root / "components"
        self._make_source_component(tmp_path / "components" / "registry")

        # Create a "previous" compiled artifact with an old generatedAt.
        old_generated_at = "2020-01-01T00:00:00Z"
        previous = {
            "generatedAt": old_generated_at,
            "schemaVersion": 1,
            "categories": ["All", "Widgets"],
            "components": [],
        }
        prev_path = registry_root / "compiled" / "components.json"
        prev_path.parent.mkdir(parents=True, exist_ok=True)
        dump_json(prev_path, previous)

        compiled, errors = build_catalog(
            repo_root=tmp_path,
            registry_root=registry_root,
            out_path=registry_root / "compiled" / "components.json",
            components_dir=components_dir,
            previous_path=prev_path,
            skip_invalid=False,
            pipeline_mode="full",
            carry_forward_generated_at=False,  # full mode: don't carry forward
        )

        assert not errors
        # generatedAt should be updated to current time, not the old one
        assert compiled["generatedAt"] != old_generated_at


# ---------------------------------------------------------------------------
# Tests: compiled.schema.json validation
# ---------------------------------------------------------------------------


class TestCompiledSchemaValidation:
    """Verify the compiled schema accepts both full and offline artifacts."""

    def _get_validator(self):
        from jsonschema import Draft202012Validator

        schema_path = _SCRIPTS_DIR.parent / "schemas" / "compiled.schema.json"
        schema = load_json(schema_path)
        return Draft202012Validator(schema)

    def _make_valid_artifact(self, *, pipeline_mode: str = "full") -> dict:
        now = datetime.now(UTC)
        comp = _make_component(
            gh_fetched_at=_iso(now - timedelta(hours=1)),
            gh_is_stale=False,
            pypi_fetched_at=_iso(now - timedelta(hours=1)),
            pypi_is_stale=False,
            pypistats_fetched_at=_iso(now - timedelta(hours=1)),
            pypistats_is_stale=False,
        )
        comp["ranking"] = {
            "score": 3.5,
            "computedAt": _iso(now),
            "isStale": False,
            "staleBuckets": [],
            "signals": {
                "starsScore": 2.0,
                "recencyScore": 0.5,
                "contributorsScore": 0.7,
                "daysSinceUpdate": 30.0,
                "daysSinceGithubPush": 30.0,
                "daysSincePypiRelease": 60.0,
                "downloadsScore": 2.3,
            },
        }
        artifact = {
            "generatedAt": _iso(now),
            "schemaVersion": 1,
            "pipelineMode": pipeline_mode,
            "categories": ["All", "Widgets"],
            "components": [comp],
        }
        if pipeline_mode == "offline":
            artifact["enrichmentSkipped"] = True
        return artifact

    def test_full_mode_artifact_validates(self):
        """A full-mode artifact should pass schema validation."""
        validator = self._get_validator()
        artifact = self._make_valid_artifact(pipeline_mode="full")
        errors = list(validator.iter_errors(artifact))
        assert not errors, f"Validation errors: {[e.message for e in errors]}"

    def test_offline_mode_artifact_validates(self):
        """An offline-mode artifact should pass schema validation."""
        validator = self._get_validator()
        artifact = self._make_valid_artifact(pipeline_mode="offline")
        errors = list(validator.iter_errors(artifact))
        assert not errors, f"Validation errors: {[e.message for e in errors]}"

    def test_stale_ranking_validates(self):
        """A ranking with isStale=true and staleBuckets should validate."""
        validator = self._get_validator()
        artifact = self._make_valid_artifact(pipeline_mode="offline")
        artifact["components"][0]["ranking"]["isStale"] = True
        artifact["components"][0]["ranking"]["staleBuckets"] = ["github", "pypi"]
        errors = list(validator.iter_errors(artifact))
        assert not errors, f"Validation errors: {[e.message for e in errors]}"

    def test_artifact_without_pipeline_mode_validates(self):
        """An artifact without pipelineMode (legacy) should still validate."""
        validator = self._get_validator()
        artifact = self._make_valid_artifact()
        del artifact["pipelineMode"]
        errors = list(validator.iter_errors(artifact))
        assert not errors, f"Validation errors: {[e.message for e in errors]}"

    def test_artifact_without_ranking_stale_fields_validates(self):
        """A ranking without isStale/staleBuckets (legacy) should still validate."""
        validator = self._get_validator()
        artifact = self._make_valid_artifact()
        del artifact["components"][0]["ranking"]["isStale"]
        del artifact["components"][0]["ranking"]["staleBuckets"]
        errors = list(validator.iter_errors(artifact))
        assert not errors, f"Validation errors: {[e.message for e in errors]}"


# ---------------------------------------------------------------------------
# Integration tests: full vs offline pipeline paths
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    """End-to-end tests comparing full vs offline pipeline output."""

    def _run_build_and_rank(
        self, tmp_path: Path, *, pipeline_mode: str, carry_forward: bool = False
    ) -> dict:
        """Run build_catalog + compute_ranking in a temp dir and return the result."""
        from build_catalog import build_catalog
        from compute_ranking import compute_rankings

        registry_root = tmp_path / "components" / "registry"
        registry_root.mkdir(parents=True, exist_ok=True)

        # Schema
        schema_dir = registry_root / "schemas"
        schema_dir.mkdir(exist_ok=True)
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": ["schemaVersion", "title", "author", "links", "governance", "categories"],
            "properties": {
                "schemaVersion": {"type": "integer", "const": 1},
                "title": {"type": "string", "minLength": 1, "maxLength": 80},
                "author": {
                    "type": "object",
                    "required": ["github"],
                    "properties": {
                        "github": {"type": "string", "pattern": "^[a-zA-Z0-9-]+$"},
                        "displayName": {"type": "string"},
                    },
                },
                "links": {
                    "type": "object",
                    "required": ["github"],
                    "properties": {
                        "github": {"type": "string"},
                        "pypi": {"type": ["string", "null"]},
                        "demo": {"type": ["string", "null"]},
                        "docs": {"type": ["string", "null"]},
                    },
                },
                "media": {
                    "type": "object",
                    "properties": {"image": {"type": ["string", "null"]}},
                },
                "install": {
                    "type": "object",
                    "properties": {"pip": {"type": ["string", "null"]}},
                },
                "governance": {
                    "type": "object",
                    "required": ["enabled"],
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "notes": {"type": ["string", "null"]},
                    },
                },
                "categories": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "enum": ["Widgets", "Charts", "LLMs"]},
                },
            },
        }
        dump_json(schema_dir / "component.schema.json", schema)

        # Source component
        comp_dir = registry_root / "components"
        comp_dir.mkdir(exist_ok=True)
        dump_json(
            comp_dir / "test-comp.json",
            {
                "schemaVersion": 1,
                "title": "Test Component",
                "author": {"github": "testuser"},
                "links": {
                    "github": "https://github.com/testuser/test-repo",
                    "pypi": "test-pkg",
                },
                "governance": {"enabled": True},
                "categories": ["Widgets"],
            },
        )

        # Previous compiled artifact with known metrics
        now = datetime.now(UTC)
        old_generated_at = "2026-06-01T00:00:00Z"
        old_fetched_at = _iso(now - timedelta(hours=100))
        previous = {
            "generatedAt": old_generated_at,
            "schemaVersion": 1,
            "pipelineMode": "full",
            "categories": ["All", "Widgets"],
            "components": [
                {
                    "title": "Test Component",
                    "author": "testuser",
                    "pipLink": "pip install test-pkg",
                    "pypi": "test-pkg",
                    "categories": ["Widgets"],
                    "image": None,
                    "gitHubUrl": "https://github.com/testuser/test-repo",
                    "enabled": True,
                    "appUrl": None,
                    "socialUrl": "https://github.com/testuser",
                    "metrics": {
                        "github": {
                            "stars": 100,
                            "forks": 10,
                            "contributorsCount": 5,
                            "openIssues": 3,
                            "lastPushAt": _iso(now - timedelta(days=30)),
                            "fetchedAt": old_fetched_at,
                            "isStale": False,
                        },
                        "pypi": {
                            "latestVersion": "1.0.0",
                            "latestReleaseAt": _iso(now - timedelta(days=60)),
                            "fetchedAt": old_fetched_at,
                            "isStale": False,
                        },
                        "pypistats": {
                            "lastDay": 30,
                            "lastWeek": 250,
                            "lastMonth": 1000,
                            "fetchedAt": old_fetched_at,
                            "isStale": False,
                        },
                    },
                    "ranking": {
                        "score": 5.0,
                        "computedAt": old_fetched_at,
                        "signals": {
                            "starsScore": 2.0,
                            "recencyScore": 0.5,
                            "contributorsScore": 0.7,
                            "daysSinceUpdate": 30.0,
                            "daysSinceGithubPush": 30.0,
                            "daysSincePypiRelease": 60.0,
                            "downloadsScore": 3.0,
                        },
                    },
                }
            ],
        }
        prev_path = registry_root / "compiled" / "components.json"
        prev_path.parent.mkdir(parents=True, exist_ok=True)
        dump_json(prev_path, previous)

        # Ranking config
        cfg_path = registry_root / "ranking_config.json"
        dump_json(cfg_path, _make_ranking_config())

        # Build
        out_path = registry_root / "compiled" / "components.json"
        compiled, errors = build_catalog(
            repo_root=tmp_path,
            registry_root=registry_root,
            out_path=out_path,
            components_dir=comp_dir,
            previous_path=prev_path,
            skip_invalid=False,
            pipeline_mode=pipeline_mode,
            carry_forward_generated_at=(pipeline_mode == "offline"),
        )
        assert not errors
        dump_json(out_path, compiled)

        # Compute ranking
        compute_rankings(
            compiled_in=out_path,
            compiled_out=out_path,
            config_path=cfg_path,
            limit=None,
            stale_threshold_hours=48.0,
        )

        return load_json(out_path)

    def test_offline_mode_preserves_generated_at(self, tmp_path):
        """Offline mode should preserve generatedAt from the previous artifact."""
        result = self._run_build_and_rank(tmp_path, pipeline_mode="offline")
        assert result["generatedAt"] == "2026-06-01T00:00:00Z"
        assert result["pipelineMode"] == "offline"
        assert result["enrichmentSkipped"] is True

    def test_full_mode_updates_generated_at(self, tmp_path):
        """Full mode should update generatedAt to current time."""
        result = self._run_build_and_rank(tmp_path, pipeline_mode="full")
        assert result["generatedAt"] != "2026-06-01T00:00:00Z"
        assert result["pipelineMode"] == "full"
        assert "enrichmentSkipped" not in result

    def test_offline_mode_ranking_marked_stale(self, tmp_path):
        """In offline mode with old metrics, ranking should be marked stale."""
        result = self._run_build_and_rank(tmp_path, pipeline_mode="offline")
        ranking = result["components"][0]["ranking"]
        assert ranking["isStale"] is True
        assert len(ranking["staleBuckets"]) > 0

    def test_offline_mode_computed_at_reflects_data_age(self, tmp_path):
        """In offline mode, ranking.computedAt should not be newer than the data."""
        result = self._run_build_and_rank(tmp_path, pipeline_mode="offline")
        ranking = result["components"][0]["ranking"]
        computed_at = datetime.fromisoformat(ranking["computedAt"].replace("Z", "+00:00"))
        now = datetime.now(UTC)
        # computedAt should be at least 48 hours old (matching our stale threshold)
        age_hours = (now - computed_at).total_seconds() / 3600
        assert age_hours > 48

    def test_offline_scores_are_explainable(self, tmp_path):
        """Offline mode should produce scores with full signal breakdown."""
        result = self._run_build_and_rank(tmp_path, pipeline_mode="offline")
        ranking = result["components"][0]["ranking"]
        signals = ranking["signals"]
        assert signals["starsScore"] is not None
        assert signals["starsScore"] > 0
        assert ranking["score"] > 0

    def test_offline_and_full_produce_different_metadata(self, tmp_path):
        """Full and offline runs should produce distinguishable artifacts."""
        offline_dir = tmp_path / "offline"
        full_dir = tmp_path / "full"
        offline_dir.mkdir()
        full_dir.mkdir()

        offline = self._run_build_and_rank(offline_dir, pipeline_mode="offline")
        full = self._run_build_and_rank(full_dir, pipeline_mode="full")

        assert offline["pipelineMode"] == "offline"
        assert full["pipelineMode"] == "full"
        assert offline.get("enrichmentSkipped") is True
        assert full.get("enrichmentSkipped") is None

        # Offline ranking should be stale, full should not (if metrics were fresh)
        # Note: both use the same carried-forward metrics, so both could be stale
        # The difference is in the metadata, not necessarily the ranking state.
