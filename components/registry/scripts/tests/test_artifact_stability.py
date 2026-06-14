"""Stability / determinism tests for the compiled catalog pipeline.

These tests pin the behavior that fixes noisy "weekly refresh" diffs:

- Re-running the build/ranking with no input change must produce a byte-identical
  artifact (no churn of timestamps or ranking signals).
- A real change (component content, metrics, or config) must still flow through and
  advance the relevant change-marker timestamp.

Run from the repo root with the project's Python (3.12+):

    python -m unittest discover -s components/registry/scripts -p "test_*.py"
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

# The pipeline scripts use absolute imports rooted at the `scripts/` directory
# (e.g. `from _utils.io import ...`), so make that importable.
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_catalog  # noqa: E402
import compute_ranking  # noqa: E402
from _utils.io import dump_json_atomic, load_json  # noqa: E402
from _utils.time import stable_timestamp  # noqa: E402

REGISTRY_ROOT = SCRIPTS_DIR.parents[0]  # components/registry
REPO_ROOT = SCRIPTS_DIR.parents[2]  # repo root

try:
    import jsonschema  # type: ignore  # noqa: F401

    HAS_JSONSCHEMA = True
except Exception:
    HAS_JSONSCHEMA = False


def _component(
    *,
    github_url: str,
    title: str,
    stars: int,
    last_push_at: str | None,
    gh_fetched_at: str | None,
    latest_release_at: str | None = None,
    pypi_fetched_at: str | None = None,
    last_month: int | None = None,
    pypistats_fetched_at: str | None = None,
) -> dict:
    """Build a compiled-shape component dict for ranking tests."""
    return {
        "title": title,
        "author": "tester",
        "pipLink": None,
        "pypi": None,
        "categories": ["Widgets"],
        "image": None,
        "gitHubUrl": github_url,
        "enabled": True,
        "appUrl": None,
        "socialUrl": "https://github.com/tester",
        "metrics": {
            "github": {
                "stars": stars,
                "forks": None,
                "openIssues": None,
                "contributorsCount": None,
                "lastPushAt": last_push_at,
                "fetchedAt": gh_fetched_at,
                "isStale": False,
            },
            "pypi": (
                {
                    "latestVersion": "1.0.0",
                    "latestReleaseAt": latest_release_at,
                    "fetchedAt": pypi_fetched_at,
                    "isStale": False,
                }
                if latest_release_at is not None
                else None
            ),
            "pypistats": (
                {
                    "lastDay": None,
                    "lastWeek": None,
                    "lastMonth": last_month,
                    "fetchedAt": pypistats_fetched_at,
                    "isStale": False,
                }
                if last_month is not None
                else None
            ),
        },
        "ranking": None,
    }


def _ranking_config() -> compute_ranking.RankingConfig:
    return compute_ranking.RankingConfig(
        half_life_days=90.0,
        w_stars=1.0,
        w_recency=2.0,
        w_contributors=0.5,
        w_downloads=0.35,
    )


class TestStableTimestamp(unittest.TestCase):
    def test_reuses_previous_when_content_unchanged(self) -> None:
        prev = {"ts": "2020-01-01T00:00:00Z", "value": {"a": [1, 2], "b": None}}
        new = {"ts": "2099-12-31T23:59:59Z", "value": {"a": [1, 2], "b": None}}
        out = stable_timestamp(new, prev, timestamp_key="ts", now="2099-12-31T23:59:59Z")
        self.assertEqual(out, "2020-01-01T00:00:00Z")

    def test_advances_when_content_changes(self) -> None:
        prev = {"ts": "2020-01-01T00:00:00Z", "value": 1}
        new = {"ts": "ignored", "value": 2}
        out = stable_timestamp(new, prev, timestamp_key="ts", now="2099-12-31T23:59:59Z")
        self.assertEqual(out, "2099-12-31T23:59:59Z")

    def test_no_previous_returns_now(self) -> None:
        new = {"ts": "ignored", "value": 1}
        self.assertEqual(
            stable_timestamp(new, None, timestamp_key="ts", now="2099-01-01T00:00:00Z"),
            "2099-01-01T00:00:00Z",
        )

    def test_previous_missing_timestamp_returns_now(self) -> None:
        new = {"ts": "ignored", "value": 1}
        prev = {"value": 1}  # equal content but no prior timestamp to carry
        self.assertEqual(
            stable_timestamp(new, prev, timestamp_key="ts", now="2099-01-01T00:00:00Z"),
            "2099-01-01T00:00:00Z",
        )


class TestRankingDeterminism(unittest.TestCase):
    def test_recency_uses_fetched_at_not_now(self) -> None:
        comp = _component(
            github_url="https://github.com/tester/repo",
            title="Repo",
            stars=35,
            last_push_at="2026-01-01T00:00:00Z",
            gh_fetched_at="2026-06-01T00:00:00Z",
            latest_release_at="2026-05-01T00:00:00Z",
            pypi_fetched_at="2026-06-01T00:00:00Z",
        )
        cfg = _ranking_config()

        # Two very different wall-clock "now" values...
        r1 = compute_ranking._compute_ranking(
            comp, cfg=cfg, now=datetime(2026, 6, 1, tzinfo=timezone.utc), now_iso="A"
        )
        r2 = compute_ranking._compute_ranking(
            comp, cfg=cfg, now=datetime(2031, 6, 1, tzinfo=timezone.utc), now_iso="B"
        )

        # ...must yield identical signals + score (recency is anchored to fetchedAt).
        self.assertEqual(r1["signals"], r2["signals"])
        self.assertEqual(r1["score"], r2["score"])
        # Sanity: exact day deltas relative to fetchedAt.
        self.assertEqual(r1["signals"]["daysSinceGithubPush"], 151.0)
        self.assertEqual(r1["signals"]["daysSincePypiRelease"], 31.0)
        self.assertEqual(r1["signals"]["daysSinceUpdate"], 31.0)

    def test_falls_back_to_now_when_no_fetched_at(self) -> None:
        comp = _component(
            github_url="https://github.com/tester/repo",
            title="Repo",
            stars=10,
            last_push_at="2026-01-01T00:00:00Z",
            gh_fetched_at=None,  # no observation time -> fallback to now
        )
        cfg = _ranking_config()
        r1 = compute_ranking._compute_ranking(
            comp, cfg=cfg, now=datetime(2026, 6, 1, tzinfo=timezone.utc), now_iso="A"
        )
        r2 = compute_ranking._compute_ranking(
            comp, cfg=cfg, now=datetime(2027, 6, 1, tzinfo=timezone.utc), now_iso="B"
        )
        self.assertNotEqual(
            r1["signals"]["daysSinceGithubPush"], r2["signals"]["daysSinceGithubPush"]
        )


class TestRankingIdempotency(unittest.TestCase):
    def _write_catalog(self, path: Path, components: list[dict]) -> None:
        dump_json_atomic(
            path,
            {
                "generatedAt": "2026-06-01T00:00:00Z",
                "schemaVersion": 1,
                "categories": ["All", "Widgets"],
                "components": components,
            },
        )

    def _config_path(self, tmp: Path) -> Path:
        cfg_path = tmp / "ranking_config.json"
        dump_json_atomic(
            cfg_path,
            {
                "schemaVersion": 1,
                "halfLifeDays": 90.0,
                "weights": {"stars": 1.0, "recency": 2.0, "contributors": 0.5, "downloads": 0.35},
            },
        )
        return cfg_path

    def test_rerun_without_change_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            catalog = tmp / "components.json"
            cfg = self._config_path(tmp)
            comps = [
                _component(
                    github_url="https://github.com/tester/active",
                    title="Active",
                    stars=120,
                    last_push_at="2026-05-20T00:00:00Z",
                    gh_fetched_at="2026-06-01T00:00:00Z",
                    latest_release_at="2026-05-25T00:00:00Z",
                    pypi_fetched_at="2026-06-01T00:00:00Z",
                    last_month=5000,
                    pypistats_fetched_at="2026-06-01T00:00:00Z",
                ),
                _component(
                    github_url="https://github.com/tester/dormant",
                    title="Dormant",
                    stars=3,
                    last_push_at="2018-01-01T00:00:00Z",
                    gh_fetched_at="2026-06-01T00:00:00Z",
                ),
            ]
            self._write_catalog(catalog, comps)

            rc = compute_ranking.compute_rankings(
                compiled_in=catalog, compiled_out=catalog, config_path=cfg, limit=None
            )
            self.assertEqual(rc, 0)
            first = catalog.read_text(encoding="utf-8")

            rc = compute_ranking.compute_rankings(
                compiled_in=catalog, compiled_out=catalog, config_path=cfg, limit=None
            )
            self.assertEqual(rc, 0)
            second = catalog.read_text(encoding="utf-8")

            self.assertEqual(first, second, "ranking rerun must not change the artifact")

    def test_real_metric_change_refreshes_only_changed_component(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            catalog = tmp / "components.json"
            cfg = self._config_path(tmp)
            comps = [
                _component(
                    github_url="https://github.com/tester/active",
                    title="Active",
                    stars=120,
                    last_push_at="2026-05-20T00:00:00Z",
                    gh_fetched_at="2026-06-01T00:00:00Z",
                ),
                _component(
                    github_url="https://github.com/tester/dormant",
                    title="Dormant",
                    stars=3,
                    last_push_at="2018-01-01T00:00:00Z",
                    gh_fetched_at="2026-06-01T00:00:00Z",
                ),
            ]
            self._write_catalog(catalog, comps)
            compute_ranking.compute_rankings(
                compiled_in=catalog, compiled_out=catalog, config_path=cfg, limit=None
            )

            before = load_json(catalog)
            by_url = {c["gitHubUrl"]: c for c in before["components"]}
            active_before = by_url["https://github.com/tester/active"]["ranking"]
            dormant_before = by_url["https://github.com/tester/dormant"]["ranking"]

            # Mutate only the "active" component's stars (a real metric change).
            obj = load_json(catalog)
            for c in obj["components"]:
                if c["gitHubUrl"].endswith("/active"):
                    c["metrics"]["github"]["stars"] = 999
            dump_json_atomic(catalog, obj)

            compute_ranking.compute_rankings(
                compiled_in=catalog, compiled_out=catalog, config_path=cfg, limit=None
            )

            after = load_json(catalog)
            by_url_after = {c["gitHubUrl"]: c for c in after["components"]}
            active_after = by_url_after["https://github.com/tester/active"]["ranking"]
            dormant_after = by_url_after["https://github.com/tester/dormant"]["ranking"]

            # Changed component: score updated and timestamp advanced.
            self.assertNotEqual(active_before["score"], active_after["score"])
            self.assertNotEqual(active_before["computedAt"], active_after["computedAt"])
            # Untouched component: ranking block (incl. computedAt) is unchanged.
            self.assertEqual(dormant_before, dormant_after)


@unittest.skipUnless(HAS_JSONSCHEMA, "jsonschema not installed")
class TestBuildCatalogStability(unittest.TestCase):
    def _write_component(self, components_dir: Path, title: str) -> None:
        components_dir.mkdir(parents=True, exist_ok=True)
        (components_dir / "sample.json").write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "title": title,
                    "author": {"github": "octocat"},
                    "links": {"github": "https://github.com/octocat/hello-world"},
                    "governance": {"enabled": True},
                    "categories": ["Widgets"],
                }
            ),
            encoding="utf-8",
        )

    def _build(self, components_dir: Path, out_path: Path, previous: Path | None) -> dict:
        compiled, errors = build_catalog.build_catalog(
            repo_root=REPO_ROOT,
            registry_root=REGISTRY_ROOT,
            out_path=out_path,
            components_dir=components_dir,
            previous_path=previous,
            skip_invalid=False,
        )
        self.assertEqual(errors, [], f"unexpected build errors: {errors}")
        return compiled

    def test_rebuild_without_change_keeps_generated_at(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            components_dir = tmp / "components"
            out_path = tmp / "components.json"
            self._write_component(components_dir, "Sample")

            compiled1 = self._build(components_dir, out_path, previous=None)
            dump_json_atomic(out_path, compiled1)

            compiled2 = self._build(components_dir, out_path, previous=out_path)
            self.assertEqual(
                compiled1, compiled2, "pure rebuild must produce an identical artifact"
            )

    def test_source_change_advances_generated_at(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            components_dir = tmp / "components"
            out_path = tmp / "components.json"
            self._write_component(components_dir, "Sample")

            compiled1 = self._build(components_dir, out_path, previous=None)
            dump_json_atomic(out_path, compiled1)

            # Real source change: rename the component title.
            self._write_component(components_dir, "Renamed Sample")
            compiled3 = self._build(components_dir, out_path, previous=out_path)

            self.assertNotEqual(compiled1["generatedAt"], compiled3["generatedAt"])
            self.assertEqual(compiled3["components"][0]["title"], "Renamed Sample")


if __name__ == "__main__":
    unittest.main()
