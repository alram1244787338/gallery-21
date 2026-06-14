"""Tests for offline-mode pipeline semantics (`--no-enrich`) vs full mode.

These tests prove the registry pipeline produces a *self-describing* artifact:
whether enrichment ran or was skipped offline, a reader can always tell the
freshness basis of every ranking, and an offline run can never silently overwrite
a previously-good score with a degenerate "no data" one.

Run directly:

    python components/registry/scripts/tests/test_offline_ranking.py

or via discovery:

    python -m unittest discover -s components/registry/scripts/tests
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Make the sibling scripts importable (these are run as loose scripts, not a package).
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import compute_ranking  # noqa: E402  (path set up above)
from _utils import provenance  # noqa: E402
from _utils.enrich import should_refetch  # noqa: E402


def _iso(hours_ago: float) -> str:
    return (datetime.now(UTC) - timedelta(hours=hours_ago)).isoformat().replace("+00:00", "Z")


def _github_bucket(*, stale: bool, hours_ago: float = 1.0, stars: int = 100) -> dict[str, Any]:
    return {
        "stars": stars,
        "forks": 5,
        "contributorsCount": 3,
        "openIssues": 1,
        "lastPushAt": _iso(hours_ago),
        "fetchedAt": _iso(hours_ago),
        "isStale": stale,
    }


def _component(
    *,
    github: dict[str, Any] | None = None,
    ranking: dict[str, Any] | None = None,
) -> dict[str, Any]:
    comp: dict[str, Any] = {
        "title": "Demo",
        "author": "octocat",
        "pipLink": "pip install demo",
        "pypi": "demo",
        "categories": ["Widgets"],
        "image": None,
        "gitHubUrl": "https://github.com/octocat/demo",
        "enabled": True,
        "appUrl": None,
        "socialUrl": "https://github.com/octocat",
        "metrics": {"github": github, "pypi": None, "pypistats": None},
    }
    if ranking is not None:
        comp["ranking"] = ranking
    return comp


class _RankingHarness:
    """Writes a temp compiled artifact + config and runs compute_rankings on it."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.config_path = tmp / "ranking_config.json"
        self.config_path.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "halfLifeDays": 90.0,
                    "weights": {
                        "stars": 1.0,
                        "recency": 2.0,
                        "contributors": 0.5,
                        "downloads": 0.35,
                    },
                }
            ),
            encoding="utf-8",
        )

    def run(
        self,
        components: list[dict[str, Any]],
        *,
        offline: bool,
        services: list[str] | None = None,
        stamp_pipeline: bool = True,
    ) -> dict[str, Any]:
        compiled_in = self.tmp / "components.json"
        obj = {
            "generatedAt": _iso(0),
            "schemaVersion": 1,
            "categories": ["All", "Widgets"],
            "components": components,
        }
        compiled_in.write_text(json.dumps(obj), encoding="utf-8")
        rc = compute_ranking.compute_rankings(
            compiled_in=compiled_in,
            compiled_out=compiled_in,
            config_path=self.config_path,
            limit=None,
            offline=offline,
            services=services or [],
            stamp_pipeline=stamp_pipeline,
        )
        assert rc == 0, f"compute_rankings returned {rc}"
        return json.loads(compiled_in.read_text(encoding="utf-8"))


class ProvenanceHelperTests(unittest.TestCase):
    def test_bucket_state(self) -> None:
        self.assertEqual(provenance.bucket_state(_github_bucket(stale=False)), "live")
        self.assertEqual(provenance.bucket_state(_github_bucket(stale=True)), "stale")
        self.assertEqual(provenance.bucket_state({"stars": 1}), "missing")  # no fetchedAt
        self.assertEqual(provenance.bucket_state(None), "missing")

    def test_component_basis(self) -> None:
        # github live, pypi/pypistats absent -> still "live" (present buckets are fresh).
        self.assertEqual(
            provenance.component_basis(_component(github=_github_bucket(stale=False))),
            "live",
        )
        self.assertEqual(
            provenance.component_basis(_component(github=_github_bucket(stale=True))),
            "stale",
        )
        # No metrics at all -> "missing".
        self.assertEqual(provenance.component_basis(_component(github=None)), "missing")

    def test_summarize_basis(self) -> None:
        self.assertEqual(provenance.summarize_basis(["live", "live"]), "live")
        self.assertEqual(provenance.summarize_basis(["stale", "stale"]), "stale")
        self.assertEqual(provenance.summarize_basis(["missing"]), "missing")
        self.assertEqual(provenance.summarize_basis(["live", "stale"]), "mixed")
        self.assertEqual(provenance.summarize_basis([]), "missing")

    def test_is_score_degenerate(self) -> None:
        self.assertTrue(provenance.is_score_degenerate(_component(github=None)))
        self.assertFalse(
            provenance.is_score_degenerate(_component(github=_github_bucket(stale=False)))
        )


class ShouldRefetchTests(unittest.TestCase):
    def test_force_when_threshold_zero(self) -> None:
        self.assertTrue(
            should_refetch(fetched_at=_iso(1), is_stale=False, refresh_older_than_hours=0)
        )

    def test_refetch_when_stale(self) -> None:
        self.assertTrue(
            should_refetch(fetched_at=_iso(1), is_stale=True, refresh_older_than_hours=24)
        )

    def test_refetch_when_no_timestamp(self) -> None:
        self.assertTrue(
            should_refetch(fetched_at=None, is_stale=False, refresh_older_than_hours=24)
        )

    def test_skip_when_fresh(self) -> None:
        self.assertFalse(
            should_refetch(fetched_at=_iso(1), is_stale=False, refresh_older_than_hours=24)
        )

    def test_refetch_when_old(self) -> None:
        self.assertTrue(
            should_refetch(fetched_at=_iso(48), is_stale=False, refresh_older_than_hours=24)
        )


class FullModeRankingTests(unittest.TestCase):
    def test_live_metrics_produce_live_basis(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = _RankingHarness(Path(d)).run(
                [_component(github=_github_bucket(stale=False))],
                offline=False,
                services=["github", "pypi", "pypistats"],
            )
        comp = out["components"][0]
        self.assertEqual(comp["ranking"]["basis"], "live")
        self.assertFalse(comp["ranking"]["offline"])
        self.assertGreater(comp["ranking"]["score"], 0.0)

        pipeline = out["pipeline"]
        self.assertFalse(pipeline["offline"])
        self.assertTrue(pipeline["enriched"])
        self.assertEqual(pipeline["rankingBasis"], "live")
        self.assertEqual(pipeline["services"], ["github", "pypi", "pypistats"])


class OfflineModeRankingTests(unittest.TestCase):
    def test_offline_stale_metrics_are_marked_but_scored(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = _RankingHarness(Path(d)).run(
                [_component(github=_github_bucket(stale=True))],
                offline=True,
            )
        comp = out["components"][0]
        # Stale values are still real numbers, so the score is computed (not zeroed),
        # but everything is clearly marked offline + stale.
        self.assertEqual(comp["ranking"]["basis"], "stale")
        self.assertTrue(comp["ranking"]["offline"])
        self.assertGreater(comp["ranking"]["score"], 0.0)
        self.assertEqual(out["pipeline"]["rankingBasis"], "stale")
        self.assertTrue(out["pipeline"]["offline"])
        self.assertFalse(out["pipeline"]["enriched"])

    def test_offline_missing_metrics_preserve_prior_good_score(self) -> None:
        prior = {
            "score": 5.0,
            "signals": {"starsScore": 2.0},
            "basis": "live",
            "offline": False,
            "computedAt": _iso(72),
        }
        with tempfile.TemporaryDirectory() as d:
            out = _RankingHarness(Path(d)).run(
                [_component(github=None, ranking=prior)],
                offline=True,
            )
        comp = out["components"][0]
        # Anti-pollution: the no-data component must NOT clobber the prior score with 0.
        self.assertEqual(comp["ranking"]["score"], 5.0)
        self.assertEqual(comp["ranking"]["basis"], "missing")
        self.assertTrue(comp["ranking"]["offline"])

    def test_online_missing_metrics_score_zero(self) -> None:
        # Same input but online: with no anti-pollution guard, a genuine no-data
        # component scores 0 and is marked missing (an honest degenerate result).
        prior = {
            "score": 5.0,
            "signals": {"starsScore": 2.0},
            "basis": "live",
            "offline": False,
            "computedAt": _iso(72),
        }
        with tempfile.TemporaryDirectory() as d:
            out = _RankingHarness(Path(d)).run(
                [_component(github=None, ranking=prior)],
                offline=False,
                services=["github"],
            )
        comp = out["components"][0]
        self.assertEqual(comp["ranking"]["score"], 0.0)
        self.assertEqual(comp["ranking"]["basis"], "missing")
        self.assertFalse(comp["ranking"]["offline"])


class CliContractTests(unittest.TestCase):
    """Exercise the real CLI entrypoint run_pipeline uses for offline runs."""

    def test_cli_offline_flag_stamps_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            harness = _RankingHarness(tmp)  # writes config
            compiled = tmp / "components.json"
            compiled.write_text(
                json.dumps(
                    {
                        "generatedAt": _iso(0),
                        "schemaVersion": 1,
                        "categories": ["All", "Widgets"],
                        "components": [_component(github=_github_bucket(stale=True))],
                    }
                ),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "compute_ranking.py"),
                    "--in",
                    str(compiled),
                    "--out",
                    str(compiled),
                    "--config",
                    str(harness.config_path),
                    "--offline",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            out = json.loads(compiled.read_text(encoding="utf-8"))
        self.assertTrue(out["pipeline"]["offline"])
        self.assertEqual(out["components"][0]["ranking"]["basis"], "stale")
        self.assertIn("OFFLINE", proc.stdout)


if __name__ == "__main__":
    unittest.main()
