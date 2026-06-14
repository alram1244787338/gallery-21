"""
Compute and persist ranking signals for the compiled component catalog.

This script reads `components/registry/compiled/components.json` and writes a `ranking` block for each
component, following the tech spec's v1 proposal:

- starsScore = log10(stars + 1)
- recencyScore = exp(-days_since_update / half_life_days)
  - days_since_update = min(days_since_github_push, days_since_pypi_release) when both exist

The final score is:
  score = w_stars * starsScore + w_recency * recencyScore

If recency data is missing, the score falls back to the stars-only term.

Run from the repo root (recommended):

    python components/registry/scripts/compute_ranking.py
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _utils.io import dump_json_atomic, load_json
from _utils.time import parse_iso8601, utc_now_iso


@dataclass(frozen=True)
class RankingConfig:
    half_life_days: float
    w_stars: float
    w_recency: float
    w_contributors: float
    w_downloads: float


def _load_ranking_config(path: Path) -> RankingConfig:
    obj = load_json(path)
    if not isinstance(obj, dict):
        raise TypeError(f"Ranking config must be a JSON object: {path}")
    half_life = obj.get("halfLifeDays", 90.0)
    weights = obj.get("weights", {})
    if not isinstance(weights, dict):
        weights = {}
    w_stars = weights.get("stars", 1.0)
    w_recency = weights.get("recency", 2.0)
    w_contributors = weights.get("contributors", 0.0)
    w_downloads = weights.get("downloads", 0.0)

    try:
        half_life_f = float(half_life)
        w_stars_f = float(w_stars)
        w_recency_f = float(w_recency)
        w_contributors_f = float(w_contributors)
        w_downloads_f = float(w_downloads)
    except Exception as e:  # pragma: no cover
        raise TypeError("Ranking config values must be numeric.") from e

    if half_life_f <= 0:
        raise ValueError("halfLifeDays must be > 0.")

    return RankingConfig(
        half_life_days=half_life_f,
        w_stars=w_stars_f,
        w_recency=w_recency_f,
        w_contributors=w_contributors_f,
        w_downloads=w_downloads_f,
    )


def _days_since(dt: datetime, now: datetime) -> float:
    delta_s = (now - dt).total_seconds()
    # If clocks or sources are weird and dt is in the future, clamp to 0.
    if delta_s < 0:
        delta_s = 0.0
    return delta_s / 86400.0


def _get_nested(comp: dict[str, Any], *path: str) -> Any:
    cur: Any = comp
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _stars_for_component(comp: dict[str, Any]) -> int:
    # Prefer nested metrics.github.stars if available.
    s = _get_nested(comp, "metrics", "github", "stars")
    if isinstance(s, int):
        return max(0, s)
    return 0


def _contributors_for_component(comp: dict[str, Any]) -> int | None:
    c = _get_nested(comp, "metrics", "github", "contributorsCount")
    if isinstance(c, int):
        return max(0, c)
    return None


def _downloads_last_month(comp: dict[str, Any]) -> int | None:
    d = _get_nested(comp, "metrics", "pypistats", "lastMonth")
    if isinstance(d, int) and d >= 0:
        return d
    return None


def _recency_days(
    comp: dict[str, Any], now: datetime
) -> tuple[float | None, float | None, float | None]:
    gh_last_push = _get_nested(comp, "metrics", "github", "lastPushAt")
    pypi_latest_release = _get_nested(comp, "metrics", "pypi", "latestReleaseAt")

    gh_dt = parse_iso8601(gh_last_push if isinstance(gh_last_push, str) else None)
    pypi_dt = parse_iso8601(pypi_latest_release if isinstance(pypi_latest_release, str) else None)

    gh_days = _days_since(gh_dt, now) if gh_dt else None
    pypi_days = _days_since(pypi_dt, now) if pypi_dt else None

    days_since_update: float | None
    if gh_days is not None and pypi_days is not None:
        days_since_update = min(gh_days, pypi_days)
    else:
        days_since_update = gh_days if gh_days is not None else pypi_days

    return days_since_update, gh_days, pypi_days


def _compute_ranking(comp: dict[str, Any], *, cfg: RankingConfig, now: datetime) -> dict[str, Any]:
    stars = _stars_for_component(comp)
    stars_score = math.log10(stars + 1)

    contributors = _contributors_for_component(comp)
    contributors_score: float | None = None
    if contributors is not None:
        contributors_score = math.log10(contributors + 1)

    downloads_last_month = _downloads_last_month(comp)
    downloads_score: float | None = None
    if downloads_last_month is not None:
        downloads_score = math.log10(downloads_last_month + 1)

    days_since_update, gh_days, pypi_days = _recency_days(comp, now)

    # Round day-count signals to whole days BEFORE computing derived scores.
    # This ensures that re-running the script within the same day produces
    # byte-identical output.  Without rounding, ``daysSince*`` drifts by the
    # second, which in turn perturbs ``recencyScore`` and ``score``.
    stable_days_since_update = round(days_since_update) if days_since_update is not None else None
    stable_gh_days = round(gh_days) if gh_days is not None else None
    stable_pypi_days = round(pypi_days) if pypi_days is not None else None

    recency_score: float | None = None
    if stable_days_since_update is not None:
        recency_score = math.exp(-stable_days_since_update / cfg.half_life_days)

    score = cfg.w_stars * stars_score
    if recency_score is not None:
        score += cfg.w_recency * recency_score
    if contributors_score is not None:
        score += cfg.w_contributors * contributors_score
    if downloads_score is not None:
        score += cfg.w_downloads * downloads_score

    # Keep ranking explainable and stable.
    return {
        "score": score,
        "signals": {
            "starsScore": stars_score,
            "recencyScore": recency_score,
            "contributorsScore": contributors_score,
            "daysSinceUpdate": stable_days_since_update,
            "daysSinceGithubPush": stable_gh_days,
            "daysSincePypiRelease": stable_pypi_days,
            "downloadsScore": downloads_score,
        },
        "computedAt": utc_now_iso(),
    }


def _ranking_changed(new_ranking: dict[str, Any], old_ranking: Any) -> bool:
    """Return True when *new_ranking* carries materially different values.

    ``computedAt`` is ignored — it is a metadata timestamp, not a signal.
    """
    if not isinstance(old_ranking, dict):
        return True
    if new_ranking.get("score") != old_ranking.get("score"):
        return True
    new_signals = new_ranking.get("signals")
    old_signals = old_ranking.get("signals")
    if not isinstance(new_signals, dict) or not isinstance(old_signals, dict):
        return True
    for key in (
        "starsScore",
        "recencyScore",
        "contributorsScore",
        "downloadsScore",
        "daysSinceUpdate",
        "daysSinceGithubPush",
        "daysSincePypiRelease",
    ):
        if new_signals.get(key) != old_signals.get(key):
            return True
    return False


def compute_rankings(
    *,
    compiled_in: Path,
    compiled_out: Path,
    config_path: Path,
    limit: int | None,
) -> int:
    obj = load_json(compiled_in)
    if not isinstance(obj, dict):
        print(
            f"ERROR: compiled catalog must be a JSON object: {compiled_in}",
            file=sys.stderr,
        )
        return 2

    comps = obj.get("components")
    if not isinstance(comps, list):
        print(
            f"ERROR: compiled catalog missing `components` array: {compiled_in}",
            file=sys.stderr,
        )
        return 2

    cfg = _load_ranking_config(config_path)
    now = datetime.now(UTC)

    processed = 0
    for comp in comps:
        if not isinstance(comp, dict):
            continue
        if limit is not None and processed >= limit:
            break
        processed += 1
        new_ranking = _compute_ranking(comp, cfg=cfg, now=now)
        old_ranking = comp.get("ranking")
        if not _ranking_changed(new_ranking, old_ranking):
            # Preserve the previous computedAt when the signals are identical
            # so pure reruns don't dirty the diff.
            new_ranking["computedAt"] = old_ranking["computedAt"]
        comp["ranking"] = new_ranking

    dump_json_atomic(compiled_out, obj)
    print(f"Wrote rankings for {processed} component(s) to {compiled_out}.")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Compute ranking fields for components/registry/compiled/components.json."
    )
    parser.add_argument(
        "--in",
        dest="compiled_in",
        default=None,
        help=(
            "Input compiled catalog path (default: components/registry/compiled/components.json)."
        ),
    )
    parser.add_argument(
        "--out",
        dest="compiled_out",
        default=None,
        help="Output compiled catalog path (default: overwrite --in).",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help=("Ranking config path (default: components/registry/ranking_config.json)."),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N components (debug).",
    )
    args = parser.parse_args(argv)

    script_path = Path(__file__).resolve()
    registry_root = script_path.parents[1]  # components/registry

    compiled_in = (
        Path(args.compiled_in)
        if args.compiled_in
        else (registry_root / "compiled" / "components.json")
    )
    compiled_out = Path(args.compiled_out) if args.compiled_out else compiled_in
    config_path = (
        Path(args.config_path) if args.config_path else (registry_root / "ranking_config.json")
    )

    if not compiled_in.is_file():
        print(f"ERROR: Missing compiled catalog: {compiled_in}", file=sys.stderr)
        return 2
    if not config_path.is_file():
        print(f"ERROR: Missing ranking config: {config_path}", file=sys.stderr)
        return 2

    return compute_rankings(
        compiled_in=compiled_in,
        compiled_out=compiled_out,
        config_path=config_path,
        limit=args.limit,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
