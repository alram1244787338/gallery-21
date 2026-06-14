"""Utilities for assessing metric freshness and staleness.

Used by compute_ranking.py to decide whether ranking scores should be marked as
stale, and by the pipeline to surface freshness information in the compiled artifact.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .time import parse_iso8601


def _get_nested(obj: Any, *path: str) -> Any:
    cur: Any = obj
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _bucket_age_hours(bucket: dict[str, Any] | None) -> float | None:
    """Return hours since `fetchedAt` in a metric bucket, or None if unparseable."""
    if not isinstance(bucket, dict):
        return None
    fetched_at = bucket.get("fetchedAt")
    if not isinstance(fetched_at, str):
        return None
    dt = parse_iso8601(fetched_at)
    if dt is None:
        return None
    age_s = (datetime.now(UTC) - dt).total_seconds()
    return max(0.0, age_s / 3600.0)


def detect_stale_metrics(
    comp: dict[str, Any],
    *,
    stale_threshold_hours: float,
) -> dict[str, bool]:
    """Return per-bucket staleness flags for a component.

    A bucket is considered stale when:
    - It has no data at all (None or missing)
    - Its `isStale` flag is True
    - Its `fetchedAt` is older than `stale_threshold_hours`

    Parameters
    ----------
    comp
        A compiled component dict.
    stale_threshold_hours
        Hours beyond which a bucket's fetchedAt is considered stale.
        Use 0 or negative to disable age-based staleness (only isStale flag is checked).

    Returns
    -------
    dict
        Keys are "github", "pypi", "pypistats". Values are True if stale.
    """
    metrics = comp.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}

    result: dict[str, bool] = {}
    for bucket_name in ("github", "pypi", "pypistats"):
        bucket = metrics.get(bucket_name)

        # No data at all → stale (but only if the component could have this data,
        # i.e., has a gitHubUrl for github, pypi/pipLink for pypi/pypistats).
        if not isinstance(bucket, dict):
            # Check whether we'd expect this bucket to exist.
            has_key = False
            if bucket_name == "github":
                has_key = bool(comp.get("gitHubUrl"))
            else:
                has_key = bool(comp.get("pypi") or comp.get("pipLink"))
            result[bucket_name] = has_key  # stale only if we expected data
            continue

        # Explicit isStale flag from enricher.
        if bucket.get("isStale") is True:
            result[bucket_name] = True
            continue

        # Age-based staleness.
        if stale_threshold_hours > 0:
            age_h = _bucket_age_hours(bucket)
            if age_h is None:
                # Has bucket but no parseable fetchedAt → treat as stale.
                result[bucket_name] = True
            else:
                result[bucket_name] = age_h > stale_threshold_hours
        else:
            result[bucket_name] = False

    return result


def oldest_metric_fetched_at(comp: dict[str, Any]) -> str | None:
    """Return the oldest `fetchedAt` timestamp across all metric buckets.

    Useful for setting ranking.computedAt to reflect actual data age rather than
    the current wall clock time when operating in offline mode.
    """
    metrics = comp.get("metrics")
    if not isinstance(metrics, dict):
        return None

    timestamps: list[str] = []
    for bucket_name in ("github", "pypi", "pypistats"):
        bucket = metrics.get(bucket_name)
        if isinstance(bucket, dict):
            fetched_at = bucket.get("fetchedAt")
            if isinstance(fetched_at, str) and fetched_at:
                timestamps.append(fetched_at)

    return min(timestamps) if timestamps else None
