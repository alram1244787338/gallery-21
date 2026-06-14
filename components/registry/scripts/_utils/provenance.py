"""Shared helpers for describing the *freshness* of enrichment data.

These helpers let the pipeline answer a single maintainer-facing question for any
compiled component: "is the data behind this artifact fresh, stale, or simply
missing?". Keeping the logic here (rather than re-deriving it ad hoc in
`compute_ranking.py` / `run_pipeline.py`) ensures every step uses the same
vocabulary, so a `--no-enrich` run and a full run report state the same way.

Vocabulary
----------
A single metrics bucket (``metrics.github`` / ``metrics.pypi`` /
``metrics.pypistats``) is in one of three states:

- ``"live"``    -- it has a ``fetchedAt`` timestamp and is not flagged stale.
- ``"stale"``   -- it is flagged ``isStale`` (a prior fetch failed), so the
                   values are carried-forward-but-known-old.
- ``"missing"`` -- it has never been fetched (no ``fetchedAt``), so there is no
                   data at all.

A *component* basis summarizes the buckets that actually carry data:

- ``"live"``    -- every present bucket is live.
- ``"stale"``   -- at least one present bucket is stale.
- ``"missing"`` -- no bucket carries any data (the ranking score is degenerate).
"""

from __future__ import annotations

from typing import Any

from .time import parse_iso8601

# Buckets that can feed ranking signals. Order is stable for deterministic output.
RANKING_BUCKETS: tuple[str, ...] = ("github", "pypi", "pypistats")

# Per-bucket and per-component freshness states.
STATE_LIVE = "live"
STATE_STALE = "stale"
STATE_MISSING = "missing"

# Overall (pipeline-level) basis can additionally be "mixed".
BASIS_MIXED = "mixed"


def bucket_state(bucket: Any) -> str:
    """Classify a single metrics bucket as live / stale / missing.

    A bucket is ``missing`` if it is absent or never carried a successful fetch
    (no ``fetchedAt``); ``stale`` if it is explicitly flagged ``isStale``; and
    ``live`` otherwise.
    """
    if not isinstance(bucket, dict):
        return STATE_MISSING

    fetched_at = bucket.get("fetchedAt")
    has_fetch = isinstance(fetched_at, str) and bool(fetched_at.strip())
    if not has_fetch:
        return STATE_MISSING

    if bucket.get("isStale") is True:
        return STATE_STALE

    return STATE_LIVE


def component_bucket_states(comp: dict[str, Any]) -> dict[str, str]:
    """Return the freshness state of each ranking bucket for one component."""
    metrics = comp.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    return {name: bucket_state(metrics.get(name)) for name in RANKING_BUCKETS}


def component_basis(comp: dict[str, Any]) -> str:
    """Summarize the freshness of the data backing a component's ranking.

    Conservative on purpose: if *any* present bucket is stale, the whole
    component is reported ``stale`` so a maintainer is never told a score is
    fresh when part of its input is known-old.
    """
    states = component_bucket_states(comp).values()
    present = [s for s in states if s != STATE_MISSING]
    if not present:
        return STATE_MISSING
    if any(s == STATE_STALE for s in present):
        return STATE_STALE
    return STATE_LIVE


def summarize_basis(component_bases: list[str]) -> str:
    """Roll component bases up into one pipeline-level basis.

    Returns the single shared state when every component agrees, ``"missing"``
    for an empty catalog, and ``"mixed"`` when components disagree.
    """
    distinct = set(component_bases)
    if not distinct:
        return STATE_MISSING
    if distinct == {STATE_LIVE}:
        return STATE_LIVE
    if distinct == {STATE_STALE}:
        return STATE_STALE
    if distinct == {STATE_MISSING}:
        return STATE_MISSING
    return BASIS_MIXED


def is_score_degenerate(comp: dict[str, Any]) -> bool:
    """True when a component has no metrics at all behind its ranking.

    Such a component can only score 0 (``log10(0 + 1) == 0``), which looks like a
    real "ranked last" result but is really just absent data. Callers use this to
    avoid overwriting a previously-good ranking during offline runs.
    """
    return component_basis(comp) == STATE_MISSING


def newest_fetched_at(comp: dict[str, Any]) -> str | None:
    """Return the most recent parseable ``fetchedAt`` across ranking buckets.

    Useful for maintainer-facing summaries ("metrics last fetched at ..."). Falls
    back to ``None`` when nothing has ever been fetched.
    """
    metrics = comp.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    best: str | None = None
    for name in RANKING_BUCKETS:
        bucket = metrics.get(name)
        if not isinstance(bucket, dict):
            continue
        raw = bucket.get("fetchedAt")
        if not isinstance(raw, str):
            continue
        if parse_iso8601(raw) is None:
            continue
        if best is None or raw > best:
            best = raw
    return best
