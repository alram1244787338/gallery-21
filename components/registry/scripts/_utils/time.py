from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def utc_now_iso() -> str:
    """UTC now in ISO8601 with Z suffix (e.g. 2025-12-19T00:00:00Z)."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def stable_timestamp(
    new_obj: Any,
    previous_obj: Any,
    *,
    timestamp_key: str,
    now: str | None = None,
) -> str:
    """Return a timestamp for ``new_obj[timestamp_key]`` that only advances on change.

    Compiled artifacts in this repo are regenerated on a schedule even when nothing
    meaningful changed. To avoid noisy diffs (and PRs that look like a full rebuild),
    a timestamp should be treated as a *change marker*: it must stay byte-stable when
    the surrounding content is unchanged, and only move forward when it isn't.

    This compares ``new_obj`` against ``previous_obj`` while ignoring ``timestamp_key``
    on both sides. If the rest is structurally equal, the previous timestamp is reused
    so the artifact stays identical across reruns. Otherwise ``now`` (or the current
    UTC time) is returned.

    Returns the current time when there is no usable previous timestamp to carry
    forward (e.g. a brand-new artifact or component).
    """
    now_iso = now if now is not None else utc_now_iso()
    if not isinstance(previous_obj, dict) or not isinstance(new_obj, dict):
        return now_iso
    prev_ts = previous_obj.get(timestamp_key)
    if not isinstance(prev_ts, str):
        return now_iso
    if _without_key(new_obj, timestamp_key) == _without_key(previous_obj, timestamp_key):
        return prev_ts
    return now_iso


def _without_key(obj: dict[str, Any], key: str) -> dict[str, Any]:
    return {k: v for k, v in obj.items() if k != key}


def parse_iso8601(dt: str | None) -> datetime | None:
    """Parse a subset of ISO8601/RFC3339 strings used by GitHub/PyPI.

    Accepts timestamps like:
    - 2025-11-30T12:33:58Z
    - 2025-11-23T22:30:23.036058Z
    - 2025-11-23T22:30:23+00:00

    Returns timezone-aware UTC datetimes when possible.
    """
    if not isinstance(dt, str) or not dt.strip():
        return None
    s = dt.strip()
    # `datetime.fromisoformat` doesn't accept "Z" suffix; normalize it.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(s)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # Assume UTC if tzinfo is missing (shouldn't happen with our sources)
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
