"""Reusable URL policy predicates shared across scripts.

This is the single home for the URL/link rules the registry enforces beyond JSON
Schema. Keeping the predicates here (instead of re-deriving them in `validate.py`
and `enrich_images.py`) means the policy is defined exactly once and stays in
sync. Callers are responsible for turning a ``True`` result into a user-facing
message, so this module has no opinion on output formatting.

The image-specific constants (disallowed hosts / signed-query keys) continue to
live in `image_url_policy.py`; this module imports and applies them.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlparse

from _utils.image_url_policy import DISALLOWED_IMAGE_HOSTS, DISALLOWED_IMAGE_QUERY_KEYS

# Schemes that must never appear in a submitted URL, even if a schema is relaxed.
# These are the classic XSS / local-file vectors.
DISALLOWED_URL_SCHEMES = frozenset({"javascript", "data", "file"})


def is_https_url(url: str) -> bool:
    """Return True if ``url`` is a well-formed ``https://`` URL with a host."""
    parsed = urlparse(url)
    return parsed.scheme == "https" and bool(parsed.netloc)


def has_disallowed_scheme(url: str) -> bool:
    """Return True if ``url`` uses a disallowed scheme (javascript/data/file)."""
    return urlparse(url).scheme.lower() in DISALLOWED_URL_SCHEMES


def is_allowed_https(url: str) -> bool:
    """Return True only if ``url`` is https and not a disallowed scheme.

    Convenience for the common ``not is_https_url(x) or has_disallowed_scheme(x)``
    check used for every accepted URL field.
    """
    return is_https_url(url) and not has_disallowed_scheme(url)


def image_host_disallowed(url: str) -> bool:
    """Return True if the URL host is a disallowed (brittle proxy) image host."""
    host = (urlparse(url).netloc or "").lower()
    return host in DISALLOWED_IMAGE_HOSTS


def image_has_signed_query(url: str) -> bool:
    """Return True if the URL carries signed/expiring query params (S3/GCS/CF)."""
    for key, _ in parse_qsl(urlparse(url).query, keep_blank_values=True):
        if key.strip().lower() in DISALLOWED_IMAGE_QUERY_KEYS:
            return True
    return False
