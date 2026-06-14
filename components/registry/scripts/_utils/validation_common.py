"""Shared validation primitives for `validate.py` and `build_catalog.py`.

This module centralizes:

- The `ValidationIssue` dataclass (with layer and severity tagging)
- JSONPath formatting for jsonschema error paths
- Missing-required-field extraction from jsonschema errors
- Schema-level validation iteration (jsonschema Draft 2020-12)

Keeping these in one place ensures that `validate.py` (CI / CLI entrypoint)
and `build_catalog.py` (compilation step) produce consistent error messages
with the same structure and granularity.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Issue taxonomy
# ---------------------------------------------------------------------------


class ValidationLayer(str, enum.Enum):
    """Which validation phase produced this issue."""

    SCHEMA = "schema"
    POLICY = "policy"
    COMPILED = "compiled"

    def __str__(self) -> str:  # pragma: no cover - Enum str is used in output
        return self.value


class Severity(str, enum.Enum):
    """How critical this issue is."""

    ERROR = "error"
    WARNING = "warning"

    def __str__(self) -> str:  # pragma: no cover
        return self.value


# Priority for sorting: lower number = shown first.
_LAYER_ORDER: dict[ValidationLayer, int] = {
    ValidationLayer.SCHEMA: 0,
    ValidationLayer.POLICY: 1,
    ValidationLayer.COMPILED: 2,
}

_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.ERROR: 0,
    Severity.WARNING: 1,
}


# ---------------------------------------------------------------------------
# ValidationIssue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation issue tied to a specific JSON file.

    Attributes
    ----------
    file
        Path to the JSON file that triggered this issue.
    schema
        Path to the JSON Schema used for validation.
    message
        Human-readable error/warning description.
    json_path
        JSONPath-like location within the JSON file (e.g. ``links.github``,
        ``media.image``, ``components[0].title``). ``None`` for file-level
        issues (e.g. file extension, file size).
    layer
        Which validation phase produced this issue (schema, policy, or
        compiled).
    severity
        How critical the issue is (error or warning).
    """

    file: Path
    schema: Path
    message: str
    json_path: str | None = None
    layer: ValidationLayer = ValidationLayer.SCHEMA
    severity: Severity = Severity.ERROR

    def sort_key(self) -> tuple[int, int, str, str, str]:
        """Stable sort key: severity → layer → file → json_path → message."""
        return (
            _SEVERITY_ORDER[self.severity],
            _LAYER_ORDER[self.layer],
            str(self.file),
            self.json_path or "",
            self.message,
        )


# ---------------------------------------------------------------------------
# JSONPath formatting
# ---------------------------------------------------------------------------


def format_json_path(parts: Iterable[Any]) -> str:
    """Format a jsonschema error path into a compact JSONPath-ish string.

    Parameters
    ----------
    parts
        Iterable of path parts (strings for object keys, ints for array
        indices), typically from ``jsonschema.ValidationError.path``.

    Returns
    -------
    str
        A compact, human-readable path (e.g. ``$``, ``author.github``,
        ``components[0].title``).
    """
    out: list[str] = []
    for p in parts:
        if isinstance(p, int):
            out.append(f"[{p}]")
        else:
            if out:
                out.append(".")
            out.append(str(p))
    return "".join(out) or "$"


# ---------------------------------------------------------------------------
# Missing-required-field extraction
# ---------------------------------------------------------------------------


def missing_required_fields(err: Any) -> list[str] | None:
    """Extract missing required field names from a jsonschema ``required`` error.

    jsonschema ``required`` errors can be noisy; this extracts the specific
    fields missing at the failing location so output stays readable.

    Parameters
    ----------
    err
        A ``jsonschema.ValidationError`` instance.

    Returns
    -------
    list[str] | None
        List of missing field names if this is a ``required`` error on a dict
        instance; otherwise ``None``.
    """
    if err.validator != "required" or not isinstance(err.validator_value, list):
        return None
    if not isinstance(err.instance, dict):
        return None
    required: list[str] = [str(x) for x in err.validator_value]
    return [k for k in required if k not in err.instance]


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def load_schema(path: Path) -> dict[str, Any]:
    """Load and sanity-check a JSON Schema from disk."""
    # Avoid circular import at module level.
    from _utils.io import load_json

    obj = load_json(path)
    if not isinstance(obj, dict):
        raise TypeError(f"Schema must be a JSON object: {path}")
    return obj


def get_validator(schema_path: Path) -> Any:
    """Create a Draft 2020-12 validator for the given schema path."""
    try:
        from jsonschema import Draft202012Validator  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `jsonschema`.\n\nInstall dependencies with:\n  uv sync --dev"
        ) from e
    schema = load_schema(schema_path)
    return Draft202012Validator(schema)


def validate_schema(
    instance_path: Path,
    schema_path: Path,
    validator: Any,
    *,
    layer: ValidationLayer = ValidationLayer.SCHEMA,
) -> list[ValidationIssue]:
    """Validate one JSON instance file against a JSON Schema.

    Parameters
    ----------
    instance_path
        Path to the JSON file to validate.
    schema_path
        Path to the JSON Schema file.
    validator
        A ``jsonschema`` validator instance (from :func:`get_validator`).
    layer
        Tag to apply to all issues produced by this validation step.

    Returns
    -------
    list[ValidationIssue]
        A de-duplicated list of validation issues. Empty means the file is
        valid.
    """
    from _utils.io import load_json

    instance = load_json(instance_path)
    issues: list[ValidationIssue] = []

    for err in sorted(validator.iter_errors(instance), key=lambda x: list(x.path)):
        json_path = format_json_path(err.path)
        message = err.message

        missing = missing_required_fields(err)
        if missing:
            message = f"Missing required field(s): {', '.join(missing)}"

        issues.append(
            ValidationIssue(
                file=instance_path,
                schema=schema_path,
                message=message,
                json_path=json_path,
                layer=layer,
            )
        )

    # De-dupe identical messages at the same path.
    deduped: list[ValidationIssue] = []
    seen: set[tuple[str, str]] = set()
    for issue in issues:
        key = (issue.json_path or "$", issue.message)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped


def validate_schema_in_memory(
    instance: Any,
    schema: dict[str, Any],
    *,
    source_label: str = "<in-memory>",
    layer: ValidationLayer = ValidationLayer.SCHEMA,
) -> list[ValidationIssue]:
    """Validate an in-memory JSON-like object against a schema.

    Used by ``build_catalog.py`` to validate submissions before compilation.
    """
    try:
        from jsonschema import Draft202012Validator  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `jsonschema`.\n\nInstall dependencies with:\n  uv sync --dev"
        ) from e

    validator = Draft202012Validator(schema)
    issues: list[ValidationIssue] = []
    fake_file = Path(source_label)
    fake_schema = Path("<schema>")

    for err in sorted(validator.iter_errors(instance), key=lambda x: list(x.path)):
        json_path = format_json_path(err.path)
        message = err.message

        missing = missing_required_fields(err)
        if missing:
            message = f"Missing required field(s): {', '.join(missing)}"

        issues.append(
            ValidationIssue(
                file=fake_file,
                schema=fake_schema,
                message=message,
                json_path=json_path,
                layer=layer,
            )
        )
    return issues
