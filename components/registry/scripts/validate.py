"""
Validate Component Gallery JSON files.

This script validates three independent layers and labels every problem with the
layer it came from, so a maintainer can tell at a glance *which JSON*, *which
field*, and *which rule layer* failed:

- ``source``   — source-of-truth submissions ``components/registry/components/*.json``
                 validated against ``components/registry/schemas/component.schema.json``.
- ``policy``   — cross-file + lint rules on those same submissions (unique GitHub
                 repo, HTTPS-only URLs, stable image URLs, file-size guardrails,
                 JSON-only directory).
- ``compiled`` — the generated artifact ``components/registry/compiled/components.json``
                 validated against ``components/registry/schemas/compiled.schema.json``
                 (only when ``--compiled`` is passed).

When a single file trips several rules, output is ordered most-critical-first
(structural/schema problems before soft policy nits) and each line is tagged
``[layer:rule]`` so the root cause is obvious without reading to the end.

Run from the repo root (recommended):

    python components/registry/scripts/validate.py
    python components/registry/scripts/validate.py --compiled
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from _utils.github import normalize_github_repo_url
from _utils.io import load_json
from _utils.paths import source_components_dir
from _utils.url_policy import (
    image_has_signed_query,
    image_host_disallowed,
    is_allowed_https,
)

# --- Validation layers -------------------------------------------------------
# The layer answers "which stage of validation produced this?" and is what makes
# source vs. compiled findings unambiguous (the two use different field names, so
# tagging the layer keeps their "vocabularies" from being confused).
LAYER_SOURCE = "source"
LAYER_POLICY = "policy"
LAYER_COMPILED = "compiled"

# Soft, non-failing marker (e.g. compiled artifact absent when not required).
CODE_SKIPPED = "skipped"

# Severity for policy rule codes. Lower number == more critical == printed first.
# Schema/decode issues are handled separately in `_severity_for`.
_POLICY_SEVERITY: dict[str, int] = {
    "bad_extension": 0,  # file can't even be processed as a submission
    "file_unreadable": 0,
    "duplicate_repo": 2,  # collides with another submission -> hard rejection
    "invalid_github_url": 2,
    "url_scheme": 3,  # unsafe / non-https URL
    "image_host": 4,  # brittle image source
    "image_query": 4,
    "image_type": 4,
    "file_too_large": 5,  # abuse guardrail; rare
}
_DEFAULT_POLICY_SEVERITY = 3

# Maps a jsonschema validator keyword to a compact, stable rule code. The same
# vocabulary is used for both the source and compiled schema layers; the issue's
# `layer` is what distinguishes them.
_SCHEMA_CODE: dict[str, str] = {
    "required": "required",
    "additionalProperties": "additional_properties",
    "type": "type",
    "const": "const",
    "enum": "enum",
    "pattern": "pattern",
    "format": "format",
    "minLength": "length",
    "maxLength": "length",
    "minItems": "items",
    "maxItems": "items",
    "uniqueItems": "unique_items",
    "minimum": "range",
    "maximum": "range",
    "exclusiveMinimum": "range",
    "exclusiveMaximum": "range",
    "oneOf": "one_of",
    "anyOf": "any_of",
    "allOf": "all_of",
}


def _severity_for(layer: str, code: str) -> int:
    """Rank an issue by how urgently it should be surfaced (lower == first)."""
    if code == "invalid_json":
        return 0
    if layer == LAYER_POLICY:
        return _POLICY_SEVERITY.get(code, _DEFAULT_POLICY_SEVERITY)
    # Schema validation issues (source or compiled) are structural.
    return 1


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation problem tied to a specific JSON file.

    Attributes
    ----------
    file
        Path to the JSON file the issue belongs to.
    layer
        Which validation stage produced it (``source`` / ``policy`` / ``compiled``).
    code
        Compact, stable rule identifier (e.g. ``required``, ``duplicate_repo``).
    message
        Human-readable explanation.
    json_path
        Location within the document (``$`` for the root, ``links.github`` etc.).
    schema
        Schema the file was validated against, when the issue is a schema error.
    """

    file: Path
    layer: str
    code: str
    message: str
    json_path: str | None = None
    schema: Path | None = None

    @property
    def severity(self) -> int:
        """Priority rank (lower is more critical, surfaced first)."""
        return _severity_for(self.layer, self.code)

    @property
    def location(self) -> str:
        """JSONPath-ish location, defaulting to ``$`` (document root)."""
        return self.json_path or "$"


@dataclass(frozen=True)
class RegistryLayout:
    """Resolved paths for the registry, computed once and shared by all checks."""

    repo_root: Path
    registry_root: Path
    components_dir: Path
    component_schema: Path
    compiled_schema: Path
    compiled_artifact: Path


def _layout(repo_root: Path) -> RegistryLayout:
    """Resolve every registry path from the repo root in one place.

    Centralizing this keeps `validate_components`, `validate_policies`, and
    `validate_compiled` from each re-deriving (and potentially drifting on) the
    schema/components/compiled locations.
    """
    registry_root = repo_root / "components" / "registry"
    return RegistryLayout(
        repo_root=repo_root,
        registry_root=registry_root,
        components_dir=source_components_dir(repo_root),
        component_schema=registry_root / "schemas" / "component.schema.json",
        compiled_schema=registry_root / "schemas" / "compiled.schema.json",
        compiled_artifact=registry_root / "compiled" / "components.json",
    )


def _format_json_path(parts: Iterable[Any]) -> str:
    """Format a jsonschema error path into a compact JSONPath-ish string.

    Parameters
    ----------
    parts
        Iterable of path parts (strings for object keys, ints for array indices),
        typically from `jsonschema.ValidationError.path`.

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


def _load_schema(path: Path) -> dict[str, Any]:
    """Load and sanity-check a JSON Schema from disk.

    Parameters
    ----------
    path
        Path to a JSON Schema file.

    Returns
    -------
    dict[str, Any]
        Parsed schema object.

    Raises
    ------
    TypeError
        If the schema file does not contain a JSON object.
    """
    obj = load_json(path)
    if not isinstance(obj, dict):
        raise TypeError(f"Schema must be a JSON object: {path}")
    return obj


def _get_validator(schema_path: Path) -> Any:
    """Create a Draft2020-12 validator for a schema path (load schema once)."""
    try:
        from jsonschema import Draft202012Validator  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `jsonschema`.\n\nInstall dependencies with:\n  uv sync --dev"
        ) from e
    schema = _load_schema(schema_path)
    return Draft202012Validator(schema)


def _schema_code(keyword: Any) -> str:
    """Map a jsonschema validator keyword to a compact rule code."""
    return _SCHEMA_CODE.get(str(keyword), "schema")


def _missing_required_fields(err: Any) -> list[str] | None:
    """Compute missing required field names for a jsonschema "required" error.

    jsonschema "required" errors can be noisy; this extracts the specific fields
    missing at the failing location so output stays readable.

    Parameters
    ----------
    err
        A `jsonschema.ValidationError` instance (typed as `Any` to keep this
        script dependency-light).

    Returns
    -------
    list[str] | None
        List of missing field names if applicable; otherwise ``None``.
    """
    if err.validator != "required" or not isinstance(err.validator_value, list):
        return None
    if not isinstance(err.instance, dict):
        return None
    # validator_value is the list of required fields for the schema at this path.
    required: list[str] = [str(x) for x in err.validator_value]
    return [k for k in required if k not in err.instance]


def _validate_schema(
    instance_path: Path, schema_path: Path, validator: Any, *, layer: str
) -> list[ValidationIssue]:
    """Validate one JSON instance against a schema, returning labeled issues.

    Used for both the ``source`` and ``compiled`` layers (the caller passes the
    right ``layer``). A malformed JSON file produces a single, clean
    ``invalid_json`` issue with a line/column instead of crashing the run.

    Parameters
    ----------
    instance_path
        Path to the JSON file to validate.
    schema_path
        Path to the JSON Schema file to validate against.
    validator
        A pre-built jsonschema validator for ``schema_path``.
    layer
        Validation layer label for the issues produced here.

    Returns
    -------
    list[ValidationIssue]
        A de-duplicated list of issues for this file (empty means valid).
    """
    try:
        instance = load_json(instance_path)
    except json.JSONDecodeError as e:
        return [
            ValidationIssue(
                file=instance_path,
                layer=layer,
                code="invalid_json",
                message=f"Invalid JSON: {e.msg} (line {e.lineno}, column {e.colno}).",
                json_path="$",
                schema=schema_path,
            )
        ]

    issues: list[ValidationIssue] = []
    seen: set[tuple[str, str, str]] = set()
    for err in sorted(validator.iter_errors(instance), key=lambda x: list(x.path)):
        location = _format_json_path(err.path)
        code = _schema_code(err.validator)
        message = err.message

        missing = _missing_required_fields(err)
        if missing:
            message = f"Missing required field(s): {', '.join(missing)}"

        # De-dupe identical findings (jsonschema can report the same root issue
        # multiple times via combinators).
        key = (location, code, message)
        if key in seen:
            continue
        seen.add(key)
        issues.append(
            ValidationIssue(
                file=instance_path,
                layer=layer,
                code=code,
                message=message,
                json_path=location,
                schema=schema_path,
            )
        )
    return issues


def validate_components(repo_root: Path) -> list[ValidationIssue]:
    """Validate all source submissions under `components/registry/components/`.

    This is purely JSON Schema validation (the ``source`` layer); directory
    hygiene and other lint rules live in `validate_policies`.

    Parameters
    ----------
    repo_root
        Path to the component-gallery repo root.

    Returns
    -------
    list[ValidationIssue]
        Schema validation issues across all source submission `*.json` files.
    """
    layout = _layout(repo_root)
    validator = _get_validator(layout.component_schema)

    issues: list[ValidationIssue] = []
    for json_file in sorted(layout.components_dir.glob("*.json")):
        issues.extend(
            _validate_schema(json_file, layout.component_schema, validator, layer=LAYER_SOURCE)
        )
    return issues


def _url_scheme_issue(file: Path, json_path: str) -> ValidationIssue:
    """Build the standard 'URL must be https / safe scheme' policy issue."""
    return ValidationIssue(
        file=file,
        layer=LAYER_POLICY,
        code="url_scheme",
        message="URL must be https:// and must not use a disallowed scheme (javascript:, data:, file:).",
        json_path=json_path,
    )


def validate_policies(repo_root: Path, *, max_component_bytes: int = 50_000) -> list[ValidationIssue]:
    """Policy/lint checks beyond JSON Schema for source submission `*.json` files.

    Covers the tech spec's CI expectations:

    - JSON-only directory (no stray non-``.json`` files).
    - Unique component identity (unique GitHub owner/repo across submissions).
    - HTTPS-only URLs with no unsafe schemes.
    - Stable preview images (no brittle proxy hosts / signed-expiring URLs).
    - File-size abuse guardrail.

    Parameters
    ----------
    repo_root
        Path to the component-gallery repo root.
    max_component_bytes
        Maximum allowed size for each source submission JSON file.

    Returns
    -------
    list[ValidationIssue]
        Policy issues across all source submission files (all in the ``policy``
        layer).
    """
    layout = _layout(repo_root)
    components_dir = layout.components_dir

    issues: list[ValidationIssue] = []

    # Directory hygiene: every file here must be a `.json` submission.
    for entry in sorted(components_dir.iterdir()):
        if entry.is_file() and entry.suffix != ".json":
            issues.append(
                ValidationIssue(
                    file=entry,
                    layer=LAYER_POLICY,
                    code="bad_extension",
                    message=(
                        "Invalid file extension in components directory. "
                        "Source component files must end with `.json`."
                    ),
                    json_path=None,
                )
            )

    first_by_repo: dict[str, Path] = {}

    for json_file in sorted(components_dir.glob("*.json")):
        # File-size abuse guardrail.
        try:
            size = json_file.stat().st_size
        except OSError as e:  # pragma: no cover
            issues.append(
                ValidationIssue(
                    file=json_file,
                    layer=LAYER_POLICY,
                    code="file_unreadable",
                    message=f"Could not stat file: {e}",
                    json_path=None,
                )
            )
            continue
        if size > max_component_bytes:
            issues.append(
                ValidationIssue(
                    file=json_file,
                    layer=LAYER_POLICY,
                    code="file_too_large",
                    message=(
                        f"File too large ({size} bytes). Max allowed is {max_component_bytes} bytes."
                    ),
                    json_path=None,
                )
            )

        # Best-effort load for lint checks; the source schema layer is
        # responsible for reporting parse/structure problems.
        try:
            obj = load_json(json_file)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        links = obj.get("links")
        if isinstance(links, dict):
            gh = links.get("github")
            if isinstance(gh, str) and gh:
                if not is_allowed_https(gh):
                    issues.append(_url_scheme_issue(json_file, "links.github"))
                else:
                    try:
                        canonical = normalize_github_repo_url(gh)
                        key = urlparse(canonical).path.lower().strip("/")
                    except Exception as e:
                        issues.append(
                            ValidationIssue(
                                file=json_file,
                                layer=LAYER_POLICY,
                                code="invalid_github_url",
                                message=str(e),
                                json_path="links.github",
                            )
                        )
                    else:
                        if key in first_by_repo:
                            issues.append(
                                ValidationIssue(
                                    file=json_file,
                                    layer=LAYER_POLICY,
                                    code="duplicate_repo",
                                    message=(
                                        f"Duplicate component identity: links.github repo `{key}` "
                                        f"already submitted in `{first_by_repo[key].name}`."
                                    ),
                                    json_path="links.github",
                                )
                            )
                        else:
                            first_by_repo[key] = json_file

            for path, val in (
                ("links.demo", links.get("demo")),
                ("links.docs", links.get("docs")),
            ):
                if isinstance(val, str) and not is_allowed_https(val):
                    issues.append(_url_scheme_issue(json_file, path))

        media = obj.get("media")
        if isinstance(media, dict):
            img = media.get("image")
            if img is None:
                # Image is optional; null is allowed.
                pass
            elif isinstance(img, str):
                if not is_allowed_https(img):
                    issues.append(_url_scheme_issue(json_file, "media.image"))
                elif image_host_disallowed(img):
                    issues.append(
                        ValidationIssue(
                            file=json_file,
                            layer=LAYER_POLICY,
                            code="image_host",
                            message=(
                                "Image host is not allowed for `media.image` "
                                "(brittle proxy). Use a stable upstream URL instead."
                            ),
                            json_path="media.image",
                        )
                    )
                elif image_has_signed_query(img):
                    issues.append(
                        ValidationIssue(
                            file=json_file,
                            layer=LAYER_POLICY,
                            code="image_query",
                            message=(
                                "Signed/expiring image URLs are not allowed for `media.image` "
                                "(disallowed query parameters detected)."
                            ),
                            json_path="media.image",
                        )
                    )
            else:
                issues.append(
                    ValidationIssue(
                        file=json_file,
                        layer=LAYER_POLICY,
                        code="image_type",
                        message="`media.image` must be a string URL or null.",
                        json_path="media.image",
                    )
                )

    return issues


def validate_compiled(repo_root: Path) -> list[ValidationIssue]:
    """Validate the compiled catalog artifact (the ``compiled`` layer).

    Parameters
    ----------
    repo_root
        Path to the component-gallery repo root.

    Returns
    -------
    list[ValidationIssue]
        Issues for the compiled artifact. If the artifact is missing, returns a
        single soft ``skipped`` issue (does not fail the run).
    """
    layout = _layout(repo_root)
    if not layout.compiled_artifact.is_file():
        return [
            ValidationIssue(
                file=layout.compiled_artifact,
                layer=LAYER_COMPILED,
                code=CODE_SKIPPED,
                message="Compiled artifact not found (skipping).",
                json_path=None,
                schema=layout.compiled_schema,
            )
        ]
    validator = _get_validator(layout.compiled_schema)
    return _validate_schema(
        layout.compiled_artifact, layout.compiled_schema, validator, layer=LAYER_COMPILED
    )


def _rel(path: Path, repo_root: Path) -> Path:
    """Best-effort path relative to the repo root for compact display."""
    try:
        return path.relative_to(repo_root)
    except (ValueError, TypeError):
        return path


@dataclass(frozen=True)
class Report:
    """Rendered validation report.

    Attributes
    ----------
    exit_code
        ``0`` if there are no hard errors, otherwise ``1``.
    error_text
        Text to emit on stderr when ``exit_code`` is ``1`` (empty otherwise).
    ok_text
        Text to emit on stdout when ``exit_code`` is ``0`` (empty otherwise).
    """

    exit_code: int
    error_text: str
    ok_text: str


def build_report(issues: list[ValidationIssue], repo_root: Path) -> Report:
    """Turn a flat list of issues into prioritized, grouped output.

    Findings are grouped by file; files and the issues within them are ordered
    most-critical-first; and every line is tagged ``[layer:rule]`` so the failing
    JSON, field, and rule layer are obvious without reading to the end.

    Parameters
    ----------
    issues
        All issues collected across the requested layers.
    repo_root
        Repo root, used to render compact relative paths.

    Returns
    -------
    Report
        Exit code plus the text to print.
    """
    hard = [i for i in issues if i.code != CODE_SKIPPED]
    notes = [i for i in issues if i.code == CODE_SKIPPED]

    if not hard:
        lines = ["OK: all validated files passed."]
        for n in notes:
            lines.append(f"NOTE: {_rel(n.file, repo_root)} - {n.message}")
        return Report(exit_code=0, error_text="", ok_text="\n".join(lines) + "\n")

    by_file: dict[Path, list[ValidationIssue]] = defaultdict(list)
    for issue in hard:
        by_file[issue.file].append(issue)

    layer_order = [LAYER_SOURCE, LAYER_POLICY, LAYER_COMPILED]
    layer_counts = Counter(i.layer for i in hard)
    breakdown = ", ".join(
        f"{layer_counts[layer]} {layer}" for layer in layer_order if layer_counts.get(layer)
    )

    out: list[str] = [
        f"Found {len(hard)} validation error(s) across {len(by_file)} file(s) [{breakdown}]:"
    ]

    def file_sort_key(fp: Path) -> tuple[int, str]:
        # Files holding the single most critical issue come first.
        return (min(i.severity for i in by_file[fp]), str(_rel(fp, repo_root)))

    for fp in sorted(by_file, key=file_sort_key):
        file_issues = sorted(
            by_file[fp], key=lambda i: (i.severity, i.location, i.code, i.message)
        )
        rel = _rel(fp, repo_root)
        layers_here = ", ".join(
            layer for layer in layer_order if any(i.layer == layer for i in file_issues)
        )
        out.append("")
        out.append(f"- {rel} ({len(file_issues)} error(s); {layers_here})")
        for schema_path in sorted(
            {i.schema for i in file_issues if i.schema}, key=lambda p: str(p)
        ):
            out.append(f"  schema: {_rel(schema_path, repo_root)}")
        for issue in file_issues:
            out.append(f"  - {issue.location} [{issue.layer}:{issue.code}]: {issue.message}")

    return Report(exit_code=1, error_text="\n".join(out) + "\n", ok_text="")


def main(argv: list[str], *, repo_root: Path | None = None) -> int:
    """CLI entrypoint.

    Parameters
    ----------
    argv
        CLI arguments excluding the program name (i.e., ``sys.argv[1:]``).
    repo_root
        Override the repo root (used by tests). Defaults to the repo inferred
        from this script's location.

    Returns
    -------
    int
        Process exit code:

        - 0: success
        - 1: validation failed
        - 2: configuration error (missing required files/dirs)
    """
    parser = argparse.ArgumentParser(description="Validate Component Gallery JSON files.")
    parser.add_argument(
        "--compiled",
        action="store_true",
        help=(
            "Also validate components/registry/compiled/components.json against "
            "components/registry/schemas/compiled.schema.json."
        ),
    )
    parser.add_argument(
        "--no-policy",
        action="store_true",
        help="Disable policy/lint checks beyond schema validation.",
    )
    parser.add_argument(
        "--max-component-bytes",
        type=int,
        default=50_000,
        help=("Max allowed size for each source submission JSON file (default: 50000)."),
    )
    args = parser.parse_args(argv)

    if repo_root is None:
        script_path = Path(__file__).resolve()
        registry_root = script_path.parents[1]  # components/registry
        repo_root = registry_root.parents[1]  # repo root

    layout = _layout(repo_root)

    # Guardrails for common mistakes (configuration errors -> exit code 2).
    if not layout.component_schema.is_file():
        print(
            "ERROR: Missing schema: components/registry/schemas/component.schema.json",
            file=sys.stderr,
        )
        return 2
    if not layout.components_dir.is_dir():
        print(
            "ERROR: Missing source components directory: components/registry/components/",
            file=sys.stderr,
        )
        return 2

    all_issues: list[ValidationIssue] = []
    all_issues.extend(validate_components(repo_root))
    if not args.no_policy:
        all_issues.extend(
            validate_policies(repo_root, max_component_bytes=args.max_component_bytes)
        )
    if args.compiled:
        all_issues.extend(validate_compiled(repo_root))

    report = build_report(all_issues, repo_root)
    if report.exit_code == 0:
        sys.stdout.write(report.ok_text)
    else:
        sys.stderr.write(report.error_text)
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
