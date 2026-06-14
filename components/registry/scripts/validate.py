"""
Validate Component Gallery JSON files.

This script validates:

- Source-of-truth component submissions: `components/registry/components/*.json`
  against `components/registry/schemas/component.schema.json`.
- Optionally, the compiled artifact: `components/registry/compiled/components.json`
  against `components/registry/schemas/compiled.schema.json` (use `--compiled`).

Run from the repo root (recommended):

    python components/registry/scripts/validate.py
    python components/registry/scripts/validate.py --compiled
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, urlparse

from _utils.github import normalize_github_repo_url
from _utils.image_url_policy import DISALLOWED_IMAGE_HOSTS, DISALLOWED_IMAGE_QUERY_KEYS
from _utils.paths import source_components_dir
from _utils.validation_common import (
    _LAYER_ORDER,
    Severity,
    ValidationIssue,
    ValidationLayer,
    get_validator,
    validate_schema,
)


# ---------------------------------------------------------------------------
# URL / image helpers (kept local — specific to policy checks)
# ---------------------------------------------------------------------------


def _is_https_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and bool(parsed.netloc)


def _is_disallowed_url(url: str) -> bool:
    """Reject obvious XSS / unsafe schemes even if schema is relaxed."""
    parsed = urlparse(url)
    return parsed.scheme in {"javascript", "data", "file"}


def _has_disallowed_image_query_params(url: str) -> bool:
    parsed = urlparse(url)
    for k, _ in parse_qsl(parsed.query, keep_blank_values=True):
        if k.strip().lower() in DISALLOWED_IMAGE_QUERY_KEYS:
            return True
    return False


def _is_disallowed_image_host(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    return host in DISALLOWED_IMAGE_HOSTS


# ---------------------------------------------------------------------------
# Source schema validation
# ---------------------------------------------------------------------------


def validate_components(repo_root: Path) -> list[ValidationIssue]:
    """Validate all source component submissions under `components/registry/components/`.

    Returns issues tagged with ``layer=SCHEMA``.
    """
    registry_root = repo_root / "components" / "registry"
    schema_path = registry_root / "schemas" / "component.schema.json"
    components_dir = source_components_dir(repo_root)
    validator = get_validator(schema_path)

    issues: list[ValidationIssue] = []
    for file_path in sorted(components_dir.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.suffix != ".json":
            issues.append(
                ValidationIssue(
                    file=file_path,
                    schema=schema_path,
                    message=(
                        "Invalid file extension in components directory. "
                        "Source component files must end with `.json`."
                    ),
                    json_path=None,
                    layer=ValidationLayer.SCHEMA,
                )
            )
    for json_file in sorted(components_dir.glob("*.json")):
        issues.extend(validate_schema(json_file, schema_path, validator, layer=ValidationLayer.SCHEMA))
    return issues


# ---------------------------------------------------------------------------
# Policy / lint validation
# ---------------------------------------------------------------------------


def validate_policies(
    repo_root: Path, *, max_component_bytes: int = 50_000
) -> list[ValidationIssue]:
    """Policy/lint checks beyond JSON Schema for source submission `*.json` files.

    Returns issues tagged with ``layer=POLICY``.  File-size violations are
    reported at ``WARNING`` severity (they indicate potential abuse but do not
    block schema correctness).
    """
    from _utils.io import load_json

    registry_root = repo_root / "components" / "registry"
    schema_path = registry_root / "schemas" / "component.schema.json"
    components_dir = source_components_dir(repo_root)

    issues: list[ValidationIssue] = []
    first_by_repo: dict[str, Path] = {}

    def _issue(
        json_file: Path, message: str, json_path: str | None, severity: Severity = Severity.ERROR
    ) -> ValidationIssue:
        return ValidationIssue(
            file=json_file,
            schema=schema_path,
            message=message,
            json_path=json_path,
            layer=ValidationLayer.POLICY,
            severity=severity,
        )

    for json_file in sorted(components_dir.glob("*.json")):
        # File size abuse guardrail
        try:
            size = json_file.stat().st_size
        except OSError as e:  # pragma: no cover
            issues.append(_issue(json_file, f"Could not stat file: {e}", None))
            continue
        if size > max_component_bytes:
            issues.append(
                _issue(
                    json_file,
                    f"File too large ({size} bytes). Max allowed is {max_component_bytes} bytes.",
                    None,
                    severity=Severity.WARNING,
                )
            )

        # Best-effort JSON load for lint checks (schema validation handled separately)
        try:
            obj = load_json(json_file)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        links = obj.get("links")
        if not isinstance(links, dict):
            continue

        # --- GitHub URL checks ---
        gh = links.get("github")
        if isinstance(gh, str) and gh:
            if _is_disallowed_url(gh) or not _is_https_url(gh):
                issues.append(
                    _issue(
                        json_file,
                        "URL must be https:// and must not use a disallowed scheme.",
                        "links.github",
                    )
                )
            else:
                try:
                    canonical = normalize_github_repo_url(gh)
                    key = urlparse(canonical).path.lower().strip("/")
                    if key in first_by_repo:
                        issues.append(
                            _issue(
                                json_file,
                                f"Duplicate component identity: links.github repo `{key}` "
                                f"already submitted in `{first_by_repo[key].name}`.",
                                "links.github",
                            )
                        )
                    else:
                        first_by_repo[key] = json_file
                except Exception as e:
                    issues.append(_issue(json_file, str(e), "links.github"))

        # --- Other URL fields ---
        for path, val in (
            ("links.demo", links.get("demo")),
            ("links.docs", links.get("docs")),
        ):
            if val is None:
                continue
            if isinstance(val, str) and (_is_disallowed_url(val) or not _is_https_url(val)):
                issues.append(
                    _issue(
                        json_file,
                        "URL must be https:// and must not use a disallowed scheme.",
                        path,
                    )
                )

        # --- Image URL checks ---
        media = obj.get("media")
        if isinstance(media, dict):
            img = media.get("image")
            if img is None:
                pass  # Optional; null is allowed.
            elif isinstance(img, str):
                if _is_disallowed_url(img) or not _is_https_url(img):
                    issues.append(
                        _issue(
                            json_file,
                            "URL must be https:// and must not use a disallowed scheme.",
                            "media.image",
                        )
                    )
                elif _is_disallowed_image_host(img):
                    issues.append(
                        _issue(
                            json_file,
                            "Image host is not allowed for `media.image` "
                            "(brittle proxy). Use a stable upstream URL instead.",
                            "media.image",
                        )
                    )
                elif _has_disallowed_image_query_params(img):
                    issues.append(
                        _issue(
                            json_file,
                            "Signed/expiring image URLs are not allowed for `media.image` "
                            "(disallowed query parameters detected).",
                            "media.image",
                        )
                    )
            else:
                issues.append(
                    _issue(
                        json_file,
                        "`media.image` must be a string URL or null.",
                        "media.image",
                    )
                )

    return issues


# ---------------------------------------------------------------------------
# Compiled artifact validation
# ---------------------------------------------------------------------------


def validate_compiled(repo_root: Path) -> list[ValidationIssue]:
    """Validate the compiled catalog artifact `components/registry/compiled/components.json`.

    Returns issues tagged with ``layer=COMPILED``.
    """
    registry_root = repo_root / "components" / "registry"
    schema_path = registry_root / "schemas" / "compiled.schema.json"
    compiled_path = registry_root / "compiled" / "components.json"
    if not compiled_path.is_file():
        return [
            ValidationIssue(
                file=compiled_path,
                schema=schema_path,
                message="Compiled artifact not found (skipping).",
                json_path=None,
                layer=ValidationLayer.COMPILED,
                severity=Severity.WARNING,
            )
        ]
    validator = get_validator(schema_path)
    return validate_schema(
        compiled_path, schema_path, validator, layer=ValidationLayer.COMPILED
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_LAYER_LABELS: dict[ValidationLayer, str] = {
    ValidationLayer.SCHEMA: "schema",
    ValidationLayer.POLICY: "policy",
    ValidationLayer.COMPILED: "compiled",
}

_SEVERITY_LABELS: dict[Severity, str] = {
    Severity.ERROR: "ERROR",
    Severity.WARNING: "WARN",
}


def _print_report(all_issues: list[ValidationIssue], repo_root: Path) -> None:
    """Print a grouped, sorted error report to stderr.

    Output order:
    1. Errors before warnings.
    2. Schema issues before policy issues before compiled issues.
    3. Within each group, sorted by file then by json_path.
    """
    hard_errors = [i for i in all_issues if i.severity == Severity.ERROR]
    warnings = [i for i in all_issues if i.severity == Severity.WARNING]

    if not hard_errors and not warnings:
        return

    def _group(
        issues: list[ValidationIssue],
    ) -> dict[Path, list[ValidationIssue]]:
        by_file: dict[Path, list[ValidationIssue]] = defaultdict(list)
        for issue in issues:
            by_file[issue.file].append(issue)
        return by_file

    def _rel(path: Path) -> Path:
        """Best-effort relative path; falls back to absolute if not under repo_root."""
        if path.is_absolute():
            try:
                return path.relative_to(repo_root)
            except ValueError:
                return path
        return path

    total_errors = len(hard_errors)
    total_warnings = len(warnings)
    parts: list[str] = []
    if total_errors:
        parts.append(f"{total_errors} error(s)")
    if total_warnings:
        parts.append(f"{total_warnings} warning(s)")

    all_affected = set(i.file for i in all_issues if i.severity == Severity.ERROR)
    total_files = len(all_affected) if all_affected else len(set(i.file for i in all_issues))

    header = f"Found {', '.join(parts)} across {total_files} file(s):"
    print(header, file=sys.stderr)

    # Print errors first, then warnings.
    for label, issues_to_print in [("ERROR", hard_errors), ("WARN", warnings)]:
        if not issues_to_print:
            continue
        by_file = _group(issues_to_print)
        for file_path in sorted(by_file.keys()):
            file_issues = by_file[file_path]
            rel = _rel(file_path)
            schema_rel = _rel(file_issues[0].schema)
            print(f"\n- {rel} ({len(file_issues)} {label.lower()}(s))", file=sys.stderr)
            print(f"  schema: {schema_rel}", file=sys.stderr)
            # Sort within file: layer order (schema→policy→compiled), then path, then message.
            for issue in sorted(
                file_issues,
                key=lambda i: (_LAYER_ORDER[i.layer], i.json_path or "$", i.message),
            ):
                layer_tag = _LAYER_LABELS.get(issue.layer, "?")
                jp = issue.json_path or "$"
                print(f"  - [{layer_tag}] {jp}: {issue.message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    """CLI entrypoint.

    Parameters
    ----------
    argv
        CLI arguments excluding the program name (i.e., ``sys.argv[1:]``).

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

    script_path = Path(__file__).resolve()
    registry_root = script_path.parents[1]  # components/registry
    repo_root = registry_root.parents[1]  # repo root

    all_issues: list[ValidationIssue] = []

    # Guardrails for common mistakes
    if not (registry_root / "schemas" / "component.schema.json").is_file():
        print(
            "ERROR: Missing schema: components/registry/schemas/component.schema.json",
            file=sys.stderr,
        )
        return 2
    components_dir = source_components_dir(repo_root)
    if not components_dir.is_dir():
        print(
            "ERROR: Missing source components directory: components/registry/components/",
            file=sys.stderr,
        )
        return 2

    all_issues.extend(validate_components(repo_root))
    if not args.no_policy:
        all_issues.extend(
            validate_policies(repo_root, max_component_bytes=args.max_component_bytes)
        )
    if args.compiled:
        all_issues.extend(validate_compiled(repo_root))

    # Sort all issues for deterministic output.
    all_issues.sort(key=lambda i: i.sort_key())

    hard_errors = [i for i in all_issues if i.severity == Severity.ERROR]

    if hard_errors:
        _print_report(all_issues, repo_root)
        return 1

    # No hard errors — print OK, then show any warnings / skipped notes.
    print("OK: all validated files passed.")
    warnings = [i for i in all_issues if i.severity == Severity.WARNING]
    if warnings:
        for w in warnings:
            layer_tag = _LAYER_LABELS.get(w.layer, "?")
            try:
                rel = w.file.relative_to(repo_root) if w.file.is_absolute() else w.file
            except ValueError:
                rel = w.file
            print(f"NOTE: [{layer_tag}] {rel}: {w.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
