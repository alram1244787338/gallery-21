"""Tests for validate.py — error reporting, layer tagging, and output ordering.

These tests verify that the validation pipeline produces clear, actionable
error messages with correct layer and severity tagging for every common
failure mode:

- Schema violations (missing fields, wrong types, invalid patterns)
- Policy violations (duplicate repos, non-HTTPS URLs, bad image URLs)
- Compiled artifact validation
- Output ordering (errors before warnings, schema before policy)
- Layer labels in formatted output
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

# conftest adds scripts/ to sys.path, so these imports work:
from validate import (  # type: ignore[import-not-found]
    Severity,
    ValidationIssue,
    ValidationLayer,
    main,
    validate_compiled,
    validate_components,
    validate_policies,
)

from conftest import _minimal_valid_component, _write_json  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_validate(repo_root: Path, *extra_args: str) -> tuple[int, str, str]:
    """Run validate.main() and capture stdout/stderr."""
    args = ["--no-policy", *extra_args] if "--no-policy" not in extra_args else list(extra_args)
    # Reset to ensure clean state
    old_argv = sys.argv
    sys.argv = ["validate.py", *args]

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout_capture, stderr_capture

    try:
        # Patch the script path resolution so it uses our temp repo
        import validate as v

        original_main = v.main

        def patched_main(argv):
            """Wrap main to use our temp repo_root."""
            from _utils.paths import source_components_dir

            parser_result = __import__("argparse").ArgumentParser()
            # Just call the real main but patch the path resolution
            return original_main(argv)

        rc = patched_main(args)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        sys.argv = old_argv

    return rc, stdout_capture.getvalue(), stderr_capture.getvalue()


def _run_validate_with_root(
    repo_root: Path, *extra_args: str
) -> tuple[int, str, str]:
    """Run validate.main() with path patching for our temp repo.

    This monkey-patches the path resolution inside validate.main so that it
    uses ``repo_root`` instead of the real repo.
    """
    import validate as v
    from _utils import paths as paths_mod

    original_source = paths_mod.source_components_dir

    def patched_source(root: Path) -> Path:
        return repo_root / "components" / "registry" / "components"

    paths_mod.source_components_dir = patched_source  # type: ignore[assignment]

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout_capture, stderr_capture

    try:
        rc = v.main(list(extra_args))
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        paths_mod.source_components_dir = original_source  # type: ignore[assignment]

    return rc, stdout_capture.getvalue(), stderr_capture.getvalue()


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Tests for schema-level validation (validate_components)."""

    def test_valid_component_passes(self, repo_root: Path, write_component) -> None:
        """A fully valid component produces zero schema issues."""
        write_component("good", _minimal_valid_component())
        issues = validate_components(repo_root)
        assert issues == []

    def test_missing_required_fields(self, repo_root: Path, write_component) -> None:
        """Missing top-level required fields produce a clear schema error."""
        # Empty object — missing schemaVersion, title, author, links, governance, categories
        write_component("bad", {})
        issues = validate_components(repo_root)
        assert len(issues) >= 1
        # Find the "required" error
        req_issues = [i for i in issues if "Missing required field" in i.message]
        assert req_issues, f"Expected 'Missing required field' issue, got: {[i.message for i in issues]}"
        # Check layer
        assert all(i.layer == ValidationLayer.SCHEMA for i in issues)
        # Check that the message lists the missing fields
        msg = req_issues[0].message
        assert "title" in msg
        assert "author" in msg

    def test_schema_type_violation(self, repo_root: Path, write_component) -> None:
        """Wrong field type produces a schema error with field path."""
        bad = _minimal_valid_component(title=12345)  # title should be string
        write_component("bad-type", bad)
        issues = validate_components(repo_root)
        assert len(issues) >= 1
        type_issues = [i for i in issues if "title" in (i.json_path or "")]
        assert type_issues, f"Expected issue at json_path containing 'title', got: {[(i.json_path, i.message) for i in issues]}"
        assert all(i.layer == ValidationLayer.SCHEMA for i in type_issues)

    def test_invalid_github_url_pattern(self, repo_root: Path, write_component) -> None:
        """A links.github URL that doesn't match the schema pattern triggers a schema error."""
        bad = _minimal_valid_component()
        bad["links"]["github"] = "not-a-url"
        write_component("bad-url", bad)
        issues = validate_components(repo_root)
        assert len(issues) >= 1
        url_issues = [i for i in issues if "links.github" in (i.json_path or "")]
        assert url_issues

    def test_invalid_category(self, repo_root: Path, write_component) -> None:
        """An unknown category value triggers a schema error."""
        bad = _minimal_valid_component(categories=["NotARealCategory"])
        write_component("bad-cat", bad)
        issues = validate_components(repo_root)
        assert len(issues) >= 1

    def test_additional_property_rejected(self, repo_root: Path, write_component) -> None:
        """Extra top-level fields are rejected (additionalProperties: false)."""
        bad = _minimal_valid_component()
        bad["customField"] = "surprise"
        write_component("extra", bad)
        issues = validate_components(repo_root)
        assert len(issues) >= 1
        extra_issues = [i for i in issues if "customField" in i.message or "additional" in i.message.lower()]
        assert extra_issues

    def test_invalid_file_extension(self, repo_root: Path, write_raw_file) -> None:
        """Non-.json files in the components dir trigger a schema error."""
        write_raw_file("readme.txt", "This is not JSON")
        issues = validate_components(repo_root)
        ext_issues = [i for i in issues if "file extension" in i.message.lower()]
        assert ext_issues
        assert ext_issues[0].layer == ValidationLayer.SCHEMA


# ---------------------------------------------------------------------------
# Policy validation tests
# ---------------------------------------------------------------------------


class TestPolicyValidation:
    """Tests for policy-level validation (validate_policies)."""

    def test_duplicate_github_repo(self, repo_root: Path, write_component) -> None:
        """Two components with the same GitHub repo produce a policy error."""
        write_component("aaa-first", _minimal_valid_component())
        # Same github URL, different file (name sorts after aaa-first)
        dup = _minimal_valid_component(title="Duplicate")
        write_component("zzz-duplicate", dup)
        issues = validate_policies(repo_root)
        dup_issues = [i for i in issues if "Duplicate" in i.message]
        assert len(dup_issues) == 1, f"Expected exactly 1 duplicate issue, got: {[i.message for i in issues]}"
        assert dup_issues[0].layer == ValidationLayer.POLICY
        assert dup_issues[0].json_path == "links.github"
        assert "aaa-first.json" in dup_issues[0].message

    def test_non_https_github_url(self, repo_root: Path, write_component) -> None:
        """An http:// GitHub URL triggers a policy error."""
        bad = _minimal_valid_component()
        bad["links"]["github"] = "http://github.com/testuser/test-repo"
        write_component("http-url", bad)
        issues = validate_policies(repo_root)
        https_issues = [i for i in issues if "https://" in i.message]
        assert https_issues
        assert https_issues[0].layer == ValidationLayer.POLICY
        assert https_issues[0].json_path == "links.github"

    def test_javascript_url_rejected(self, repo_root: Path, write_component) -> None:
        """A javascript: URL in links.demo triggers a policy error."""
        bad = _minimal_valid_component()
        bad["links"]["demo"] = "javascript:alert(1)"
        write_component("xss", bad)
        issues = validate_policies(repo_root)
        xss_issues = [i for i in issues if i.json_path == "links.demo"]
        assert xss_issues
        assert xss_issues[0].layer == ValidationLayer.POLICY

    def test_signed_image_url_rejected(self, repo_root: Path, write_component) -> None:
        """A signed/expiring image URL triggers a policy error."""
        bad = _minimal_valid_component()
        bad["media"] = {
            "image": "https://cdn.example.com/img.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=abc"
        }
        write_component("signed-img", bad)
        issues = validate_policies(repo_root)
        img_issues = [i for i in issues if "Signed/expiring" in i.message]
        assert len(img_issues) == 1
        assert img_issues[0].layer == ValidationLayer.POLICY
        assert img_issues[0].json_path == "media.image"

    def test_disallowed_image_host(self, repo_root: Path, write_component) -> None:
        """A camo.githubusercontent.com image URL triggers a policy error."""
        bad = _minimal_valid_component()
        bad["media"] = {
            "image": "https://camo.githubusercontent.com/abc123/xyz.png"
        }
        write_component("proxy-img", bad)
        issues = validate_policies(repo_root)
        img_issues = [i for i in issues if "Image host" in i.message]
        assert len(img_issues) == 1
        assert img_issues[0].layer == ValidationLayer.POLICY
        assert img_issues[0].json_path == "media.image"

    def test_non_https_image_url(self, repo_root: Path, write_component) -> None:
        """An http:// image URL triggers a policy error."""
        bad = _minimal_valid_component()
        bad["media"] = {"image": "http://example.com/img.png"}
        write_component("http-img", bad)
        issues = validate_policies(repo_root)
        img_issues = [i for i in issues if i.json_path == "media.image" and "https://" in i.message]
        assert img_issues

    def test_file_too_large_is_warning(self, repo_root: Path, write_raw_file) -> None:
        """An oversized file produces a WARNING, not an ERROR."""
        # Create a 100-byte file but set max to 50
        write_raw_file("huge.json", json.dumps(_minimal_valid_component()))
        # The default max is 50_000 bytes; our test file is small, so we
        # call with a tiny max_component_bytes to trigger the warning.
        issues = validate_policies(repo_root, max_component_bytes=10)
        size_issues = [i for i in issues if "too large" in i.message]
        assert len(size_issues) == 1
        assert size_issues[0].severity == Severity.WARNING
        assert size_issues[0].layer == ValidationLayer.POLICY

    def test_valid_component_no_policy_issues(self, repo_root: Path, write_component) -> None:
        """A valid component produces zero policy issues."""
        write_component("good", _minimal_valid_component())
        issues = validate_policies(repo_root)
        # Filter out warnings (like file size if the test file is somehow large)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert errors == []

    def test_image_null_is_allowed(self, repo_root: Path, write_component) -> None:
        """media.image = null is valid — no policy error."""
        comp = _minimal_valid_component()
        comp["media"] = {"image": None}
        write_component("null-img", comp)
        issues = validate_policies(repo_root)
        img_issues = [i for i in issues if i.json_path == "media.image"]
        assert img_issues == []

    def test_multiple_policy_errors_same_file(self, repo_root: Path, write_component) -> None:
        """A single file with multiple issues produces separate ValidationIssue objects."""
        bad = _minimal_valid_component()
        bad["links"]["demo"] = "javascript:void(0)"
        bad["media"] = {"image": "https://camo.githubusercontent.com/bad.png"}
        write_component("multi-bad", bad)
        issues = validate_policies(repo_root)
        # Should have at least 2 issues: one for demo URL, one for image host
        demo_issues = [i for i in issues if i.json_path == "links.demo"]
        img_issues = [i for i in issues if i.json_path == "media.image"]
        assert demo_issues
        assert img_issues


# ---------------------------------------------------------------------------
# Compiled validation tests
# ---------------------------------------------------------------------------


class TestCompiledValidation:
    """Tests for compiled artifact validation (validate_compiled)."""

    def test_missing_compiled_is_warning(self, repo_root: Path) -> None:
        """A missing compiled artifact is a warning, not an error."""
        issues = validate_compiled(repo_root)
        assert len(issues) == 1
        assert issues[0].severity == Severity.WARNING
        assert issues[0].layer == ValidationLayer.COMPILED
        assert "skipping" in issues[0].message.lower()

    def test_valid_compiled_passes(self, repo_root: Path) -> None:
        """A valid compiled artifact produces zero errors."""
        compiled_path = repo_root / "components" / "registry" / "compiled" / "components.json"
        # Minimal valid compiled artifact matching compiled.schema.json
        compiled = {
            "generatedAt": "2025-01-01T00:00:00Z",
            "schemaVersion": 1,
            "categories": ["All", "Widgets"],
            "components": [
                {
                    "title": "Test Component",
                    "author": "testuser",
                    "socialUrl": "https://github.com/testuser",
                    "pipLink": "pip install test-repo",
                    "pypi": "test-repo",
                    "categories": ["Widgets"],
                    "image": None,
                    "gitHubUrl": "https://github.com/testuser/test-repo",
                    "enabled": True,
                    "appUrl": None,
                }
            ],
        }
        _write_json(compiled_path, compiled)
        issues = validate_compiled(repo_root)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert errors == [], f"Unexpected errors: {[(i.json_path, i.message) for i in errors]}"

    def test_invalid_compiled_schema_violation(self, repo_root: Path) -> None:
        """A compiled artifact with schema violations produces errors tagged COMPILED."""
        compiled_path = repo_root / "components" / "registry" / "compiled" / "components.json"
        # Missing required fields
        compiled = {
            "generatedAt": "2025-01-01T00:00:00Z",
            "schemaVersion": 1,
            # Missing "categories" and "components"
        }
        _write_json(compiled_path, compiled)
        issues = validate_compiled(repo_root)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1
        assert all(i.layer == ValidationLayer.COMPILED for i in errors)


# ---------------------------------------------------------------------------
# Output ordering and formatting tests
# ---------------------------------------------------------------------------


class TestOutputOrdering:
    """Tests for error report output ordering and layer labels."""

    def test_errors_before_warnings(self, repo_root: Path, write_component, write_raw_file) -> None:
        """When both errors and warnings exist, errors are printed first."""
        # Schema error: missing required fields
        write_component("bad-schema", {})
        # Policy warning: file too large (set max to 1 byte)
        # We need a valid-ish JSON that will pass schema but be >1 byte
        write_raw_file("big.json", json.dumps(_minimal_valid_component()))

        import validate as v

        schema_issues = validate_components(repo_root)
        policy_issues = validate_policies(repo_root, max_component_bytes=1)
        all_issues = schema_issues + policy_issues
        all_issues.sort(key=lambda i: i.sort_key())

        errors = [i for i in all_issues if i.severity == Severity.ERROR]
        warnings = [i for i in all_issues if i.severity == Severity.WARNING]
        assert errors, "Expected at least one schema error"
        assert warnings, "Expected at least one size warning"
        # First items should be errors
        first_warning_idx = next(
            idx for idx, i in enumerate(all_issues) if i.severity == Severity.WARNING
        )
        last_error_idx = max(
            idx for idx, i in enumerate(all_issues) if i.severity == Severity.ERROR
        )
        assert last_error_idx < first_warning_idx

    def test_schema_before_policy_in_sort(self, repo_root: Path, write_component) -> None:
        """Schema errors sort before policy errors for the same file."""
        # A component that fails both schema (bad type) and policy (duplicate)
        write_component("aaa-first", _minimal_valid_component())
        bad = _minimal_valid_component(title=12345)  # schema error on title
        write_component("zzz-bad-and-dup", bad)  # also has same github = policy error

        schema_issues = validate_components(repo_root)
        policy_issues = validate_policies(repo_root)
        all_issues = schema_issues + policy_issues
        all_issues.sort(key=lambda i: i.sort_key())

        # Filter to issues for "zzz-bad-and-dup"
        bad_issues = [i for i in all_issues if "zzz-bad-and-dup" in str(i.file)]
        if bad_issues:
            layers = [i.layer for i in bad_issues]
            schema_idx = next((i for i, l in enumerate(layers) if l == ValidationLayer.SCHEMA), None)
            policy_idx = next((i for i, l in enumerate(layers) if l == ValidationLayer.POLICY), None)
            if schema_idx is not None and policy_idx is not None:
                assert schema_idx < policy_idx

    def test_print_report_layer_labels(self, repo_root: Path, write_component, capsys) -> None:
        """The printed report includes [schema] and [policy] layer labels."""
        # Schema error
        write_component("missing", {})
        # Policy error — use name that sorts after "aaa-first"
        write_component("aaa-first", _minimal_valid_component())
        dup = _minimal_valid_component(title="Dup")
        write_component("zzz-dup", dup)

        import validate as v

        schema_issues = validate_components(repo_root)
        policy_issues = validate_policies(repo_root)
        all_issues = schema_issues + policy_issues
        all_issues.sort(key=lambda i: i.sort_key())

        hard_errors = [i for i in all_issues if i.severity == Severity.ERROR]
        assert hard_errors, "Expected hard errors for this test"

        # Capture stderr
        old_stderr = sys.stderr
        sys.stderr = captured = io.StringIO()
        try:
            v._print_report(all_issues, repo_root)
        finally:
            sys.stderr = old_stderr

        output = captured.getvalue()
        assert "[schema]" in output, f"Expected '[schema]' label in output:\n{output}"
        assert "[policy]" in output, f"Expected '[policy]' label in output:\n{output}"

    def test_print_report_shows_file_path(self, repo_root: Path, write_component, capsys) -> None:
        """Error output includes the relative file path."""
        write_component("my-component", {})
        issues = validate_components(repo_root)
        issues.sort(key=lambda i: i.sort_key())

        old_stderr = sys.stderr
        sys.stderr = captured = io.StringIO()
        try:
            from validate import _print_report
            _print_report(issues, repo_root)
        finally:
            sys.stderr = old_stderr

        output = captured.getvalue()
        assert "my-component.json" in output

    def test_sort_key_deterministic(self, repo_root: Path, write_component) -> None:
        """sort_key produces a stable, deterministic ordering."""
        write_component("aaa-first", _minimal_valid_component())
        dup = _minimal_valid_component(title="Dup")
        write_component("zzz-dup", dup)

        schema_issues = validate_components(repo_root)
        policy_issues = validate_policies(repo_root)
        all_issues = schema_issues + policy_issues

        # Sort twice and verify identical
        sorted1 = sorted(all_issues, key=lambda i: i.sort_key())
        sorted2 = sorted(all_issues, key=lambda i: i.sort_key())
        assert [i.sort_key() for i in sorted1] == [i.sort_key() for i in sorted2]


# ---------------------------------------------------------------------------
# ValidationIssue dataclass tests
# ---------------------------------------------------------------------------


class TestValidationIssueDataclass:
    """Tests for the ValidationIssue data structure itself."""

    def test_default_layer_is_schema(self) -> None:
        """Default layer is SCHEMA (backwards compat)."""
        issue = ValidationIssue(
            file=Path("test.json"),
            schema=Path("schema.json"),
            message="test",
        )
        assert issue.layer == ValidationLayer.SCHEMA

    def test_default_severity_is_error(self) -> None:
        """Default severity is ERROR."""
        issue = ValidationIssue(
            file=Path("test.json"),
            schema=Path("schema.json"),
            message="test",
        )
        assert issue.severity == Severity.ERROR

    def test_frozen_dataclass(self) -> None:
        """ValidationIssue is immutable."""
        issue = ValidationIssue(
            file=Path("test.json"),
            schema=Path("schema.json"),
            message="test",
        )
        with pytest.raises(AttributeError):
            issue.message = "changed"  # type: ignore[misc]

    def test_sort_key_ordering(self) -> None:
        """sort_key puts errors before warnings and schema before policy."""
        err_schema = ValidationIssue(
            file=Path("a.json"), schema=Path("s.json"), message="e",
            layer=ValidationLayer.SCHEMA, severity=Severity.ERROR,
        )
        err_policy = ValidationIssue(
            file=Path("a.json"), schema=Path("s.json"), message="e",
            layer=ValidationLayer.POLICY, severity=Severity.ERROR,
        )
        warn_schema = ValidationIssue(
            file=Path("a.json"), schema=Path("s.json"), message="w",
            layer=ValidationLayer.SCHEMA, severity=Severity.WARNING,
        )
        keys = [err_policy.sort_key(), warn_schema.sort_key(), err_schema.sort_key()]
        sorted_keys = sorted(keys)
        # err_schema < err_policy < warn_schema
        assert sorted_keys == [err_schema.sort_key(), err_policy.sort_key(), warn_schema.sort_key()]


# ---------------------------------------------------------------------------
# Shared validation_common module tests
# ---------------------------------------------------------------------------


class TestValidationCommon:
    """Tests for the shared _utils.validation_common module."""

    def test_format_json_path_nested(self) -> None:
        from _utils.validation_common import format_json_path
        assert format_json_path(["author", "github"]) == "author.github"

    def test_format_json_path_array_index(self) -> None:
        from _utils.validation_common import format_json_path
        assert format_json_path(["components", 0, "title"]) == "components[0].title"

    def test_format_json_path_root(self) -> None:
        from _utils.validation_common import format_json_path
        assert format_json_path([]) == "$"

    def test_missing_required_fields_extraction(self) -> None:
        """missing_required_fields returns None for non-required errors."""
        from _utils.validation_common import missing_required_fields

        class FakeErr:
            validator = "type"
            validator_value = "string"
            instance = "not a dict"

        assert missing_required_fields(FakeErr()) is None

    def test_validate_schema_in_memory(self) -> None:
        """validate_schema_in_memory produces issues for invalid data."""
        from _utils.validation_common import validate_schema_in_memory, load_schema

        schema_path = Path(__file__).resolve().parents[1] / "schemas" / "component.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        issues = validate_schema_in_memory({}, schema)
        assert len(issues) >= 1
        assert any("Missing required field" in i.message for i in issues)

    def test_validate_schema_in_memory_valid(self) -> None:
        """validate_schema_in_memory returns empty for valid data."""
        from _utils.validation_common import validate_schema_in_memory

        schema_path = Path(__file__).resolve().parents[1] / "schemas" / "component.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        valid = _minimal_valid_component()
        issues = validate_schema_in_memory(valid, schema)
        assert issues == []
