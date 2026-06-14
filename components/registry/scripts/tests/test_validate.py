"""Tests for the layered component-registry validator (`validate.py`).

These tests build throwaway registry layouts in temp dirs (copying the *real*
schemas so behavior matches production) and assert that each kind of common
failure is reported with a clear, locatable signal:

- which file (grouped by path),
- which field (``json_path`` / ``location``),
- which rule layer (``source`` schema / ``policy`` lint / ``compiled`` schema),
- and, when a file trips several rules, that the most critical one is surfaced first.

Run directly (no pytest required):

    python components/registry/scripts/tests/test_validate.py
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Make `validate` and the `_utils` package importable the same way the CLI does
# (the script's own directory is on sys.path when run as `python .../validate.py`).
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import validate  # noqa: E402
from _utils import url_policy  # noqa: E402

REGISTRY_ROOT = SCRIPTS_DIR.parent
REAL_COMPONENT_SCHEMA = REGISTRY_ROOT / "schemas" / "component.schema.json"
REAL_COMPILED_SCHEMA = REGISTRY_ROOT / "schemas" / "compiled.schema.json"


def _valid_component(github_user: str = "alice", repo: str = "streamlit-thing", **overrides):
    """Return a minimal schema-valid + policy-clean source submission dict."""
    obj = {
        "schemaVersion": 1,
        "title": "Thing",
        "author": {"github": github_user},
        "links": {
            "github": f"https://github.com/{github_user}/{repo}",
            "pypi": None,
            "demo": None,
            "docs": None,
        },
        "media": {"image": None},
        "install": {"pip": "pip install thing"},
        "governance": {"enabled": True, "notes": None},
        "categories": ["Widgets"],
    }
    obj.update(overrides)
    return obj


def _valid_compiled(**component_overrides):
    """Return a minimal schema-valid compiled artifact (one component)."""
    component = {
        "title": "Thing",
        "author": "alice",
        "socialUrl": None,
        "pipLink": "pip install thing",
        "categories": ["Widgets"],
        "image": None,
        "gitHubUrl": "https://github.com/alice/streamlit-thing",
        "enabled": True,
        "appUrl": None,
    }
    component.update(component_overrides)
    return {
        "generatedAt": "2026-01-01T00:00:00Z",
        "schemaVersion": 1,
        "categories": ["Widgets"],
        "components": [component],
    }


class ValidatorTestBase(unittest.TestCase):
    def make_repo(self) -> Path:
        """Create a temp repo with the real schemas wired up; auto-cleaned."""
        root = Path(tempfile.mkdtemp(prefix="reg-validate-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        registry = root / "components" / "registry"
        (registry / "components").mkdir(parents=True)
        (registry / "schemas").mkdir(parents=True)
        shutil.copy(REAL_COMPONENT_SCHEMA, registry / "schemas" / "component.schema.json")
        shutil.copy(REAL_COMPILED_SCHEMA, registry / "schemas" / "compiled.schema.json")
        return root

    def components_dir(self, root: Path) -> Path:
        return root / "components" / "registry" / "components"

    def write_component(self, root: Path, name: str, obj) -> Path:
        path = self.components_dir(root) / name
        if isinstance(obj, str):
            path.write_text(obj, encoding="utf-8")
        else:
            path.write_text(json.dumps(obj), encoding="utf-8")
        return path

    def write_compiled(self, root: Path, obj) -> Path:
        path = root / "components" / "registry" / "compiled" / "components.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), encoding="utf-8")
        return path

    def codes(self, issues):
        return [i.code for i in issues]


class TestSharedUrlPolicy(unittest.TestCase):
    """The consolidated predicates back both validate.py and enrich_images.py."""

    def test_https_and_scheme(self):
        self.assertTrue(url_policy.is_https_url("https://example.com/a.png"))
        self.assertFalse(url_policy.is_https_url("http://example.com/a.png"))
        self.assertTrue(url_policy.has_disallowed_scheme("javascript:alert(1)"))
        self.assertFalse(url_policy.is_allowed_https("http://example.com"))
        self.assertTrue(url_policy.is_allowed_https("https://example.com"))

    def test_image_host_and_signed_query(self):
        self.assertTrue(
            url_policy.image_host_disallowed("https://camo.githubusercontent.com/x.png")
        )
        self.assertFalse(url_policy.image_host_disallowed("https://raw.githubusercontent.com/x"))
        self.assertTrue(
            url_policy.image_has_signed_query("https://s3.example.com/x.png?X-Amz-Signature=abc")
        )
        self.assertFalse(url_policy.image_has_signed_query("https://example.com/x.png?v=1"))


class TestSourceSchemaLayer(ValidatorTestBase):
    def test_valid_component_passes(self):
        root = self.make_repo()
        self.write_component(root, "thing.json", _valid_component())

        self.assertEqual(validate.validate_components(root), [])
        self.assertEqual(validate.validate_policies(root), [])

        # End-to-end: a clean repo exits 0 and prints the canonical OK line.
        out, err, code = run_main([], root)
        self.assertEqual(code, 0)
        self.assertIn("OK: all validated files passed.", out)
        self.assertEqual(err, "")

    def test_missing_required_fields_reports_field_and_layer(self):
        root = self.make_repo()
        obj = _valid_component()
        del obj["title"]
        del obj["categories"]
        self.write_component(root, "thing.json", obj)

        issues = validate.validate_components(root)
        required = [i for i in issues if i.code == "required"]
        # De-duplicated to a single root-level finding listing both missing fields.
        self.assertEqual(len(required), 1)
        issue = required[0]
        self.assertEqual(issue.layer, validate.LAYER_SOURCE)
        self.assertEqual(issue.location, "$")
        self.assertEqual(issue.severity, 1)
        self.assertIn("title", issue.message)
        self.assertIn("categories", issue.message)
        self.assertTrue(str(issue.schema).endswith("component.schema.json"))

    def test_invalid_json_is_clean_not_a_crash(self):
        root = self.make_repo()
        self.write_component(root, "broken.json", "{ this is not json ")

        # Must not raise; must report a locatable, top-priority issue.
        issues = validate.validate_components(root)
        invalid = [i for i in issues if i.code == "invalid_json"]
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0].layer, validate.LAYER_SOURCE)
        self.assertEqual(invalid[0].location, "$")
        self.assertEqual(invalid[0].severity, 0)
        self.assertTrue(invalid[0].message.startswith("Invalid JSON"))


class TestPolicyLayer(ValidatorTestBase):
    def test_duplicate_github_repo(self):
        root = self.make_repo()
        self.write_component(root, "a-first.json", _valid_component("alice", "dup-repo"))
        self.write_component(root, "b-second.json", _valid_component("alice", "dup-repo"))

        issues = validate.validate_policies(root)
        dups = [i for i in issues if i.code == "duplicate_repo"]
        self.assertEqual(len(dups), 1)
        issue = dups[0]
        self.assertEqual(issue.layer, validate.LAYER_POLICY)
        self.assertEqual(issue.location, "links.github")
        self.assertEqual(issue.severity, 2)
        # Names the *first* submission so the maintainer knows the collision.
        self.assertIn("a-first.json", issue.message)
        self.assertTrue(issue.file.name == "b-second.json")

    def test_invalid_image_urls(self):
        root = self.make_repo()
        self.write_component(
            root,
            "camo.json",
            _valid_component("u1", "r1", media={"image": "https://camo.githubusercontent.com/x"}),
        )
        self.write_component(
            root,
            "signed.json",
            _valid_component(
                "u2", "r2", media={"image": "https://s3.example.com/x.png?X-Amz-Signature=abc"}
            ),
        )
        self.write_component(
            root,
            "insecure.json",
            _valid_component("u3", "r3", media={"image": "http://example.com/x.png"}),
        )
        self.write_component(
            root,
            "xss.json",
            _valid_component("u4", "r4", media={"image": "javascript:alert(1)"}),
        )

        by_file = {i.file.name: i for i in validate.validate_policies(root)}
        self.assertEqual(by_file["camo.json"].code, "image_host")
        self.assertEqual(by_file["signed.json"].code, "image_query")
        self.assertEqual(by_file["insecure.json"].code, "url_scheme")
        self.assertEqual(by_file["xss.json"].code, "url_scheme")
        for issue in by_file.values():
            self.assertEqual(issue.layer, validate.LAYER_POLICY)
            self.assertEqual(issue.location, "media.image")

    def test_file_too_large(self):
        root = self.make_repo()
        self.write_component(root, "thing.json", _valid_component())

        issues = validate.validate_policies(root, max_component_bytes=10)
        big = [i for i in issues if i.code == "file_too_large"]
        self.assertEqual(len(big), 1)
        self.assertEqual(big[0].layer, validate.LAYER_POLICY)
        self.assertEqual(big[0].severity, 5)
        self.assertIn("bytes", big[0].message)

    def test_bad_extension_flagged_by_default_but_skipped_by_no_policy(self):
        root = self.make_repo()
        self.write_component(root, "thing.json", _valid_component())
        # A stray non-.json file (e.g. a placeholder) in the components dir.
        (self.components_dir(root) / ".gitkeep").write_text("", encoding="utf-8")

        # Default run treats the directory-hygiene rule as a (failing) policy check.
        out, err, code = run_main([], root)
        self.assertEqual(code, 1)
        self.assertIn("[policy:bad_extension]", err)

        # `--no-policy` is the documented escape hatch for policy/lint rules, so the
        # stray file is no longer flagged and the valid submission passes schema.
        out2, err2, code2 = run_main(["--no-policy"], root)
        self.assertEqual(code2, 0)
        self.assertIn("OK: all validated files passed.", out2)


class TestCompiledLayer(ValidatorTestBase):
    def test_compiled_anomaly_reports_compiled_field_and_schema(self):
        root = self.make_repo()
        self.write_component(root, "thing.json", _valid_component())
        artifact = _valid_compiled()
        del artifact["components"][0]["appUrl"]  # required compiled field
        self.write_compiled(root, artifact)

        issues = validate.validate_compiled(root)
        required = [i for i in issues if i.code == "required"]
        self.assertEqual(len(required), 1)
        issue = required[0]
        self.assertEqual(issue.layer, validate.LAYER_COMPILED)
        self.assertEqual(issue.location, "components[0]")
        self.assertIn("appUrl", issue.message)
        self.assertTrue(str(issue.schema).endswith("compiled.schema.json"))

    def test_missing_compiled_is_soft_skip(self):
        root = self.make_repo()
        self.write_component(root, "thing.json", _valid_component())

        issues = validate.validate_compiled(root)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, validate.CODE_SKIPPED)
        self.assertEqual(issues[0].layer, validate.LAYER_COMPILED)

        # A soft skip must not fail the run, even when --compiled is requested.
        report = validate.build_report(issues, root)
        self.assertEqual(report.exit_code, 0)
        self.assertIn("NOTE:", report.ok_text)

        out, err, code = run_main(["--compiled"], root)
        self.assertEqual(code, 0)
        self.assertIn("OK: all validated files passed.", out)

    def test_source_and_compiled_vocabularies_do_not_conflict(self):
        """A compiled-only failure stays in the compiled layer; source is untouched."""
        root = self.make_repo()
        self.write_component(root, "thing.json", _valid_component())  # valid source
        artifact = _valid_compiled()
        del artifact["components"][0]["gitHubUrl"]  # compiled-only field name
        self.write_compiled(root, artifact)

        source_issues = validate.validate_components(root)
        compiled_issues = validate.validate_compiled(root)

        # Source remains clean; the failure is isolated to the compiled layer.
        self.assertEqual(source_issues, [])
        self.assertTrue(compiled_issues)
        for issue in compiled_issues:
            self.assertEqual(issue.layer, validate.LAYER_COMPILED)
            self.assertTrue(str(issue.schema).endswith("compiled.schema.json"))
            # Must not bleed source field names into a compiled finding.
            self.assertNotIn("links.github", issue.message)
        self.assertTrue(any("gitHubUrl" in i.message for i in compiled_issues))


class TestPrioritizedReport(ValidatorTestBase):
    def test_most_critical_issue_is_listed_first(self):
        root = self.make_repo()
        # One file that trips both a structural schema error (missing title) and a
        # soft image policy error (brittle host).
        obj = _valid_component(media={"image": "https://camo.githubusercontent.com/x"})
        del obj["title"]
        self.write_component(root, "thing.json", obj)

        issues = validate.validate_components(root) + validate.validate_policies(root)
        report = validate.build_report(issues, root)
        self.assertEqual(report.exit_code, 1)

        lines = report.error_text.splitlines()
        idx_required = _line_index(lines, "[source:required]")
        idx_image = _line_index(lines, "[policy:image_host]")
        self.assertLess(idx_required, idx_image, "schema error should precede soft policy nit")

        # The file header advertises both layers and the per-line tags are present.
        header = next(line for line in lines if line.startswith("- "))
        self.assertIn("source", header)
        self.assertIn("policy", header)
        self.assertIn("[source:required]", report.error_text)
        self.assertIn("[policy:image_host]", report.error_text)

    def test_report_breakdown_counts_by_layer(self):
        root = self.make_repo()
        obj = _valid_component(media={"image": "http://example.com/x.png"})
        del obj["title"]
        self.write_component(root, "thing.json", obj)

        issues = validate.validate_components(root) + validate.validate_policies(root)
        report = validate.build_report(issues, root)
        first = report.error_text.splitlines()[0]
        self.assertIn("1 source", first)
        self.assertIn("1 policy", first)


def _line_index(lines, needle: str) -> int:
    for idx, line in enumerate(lines):
        if needle in line:
            return idx
    raise AssertionError(f"expected a line containing {needle!r} in:\n" + "\n".join(lines))


def run_main(argv, repo_root: Path):
    """Run validate.main with captured stdout/stderr; return (out, err, code)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = validate.main(argv, repo_root=repo_root)
    return out.getvalue(), err.getvalue(), code


if __name__ == "__main__":
    unittest.main(verbosity=2)
