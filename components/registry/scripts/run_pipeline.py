"""
Run the full Component Gallery pipeline with a single entrypoint.

Default pipeline:

  1) Validate `components/registry/components/*.json` submissions
  2) Build `components/registry/compiled/components.json`
  3) Validate `components/registry/compiled/components.json`
  4) Enrich GitHub metrics
  5) Enrich PyPI metrics
  6) Compute ranking signals
  7) Validate `components/registry/compiled/components.json` again

Run from the repo root (recommended):

    python components/registry/scripts/run_pipeline.py

Typical CI usage:

    # Build + validate only (no network)
    python components/registry/scripts/run_pipeline.py --no-enrich

Offline mode behavior (--no-enrich):

    - Skips all network-dependent steps (images, GitHub, PyPI, pypistats enrichment).
    - build_catalog runs with --pipeline-mode offline:
      * generatedAt is preserved from the previous artifact (not updated to now).
      * The artifact is tagged with pipelineMode=offline, enrichmentSkipped=true.
    - compute_ranking still runs but:
      * Metrics older than --stale-threshold-hours trigger ranking.isStale=true.
      * ranking.computedAt reflects the oldest metric's fetchedAt, not wall-clock time.
    - A summary is printed at the end explaining what ran and what was skipped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _utils.github_token import has_github_token


def _run(cmd: list[str]) -> int:
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Run validate -> build -> enrich pipeline for the component gallery."
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation steps (not recommended).",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help=(
            "Skip build step (assumes components/registry/compiled/components.json already exists)."
        ),
    )
    parser.add_argument(
        "--no-github",
        action="store_true",
        help="Skip GitHub enrichment.",
    )
    parser.add_argument(
        "--no-pypi",
        action="store_true",
        help="Skip PyPI enrichment.",
    )
    parser.add_argument(
        "--no-pypistats",
        action="store_true",
        help="Skip PyPI download enrichment (pypistats).",
    )
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help=(
            "Skip all network-dependent steps "
            "(equivalent to --no-github --no-pypi --no-pypistats --no-images). "
            "In this mode, build_catalog preserves the previous generatedAt and tags "
            "the artifact as pipelineMode=offline."
        ),
    )
    parser.add_argument(
        "--no-ranking",
        action="store_true",
        help="Skip ranking computation (not recommended).",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip image URL checking (requires outbound network).",
    )
    parser.add_argument(
        "--allow-enrich-failures",
        action="store_true",
        help="Do not fail the pipeline if some enrichment fetches fail.",
    )
    parser.add_argument(
        "--refresh-older-than-hours",
        type=float,
        default=24.0,
        help=(
            "Only refetch enrichment metrics if existing fetchedAt values are older "
            "than this many hours (default: 24). Use 0 to force refetching everything."
        ),
    )
    parser.add_argument(
        "--stale-threshold-hours",
        type=float,
        default=48.0,
        help=(
            "Forwarded to compute_ranking.py. A metric bucket is considered stale when "
            "its fetchedAt is older than this many hours (default: 48). "
            "Stale metrics cause ranking.isStale=true."
        ),
    )
    parser.add_argument(
        "--enrich-progress-every",
        type=int,
        default=None,
        help=(
            "Forwarded to enrichers as --progress-every N. Default: use each enricher's default."
        ),
    )
    parser.add_argument(
        "--enrich-verbose",
        action="store_true",
        help="Forwarded to enrichers as --verbose (prints per-request failures as they happen).",
    )
    parser.add_argument(
        "--enrich-sleep-github",
        type=float,
        default=None,
        help=(
            "Sleep between unique GitHub API requests in seconds. "
            "Default: 0.2 with GH_TOKEN set, else 1.0 (safer for large catalogs)."
        ),
    )
    parser.add_argument(
        "--enrich-sleep-pypi",
        type=float,
        default=None,
        help=(
            "Sleep between unique PyPI API requests in seconds. "
            "Default: 0.3 (safer for large catalogs)."
        ),
    )
    parser.add_argument(
        "--enrich-sleep-pypistats",
        type=float,
        default=None,
        help=(
            "Sleep between unique pypistats API requests in seconds. "
            "Default: 3.0 (pypistats is aggressively rate-limited)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N components for each enrichment step (debug).",
    )
    args = parser.parse_args(argv)

    script_path = Path(__file__).resolve()
    registry_root = script_path.parents[1]  # components/registry
    scripts_dir = registry_root / "scripts"
    py = sys.executable

    is_offline = bool(args.no_enrich)
    if is_offline:
        args.no_github = True
        args.no_pypi = True
        args.no_pypistats = True
        args.no_images = True

    # Choose conservative enrichment pacing defaults, especially for large catalogs.
    has_gh_token = has_github_token()
    github_sleep = (
        float(args.enrich_sleep_github)
        if args.enrich_sleep_github is not None
        else (0.2 if has_gh_token else 1.0)
    )
    pypi_sleep = float(args.enrich_sleep_pypi) if args.enrich_sleep_pypi is not None else 0.3
    pypistats_sleep = (
        float(args.enrich_sleep_pypistats) if args.enrich_sleep_pypistats is not None else 3.0
    )

    def run_step(name: str, cmd: list[str]) -> int:
        # Flush so headers appear before subprocess output in buffered environments.
        print(f"\n==> {name}\n$ {' '.join(cmd)}", flush=True)
        return _run(cmd)

    steps_run: list[str] = []
    steps_skipped: list[str] = []

    # 1) Validate submissions
    if not args.no_validate:
        rc = run_step("Validate submissions", [py, str(scripts_dir / "validate.py")])
        if rc != 0:
            return rc
        steps_run.append("validate")
    else:
        steps_skipped.append("validate")

    # 1b) Check image URLs (network). Keep this separate from schema validation so
    # CI can enforce it while local/offline runs can skip it.
    if not args.no_images:
        rc = run_step(
            "Check images",
            [py, str(scripts_dir / "enrich_images.py"), "--check-only"],
        )
        if rc != 0:
            return rc
        steps_run.append("images")
    else:
        steps_skipped.append("images")

    # 2) Build compiled artifact
    if not args.no_build:
        build_cmd = [py, str(scripts_dir / "build_catalog.py")]
        if is_offline:
            build_cmd += [
                "--pipeline-mode",
                "offline",
                "--carry-forward-generated-at",
            ]
        rc = run_step("Build compiled catalog", build_cmd)
        if rc != 0:
            return rc
        steps_run.append("build")
    else:
        steps_skipped.append("build")

    # 3) Validate compiled artifact
    if not args.no_validate:
        rc = run_step(
            "Validate compiled catalog",
            [py, str(scripts_dir / "validate.py"), "--compiled"],
        )
        if rc != 0:
            return rc
        steps_run.append("validate-compiled")
    else:
        steps_skipped.append("validate-compiled")

    # 4) Enrich (GitHub/PyPI/pypistats)
    services: list[str] = []
    if not args.no_github:
        services.append("github")
    if not args.no_pypi:
        services.append("pypi")
    if not args.no_pypistats:
        services.append("pypistats")

    if services:
        cmd = [
            py,
            str(scripts_dir / "enrich.py"),
            "--services",
            *services,
            "--sleep-github",
            str(github_sleep),
            "--sleep-pypi",
            str(pypi_sleep),
            "--sleep-pypistats",
            str(pypistats_sleep),
            "--refresh-older-than-hours",
            str(args.refresh_older_than_hours),
        ]
        if args.enrich_progress_every is not None:
            cmd += ["--progress-every", str(args.enrich_progress_every)]
        if args.enrich_verbose:
            cmd += ["--verbose"]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if args.allow_enrich_failures:
            cmd += ["--allow-failures"]
        rc = run_step("Enrich catalog", cmd)
        if rc != 0:
            return rc
        steps_run.append(f"enrich({','.join(services)})")
    else:
        steps_skipped.append("enrich")

    # 6) Compute ranking
    if not args.no_ranking:
        ranking_cmd = [
            py,
            str(scripts_dir / "compute_ranking.py"),
            "--stale-threshold-hours",
            str(args.stale_threshold_hours),
        ]
        if args.limit is not None:
            ranking_cmd += ["--limit", str(args.limit)]
        rc = run_step("Compute ranking", ranking_cmd)
        if rc != 0:
            return rc
        steps_run.append("ranking")
    else:
        steps_skipped.append("ranking")

    # 7) Final validate compiled artifact
    if not args.no_validate:
        rc = run_step(
            "Final validate compiled catalog",
            [py, str(scripts_dir / "validate.py"), "--compiled"],
        )
        if rc != 0:
            return rc
        steps_run.append("final-validate")
    else:
        steps_skipped.append("final-validate")

    # --- Pipeline summary ---
    mode = "OFFLINE" if is_offline else "FULL"
    print(f"\n{'=' * 60}")
    print(f"Pipeline completed ({mode} mode)")
    print(f"{'=' * 60}")
    print(f"  Steps run:     {', '.join(steps_run) if steps_run else '(none)'}")
    print(f"  Steps skipped: {', '.join(steps_skipped) if steps_skipped else '(none)'}")
    if is_offline:
        print(
            "\n  OFFLINE mode notes:"
            "\n    - No network calls were made (enrichment and image checks skipped)."
            "\n    - generatedAt was preserved from the previous artifact."
            "\n    - Artifact tagged with pipelineMode=offline, enrichmentSkipped=true."
            "\n    - Rankings computed on carried-forward metrics; stale metrics are flagged"
            f"\n      (threshold: {args.stale_threshold_hours:.0f}h). Check ranking.isStale"
            "\n      in the output to identify components with outdated scores."
        )
    if "ranking" in steps_run and not services:
        print(
            "\n  WARNING: ranking was computed without fresh enrichment data."
            "\n  Scores reflect carried-forward metrics and may be outdated."
        )
    print()

    print("OK: pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
