"""CLI for GitHub Actions secret-hygiene scanning and schedule gating."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from colour_by_numbers.gha_secret_hygiene import (
    DEFAULT_LOOKBACK_HOURS,
    DEFAULT_WORKFLOWS_DIR,
    decide_schedule_gate,
    scan_workflows,
)


def _cmd_check(args: argparse.Namespace) -> int:
    report = scan_workflows(Path(args.workflows_dir))
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(
            f"scanned={len(report.scanned_files)} "
            f"errors={sum(1 for item in report.findings if item.severity == 'error')} "
            f"warnings={sum(1 for item in report.findings if item.severity == 'warning')}"
        )
        for item in report.findings:
            print(f"{item.severity.upper()} {item.path} [{item.rule}] {item.message}")
        if report.ok:
            print("GHA secret hygiene: OK")
        else:
            print("GHA secret hygiene: FAILED", file=sys.stderr)
    return 0 if report.ok else 1


def _cmd_schedule_gate(args: argparse.Namespace) -> int:
    token = (
        args.token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    ).strip()
    repo = (args.repo or os.environ.get("GITHUB_REPOSITORY") or "").strip()
    force = bool(args.force)
    if not force and (not token or not repo):
        print(
            "GITHUB_TOKEN/GH_TOKEN and GITHUB_REPOSITORY (or --token/--repo) required "
            "unless --force",
            file=sys.stderr,
        )
        return 2
    decision = decide_schedule_gate(
        force=force,
        lookback_hours=int(args.lookback_hours),
        repo=repo or None,
        token=token or None,
    )
    if args.json:
        print(json.dumps(decision.to_dict(), indent=2, sort_keys=True))
    else:
        print(
            f"should_run={str(decision.should_run).lower()} "
            f"reason={decision.reason} "
            f"merged_prs={decision.merged_pr_count} "
            f"workflow_touches={decision.workflow_touch_count} "
            f"lookback_hours={decision.lookback_hours}"
        )
    if args.github_output:
        out = Path(os.environ.get("GITHUB_OUTPUT", ""))
        if not str(out):
            print("GITHUB_OUTPUT is unset", file=sys.stderr)
            return 2
        with out.open("a", encoding="utf-8") as handle:
            handle.write(f"should_run={'true' if decision.should_run else 'false'}\n")
            handle.write(f"reason={decision.reason}\n")
            handle.write(f"merged_pr_count={decision.merged_pr_count}\n")
            handle.write(f"workflow_touch_count={decision.workflow_touch_count}\n")
    # Gate command exits 0 even when skipping — caller checks should_run.
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan GitHub Actions workflows for secret-exposure patterns",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check_p = sub.add_parser("check", help="Scan workflows and exit non-zero on errors")
    check_p.add_argument(
        "--workflows-dir",
        type=Path,
        default=DEFAULT_WORKFLOWS_DIR,
        help="Workflows directory (default: .github/workflows)",
    )
    check_p.add_argument("--json", action="store_true")
    check_p.set_defaults(func=_cmd_check)

    gate_p = sub.add_parser(
        "schedule-gate",
        help="Decide whether a scheduled hygiene run should execute",
    )
    gate_p.add_argument("--force", action="store_true", help="Always run")
    gate_p.add_argument(
        "--lookback-hours",
        type=int,
        default=DEFAULT_LOOKBACK_HOURS,
        help=f"Merged-PR / workflow-touch window (default {DEFAULT_LOOKBACK_HOURS})",
    )
    gate_p.add_argument("--repo", default="", help="owner/name (default GITHUB_REPOSITORY)")
    gate_p.add_argument("--token", default="", help="GitHub token (default GITHUB_TOKEN)")
    gate_p.add_argument("--json", action="store_true")
    gate_p.add_argument(
        "--github-output",
        action="store_true",
        help="Append should_run/reason fields to $GITHUB_OUTPUT",
    )
    gate_p.set_defaults(func=_cmd_schedule_gate)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
