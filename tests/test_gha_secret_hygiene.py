"""Guards against reintroducing GHA secret-exposure patterns."""

from __future__ import annotations

from pathlib import Path

from colour_by_numbers.gha_secret_hygiene import (
    decide_schedule_gate,
    extract_run_blocks,
    scan_workflow_text,
    scan_workflows,
)

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def test_scan_workflows_clean_on_repo() -> None:
    report = scan_workflows(WORKFLOWS)
    errors = [item for item in report.findings if item.severity == "error"]
    assert errors == [], errors
    assert report.ok


def test_pages_workflow_has_minimal_permissions() -> None:
    text = (WORKFLOWS / "pages.yml").read_text(encoding="utf-8")
    findings = scan_workflow_text("pages.yml", text)
    assert not any(item.severity == "error" for item in findings)
    assert "contents: read" in text
    assert "id-token: write" in text


def test_detects_untrusted_expr_in_run_block() -> None:
    evil = """
name: Evil
on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]
jobs:
  x:
    runs-on: ubuntu-latest
    steps:
      - run: |
          echo "${{ github.event.workflow_run.head_branch }}"
"""
    findings = scan_workflow_text("evil.yml", evil)
    assert any(item.rule == "untrusted_expr_in_run" for item in findings)


def test_detects_dispatch_input_in_run_block() -> None:
    evil = """
name: Evil
on:
  workflow_dispatch:
    inputs:
      task_id:
        type: string
jobs:
  x:
    runs-on: ubuntu-latest
    steps:
      - run: |
          echo "${{ github.event.inputs.task_id }}"
"""
    findings = scan_workflow_text("evil.yml", evil)
    assert any(item.rule == "dispatch_input_in_run" for item in findings)


def test_dispatch_input_via_env_is_allowed() -> None:
    ok = """
name: Ok
on:
  workflow_dispatch:
    inputs:
      task_id:
        type: string
jobs:
  x:
    runs-on: ubuntu-latest
    steps:
      - env:
          TASK_ID: ${{ github.event.inputs.task_id }}
        run: |
          echo "$TASK_ID"
"""
    findings = scan_workflow_text("ok.yml", ok)
    assert not any(item.rule == "dispatch_input_in_run" for item in findings)


def test_detects_missing_same_repo_gate_on_head_checkout() -> None:
    evil = """
name: Evil
on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]
jobs:
  x:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
        with:
          ref: ${{ github.event.workflow_run.head_sha }}
"""
    findings = scan_workflow_text("evil.yml", evil)
    assert any(item.rule == "workflow_run_missing_same_repo_gate" for item in findings)


def test_schedule_gate_force_and_recent_changes() -> None:
    forced = decide_schedule_gate(force=True)
    assert forced.should_run and forced.reason == "force"

    skip = decide_schedule_gate(
        force=False,
        merged_pr_count=0,
        workflow_touch_count=0,
        lookback_hours=36,
    )
    assert not skip.should_run
    assert skip.reason == "no_recent_merges_or_workflow_changes"

    run = decide_schedule_gate(
        force=False,
        merged_pr_count=2,
        workflow_touch_count=0,
        lookback_hours=36,
    )
    assert run.should_run and run.reason == "recent_main_changes"


def test_extract_run_blocks_ignores_env() -> None:
    text = """
jobs:
  x:
    steps:
      - env:
          BRANCH: ${{ github.event.workflow_run.head_branch }}
        run: |
          echo "$BRANCH"
"""
    blocks = extract_run_blocks(text)
    assert len(blocks) == 1
    assert "$BRANCH" in blocks[0]
    assert "github.event" not in blocks[0]
