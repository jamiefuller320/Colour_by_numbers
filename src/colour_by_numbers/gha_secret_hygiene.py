"""Static checks for GitHub Actions secret-exposure patterns.

Public-repo ``workflow_run`` jobs run with base-repo privileges. This module
flags dangerous patterns so CI / a daily scheduled job can fail closed.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_WORKFLOWS_DIR = Path(".github/workflows")
DEFAULT_LOOKBACK_HOURS = 36

# Privileged workflows that must keep explicit same-repo / trusted-install gates.
REQUIRED_SAME_REPO_WORKFLOWS: tuple[str, ...] = ()

# Expressions that must never appear inside ``run: |`` / ``run: >`` bodies.
_UNTRUSTED_RUN_INTERP = re.compile(
    r"\$\{\{\s*github\.event\."
    r"(?:pull_request\.head\.ref|workflow_run\.head_branch|workflow_run\.name)"
    r"\s*\}\}"
)

# workflow_dispatch inputs interpolated into run: enable shell injection when a
# write token / WORKFLOW_DISPATCH_PAT can dispatch (and later steps may load secrets).
_DISPATCH_INPUT_IN_RUN = re.compile(r"\$\{\{\s*github\.event\.inputs\.")

_WORKFLOW_RUN_TRIGGER = re.compile(r"(?m)^\s*workflow_run\s*:")
_HEAD_REPO_GATE = re.compile(r"head_repository\.full_name\s*==\s*github\.repository")
_EDITABLE_PIP = re.compile(r"pip\s+install\s+-e\b")
_USES_WORKFLOW_RUN_HEAD = re.compile(r"github\.event\.workflow_run\.(?:head_branch|head_sha)")
_CHECKOUT_UNTRUSTED_REF = re.compile(
    r"ref:\s*\$\{\{\s*github\.event\.workflow_run\.(?:head_branch|head_sha)\s*\}\}"
)


@dataclass(frozen=True)
class HygieneFinding:
    severity: str  # "error" | "warning"
    path: str
    rule: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class HygieneReport:
    findings: list[HygieneFinding] = field(default_factory=list)
    scanned_files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(item.severity == "error" for item in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error_count": sum(1 for item in self.findings if item.severity == "error"),
            "warning_count": sum(1 for item in self.findings if item.severity == "warning"),
            "scanned_files": list(self.scanned_files),
            "findings": [item.to_dict() for item in self.findings],
        }


@dataclass(frozen=True)
class ScheduleGateDecision:
    should_run: bool
    reason: str
    merged_pr_count: int = 0
    workflow_touch_count: int = 0
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def iter_workflow_files(workflows_dir: Path | None = None) -> list[Path]:
    root = Path(workflows_dir or DEFAULT_WORKFLOWS_DIR)
    if not root.is_dir():
        return []
    return sorted(path for path in root.glob("*.yml") if path.is_file()) + sorted(
        path for path in root.glob("*.yaml") if path.is_file()
    )


def extract_run_blocks(text: str) -> list[str]:
    """Return bodies of ``run: |`` / ``run: >`` steps (shell scripts only)."""
    blocks: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        # Match both `run: |` and `- run: |` step forms.
        match = re.match(r"^(\s*)(?:-\s+)?run:\s*[|>]\s*$", lines[i])
        if not match:
            i += 1
            continue
        indent = len(match.group(1))
        i += 1
        body: list[str] = []
        while i < len(lines):
            line = lines[i]
            if line.strip() == "":
                body.append(line)
                i += 1
                continue
            leading = len(line) - len(line.lstrip(" "))
            if leading <= indent:
                break
            body.append(line)
            i += 1
        blocks.append("\n".join(body))
    return blocks


def scan_workflow_text(path: str, text: str) -> list[HygieneFinding]:
    findings: list[HygieneFinding] = []
    name = Path(path).name
    has_workflow_run = bool(_WORKFLOW_RUN_TRIGGER.search(text))
    uses_workflow_run_head = bool(_USES_WORKFLOW_RUN_HEAD.search(text))
    checks_out_untrusted_ref = bool(_CHECKOUT_UNTRUSTED_REF.search(text))
    has_same_repo_gate = bool(_HEAD_REPO_GATE.search(text))

    for block in extract_run_blocks(text):
        hit = _UNTRUSTED_RUN_INTERP.search(block)
        if hit:
            findings.append(
                HygieneFinding(
                    severity="error",
                    path=path,
                    rule="untrusted_expr_in_run",
                    message=(
                        f"Untrusted GitHub expression inside run script: {hit.group(0)}. "
                        "Pass via env: and validate with a strict regex."
                    ),
                )
            )
        input_hit = _DISPATCH_INPUT_IN_RUN.search(block)
        if input_hit:
            findings.append(
                HygieneFinding(
                    severity="error",
                    path=path,
                    rule="dispatch_input_in_run",
                    message=(
                        f"workflow_dispatch input interpolated inside run script: "
                        f"{input_hit.group(0)}. Pass via env: (and allowlist) so a "
                        "stolen dispatch PAT cannot shell-inject into secret-bearing jobs."
                    ),
                )
            )

    if has_workflow_run and uses_workflow_run_head and not has_same_repo_gate:
        findings.append(
            HygieneFinding(
                severity="error",
                path=path,
                rule="workflow_run_missing_same_repo_gate",
                message=(
                    "workflow_run uses PR head fields without "
                    "head_repository.full_name == github.repository"
                ),
            )
        )

    if name in REQUIRED_SAME_REPO_WORKFLOWS and not has_same_repo_gate:
        findings.append(
            HygieneFinding(
                severity="error",
                path=path,
                rule="required_same_repo_gate",
                message=f"{name} must keep head_repository.full_name == github.repository",
            )
        )

    if name == "ci-pr-autofix.yml":
        if _EDITABLE_PIP.search(text):
            findings.append(
                HygieneFinding(
                    severity="error",
                    path=path,
                    rule="editable_install_from_pr",
                    message=(
                        "ci-pr-autofix.yml must not use pip install -e; install a "
                        "non-editable package from main before checking out PR code"
                    ),
                )
            )
        if 'pip install ".[dev]"' not in text and "pip install '.[dev]'" not in text:
            findings.append(
                HygieneFinding(
                    severity="error",
                    path=path,
                    rule="trusted_install_missing",
                    message='ci-pr-autofix.yml must pip install ".[dev]" from main',
                )
            )
        if "cp scripts/ci_pr_autofix.py /tmp/ci_pr_autofix.py" not in text:
            findings.append(
                HygieneFinding(
                    severity="error",
                    path=path,
                    rule="trusted_script_copy_missing",
                    message="ci-pr-autofix.yml must copy autofix scripts from main to /tmp",
                )
            )

    if name == "engineering-auto-merge.yml":
        if '--branch "${{ github.event.workflow_run.head_branch }}"' in text:
            findings.append(
                HygieneFinding(
                    severity="error",
                    path=path,
                    rule="branch_expr_in_shell",
                    message=(
                        "engineering-auto-merge.yml must pass head_branch via env, "
                        "not interpolate it into the shell command"
                    ),
                )
            )
        if r"^cursor/eng-[0-9]{8}-[0-9]{2}-1de3$" not in text:
            findings.append(
                HygieneFinding(
                    severity="error",
                    path=path,
                    rule="eng_branch_regex_missing",
                    message=(
                        "engineering-auto-merge.yml must validate "
                        "cursor/eng-YYYYMMDD-NN-1de3 branch names"
                    ),
                )
            )

    if has_workflow_run and _EDITABLE_PIP.search(text) and checks_out_untrusted_ref:
        findings.append(
            HygieneFinding(
                severity="error",
                path=path,
                rule="editable_install_with_untrusted_checkout",
                message=(
                    "workflow_run job checks out PR head with ref: and uses pip install -e "
                    "(package code from the PR runs with write token)"
                ),
            )
        )

    return findings


def scan_workflows(workflows_dir: Path | None = None) -> HygieneReport:
    report = HygieneReport()
    for path in iter_workflow_files(workflows_dir):
        rel = str(path).replace("\\", "/")
        report.scanned_files.append(rel)
        text = path.read_text(encoding="utf-8")
        report.findings.extend(scan_workflow_text(rel, text))
    return report


def _gh_api_json(
    url: str,
    *,
    token: str,
    accept: str = "application/vnd.github+json",
) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": accept,
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "colour-by-numbers-gha-secret-hygiene",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {exc.code} for {url}: {detail}") from exc


def _parse_repo(repo: str) -> tuple[str, str]:
    parts = (repo or "").strip().split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"repo must be owner/name, got {repo!r}")
    return parts[0], parts[1]


def count_merged_prs_since(
    *,
    repo: str,
    token: str,
    since: datetime,
    base: str = "main",
) -> int:
    owner, name = _parse_repo(repo)
    since_iso = since.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    query = f"repo:{owner}/{name} is:pr is:merged base:{base} merged:>={since_iso}"
    url = "https://api.github.com/search/issues?" + urllib.parse.urlencode(
        {"q": query, "per_page": "1"}
    )
    payload = _gh_api_json(url, token=token)
    return int(payload.get("total_count") or 0)


def count_workflow_commits_since(
    *,
    repo: str,
    token: str,
    since: datetime,
    branch: str = "main",
    path: str = ".github/workflows",
) -> int:
    owner, name = _parse_repo(repo)
    since_iso = since.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = f"https://api.github.com/repos/{owner}/{name}/commits?" + urllib.parse.urlencode(
        {
            "sha": branch,
            "since": since_iso,
            "path": path,
            "per_page": "100",
        }
    )
    payload = _gh_api_json(url, token=token)
    if not isinstance(payload, list):
        return 0
    return len(payload)


def decide_schedule_gate(
    *,
    force: bool = False,
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
    merged_pr_count: int | None = None,
    workflow_touch_count: int | None = None,
    repo: str | None = None,
    token: str | None = None,
    now: datetime | None = None,
) -> ScheduleGateDecision:
    """Return whether a scheduled hygiene run should execute.

    Runs when forced, or when main gained merged PRs / workflow-file commits
    inside the lookback window (default 36h so a daily job catches prior-day merges).
    """
    hours = max(1, int(lookback_hours))
    if force:
        return ScheduleGateDecision(
            should_run=True,
            reason="force",
            lookback_hours=hours,
            merged_pr_count=int(merged_pr_count or 0),
            workflow_touch_count=int(workflow_touch_count or 0),
        )

    stamp = now or datetime.now(UTC)
    since = stamp - timedelta(hours=hours)

    if merged_pr_count is None or workflow_touch_count is None:
        if not repo or not token:
            raise ValueError("repo and token are required unless counts are provided")
        if merged_pr_count is None:
            merged_pr_count = count_merged_prs_since(repo=repo, token=token, since=since)
        if workflow_touch_count is None:
            workflow_touch_count = count_workflow_commits_since(repo=repo, token=token, since=since)

    merged = int(merged_pr_count or 0)
    touches = int(workflow_touch_count or 0)
    if merged > 0 or touches > 0:
        return ScheduleGateDecision(
            should_run=True,
            reason="recent_main_changes",
            merged_pr_count=merged,
            workflow_touch_count=touches,
            lookback_hours=hours,
        )
    return ScheduleGateDecision(
        should_run=False,
        reason="no_recent_merges_or_workflow_changes",
        merged_pr_count=merged,
        workflow_touch_count=touches,
        lookback_hours=hours,
    )


__all__ = [
    "DEFAULT_LOOKBACK_HOURS",
    "DEFAULT_WORKFLOWS_DIR",
    "HygieneFinding",
    "HygieneReport",
    "REQUIRED_SAME_REPO_WORKFLOWS",
    "ScheduleGateDecision",
    "count_merged_prs_since",
    "count_workflow_commits_since",
    "decide_schedule_gate",
    "extract_run_blocks",
    "iter_workflow_files",
    "scan_workflow_text",
    "scan_workflows",
]
