# GitHub Actions secret hygiene

This repo is **public**. Any `workflow_run` job runs in the base-repo context and
receives the default `GITHUB_TOKEN` (and repository secrets when configured). Treat
PR head branch names and fork commits as **untrusted input**.

## High-risk patterns (blocked)

| Pattern | Risk | Mitigation in this repo |
|---------|------|-------------------------|
| `workflow_run` after PR CI, then `${{ github.event.workflow_run.head_branch }}` inside `run:` | Shell injection → token / secret theft | Pass via `env:` + strict regex; never `${{ }}` into the script body |
| `workflow_run` autofix that `pip install -e .` from the PR ref | Malicious `pyproject` / package code runs with write token | Install **non-editable** package from `main`, copy trusted scripts to `/tmp`, then check out the PR SHA |
| `workflow_run` without a same-repo gate | Public **fork** PRs trigger privileged jobs | Require `head_repository.full_name == github.repository` |
| Logging full API keys | Key leak via Actions logs | Use `api_key_fingerprint()` / `env_key_status()` only |
| `${{ github.event.inputs.* }}` inside `run:` (string dispatch inputs) | Shell / Python injection → secret theft if a write collaborator or stolen `WORKFLOW_DISPATCH_PAT` can dispatch | Pass all inputs via `env:` + allowlists; never `${{ }}` into the script body |

## Application secrets

Illustration backends read keys from environment variables (never hardcoded):

| Env var | Used for |
|---------|----------|
| `FAL_KEY` | fal.ai Flux (primary generator) |
| `OPENAI_API_KEY` | Optional OpenAI illustration + vision critique |
| `POLLINATIONS_API_KEY` | Legacy Pollinations backend |

For safe logging, use helpers from `colour_by_numbers.api_secrets`:

```python
from colour_by_numbers.api_secrets import api_key_fingerprint, env_key_status

print(f"FAL_KEY: {env_key_status('FAL_KEY')}")  # never print the raw key
```

Local `.env` files are gitignored. Copy `.env.example` to `.env` for development.

## Automated daily check

`gha-secret-hygiene.yml` runs:

1. **Daily** (~06:20 UTC via cron, GitHub `schedule` as backup)
2. **On PRs / pushes** that touch `.github/workflows/**` or the scanner itself
3. **Manual** `workflow_dispatch` with optional `force=true`

The daily job **skips** when no PRs were merged to `main` and no commits touched
`.github/workflows/` in the last **36 hours** (override with `force`). That keeps
noise low while still catching workflow changes introduced by merges.

Local / CI commands:

```bash
cbn-gha-secret-hygiene check
cbn-gha-secret-hygiene schedule-gate --force
pytest -q tests/test_gha_secret_hygiene.py tests/test_api_secrets.py
```

## If an API key may already be compromised

1. Revoke the key at the provider dashboard (fal.ai, OpenAI, etc.).
2. Create a new key; update local `.env`, Streamlit secrets, and any Cloud Agent / CI secrets.
3. Review recent Actions runs for unexpected `workflow_run` jobs on odd branch names.
4. Confirm `main` workflow files were not modified by an unexpected actor.
5. Prefer branch protection on `main` (required reviews / block GITHUB_TOKEN force-push) so a stolen Actions token cannot silently plant a secret-exfiltrating workflow.

## `workflow_dispatch` inputs and PAT blast radius

A stolen dispatch PAT can start any `workflow_dispatch` job that loads secrets.
Free-form string inputs interpolated with `${{ }}` into `run:` enable shell injection
in those jobs (and cross-step `GITHUB_PATH` poisoning into later secret-bearing steps).

Hardening rule: put every `github.event.inputs.*` value into `env:`, quote it in the
shell, and allowlist free-form strings with a strict regex before use. The daily
`cbn-gha-secret-hygiene` scan fails on `dispatch_input_in_run`.

## Current workflows

| Workflow | Secrets | Notes |
|----------|---------|-------|
| `pages.yml` | None | Deploys static `docs/` to GitHub Pages; minimal permissions |
| `gha-secret-hygiene.yml` | None | Static workflow scanner only |

When adding workflows that use `FAL_KEY` or other secrets, keep them on `workflow_dispatch`
/ schedule from `main` only — never in ungated `workflow_run` jobs triggered by fork PRs.
