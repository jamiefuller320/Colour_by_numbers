"""Safe handling and diagnostics for API keys loaded from environment variables."""

from __future__ import annotations

import os

FAL_KEY_ENV = "FAL_KEY"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
POLLINATIONS_API_KEY_ENV = "POLLINATIONS_API_KEY"

# Env vars that must never appear in logs or error messages in full.
SENSITIVE_ENV_VARS = (
    FAL_KEY_ENV,
    OPENAI_API_KEY_ENV,
    POLLINATIONS_API_KEY_ENV,
)


def api_key_fingerprint(key: str | None) -> str:
    """Safe fingerprint for logs (prefix + length, never the full secret)."""
    normalized = (key or "").strip()
    if not normalized:
        return "(empty)"
    prefix = normalized[:5]
    return f"{prefix}… len={len(normalized)}"


def env_key_status(env_name: str) -> str:
    """Describe whether an env var is unset, blank, or populated."""
    if env_name not in os.environ:
        return "not set"
    normalized = (os.environ.get(env_name) or "").strip()
    if not normalized:
        return "set but blank"
    return api_key_fingerprint(normalized)


def resolve_env_key(*, env_name: str) -> tuple[str, str | None]:
    """Return (api_key, env_name_used) for a single env var."""
    key = (os.environ.get(env_name) or "").strip()
    if key:
        return key, env_name
    return "", None


def api_key_diagnostics(*, env_names: tuple[str, ...] = SENSITIVE_ENV_VARS) -> list[str]:
    """Diagnostic lines for configured env vars without printing secrets."""
    lines = [f"  {name}: {env_key_status(name)}" for name in env_names]
    return lines
