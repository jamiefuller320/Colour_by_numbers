"""Tests for API key env resolution and safe fingerprinting."""

from __future__ import annotations

from colour_by_numbers.api_secrets import (
    FAL_KEY_ENV,
    OPENAI_API_KEY_ENV,
    POLLINATIONS_API_KEY_ENV,
    api_key_diagnostics,
    api_key_fingerprint,
    resolve_env_key,
)


def test_fingerprint_masks_secret() -> None:
    assert api_key_fingerprint("sk-abcdefghijklmnop") == "sk-ab… len=19"
    assert api_key_fingerprint("") == "(empty)"
    assert api_key_fingerprint("   ") == "(empty)"


def test_resolve_env_key() -> None:
    import os

    os.environ[FAL_KEY_ENV] = "fal-test-key"
    try:
        key, source = resolve_env_key(env_name=FAL_KEY_ENV)
        assert key == "fal-test-key"
        assert source == FAL_KEY_ENV
    finally:
        del os.environ[FAL_KEY_ENV]


def test_resolve_env_key_missing() -> None:
    import os

    os.environ.pop(FAL_KEY_ENV, None)
    key, source = resolve_env_key(env_name=FAL_KEY_ENV)
    assert key == ""
    assert source is None


def test_diagnostics_lists_env_vars(monkeypatch) -> None:
    monkeypatch.setenv(FAL_KEY_ENV, "fal_secret_key_value")
    monkeypatch.delenv(OPENAI_API_KEY_ENV, raising=False)
    monkeypatch.delenv(POLLINATIONS_API_KEY_ENV, raising=False)
    lines = api_key_diagnostics()
    assert any("FAL_KEY" in line and "fal_s…" in line for line in lines)
    assert any("OPENAI_API_KEY: not set" in line for line in lines)
    assert any("POLLINATIONS_API_KEY: not set" in line for line in lines)
