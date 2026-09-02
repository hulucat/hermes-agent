"""Tests for the non-stream stale-call detector context estimator.

Covers:
- ``estimate_request_context_tokens`` for Chat Completions, Responses API,
  bare lists, and mixed-shape dicts.
- ``AIAgent._compute_non_stream_stale_timeout`` with both legacy ``messages``
  list and full ``api_kwargs`` dicts.
- The May 2026 default-base change (300s -> 90s) and the lowered
  context-tier ceilings (450/600 -> 150/240).
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace



def _write_config(tmp_path: Path, body: str) -> None:
    hermes_home = tmp_path
    (hermes_home / "config.yaml").write_text(body or "{}\n", encoding="utf-8")


def _make_agent(tmp_path: Path, **overrides):
    from run_agent import AIAgent
    kwargs = dict(
        model="gpt-5.5",
        provider="openai-codex",
        api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    kwargs.update(overrides)
    return AIAgent(**kwargs)


# ── estimator ──────────────────────────────────────────────────────────────




def test_estimator_responses_api_input():
    from agent.chat_completion_helpers import estimate_request_context_tokens
    payload = {
        "model": "gpt-5.5",
        "instructions": "i" * 1000,
        "input": "x" * 4000,
        "tools": [{"name": "t", "description": "d" * 200}],
    }
    # input(4000) + instructions(1000) + tools (~stringified) -> well over 1000 tokens
    tokens = estimate_request_context_tokens(payload)
    assert tokens >= 1200, f"Responses API estimator returned {tokens}"






def test_estimator_empty_inputs():
    from agent.chat_completion_helpers import estimate_request_context_tokens
    assert estimate_request_context_tokens({}) == 0
    assert estimate_request_context_tokens([]) == 0
    assert estimate_request_context_tokens(None) == 0




# ── default base + tier scaling ────────────────────────────────────────────


def test_default_base_is_90s(monkeypatch, tmp_path):
    """Default base stale timeout dropped from 300s to 90s (May 2026)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    base, implicit = agent._resolved_api_call_stale_timeout_base()
    assert base == 90.0
    assert implicit is True










def test_explicit_user_config_overrides_default(monkeypatch, tmp_path):
    """If the user explicitly sets a stale_timeout, the new defaults don't apply."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    _write_config(tmp_path, """\
providers:
  openai-codex:
    stale_timeout_seconds: 1800
""")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)

    import importlib
    from hermes_cli import timeouts as to_mod
    importlib.reload(to_mod)

    agent = _make_agent(tmp_path)
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == 1800.0


def test_named_custom_provider_stale_timeout_applies_to_both_watchdogs(
    monkeypatch, tmp_path
):
    """WorkMate's ``providers.upstream`` applies after custom normalization."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    monkeypatch.delenv("HERMES_STREAM_STALE_TIMEOUT", raising=False)
    (tmp_path / ".env").write_text("", encoding="utf-8")
    _write_config(
        tmp_path,
        """\
providers:
  upstream:
    stale_timeout_seconds: 600
model:
  provider: custom:upstream
""",
    )

    from agent.chat_completion_helpers import (
        _derive_stream_stale_timeout,
        _resolve_stream_stale_timeout_base,
    )
    from hermes_cli.timeouts import resolve_provider_stale_timeout

    agent = _make_agent(
        tmp_path,
        provider="custom",
        requested_provider="custom:upstream",
        base_url="https://gateway.example/v1",
    )

    assert resolve_provider_stale_timeout("custom:upstream", "gpt-5.5") == (
        600.0,
        "provider",
    )
    assert agent._compute_non_stream_stale_timeout({"model": "gpt-5.5", "input": "hi"}) == 600.0
    assert _resolve_stream_stale_timeout_base(agent) == (600.0, "provider")
    assert _derive_stream_stale_timeout(agent, {"model": "gpt-5.5", "input": "hi"}) == 600.0


def test_stale_nonstream_log_contains_correlation_fields_without_payload(
    monkeypatch, tmp_path, caplog
):
    """A stale non-stream call emits safe, queryable diagnostics."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
providers:
  upstream:
    stale_timeout_seconds: 600
""",
    )
    from agent.chat_completion_helpers import _report_stale_nonstream_kill

    agent = SimpleNamespace(
        provider="custom",
        requested_provider="custom:upstream",
        model="gpt-5.5",
        session_id="session-1",
        run_id="run-1",
        _current_api_request_id="logical-1",
        _buffer_status=lambda _message: None,
    )
    with caplog.at_level(logging.WARNING, logger="agent.chat_completion_helpers"):
        _report_stale_nonstream_kill(
            agent,
            {"model": "gpt-5.5", "input": "secret-prompt-must-not-appear"},
            600.5,
            600.0,
        )

    event = next(
        record.getMessage()
        for record in caplog.records
        if "event=stale_nonstream_kill" in record.getMessage()
    )
    assert "effective_threshold_seconds=600.000" in event
    assert "threshold_source=provider" in event
    assert "provider=custom" in event
    assert "session_id=session-1" in event
    assert "run_id=run-1" in event
    assert "logical_api_request_id=logical-1" in event
    assert "secret-prompt-must-not-appear" not in event


# ── openai-codex gateway-scale stale floor ────────────────────────────────




def test_openai_codex_stale_floor_tiers():
    from agent.chat_completion_helpers import openai_codex_stale_timeout_floor

    assert openai_codex_stale_timeout_floor(55_000) == 900.0
    assert openai_codex_stale_timeout_floor(120_000) == 1200.0
