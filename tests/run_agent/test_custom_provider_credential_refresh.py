"""PATCH-026: named custom provider 401 credential self-heal.

WorkMate's managed ``custom:<name>`` provider carries a short-TTL data-plane key
in dotenv. ``_try_refresh_custom_provider_credentials`` re-reads the key on 401
(optionally asking the host backend's loopback bridge to re-sign first) and
rebuilds the shared client in place. These tests mirror the vertex/copilot 401
refresh tests: hermetic, network-free, no process restarts.
"""

import sys
import types
from types import SimpleNamespace

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent  # noqa: E402
from agent.chat_completion_helpers import (  # noqa: E402
    _provider_stream_error_from_text,
    _status_code_from_payload,
    _status_code_from_value,
)

KEY_ENV = "HL_MODEL_API_KEY"


def _build_custom_agent(*, api_key: str = "old-key"):
    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    agent.api_mode = "chat_completions"
    agent.provider = "custom"
    agent.requested_provider = "custom:upstream"
    agent.api_key = api_key
    agent.base_url = "https://upstream.example/v1"
    agent._client_kwargs = {"api_key": api_key, "base_url": agent.base_url}
    agent.client = SimpleNamespace(closed=False)
    calls: dict = {"replace": [], "reapply": 0}

    def _fake_replace(*, reason: str) -> bool:
        calls["replace"].append(reason)
        agent.client = SimpleNamespace(closed=False)
        return True

    def _fake_reapply(*, route_changed: bool) -> None:
        calls["reapply"] += 1

    agent._replace_primary_openai_client = _fake_replace
    agent._reapply_route_client_config = _fake_reapply
    return agent, calls


def _patch_env(monkeypatch, values: dict):
    import agent.credential_pool as credential_pool

    monkeypatch.setattr(
        credential_pool,
        "get_env_prefer_dotenv",
        lambda name: values.get(name, ""),
    )


def _patch_named_provider(monkeypatch, provider: dict | None):
    import hermes_cli.runtime_provider as runtime_provider

    monkeypatch.setattr(
        runtime_provider,
        "_get_named_custom_provider",
        lambda requested: provider,
    )


_PROVIDER = {"key_env": KEY_ENV, "base_url": "https://upstream.example/v1"}


def test_refresh_adopts_changed_dotenv_key(monkeypatch):
    agent, calls = _build_custom_agent()
    _patch_named_provider(monkeypatch, _PROVIDER)
    _patch_env(monkeypatch, {KEY_ENV: "new-key"})

    assert agent._try_refresh_custom_provider_credentials() is True
    assert agent.api_key == "new-key"
    assert agent._client_kwargs["api_key"] == "new-key"
    assert calls["replace"] == ["custom_provider_credential_refresh"]
    assert calls["reapply"] == 1


def test_refresh_calls_bridge_then_adopts(monkeypatch):
    agent, calls = _build_custom_agent()
    _patch_named_provider(monkeypatch, _PROVIDER)
    reads = {"count": 0}
    values = {
        KEY_ENV: "old-key",
        "HL_WM_CREDENTIAL_REFRESH_URL": "http://127.0.0.1:8008/api/wm/internal/credential-refresh",
        "HL_WM_CREDENTIAL_REFRESH_TOKEN": "bridge-token",
    }

    def _flip_after_bridge(name):
        if name == KEY_ENV:
            if reads["count"] >= 1:
                return "new-key"
            reads["count"] += 1
            return "old-key"
        return values.get(name, "")

    import agent.credential_pool as credential_pool

    monkeypatch.setattr(credential_pool, "get_env_prefer_dotenv", _flip_after_bridge)

    posted = {}

    def _fake_post(url, json=None, timeout=None, trust_env=False):  # noqa: A002
        posted["url"] = url
        posted["json"] = json
        return SimpleNamespace(status_code=204)

    import httpx

    monkeypatch.setattr(httpx, "post", _fake_post)

    assert agent._try_refresh_custom_provider_credentials() is True
    assert agent.api_key == "new-key"
    assert posted["json"] == {"token": "bridge-token"}
    assert calls["replace"] == ["custom_provider_credential_refresh"]


def test_refresh_fails_when_bridge_rejected_and_key_unchanged(monkeypatch):
    agent, calls = _build_custom_agent()
    _patch_named_provider(monkeypatch, _PROVIDER)
    _patch_env(
        monkeypatch,
        {
            KEY_ENV: "old-key",
            "HL_WM_CREDENTIAL_REFRESH_URL": "http://127.0.0.1:8008/x",
            "HL_WM_CREDENTIAL_REFRESH_TOKEN": "bridge-token",
        },
    )

    import httpx

    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: SimpleNamespace(status_code=503)
    )

    assert agent._try_refresh_custom_provider_credentials() is False
    assert agent.api_key == "old-key"
    assert calls["replace"] == []


def test_refresh_without_bridge_env_only_rereads(monkeypatch):
    agent, calls = _build_custom_agent()
    _patch_named_provider(monkeypatch, _PROVIDER)
    _patch_env(monkeypatch, {KEY_ENV: "old-key"})

    import httpx

    posted = []
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: posted.append(1) or SimpleNamespace(status_code=204)
    )

    assert agent._try_refresh_custom_provider_credentials() is False
    assert posted == []
    assert agent.api_key == "old-key"


def test_refresh_skips_non_custom_or_missing_key_env(monkeypatch):
    agent, _calls = _build_custom_agent()
    agent.provider = "openai"
    assert agent._try_refresh_custom_provider_credentials() is False

    agent2, _ = _build_custom_agent()
    _patch_named_provider(monkeypatch, {"base_url": "https://upstream.example/v1"})
    assert agent2._try_refresh_custom_provider_credentials() is False


def test_status_code_from_value_maps_unauthorized():
    assert _status_code_from_value("UNAUTHORIZED") == 401
    assert _status_code_from_value("unauthorized ") == 401
    assert _status_code_from_value("403 Forbidden") == 403
    assert _status_code_from_value("RATE_LIMITED_LATER") is None


def test_unauthorized_sse_error_event_classifies_as_401():
    """合作方数据面流中以 SSE error 事件承载鉴权失败,载荷只有 error.code 字符串。

    该事件必须被提取为 status_code=401,否则 PATCH-026 的 401 阶梯收不到流中失效。
    """
    text = (
        'event: error\n'
        'data: {"error":{"code":"UNAUTHORIZED","message":"authentication required"}}\n\n'
    )
    error = _provider_stream_error_from_text(text, "error")
    assert error is not None
    assert error.status_code == 401


def test_status_code_from_payload_maps_error_code_word():
    payload = {"error": {"code": "UNAUTHORIZED", "message": "authentication required"}}
    assert _status_code_from_payload(payload) == 401
