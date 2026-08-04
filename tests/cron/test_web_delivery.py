"""Cron delivery tests for persisted API Server conversations."""

from __future__ import annotations

import sys
import types

from cron.scheduler import _deliver_result, _resolve_delivery_targets
from cron.web_delivery import deliver_web_result, parse_web_delivery_target


def test_resolve_web_and_api_server_origin_targets() -> None:
    assert _resolve_delivery_targets({"deliver": "web:api-abc_123"}) == [
        {"platform": "api_server", "chat_id": "api-abc_123", "thread_id": None}
    ]
    assert _resolve_delivery_targets(
        {
            "deliver": "origin",
            "origin": {"platform": "api_server", "chat_id": "api-abc_123"},
        }
    ) == [{"platform": "api_server", "chat_id": "api-abc_123", "thread_id": None}]


def test_deliver_web_result_appends_marked_event_to_resume_tip(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeSessionDB:
        def resolve_resume_session_id(self, session_id: str) -> str:
            calls.append(("resolve", session_id))
            return "api-current"

        def get_session(self, session_id: str):
            calls.append(("get", session_id))
            return {"id": session_id, "source": "api_server"}

        def append_message(self, **kwargs):
            calls.append(("append", kwargs))

        def close(self) -> None:
            calls.append(("close",))

    monkeypatch.setitem(sys.modules, "hermes_state", types.SimpleNamespace(SessionDB=FakeSessionDB))

    assert deliver_web_result(
        session_id="api-original",
        job_name="每小时摘要",
        content="本小时无异常。",
    ) is None
    assert calls == [
        ("resolve", "api-original"),
        ("get", "api-current"),
        (
            "append",
            {
                "session_id": "api-current",
                "role": "user",
                "tool_name": "automation_result",
                "content": "**每小时摘要**\n\n本小时无异常。",
            },
        ),
        ("close",),
    ]


def test_deliver_result_uses_web_store_without_gateway_platform_config(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_delivery(**kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr("cron.web_delivery.deliver_web_result", fake_delivery)

    assert _deliver_result(
        {"id": "aabbccddeeff", "name": "每小时摘要", "deliver": "web:api-abc"},
        "本小时无异常。",
    ) is None
    assert calls == [
        {
            "session_id": "api-abc",
            "job_name": "每小时摘要",
            "content": "本小时无异常。",
        }
    ]


def test_web_delivery_parser_rejects_cron_separator_injection() -> None:
    assert parse_web_delivery_target("web:api-good") == "api-good"
    assert parse_web_delivery_target("web:api-good,telegram:123") is None
