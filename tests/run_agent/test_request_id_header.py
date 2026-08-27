"""Per-attempt request IDs for OpenAI-compatible model transports."""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import httpx
import pytest

from agent.agent_runtime_helpers import dump_api_request_debug
from agent.process_bootstrap import (
    _request_id_event_hooks,
    build_keepalive_http_client,
    request_id_from_error,
)
from run_agent import AIAgent


def _assert_uuid(value: str) -> None:
    assert str(uuid.UUID(value)) == value


def test_sync_client_generates_a_fresh_request_id_per_attempt():
    ids: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        ids.append(request.headers["X-WM-Request-Id"])
        return httpx.Response(204, request=request)

    hooks = _request_id_event_hooks("X-WM-Request-Id")
    assert hooks is not None
    with httpx.Client(transport=httpx.MockTransport(respond), event_hooks=hooks) as client:
        client.get("https://partner.test/v1/models")
        client.get("https://partner.test/v1/models")

    assert len(set(ids)) == 2
    for request_id in ids:
        _assert_uuid(request_id)


@pytest.mark.asyncio
async def test_async_client_generates_a_fresh_request_id_per_attempt():
    ids: list[str] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        ids.append(request.headers["X-WM-Request-Id"])
        return httpx.Response(204, request=request)

    hooks = _request_id_event_hooks("X-WM-Request-Id", async_mode=True)
    assert hooks is not None
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
        event_hooks=hooks,
    ) as client:
        await client.get("https://partner.test/v1/models")
        await client.get("https://partner.test/v1/models")

    assert len(set(ids)) == 2
    for request_id in ids:
        _assert_uuid(request_id)


def test_invalid_header_disables_request_hook():
    client = build_keepalive_http_client(
        "https://partner.test/v1",
        request_id_header="bad header\r\nX-Evil",
    )
    assert client is not None
    try:
        assert not client._event_hooks["request"]
    finally:
        client.close()


def test_error_summary_and_dump_use_only_local_request_id(tmp_path):
    request = httpx.Request(
        "POST",
        "https://partner.test/v1/chat/completions",
        extensions={
            "hermes.request_id_header": "X-WM-Request-Id",
            "hermes.request_id": "f095e9b2-fcd3-41f4-bb9b-9d17f2bfa857",
        },
    )
    error = Exception("request blocked")
    error.status_code = 403
    error.body = {"error": {"message": "request rejected"}}
    error.response = httpx.Response(403, request=request)
    error.request_id = "provider-controlled-id"

    assert request_id_from_error(error) == (
        "X-WM-Request-Id",
        "f095e9b2-fcd3-41f4-bb9b-9d17f2bfa857",
    )
    summary = AIAgent._summarize_api_error(error)
    assert "Request ID: f095e9b2-fcd3-41f4-bb9b-9d17f2bfa857" in summary
    assert "provider-controlled-id" not in summary

    agent = SimpleNamespace(
        client=SimpleNamespace(api_key="secret"),
        session_id="session-1",
        base_url="https://partner.test/v1",
        api_mode="chat_completions",
        logs_dir=tmp_path,
        log_prefix="",
        _mask_api_key_for_logs=lambda value: "***",
        _vprint=lambda *args, **kwargs: None,
        verbose_logging=False,
    )
    dump_file = dump_api_request_debug(
        agent,
        {"model": "test-model", "messages": []},
        reason="test",
        error=error,
    )
    assert dump_file is not None
    payload = json.loads(dump_file.read_text())
    assert payload["request"]["headers"]["X-WM-Request-Id"].startswith("f095")
    assert payload["error"]["request_id"].startswith("f095")
    assert payload["error"]["provider_request_id"] == "provider-controlled-id"
