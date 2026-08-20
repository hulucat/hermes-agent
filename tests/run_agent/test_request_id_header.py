"""Request-ID injection for OpenAI-compatible model clients."""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import httpx
import pytest

from agent.process_bootstrap import (
    _request_id_event_hooks,
    build_keepalive_http_client,
    request_id_from_error,
)
from agent.agent_runtime_helpers import dump_api_request_debug
from run_agent import AIAgent


def _request_ids_after_two_requests(event_hooks: dict[str, list]) -> list[str]:
    request_ids: list[str] = []

    def record_request(request: httpx.Request) -> httpx.Response:
        request_ids.append(request.headers.get("X-WM-Request-Id", ""))
        return httpx.Response(204, request=request)

    client = httpx.Client(
        transport=httpx.MockTransport(record_request),
        event_hooks=event_hooks,
    )
    try:
        assert client.get("https://partner.test/v1/models").status_code == 204
        assert client.get("https://partner.test/v1/models").status_code == 204
    finally:
        client.close()
    return request_ids


def _assert_fresh_uuids(request_ids: list[str]) -> None:
    assert len(request_ids) == 2
    assert request_ids[0] != request_ids[1]
    for request_id in request_ids:
        assert str(uuid.UUID(request_id)) == request_id


def test_keepalive_client_generates_a_new_request_id_per_request():
    client = build_keepalive_http_client(
        "https://partner.test/v1",
        request_id_header="X-WM-Request-Id",
    )
    assert client is not None
    try:
        event_hooks = client._event_hooks
    finally:
        client.close()
    _assert_fresh_uuids(_request_ids_after_two_requests(event_hooks))


def test_agent_client_reads_the_configured_request_id_header(monkeypatch):
    import hermes_cli.config as config

    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {"model": {"request_id_header": "X-WM-Request-Id"}},
    )
    client = AIAgent._build_keepalive_http_client("https://partner.test/v1")
    assert client is not None
    try:
        event_hooks = client._event_hooks
    finally:
        client.close()
    _assert_fresh_uuids(_request_ids_after_two_requests(event_hooks))


def test_request_id_from_error_uses_only_the_locally_injected_value():
    hooks = _request_id_event_hooks("X-WM-Request-Id")
    assert hooks is not None
    request = httpx.Request("POST", "https://partner.test/v1/chat/completions")
    for hook in hooks["request"]:
        hook(request)
    response = httpx.Response(403, request=request)
    with pytest.raises(httpx.HTTPStatusError) as caught:
        response.raise_for_status()

    header, request_id = request_id_from_error(caught.value) or (None, None)
    assert header == "X-WM-Request-Id"
    assert request_id is not None
    assert str(uuid.UUID(request_id)) == request_id


def test_request_dump_persists_locally_injected_request_id(tmp_path):
    request = httpx.Request(
        "POST",
        "https://partner.test/v1/chat/completions",
        extensions={
            "hermes.request_id_header": "X-WM-Request-Id",
            "hermes.request_id": "f095e9b2-fcd3-41f4-bb9b-9d17f2bfa857",
        },
    )
    error = Exception("request blocked")
    error.response = httpx.Response(403, request=request)
    error.request_id = "partner-returned-id"
    agent = SimpleNamespace(
        client=SimpleNamespace(api_key="test-key"),
        session_id="session-1",
        base_url="https://partner.test/v1",
        api_mode="chat_completions",
        logs_dir=tmp_path,
        log_prefix="",
        _mask_api_key_for_logs=lambda value: "***" if value else None,
        _vprint=lambda *args, **kwargs: None,
        verbose_logging=False,
    )

    dump_file = dump_api_request_debug(
        agent,
        {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        reason="non_retryable_client_error",
        error=error,
    )

    assert dump_file is not None
    payload = json.loads(dump_file.read_text())
    assert payload["request"]["headers"]["X-WM-Request-Id"] == "f095e9b2-fcd3-41f4-bb9b-9d17f2bfa857"
    assert payload["error"]["request_id"] == "f095e9b2-fcd3-41f4-bb9b-9d17f2bfa857"
    assert payload["error"]["provider_request_id"] == "partner-returned-id"
