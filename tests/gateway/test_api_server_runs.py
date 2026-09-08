"""Tests for /v1/runs endpoints: start, status, events, steer, and stop.

Covers:
- POST /v1/runs — start a run (202)
- GET /v1/runs/{run_id} — poll run status
- GET /v1/runs/{run_id}/events — SSE event stream
- POST /v1/runs/{run_id}/steer — inject guidance into a running agent
- POST /v1/runs/{run_id}/stop — interrupt a running agent
- Auth, error handling, and cleanup
"""

import asyncio
import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    MAX_RUN_REASONING_SEGMENT_CHARS,
    _approval_event_choices,
    _extract_x_wm_request_id,
    _extract_x_wm_tool_error,
    cors_middleware,
    security_headers_middleware,
)
from tools import approval as approval_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("smart_denied", "allow_permanent", "expected"),
    [
        (False, True, ["once", "session", "always", "deny"]),
        (False, False, ["once", "session", "deny"]),
        (True, True, ["once", "deny"]),
        (True, False, ["once", "deny"]),
    ],
)
def test_approval_event_choices_follow_backend_capabilities(
    smart_denied, allow_permanent, expected
):
    assert _approval_event_choices(
        smart_denied=smart_denied,
        allow_permanent=allow_permanent,
    ) == expected


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            {
                "success": False,
                "error": "failed",
                "x_wm_request_id": "11111111-1111-4111-8111-111111111111",
            },
            "11111111-1111-4111-8111-111111111111",
        ),
        (
            {
                "results": [
                    {
                        "error": "failed",
                        "x_wm_request_id": "22222222-2222-4222-8222-222222222222",
                    }
                ]
            },
            "22222222-2222-4222-8222-222222222222",
        ),
        ({"x_wm_request_id": "not-a-uuid"}, None),
        (
            {
                "results": [
                    {
                        "error": "first",
                        "x_wm_request_id": "33333333-3333-4333-8333-333333333333",
                    },
                    {
                        "error": "second",
                        "x_wm_request_id": "44444444-4444-4444-8444-444444444444",
                    },
                ]
            },
            None,
        ),
    ],
)
def test_extract_x_wm_request_id_accepts_one_canonical_uuid4(result, expected):
    assert _extract_x_wm_request_id(json.dumps(result)) == expected


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            {
                "contract_version": "0.2",
                "ok": False,
                "error": {
                    "code": "PYTHON_COMPUTE_OUTPUT_REJECTED",
                    "message": "rejected",
                    "retryable": False,
                    "details": {"reason": "required_output_missing"},
                },
            },
            {
                "code": "PYTHON_COMPUTE_OUTPUT_REJECTED",
                "message": "rejected",
                "reason": "required_output_missing",
            },
        ),
        # 成功信封(ok:true 或无 error)不提取
        ({"ok": True, "error": None}, None),
        ({"ok": False, "error": {"code": "WEB_TOOL_REJECTED", "message": "x"}}, None),
        ({"ok": False, "error": {"code": "PYTHON_COMPUTE_X", "message": ""}}, None),
        ({"error": {"code": "PYTHON_COMPUTE_X", "message": "x"}}, None),
        ("not-json", None),
        (None, None),
    ],
)
def test_extract_x_wm_tool_error_only_accepts_managed_envelope(result, expected):
    # dict 直接传入;字符串走 JSON 解析路径("not-json" 覆盖解析失败)。
    assert _extract_x_wm_tool_error(result) == expected


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    """Create an adapter with optional API key."""
    extra = {}
    if api_key:
        extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=extra)
    adapter = APIServerAdapter(config)
    return adapter


def _create_runs_app(adapter: APIServerAdapter) -> web.Application:
    """Create an aiohttp app with /v1/runs routes registered."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_get(
        "/v1/runs/{run_id}/model-usage", adapter._handle_run_model_usage
    )
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    app.router.add_post("/v1/runs/{run_id}/steer", adapter._handle_steer_run)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    return app


def _run_body(**fields):
    body = {
        "input": "hello",
        "workspace_root": str(Path.cwd()),
        "mode": "execute",
    }
    body.update(fields)
    return body


def _make_slow_agent(**kwargs):
    """Create a mock agent that blocks in run_conversation until interrupted.

    Returns (mock_agent, agent_ready_event, interrupt_event) where
    agent_ready_event is set once run_conversation starts, and
    interrupt_event is set when interrupt() is called.
    """
    ready = threading.Event()
    interrupted = threading.Event()

    mock_agent = MagicMock()

    def _do_interrupt(message=None):
        interrupted.set()

    mock_agent.interrupt = MagicMock(side_effect=_do_interrupt)

    def _slow_run(user_message=None, conversation_history=None, task_id=None):
        ready.set()
        # Block until interrupt() is called
        interrupted.wait(timeout=10)
        return {"final_response": "interrupted"}

    mock_agent.run_conversation.side_effect = _slow_run
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0

    return mock_agent, ready, interrupted


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


# ---------------------------------------------------------------------------
# POST /v1/runs — start a run
# ---------------------------------------------------------------------------


class TestStartRun:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body,error_field",
        [
            ({"input": "hello", "mode": "execute"}, "workspace_root"),
            (
                {"input": "hello", "workspace_root": "relative", "mode": "execute"},
                "workspace_root",
            ),
            (
                {
                    "input": "hello",
                    "workspace_root": str(Path.cwd()),
                    "mode": "edit",
                },
                "mode",
            ),
            (
                {
                    "input": "hello",
                    "workspace_root": str(Path.cwd()),
                    "mode": "ask",
                },
                "mode",
            ),
            (
                {
                    "input": "hello",
                    "workspace_root": str(Path.cwd()),
                    "mode": "craft",
                },
                "mode",
            ),
            (
                {
                    "input": "hello",
                    "workspace_root": str(Path.cwd()),
                    "mode": "execute",
                    "include_reasoning": "true",
                },
                "include_reasoning",
            ),
        ],
    )
    async def test_start_validates_workmate_run_fields(
        self, adapter, body, error_field
    ):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post("/v1/runs", json=body)
            payload = await response.json()

        assert response.status == 400
        assert error_field in payload["error"]["message"]
        assert adapter._run_statuses == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["execute", "plan", "visualize"])
    async def test_start_returns_202(self, adapter, mode):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json=_run_body(mode=mode))
                assert resp.status == 202
                data = await resp.json()
                assert data["status"] == "started"
                assert data["run_id"].startswith("run_")

                status_resp = await cli.get(f"/v1/runs/{data['run_id']}")
                assert status_resp.status == 200
                status = await status_resp.json()
                assert status["run_id"] == data["run_id"]
                assert status["status"] in {"queued", "running", "completed"}
                assert status["object"] == "hermes.run"

    @pytest.mark.asyncio
    async def test_start_binds_chat_id_for_delegation_wake_target(self, adapter):
        """/v1/runs must bind the raw session id as the api_server chat_id
        (like every other agent-entry route does via _run_agent): the async
        delegation dispatch reads HERMES_SESSION_CHAT_ID to pick its wake
        self-post target, and an empty binding forces background delegations
        on this route back to synchronous execution."""
        app = _create_runs_app(adapter)
        captured = {}

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()

                def _capture_run(user_message=None, conversation_history=None, task_id=None):
                    from tools.async_delegation import _current_origin_session_id

                    captured["origin_session_id"] = _current_origin_session_id()
                    return {"final_response": "done"}

                mock_agent.run_conversation.side_effect = _capture_run
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json=_run_body(session_id="runs-raw-sid"),
                )
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(40):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert captured.get("origin_session_id") == "runs-raw-sid", (
            "runs route must bind chat_id so delegation dispatch sees a wake target"
        )


    @pytest.mark.asyncio
    async def test_start_rejects_conflicting_route_and_request_provider(self):
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "model_routes": {
                        "alias": {
                            "model": "route/model",
                            "provider": "openrouter",
                        }
                    }
                },
            )
        )
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                resp = await cli.post(
                    "/v1/runs",
                    json=_run_body(model="alias", provider="minimax"),
                )
                data = await resp.json()

        assert resp.status == 400
        assert "provider" in data["error"]["message"].lower()
        assert adapter._run_streams == {}
        assert adapter._run_statuses == {}
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_passes_request_model_provider_options_to_create_agent(self, adapter):
        app = _create_runs_app(adapter)
        model_options = {"reasoning_effort": "medium", "service_tier": "priority"}
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json=_run_body(
                        model="MiniMax-M3",
                        provider="minimax",
                        model_options=model_options,
                    ),
                )
                assert resp.status == 202
                for _ in range(20):
                    if mock_create.call_args is not None:
                        break
                    await asyncio.sleep(0.05)

        kwargs = mock_create.call_args.kwargs
        assert kwargs["requested_model"] == "MiniMax-M3"
        assert kwargs["requested_provider"] == "minimax"
        assert kwargs["model_options"] == model_options


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id} — poll run status
# ---------------------------------------------------------------------------


class TestRunStatus:

    @pytest.mark.asyncio
    async def test_status_reflects_explicit_session_id(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json=_run_body(session_id="space-session"),
                )
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(20):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

                mock_agent.run_conversation.assert_called_once()
                assert mock_agent.run_conversation.call_args.kwargs["task_id"] == run_id
                assert status["session_id"] == "space-session"


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id}/events — SSE event stream
# ---------------------------------------------------------------------------


class TestRunEvents:
    @pytest.mark.asyncio
    async def test_shared_session_runs_isolate_workspace_mode_and_cleanup(
        self, adapter, tmp_path
    ):
        workspaces = [tmp_path / "one", tmp_path / "two"]
        for workspace in workspaces:
            workspace.mkdir()
        barrier = threading.Barrier(2)
        captured: list[tuple[str, str, str, str]] = []

        def _create_agent(**_kwargs):
            mock_agent = MagicMock()

            def _run_conversation(**run_kwargs):
                from agent.runtime_cwd import resolve_agent_cwd
                from tools.terminal_tool import resolve_task_overrides

                task_id = run_kwargs["task_id"]
                overrides = resolve_task_overrides(task_id)
                captured.append(
                    (
                        task_id,
                        overrides["cwd"],
                        overrides["wm_mode"],
                        str(resolve_agent_cwd()),
                    )
                )
                barrier.wait(timeout=5)
                return {"final_response": "done"}

            mock_agent.run_conversation.side_effect = _run_conversation
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            return mock_agent

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=_create_agent):
                responses = [
                    await cli.post(
                        "/v1/runs",
                        json=_run_body(
                            workspace_root=str(workspaces[index]),
                            mode=mode,
                            session_id="shared-session",
                        ),
                    )
                    for index, mode in enumerate(("execute", "plan"))
                ]
                run_ids = [(await response.json())["run_id"] for response in responses]
                for run_id in run_ids:
                    await (await cli.get(f"/v1/runs/{run_id}/events")).text()

        assert {item[0] for item in captured} == set(run_ids)
        assert {(item[1], item[2], item[3]) for item in captured} == {
            (str(workspaces[0].resolve()), "execute", str(workspaces[0].resolve())),
            (str(workspaces[1].resolve()), "plan", str(workspaces[1].resolve())),
        }
        from tools.terminal_tool import resolve_task_overrides

        assert all(resolve_task_overrides(run_id) == {} for run_id in run_ids)

    @pytest.mark.asyncio
    async def test_reasoning_and_message_events_are_ordered_redacted_and_sequenced(
        self, adapter
    ):
        def _create_agent(**kwargs):
            mock_agent = MagicMock()

            def _run_conversation(**_run_kwargs):
                kwargs["reasoning_callback"](
                    "Inspect OPENAI_API_KEY=sk-live-secret-1234567890"
                )
                kwargs["tool_progress_callback"]("tool.started", "web_search")
                kwargs["reasoning_callback"]("Synthesize")
                kwargs["stream_delta_callback"]("first")
                kwargs["stream_delta_callback"]("second")
                return {"final_response": "firstsecond"}

            mock_agent.run_conversation.side_effect = _run_conversation
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            return mock_agent

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=_create_agent):
                response = await cli.post(
                    "/v1/runs",
                    json=_run_body(include_reasoning=True),
                )
                run_id = (await response.json())["run_id"]
                body = await (await cli.get(f"/v1/runs/{run_id}/events")).text()

        events = [
            json.loads(line.removeprefix("data: "))
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        assert "sk-live-secret-1234567890" not in body
        assert [event["event"] for event in events] == [
            "reasoning.started",
            "reasoning.completed",
            "tool.started",
            "reasoning.started",
            "reasoning.completed",
            "message.delta",
            "message.delta",
            "run.completed",
        ]
        deltas = [event for event in events if event["event"] == "message.delta"]
        assert [event["seq"] for event in deltas] == [1, 2]
        segments = [
            event for event in events if event["event"] == "reasoning.completed"
        ]
        assert [event["segment_seq"] for event in segments] == [1, 2]

    @pytest.mark.asyncio
    async def test_reasoning_segment_is_truncated_and_disabled_by_default(
        self, adapter
    ):
        captured = {}

        def _create_agent(**kwargs):
            captured.update(kwargs)
            mock_agent = MagicMock()

            def _run_conversation(**_run_kwargs):
                if "reasoning_callback" in kwargs:
                    kwargs["reasoning_callback"](
                        "x" * MAX_RUN_REASONING_SEGMENT_CHARS
                    )
                    kwargs["reasoning_callback"]("overflow")
                return {"final_response": "done"}

            mock_agent.run_conversation.side_effect = _run_conversation
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            return mock_agent

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=_create_agent):
                response = await cli.post(
                    "/v1/runs", json=_run_body(include_reasoning=True)
                )
                run_id = (await response.json())["run_id"]
                body = await (await cli.get(f"/v1/runs/{run_id}/events")).text()
        assert '"truncated": true' in body

        captured.clear()
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=_create_agent):
                response = await cli.post("/v1/runs", json=_run_body())
                run_id = (await response.json())["run_id"]
                await (await cli.get(f"/v1/runs/{run_id}/events")).text()
        assert "reasoning_callback" not in captured

    @pytest.mark.asyncio
    async def test_file_paths_are_deduplicated_workspace_relative_and_bounded(
        self, adapter, tmp_path
    ):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        inside = workspace / "src" / "app.py"
        outside = tmp_path / "secret.txt"

        def _create_agent(**kwargs):
            mock_agent = MagicMock()

            def _run_conversation(**_run_kwargs):
                kwargs["tool_progress_callback"](
                    "tool.completed",
                    "write_file",
                    duration=0.1,
                    is_error=False,
                    file_paths=[str(inside), str(outside), str(inside)],
                )
                return {"final_response": "done"}

            mock_agent.run_conversation.side_effect = _run_conversation
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            return mock_agent

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=_create_agent):
                response = await cli.post(
                    "/v1/runs",
                    json=_run_body(workspace_root=str(workspace)),
                )
                run_id = (await response.json())["run_id"]
                body = await (await cli.get(f"/v1/runs/{run_id}/events")).text()

        events = [
            json.loads(line.removeprefix("data: "))
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        completed = next(event for event in events if event["event"] == "tool.completed")
        assert completed["file_paths"] == ["src/app.py"]
        assert str(workspace) not in body
        assert str(outside) not in body

    @pytest.mark.asyncio
    async def test_failed_tool_event_exposes_only_valid_x_wm_request_id(
        self, adapter
    ):
        request_id = "55555555-5555-4555-8555-555555555555"

        def _create_agent(**kwargs):
            mock_agent = MagicMock()

            def _run_conversation(**_run_kwargs):
                kwargs["tool_progress_callback"](
                    "tool.completed",
                    "web_search",
                    duration=0.1,
                    is_error=True,
                    result=json.dumps(
                        {
                            "success": False,
                            "error": "cloud rejected",
                            "x_wm_request_id": request_id,
                            "secret": "must-not-leak",
                        }
                    ),
                )
                return {"final_response": "done"}

            mock_agent.run_conversation.side_effect = _run_conversation
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            return mock_agent

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=_create_agent):
                response = await cli.post("/v1/runs", json=_run_body())
                run_id = (await response.json())["run_id"]
                body = await (await cli.get(f"/v1/runs/{run_id}/events")).text()

        events = [
            json.loads(line.removeprefix("data: "))
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        completed = next(event for event in events if event["event"] == "tool.completed")
        assert completed["x_wm_request_id"] == request_id
        assert "must-not-leak" not in body

    @pytest.mark.asyncio
    async def test_failed_managed_tool_event_carries_structured_error(
        self, adapter
    ):
        """受管 Python Compute 失败事件携带最小结构化错误,供宿主精确归因。"""
        envelope = {
            "contract_version": "0.2",
            "ok": False,
            "execution_id": "pce_" + "a" * 32,
            "state": "failed",
            "artifacts": [],
            "error": {
                "code": "PYTHON_COMPUTE_OUTPUT_REJECTED",
                "message": "validated artifacts rejected",
                "retryable": False,
                "details": {"reason": "required_output_missing"},
            },
        }

        def _create_agent(**kwargs):
            mock_agent = MagicMock()

            def _run_conversation(**_run_kwargs):
                kwargs["tool_progress_callback"](
                    "tool.completed",
                    "python_compute",
                    duration=0.1,
                    is_error=True,
                    result=json.dumps({**envelope, "secret": "must-not-leak"}),
                )
                return {"final_response": "done"}

            mock_agent.run_conversation.side_effect = _run_conversation
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            return mock_agent

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=_create_agent):
                response = await cli.post("/v1/runs", json=_run_body())
                run_id = (await response.json())["run_id"]
                body = await (await cli.get(f"/v1/runs/{run_id}/events")).text()

        events = [
            json.loads(line.removeprefix("data: "))
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        completed = next(event for event in events if event["event"] == "tool.completed")
        assert completed["x_wm_error"] == {
            "code": "PYTHON_COMPUTE_OUTPUT_REJECTED",
            "message": "validated artifacts rejected",
            "reason": "required_output_missing",
        }
        assert "must-not-leak" not in body

    @pytest.mark.asyncio
    async def test_events_stream_returns_completed(self, adapter):
        """Events stream should receive run.completed when agent finishes."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "Hello!"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json=_run_body())
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Subscribe to events
                events_resp = await cli.get(f"/v1/runs/{run_id}/events")
                assert events_resp.status == 200
                body = await events_resp.text()

                # Should contain run.completed
                assert "run.completed" in body
                assert "Hello!" in body


    @pytest.mark.asyncio
    async def test_approval_resolve_all_is_scoped_to_target_run(self, auth_adapter):
        """Same client session_id must not let one run approve another run's queue."""
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent") as mock_create:
                victim_agent, victim_ready, victim_interrupted = _make_slow_agent()
                attacker_agent, attacker_ready, attacker_interrupted = _make_slow_agent()
                mock_create.side_effect = [victim_agent, attacker_agent]

                victim_resp = await cli.post(
                    "/v1/runs",
                    json=_run_body(input="victim", session_id="shared-project"),
                    headers={"Authorization": "Bearer sk-secret"},
                )
                attacker_resp = await cli.post(
                    "/v1/runs",
                    json=_run_body(input="attacker", session_id="shared-project"),
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert victim_resp.status == 202
                assert attacker_resp.status == 202
                victim_run = (await victim_resp.json())["run_id"]
                attacker_run = (await attacker_resp.json())["run_id"]

                victim_ready.wait(timeout=3.0)
                attacker_ready.wait(timeout=3.0)
                assert auth_adapter._run_approval_sessions[victim_run] == victim_run
                assert auth_adapter._run_approval_sessions[attacker_run] == attacker_run
                assert auth_adapter._run_approval_sessions[victim_run] != auth_adapter._run_approval_sessions[attacker_run]

                victim_entry = approval_mod._ApprovalEntry({
                    "command": "bash -c victim-danger",
                    "description": "victim approval",
                    "pattern_keys": ["shell-c"],
                })
                attacker_entry = approval_mod._ApprovalEntry({
                    "command": "bash -c attacker-danger",
                    "description": "attacker approval",
                    "pattern_keys": ["shell-c"],
                })
                with approval_mod._lock:
                    approval_mod._gateway_queues[victim_run] = [victim_entry]
                    approval_mod._gateway_queues[attacker_run] = [attacker_entry]

                approval_resp = await cli.post(
                    f"/v1/runs/{attacker_run}/approval",
                    json={"choice": "session", "resolve_all": True},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                approval_data = await approval_resp.json()

                assert approval_resp.status == 200
                assert approval_data["resolved"] == 1
                assert attacker_entry.result == "session"
                assert attacker_entry.event.is_set()
                assert victim_entry.result is None
                assert not victim_entry.event.is_set()
                with approval_mod._lock:
                    assert approval_mod._gateway_queues[victim_run] == [victim_entry]
                    assert victim_run in approval_mod._gateway_queues
                    assert attacker_run not in approval_mod._gateway_queues

                # Clean up the synthetic pending victim approval and unblock the
                # slow test agents so their background run tasks can finish.
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(victim_run, None)
                victim_interrupted.set()
                attacker_interrupted.set()


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/steer — steer a running agent
# ---------------------------------------------------------------------------


class TestSteerRun:
    @pytest.mark.asyncio
    async def test_steer_running_agent(self, adapter):
        app = _create_runs_app(adapter)
        agent = MagicMock()
        agent.steer.return_value = True
        queue = asyncio.Queue()
        adapter._active_run_agents["run_123"] = agent
        adapter._run_streams["run_123"] = queue
        adapter._set_run_status("run_123", "running")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_123/steer", json={"input": "tighten the ending"})
            payload = await resp.json()

        assert resp.status == 200
        assert payload == {
            "object": "hermes.run.steer",
            "run_id": "run_123",
            "accepted": True,
        }
        agent.steer.assert_called_once_with("tighten the ending")
        assert adapter._run_statuses["run_123"]["last_event"] == "run.steered"
        event = queue.get_nowait()
        assert event["event"] == "run.steered"
        assert event["run_id"] == "run_123"
        assert event["accepted"] is True

    @pytest.mark.asyncio
    async def test_steer_nonexistent_run_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_missing/steer", json={"input": "hello"})
            payload = await resp.json()

        assert resp.status == 404
        assert payload["error"]["code"] == "run_not_found"

    @pytest.mark.asyncio
    async def test_steer_inactive_run_returns_409(self, adapter):
        app = _create_runs_app(adapter)
        adapter._set_run_status("run_done", "completed")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_done/steer", json={"input": "hello"})
            payload = await resp.json()

        assert resp.status == 409
        assert payload["error"]["code"] == "run_not_accepting_steer"

    @pytest.mark.asyncio
    async def test_steer_missing_input_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        agent = MagicMock()
        agent.steer.return_value = True
        adapter._active_run_agents["run_123"] = agent
        adapter._set_run_status("run_123", "running")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_123/steer", json={"input": ""})
            payload = await resp.json()

        assert resp.status == 400
        assert payload["error"]["code"] == "invalid_steer_input"
        agent.steer.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_then_steer_rejects_retained_agent_ref(self, adapter):
        """Steer must reject a stopping run even if the executor thread is still live."""
        app = _create_runs_app(adapter)
        run_can_finish = threading.Event()
        run_started = threading.Event()

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_agent.steer = MagicMock(return_value=True)

                def _interrupt(_message=None):
                    return None

                def _run_conversation(*_args, **_kwargs):
                    run_started.set()
                    run_can_finish.wait(timeout=5)
                    return {"final_response": "late result"}

                mock_agent.interrupt = MagicMock(side_effect=_interrupt)
                mock_agent.run_conversation.side_effect = _run_conversation
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json=_run_body())
                run_id = (await start_resp.json())["run_id"]
                assert run_started.wait(timeout=3.0)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                assert run_id in adapter._active_run_agents

                steer_resp = await cli.post(
                    f"/v1/runs/{run_id}/steer",
                    json={"input": "tighten the ending"},
                )
                steer_data = await steer_resp.json()

                assert steer_resp.status == 409
                assert steer_data["error"]["code"] == "run_not_accepting_steer"
                mock_agent.steer.assert_not_called()

                run_can_finish.set()
                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_pending_steer_preserved_on_run_completed(self, adapter):
        """A steer drained by the turn finalizer (accepted after the final
        response) must surface as pending_steer on the terminal run status
        instead of being silently dropped."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_agent.run_conversation.return_value = {
                    "final_response": "done",
                    "pending_steer": "tighten the ending",
                }
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json=_run_body())
                run_id = (await start_resp.json())["run_id"]

                for _ in range(40):
                    status = adapter._run_statuses.get(run_id, {})
                    if status.get("status") == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert adapter._run_statuses[run_id]["status"] == "completed"
        assert adapter._run_statuses[run_id]["pending_steer"] == "tighten the ending"

    @pytest.mark.asyncio
    async def test_steer_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_any/steer", json={"input": "hello"})

        assert resp.status == 401


# ---------------------------------------------------------------------------
# Run lifecycle TTL sweeping
# ---------------------------------------------------------------------------


class TestRunLifecycleSweep:

    @pytest.mark.asyncio
    async def test_expired_live_run_drops_transport_but_keeps_control_state(self, adapter):
        """Stream TTL bounds buffering without detaching a live run."""
        app = _create_runs_app(adapter)
        adapter._max_concurrent_runs = 1

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json=_run_body())
                assert start_resp.status == 202
                run_id = (await start_resp.json())["run_id"]
                assert agent_ready.wait(timeout=3.0)

                task = adapter._active_run_tasks[run_id]
                assert isinstance(task, asyncio.Task)
                assert not task.done()

                pending = approval_mod._ApprovalEntry({
                    "command": "bash -c long-running",
                    "description": "approval after stream TTL",
                    "pattern_keys": ["shell-c"],
                })
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [pending]

                adapter._run_streams_created[run_id] -= adapter._RUN_STREAM_TTL + 1
                # Exercise one real sweeper iteration without waiting 60 seconds.
                with patch(
                    "gateway.platforms.api_server.asyncio.sleep",
                    side_effect=[None, asyncio.CancelledError()],
                ):
                    with pytest.raises(asyncio.CancelledError):
                        await adapter._sweep_orphaned_runs()

                assert adapter._active_run_tasks[run_id] is task
                assert adapter._active_run_agents[run_id] is mock_agent
                assert run_id not in adapter._run_streams
                assert run_id not in adapter._run_streams_created
                assert adapter._run_approval_sessions[run_id] == run_id

                limited = adapter._concurrency_limited_response()
                assert limited is not None
                assert limited.status == 429

                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "session"},
                )
                assert approval_resp.status == 200
                assert pending.event.is_set()
                assert pending.result == "session"

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                mock_agent.interrupt.assert_called_once_with("Stop requested via API")


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/stop — interrupt a running agent
# ---------------------------------------------------------------------------


class TestStopRun:

    @pytest.mark.asyncio
    async def test_stop_keeps_uncooperative_executor_tracked_until_exit(self, adapter):
        """Cancelling an asyncio wrapper must not hide its live executor thread."""
        app = _create_runs_app(adapter)
        run_can_finish = threading.Event()
        run_finished = threading.Event()

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                started = threading.Event()

                def _run_conversation(*_args, **_kwargs):
                    started.set()
                    run_can_finish.wait(timeout=5)
                    run_finished.set()
                    return {"final_response": "late result"}

                mock_agent.run_conversation.side_effect = _run_conversation
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json=_run_body())
                run_id = (await resp.json())["run_id"]
                assert started.wait(timeout=3)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                await asyncio.sleep(0.1)

                assert not run_finished.is_set()
                assert run_id in adapter._active_run_agents
                assert run_id in adapter._active_run_tasks
                assert adapter._run_statuses[run_id]["status"] == "stopping"

                run_can_finish.set()
                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks
                assert adapter._run_statuses[run_id]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_stop_running_agent(self, adapter):
        """Stop should interrupt the agent and cancel the task."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json=_run_body())
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Wait for agent to start running in the thread
                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                # Verify agent ref is stored
                assert run_id in adapter._active_run_agents

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                stop_data = await stop_resp.json()
                assert stop_data["run_id"] == run_id
                assert stop_data["status"] == "stopping"

                # Agent interrupt should have been called
                mock_agent.interrupt.assert_called_once_with("Stop requested via API")

                status_resp = await cli.get(f"/v1/runs/{run_id}")
                assert status_resp.status == 200
                status_data = await status_resp.json()
                assert status_data["status"] in {"stopping", "cancelled"}

                # Refs should be cleaned up
                await asyncio.sleep(0.2)
                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks


    @pytest.mark.asyncio
    async def test_stop_sends_sentinel_to_events_stream(self, adapter):
        """After stop, the events stream should close."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json=_run_body())
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                # Subscribe to events in background
                events_task = asyncio.ensure_future(
                    cli.get(f"/v1/runs/{run_id}/events")
                )

                await asyncio.sleep(0.1)

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200

                # Events stream should close
                events_resp = await asyncio.wait_for(events_task, timeout=5.0)
                assert events_resp.status == 200
                body = await events_resp.text()
                # Stream should have received run.failed and closed
                assert "run.failed" in body or "stream closed" in body


class TestRunsProviderAuthFailure:
    @pytest.mark.asyncio
    async def test_status_reports_provider_auth_failure_distinctly(self, adapter):
        """/v1/runs builds its own agent via _create_agent() and does not
        route through _run_agent(), so the controlled "Provider
        authentication failed" message added there does not cover this
        endpoint. _handle_runs()'s own _ProviderAuthResolutionError branch
        must give the same distinguished message instead of the generic
        except-Exception "run failed" text."""
        from gateway.platforms.api_server import _ProviderAuthResolutionError

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.side_effect = _ProviderAuthResolutionError(
                    "No credentials found for provider 'nous'"
                )

                resp = await cli.post("/v1/runs", json=_run_body())
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(40):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "failed":
                        break
                    await asyncio.sleep(0.05)

                assert status["status"] == "failed"
                assert status["error"] == "⚠️ Provider authentication failed: No credentials found for provider 'nous'"
                assert status["last_event"] == "run.failed"
