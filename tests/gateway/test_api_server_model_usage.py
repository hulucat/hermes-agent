"""Run-scoped resolved-model SSE, persistence and query contracts."""

import asyncio
import threading

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


class _UsageDB:
    def __init__(self):
        self.rows = []

    def record_run_model_usage(
        self, run_id, session_id, seq, resolved_model, *, reason, timestamp
    ):
        self.rows.append({
            "run_id": run_id,
            "session_id": session_id,
            "seq": seq,
            "resolved_model": resolved_model,
            "reason": reason,
            "timestamp": timestamp,
        })

    def get_run_model_usage(self, run_id):
        return sorted(
            (row for row in self.rows if row["run_id"] == run_id),
            key=lambda row: row["seq"],
        )


def _bind_run(adapter, run_id, session_id, db, loop):
    adapter._run_streams[run_id] = asyncio.Queue()
    adapter._run_model_contexts[run_id] = {
        "loop": loop,
        "session_id": session_id,
        "db": db,
        "seq": 0,
        "persist_tasks": set(),
    }


@pytest.mark.asyncio
async def test_model_used_isolated_for_concurrent_runs_sharing_session():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    loop = asyncio.get_running_loop()
    db = _UsageDB()
    _bind_run(adapter, "run_a", "shared-session", db, loop)
    _bind_run(adapter, "run_b", "shared-session", db, loop)

    threads = [
        threading.Thread(
            target=adapter._dispatch_model_usage_hook,
            kwargs={
                "run_id": run_id,
                "session_id": "shared-session",
                "response_model": model,
            },
        )
        for run_id, model in (("run_a", "model-a"), ("run_b", "model-b"))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    await asyncio.sleep(0)
    await adapter._drain_run_model_usage("run_a")
    await adapter._drain_run_model_usage("run_b")

    event_a = await adapter._run_streams["run_a"].get()
    event_b = await adapter._run_streams["run_b"].get()
    assert (event_a["run_id"], event_a["seq"], event_a["model"]) == (
        "run_a", 1, "model-a"
    )
    assert (event_b["run_id"], event_b["seq"], event_b["model"]) == (
        "run_b", 1, "model-b"
    )
    assert [(row["run_id"], row["seq"]) for row in db.rows] == [
        ("run_a", 1),
        ("run_b", 1),
    ]


@pytest.mark.asyncio
async def test_model_used_ignores_calls_without_active_run_context():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._dispatch_model_usage_hook(
        run_id=None,
        session_id="session-x",
        response_model="model-x",
    )
    adapter._dispatch_model_usage_hook(
        run_id="unknown-run",
        session_id="session-x",
        response_model="model-x",
    )
    await asyncio.sleep(0)
    assert adapter._run_model_contexts == {}


@pytest.mark.asyncio
async def test_model_usage_query_requires_auth_and_returns_persisted_rows():
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "sk-secret"})
    )
    db = _UsageDB()
    db.rows.extend([
        {
            "run_id": "run_x",
            "session_id": "session-x",
            "seq": 2,
            "resolved_model": "model-b",
            "reason": "api",
            "timestamp": "2.0",
        },
        {
            "run_id": "run_x",
            "session_id": "session-x",
            "seq": 1,
            "resolved_model": "model-a",
            "reason": "api",
            "timestamp": "1.0",
        },
    ])
    adapter._session_db = db
    app = web.Application()
    app.router.add_get(
        "/v1/runs/{run_id}/model-usage", adapter._handle_run_model_usage
    )

    async with TestClient(TestServer(app)) as client:
        unauthorized = await client.get("/v1/runs/run_x/model-usage")
        assert unauthorized.status == 401
        response = await client.get(
            "/v1/runs/run_x/model-usage",
            headers={"Authorization": "Bearer sk-secret"},
        )
        assert response.status == 200
        rows = await response.json()

    assert [row["seq"] for row in rows] == [1, 2]
    assert [row["resolved_model"] for row in rows] == ["model-a", "model-b"]
