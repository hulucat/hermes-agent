"""PATCH-004: per-run ``model.used`` SSE attribution (api_server side).

These tests pin the core correctness guarantee of PATCH-004: two ``/v1/runs``
sharing one ``session_id`` each receive their OWN ``model.used`` SSE events
with independent ``seq`` counters and no cross-talk, even though the
``post_api_request`` hook fires from the shared agent executor pool.

Coverage here is deliberately split from the state.db layer
(``tests/test_hermes_state_run_model_usage.py``) and from end-to-end:

* Hook dispatch / per-run seq / no-cross-talk — here (unit, no live agent).
* record/get + FK CASCADE — ``test_hermes_state_run_model_usage.py``.
* End-to-end concurrent ``POST /v1/runs`` with a real agent — deferred to the
  TaskFacade stage, where a real agent-creation path exists; the fragile
  ``_handle_runs`` internal mocking it would require today adds no signal
  beyond what these two layers already prove.
"""
import asyncio
import threading
from unittest.mock import patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli.plugins import get_plugin_manager, invoke_hook


def _register_isolated(adapter: APIServerAdapter, loop) -> list:
    """Register the PATCH-004 sink against an isolated global hook list.

    Returns the previously-registered callbacks so the caller can restore
    them in a finally block — the plugin manager is a process-wide singleton.
    """
    mgr = get_plugin_manager()
    saved = mgr._hooks.get("post_api_request", [])
    mgr._hooks["post_api_request"] = []
    adapter._register_model_usage_hook(loop)
    return saved


@pytest.mark.asyncio
async def test_model_used_dispatches_per_run_id_without_crosstalk():
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    loop = asyncio.get_running_loop()
    saved = _register_isolated(adapter, loop)
    try:
        q_a: asyncio.Queue = asyncio.Queue()
        q_b: asyncio.Queue = asyncio.Queue()
        adapter._run_streams["run_a"] = q_a
        adapter._run_streams["run_b"] = q_b

        def fire(run_id: str, model: str) -> None:
            # The hook fires on the agent executor thread; mirror that here.
            invoke_hook(
                "post_api_request",
                run_id=run_id,
                session_id="shared-session",
                response_model=model,
            )

        # Persistence is covered by the state.db test; keep it a no-op here so
        # the assertion targets dispatch only and never touches hermes home.
        with patch.object(adapter, "_persist_model_usage", lambda *a, **k: None):
            t_a = threading.Thread(target=fire, args=("run_a", "claude-A"))
            t_b = threading.Thread(target=fire, args=("run_b", "claude-B"))
            t_a.start()
            t_b.start()
            t_a.join()
            t_b.join()
            # call_soon_threadsafe scheduled _on_model_used; let the loop run it.
            await asyncio.sleep(0.08)

            async def collect(q: asyncio.Queue) -> list:
                out = []
                for _ in range(8):
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=0.3)
                    except asyncio.TimeoutError:
                        break
                    if ev and ev.get("event") == "model.used":
                        out.append(ev)
                return out

            a_events = await collect(q_a)
            b_events = await collect(q_b)

        assert len(a_events) == 1
        assert a_events[0]["run_id"] == "run_a"
        assert a_events[0]["model"] == "claude-A"
        assert a_events[0]["seq"] == 1  # independent per-run counter

        assert len(b_events) == 1
        assert b_events[0]["run_id"] == "run_b"
        assert b_events[0]["model"] == "claude-B"
        assert b_events[0]["seq"] == 1  # NOT 2 — no carry-over from run_a
    finally:
        get_plugin_manager()._hooks["post_api_request"] = saved


@pytest.mark.asyncio
async def test_model_used_skipped_when_no_run_id():
    """Agents outside /v1/runs (chat/completions, CLI, gateway) never get
    ``run_id`` tagged on their instance (PATCH-004 1A only tags /v1/runs
    agents), so the sink MUST skip them — otherwise every chat turn would
    emit a spurious ``model.used`` and try to persist with run_id=None."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    loop = asyncio.get_running_loop()
    saved = _register_isolated(adapter, loop)
    try:
        q: asyncio.Queue = asyncio.Queue()
        adapter._run_streams["run_x"] = q  # would receive a leak if any

        with patch.object(adapter, "_persist_model_usage", lambda *a, **k: None):
            invoke_hook(
                "post_api_request",
                run_id=None,  # no /v1/runs correlation
                session_id="s",
                response_model="claude-X",
            )
            await asyncio.sleep(0.08)

        assert q.empty(), "model.used leaked for a run_id-less agent"
    finally:
        get_plugin_manager()._hooks["post_api_request"] = saved
