"""PATCH-004: ``run_model_usage`` table — record/get + FK CASCADE (state.db side).

The api_server dispatch layer is covered by
``tests/gateway/test_api_server_model_usage.py``; this file pins the
persistence contract: per-(run_id, seq) rows, ordering by seq, and that a
backend deletion saga (``DELETE SESSION``) CASCADE-clears every belonging
run automatically — the guarantee AD-10 reliability degrades to, since the
SSE stream itself is non-recoverable.
"""
from hermes_state import SessionDB


def test_record_get_and_cascade_run_model_usage(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    # FK target: run_model_usage.session_id REFERENCES sessions(id).
    db.create_session("s1", source="test")

    db.record_run_model_usage("run_a", "s1", 1, "claude-A")
    db.record_run_model_usage("run_a", "s1", 2, "claude-A2")
    db.record_run_model_usage("run_b", "s1", 1, "claude-B")

    rows_a = db.get_run_model_usage("run_a")
    assert [r["seq"] for r in rows_a] == [1, 2]  # ordered by seq
    assert all(r["run_id"] == "run_a" and r["session_id"] == "s1" for r in rows_a)
    assert rows_a[0]["resolved_model"] == "claude-A"
    assert rows_a[0]["reason"] == "api"  # default

    rows_b = db.get_run_model_usage("run_b")
    assert len(rows_b) == 1
    assert rows_b[0]["resolved_model"] == "claude-B"
    assert rows_b[0]["seq"] == 1  # independent of run_a's seq

    # Unknown run → empty, not error.
    assert db.get_run_model_usage("nope") == []

    # DELETE SESSION must CASCADE-clear every belonging run (backend saga).
    assert db.delete_session("s1") is True
    assert db.get_run_model_usage("run_a") == []
    assert db.get_run_model_usage("run_b") == []
