"""Declarative schema and lifecycle tests for run_model_usage."""

import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_common import SCHEMA_VERSION


@pytest.mark.parametrize("legacy_version", [22, 23, 26])
def test_declarative_table_lands_without_new_schema_version(
    tmp_path, legacy_version
):
    db_path = tmp_path / f"state-v{legacy_version}.db"
    db = SessionDB(db_path=db_path)
    db.close()

    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE run_model_usage")
    conn.execute("UPDATE schema_version SET version = ?", (legacy_version,))
    conn.commit()
    conn.close()

    reopened = SessionDB(db_path=db_path)
    try:
        columns = {
            row["name"]
            for row in reopened._conn.execute(
                "PRAGMA table_info('run_model_usage')"
            ).fetchall()
        }
        assert columns == {
            "run_id",
            "session_id",
            "seq",
            "resolved_model",
            "reason",
            "timestamp",
        }
        version = reopened._conn.execute(
            "SELECT version FROM schema_version"
        ).fetchone()[0]
        assert version == SCHEMA_VERSION == 26
    finally:
        reopened.close()


def test_existing_patched_table_is_reused_without_data_loss(tmp_path):
    db_path = tmp_path / "patched-v23.db"
    db = SessionDB(db_path=db_path)
    db.create_session("session-a", source="test")
    db.record_run_model_usage(
        "run-a", "session-a", 1, "model-a", timestamp="123.0"
    )
    db.close()

    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE schema_version SET version = 23")
    conn.commit()
    conn.close()

    reopened = SessionDB(db_path=db_path)
    try:
        assert reopened.get_run_model_usage("run-a") == [{
            "run_id": "run-a",
            "session_id": "session-a",
            "seq": 1,
            "resolved_model": "model-a",
            "reason": "api",
            "timestamp": "123.0",
        }]
    finally:
        reopened.close()


def test_run_usage_orders_per_run_and_cascades_with_session(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("session-a", source="test")
        db.record_run_model_usage("run-a", "session-a", 2, "model-a2")
        db.record_run_model_usage("run-a", "session-a", 1, "model-a1")
        db.record_run_model_usage("run-b", "session-a", 1, "model-b1")

        assert [row["seq"] for row in db.get_run_model_usage("run-a")] == [1, 2]
        assert [row["seq"] for row in db.get_run_model_usage("run-b")] == [1]
        assert db.delete_session("session-a") is True
        assert db.get_run_model_usage("run-a") == []
        assert db.get_run_model_usage("run-b") == []
    finally:
        db.close()
