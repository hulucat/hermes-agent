"""Profile-local durable audit ledger for cron execution attempts.

The ledger records what is known about each attempt; it is not a retry queue.
Interrupted attempts become ``unknown`` only after their exact owner process is
proved gone. Terminal states are immutable.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

EXECUTIONS_FILE = get_hermes_home().resolve() / "cron" / "executions.db"
MAX_TERMINAL_EXECUTIONS = 1000
_TERMINAL_STATES = ("completed", "failed", "unknown", "skipped")
_lock = threading.RLock()
_PROCESS_ID = uuid.uuid4().hex


def _create_execution_table(conn: sqlite3.Connection) -> None:
    """Create the current execution ledger schema in an empty database."""
    conn.execute(
        """CREATE TABLE executions (
             id TEXT PRIMARY KEY,
             job_id TEXT NOT NULL,
             source TEXT NOT NULL,
             process_id TEXT NOT NULL,
             pid INTEGER NOT NULL,
             process_started_at INTEGER,
             status TEXT NOT NULL CHECK(status IN
               ('claimed','running','completed','failed','unknown','skipped')),
             claimed_at TEXT NOT NULL,
             started_at TEXT,
             finished_at TEXT,
             error TEXT,
             output_file TEXT
           )"""
    )


def _migrate_execution_table_if_needed(conn: sqlite3.Connection) -> None:
    """Upgrade old ledgers so they can persist HLMate's ``skipped`` state."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='executions'"
    ).fetchone()
    schema = str(row["sql"] or "") if row is not None else ""
    if "'skipped'" in schema and "output_file" in schema:
        return

    # SQLite CHECK constraints cannot be altered. Rebuild atomically while
    # retaining every audit row; no scheduled work is inferred from this ledger.
    legacy_columns = {item["name"] for item in conn.execute("PRAGMA table_info(executions)")}
    conn.execute("DROP INDEX IF EXISTS idx_executions_job_claimed")
    conn.execute("DROP INDEX IF EXISTS idx_executions_status_claimed")
    conn.execute("ALTER TABLE executions RENAME TO executions_legacy")
    _create_execution_table(conn)
    columns = [
        "id", "job_id", "source", "process_id", "pid", "process_started_at",
        "status", "claimed_at", "started_at", "finished_at", "error",
    ]
    target = ", ".join(columns + ["output_file"])
    source = ", ".join(columns + (["output_file"] if "output_file" in legacy_columns else ["NULL"]))
    conn.execute(
        f"INSERT INTO executions ({target}) SELECT {source} FROM executions_legacy"
    )
    conn.execute("DROP TABLE executions_legacy")


def _connect() -> sqlite3.Connection:
    EXECUTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(EXECUTIONS_FILE, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='executions'"
    ).fetchone()
    if table_exists:
        _migrate_execution_table_if_needed(conn)
    else:
        _create_execution_table(conn)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_job_claimed "
        "ON executions(job_id, claimed_at DESC, id DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_status_claimed "
        "ON executions(status, claimed_at DESC, id DESC)"
    )
    return conn


def _record(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    return dict(row) if row is not None else None


def _process_start_time(pid: int) -> Optional[int]:
    try:
        from gateway.status import get_process_start_time
        return get_process_start_time(pid)
    except Exception:
        return None


def _owner_is_live(pid: int, started_at: Optional[int]) -> bool:
    try:
        from gateway.status import _pid_exists
        if not _pid_exists(pid):
            return False
    except Exception:
        return True  # fail safe: inability to prove death must not rewrite state
    if started_at is None:
        return pid == os.getpid()
    current = _process_start_time(pid)
    return current is not None and current == started_at


def _prune_unlocked(conn: sqlite3.Connection) -> None:
    limit = max(0, int(MAX_TERMINAL_EXECUTIONS))
    conn.execute(
        """DELETE FROM executions WHERE id IN (
             SELECT id FROM executions
             WHERE status IN ('completed','failed','unknown','skipped')
             ORDER BY claimed_at DESC, id DESC LIMIT -1 OFFSET ?
           )""",
        (limit,),
    )


def create_execution(job_id: str, *, source: str) -> Dict[str, Any]:
    """Persist a claimed attempt before executor/provider dispatch."""
    now = _hermes_now().isoformat()
    execution_id = uuid.uuid4().hex
    pid = os.getpid()
    with _lock, _connect() as conn:
        conn.execute(
            """INSERT INTO executions
               (id, job_id, source, process_id, pid, process_started_at,
                status, claimed_at)
               VALUES (?, ?, ?, ?, ?, ?, 'claimed', ?)""",
            (execution_id, str(job_id), str(source), _PROCESS_ID, pid,
             _process_start_time(pid), now),
        )
        row = conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone()
    return _record(row)  # type: ignore[return-value]


def mark_execution_running(execution_id: str) -> Optional[Dict[str, Any]]:
    """Transition one claimed attempt to running exactly once."""
    now = _hermes_now().isoformat()
    with _lock, _connect() as conn:
        cur = conn.execute(
            """UPDATE executions SET status='running', started_at=?
               WHERE id=? AND status='claimed'""",
            (now, execution_id),
        )
        if cur.rowcount != 1:
            return None
        return _record(conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone())


def finish_execution(
    execution_id: str, *, success: bool, error: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Write a terminal result once; terminal attempts cannot be rewritten."""
    now = _hermes_now().isoformat()
    status = "completed" if success else "failed"
    detail = None if success else (str(error) if error else "unknown failure")
    with _lock, _connect() as conn:
        cur = conn.execute(
            """UPDATE executions SET status=?, finished_at=?, error=?
               WHERE id=? AND status IN ('claimed','running')""",
            (status, now, detail, execution_id),
        )
        if cur.rowcount != 1:
            return None
        _prune_unlocked(conn)
        return _record(conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone())


def set_execution_output(execution_id: str, output_file: str) -> Optional[Dict[str, Any]]:
    """Attach the profile-relative output path produced by one execution.

    Output is persisted before delivery, so this association remains useful
    even when the destination channel later fails.  The path is deliberately
    relative to ``cron/output``; callers must resolve it under that directory.
    """
    with _lock, _connect() as conn:
        cur = conn.execute(
            """UPDATE executions SET output_file=?
               WHERE id=? AND status IN ('claimed','running','completed','failed')""",
            (str(output_file), execution_id),
        )
        if cur.rowcount != 1:
            return None
        return _record(conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone())


def record_skipped_execution(job_id: str, *, reason: str, source: str = "hlmate") -> Dict[str, Any]:
    """Append an immutable terminal record for work intentionally not run."""
    now = _hermes_now().isoformat()
    execution_id = uuid.uuid4().hex
    pid = os.getpid()
    with _lock, _connect() as conn:
        conn.execute(
            """INSERT INTO executions
               (id, job_id, source, process_id, pid, process_started_at,
                status, claimed_at, finished_at, error)
               VALUES (?, ?, ?, ?, ?, ?, 'skipped', ?, ?, ?)""",
            (execution_id, str(job_id), str(source), _PROCESS_ID, pid,
             _process_start_time(pid), now, now, str(reason)),
        )
        _prune_unlocked(conn)
        row = conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone()
    return _record(row)  # type: ignore[return-value]


def recover_interrupted_executions() -> int:
    """Mark provably abandoned attempts unknown without scheduling retries."""
    now = _hermes_now().isoformat()
    changed = 0
    with _lock, _connect() as conn:
        rows = conn.execute(
            """SELECT id, process_id, pid, process_started_at FROM executions
               WHERE status IN ('claimed','running')"""
        ).fetchall()
        for row in rows:
            if row["process_id"] == _PROCESS_ID:
                continue
            if _owner_is_live(int(row["pid"]), row["process_started_at"]):
                continue
            cur = conn.execute(
                """UPDATE executions SET status='unknown', finished_at=?, error=?
                   WHERE id=? AND status IN ('claimed','running')""",
                (now,
                 "Scheduler restarted after this execution's owner exited before a durable "
                 "terminal state; whether side effects ran is unknown.",
                 row["id"]),
            )
            changed += cur.rowcount
        if changed:
            _prune_unlocked(conn)
    return changed


def list_executions(
    *, job_id: Optional[str] = None, limit: int = 50,
    before_claimed_at: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return indexed, newest-first execution history with cursor pagination."""
    clauses: List[str] = []
    params: List[Any] = []
    if job_id is not None:
        clauses.append("job_id=?")
        params.append(str(job_id))
    if before_claimed_at is not None:
        clauses.append("claimed_at < ?")
        params.append(str(before_claimed_at))
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(max(1, min(int(limit), 500)))
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM executions" + where
            + " ORDER BY claimed_at DESC, id DESC LIMIT ?",
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def latest_execution(job_id: str) -> Optional[Dict[str, Any]]:
    rows = list_executions(job_id=job_id, limit=1)
    return rows[0] if rows else None


def latest_executions(job_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load latest execution for many jobs in one indexed query."""
    clean = [str(job_id) for job_id in dict.fromkeys(job_ids) if job_id]
    if not clean:
        return {}
    placeholders = ",".join("?" for _ in clean)
    with _lock, _connect() as conn:
        rows = conn.execute(
            f"""SELECT e.* FROM executions e
                WHERE e.job_id IN ({placeholders})
                  AND e.id=(SELECT e2.id FROM executions e2
                            WHERE e2.job_id=e.job_id
                            ORDER BY e2.claimed_at DESC, e2.id DESC LIMIT 1)""",
            clean,
        ).fetchall()
    return {row["job_id"]: dict(row) for row in rows}
