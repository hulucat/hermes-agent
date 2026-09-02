"""Session-scoped read authorization for WorkMate external files.

The registry is intentionally in-memory: a process restart or session end
must revoke grants. Callers still need to canonicalize paths before checking
or granting a scope.
"""

from __future__ import annotations

import threading
from pathlib import Path


_lock = threading.RLock()
_grants: dict[str, set[tuple[str, str]]] = {}


def _key(session_key: str, task_id: str = "") -> str:
    return (session_key or task_id or "default").strip() or "default"


def _canonical(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def is_allowed(
    path: str | Path,
    *,
    session_key: str,
    task_id: str = "",
    workspace_root: str | Path | None = None,
) -> bool:
    """Return whether *path* is inside the workspace or a granted scope."""
    target = _canonical(path)
    roots = []
    if workspace_root:
        roots.append(_canonical(workspace_root))
    # A supplied session key is authoritative. Do not fall back to task_id:
    # task IDs can be reused by a later conversation and must not inherit a
    # previous session's external read grant.
    key_candidates = {_key(session_key, task_id)}
    with _lock:
        grants = [scope for key in key_candidates for scope in _grants.get(key, set())]
    for root in roots:
        try:
            Path(target).relative_to(root)
            return True
        except ValueError:
            pass
    for scope_kind, scope_path in grants:
        try:
            if scope_kind == "file":
                if target == scope_path:
                    return True
            elif scope_kind == "directory":
                Path(target).relative_to(scope_path)
                return True
        except ValueError:
            continue
    return False


def grant(path: str | Path, *, session_key: str, scope_kind: str) -> str:
    """Grant an exact file or directory subtree for the current session."""
    if scope_kind not in {"file", "directory"}:
        raise ValueError(f"unsupported read scope: {scope_kind}")
    scope_path = _canonical(path)
    key = _key(session_key)
    with _lock:
        _grants.setdefault(key, set()).add((scope_kind, scope_path))
    return scope_path


def revoke(path: str | Path, *, session_key: str, scope_kind: str) -> None:
    scope = (scope_kind, _canonical(path))
    key = _key(session_key)
    with _lock:
        scopes = _grants.get(key)
        if not scopes:
            return
        scopes.discard(scope)
        if not scopes:
            _grants.pop(key, None)


def clear(session_key: str) -> None:
    with _lock:
        _grants.pop(_key(session_key), None)


def clear_all() -> None:
    with _lock:
        _grants.clear()
