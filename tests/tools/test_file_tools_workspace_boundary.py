"""Per-Run workspace write boundary for WorkMate runs."""

from __future__ import annotations

import json

from tools.file_tools import patch_tool, read_file_tool, search_tool, write_file_tool
from tools.terminal_tool import clear_task_env_overrides, register_task_env_overrides
from agent.read_scope import clear as clear_read_scope, grant as grant_read_scope


def test_write_and_patch_block_absolute_and_parent_workspace_escapes(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inside = workspace / "inside.txt"
    outside = tmp_path / "outside.txt"
    task_id = "run-boundary"
    register_task_env_overrides(task_id, {"cwd": str(workspace)})
    try:
        inside_result = json.loads(
            write_file_tool("inside.txt", "inside", task_id=task_id)
        )
        absolute_result = json.loads(
            write_file_tool(str(outside), "outside", task_id=task_id)
        )
        parent_result = json.loads(
            patch_tool(
                mode="replace",
                path="../outside.txt",
                old_string="outside",
                new_string="changed",
                task_id=task_id,
            )
        )
    finally:
        clear_task_env_overrides(task_id)

    assert inside_result["files_modified"] == [str(inside.resolve())]
    assert "escapes allowed directory" in absolute_result["error"]
    assert "escapes allowed directory" in parent_result["error"]
    assert not outside.exists()


def test_write_blocks_symlink_escape(tmp_path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "link").symlink_to(outside, target_is_directory=True)
    task_id = "run-symlink"
    register_task_env_overrides(task_id, {"cwd": str(workspace)})
    try:
        result = json.loads(
            write_file_tool("link/escaped.txt", "secret", task_id=task_id)
        )
    finally:
        clear_task_env_overrides(task_id)

    assert "escapes allowed directory" in result["error"]
    assert not (outside / "escaped.txt").exists()


def test_read_and_search_block_external_paths_until_granted(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "note.txt"
    target.write_text("external material\n")
    task_id = "read-boundary"
    session_id = "read-session"
    register_task_env_overrides(task_id, {"cwd": str(workspace), "wm_session_id": session_id})
    try:
        blocked = json.loads(read_file_tool(str(target), task_id=task_id))
        assert "outside the active workspace" in blocked["error"]
        blocked_search = json.loads(search_tool("external", path=str(outside), task_id=task_id))
        assert "outside the active workspace" in blocked_search["error"]

        grant_read_scope(outside, session_key=session_id, scope_kind="directory")
        allowed = json.loads(read_file_tool(str(target), task_id=task_id))
        assert "external material" in allowed["content"]
        allowed_search = json.loads(search_tool("external", path=str(outside), task_id=task_id))
        assert allowed_search["matches"]
    finally:
        clear_read_scope(session_id)
        clear_task_env_overrides(task_id)


def test_read_scope_does_not_follow_granted_directory_symlink(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "secret.txt"
    target.write_text("secret\n")
    link = workspace / "link.txt"
    link.symlink_to(target)
    task_id = "read-symlink"
    register_task_env_overrides(task_id, {"cwd": str(workspace), "wm_session_id": task_id})
    try:
        result = json.loads(read_file_tool(str(link), task_id=task_id))
        assert "outside the active workspace" in result["error"]
    finally:
        clear_read_scope(task_id)
        clear_task_env_overrides(task_id)


def test_sensitive_external_file_remains_hard_blocked_after_grant(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / ".env"
    secret.write_text("TOKEN=do-not-read\n")
    task_id = "read-sensitive"
    register_task_env_overrides(task_id, {"cwd": str(workspace), "wm_session_id": task_id})
    grant_read_scope(outside, session_key=task_id, scope_kind="directory")
    try:
        result = json.loads(read_file_tool(str(secret), task_id=task_id))
    finally:
        clear_read_scope(task_id)
        clear_task_env_overrides(task_id)

    assert "environment file" in result["error"].lower()
