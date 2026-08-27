"""Per-Run workspace write boundary for WorkMate runs."""

from __future__ import annotations

import json

from tools.file_tools import patch_tool, write_file_tool
from tools.terminal_tool import clear_task_env_overrides, register_task_env_overrides


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
