"""Session-scoped external read authorization."""

from pathlib import Path

from agent import read_scope


def setup_function() -> None:
    read_scope.clear_all()


def teardown_function() -> None:
    read_scope.clear_all()


def test_workspace_and_granted_file_are_allowed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external.txt"
    external.write_text("data")

    assert read_scope.is_allowed(
        workspace / "new.txt",
        session_key="s1",
        workspace_root=workspace,
    )
    assert not read_scope.is_allowed(
        external,
        session_key="s1",
        workspace_root=workspace,
    )

    read_scope.grant(external, session_key="s1", scope_kind="file")
    assert read_scope.is_allowed(external, session_key="s1", workspace_root=workspace)
    assert not read_scope.is_allowed(external, session_key="s2", workspace_root=workspace)
    assert not read_scope.is_allowed(
        external, session_key="s2", task_id="s1", workspace_root=workspace
    )


def test_directory_grant_is_recursive_but_not_sibling(tmp_path: Path) -> None:
    root = tmp_path / "materials"
    nested = root / "nested"
    sibling = tmp_path / "materials-2"
    nested.mkdir(parents=True)
    sibling.mkdir()
    read_scope.grant(root, session_key="s1", scope_kind="directory")

    assert read_scope.is_allowed(nested / "note.txt", session_key="s1")
    assert not read_scope.is_allowed(sibling / "note.txt", session_key="s1")


def test_revoke_and_clear_remove_session_grants(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("data")
    read_scope.grant(target, session_key="s1", scope_kind="file")
    assert read_scope.is_allowed(target, session_key="s1")

    read_scope.revoke(target, session_key="s1", scope_kind="file")
    assert not read_scope.is_allowed(target, session_key="s1")

    read_scope.grant(target, session_key="s1", scope_kind="file")
    read_scope.clear("s1")
    assert not read_scope.is_allowed(target, session_key="s1")
