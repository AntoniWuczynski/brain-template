"""Isolated tests for the vault_replace_note write tool.

Runs the tool's real logic (safety + atomic write + git commit) against a
throwaway temporary git repo, so it never touches the real vault and is
safe to run on a dirty working tree (unlike manual_test.py, which does a
``git reset --hard`` on the live repo).

Run with::

    uv run python -m mcp_server.test_replace_note

Exits 0 if every check passes; non-zero with a diagnostic otherwise.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from mcp_server.audit import AuditLog
from mcp_server.config import MAX_NOTE_BYTES, ServerConfig
from mcp_server.push_queue import PushWorker
from mcp_server.reindex import IndexRefresher
from mcp_server.runtime import Runtime
from mcp_server.safety import SafetyError
from mcp_server.tools import ToolError, tool_create_note, tool_replace_note


def _make_vault() -> Path:
    """Create a throwaway git repo to act as a vault root."""
    root = Path(tempfile.mkdtemp(prefix="brain-replace-test-")).resolve()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "test"], check=True)
    return root


def _cfg(root: Path) -> ServerConfig:
    return ServerConfig(
        vault_root=root,
        tokens=(("x" * 24, "default"),),
        bind_host="127.0.0.1",
        bind_port=0,
        git_push_on_write=False,
        git_remote="origin",
        git_branch="main",
        log_level="warning",
        allowed_hosts=(),
        profile_max_bytes=4096,
    )


def _runtime(cfg: ServerConfig) -> Runtime:
    """Real Runtime over the throwaway vault, with both background
    workers disabled: push reports "disabled", reindex reports "off"."""
    return Runtime(
        audit=AuditLog(cfg.vault_root),
        push_worker=PushWorker(
            cfg.vault_root,
            remote=cfg.git_remote,
            branch=cfg.git_branch,
            enabled=False,
        ),
        refresher=IndexRefresher(
            cfg.vault_root, audit=AuditLog(cfg.vault_root), enabled=False
        ),
    )


class Runner:
    def __init__(self) -> None:
        self.passes: list[str] = []
        self.failures: list[str] = []

    def expect(self, label: str, cond: bool, detail: str = "") -> None:
        if cond:
            self.passes.append(label)
            print(f"  PASS  {label}")
        else:
            self.failures.append(f"{label}: {detail}")
            print(f"  FAIL  {label}  ({detail})")

    def expect_raises(self, label: str, fn, expect_in_msg: str = "") -> None:
        try:
            fn()
        except (ToolError, SafetyError) as exc:
            msg = str(exc).lower()
            if expect_in_msg and expect_in_msg.lower() not in msg:
                self.expect(label, False, f"raised but message lacks {expect_in_msg!r}: {exc}")
            else:
                self.passes.append(label)
                print(f"  PASS  {label}  (refused: {str(exc)[:70]})")
        except Exception as exc:  # noqa: BLE001
            self.expect(label, False, f"wrong exception type: {type(exc).__name__}: {exc}")
        else:
            self.expect(label, False, "expected an error but none was raised")


def main() -> int:
    r = Runner()
    root = _make_vault()
    cfg = _cfg(root)
    runtime = _runtime(cfg)
    note = "knowledge/notes/_replace_test.md"

    print("\n[vault_replace_note]")

    # Seed a note to replace.
    tool_create_note(cfg, runtime, path=note, content="# original\n\nfirst body.\n")

    # 1. Replaces the full content of an existing note. The server stamps
    # provenance into the frontmatter; the client body must survive intact.
    new_body = "# rewritten\n\ncompletely new body.\n"
    res = tool_replace_note(cfg, runtime, path=note, content=new_body)
    on_disk = (root / note).read_text(encoding="utf-8")
    r.expect("replace: client body preserved", on_disk.endswith(new_body),
             f"on disk: {on_disk!r}")
    r.expect("replace: provenance stamped", "last_written_by:" in on_disk
             and "written_via: mcp" in on_disk, f"on disk: {on_disk!r}")
    r.expect("replace: reports bytes written",
             res.bytes_written == len(on_disk.encode()), f"got {res.bytes_written}")
    r.expect("replace: committed locally, push is async",
             res.committed and not res.pushed,
             f"committed={res.committed} pushed={res.pushed}")
    r.expect("replace: returns a commit sha", bool(res.commit_sha))
    r.expect("replace: push_state disabled (worker off)",
             res.push_state == "disabled", f"got {res.push_state!r}")
    r.expect("replace: index_refresh off (refresher off)",
             res.index_refresh == "off", f"got {res.index_refresh!r}")

    # 2. Refuses a note that does not exist (replace is not create).
    r.expect_raises(
        "replace: refuses non-existent note",
        lambda: tool_replace_note(cfg, runtime, path="knowledge/notes/_absent.md", content="x"),
        expect_in_msg="does not exist",
    )

    # 3. Refuses oversize content.
    r.expect_raises(
        "replace: refuses oversize content",
        lambda: tool_replace_note(cfg, runtime, path=note, content="x" * (MAX_NOTE_BYTES + 1)),
        expect_in_msg="max",
    )

    # 4. Refuses a path outside the write allowlist.
    r.expect_raises(
        "replace: refuses path outside write allowlist",
        lambda: tool_replace_note(cfg, runtime, path="archive/processed/x.md", content="nope"),
        expect_in_msg="denied",
    )

    # 5. Refuses a traversal escape.
    r.expect_raises(
        "replace: refuses ../ escape",
        lambda: tool_replace_note(cfg, runtime, path="../escape.md", content="nope"),
        expect_in_msg="escapes",
    )

    # 6. inbox/ is writable only via drop_inbox_file: the note verbs must
    # refuse it. replace on an existing inbox file would otherwise be the
    # only tool able to destroy a pending (typically uncommitted) source.
    (root / "inbox").mkdir(exist_ok=True)
    (root / "inbox" / "pending.md").write_text("original source\n", encoding="utf-8")
    r.expect_raises(
        "replace: refuses inbox/ path",
        lambda: tool_replace_note(cfg, runtime, path="inbox/pending.md", content="DESTROYED"),
        expect_in_msg="denied",
    )
    r.expect(
        "replace: inbox file untouched after refusal",
        (root / "inbox" / "pending.md").read_text(encoding="utf-8") == "original source\n",
    )
    r.expect_raises(
        "create: refuses inbox/ path",
        lambda: tool_create_note(cfg, runtime, path="inbox/new.md", content="nope"),
        expect_in_msg="denied",
    )

    print(f"\nresults: {len(r.passes)} passed, {len(r.failures)} failed")
    if r.failures:
        print("\nfailures:")
        for f in r.failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
