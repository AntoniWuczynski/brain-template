"""Write-policy and search-gate coverage for the MCP tools.

These invariants previously lived only in throwaway ``python -m`` scripts
(mcp_server/test_replace_note.py, mcp_server/test_search_gate.py) that CI
never ran — so when derived_note_relpath changed to keep the source
extension, the gate script's expectations silently went stale. Ported here
as real pytest cases over a throwaway git vault so they run every CI push.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# Type-only import of the shared harness contracts. At runtime the module
# is pytest's ``conftest``; ``tests.conftest`` is the name mypy resolves
# (explicit_package_bases), and importing it for real would collide with an
# unrelated installed ``tests`` package. Annotations are postponed, so
# nothing here is evaluated at import time.
if TYPE_CHECKING:
    from tests.conftest import CfgFactory, GitRunner, RuntimeFactory

from mcp_server.config import MAX_NOTE_BYTES, ServerConfig
from mcp_server.errors import ToolError
from mcp_server.safety import SafetyError
from mcp_server.runtime import Runtime
from mcp_server.tools import _atomic_write_text, tool_create_note, tool_replace_note
from mcp_server.tools_read import _hit_gate_path

# The vault + ServerConfig + Runtime harness is shared (tests/conftest.py):
# mcp_vault / make_cfg / make_runtime, plus the roomy rate buckets every
# write-heavy module needs so the process-global bucket stays order-neutral.
pytestmark = pytest.mark.usefixtures("roomy_rate_buckets")


def _audit_outcomes(root: Path) -> list[str]:
    """`outcome` of every audit row written so far, oldest first."""
    path = root / "logs" / "mcp-audit.jsonl"
    if not path.exists():
        return []
    return [
        str(json.loads(line)["outcome"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ------------------------------------------------------- search-hit read gate

def test_source_hit_gates_on_processed_twin() -> None:
    # An ingested source's content is only returned if its processed artifact
    # is readable — gate on archive/processed/<rel>.<ext>.md (the twin keeps
    # the source extension since the D7 collision fix).
    assert _hit_gate_path(
        "university/COMP0023/04_error_coding.pdf", "pdf-mineru"
    ) == "archive/processed/university/COMP0023/04_error_coding.pdf.md"


def test_knowledge_note_hit_gates_on_its_own_path() -> None:
    assert _hit_gate_path(
        "knowledge/projects/brain/brain.md", "knowledge-note"
    ) == "knowledge/projects/brain/brain.md"


def test_ingested_source_labelled_knowledge_gates_on_twin() -> None:
    # A source dropped at inbox/knowledge/x.pdf is LABELLED knowledge/ but is
    # NOT a vault note — origin, not the path prefix, decides.
    assert _hit_gate_path(
        "knowledge/x.pdf", "pdf-pypdf-fallback"
    ) == "archive/processed/knowledge/x.pdf.md"


def test_legacy_row_without_origin_falls_back_to_prefix() -> None:
    # Rows built before `origin` existed: a knowledge/ path gates on itself,
    # a non-knowledge path gates on its processed twin.
    assert _hit_gate_path("knowledge/projects/brain/brain.md", "") == (
        "knowledge/projects/brain/brain.md"
    )
    assert _hit_gate_path("uni/x.pdf", "") == "archive/processed/uni/x.pdf.md"


# ------------------------------------------------------------- write policy

def test_replace_refuses_oversize_content(
    mcp_vault: Path, make_cfg: CfgFactory, make_runtime: RuntimeFactory
) -> None:
    root = mcp_vault
    cfg, runtime = make_cfg(root), make_runtime(root)
    note = "knowledge/notes/x.md"
    tool_create_note(cfg, runtime, path=note, content="# x\n")
    with pytest.raises((ToolError, SafetyError)) as exc:
        tool_replace_note(cfg, runtime, path=note, content="x" * (MAX_NOTE_BYTES + 1))
    assert "max" in str(exc.value).lower()


def test_replace_refuses_nonexistent_note(
    mcp_vault: Path, make_cfg: CfgFactory, make_runtime: RuntimeFactory
) -> None:
    root = mcp_vault
    cfg, runtime = make_cfg(root), make_runtime(root)
    with pytest.raises(ToolError) as exc:
        tool_replace_note(cfg, runtime, path="knowledge/notes/absent.md", content="x")
    assert "does not exist" in str(exc.value).lower()


def test_note_verbs_refuse_inbox_paths(
    mcp_vault: Path, make_cfg: CfgFactory, make_runtime: RuntimeFactory
) -> None:
    # inbox/ is writable only via drop_inbox_file; the note verbs must refuse
    # it (WRITE_ALLOW_PREFIXES excludes inbox/) and leave a pending source
    # byte-identical — replace would otherwise be the one tool able to
    # destroy an uncommitted inbox file.
    root = mcp_vault
    cfg, runtime = make_cfg(root), make_runtime(root)
    (root / "inbox").mkdir()
    pending = root / "inbox" / "pending.md"
    pending.write_text("original source\n", encoding="utf-8")

    with pytest.raises((ToolError, SafetyError)) as exc:
        tool_replace_note(cfg, runtime, path="inbox/pending.md", content="DESTROYED")
    assert "denied" in str(exc.value).lower()
    assert pending.read_text(encoding="utf-8") == "original source\n"

    with pytest.raises((ToolError, SafetyError)) as exc2:
        tool_create_note(cfg, runtime, path="inbox/new.md", content="nope")
    assert "denied" in str(exc2.value).lower()


# ------------------------------------------------------ relations frontmatter

_RELATIONS_DICT = (
    "---\ntitle: x\nrelations:\n  rel: related_to\n  target: projects/other\n---\nbody\n"
)
_RELATIONS_MISSING_KEYS = (
    "---\ntitle: x\nrelations:\n  - related_to: knowledge/projects/x.md\n---\nbody\n"
)
_RELATIONS_UNKNOWN_REL = (
    "---\ntitle: x\nrelations:\n  - rel: same_as\n    target: people/anna\n---\nbody\n"
)
_RELATIONS_VALID = (
    "---\ntitle: x\nrelations:\n  - rel: related_to\n    target: projects/other\n---\nbody\n"
)


@pytest.mark.parametrize(
    "content", [_RELATIONS_DICT, _RELATIONS_MISSING_KEYS, _RELATIONS_UNKNOWN_REL]
)
def test_create_note_refuses_bad_relations(
    content: str, mcp_vault: Path, make_cfg: CfgFactory, make_runtime: RuntimeFactory,
    run_git: GitRunner,
) -> None:
    root = mcp_vault
    cfg, runtime = make_cfg(root), make_runtime(root)
    with pytest.raises(ToolError) as exc:
        tool_create_note(cfg, runtime, path="knowledge/notes/bad.md", content=content)
    assert "relations frontmatter rejected" in str(exc.value)
    # The message alone would still pass if the check moved AFTER the write
    # (AUD-006 itself): pin that nothing reached disk, git, or the audit log
    # as anything but a refusal.
    assert not (root / "knowledge/notes/bad.md").exists()
    assert run_git(root, "status", "--porcelain", "--untracked-files=no") == ""
    outcomes = _audit_outcomes(root)
    assert len(outcomes) == 1
    assert outcomes[0].startswith("refused:")


def test_create_note_accepts_valid_relations(
    mcp_vault: Path, make_cfg: CfgFactory, make_runtime: RuntimeFactory
) -> None:
    root = mcp_vault
    cfg, runtime = make_cfg(root), make_runtime(root)
    result = tool_create_note(cfg, runtime, path="knowledge/notes/good.md", content=_RELATIONS_VALID)
    assert result.committed


@pytest.mark.parametrize(
    "content", [_RELATIONS_DICT, _RELATIONS_MISSING_KEYS, _RELATIONS_UNKNOWN_REL]
)
def test_replace_note_refuses_bad_relations(
    content: str, mcp_vault: Path, make_cfg: CfgFactory, make_runtime: RuntimeFactory,
    run_git: GitRunner,
) -> None:
    root = mcp_vault
    cfg, runtime = make_cfg(root), make_runtime(root)
    note = "knowledge/notes/replace-target.md"
    tool_create_note(cfg, runtime, path=note, content="# x\n")
    before = (root / note).read_text(encoding="utf-8")
    head_before = run_git(root, "rev-parse", "HEAD")
    with pytest.raises(ToolError) as exc:
        tool_replace_note(cfg, runtime, path=note, content=content)
    assert "relations frontmatter rejected" in str(exc.value)
    # Replace is the destructive verb: the refusal must land before the
    # write, so the prior note is byte-identical and no commit happened.
    assert (root / note).read_text(encoding="utf-8") == before
    assert run_git(root, "rev-parse", "HEAD") == head_before
    assert run_git(root, "status", "--porcelain", "--untracked-files=no") == ""
    outcomes = _audit_outcomes(root)
    assert outcomes[0].startswith("ok")           # the create
    assert len(outcomes) == 2
    assert outcomes[1].startswith("refused:")


def test_replace_note_accepts_valid_relations(
    mcp_vault: Path, make_cfg: CfgFactory, make_runtime: RuntimeFactory
) -> None:
    root = mcp_vault
    cfg, runtime = make_cfg(root), make_runtime(root)
    note = "knowledge/notes/replace-good.md"
    tool_create_note(cfg, runtime, path=note, content="# x\n")
    result = tool_replace_note(cfg, runtime, path=note, content=_RELATIONS_VALID)
    assert result.committed


# ----------------------------------------------- pre-write branch guard (IMP-014)

def test_write_refused_before_disk_when_vault_on_other_branch(
    mcp_vault: Path, make_cfg: CfgFactory, make_runtime: RuntimeFactory,
    run_git: GitRunner,
) -> None:
    from mcp_server.identity import AGENT_VAR

    root = mcp_vault
    run_git(root, "checkout", "-q", "-b", "side")
    # Reset the identity afterwards: an unreset contextvar leaks into every
    # later test that runs in this task (it broke test_mcp_identity's
    # "identity must not leak" cases when this file collected first).
    token = AGENT_VAR.set("default")
    target = root / "knowledge" / "notes" / "stranded.md"
    try:
        with pytest.raises(ToolError, match="checked out"):
            tool_create_note(
                make_cfg(root), make_runtime(root), "knowledge/notes/stranded.md", "# x\n"
            )
    finally:
        AGENT_VAR.reset(token)
    assert not target.exists(), "the guard must fire before anything reaches disk"
    outcomes = _audit_outcomes(root)
    assert outcomes and outcomes[-1].startswith("refused:")
    assert "side" in outcomes[-1]


# ------------------------------------- entity-history preservation (F7)

# The dream pass rewrites entity notes in full via vault_replace_note. A
# rewrite may add and supersede, never drop a relation interval or a ## Log
# line — AGENTS.md's "supersede, never delete" was otherwise enforced only by
# the typed entity tools, which that path bypasses.
_ENTITY_NOTE = (
    "---\n"
    "title: Anna Kowalska\n"
    "type: person\n"
    "relations:\n"
    "  - rel: works_at\n"
    "    target: organisations/acme\n"
    "    valid_from: '2020-01-01'\n"
    "  - rel: attended\n"
    "    target: meetings/2026/2026-06-12-kern-call\n"
    "---\n"
    "\n"
    "# Anna Kowalska\n"
    "\n"
    "## Log\n"
    "\n"
    "- 2026-01-01 — joined acme ([[knowledge/assistant/inbox/a]])\n"
    "- 2026-02-01 — moved to Krakow ([[knowledge/assistant/inbox/b]])\n"
    "- 2026-03-01 — took over kern ([[knowledge/assistant/inbox/c]])\n"
)
_ENTITY_PATH = "knowledge/people/anna-kowalska.md"


def _entity_vault(
    root: Path, make_cfg: CfgFactory, make_runtime: RuntimeFactory
) -> tuple[ServerConfig, Runtime, Path]:
    cfg, runtime = make_cfg(root), make_runtime(root)
    tool_create_note(cfg, runtime, path=_ENTITY_PATH, content=_ENTITY_NOTE)
    return cfg, runtime, root


def test_replace_refuses_dropping_a_relation_and_writes_nothing(
    mcp_vault: Path, make_cfg: CfgFactory, make_runtime: RuntimeFactory
) -> None:
    cfg, runtime, root = _entity_vault(mcp_vault, make_cfg, make_runtime)
    before = (root / _ENTITY_PATH).read_text(encoding="utf-8")
    dropped = _ENTITY_NOTE.replace(
        "  - rel: attended\n    target: meetings/2026/2026-06-12-kern-call\n", ""
    )
    with pytest.raises(ToolError) as exc:
        tool_replace_note(cfg, runtime, path=_ENTITY_PATH, content=dropped)
    assert "drops history" in str(exc.value)
    assert "attended -> meetings/2026/2026-06-12-kern-call" in str(exc.value)
    assert "supersede" in str(exc.value)
    assert (root / _ENTITY_PATH).read_text(encoding="utf-8") == before


def test_replace_refuses_dropping_a_log_bullet(
    mcp_vault: Path, make_cfg: CfgFactory, make_runtime: RuntimeFactory
) -> None:
    cfg, runtime, root = _entity_vault(mcp_vault, make_cfg, make_runtime)
    before = (root / _ENTITY_PATH).read_text(encoding="utf-8")
    dropped = _ENTITY_NOTE.replace(
        "- 2026-02-01 — moved to Krakow ([[knowledge/assistant/inbox/b]])\n", ""
    )
    with pytest.raises(ToolError) as exc:
        tool_replace_note(cfg, runtime, path=_ENTITY_PATH, content=dropped)
    assert "## Log lines" in str(exc.value)
    assert "moved to Krakow" in str(exc.value)
    assert (root / _ENTITY_PATH).read_text(encoding="utf-8") == before


def test_replace_allows_closing_an_open_relation(
    mcp_vault: Path, make_cfg: CfgFactory, make_runtime: RuntimeFactory
) -> None:
    # Superseding is the sanctioned mutation: the open span comes back with
    # a valid_until. That is history, not loss.
    cfg, runtime, root = _entity_vault(mcp_vault, make_cfg, make_runtime)
    closed = _ENTITY_NOTE.replace(
        "  - rel: attended\n    target: meetings/2026/2026-06-12-kern-call\n",
        "  - rel: attended\n"
        "    target: meetings/2026/2026-06-12-kern-call\n"
        "    valid_until: '2026-06-12'\n",
    )
    result = tool_replace_note(cfg, runtime, path=_ENTITY_PATH, content=closed)
    assert result.committed
    assert "valid_until: '2026-06-12'" in (root / _ENTITY_PATH).read_text(encoding="utf-8")


def test_replace_allows_adding_a_relation_and_a_bullet(
    mcp_vault: Path, make_cfg: CfgFactory, make_runtime: RuntimeFactory
) -> None:
    cfg, runtime, root = _entity_vault(mcp_vault, make_cfg, make_runtime)
    grown = _ENTITY_NOTE.replace(
        "  - rel: attended\n",
        "  - rel: member_of\n    target: organisations/kern\n  - rel: attended\n",
    ).replace(
        "- 2026-03-01 — took over kern ([[knowledge/assistant/inbox/c]])\n",
        "- 2026-03-01 — took over kern ([[knowledge/assistant/inbox/c]])\n"
        "- 2026-04-01 — joined kern ([[knowledge/assistant/inbox/d]])\n",
    )
    result = tool_replace_note(cfg, runtime, path=_ENTITY_PATH, content=grown)
    assert result.committed
    on_disk = (root / _ENTITY_PATH).read_text(encoding="utf-8")
    assert "organisations/kern" in on_disk
    assert "2026-04-01 — joined kern" in on_disk
    assert "2026-02-01 — moved to Krakow" in on_disk


def test_replace_unaffected_for_note_without_relations_or_log(
    mcp_vault: Path, make_cfg: CfgFactory, make_runtime: RuntimeFactory
) -> None:
    # A plain note is not an entity note: a full rewrite that drops most of
    # its body stays legal.
    root = mcp_vault
    cfg, runtime = make_cfg(root), make_runtime(root)
    note = "knowledge/notes/plain.md"
    tool_create_note(cfg, runtime, path=note, content="# Plain\n\n- a bullet\n- another\n")
    result = tool_replace_note(cfg, runtime, path=note, content="# Plain\n\nrewritten\n")
    assert result.committed
    assert "a bullet" not in (root / note).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# AUD-079: the server's durability guarantee, now on ingest_lib.atomic
# ---------------------------------------------------------------------------

def test_atomic_write_interrupted_leaves_no_temp_and_keeps_the_old_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup used to run in `except OSError` / `except Exception`, neither of
    which catches KeyboardInterrupt: a Ctrl-C between mkstemp and os.replace
    left a `.<note>.*.tmp` in the vault for Obsidian to index and the next
    commit to pick up. The target keeps its old bytes either way."""
    target = tmp_path / "knowledge" / "notes" / "n.md"
    target.parent.mkdir(parents=True)
    target.write_text("old\n", encoding="utf-8")

    def interrupt(src: object, dst: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        _atomic_write_text(target, "new\n")

    assert [p.name for p in target.parent.iterdir()] == ["n.md"]
    assert target.read_text(encoding="utf-8") == "old\n"


def test_atomic_write_keeps_0600_and_redacts_os_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two halves of the write contract the refactor had to preserve: notes
    land at mkstemp's 0600 (the same mode ingest_lib.notes gives them, so a
    note's permissions don't flip with whichever writer touched it last), and
    an OSError never reaches the agent carrying absolute paths."""
    target = tmp_path / "knowledge" / "notes" / "n.md"
    _atomic_write_text(target, "body\n")
    assert target.stat().st_mode & 0o777 == 0o600

    def boom(src: object, dst: object) -> None:
        raise OSError(28, "No space left on device", str(target))

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(ToolError) as excinfo:
        _atomic_write_text(target, "more\n")
    assert str(excinfo.value) == "write failed"
    assert str(tmp_path) not in str(excinfo.value)
    assert [p.name for p in target.parent.iterdir()] == ["n.md"]
