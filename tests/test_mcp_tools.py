"""``vault_update_compiled_truth``: the marker-scoped write tool.

The dream pass's compiled-truth job used to send whole entity notes through
``vault_replace_note``, with "only the text between the markers changes"
enforced by prose in a skill file. These tests are the mechanical version of
that promise: the server splices between the fence and refuses everything
else.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.conftest import McpEnv

from ingest_lib.dream import COMPILED_TRUTH_END, COMPILED_TRUTH_START
from mcp_server.errors import ToolError
from mcp_server.tools import tool_update_compiled_truth

pytestmark = pytest.mark.usefixtures("roomy_rate_buckets")

_NOTE = (
    "---\n"
    "title: Anna\n"
    "type: person\n"
    "relations:\n"
    "  - rel: works_at\n"
    "    target: organisations/acme\n"
    "---\n"
    "\n"
    f"{COMPILED_TRUTH_START}\n"
    "Anna works at Acme.\n"
    f"{COMPILED_TRUTH_END}\n"
    "\n"
    "## Overview\n"
    "\n"
    "Hand-written prose that must survive.\n"
    "\n"
    "## Log\n"
    "\n"
    "- 2026-06-01 — joined the platform team ([[knowledge/x]])\n"
)


# The server stamps these on every write (AGENTS.md); they are the only
# frontmatter lines this tool is allowed to add.
_PROVENANCE_KEYS = ("last_written_by:", "written_via:")


def _write(root: Path, rel: str, text: str) -> Path:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def _without_provenance(text: str) -> str:
    """``text`` with the server-stamped provenance lines dropped, so a test
    can compare every OTHER byte of the note."""
    return "".join(
        line for line in text.splitlines(keepends=True)
        if not line.startswith(_PROVENANCE_KEYS)
    )


def test_update_compiled_truth_replaces_only_the_fenced_block(
    mcp_env: McpEnv,
) -> None:
    root, cfg, runtime = mcp_env
    note = _write(root, "knowledge/people/anna.md", _NOTE)

    tool_update_compiled_truth(
        cfg, runtime,
        path="knowledge/people/anna.md",
        content="Anna left Acme in June 2026 and has no current employer.",
    )

    after = note.read_text(encoding="utf-8")
    assert _without_provenance(after) == _NOTE.replace(
        "Anna works at Acme.",
        "Anna left Acme in June 2026 and has no current employer.",
    )
    # Everything outside the fence, byte for byte.
    assert "  - rel: works_at" in after
    assert "Hand-written prose that must survive." in after
    assert "- 2026-06-01 — joined the platform team ([[knowledge/x]])" in after


def test_update_compiled_truth_stamps_provenance(mcp_env: McpEnv) -> None:
    """N6: AGENTS.md says the server stamps every note it writes. The
    stamped values come from the agent, not the clock, so two runs by the
    same agent leave the frontmatter byte-identical — which is what keeps
    ``dream._only_compiled_truth_changed`` seeing a fence-only edit."""
    root, cfg, runtime = mcp_env
    note = _write(root, "knowledge/people/anna.md", _NOTE)

    tool_update_compiled_truth(
        cfg, runtime, path="knowledge/people/anna.md", content="First truth.",
    )
    first = note.read_text(encoding="utf-8")
    assert "last_written_by: 'agent:agent-a'" in first
    assert "written_via: mcp" in first

    tool_update_compiled_truth(
        cfg, runtime, path="knowledge/people/anna.md", content="Second truth.",
    )
    second = note.read_text(encoding="utf-8")
    assert second.replace("Second truth.", "First truth.") == first


def test_update_compiled_truth_bootstraps_the_first_fence_below_the_frontmatter(
    mcp_env: McpEnv,
) -> None:
    """N2: the note has no fence yet. The server puts the first one directly
    under the frontmatter — never appended, because an entity note's last
    section is ``## Log`` and a fence inside it is read back as Log evidence
    by the very pass that wrote it."""
    root, cfg, runtime = mcp_env
    plain = _NOTE.replace(
        f"{COMPILED_TRUTH_START}\nAnna works at Acme.\n{COMPILED_TRUTH_END}\n\n", ""
    )
    note = _write(root, "knowledge/people/anna.md", plain)

    tool_update_compiled_truth(
        cfg, runtime, path="knowledge/people/anna.md", content="Anna works at Acme.",
    )

    after = _without_provenance(note.read_text(encoding="utf-8"))
    assert after == _NOTE
    assert after.index(COMPILED_TRUTH_END) < after.index("## Overview")
    assert after.index(COMPILED_TRUTH_START) < after.index("## Log")


def test_update_compiled_truth_bootstrap_ignores_a_documented_fence(
    mcp_env: McpEnv,
) -> None:
    """N7: a note quoting the markers inside a ``` block has no fence of its
    own. Counting the quoted pair read the note as fenced and overwrote the
    example."""
    root, cfg, runtime = mcp_env
    documented = (
        "---\ntitle: Anna\ntype: person\n---\n\n"
        "## Overview\n\nThe block looks like this:\n\n"
        f"```markdown\n{COMPILED_TRUTH_START}\n...\n{COMPILED_TRUTH_END}\n```\n"
    )
    note = _write(root, "knowledge/people/anna.md", documented)

    tool_update_compiled_truth(
        cfg, runtime, path="knowledge/people/anna.md", content="Anna works at Acme.",
    )

    after = _without_provenance(note.read_text(encoding="utf-8"))
    assert after == documented.replace(
        "---\n\n## Overview",
        f"---\n\n{COMPILED_TRUTH_START}\nAnna works at Acme.\n"
        f"{COMPILED_TRUTH_END}\n\n## Overview",
    )
    assert f"```markdown\n{COMPILED_TRUTH_START}\n...\n{COMPILED_TRUTH_END}\n```" in after


def test_update_compiled_truth_refuses_content_carrying_a_marker(
    mcp_env: McpEnv,
) -> None:
    """N1 case A: a payload that closes the fence itself escapes it. The
    splice was verbatim, so the text landed AFTER the END marker and the
    note was left with two ENDs — unwritable by this tool for ever."""
    root, cfg, runtime = mcp_env
    note = _write(root, "knowledge/people/anna.md", _NOTE)

    with pytest.raises(ToolError, match="must not contain the fence markers"):
        tool_update_compiled_truth(
            cfg, runtime, path="knowledge/people/anna.md",
            content=f"evil\n{COMPILED_TRUTH_END}\nESCAPED PAST THE FENCE\n",
        )
    assert note.read_text(encoding="utf-8") == _NOTE


def test_update_compiled_truth_refuses_the_skills_own_example_block(
    mcp_env: McpEnv,
) -> None:
    """N1 case L, the likeliest real trigger: the model copies the skill's
    step-6 example and sends the whole block, markers included."""
    root, cfg, runtime = mcp_env
    note = _write(root, "knowledge/people/anna.md", _NOTE)

    with pytest.raises(ToolError, match="must not contain the fence markers"):
        tool_update_compiled_truth(
            cfg, runtime, path="knowledge/people/anna.md",
            content=(
                f"{COMPILED_TRUTH_START}\n"
                "Anna works at Acme.\n"
                f"{COMPILED_TRUTH_END}"
            ),
        )
    assert note.read_text(encoding="utf-8") == _NOTE


def test_update_compiled_truth_refuses_a_broken_fence(mcp_env: McpEnv) -> None:
    root, cfg, runtime = mcp_env
    half = _NOTE.replace(f"{COMPILED_TRUTH_END}\n", "")
    note = _write(root, "knowledge/people/anna.md", half)

    with pytest.raises(ToolError, match="ambiguous compiled-truth fence"):
        tool_update_compiled_truth(
            cfg, runtime, path="knowledge/people/anna.md", content="anything",
        )
    assert note.read_text(encoding="utf-8") == half


def test_update_compiled_truth_refuses_the_memory_areas(mcp_env: McpEnv) -> None:
    root, cfg, runtime = mcp_env
    fact = _write(root, "knowledge/assistant/inbox/x.md", _NOTE)

    with pytest.raises(ToolError, match="memory lifecycle"):
        tool_update_compiled_truth(
            cfg, runtime, path="knowledge/assistant/inbox/x.md", content="anything",
        )
    assert fact.read_text(encoding="utf-8") == _NOTE


def test_update_compiled_truth_refuses_a_missing_note(mcp_env: McpEnv) -> None:
    _root, cfg, runtime = mcp_env
    with pytest.raises(ToolError, match="does not exist"):
        tool_update_compiled_truth(
            cfg, runtime, path="knowledge/people/nobody.md", content="anything",
        )


def test_update_compiled_truth_audits_the_write(mcp_env: McpEnv) -> None:
    root, cfg, runtime = mcp_env
    _write(root, "knowledge/people/anna.md", _NOTE)

    tool_update_compiled_truth(
        cfg, runtime, path="knowledge/people/anna.md", content="Fresh truth.",
    )

    audit = (root / "logs" / "mcp-audit.jsonl").read_text(encoding="utf-8")
    assert "vault_update_compiled_truth" in audit
