"""Server-asserted provenance stamping for MCP-written notes.

Every Markdown note written through the MCP write tools gets provenance
keys asserted by the SERVER, never trusted from the client:

- ``author`` / ``written_via`` (+ ``memory_status`` in the memory areas)
  on create;
- ``last_written_by`` / ``written_via`` on replace and append, leaving
  any existing ``author`` / ``memory_status`` lines alone — the original
  attribution was server-asserted at create time and must survive edits.

Implementation is textual line surgery inside the frontmatter fence, not
a YAML round-trip: hand-edited frontmatter keeps its exact formatting
(quote style, key order, comments), and the body below the fence is
preserved byte-for-byte. ``yaml.safe_dump`` would normalise all of that
on every write. The trade-off is that we only manage whole top-level
``key: value`` lines — which is all these keys ever are.

Both helpers are pure ``text -> value`` so they unit-test without a
filesystem and the write tools stay the single place doing I/O.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Literal

# Make the ingest_lib package importable. Same shim the CLI scripts use.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from ingest_lib.concepts import slugify as _slugify  # type: ignore[import-not-found]  # noqa: E402
from ingest_lib.notes import _split_frontmatter  # type: ignore[import-not-found]  # noqa: E402
from ingest_lib.relations import parse_relations  # type: ignore[import-not-found]  # noqa: E402

# The four keys this module owns. Asserted lines matching ^<key>: are
# replaced wholesale; everything else in the fence is untouched.
_PROVENANCE_KEYS: tuple[str, ...] = (
    "author",
    "written_via",
    "last_written_by",
    "memory_status",
)

# Agent names are validated slugs (identity._NAME_RE) before they ever
# reach this module; the filter is defense in depth so a future caller
# can't smuggle YAML syntax or newlines into a frontmatter value.
_AGENT_SAFE = re.compile(r"[^a-z0-9_-]")


def _safe_agent(agent: str) -> str:
    cleaned = _AGENT_SAFE.sub("", agent.lower())[:32]
    return cleaned or "unknown"


# Whitespace-tolerant matcher for a top-level ``<key>:`` line. ``author :``
# (space before the colon) is still a YAML key, so an exact ``author:``
# match would let a client smuggle a duplicate spoofed key past the strip.
def _key_line_re(key: str) -> "re.Pattern[str]":
    return re.compile(rf"^\s*{re.escape(key)}\s*:")


def _carry_forward(prior: str) -> dict[str, str]:
    """Extract the server-owned ``author`` / ``memory_status`` lines from a
    note's EXISTING on-disk frontmatter, so a replace/append re-asserts the
    create-time attribution and consolidation state from what the server
    itself last wrote — never from the (forgeable) client-supplied body."""
    _fm, body = _split_frontmatter(prior)
    if body is prior:
        return {}
    lines = prior.splitlines()
    close_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            close_idx = i
            break
    if close_idx < 0:
        return {}
    carried: dict[str, str] = {}
    for line in lines[1:close_idx]:
        for key in ("author", "memory_status"):
            if key not in carried and _key_line_re(key).match(line):
                carried[key] = line.rstrip("\r\n")
    return carried


def _asserted_lines(
    *,
    agent: str,
    mode: Literal["create", "replace", "append"],
    memory_area: bool,
    prior: str | None,
) -> list[tuple[str, str]]:
    """(key, full line) pairs the server asserts for this write mode.

    On create the server mints author/written_via (+ memory_status in a
    memory area). On replace/append it stamps last_written_by/written_via
    and carries author/memory_status forward from the PRIOR on-disk note,
    so those server-owned keys survive an edit without ever trusting the
    client's copy of them.
    """
    who = f"agent:{_safe_agent(agent)}"
    if mode == "create":
        lines = [("author", f"author: {who!r}"), ("written_via", "written_via: mcp")]
        if memory_area:
            lines.append(("memory_status", "memory_status: unconsolidated"))
        return lines
    lines = [
        ("last_written_by", f"last_written_by: {who!r}"),
        ("written_via", "written_via: mcp"),
    ]
    carried = _carry_forward(prior) if prior is not None else {}
    for key in ("author", "memory_status"):
        if key in carried:
            lines.append((key, carried[key]))
    return lines


def stamp_provenance(
    content: str,
    *,
    agent: str,
    mode: Literal["create", "replace", "append"],
    memory_area: bool,
    prior: str | None = None,
) -> str:
    """Return ``content`` with server-asserted provenance keys in its
    frontmatter. The body is preserved byte-for-byte.

    ALL server-owned keys (``author``, ``written_via``, ``last_written_by``,
    ``memory_status``) are stripped from the client-supplied content first,
    then re-asserted by the server — a tool caller must never be able to
    spoof ``author: agent:someone-else`` or self-promote
    ``memory_status: consolidated`` past the consolidation gate. On
    replace/append the server-owned ``author``/``memory_status`` are
    re-derived from ``prior`` (the existing on-disk note), not from the
    client's body. Content without parseable frontmatter gets a minimal
    block prepended containing only the provenance keys.
    """
    asserted = _asserted_lines(
        agent=agent, mode=mode, memory_area=memory_area, prior=prior
    )

    # Agreement with notes._split_frontmatter is the contract: if the
    # downstream parsers would not see a frontmatter mapping here (no
    # fence, unterminated fence, non-mapping YAML), don't pretend one
    # exists — prepend a fresh minimal block instead. The parser returns
    # the INPUT object itself as the body in every no-frontmatter path,
    # so an identity check is the exact "was a fence parsed" signal.
    _parsed, body = _split_frontmatter(content)
    if body is content:
        block = "\n".join(line for _key, line in asserted)
        return f"---\n{block}\n---\n{content}"

    lines = content.splitlines(keepends=True)
    close_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            close_idx = i
            break
    if close_idx < 0:  # unreachable after the parse check; fail safe anyway
        block = "\n".join(line for _key, line in asserted)
        return f"---\n{block}\n---\n{content}"

    # Strip EVERY server-owned key the client may have supplied (not just
    # the ones re-asserted this mode) so nothing forged survives.
    strip_res = [_key_line_re(k) for k in _PROVENANCE_KEYS]
    kept: list[str] = []
    for line in lines[1:close_idx]:
        if any(r.match(line) for r in strip_res):
            continue  # client-supplied server-owned key: dropped
        kept.append(line)
    # New keys land just before the closing fence, after the user's keys.
    inserted = [line + "\n" for _key, line in asserted]
    return lines[0] + "".join(kept) + "".join(inserted) + "".join(lines[close_idx:])


def frontmatter_signature(content: str) -> tuple:
    """Normalised (topics, relations) from a note's frontmatter.

    This is the answer to "did this write change the concept/graph
    inputs?": topics feed concept notes (compared by their slug, exactly
    how ``rebuild_concepts`` groups them) and relations feed typed edges
    (compared by the fields that land in the graph — ``source`` is pure
    provenance and deliberately excluded). Body edits never change it.
    """
    frontmatter, _body = _split_frontmatter(content)
    topics = tuple(sorted({
        slug for slug in (_slugify(str(t)) for t in _topic_list(frontmatter)) if slug
    }))
    relations, _problems = parse_relations(frontmatter)
    relation_sig = tuple(sorted(
        (r.rel, r.target, r.valid_from, r.valid_until) for r in relations
    ))
    return (topics, relation_sig)


def _topic_list(frontmatter: dict[str, Any]) -> list[object]:
    """Tolerant read of ``topics:`` — a bare string counts as one topic,
    anything that isn't a string/list counts as none (mirrors how the
    knowledge scanner shrugs at malformed frontmatter)."""
    raw = frontmatter.get("topics")
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, list):
        return raw
    return []
