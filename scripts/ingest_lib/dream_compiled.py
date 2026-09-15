"""Compiled truth: a marker-guarded "current best understanding" block on
entity notes.

Split out of ``dream.py`` (AUD-121). This is the module every other dream
sibling treats as the shared base for compiled-truth primitives: the gate
(``dream_gate_core``) needs :func:`marker_state` and
:func:`_strip_compiled_truth` to tell a fence-only edit from real new
information, and the contradiction detector (``dream_contradictions``)
reads Log bullets and entity note paths through the same helpers — so
those live here rather than in either of them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final, TypedDict

from .config import VaultPaths
from .dream_git import GitError, _git
from .notes import _split_frontmatter
from .relations import (  # private helpers: one definition of "what is a ## Log bullet"
    _HEADING_RE,
    _LOG_HEADING,
    Relation,
    _code_fence_flags,
    node_id_for_note,
    parse_relations,
)


class CompiledTruthCandidate(TypedDict):
    """An entity note whose compiled-truth block is worth refreshing.

    ``marker_state`` is ``absent`` (no block yet - the job writes the first
    one) or ``ok`` (a block exists between the markers and is replaced in
    place). ``broken`` notes never become candidates; they land in the
    packet's ``compiled_truth_blocked`` list instead.

    The three evidence fields are what makes a note a candidate: the block
    is a synthesis of the entity's ``## Log`` AND its open relations, so
    either moving is new evidence. ``new_log_bullets`` counts bullets added
    anywhere under the entity (a folder-backed project's evidence lands in
    ``log/<date>.md``, not in the overview's own ``## Log``),
    ``changed_notes`` counts the changeset's notes under the entity, and
    ``relations_changed`` says the rendered open relations differ from
    ``base_commit`` — a closed relation appends no Log bullet at all.
    """

    node_id: str            # canonical id: people/anna, projects/brain/brain
    note_path: str          # vault-relative, including .md
    marker_state: str       # "absent" | "ok"
    log_bullets: int        # total ## Log bullets on the entity note
    new_log_bullets: int    # ## Log bullets added under the entity since base_commit
    changed_notes: int      # changeset notes under the entity, besides its own note
    relations_changed: bool  # open relations differ from base_commit
    open_relations: list[str]   # "<rel> -> <target> (since <date>)", no valid_until


# The block the compiled-truth job owns is FENCED by these two markers, the
# same idea as the concept notes' AUTO-GENERATED-END marker
# (mcp_server.tools.tool_update_concept_user_section): everything outside is
# hand-written and survives untouched. A fence rather than a single marker
# because an entity note has hand-written prose on BOTH sides — frontmatter
# and the note's own body above, the ## Log below — where a concept note has
# only generated-above / hand-written-below.
COMPILED_TRUTH_START: Final = "<!-- COMPILED-TRUTH-START -->"
COMPILED_TRUTH_END: Final = "<!-- COMPILED-TRUTH-END -->"

# The skill's tripwire: at most this many pre-existing notes replaced per run.
EDIT_CAP: Final = 10
# Compiled truth's slice of it. Deliberately small: digests, consolidation
# and the questions rewrite need the rest, and a compiled-truth block that
# lags one night is harmless.
COMPILED_TRUTH_BUDGET: Final = 3
# Contradiction findings are note CREATIONS (fact notes in the assistant
# inbox), which the edit cap does not count — this bounds the packet and the
# human's review queue instead.
CONTRADICTION_BUDGET: Final = 5


def _marker_offsets(text: str) -> tuple[list[int], list[int]]:
    """Offsets of every START / END marker that is real Markdown, in file
    order.

    Occurrences inside a fenced code block do not count — a note DOCUMENTING
    the fence (this skill's own step 6 example, an AGENTS.md excerpt) carries
    both markers in a ```` ``` ```` block, and reading those as the note's own
    fence would have the tool overwrite the example. Same fence detection
    ``_log_bullets`` already uses, so "what counts as real Markdown" has one
    definition in this module.
    """
    lines = text.split("\n")
    in_fence = _code_fence_flags(lines)
    starts: list[int] = []
    ends: list[int] = []
    pos = 0
    for i, line in enumerate(lines):
        if not in_fence[i]:
            for marker, found in ((COMPILED_TRUTH_START, starts), (COMPILED_TRUTH_END, ends)):
                at = line.find(marker)
                while at >= 0:
                    found.append(pos + at)
                    at = line.find(marker, at + 1)
        pos += len(line) + 1
    return starts, ends


def marker_state(text: str) -> str:
    """``"absent"`` | ``"ok"`` | ``"broken"`` for a note's compiled-truth fence.

    ``broken`` is any fence a rewrite could not land on safely: one marker
    without the other, END before START, or a duplicated marker. Mirrors
    concepts.py's refusal to regenerate a note with ambiguous markers —
    guessing where the block ends is how a hand-written section gets eaten.
    """
    starts, ends = _marker_offsets(text)
    if not starts and not ends:
        return "absent"
    if len(starts) != 1 or len(ends) != 1:
        return "broken"
    return "ok" if starts[0] < ends[0] else "broken"


def compiled_truth_span(text: str) -> tuple[int, int] | None:
    """``(first byte of the block, offset of the END marker)`` for a note
    whose fence is ``ok``; ``None`` otherwise.

    The one place that decides WHERE the block is: ``_strip_compiled_truth``
    here and ``mcp_server.tools.tool_update_compiled_truth`` both slice on
    it, so neither can land on a marker the other would have ignored (one
    inside a code fence, say).
    """
    starts, ends = _marker_offsets(text)
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        return None
    return starts[0] + len(COMPILED_TRUTH_START), ends[0]


def _strip_compiled_truth(text: str) -> str:
    """The note with its compiled-truth block emptied, so two revisions that
    differ only inside the fence compare equal — and so no reader of the
    note's prose (``_log_bullets`` above all) can mistake the dream's own
    summary for evidence the vault gained."""
    span = compiled_truth_span(text)
    if span is None:
        return text
    start, end = span
    return text[:start] + text[end:]


def _log_bullets(text: str) -> list[str]:
    """The note's ``## Log`` bullets, stripped, in file order.

    Same section detection ``relations.append_fact_to_log`` writes with (and
    ``mcp_server.tools._log_section`` guards with): the first non-fenced line
    that is exactly ``## Log``, ending at the next heading of any level.
    """
    lines = text.split("\n")
    in_fence = _code_fence_flags(lines)
    heading_idx = -1
    for i, line in enumerate(lines):
        if not in_fence[i] and line.strip() == _LOG_HEADING:
            heading_idx = i
            break
    if heading_idx < 0:
        return []
    end = len(lines)
    for j in range(heading_idx + 1, len(lines)):
        if not in_fence[j] and _HEADING_RE.match(lines[j]):
            end = j
            break
    return [
        lines[j].rstrip("\r").strip()
        for j in range(heading_idx + 1, end)
        if lines[j].rstrip("\r").strip().startswith("- ")
    ]


def _entity_note_path(root: Path, active_entity: str) -> str | None:
    """``knowledge/people/anna`` -> ``knowledge/people/anna.md``;
    ``knowledge/projects/brain`` -> ``knowledge/projects/brain/brain.md``.

    Resolution is by existence only (AGENTS.md 'folder-backed entities'):
    a project's canonical note is the overview INSIDE its folder, and the
    flat path is checked first because a person note is flat.
    """
    flat = f"{active_entity}.md"
    if (root / flat).is_file():
        return flat
    slug = active_entity.rsplit("/", 1)[-1]
    overview = f"{active_entity}/{slug}.md"
    if (root / overview).is_file():
        return overview
    return None


def _text_at(root: Path, commit: str, rel: str) -> str | None:
    """The file's content at ``commit``, or None if it wasn't there."""
    if not commit:
        return None
    try:
        return _git(root, "show", f"{commit}:{rel}")
    except GitError:
        return None


def _render_open_relations(relations: list[Relation]) -> list[str]:
    """``"<rel> -> <target>"``, with ``(since <date>)`` when the entry says
    so — the block is asked to state since when, and re-deriving that from
    the note is work the packet can do once, deterministically."""
    return sorted(
        f"{r.rel} -> {r.target}" + (f" (since {r.valid_from})" if r.valid_from else "")
        for r in relations if not r.valid_until
    )


def _new_log_bullets(root: Path, base_commit: str, rel: str, text: str) -> int:
    """``## Log`` bullets on ``rel`` that were not there at ``base_commit``.

    Both revisions are read with the compiled-truth block stripped. The
    fence lives above ``## Log`` on a well-formed note, but nothing enforces
    that — and a block that ends up INSIDE the Log section would otherwise
    be counted as new evidence by the very job that wrote it.
    """
    bullets = _log_bullets(_strip_compiled_truth(text))
    if not bullets:
        return 0
    old_text = _text_at(root, base_commit, rel)
    if old_text is None:
        return len(bullets)
    seen = set(_log_bullets(_strip_compiled_truth(old_text)))
    return sum(1 for b in bullets if b not in seen)


def _entity_changeset(
    entity: str, note_rel: str, changed_notes: tuple[str, ...]
) -> list[str]:
    """The changeset's notes belonging to this entity, other than its own
    note: a folder-backed project's evidence is its ``log/<date>.md`` and
    curated sub-notes, which is where every real project's activity lands."""
    prefix = f"{entity}/"
    return [n for n in changed_notes if n != note_rel and n.startswith(prefix)]


def _compiled_truth_candidates(
    paths: VaultPaths,
    *,
    active_entities: list[str],
    changed_notes: tuple[str, ...],
    base_commit: str,
    budget: int,
) -> tuple[list[CompiledTruthCandidate], list[str]]:
    """Entity notes worth a compiled-truth refresh, plus the blocked ones.

    A note qualifies when it has something to compile AND new evidence to
    compile it from. The block synthesises the ``## Log`` PLUS the open
    relations, so evidence is either of them moving:

    - ``## Log`` bullets added since ``base_commit`` anywhere under the
      entity — the overview's own Log, and every changed note under a
      folder-backed project;
    - any other changed note under the entity (a project's ``log/<date>.md``
      carries its bullets under its own headings, not under a ``## Log``);
    - a difference in the rendered open relations, which is how a CLOSED
      relation registers: ``upsert_relation_in_text`` edits frontmatter and
      appends no bullet, so without this the block would assert a fact the
      note has since retired, for ever.

    A quiet entity — nothing added, nothing closed — never burns an edit,
    whether or not it already has a block. Ranked by how much new evidence
    landed, then by node id; truncated to ``budget``.
    """
    candidates: list[CompiledTruthCandidate] = []
    blocked: list[str] = []
    for entity in active_entities:
        rel = _entity_note_path(paths.root, entity)
        if rel is None:
            continue
        try:
            text = (paths.root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        state = marker_state(text)
        if state == "broken":
            blocked.append(rel)
            continue
        bullets = _log_bullets(_strip_compiled_truth(text))
        frontmatter, _body = _split_frontmatter(text)
        relations, _problems = parse_relations(frontmatter)
        open_rels = _render_open_relations(relations)

        new_bullets = _new_log_bullets(paths.root, base_commit, rel, text)
        members = _entity_changeset(entity, rel, changed_notes)
        for member in members:
            try:
                member_text = (paths.root / member).read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                continue
            new_bullets += _new_log_bullets(paths.root, base_commit, member, member_text)

        old_text = _text_at(paths.root, base_commit, rel)
        if old_text is None:
            relations_changed = bool(open_rels)
        else:
            old_fm, _old_body = _split_frontmatter(old_text)
            old_relations, _old_problems = parse_relations(old_fm)
            relations_changed = _render_open_relations(old_relations) != open_rels

        if not new_bullets and not members and not relations_changed:
            continue    # no new evidence: a refresh would say the same thing
        if state == "absent" and not (bullets or open_rels or new_bullets or members):
            # Nothing to compile a first block out of: no Log, no open
            # relations, and nothing under the entity to synthesise from.
            # (An entity whose relations all just CLOSED has something to
            # say, but only once it has a block saying the opposite.)
            continue

        candidates.append(CompiledTruthCandidate(
            node_id=node_id_for_note(rel),
            note_path=rel,
            marker_state=state,
            log_bullets=len(bullets),
            new_log_bullets=new_bullets,
            changed_notes=len(members),
            relations_changed=relations_changed,
            open_relations=open_rels,
        ))
    candidates.sort(key=lambda c: (
        -(c["new_log_bullets"] + c["changed_notes"]),
        not c["relations_changed"],
        c["node_id"],
    ))
    return candidates[:budget], sorted(blocked)
