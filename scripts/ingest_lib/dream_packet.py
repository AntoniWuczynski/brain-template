"""The dream session's worklist: computed reproducibly from git history,
``metadata/index.jsonl`` and ``metadata/connections.jsonl``, and the
deterministic writer for what the LLM adjudication kept.

Split out of ``dream.py`` (AUD-121). This is the top of the dream sibling
stack: it assembles :class:`DreamPacket` out of the gate's
:class:`~.dream_gate_core.GateVerdict`, the compiled-truth candidates from
``dream_compiled`` and the contradiction hints from ``dream_contradictions``.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import TypedDict

from .config import VaultPaths
from .connections import load_edges
from .dream_compiled import (
    COMPILED_TRUTH_BUDGET,
    CONTRADICTION_BUDGET,
    EDIT_CAP,
    CompiledTruthCandidate,
    _compiled_truth_candidates,
)
from .dream_contradictions import (
    PROPOSAL_KIND,
    ContradictionCandidate,
    ProposalRelation,
    _contradictions,
    _relation_from_entry,
)
from .dream_gate_core import GateVerdict
from .dream_git import _git
from .dream_meetings import (
    MEETING_FILING_BUDGET,
    ProjectCatalogueEntry,
    UnfiledMeeting,
    project_catalogue,
    unfiled_meetings,
)
from .propose import ProposeResult, propose_fact
from .relations import entity_notes


class PairCandidate(TypedDict):
    """A semantic edge with no written-down relationship yet."""

    a: str
    b: str
    weight: float


class EditBudget(TypedDict):
    """How the SKILL's per-run edit cap is split across the jobs.

    ``total_note_edits`` mirrors the skill's tripwire (replaces of
    pre-existing notes). Compiled truth gets a fixed slice so it cannot
    starve digests, consolidation and the questions rewrite.
    ``contradiction_proposals`` bounds a different resource: fact-note
    CREATIONS, which the cap does not count.
    """

    total_note_edits: int
    compiled_truth: int
    reserved_for_other_jobs: int
    contradiction_proposals: int
    meeting_filings: int


class DreamPacket(TypedDict):
    """The dream session's entire worklist, computed deterministically."""

    since: str
    base_commit: str
    head_commit: str
    changed_notes: list[str]
    new_sources: list[str]
    candidate_pairs: list[PairCandidate]
    active_entities: list[str]
    existing_dream_notes: list[str]
    compiled_truth: list[CompiledTruthCandidate]
    compiled_truth_blocked: list[str]
    contradictions: list[ContradictionCandidate]
    unfiled_meetings: list[UnfiledMeeting]
    project_catalogue: list[ProjectCatalogueEntry]
    edit_budget: EditBudget


_ENTITY_ROOTS = ("knowledge/people/", "knowledge/projects/", "knowledge/organisations/")


def _written_connection_stems(paths: VaultPaths) -> set[str]:
    """File stems of the connection notes the dream has already written.
    Matched whole (never split on ``--``, which can occur inside a slug)."""
    conn_dir = paths.knowledge / "notes" / "dreams" / "connections"
    if not conn_dir.is_dir():
        return set()
    return {p.stem for p in conn_dir.glob("*.md") if p.is_file()}


def _candidate_pairs(paths: VaultPaths, *, top_k: int) -> list[PairCandidate]:
    """Semantic-but-never-cooccurring pairs: the embedding space thinks
    these concepts relate but no note records why. Weight-ranked.

    Pairs whose connection note already exists are dropped before the
    top_k cut — in either name order, since notes predating the edge
    list's ``a <= b`` invariant are stored as ``<b>--<a>`` — so handled
    pairs cannot permanently occupy the packet's slots."""
    edges = load_edges(paths)
    written = _written_connection_stems(paths)
    linked = {(e.a, e.b) for e in edges if e.kind != "semantic"}
    semantic = [e for e in edges if e.kind == "semantic" and (e.a, e.b) not in linked]
    semantic.sort(key=lambda e: (-e.weight, e.a, e.b))
    fresh = [e for e in semantic if f"{e.a}--{e.b}" not in written and f"{e.b}--{e.a}" not in written]
    return [PairCandidate(a=e.a, b=e.b, weight=e.weight) for e in fresh[:top_k]]


def _active_entities(changed_notes: tuple[str, ...]) -> list[str]:
    """Entity identifiers touched by the changeset. People notes are flat
    files (knowledge/people/anna.md -> knowledge/people/anna); projects and
    organisations are folders (knowledge/projects/brain/log/x.md ->
    knowledge/projects/brain)."""
    entities: set[str] = set()
    for note in changed_notes:
        for prefix in _ENTITY_ROOTS:
            if note.startswith(prefix):
                parts = note.split("/")
                if len(parts) > 3:
                    entities.add("/".join(parts[:3]))
                else:
                    entities.add(note.removesuffix(".md"))
    return sorted(entities)


def _existing_dream_notes(paths: VaultPaths) -> list[str]:
    dreams_dir = paths.knowledge / "notes" / "dreams"
    if not dreams_dir.is_dir():
        return []
    return sorted(
        str(p.relative_to(paths.root)) for p in dreams_dir.rglob("*.md") if p.is_file()
    )


def build_packet(
    paths: VaultPaths,
    *,
    verdict: GateVerdict,
    top_k: int = 10,
    compiled_truth_budget: int = COMPILED_TRUTH_BUDGET,
    contradiction_budget: int = CONTRADICTION_BUDGET,
    logger: logging.Logger,
) -> DreamPacket:
    head = _git(paths.root, "rev-parse", "HEAD").strip()
    active = _active_entities(verdict.changed_notes)
    entities = entity_notes(paths)
    compiled, blocked = _compiled_truth_candidates(
        paths,
        active_entities=active,
        changed_notes=verdict.changed_notes,
        base_commit=verdict.base_commit,
        budget=compiled_truth_budget,
    )
    packet = DreamPacket(
        since=verdict.since,
        base_commit=verdict.base_commit,
        head_commit=head,
        changed_notes=list(verdict.changed_notes),
        new_sources=list(verdict.new_sources),
        candidate_pairs=_candidate_pairs(paths, top_k=top_k),
        active_entities=active,
        existing_dream_notes=_existing_dream_notes(paths),
        compiled_truth=compiled,
        compiled_truth_blocked=blocked,
        contradictions=_contradictions(
            paths, active_entities=active, budget=contradiction_budget
        ),
        unfiled_meetings=unfiled_meetings(paths, entities),
        project_catalogue=project_catalogue(paths, entities),
        edit_budget=EditBudget(
            total_note_edits=EDIT_CAP,
            compiled_truth=compiled_truth_budget,
            reserved_for_other_jobs=EDIT_CAP - compiled_truth_budget,
            contradiction_proposals=contradiction_budget,
            meeting_filings=MEETING_FILING_BUDGET,
        ),
    )
    logger.info(
        "dream packet: %d changed note(s), %d new source(s), %d candidate pair(s), "
        "%d compiled-truth candidate(s), %d contradiction hint(s), "
        "%d unfiled meeting(s)",
        len(packet["changed_notes"]), len(packet["new_sources"]),
        len(packet["candidate_pairs"]), len(packet["compiled_truth"]),
        len(packet["contradictions"]), len(packet["unfiled_meetings"]),
    )
    return packet


# ---------------------------------------------------------------------------
# writing the survivors the LLM kept
# ---------------------------------------------------------------------------

class AdjudicatedFinding(TypedDict):
    """One packet contradiction the dream session decided is real.

    Every field except ``title``/``body`` comes from the packet verbatim —
    the proposal's path is a hash of its own content, so a reworded
    ``proposal_fact`` addresses a different file and the idempotency the
    inbox/archive checks depend on is gone. The LLM's own words go in the
    title and the body, which the path does not depend on.
    """

    node_id: str
    note_path: str
    proposal_path: str
    proposal_fact: str
    promote_relations: list[ProposalRelation]
    title: str
    body: str


class ProposalOutcome(TypedDict):
    """What :func:`write_proposals` did with one adjudicated finding."""

    proposal_path: str
    action: str     # "proposed" (written now) | "exists" (already in the inbox)


def _as_mapping(raw: object, where: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected a JSON object")
    return raw


def _require_str(mapping: dict[str, object], key: str, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: {key!r} must be a non-empty string")
    return value


def _parse_promote_relations(raw: object, where: str) -> list[ProposalRelation]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{where}: 'promote_relations' must be a list")
    entries: list[ProposalRelation] = []
    for i, item in enumerate(raw):
        entry = _as_mapping(item, f"{where} relation {i}")
        rel = _require_str(entry, "rel", f"{where} relation {i}")
        target = _require_str(entry, "target", f"{where} relation {i}")
        optional: dict[str, str] = {}
        for key in ("valid_from", "valid_until", "source"):
            value = entry.get(key, "")
            if not isinstance(value, str):
                raise ValueError(f"{where} relation {i}: {key!r} must be a string")
            optional[key] = value
        entries.append(ProposalRelation(
            rel=rel, target=target,
            valid_from=optional["valid_from"],
            valid_until=optional["valid_until"],
            source=optional["source"],
        ))
    return entries


def parse_adjudicated(raw: object) -> list[AdjudicatedFinding]:
    """Validate the JSON the dream session hands back after adjudication.

    Strict on purpose: this is the boundary between an LLM's output and a
    deterministic writer, so a missing or mistyped field is an error the
    caller sees, never a proposal written with half a contract.
    """
    if not isinstance(raw, list):
        raise ValueError("adjudicated findings must be a JSON array")
    findings: list[AdjudicatedFinding] = []
    for i, item in enumerate(raw):
        where = f"finding {i}"
        entry = _as_mapping(item, where)
        findings.append(AdjudicatedFinding(
            node_id=_require_str(entry, "node_id", where),
            note_path=_require_str(entry, "note_path", where),
            proposal_path=_require_str(entry, "proposal_path", where),
            proposal_fact=_require_str(entry, "proposal_fact", where),
            promote_relations=_parse_promote_relations(
                entry.get("promote_relations"), where
            ),
            title=_require_str(entry, "title", where),
            body=_require_str(entry, "body", where),
        ))
    return findings


def write_proposals(
    paths: VaultPaths,
    findings: list[AdjudicatedFinding],
    *,
    now: datetime,
    logger: logging.Logger,
) -> list[ProposalOutcome]:
    """Write the adjudicated survivors as fact notes, through the shared
    proposer.

    This is the deterministic half doing the writing, which is what makes
    AGENTS.md's "the fact-note proposer ... used by the dream pass's
    contradiction detector" true: every proposal carries the memory-fact
    contract (``approved: false``, ``memory_status: unconsolidated``) and
    the honest ``written_via: script`` / ``author: script:<kind>`` stamp
    that only a script may claim. The LLM adjudicates; it does not author
    the frontmatter.

    A finding whose recomputed path differs from the packet's is refused
    outright: the only way that happens is a reworded ``proposal_fact`` or
    an edited ``promote_relations``, and the packet's inbox/archive dedup
    (and therefore the budget) is derived from that path. Every finding is
    checked BEFORE any of them is written, so a refused run leaves no half
    of itself in the inbox.
    """
    def _propose(finding: AdjudicatedFinding, *, dry_run: bool) -> ProposeResult:
        return propose_fact(
            paths,
            kind=PROPOSAL_KIND,
            title=finding["title"],
            target=finding["node_id"],
            source=finding["note_path"].removesuffix(".md"),
            reason=finding["body"],
            relations=[_relation_from_entry(e) for e in finding["promote_relations"]],
            fact=finding["proposal_fact"],
            now=now,
            dry_run=dry_run,
        )

    for finding in findings:
        addressed = _propose(finding, dry_run=True)
        if addressed.path != finding["proposal_path"]:
            raise ValueError(
                f"proposal for {finding['node_id']} addresses {addressed.path!r}, "
                f"not the packet's {finding['proposal_path']!r} — the fact or "
                "its relations were edited; send them back unaltered"
            )

    outcomes: list[ProposalOutcome] = []
    for finding in findings:
        result = _propose(finding, dry_run=False)
        logger.info("dream proposal %s: %s", result.action, result.path)
        outcomes.append(ProposalOutcome(
            proposal_path=result.path, action=result.action
        ))
    return outcomes
