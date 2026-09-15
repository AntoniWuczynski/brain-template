"""Contradiction detection: propose-only hints.

Split out of ``dream.py`` (AUD-121). Two claims on one entity note that
look like they disagree — a HINT, never a verdict: the detectors are
lexical and interval arithmetic, so the LLM adjudicates and a human gates
the result. The job may only propose - it never edits the entity note.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, TypedDict

from .config import VaultPaths
from .dream_compiled import _entity_note_path, _log_bullets, _strip_compiled_truth
from .notes import _split_frontmatter
from .propose import propose_fact
from .relations import Relation, node_id_for_note, parse_relations

_DATE_RE: Final = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_BULLET_DATE_RE: Final = re.compile(r"^\d{4}-\d{2}-\d{2}\s*[\u2014\u2013-]\s*")
# The same leading date, read off a bullet that still carries its "- ".
_BULLET_LEADING_DATE_RE: Final = re.compile(r"^-\s*(\d{4}-\d{2}-\d{2})\b")
_WIKILINK_RE: Final = re.compile(r"\[\[[^\]]*\]\]")
_WORD_RE: Final = re.compile(r"[a-z0-9]{3,}")

# Closed lexicon of "this ended" cues. Lexical only: 'left the meeting' will
# match too, which is why the finding is a proposal a human gates, never an
# edit. Kept small on purpose — a long list is a long false-positive list.
_TERMINATION_CUES: Final[tuple[str, ...]] = (
    "departed",
    "ended",
    "former",
    "left",
    "leaving",
    "no longer",
    "resigned",
    "stepped down",
    "stopped working",
    "wound down",
)
_CUE_RE: Final = re.compile(
    r"\b(?:" + "|".join(re.escape(c) for c in _TERMINATION_CUES) + r")\b"
)

# The relations a termination cue can contradict: an edge that is meant to
# RUN until something ends it. The rest of the closed vocabulary is
# point-in-time (``attended``, ``met_at`` — the meeting happened, and then
# it was over) or deliberately timeless (``related_to``); an open entry
# there is the correct state, not a contradiction waiting to be flagged.
_CLOSABLE_RELS: Final[frozenset[str]] = frozenset(
    {"works_at", "member_of", "stakeholder_in", "collaborator_on"}
)

# Function words carry no claim: with them in the token set, "ran the dream
# pass" and "ran the dream gate" share 3 of 5 words and clear the bar. Kept
# to the words that actually show up in a dated one-line Log fact.
_STOPWORDS: Final[frozenset[str]] = frozenset({
    "and", "are", "for", "from", "had", "has", "have", "into", "not", "our",
    "that", "the", "this", "was", "were", "with",
})
# Two bullets sharing at least this share of their words are the same claim
# told twice often enough to be worth a human's glance.
_RESTATEMENT_MIN_OVERLAP: Final = 0.6
# Pairwise comparison is quadratic; a Log longer than this is compared over
# its most recent bullets only (newest last), which is where restatements land.
_RESTATEMENT_WINDOW: Final = 200
# The window bounds the INPUTS; a Log of near-identical bullets still emits
# O(n^2) findings out of them, and the packet's budget cut happens much
# later. This bounds the outputs per note.
_RESTATEMENT_MAX_FINDINGS: Final = 50
# Severity order for the packet's truncation: a self-contradicting relation
# interval is a hard error, a lexical cue is a hint, a restatement is tidying.
_KIND_RANK: Final[dict[str, int]] = {
    "relation_interval_conflict": 0,
    "relation_vs_log": 1,
    "restated_fact": 2,
}


class ProposalRelation(TypedDict):
    """One ``promote.relations`` entry on a contradiction proposal.

    Every key is present (empty string when unset) so the mapping converts
    to a :class:`~ingest_lib.relations.Relation` — and back — without
    optional-key narrowing, and so the proposal's content hash is stable.
    """

    rel: str
    target: str
    valid_from: str
    valid_until: str
    source: str


class ContradictionCandidate(TypedDict):
    """Two claims on one entity note that look like they disagree.

    A HINT, never a verdict: the detectors are lexical and interval
    arithmetic, so the LLM adjudicates and a human gates the result. The
    job may only propose - it never edits the entity note.
    """

    kind: str               # relation_interval_conflict | relation_vs_log | restated_fact
    node_id: str
    note_path: str
    left: str               # the first claim, verbatim
    right: str              # the second claim, verbatim
    detail: str             # why the detector paired them
    proposal_path: str      # vault-relative fact note to create (incl. .md)
    proposal_fact: str      # the promote.fact line that path is addressed by
    # The closure a human's approval would apply: the same (rel, target)
    # carrying valid_until. Empty when the finding implies no closure.
    promote_relations: list[ProposalRelation]


# The ``kind`` this job proposes under, in ingest_lib.propose's sense: it
# seeds the fact note's filename and its ``author: script:...`` stamp, and
# keeps these proposals from ever colliding with another pass's.
PROPOSAL_KIND: Final = "dream-contradiction"
_PROPOSAL_KIND: Final = PROPOSAL_KIND
# Every promoted contradiction line starts with this, and the detectors skip
# any ``## Log`` bullet that does. Without it the pass eats its own output:
# consolidate appends promote.fact to the Log, that line quotes the offending
# bullet (so it carries the target mention AND the termination cue), and the
# next run re-detects the settled contradiction against different text —
# a different content hash, so neither the inbox nor the archive check
# suppresses it. Stable by contract: changing this string re-arms every
# proposal a human has already promoted.
CONTRADICTION_FLAG_PREFIX: Final = "contradiction flagged by the dream pass"


def _relation_from_entry(entry: ProposalRelation) -> Relation:
    return Relation(
        rel=entry["rel"],
        target=entry["target"],
        valid_from=entry["valid_from"],
        valid_until=entry["valid_until"],
        source=entry["source"],
    )


@dataclass(frozen=True)
class _RawFinding:
    """A detector hit, before it is addressed to a fact note."""

    kind: str
    node_id: str
    note_path: str
    left: str
    right: str
    detail: str
    # The closure a human's approval would apply, if the finding implies one.
    relations: tuple[ProposalRelation, ...] = ()
    # Why this finding carries no closure, when the reason is worth a human's
    # attention. Rides in the fact text, so it reaches the entity's ## Log.
    flag: str = ""


def _finding(
    kind: str, node_id: str, note_path: str, left: str, right: str, detail: str,
    relations: tuple[ProposalRelation, ...] = (),
    flag: str = "",
) -> _RawFinding:
    return _RawFinding(
        kind=kind, node_id=node_id, note_path=note_path,
        left=left, right=right, detail=detail, relations=relations, flag=flag,
    )


def _claim_text(claim: str) -> str:
    """A claim as it reads inside a one-line sentence (## Log bullets arrive
    with their ``- `` marker, which would nest a bullet inside a bullet)."""
    return claim.removeprefix("- ").strip()


def _proposal_fact(raw: _RawFinding) -> str:
    """The one line a human's approval would append to the entity's ``## Log``.

    It records THAT the two claims disagree, never a resolution in prose:
    the resolution a ``relation_vs_log`` finding implies rides in
    ``promote.relations`` instead, as the closure of the relation the Log
    line says ended — so approving the proposal actually settles it.
    Deterministic, and it carries both claims, so it is also what makes one
    finding's proposal path differ from another's on the same note.
    """
    line = (
        f"{CONTRADICTION_FLAG_PREFIX} ({raw.kind}): "
        f'"{_claim_text(raw.left)}" vs "{_claim_text(raw.right)}"'
    )
    return f"{line} — {raw.flag}" if raw.flag else line


def _is_contradiction_flag(bullet: str) -> bool:
    """True for a ``## Log`` bullet this job's own proposal put there."""
    text = _BULLET_DATE_RE.sub("", _claim_text(bullet))
    return text.startswith(CONTRADICTION_FLAG_PREFIX)


def _describe_relation(relation: Relation) -> str:
    span = relation.valid_from or "(open start)"
    span += f"..{relation.valid_until}" if relation.valid_until else ".. (open)"
    return f"{relation.rel} -> {relation.target} [{span}]"


def _interval_conflicts(
    node_id: str, note_path: str, relations: list[Relation]
) -> list[_RawFinding]:
    """One (rel, target) both closed and continuously open across the closure.

    Superseding is legal and common — close the old interval, append a fresh
    one starting later. The contradiction is the other shape: an open entry
    that starts BEFORE (or without) a closure of the same edge, so the note
    asserts the relation both ended on a date and never stopped.
    """
    found: list[_RawFinding] = []
    by_edge: dict[tuple[str, str], list[Relation]] = {}
    for r in relations:
        by_edge.setdefault((r.rel, r.target), []).append(r)
    for entries in by_edge.values():
        closed = [r for r in entries if _DATE_RE.match(r.valid_until)]
        for open_entry in (r for r in entries if not r.valid_until):
            for closed_entry in closed:
                # A malformed valid_from ("last spring") would compare
                # lexically against the closure date and silently decide
                # the question; only a real date can rule the pair out.
                if (
                    _DATE_RE.match(open_entry.valid_from)
                    and open_entry.valid_from >= closed_entry.valid_until
                ):
                    continue    # a fresh interval starting after the closure
                found.append(_finding(
                    "relation_interval_conflict", node_id, note_path,
                    _describe_relation(open_entry), _describe_relation(closed_entry),
                    f"the note closes {open_entry.rel} -> {open_entry.target} on "
                    f"{closed_entry.valid_until} and also carries it as open from "
                    f"{open_entry.valid_from or '(open start)'}",
                ))
    return found


def _mentions_target(lowered_bullet: str, target: str) -> bool:
    """Does this ``## Log`` line talk about the relation's target?

    Two surface forms: the node id as a wikilink writes it (a substring —
    it is already a distinctive path), and the slug as words, matched on
    word boundaries the way ``_CUE_RE`` matches cues. Without the boundary,
    a project slugged ``check`` matches "checklist", "checked" and
    "checking" — and this vault has slugs called ``check``, ``agent``,
    ``brain``, ``claude`` and ``server``.
    """
    if target.lower() in lowered_bullet:
        return True
    words = target.rsplit("/", 1)[-1].replace("-", " ").lower()
    if not words:
        return False
    return re.search(rf"\b{re.escape(words)}\b", lowered_bullet) is not None


def _closure_relation(
    relation: Relation, bullet: str
) -> tuple[tuple[ProposalRelation, ...], str]:
    """The closure this finding implies — the same edge, ending on the date
    the ``## Log`` line carries — plus a flag when it implies none.

    It rides in ``promote.relations``, so a human approving the proposal
    hands ``consolidate.py`` a real ``valid_until`` and
    ``upsert_relation_in_text`` closes the open span — supersede-never-
    delete, done by the one code path that owns it. Without a date on the
    bullet there is nothing to close it ON, and the proposal carries the
    disagreement alone.

    A date EARLIER than the relation's own ``valid_from`` is not a closure
    either: that is the rejoin shape (left in June, came back in August, the
    old span already closed), and writing it would hand consolidate an
    incoherent ``from 2026-08-01 until 2026-06-02``. Propose nothing and say
    why in the fact text, so the human sees the stale Log line rather than a
    silently mangled interval.
    """
    match = _BULLET_LEADING_DATE_RE.match(bullet)
    if match is None:
        return (), ""
    ended = match.group(1)
    if _DATE_RE.match(relation.valid_from) and ended < relation.valid_from:
        return (), (
            f"no closure proposed: the Log line's date {ended} precedes the "
            f"relation's own start {relation.valid_from}"
        )
    return (ProposalRelation(
        rel=relation.rel,
        target=relation.target,
        valid_from=relation.valid_from,
        valid_until=ended,
        source=relation.source,
    ),), ""


def _relation_vs_log(
    node_id: str, note_path: str, relations: list[Relation], bullets: list[str]
) -> list[_RawFinding]:
    """An open relation while a Log line about that same target says it ended.

    Only the CLOSABLE relations: ``attended`` and ``met_at`` are
    point-in-time by design (the closed vocabulary's own reading), so such
    an edge is always open and correctly so — "the meeting ended" is not a
    contradiction, and one busy person note carries dozens of them.
    """
    found: list[_RawFinding] = []
    for relation in relations:
        if relation.valid_until or relation.rel not in _CLOSABLE_RELS:
            continue
        for bullet in bullets:
            lowered = bullet.lower()
            if not _mentions_target(lowered, relation.target):
                continue
            cue = _CUE_RE.search(lowered)
            if cue is None:
                continue
            closure, flag = _closure_relation(relation, bullet)
            found.append(_finding(
                "relation_vs_log", node_id, note_path,
                _describe_relation(relation), bullet,
                f"relation is open while a ## Log line about {relation.target} "
                f"carries the termination cue {cue.group(0)!r}",
                closure, flag,
            ))
    return found


def _bullet_tokens(bullet: str) -> frozenset[str]:
    stripped = bullet.removeprefix("- ").strip()
    stripped = _BULLET_DATE_RE.sub("", stripped)
    stripped = _WIKILINK_RE.sub(" ", stripped)
    return frozenset(
        w for w in _WORD_RE.findall(stripped.lower()) if w not in _STOPWORDS
    )


def _restated_facts(
    node_id: str, note_path: str, bullets: list[str]
) -> list[_RawFinding]:
    """The same fact asserted twice in one Log, worded differently.

    Bounded at both ends: the window caps how many bullets are compared,
    ``_RESTATEMENT_MAX_FINDINGS`` caps how many pairs one note may emit —
    a Log of near-identical bullets is quadratic in findings long before
    the packet's budget cut would notice.
    """
    window = bullets[-_RESTATEMENT_WINDOW:]
    tokens = [_bullet_tokens(b) for b in window]
    found: list[_RawFinding] = []
    for i in range(len(window)):
        if not tokens[i]:
            continue
        for j in range(i + 1, len(window)):
            if len(found) >= _RESTATEMENT_MAX_FINDINGS:
                return found
            if not tokens[j] or window[i] == window[j]:
                continue
            union = tokens[i] | tokens[j]
            overlap = len(tokens[i] & tokens[j]) / len(union)
            if overlap < _RESTATEMENT_MIN_OVERLAP:
                continue
            found.append(_finding(
                "restated_fact", node_id, note_path, window[i], window[j],
                f"{overlap:.2f} word overlap between two ## Log lines",
            ))
    return found


def _address_proposals(
    paths: VaultPaths, raws: list[_RawFinding], *, budget: int
) -> list[ContradictionCandidate]:
    """Give each hit the fact note it would become, dropping handled ones.

    The destination comes from :func:`ingest_lib.propose.propose_fact` in
    ``dry_run`` mode — the shared proposer owns the inbox filename contract,
    so this job and every other deterministic proposer address the same
    candidate to the same file. Dry-run because the packet must not write:
    these detectors are lexical, most hits are false positives, and the
    LLM half adjudicates before anything reaches the human's queue.

    Skipping proposals that already exist (inbox) or were already promoted
    (archive) is what keeps the budget useful: a standing high-severity
    finding would otherwise occupy the same slot every night and starve
    everything detected since — the guard ``_candidate_pairs`` already
    applies to connection notes.
    """
    addressed: list[ContradictionCandidate] = []
    seen: set[str] = set()
    for raw in raws:
        if len(addressed) >= budget:
            break
        fact = _proposal_fact(raw)
        result = propose_fact(
            paths,
            kind=_PROPOSAL_KIND,
            title=f"Contradiction on {raw.node_id}",
            target=raw.node_id,
            source=raw.note_path.removesuffix(".md"),
            reason=raw.detail,
            relations=[_relation_from_entry(e) for e in raw.relations],
            fact=fact,
            dry_run=True,
        )
        if result.action in ("exists", "archived"):
            continue
        # Two findings can address ONE file (a note declaring the same edge
        # twice detects the same conflict twice); without this the packet
        # spends two budget slots on one proposal.
        if result.path in seen:
            continue
        seen.add(result.path)
        addressed.append(ContradictionCandidate(
            kind=raw.kind,
            node_id=raw.node_id,
            note_path=raw.note_path,
            left=raw.left,
            right=raw.right,
            detail=raw.detail,
            proposal_path=result.path,
            proposal_fact=fact,
            promote_relations=list(raw.relations),
        ))
    return addressed


def _contradictions(
    paths: VaultPaths, *, active_entities: list[str], budget: int
) -> list[ContradictionCandidate]:
    """Contradiction hints across the changeset's entity notes.

    Scope is deliberately the entities the changeset touched — the dream pass
    is change-driven, and a whole-vault pairwise scan is neither bounded nor
    something the edit budget could act on. Findings are sorted by severity
    then note, then addressed to fact notes and truncated to ``budget``.
    """
    found: list[_RawFinding] = []
    for entity in active_entities:
        rel = _entity_note_path(paths.root, entity)
        if rel is None:
            continue
        try:
            text = (paths.root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        node_id = node_id_for_note(rel)
        frontmatter, _body = _split_frontmatter(text)
        relations, _problems = parse_relations(frontmatter)
        # A promoted proposal's own Log line quotes the claims that produced
        # it; reading it back as evidence is how the detector re-arms itself
        # against a contradiction a human has already answered.
        # Stripped of the compiled-truth block first: it is this pass's own
        # summary of the Log, so reading it back as a Log line manufactures a
        # ``restated_fact`` against every bullet it summarises.
        bullets = [
            b for b in _log_bullets(_strip_compiled_truth(text))
            if not _is_contradiction_flag(b)
        ]
        found.extend(_interval_conflicts(node_id, rel, relations))
        found.extend(_relation_vs_log(node_id, rel, relations, bullets))
        found.extend(_restated_facts(node_id, rel, bullets))
    found.sort(key=lambda r: (_KIND_RANK[r.kind], r.node_id, r.left, r.right))
    return _address_proposals(paths, found, budget=budget)
