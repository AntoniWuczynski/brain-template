"""Sweep checks over the vault's link/relation/entity graph — split out of
``sweep.py`` (AUD-121). Pure compute, same "never raises, always a finding"
contract as the rest of the sweep.

Finding categories produced here: ``dangling-wikilink``,
``wikilink-source-local-only``, ``wikilink-not-canonical``,
``wikilink-ambiguous``, ``relation-problem``, ``relation-dangling-target``,
``relation-bad-date``, ``relation-inverted-interval``, ``relation-overlap``,
``concept-fragmentation``, ``entity-duplicate``, ``alias-collision``."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .concepts import slugify
from .config import VaultPaths
from .knowledge import KNOWLEDGE_NOTE_DIRS
from .metadata import IndexRecord
from .notes import _split_frontmatter  # private helper, but module-internal
from .relations import (
    _ASSISTANT_HISTORY_PREFIXES,  # private, but the graph and the sweep must agree
    Relation,
    note_path_for_node,
    parse_relations,
)
from .sweep_common import Finding, _git_ignored_paths, _read_text

_WIKILINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DIGITS_RE = re.compile(r"\d+")
# A title that is nothing but an email address: the calendar payload carried
# no display name, so the MCP write path slugged the address into a node id.
_EMAIL_TITLE_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
# An entity note whose project/lifecycle status says it is no longer the live
# node — its aliases may overlap a live note's without ambiguity.
_INACTIVE_STATUSES = frozenset({"archived", "complete", "completed", "superseded"})
_ENTITY_DIRS: tuple[str, ...] = ("people", "organisations", "projects", "chats")

# Edit-distance ceiling for two concept slugs to count as "the same
# concept spelled twice". 2 covers a typo or a joined/hyphenated variant
# without pulling genuinely different short slugs together too often.
_FRAGMENT_MAX_DISTANCE = 2


def _has_stem_sibling(candidate: Path) -> bool:
    """Extension-stripped link to a binary source: per the vault-wide
    convention (AGENTS.md mandates extensionless wikilinks) generated index
    notes write ``[[archive/raw/x/y]]`` for ``y.pdf``. The link resolves
    when the parent directory holds any non-md file matching ``<stem>.*``."""
    parent, prefix = candidate.parent, candidate.name + "."
    if not candidate.name or not parent.is_dir():
        return False
    return any(
        f.is_file() and f.name.startswith(prefix) and not f.name.endswith(".md")
        for f in parent.iterdir()
    )


def _note_stem_index(root: Path) -> dict[str, list[str]]:
    """Every note in the vault, grouped by file stem — the index Obsidian's
    short links resolve against. Dot-directories (``.git``, ``.obsidian``,
    ``.venv``) are not part of the vault. Deterministic: paths stay sorted."""
    index: dict[str, list[str]] = defaultdict(list)
    for md in sorted(root.rglob("*.md")):
        if not md.is_file():
            continue
        rel = md.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        index[md.stem].append(rel.as_posix())
    return index


def _short_link_matches(target: str, stems: dict[str, list[str]]) -> list[str]:
    """The notes a non-vault-relative ``[[target]]`` resolves to the way
    Obsidian does — any note whose path ENDS with the link, so ``[[foo]]``
    matches ``a/b/foo.md`` and ``[[log/foo]]`` matches ``a/log/foo.md``."""
    suffix = f"/{target}.md"
    return [p for p in stems.get(target.rsplit("/", 1)[-1], ()) if p.endswith(suffix)]


def _check_wikilinks(
    paths: VaultPaths, latest: dict[str, IndexRecord]
) -> list[Finding]:
    """dangling-wikilink: a [[target]] in a knowledge/ note that resolves
    to none of ``<vault>/<target>.md``, ``<vault>/<target>`` (embeds of
    existing assets), or ``<vault>/<target>.*`` (extension-stripped links
    to binary sources). External http(s) links are skipped.

    A link that is none of those may still be a short link, which Obsidian
    resolves by name across the whole vault (F8: most of the "dangling"
    findings were these, burying the real breaks). Such links get their own
    categories — ``wikilink-not-canonical`` when they resolve, and
    ``wikilink-ambiguous`` when the name matches several notes and the
    reader cannot know which one was meant."""
    findings: list[Finding] = []
    if not paths.knowledge.is_dir():
        return findings
    stems = _note_stem_index(paths.root)
    # Templates carry placeholder links ([[knowledge/people/<slug>]]) by
    # design — linting them would flag every fresh vault.
    templates = paths.knowledge_index / "templates"
    # The sweep report itself embeds real [[target]] links inside its
    # dangling-link findings, so linting it re-flags its own previous copy
    # forever (a count that never converges). Exclude it.
    report_note = paths.knowledge_index / "sweep-report.md"
    # A wikilink to a raw source is written extension-stripped (AGENTS.md),
    # but .gitignore entries and git's index are keyed by the real filename
    # — so a dangling ``archive/raw/...`` target must be resolved back to
    # its recorded ``raw_path`` before asking git whether it is ignored.
    raw_by_stem = {
        rec.raw_path.rsplit(".", 1)[0]: rec.raw_path
        for rec in latest.values() if rec.raw_path
    }
    # Collected first and resolved against git in one batched call, the
    # same way _check_archive tells a real loss from a source .gitignore
    # deliberately keeps out of the repository (AUD-117).
    dangling: list[tuple[str, str]] = []  # (note rel path, target)
    for md in sorted(paths.knowledge.rglob("*.md")):
        if not md.is_file() or templates in md.parents or md == report_note:
            continue
        text = _read_text(md)
        if text is None:
            continue
        rel = md.relative_to(paths.root).as_posix()
        rel_dir = md.parent.relative_to(paths.root).as_posix()
        seen: set[str] = set()  # one finding per distinct target per note
        for raw in _WIKILINK_RE.findall(text):
            target = raw.strip()
            if not target or target.startswith(("http://", "https://")):
                continue
            target = target.split("|", 1)[0].split("#", 1)[0].strip()
            if not target or target in seen:
                continue
            # `[[archive/raw/<rel/path>]]` in a note that documents the note
            # format is a placeholder, not a link: an angle-bracketed
            # segment can never name a real file (F4 residual).
            if "<" in target or ">" in target:
                continue
            seen.add(target)
            if (paths.root / f"{target}.md").is_file() or (paths.root / target).exists():
                continue
            if _has_stem_sibling(paths.root / target):
                continue
            matches = _short_link_matches(target, stems)
            # Obsidian prefers a match in the linking note's own folder;
            # otherwise the name has to be unique to resolve at all.
            same_folder = [m for m in matches if m.rsplit("/", 1)[0] == rel_dir]
            if same_folder or len(matches) == 1:
                findings.append(Finding(
                    "wikilink-not-canonical", rel,
                    f"[[{target}]] is a short link — Obsidian resolves it to "
                    f"{(same_folder or matches)[0]}; AGENTS.md asks for the "
                    "full vault-relative path",
                ))
            elif matches:
                findings.append(Finding(
                    "wikilink-ambiguous", rel,
                    f"[[{target}]] is a short link matching {len(matches)} notes "
                    f"({', '.join(matches)}) — write the full vault-relative path",
                ))
            else:
                dangling.append((rel, target))

    probe_paths = sorted({raw_by_stem.get(t, t) for _, t in dangling})
    ignored = _git_ignored_paths(paths.root, probe_paths)
    for rel, target in dangling:
        if raw_by_stem.get(target, target) in ignored:
            findings.append(Finding(
                "wikilink-source-local-only", rel,
                f"[[{target}]] absent and git-ignored, so this source is "
                f"expected to be missing on any other checkout: {target}",
            ))
        else:
            findings.append(Finding(
                "dangling-wikilink", rel,
                f"[[{target}]] resolves to no note or file",
            ))
    return findings


def _parse_day(raw: str) -> date | None:
    """Strict YYYY-MM-DD parse — anything else is None (and a finding)."""
    if not _DATE_RE.match(raw):
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _interval_label(r: Relation) -> str:
    return f"[{r.valid_from or 'open'}..{r.valid_until or 'open'}]"


def _check_relations(paths: VaultPaths) -> list[Finding]:
    """relation-problem / relation-dangling-target / relation-bad-date /
    relation-inverted-interval / relation-overlap, one note at a time.

    Skips the same assistant history ``entity_notes`` skips (F20): a note
    the graph will never read must not be linted as if it were a node."""
    findings: list[Finding] = []
    for sub in KNOWLEDGE_NOTE_DIRS:
        base = paths.knowledge / sub
        if not base.is_dir():
            continue
        for md in sorted(base.rglob("*.md")):
            if not md.is_file():
                continue
            rel_path = md.relative_to(paths.root).as_posix()
            if rel_path.startswith(_ASSISTANT_HISTORY_PREFIXES):
                continue
            text = _read_text(md)
            if text is None or not text.strip():
                continue
            frontmatter, _body = _split_frontmatter(text)
            relations, problems = parse_relations(frontmatter)
            for problem in problems:
                findings.append(Finding("relation-problem", rel_path, problem))

            # (rel, target) -> clean date spans, for the overlap check.
            # Entries with bad or inverted dates are excluded so one bad
            # entry doesn't cascade into spurious overlap findings.
            spans: dict[tuple[str, str], list[tuple[Relation, date | None, date | None]]] = defaultdict(list)
            for r in relations:
                if not (paths.root / note_path_for_node(r.target)).is_file():
                    findings.append(Finding(
                        "relation-dangling-target", rel_path,
                        f"{r.rel} -> {r.target}: target note missing "
                        f"({note_path_for_node(r.target)})",
                    ))
                clean = True
                parsed: dict[str, date | None] = {}
                for label, value in (
                    ("valid_from", r.valid_from),
                    ("valid_until", r.valid_until),
                ):
                    if not value:
                        parsed[label] = None
                        continue
                    day = _parse_day(value)
                    if day is None:
                        findings.append(Finding(
                            "relation-bad-date", rel_path,
                            f"{r.rel} -> {r.target}: {label} '{value}' is not YYYY-MM-DD",
                        ))
                        clean = False
                    parsed[label] = day
                start, end = parsed["valid_from"], parsed["valid_until"]
                if clean and start and end and end < start:
                    findings.append(Finding(
                        "relation-inverted-interval", rel_path,
                        f"{r.rel} -> {r.target}: valid_until {r.valid_until} "
                        f"precedes valid_from {r.valid_from}",
                    ))
                    clean = False
                if clean:
                    # Keep the RAW parsed bounds (None == open). Don't coalesce
                    # an open start to date.min here: the documented supersede
                    # flow (undated open -> close -> undated reopen) is all
                    # open-start entries, and treating them as starting at
                    # date.min made every such pair a spurious overlap.
                    spans[(r.rel, r.target)].append((r, start, end))

            for (rel_name, target), entries in sorted(spans.items()):
                for i in range(len(entries)):
                    for j in range(i + 1, len(entries)):
                        (ra, sa, ea), (rb, sb, eb) = entries[i], entries[j]
                        # Only flag overlaps PROVABLE from concrete start dates.
                        # An open (undated) start can't be located on the
                        # calendar, so overlap with it is unprovable — skip it
                        # rather than assume the worst (mirrors AGENTS.md's
                        # "valid_from optional" supersede contract).
                        if sa is None or sb is None:
                            continue
                        ea_eff = ea or date.max
                        eb_eff = eb or date.max
                        if max(sa, sb) <= min(ea_eff, eb_eff):
                            findings.append(Finding(
                                "relation-overlap", rel_path,
                                f"{rel_name} -> {target}: {_interval_label(ra)} "
                                f"overlaps {_interval_label(rb)}",
                            ))
    return findings


def _levenshtein(a: str, b: str) -> int:
    """Classic O(len(a)·len(b)) edit-distance DP, two rows of memory.
    Vault slugs are short; no dependency is worth importing for this."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                prev[j] + 1,                  # deletion
                cur[j - 1] + 1,               # insertion
                prev[j - 1] + (ca != cb),     # substitution
            ))
        prev = cur
    return prev[-1]


def _concept_distinct_pairs(paths: VaultPaths) -> set[frozenset[str]]:
    """Concept pairs a reader has already reviewed and marked as genuinely
    different, from ``distinct_from: [slug]`` in a concept note's
    frontmatter — an aliases-adjacent user key that ``concepts.py``
    preserves across regeneration. One side is enough."""
    pairs: set[frozenset[str]] = set()
    concepts_dir = paths.knowledge / "concepts"
    if not concepts_dir.is_dir():
        return pairs
    for md in sorted(concepts_dir.glob("*.md")):
        text = _read_text(md)
        if text is None:
            continue
        raw = _split_frontmatter(text)[0].get("distinct_from")
        values = raw if isinstance(raw, list) else [raw]
        for value in values:
            if isinstance(value, str) and slugify(value):
                pairs.add(frozenset((md.stem, slugify(value))))
    return pairs


def _differs_only_in_numbers(a: str, b: str) -> bool:
    """``android-16`` vs ``android-17``: the same words around different
    numbers. That is a version or edition pair — never one concept spelled
    twice — and the edit distance between them is always small."""
    if _DIGITS_RE.search(a) is None or _DIGITS_RE.search(b) is None:
        return False
    return _DIGITS_RE.sub("#", a) == _DIGITS_RE.sub("#", b)


def _check_fragmentation(
    records: list[IndexRecord], *, distinct: set[frozenset[str]]
) -> list[Finding]:
    """concept-fragmentation: two slugs that differ by a couple of edits
    (or a bare plural) AND co-occur on at least one document are almost
    certainly one concept spelled twice — suggest an aliases merge.

    Pairs marked ``distinct_from`` on either note, and pairs that differ
    only in a number, are left alone (F8: four of the five findings were
    distinct concepts, re-reported every night)."""
    slug_sources: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        for topic in rec.topics or []:
            slug = slugify(topic)
            if slug:
                slug_sources[slug].add(rec.relative_path)

    findings: list[Finding] = []
    slugs = sorted(slug_sources)
    for i, a in enumerate(slugs):
        for b in slugs[i + 1:]:
            if abs(len(a) - len(b)) > _FRAGMENT_MAX_DISTANCE:
                continue
            similar = (
                b == a + "s"
                or a == b + "s"
                or _levenshtein(a, b) <= _FRAGMENT_MAX_DISTANCE
            )
            if not similar:
                continue
            if frozenset((a, b)) in distinct or _differs_only_in_numbers(a, b):
                continue
            shared = slug_sources[a] & slug_sources[b]
            if not shared:
                continue
            findings.append(Finding(
                "concept-fragmentation", f"knowledge/concepts/{a}.md",
                f"'{a}' and '{b}' look like the same concept "
                f"({len(shared)} shared source(s)) — consider merging via aliases",
            ))
    return findings


@dataclass(frozen=True)
class _EntityNote:
    """The frontmatter of one entity note, reduced to what the duplicate
    and alias checks need."""

    rel_path: str
    node_id: str
    slug: str
    title: str
    aliases: tuple[str, ...]
    status: str
    superseded: bool
    related: frozenset[str]


def _entity_notes(paths: VaultPaths) -> list[_EntityNote]:
    """Every people/organisations/projects/chats note, frontmatter parsed once.
    Unreadable notes are skipped, as everywhere else in the sweep."""
    notes: list[_EntityNote] = []
    for sub in _ENTITY_DIRS:
        base = paths.knowledge / sub
        if not base.is_dir():
            continue
        for md in sorted(base.rglob("*.md")):
            if not md.is_file():
                continue
            text = _read_text(md)
            if text is None or not text.strip():
                continue
            frontmatter, _body = _split_frontmatter(text)
            rel_path = md.relative_to(paths.root).as_posix()
            raw_aliases = frontmatter.get("aliases")
            aliases = tuple(
                a for a in (raw_aliases if isinstance(raw_aliases, list) else [])
                if isinstance(a, str) and a.strip()
            )
            title = frontmatter.get("title")
            status = frontmatter.get("status")
            relations, _problems = parse_relations(frontmatter)
            notes.append(_EntityNote(
                rel_path=rel_path,
                node_id=md.relative_to(paths.knowledge).with_suffix("").as_posix(),
                slug=md.stem,
                title=title.strip() if isinstance(title, str) else "",
                aliases=aliases,
                status=status.strip() if isinstance(status, str) else "",
                superseded="superseded_by" in frontmatter,
                related=frozenset(r.target for r in relations if r.rel == "related_to"),
            ))
    return notes


def _already_merged(email_node: _EntityNote, named: _EntityNote) -> bool:
    """The vault's merge shape (AGENTS.md: supersede, never delete): the
    email node marked ``superseded_by``, a ``related_to`` edge either way,
    or the address kept as an alias on the named node."""
    return (
        email_node.superseded
        or named.node_id in email_node.related
        or email_node.node_id in named.related
        or email_node.title.casefold() in {a.casefold() for a in named.aliases}
    )


@dataclass(frozen=True)
class DuplicateEntityPair:
    """One email-slugged node beside the named node(s) for the same local
    part, not yet merged the vault's way. Shared by the ``entity-duplicate``
    sweep check and ``ingest_lib.duplicates``'s merge proposer, so both
    agree on exactly what counts as a duplicate and what counts as already
    merged (:func:`_already_merged`)."""

    email_node: _EntityNote
    named_candidates: tuple[_EntityNote, ...]


def find_duplicate_entity_pairs(paths: VaultPaths) -> list[DuplicateEntityPair]:
    """entity-duplicate detection: a person/organisation node whose whole
    title is an email address, sitting beside a named node for the same
    local part.

    `meeting_create` mints a node per attendee id it is handed, so a
    calendar payload carrying an address instead of a display name splits
    the person in two and each half then collects its own ``attended``
    history (F2). Only returns pairs not yet linked the vault's way.
    Deterministically ordered by the email node's path, then candidates by
    their own path."""
    entities = [
        n for n in _entity_notes(paths)
        if n.rel_path.startswith(("knowledge/people/", "knowledge/organisations/"))
    ]
    named = [n for n in entities if not _EMAIL_TITLE_RE.match(n.title)]
    pairs: list[DuplicateEntityPair] = []
    for node in sorted(entities, key=lambda n: n.rel_path):
        if not _EMAIL_TITLE_RE.match(node.title):
            continue
        local = slugify(node.title.split("@", 1)[0])
        if not local:
            continue
        candidates = tuple(sorted(
            (other for other in named
             if (other.slug == local or other.slug.startswith(f"{local}-"))
             and not _already_merged(node, other)),
            key=lambda n: n.rel_path,
        ))
        if candidates:
            pairs.append(DuplicateEntityPair(email_node=node, named_candidates=candidates))
    return pairs


def _check_entity_duplicates(paths: VaultPaths) -> list[Finding]:
    """entity-duplicate: a person/organisation node whose whole title is an
    email address, sitting beside a named node for the same local part.
    See :func:`find_duplicate_entity_pairs` for the detection itself."""
    findings: list[Finding] = []
    for pair in find_duplicate_entity_pairs(paths):
        node = pair.email_node
        candidates = [c.rel_path for c in pair.named_candidates]
        findings.append(Finding(
            "entity-duplicate", node.rel_path,
            f"title '{node.title}' is a bare email address and "
            f"{', '.join(candidates)} name(s) the same local part — merge "
            "them (address as an alias, related_to both ways, supersede "
            "the email node) or link them if they are different people",
        ))
    return findings


def _check_alias_collisions(paths: VaultPaths) -> list[Finding]:
    """alias-collision: one alias claimed by two entity notes that are both
    live, so every alias lookup for it is ambiguous (F9). A note that is
    archived, complete or superseded is out of the running by definition."""
    by_alias: dict[str, list[str]] = defaultdict(list)
    for node in _entity_notes(paths):
        if node.superseded or node.status.casefold() in _INACTIVE_STATUSES:
            continue
        for alias in sorted({a.strip().casefold() for a in node.aliases}):
            by_alias[alias].append(node.rel_path)
    return [
        Finding(
            "alias-collision", sorted(claimants)[0],
            f"alias '{alias}' is claimed by {len(claimants)} live notes "
            f"({', '.join(sorted(claimants))}) — leave it on the canonical "
            "note and mark the others archived or superseded",
        )
        for alias, claimants in sorted(by_alias.items())
        if len(claimants) > 1
    ]

