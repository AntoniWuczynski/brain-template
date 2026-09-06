"""Vault linter — a deterministic consistency sweep over the whole vault.

Pure-compute core: :func:`run_sweep` reads the vault and returns a
:class:`SweepReport`; :func:`render_report` turns one into a Markdown
note. Neither writes anything — the report file and the per-run log are
the CLI's job (``scripts/sweep.py``). A linter never raises over a
malformed file: every problem is a finding, every unreadable input a
graceful skip.

Finding categories (exact strings):

- ``archive-orphan-file``        raw file with no index.jsonl record
- ``archive-orphan-record``      record whose raw_path file is missing
- ``archive-source-local-only``  record whose raw_path file is missing
                                 *and* git-ignored: a source deliberately
                                 kept out of the repository, absent by
                                 design on any other checkout
- ``archive-corrupt``            raw file whose bytes no longer match its
                                 recorded source_hash (opt-in; re-hashes
                                 archive/raw)
- ``archive-processed-large``    processed tree past the git-vs-git-lfs
                                 decision size (a few hundred MB)
- ``missing-artifact``           record's processed/index note missing
- ``dangling-wikilink``          ``[[target]]`` that resolves to nothing
- ``wikilink-not-canonical``     short ``[[target]]`` Obsidian resolves by
                                 name, written without its vault path
- ``wikilink-ambiguous``         short ``[[target]]`` matching several notes
- ``dream-stalled``              the dream gate fired but no dream run
                                 completed within 2 days
- ``relation-problem``           parse_relations problems, verbatim
- ``relation-dangling-target``   relation target note missing on disk
- ``relation-bad-date``          valid_from/valid_until not YYYY-MM-DD
- ``relation-inverted-interval`` valid_until before valid_from
- ``relation-overlap``           same (rel, target) with overlapping spans
- ``concept-fragmentation``      near-identical slugs sharing a source
- ``entity-duplicate``           an email-titled entity node beside the
                                 named node for the same entity
- ``alias-collision``            one alias claimed by two live entity notes
- ``index-drift-stale``          embeddings row hash != current content
- ``index-drift-missing``        embeddings row whose backing is gone
- ``index-drift-unindexed``      source/note with zero embeddings rows
- ``stale-unconsolidated``       old unconsolidated assistant memory
- ``memory-unconsolidatable``    assistant memory neither consolidate nor
                                 the staleness check can ever act on
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from .concepts import slugify
from .config import VaultPaths
from .dream import load_pending_since
from .hashing import sha256_of
from .knowledge import KNOWLEDGE_EXTRACTOR, KNOWLEDGE_NOTE_DIRS, scan_knowledge
from .metadata import IndexRecord, latest_records_by_path
from .notes import _split_frontmatter  # private helper, but module-internal
from .relations import (
    _ASSISTANT_HISTORY_PREFIXES,  # private, but the graph and the sweep must agree
    Relation,
    note_path_for_node,
    parse_relations,
)

_WIKILINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DIGITS_RE = re.compile(r"\d+")
# A title that is nothing but an email address: the calendar payload carried
# no display name, so the MCP write path slugged the address into a node id.
_EMAIL_TITLE_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
# An entity note whose project/lifecycle status says it is no longer the live
# node — its aliases may overlap a live note's without ambiguity.
_INACTIVE_STATUSES = frozenset({"archived", "complete", "completed", "superseded"})
_ENTITY_DIRS: tuple[str, ...] = ("people", "organisations", "projects")

# Edit-distance ceiling for two concept slugs to count as "the same
# concept spelled twice". 2 covers a typo or a joined/hyphenated variant
# without pulling genuinely different short slugs together too often.
_FRAGMENT_MAX_DISTANCE = 2


@dataclass(frozen=True)
class Finding:
    """One lint finding. ``path`` is vault-relative; ``detail`` is a
    human-readable, deterministic one-liner."""

    category: str
    path: str
    detail: str


@dataclass(frozen=True)
class SweepReport:
    findings: tuple[Finding, ...]

    @property
    def counts(self) -> dict[str, int]:
        """Per-category finding counts, sorted by category."""
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.category] = out.get(f.category, 0) + 1
        return dict(sorted(out.items()))


def run_sweep(
    paths: VaultPaths,
    *,
    logger: logging.Logger,
    as_of: date,
    stale_days: int = 30,
    check_integrity: bool = False,
) -> SweepReport:
    """Run every check and return the findings, deterministically ordered
    by ``(category, path, detail)``. ``as_of`` (not the wall clock) anchors
    the staleness check so the same vault state always sweeps the same.

    ``check_integrity`` additionally re-hashes every ``archive/raw`` file
    against its recorded ``source_hash`` (``archive-corrupt``) — off by
    default because it reads the whole immutable archive (GBs)."""
    latest = latest_records_by_path(paths.metadata_index_jsonl)
    # Virtual knowledge records: topics for fragmentation, paths for the
    # unindexed check — exactly the record set the enrichment pipeline sees.
    knowledge_recs = scan_knowledge(paths, logger=logger).records

    findings: list[Finding] = []
    findings += _check_archive(paths, latest, check_integrity=check_integrity)
    findings += _check_archive_processed_size(paths)
    findings += _check_wikilinks(paths)
    findings += _check_relations(paths)
    findings += _check_fragmentation(
        list(latest.values()) + knowledge_recs,
        distinct=_concept_distinct_pairs(paths),
    )
    findings += _check_entity_duplicates(paths)
    findings += _check_alias_collisions(paths)
    findings += _check_index_drift(paths, latest, knowledge_recs)
    findings += _check_stale_memory(paths, as_of=as_of, stale_days=stale_days)
    findings += _check_inbox_unprocessable(paths)
    findings += _check_dream_stalled(paths, as_of=as_of)

    findings.sort(key=lambda f: (f.category, f.path, f.detail))
    report = SweepReport(findings=tuple(findings))
    logger.info(
        "sweep: %d finding(s) in %d category(ies) (as_of=%s, stale_days=%d)",
        len(report.findings), len(report.counts), as_of.isoformat(), stale_days,
    )
    return report


def render_report(report: SweepReport, *, as_of: date) -> str:
    """Markdown report note. Fully deterministic given (report, as_of):
    the only timestamp is ``updated:`` in frontmatter, derived from
    ``as_of`` — never from the wall clock."""
    counts = report.counts
    lines: list[str] = [
        "---",
        "title: Vault sweep report",
        "type: report",
        f"updated: '{as_of.isoformat()}T00:00:00Z'",
    ]
    if counts:
        lines.append("counts:")
        lines.extend(f"  {category}: {n}" for category, n in counts.items())
    else:
        lines.append("counts: {}")
    lines += ["---", "", "# Vault sweep report", ""]

    if not report.findings:
        lines += ["_(no findings)_", ""]
        return "\n".join(lines)

    by_category: dict[str, list[Finding]] = defaultdict(list)
    for f in report.findings:  # already sorted: groups stay sorted too
        by_category[f.category].append(f)
    for category in sorted(by_category):
        group = by_category[category]
        lines.append(f"## {category} ({len(group)})")
        lines.append("")
        lines.extend(f"- `{f.path}` — {f.detail}" for f in group)
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> str | None:
    """Tolerant read: a vanished or unreadable file is a skip, not a crash."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


# ``git check-ignore`` answers "is this path excluded by the repository's
# ignore rules?" from .gitignore itself, so no second list of expected-absent
# sources has to be maintained alongside it and the two can never drift.
_GIT_CHECK_IGNORE_TIMEOUT_S = 30


def _git_ignored_paths(root: Path, rels: Sequence[str]) -> frozenset[str]:
    """The subset of ``rels`` (root-relative POSIX paths) that git reports
    as excluded by the vault's ignore rules.

    One batched ``git check-ignore`` call, NUL-delimited in both directions
    so spaces and other odd characters in slide filenames survive. Tracked
    paths are never reported ignored (git consults the index first), so a
    committed file that vanished stays a real finding.

    Returns an empty set whenever the answer cannot be obtained — git not
    installed, the vault not a repository, the call timing out. The honest
    default is to keep reporting an absent file as missing rather than to
    excuse it on a guess.
    """
    if not rels:
        return frozenset()
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "--stdin", "-z"],
            input="\0".join(rels),
            capture_output=True,
            text=True,
            timeout=_GIT_CHECK_IGNORE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    # 0 = at least one path ignored, 1 = none ignored. Anything else (128 =
    # not a repository) means the output carries no answer.
    if proc.returncode not in (0, 1):
        return frozenset()
    return frozenset(line for line in proc.stdout.split("\0") if line)


def _check_archive(
    paths: VaultPaths, latest: dict[str, IndexRecord], *, check_integrity: bool = False
) -> list[Finding]:
    """archive-orphan-file / archive-orphan-record /
    archive-source-local-only / missing-artifact, and (when
    ``check_integrity``) archive-corrupt."""
    findings: list[Finding] = []

    known_raw = {rec.raw_path for rec in latest.values() if rec.raw_path}
    if paths.archive_raw.is_dir():
        for f in sorted(paths.archive_raw.rglob("*")):
            if not f.is_file():
                continue
            # Dotfiles (.gitkeep, .DS_Store, AppleDouble ._*) are
            # scaffolding, not sources.
            if f.name.startswith("."):
                continue
            rel = f.relative_to(paths.root).as_posix()
            if rel not in known_raw:
                findings.append(Finding(
                    "archive-orphan-file", rel,
                    "no index.jsonl record references this raw file",
                ))

    # An absent source is either a real loss or a file .gitignore deliberately
    # keeps out of the repository (oversized slides kept local only). Ask git
    # once, for every absent path at once, and split the two apart.
    missing_raw = {
        rec.raw_path for rec in latest.values()
        if rec.raw_path and not (paths.root / rec.raw_path).is_file()
    }
    ignored_raw = _git_ignored_paths(paths.root, sorted(missing_raw))

    for rel, rec in sorted(latest.items()):
        raw_file = (paths.root / rec.raw_path) if rec.raw_path else None
        if rec.raw_path and rec.raw_path in missing_raw:
            if rec.raw_path in ignored_raw:
                findings.append(Finding(
                    "archive-source-local-only", rel,
                    f"raw_path absent and git-ignored, so this source is "
                    f"expected to be missing on any other checkout: "
                    f"{rec.raw_path}",
                ))
            else:
                findings.append(Finding(
                    "archive-orphan-record", rel,
                    f"raw_path missing on disk: {rec.raw_path}",
                ))
        elif check_integrity and raw_file is not None and rec.source_hash:
            # Bit-rot / accidental edit / immutability violation: the raw
            # bytes no longer hash to what was recorded at ingest time. Skip
            # unreadable files (an OSError is not a corruption finding).
            try:
                current = sha256_of(raw_file)
            except OSError:
                current = ""
            if current and current != rec.source_hash:
                findings.append(Finding(
                    "archive-corrupt", rel,
                    f"raw file hash changed since ingest "
                    f"(recorded {rec.source_hash[:12]}, now {current[:12]}) "
                    "— archive/raw must be immutable",
                ))
        for label, artifact in (
            ("processed_path", rec.processed_path),
            ("index_note_path", rec.index_note_path),
        ):
            if artifact and not (paths.root / artifact).is_file():
                findings.append(Finding(
                    "missing-artifact", rel,
                    f"{label} missing on disk: {artifact}",
                ))
    return findings


# Tripwire for the "keep archive/processed under git vs git-lfs" decision.
# Settled 2026-09-04 (FOUNDER_DECISIONS.md IMP-010): keep it in plain git,
# because re-ingesting the tree costs MinerU weights and hours of compute
# while the pack cost is bounded. Raised from 300 MB to 2 GB so the tripwire
# flags a genuinely new problem rather than the decision already taken.
_PROCESSED_SIZE_THRESHOLD_BYTES = 2 * 1024 * 1024 * 1024


def _check_archive_processed_size(
    paths: VaultPaths, *, threshold_bytes: int | None = None
) -> list[Finding]:
    """archive-processed-large: flag when archive/processed grows past the
    git-vs-git-lfs decision point. Cheap: sums file sizes, reads nothing.

    The threshold is read at call time (not bound as a default) so the module
    constant can be monkeypatched in tests."""
    if threshold_bytes is None:
        threshold_bytes = _PROCESSED_SIZE_THRESHOLD_BYTES
    if not paths.archive_processed.is_dir():
        return []
    total = 0
    for f in paths.archive_processed.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    if total < threshold_bytes:
        return []
    mb = total / (1024 * 1024)
    return [Finding(
        "archive-processed-large", "archive/processed",
        f"regenerable processed tree is {mb:.0f} MB (>= {threshold_bytes // (1024*1024)} MB) "
        "— decide git-lfs vs dropping it from git (it's regenerable)",
    )]


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


def _check_wikilinks(paths: VaultPaths) -> list[Finding]:
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
    """Every people/organisations/projects note, frontmatter parsed once.
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


def _check_index_drift(
    paths: VaultPaths,
    latest: dict[str, IndexRecord],
    knowledge_recs: list[IndexRecord],
) -> list[Finding]:
    """index-drift-stale / index-drift-missing / index-drift-unindexed.

    Knowledge-note rows (origin ``knowledge-note``) embed the note itself,
    so staleness compares against the note's current hash; archive rows
    compare against the latest index.jsonl record. No embeddings index on
    disk (fresh clone) -> all three checks are skipped gracefully.
    """
    meta_path = paths.metadata / "embeddings_meta.jsonl"
    if not meta_path.is_file():
        return []
    meta_text = _read_text(meta_path)
    if meta_text is None:
        return []

    # All rows of one source share its hash/origin — the first one speaks
    # for the source. Malformed lines are skipped, never fatal.
    first_by_source: dict[str, dict[str, object]] = {}
    for line in meta_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        src = row.get("source_relative_path")
        if isinstance(src, str) and src and src not in first_by_source:
            first_by_source[src] = row

    findings: list[Finding] = []
    for src in sorted(first_by_source):
        row = first_by_source[src]
        indexed_hash = str(row.get("source_hash") or "")
        if str(row.get("origin") or "") == KNOWLEDGE_EXTRACTOR:
            note = paths.root / src
            if not note.is_file():
                findings.append(Finding(
                    "index-drift-missing", src,
                    "indexed knowledge note no longer exists on disk",
                ))
            else:
                # A note that vanishes/loses read permission between is_file()
                # and the hash must not crash the whole linter (its contract is
                # 'never raises'); every other read here goes through _read_text.
                try:
                    current_hash = sha256_of(note)
                except OSError:
                    findings.append(Finding(
                        "index-drift-missing", src,
                        "indexed knowledge note is unreadable",
                    ))
                    continue
                if current_hash != indexed_hash:
                    findings.append(Finding(
                        "index-drift-stale", src,
                        "note content changed since indexing — rebuild the search index",
                    ))
        else:
            rec = latest.get(src)
            if rec is None:
                findings.append(Finding(
                    "index-drift-missing", src,
                    "no index.jsonl record for this indexed source",
                ))
            elif rec.source_hash != indexed_hash:
                findings.append(Finding(
                    "index-drift-stale", src,
                    "source_hash changed since indexing — rebuild the search index",
                ))

    indexed = set(first_by_source)
    for rel, rec in sorted(latest.items()):
        # Only sources the index build would actually embed (processed,
        # with a processed_path) can meaningfully be "unindexed".
        if rec.status == "processed" and rec.processed_path and rel not in indexed:
            findings.append(Finding(
                "index-drift-unindexed", rel,
                "processed record has no rows in the embeddings index",
            ))
    for rec in knowledge_recs:
        if rec.relative_path not in indexed:
            findings.append(Finding(
                "index-drift-unindexed", rec.relative_path,
                "knowledge note has no rows in the embeddings index",
            ))
    return findings


def _coerce_day(raw: object) -> date | None:
    """Frontmatter ``created:`` -> date. Quoted strings, bare YAML dates,
    and full ISO timestamps all collapse to the calendar day; anything
    unparseable is None (a note without a clock can't be aged)."""
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if isinstance(raw, str) and len(raw.strip()) >= 10:
        return _parse_day(raw.strip()[:10])
    return None


def _check_stale_memory(
    paths: VaultPaths, *, as_of: date, stale_days: int
) -> list[Finding]:
    """stale-unconsolidated: assistant memory left unconsolidated longer
    than ``stale_days`` should be promoted or digested, not forgotten —
    plus memory-unconsolidatable for the notes whose age is unknowable."""
    findings: list[Finding] = []
    base = paths.knowledge / "assistant"
    if not base.is_dir():
        return findings
    # F8: archive/ and digests/ are consolidate's HANDLED history, not a
    # backlog. Digested notes keep ``memory_status: unconsolidated`` after
    # being moved into archive/, so without this skip the sweep would flag
    # them stale forever and contradict consolidate. Only inbox/ holds the
    # actionable backlog.
    archive_dir = base / "archive"
    digests_dir = base / "digests"
    for md in sorted(base.rglob("*.md")):
        if not md.is_file():
            continue
        if archive_dir in md.parents or digests_dir in md.parents:
            continue
        text = _read_text(md)
        if text is None or not text.strip():
            continue
        frontmatter, _body = _split_frontmatter(text)
        raw_status = frontmatter.get("memory_status")
        status = raw_status.strip() if isinstance(raw_status, str) else ""
        if status != "unconsolidated":
            continue
        created = _coerce_day(frontmatter.get("created"))
        if created is None:
            # F13: no clock, so this note can never age out — and if it also
            # sits outside consolidate's flat inbox, nothing else will ever
            # look at it either. Report it instead of skipping in silence.
            findings.append(Finding(
                "memory-unconsolidatable",
                md.relative_to(paths.root).as_posix(),
                "memory_status: unconsolidated but created: is missing or "
                "unparseable — the staleness check can never fire",
            ))
            continue
        age = (as_of - created).days
        if age > stale_days:
            findings.append(Finding(
                "stale-unconsolidated",
                md.relative_to(paths.root).as_posix(),
                f"unconsolidated for {age} day(s) "
                f"(created {created.isoformat()}, threshold {stale_days})",
            ))
    return findings


def _check_inbox_unprocessable(paths: VaultPaths) -> list[Finding]:
    """memory-unconsolidatable: a note in consolidate's inbox that
    consolidate itself classifies as "not a memory fact note" (no
    ``promote`` mapping and no ``memory_status``) — it is skipped on every
    run, never promoted, never digested, and never stale (F13)."""
    findings: list[Finding] = []
    inbox = paths.knowledge / "assistant" / "inbox"
    if not inbox.is_dir():
        return findings
    for md in sorted(inbox.glob("*.md")):  # flat, exactly like consolidate
        if not md.is_file():
            continue
        text = _read_text(md)
        if text is None or not text.strip():
            continue
        frontmatter, _body = _split_frontmatter(text)
        raw_status = frontmatter.get("memory_status")
        status = raw_status.strip() if isinstance(raw_status, str) else ""
        if isinstance(frontmatter.get("promote"), dict) or status:
            continue
        findings.append(Finding(
            "memory-unconsolidatable",
            md.relative_to(paths.root).as_posix(),
            "not a memory fact note (no promote mapping, no memory_status) — "
            "consolidate skips it on every run",
        ))
    return findings


def _check_dream_stalled(paths: VaultPaths, *, as_of: date) -> list[Finding]:
    """The dream gate records ``metadata/dream.pending`` when it fires; a
    completed run clears it via ``dream_gate.py --mark-done``. A marker
    two or more days old means dream runs keep failing or never launch."""
    since = load_pending_since(paths)
    if since is None:
        return []
    age_days = (as_of - since.date()).days
    if age_days < 2:
        return []
    return [
        Finding(
            "dream-stalled",
            "metadata/dream.pending",
            f"dream gate fired {age_days} days ago but no run completed "
            "(dream_gate.py --mark-done never ran) — check logs/dream-*.log",
        )
    ]


__all__ = ["Finding", "SweepReport", "render_report", "run_sweep"]
