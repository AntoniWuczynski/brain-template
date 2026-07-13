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
- ``archive-corrupt``            raw file whose bytes no longer match its
                                 recorded source_hash (opt-in; re-hashes
                                 archive/raw)
- ``archive-processed-large``    processed tree past the git-vs-git-lfs
                                 decision size (a few hundred MB)
- ``missing-artifact``           record's processed/index note missing
- ``dangling-wikilink``          ``[[target]]`` that resolves to nothing
- ``relation-problem``           parse_relations problems, verbatim
- ``relation-dangling-target``   relation target note missing on disk
- ``relation-bad-date``          valid_from/valid_until not YYYY-MM-DD
- ``relation-inverted-interval`` valid_until before valid_from
- ``relation-overlap``           same (rel, target) with overlapping spans
- ``concept-fragmentation``      near-identical slugs sharing a source
- ``index-drift-stale``          embeddings row hash != current content
- ``index-drift-missing``        embeddings row whose backing is gone
- ``index-drift-unindexed``      source/note with zero embeddings rows
- ``stale-unconsolidated``       old unconsolidated assistant memory
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from .concepts import slugify
from .config import VaultPaths
from .hashing import sha256_of
from .knowledge import KNOWLEDGE_EXTRACTOR, KNOWLEDGE_NOTE_DIRS, scan_knowledge
from .metadata import IndexRecord, latest_records_by_path
from .notes import _split_frontmatter  # private helper, but module-internal
from .relations import Relation, note_path_for_node, parse_relations

_WIKILINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

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
    findings += _check_fragmentation(list(latest.values()) + knowledge_recs)
    findings += _check_index_drift(paths, latest, knowledge_recs)
    findings += _check_stale_memory(paths, as_of=as_of, stale_days=stale_days)

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


def _check_archive(
    paths: VaultPaths, latest: dict[str, IndexRecord], *, check_integrity: bool = False
) -> list[Finding]:
    """archive-orphan-file / archive-orphan-record / missing-artifact, and
    (when ``check_integrity``) archive-corrupt."""
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

    for rel, rec in sorted(latest.items()):
        raw_file = (paths.root / rec.raw_path) if rec.raw_path else None
        if rec.raw_path and (raw_file is None or not raw_file.is_file()):
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


# Tripwire for the "keep archive/processed under git vs git-lfs" decision
# (TODO.md): once the regenerable processed tree passes a few hundred MB it
# starts to bloat the git pack, at which point git-lfs (or dropping it from
# git, since it is regenerable) is worth deciding.
_PROCESSED_SIZE_THRESHOLD_BYTES = 300 * 1024 * 1024


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


def _check_wikilinks(paths: VaultPaths) -> list[Finding]:
    """dangling-wikilink: a [[target]] in a knowledge/ note that resolves
    to none of ``<vault>/<target>.md``, ``<vault>/<target>`` (embeds of
    existing assets), or ``<vault>/<target>.*`` (extension-stripped links
    to binary sources). External http(s) links are skipped."""
    findings: list[Finding] = []
    if not paths.knowledge.is_dir():
        return findings
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
        seen: set[str] = set()  # one finding per distinct target per note
        for raw in _WIKILINK_RE.findall(text):
            target = raw.strip()
            if not target or target.startswith(("http://", "https://")):
                continue
            target = target.split("|", 1)[0].split("#", 1)[0].strip()
            if not target or target in seen:
                continue
            seen.add(target)
            if (paths.root / f"{target}.md").is_file() or (paths.root / target).exists():
                continue
            if _has_stem_sibling(paths.root / target):
                continue
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
    relation-inverted-interval / relation-overlap, one note at a time."""
    findings: list[Finding] = []
    for sub in KNOWLEDGE_NOTE_DIRS:
        base = paths.knowledge / sub
        if not base.is_dir():
            continue
        for md in sorted(base.rglob("*.md")):
            if not md.is_file():
                continue
            text = _read_text(md)
            if text is None or not text.strip():
                continue
            rel_path = md.relative_to(paths.root).as_posix()
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


def _check_fragmentation(records: list[IndexRecord]) -> list[Finding]:
    """concept-fragmentation: two slugs that differ by a couple of edits
    (or a bare plural) AND co-occur on at least one document are almost
    certainly one concept spelled twice — suggest an aliases merge."""
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
            shared = slug_sources[a] & slug_sources[b]
            if not shared:
                continue
            findings.append(Finding(
                "concept-fragmentation", f"knowledge/concepts/{a}.md",
                f"'{a}' and '{b}' look like the same concept "
                f"({len(shared)} shared source(s)) — consider merging via aliases",
            ))
    return findings


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
    than ``stale_days`` should be promoted or digested, not forgotten."""
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


__all__ = ["Finding", "SweepReport", "render_report", "run_sweep"]
