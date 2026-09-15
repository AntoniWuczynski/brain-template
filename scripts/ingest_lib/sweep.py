"""Vault linter — a deterministic consistency sweep over the whole vault.

Pure-compute core: :func:`run_sweep` reads the vault and returns a
:class:`SweepReport`; :func:`render_report` turns one into a Markdown
note. Neither writes anything — the report file and the per-run log are
the CLI's job (``scripts/sweep.py``). A linter never raises over a
malformed file: every problem is a finding, every unreadable input a
graceful skip.

The individual checks live in sibling ``sweep_checks_*`` modules (split
out of this one, AUD-121) grouped by what they check —
``sweep_checks_archive`` (archive/raw and archive/processed),
``sweep_checks_graph`` (wikilinks, relations, concepts, entities) and
``sweep_checks_index`` (the embeddings index, assistant memory, the dream
gate). This module re-exports every name it used to define so nothing
importing from ``ingest_lib.sweep`` has to change. One check —
``_check_archive_processed_size`` and its threshold constant — stays
defined here rather than in a sibling: a test exercises it by
monkeypatching ``ingest_lib.sweep._PROCESSED_SIZE_THRESHOLD_BYTES`` and
then calling :func:`run_sweep`, which only works if the constant a
sibling's function would read and the constant the test patches are the
same module-global binding; a sibling cannot import this module back
(that would be a cycle), so the check and its constant have to live
wherever :func:`run_sweep` calls it from.

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
- ``wikilink-source-local-only`` ``[[target]]`` under ``archive/raw/`` that
                                 resolves to nothing *and* is git-ignored: a
                                 source deliberately kept out of the
                                 repository, absent by design on any other
                                 checkout
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

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from .config import VaultPaths
from .knowledge import scan_knowledge
from .metadata import latest_records_by_path
from .sweep_checks_archive import _check_archive
from .sweep_checks_graph import (
    _check_alias_collisions,
    _check_entity_duplicates,
    _check_fragmentation,
    _check_relations,
    _check_wikilinks,
    _concept_distinct_pairs,
    _differs_only_in_numbers as _differs_only_in_numbers,
    _EMAIL_TITLE_RE as _EMAIL_TITLE_RE,
    _EntityNote as _EntityNote,
    _entity_notes as _entity_notes,
    _ENTITY_DIRS as _ENTITY_DIRS,
    _FRAGMENT_MAX_DISTANCE as _FRAGMENT_MAX_DISTANCE,
    _already_merged as _already_merged,
    _DATE_RE as _DATE_RE,
    _DIGITS_RE as _DIGITS_RE,
    _has_stem_sibling as _has_stem_sibling,
    _INACTIVE_STATUSES as _INACTIVE_STATUSES,
    _interval_label as _interval_label,
    _levenshtein as _levenshtein,
    _note_stem_index as _note_stem_index,
    _parse_day as _parse_day,
    _short_link_matches as _short_link_matches,
    _WIKILINK_RE as _WIKILINK_RE,
    DuplicateEntityPair as DuplicateEntityPair,
    find_duplicate_entity_pairs as find_duplicate_entity_pairs,
)
from .sweep_checks_index import (
    _check_dream_stalled,
    _check_index_drift,
    _check_inbox_unprocessable,
    _check_stale_memory,
    _coerce_day as _coerce_day,
)
from .sweep_common import (
    Finding,
    _GIT_CHECK_IGNORE_TIMEOUT_S as _GIT_CHECK_IGNORE_TIMEOUT_S,
    _git_ignored_paths as _git_ignored_paths,
    _read_text as _read_text,
)

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
    findings += _check_wikilinks(paths, latest)
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


__all__ = ["Finding", "SweepReport", "render_report", "run_sweep"]
