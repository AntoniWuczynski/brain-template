"""Sweep checks over ``archive/raw`` and ``archive/processed`` — split out
of ``sweep.py`` (AUD-121). Pure compute, same "never raises, always a
finding" contract as the rest of the sweep.

Finding categories produced here: ``archive-orphan-file``,
``archive-orphan-record``, ``archive-source-local-only``,
``archive-corrupt``, ``missing-artifact``."""
from __future__ import annotations

from .config import VaultPaths
from .hashing import sha256_of
from .metadata import IndexRecord
from .sweep_common import Finding, _git_ignored_paths


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
