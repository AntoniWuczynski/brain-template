"""The one atomic-write implementation every vault writer uses.

AGENTS.md rule 4 (atomic writes) used to be hand-rolled at each call site:
seven copies of ``mkstemp`` -> write -> ``flush`` -> ``fsync`` ->
``os.replace``, cleaning up in ``except Exception``. ``except Exception``
does not catch ``KeyboardInterrupt`` or ``SystemExit``, so a Ctrl-C between
``mkstemp`` and ``os.replace`` left a ``.note-*.md`` / ``.index-*.jsonl``
behind in the vault, where Obsidian indexes it and ``git status`` shows it.
Cleanup here runs on ``BaseException`` instead.

Callers keep their own temp-file ``prefix``/``suffix`` — the ingest scanner
skips ``.tmp-*`` names and ``.gitignore`` lists the metadata prefixes by
name, so the naming is part of each caller's contract, not this module's.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def umask_mode() -> int:
    """The mode a normally-created file would get (0644 under the usual 022).

    ``mkstemp`` hardcodes 0600. Files a human reads and edits in the vault
    pass this so the rename doesn't leave them unreadable to anything but
    the owner.
    """
    current = os.umask(0o022)
    os.umask(current)
    return 0o666 & ~current


def atomic_write_bytes(
    path: Path, data: bytes, *, prefix: str, suffix: str, mode: int | None = None
) -> None:
    """Write ``data`` to ``path`` via a temp file in the same directory.

    ``mode`` is applied to the temp file before the rename; ``None`` keeps
    ``mkstemp``'s 0600. Any failure — exception or interrupt — unlinks the
    temp file and re-raises; ``path`` is either the old bytes or the new
    ones, never a truncated mix.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            if mode is not None:
                os.fchmod(fh.fileno(), mode)
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(
    path: Path, text: str, *, prefix: str, suffix: str, mode: int | None = None
) -> None:
    """UTF-8 ``atomic_write_bytes``."""
    atomic_write_bytes(path, text.encode("utf-8"), prefix=prefix, suffix=suffix, mode=mode)


def append_jsonl_line(path: Path, line: str, *, prefix: str, suffix: str = ".jsonl") -> None:
    """Append one already-newline-terminated ``line`` to a JSONL file.

    Creates the file atomically when it doesn't exist yet. When it does,
    appends with ``fsync`` after self-healing a torn tail: records routinely
    exceed the PIPE_BUF (4 KiB) single-write atomicity window, so a crash or
    concurrent run can leave a partial line. If the file doesn't end in
    ``\\n`` one is written first, so at most one record is lost to the torn
    tail instead of two silently merging.

    The last byte is probed in BINARY. These files are written with
    ``ensure_ascii=False``, and a text-mode read of a tail cut mid-UTF-8
    would raise ``UnicodeDecodeError`` before any write, crashing every
    retry.
    """
    if not path.exists():
        atomic_write_text(path, line, prefix=prefix, suffix=suffix)
        return
    with path.open("rb") as probe:
        probe.seek(0, os.SEEK_END)
        needs_nl = probe.tell() > 0
        if needs_nl:
            probe.seek(-1, os.SEEK_END)
            needs_nl = probe.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as fh:
        if needs_nl:
            fh.write("\n")
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())
