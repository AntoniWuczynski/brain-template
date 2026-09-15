"""Legacy ``.ppt`` extractor.

Strategy: convert the binary PowerPoint 97-2003 deck to ``.pptx`` with a
headless LibreOffice (``soffice``) subprocess, into a throwaway temp
directory, then hand the converted file to the existing ``.pptx``
extractor so the resulting note has the identical shape to a native
``.pptx`` deck.

``archive/raw`` is never written: the conversion output lands only in the
temp directory, which is always removed. When ``soffice`` is missing, or
the conversion fails or times out, the result is an honest
``manual_review`` carrying the real error — never a summary guessed from
the filename (AGENTS.md rule 6: honest extraction).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from . import pptx as _pptx_mod
from .base import ExtractionResult

# Format conversion only (no ML models involved), unlike MinerU's PDF
# pipeline — a generous few minutes is plenty even for a large deck.
_SOFFICE_TIMEOUT_S = 5 * 60
_VERSION_TIMEOUT_S = 30


def extract(src: Path, assets_dir: Path) -> ExtractionResult:
    if not _soffice_on_path():
        return ExtractionResult(
            status="manual_review",
            extractor="ppt-soffice",
            markdown="",
            error=(
                "soffice (LibreOffice) not on PATH; install LibreOffice to "
                "convert legacy .ppt files (brew install --cask libreoffice)"
            ),
        )

    tmp_root = Path(tempfile.mkdtemp(prefix="ppt-convert-"))
    try:
        return _convert_and_extract(src, assets_dir, tmp_root)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _convert_and_extract(src: Path, assets_dir: Path, tmp_root: Path) -> ExtractionResult:
    # Isolate LibreOffice's user profile per run: a shared/default profile
    # can deadlock or refuse to start a second headless instance while
    # another soffice process (or a leftover lock) is around.
    profile_dir = tmp_root / "lo_profile"
    cmd = [
        "soffice",
        f"-env:UserInstallation=file://{profile_dir}",
        "--headless",
        "--convert-to", "pptx",
        "--outdir", str(tmp_root),
        str(src),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_SOFFICE_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ExtractionResult(
            status="manual_review",
            extractor="ppt-soffice",
            markdown="",
            error=f"soffice timed out after {_SOFFICE_TIMEOUT_S}s converting to pptx: {exc}",
        )
    except OSError as exc:
        return ExtractionResult(
            status="manual_review",
            extractor="ppt-soffice",
            markdown="",
            error=f"soffice invocation failed: {exc!r}",
        )

    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip().splitlines()[-12:]
        return ExtractionResult(
            status="manual_review",
            extractor="ppt-soffice",
            markdown="",
            error=f"soffice exit={proc.returncode}: {' | '.join(stderr_tail)}",
        )

    converted = tmp_root / f"{src.stem}.pptx"
    if not converted.is_file():
        stdout_tail = (proc.stdout or "").strip().splitlines()[-12:]
        return ExtractionResult(
            status="manual_review",
            extractor="ppt-soffice",
            markdown="",
            error=(
                f"soffice reported success but produced no {converted.name} "
                f"under {tmp_root}: {' | '.join(stdout_tail)}"
            ),
        )

    inner = _pptx_mod.extract(converted, assets_dir)
    conversion_note = (
        f"converted from legacy .ppt to .pptx via LibreOffice "
        f"({_soffice_version()}) before extraction"
    )
    return ExtractionResult(
        status=inner.status,
        extractor="ppt-soffice+pptx",
        markdown=inner.markdown,
        assets=inner.assets,
        error=inner.error,
        notes=[conversion_note, *inner.notes],
    )


def _soffice_on_path() -> bool:
    return shutil.which("soffice") is not None


def _soffice_version() -> str:
    """Best-effort LibreOffice release string for the processing notes.

    ``soffice --version`` prints e.g. ``LibreOffice 25.8.1.1
    54047653041915e595ad4e45cccea684809c77b5`` — that trailing token is a
    per-build git hash, machine-specific and not the same across two
    installs of the identical release. AGENTS.md rule 7 requires the same
    input to produce the same output, so only the product name and release
    number (the first two whitespace-separated tokens) go into the note
    body; the full string carries no extra information a reader needs.
    """
    try:
        proc = subprocess.run(
            ["soffice", "--version"],
            capture_output=True,
            text=True,
            timeout=_VERSION_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "version unknown"
    lines = (proc.stdout or "").strip().splitlines()
    if not lines:
        return "version unknown"
    tokens = lines[0].split()
    return " ".join(tokens[:2]) if len(tokens) >= 2 else lines[0]
