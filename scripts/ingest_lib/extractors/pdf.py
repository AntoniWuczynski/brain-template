"""PDF extractor.

Strategy:
1. Try MinerU via its CLI (``mineru``). MinerU auto-downloads its model
   weights from Hugging Face on first use (one-off ~14 GB) and produces
   a Markdown file plus a directory of extracted images per PDF.
2. If the ``mineru`` CLI is not on PATH, or it errors on a particular
   file, fall back to ``pypdf`` text-only extraction and mark the result
   ``status="partial"``.

The fallback exists so the system still functions on a fresh clone
(no models yet) and so a single broken PDF doesn't block a batch.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .base import ExtractionResult


_MINERU_TIMEOUT_S = 30 * 60   # generous: full extraction can be slow on CPU
_MINERU_BACKEND = "pipeline"  # most general; works with --lang for OCR
_MINERU_METHOD = "auto"       # let MinerU pick text vs OCR


def _mineru_lang() -> str:
    """OCR language for MinerU. Defaults to English (the vault's material);
    override with BRAIN_MINERU_LANG for other-language scans. Resolved at
    call time, not import, so the env var actually takes effect."""
    return os.environ.get("BRAIN_MINERU_LANG", "en")


def extract(src: Path, assets_dir: Path) -> ExtractionResult:
    # Handwritten / scanned material: route to the vision-LLM extractor.
    # MinerU's OCR is printed-text only and its formula model fabricates
    # LaTeX on handwriting, so set BRAIN_PDF_EXTRACTOR=vlm for such modules.
    if (os.environ.get("BRAIN_PDF_EXTRACTOR") or "").lower() == "vlm":
        from . import vlm as _vlm_mod
        return _vlm_mod.extract(src, assets_dir)
    if _mineru_on_path():
        result = _extract_with_mineru(src, assets_dir)
        if result.status != "manual_review":
            return result
        # MinerU failed: fall back to pypdf and note the failure.
        fallback = _extract_with_pypdf(src)
        # When the fallback ALSO fails, keep both errors — otherwise the
        # pypdf failure reason (the actual reason nothing was produced) is
        # lost and the failure is undiagnosable from the log / index.jsonl.
        if fallback.status == "manual_review":
            error = f"mineru: {result.error}; pypdf: {fallback.error}"
        else:
            error = result.error
        return ExtractionResult(
            status=fallback.status if fallback.status == "manual_review" else "partial",
            extractor="pdf-pypdf-fallback",
            markdown=fallback.markdown,
            assets=[],
            error=error,
            notes=fallback.notes
            + [f"MinerU failed; fell back to pypdf. mineru-error: {result.error}"],
        )
    return _extract_with_pypdf(src)


def _mineru_on_path() -> bool:
    return shutil.which("mineru") is not None


# --------------------------------------------------------------------------
# MinerU implementation (CLI-based)
# --------------------------------------------------------------------------

def _extract_with_mineru(src: Path, assets_dir: Path) -> ExtractionResult:
    """Run ``mineru -p <src> -o <tmp>`` and gather its outputs.

    Layout MinerU produces (as of mineru 3.x with the ``pipeline`` backend):

        <tmp>/<pdf-stem>/auto/<pdf-stem>.md
        <tmp>/<pdf-stem>/auto/images/*.{jpg,png}

    Older versions used ``<backend>`` instead of ``auto``. We're tolerant.
    """
    tmp_root = Path(tempfile.mkdtemp(prefix="mineru-out-"))
    lang = _mineru_lang()
    cmd = [
        "mineru",
        "-p", str(src),
        "-o", str(tmp_root),
        "-l", lang,
        "-b", _MINERU_BACKEND,
        "-m", _MINERU_METHOD,
    ]
    # The UniMerNet formula model hallucinates dense fake LaTeX on
    # handwriting; BRAIN_MINERU_FORMULA=false disables it (keeps text +
    # figures). Default keeps formula parsing on for printed math.
    formula_on = (os.environ.get("BRAIN_MINERU_FORMULA") or "true").lower() \
        not in ("0", "false", "no", "off")
    if not formula_on:
        cmd += ["-f", "false"]
    env = os.environ.copy()
    env.setdefault("MINERU_DEVICE_MODE", _default_device())
    env.setdefault("MINERU_MODEL_SOURCE", env.get("MINERU_MODEL_SOURCE", "huggingface"))

    try:
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=str(tmp_root),
            capture_output=True,
            text=True,
            timeout=_MINERU_TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError as exc:
        shutil.rmtree(tmp_root, ignore_errors=True)
        return ExtractionResult(
            status="manual_review",
            extractor="pdf-mineru",
            markdown="",
            error=f"mineru not on PATH: {exc}",
        )
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(tmp_root, ignore_errors=True)
        return ExtractionResult(
            status="manual_review",
            extractor="pdf-mineru",
            markdown="",
            error=f"mineru timed out after {_MINERU_TIMEOUT_S}s: {exc}",
        )

    try:
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "").strip().splitlines()[-12:]
            return ExtractionResult(
                status="manual_review",
                extractor="pdf-mineru",
                markdown="",
                error=f"mineru exit={proc.returncode}: {' | '.join(stderr_tail)}",
            )

        md_path, images_dir = _locate_mineru_outputs(tmp_root, src.stem)
        if md_path is None or not md_path.is_file():
            return ExtractionResult(
                status="manual_review",
                extractor="pdf-mineru",
                markdown="",
                error=f"mineru produced no markdown under {tmp_root}",
            )

        md_text = md_path.read_text(encoding="utf-8")

        copied: list[Path] = []
        if images_dir is not None and images_dir.is_dir():
            assets_dir.mkdir(parents=True, exist_ok=True)
            for asset in sorted(images_dir.iterdir()):
                if not asset.is_file():
                    continue
                target = assets_dir / asset.name
                shutil.copy2(asset, target)
                copied.append(target)
            # Rewrite ``images/<file>`` ONLY inside Markdown image links, not
            # by a blind string replace: prose or a URL containing the
            # substring "images/" (e.g. https://x/images/logo.png) would
            # otherwise be silently corrupted.
            assets_rel = assets_dir.name + "/"
            md_text = re.sub(
                r"(!\[[^\]]*\]\()images/",
                lambda m: m.group(1) + assets_rel,
                md_text,
            )

        notes = [f"backend={_MINERU_BACKEND} lang={lang} method={_MINERU_METHOD}"]
        if copied:
            notes.append(f"extracted {len(copied)} image asset(s)")

        # A MinerU run that exits 0 but emits empty markdown (image-only /
        # pathological PDF) must not be recorded as fully 'processed' — the
        # idempotency skip would then make the empty note permanent.
        if not md_text.strip():
            if copied:
                return ExtractionResult(
                    status="partial",
                    extractor="pdf-mineru",
                    markdown=md_text,
                    assets=copied,
                    notes=notes + ["mineru produced empty markdown (figures only)"],
                )
            return ExtractionResult(
                status="manual_review",
                extractor="pdf-mineru",
                markdown="",
                error="mineru produced empty markdown",
            )

        return ExtractionResult(
            status="processed",
            extractor="pdf-mineru",
            markdown=md_text,
            assets=copied,
            notes=notes,
        )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _locate_mineru_outputs(tmp_root: Path, stem: str) -> tuple[Path | None, Path | None]:
    """Find the .md and images/ produced by MinerU under tmp_root."""
    candidate_md = None
    # Match by exact name, not rglob(f"{stem}.md"): a stem with glob
    # metacharacters (report[2024].pdf -> [2024] is a character class) would
    # match nothing and silently fall through to the any-*.md branch.
    target_name = f"{stem}.md"
    for p in tmp_root.rglob("*.md"):
        if p.name == target_name:
            candidate_md = p
            break
    if candidate_md is None:
        # Fallback: any markdown file at all.
        for p in tmp_root.rglob("*.md"):
            candidate_md = p
            break
    if candidate_md is None:
        return None, None
    images_dir = candidate_md.parent / "images"
    return candidate_md, images_dir if images_dir.is_dir() else None


def _default_device() -> str:
    """Pick a sensible default device. User can override via MINERU_DEVICE_MODE."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return "cpu"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --------------------------------------------------------------------------
# pypdf fallback
# --------------------------------------------------------------------------

def _extract_with_pypdf(src: Path) -> ExtractionResult:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
        from pypdf.errors import PdfReadError  # type: ignore[import-not-found]
    except ImportError as exc:
        return ExtractionResult(
            status="manual_review",
            extractor="pdf-pypdf",
            markdown="",
            error=f"pypdf missing: {exc}",
        )

    try:
        reader = PdfReader(str(src))
    except PdfReadError as exc:
        return ExtractionResult(
            status="manual_review",
            extractor="pdf-pypdf",
            markdown="",
            error=f"pypdf open failed: {exc}",
        )
    except Exception as exc:  # noqa: BLE001
        return ExtractionResult(
            status="manual_review",
            extractor="pdf-pypdf",
            markdown="",
            error=f"pypdf unexpected error: {exc!r}",
        )

    # Encrypted PDFs raise FileNotDecryptedError on page-tree access (not at
    # construction). Try an empty-password decrypt; if that fails, surface a
    # clear, greppable 'encrypted' failure instead of a generic crash.
    if getattr(reader, "is_encrypted", False):
        try:
            if not reader.decrypt(""):
                return ExtractionResult(
                    status="manual_review",
                    extractor="pdf-pypdf",
                    markdown="",
                    error="pypdf: PDF is encrypted (password required)",
                )
        except Exception as exc:  # noqa: BLE001
            return ExtractionResult(
                status="manual_review",
                extractor="pdf-pypdf",
                markdown="",
                error=f"pypdf: PDF is encrypted and could not be decrypted: {exc!r}",
            )

    pages: list[str] = []
    page_errors = 0
    try:
        page_iter = list(reader.pages)
    except Exception as exc:  # noqa: BLE001 — page-tree access can still fail
        return ExtractionResult(
            status="manual_review",
            extractor="pdf-pypdf",
            markdown="",
            error=f"pypdf: could not read pages: {exc!r}",
        )
    for i, page in enumerate(page_iter, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:  # noqa: BLE001 — per-page errors shouldn't kill the doc
            page_errors += 1
            txt = ""
        pages.append(f"## Page {i}\n\n{txt.strip()}\n")
    body = "\n".join(pages) if pages else "_(no pages extracted)_\n"

    notes = ["pypdf text-only extraction; figures/tables/formulas are not captured"]
    if page_errors:
        notes.append(f"{page_errors} page(s) raised errors during text extraction")

    nonempty_pages = sum(1 for p in pages if p.split("\n\n", 1)[-1].strip())
    if nonempty_pages == 0:
        return ExtractionResult(
            status="partial",
            extractor="pdf-pypdf",
            markdown=body,
            notes=notes
            + ["no text layer detected — likely scanned. install MinerU for OCR."],
        )

    return ExtractionResult(
        status="partial",
        extractor="pdf-pypdf",
        markdown=body,
        notes=notes,
    )
