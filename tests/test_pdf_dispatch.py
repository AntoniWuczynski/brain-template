"""Tests for ``ingest_lib.extractors.pdf.extract`` — the dispatch function
and its pypdf-fallback path (F1: never executed by the existing suite,
which only calls ``_extract_with_mineru`` directly).

Real fixture PDFs are generated in-process with ``pypdf.PdfWriter`` (a
declared dependency); no external ``mineru`` binary is invoked — the
``mineru`` side is faked at the ``shutil.which`` / ``_extract_with_mineru``
seam so the test stays hermetic and fast.
"""
from __future__ import annotations

import io
import shutil
from pathlib import Path

import pytest
from pypdf import PdfWriter

from ingest_lib.extractors import pdf
from ingest_lib.extractors.base import ExtractionResult


def _write_blank_pdf(path: Path, *, width: float = 200, height: float = 200) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=width, height=height)
    buf = io.BytesIO()
    writer.write(buf)
    path.write_bytes(buf.getvalue())


def test_extract_takes_pypdf_path_when_mineru_not_on_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    src = tmp_path / "doc.pdf"
    _write_blank_pdf(src)

    res = pdf.extract(src, tmp_path / "assets")

    assert res.status == "partial"
    assert res.extractor == "pdf-pypdf"
    assert res.error is None
    assert any("no text layer detected" in n for n in res.notes)


def test_extract_falls_back_to_pypdf_when_mineru_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Simulate "mineru is on PATH" without a real binary...
    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/bin/mineru" if name == "mineru" else None
    )
    # ...and "it fails on this file" by faking its result directly (the CLI
    # invocation itself is exercised separately by test_extractors.py).
    monkeypatch.setattr(
        pdf,
        "_extract_with_mineru",
        lambda _src, _assets_dir: ExtractionResult(
            status="manual_review",
            extractor="pdf-mineru",
            markdown="",
            error="mineru exit=1: page has invalid encoding",
        ),
    )
    # A genuinely unreadable file, so the REAL pypdf fallback also fails —
    # exercising the "both errors kept" branch (pdf.py:55-57), not the
    # "fallback partially succeeded" one.
    src = tmp_path / "corrupt.pdf"
    src.write_bytes(b"not actually a pdf")

    res = pdf.extract(src, tmp_path / "assets")

    assert res.status == "manual_review"
    assert res.extractor == "pdf-pypdf-fallback"
    assert res.error is not None
    assert "mineru exit=1: page has invalid encoding" in res.error
    assert "pypdf" in res.error
    assert any(
        "MinerU failed; fell back to pypdf" in n and "page has invalid encoding" in n
        for n in res.notes
    )


def test_extract_routes_to_vlm_when_pdf_extractor_env_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BRAIN_PDF_EXTRACTOR", "vlm")
    from ingest_lib.extractors import vlm as vlm_mod

    seen: dict[str, Path] = {}

    def fake_vlm_extract(src: Path, assets_dir: Path) -> ExtractionResult:
        seen["src"] = src
        seen["assets_dir"] = assets_dir
        return ExtractionResult(status="processed", extractor="pdf-vlm", markdown="ok")

    monkeypatch.setattr(vlm_mod, "extract", fake_vlm_extract)
    src = tmp_path / "scan.pdf"
    src.write_bytes(b"irrelevant for this routing test")
    assets = tmp_path / "assets"

    res = pdf.extract(src, assets)

    assert res.extractor == "pdf-vlm"
    assert seen == {"src": src, "assets_dir": assets}
