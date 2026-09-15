"""Extractor unit tests over small synthetic fixtures.

The extractors package had no coverage despite holding the riskiest
branching in the pipeline (fence collisions, table escaping, honesty
downgrades). These pin the correctness fixes without needing MinerU,
torch, or a vision API.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import pytest

from ingest_lib.extractors.base import fence
from ingest_lib.extractors import dataset as ds
from ingest_lib.extractors import text as text_ex
from ingest_lib.notes import NoteContent, write_index_note

if TYPE_CHECKING:
    from types import SimpleNamespace
    from google.genai.types import FinishReason


# --------------------------------------------------------------- fence()

def test_fence_plain_uses_three_backticks() -> None:
    out = fence("hello", "py")
    assert out == "```py\nhello\n```"


def test_fence_grows_past_inner_backtick_run() -> None:
    # Content containing ``` must be wrapped in a LONGER fence, else it
    # closes early.
    content = "before\n```\ninner\n```\nafter"
    out = fence(content, "md")
    assert out.startswith("````md\n")
    assert out.endswith("\n````")
    # The inner ``` no longer terminates the block.
    assert out.count("````") == 2


# ---------------------------------------------------------------- text.py

def test_text_extractor_survives_embedded_code_fence(tmp_path: Path) -> None:
    src = tmp_path / "note.md"
    src.write_text("# Title\n\n```python\nprint('x')\n```\n", encoding="utf-8")
    res = text_ex.extract(src, tmp_path / "assets")
    assert res.status == "processed"
    # The wrapping fence is longer than the inner ``` so nothing spills out.
    assert res.markdown.startswith("````md\n")
    assert "print('x')" in res.markdown


def test_text_extractor_byte_cap_is_true_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(text_ex, "_MAX_BYTES", 8)
    src = tmp_path / "big.txt"
    src.write_text("abcdefghijklmnop", encoding="utf-8")  # 16 bytes
    res = text_ex.extract(src, tmp_path / "a")
    assert res.status == "partial"
    assert any("truncated" in n for n in res.notes)


# ------------------------------------------------------------- dataset.py

def test_dataset_csv_escapes_pipe_in_header(tmp_path: Path) -> None:
    src = tmp_path / "d.csv"
    src.write_text("price|usd,name\n10,widget\n", encoding="utf-8")
    res = ds.extract(src, tmp_path / "a")
    assert res.status == "processed"
    # The '|' in the header is escaped so it can't add a phantom column.
    assert "price\\|usd" in res.markdown


def test_dataset_jsonl_marks_partial_on_unparseable_lines(tmp_path: Path) -> None:
    src = tmp_path / "d.jsonl"
    src.write_text('{"a": 1}\nnot json\n{"a": 2}\n', encoding="utf-8")
    res = ds.extract(src, tmp_path / "a")
    assert res.status == "partial"
    assert any("unparseable" in n for n in res.notes)
    assert "**Records:** 2" in res.markdown


def test_dataset_csv_strips_excel_bom_from_header(tmp_path: Path) -> None:
    # Excel's "CSV UTF-8" export prepends a BOM; it must not leak into the
    # first column name of the schema table.
    src = tmp_path / "d.csv"
    src.write_bytes(b"\xef\xbb\xbfname,age\nalice,30\n")
    res = ds.extract(src, tmp_path / "a")
    assert res.status == "processed"
    assert "| 1 | `name` |" in res.markdown
    assert "﻿" not in res.markdown


def test_dataset_jsonl_bom_does_not_drop_first_record(tmp_path: Path) -> None:
    src = tmp_path / "d.jsonl"
    src.write_bytes(b'\xef\xbb\xbf{"a": 1}\n{"a": 2}\n')
    res = ds.extract(src, tmp_path / "a")
    assert res.status == "processed"
    assert res.notes == []
    assert "**Records:** 2" in res.markdown


def test_dataset_csv_quoted_crlf_cell_stays_on_one_preview_row(tmp_path: Path) -> None:
    # A quoted multiline cell (Excel Alt+Enter) reaches _clip with \r\n
    # intact; a bare CR surviving into the note splits the table row.
    src = tmp_path / "d.csv"
    src.write_bytes(b'name,notes\r\nalice,"line1\r\nline2"\r\nbob,plain\r\n')
    res = ds.extract(src, tmp_path / "a")
    assert res.status == "processed"
    assert "\r" not in res.markdown
    assert "| alice | line1 line2 |" in res.markdown


def test_dataset_csv_oversized_field_is_manual_review(tmp_path: Path) -> None:
    import csv
    src = tmp_path / "d.csv"
    big = "x" * (csv.field_size_limit() + 10)
    src.write_text(f'col\n"{big}"\n', encoding="utf-8")
    res = ds.extract(src, tmp_path / "a")
    assert res.status == "manual_review"
    assert "csv parse failed" in (res.error or "")


# ------------------------------------------------------------- notebook.py

def _nb_json(cells: list[dict[str, object]], *, language: str = "python") -> str:
    # `id` on every cell: nbformat >= 5.1 warns (and will eventually error)
    # on v4.5 cells without one.
    stamped = [{**c, "id": f"c{i}"} for i, c in enumerate(cells)]
    return json.dumps({
        "cells": stamped,
        "metadata": {"kernelspec": {"language": language, "name": language}},
        "nbformat": 4,
        "nbformat_minor": 5,
    })


def test_notebook_extracts_cells_and_reports_omitted_outputs(tmp_path: Path) -> None:
    from ingest_lib.extractors import notebook as nb_ex
    src = tmp_path / "analysis.ipynb"
    src.write_text(_nb_json([
        {"cell_type": "markdown", "metadata": {}, "source": "# Heading\n\ntext"},
        {
            "cell_type": "code", "metadata": {}, "execution_count": 1,
            "source": "print('x')",
            "outputs": [
                {"output_type": "stream", "name": "stdout", "text": "x\n"},
                {"output_type": "display_data", "metadata": {},
                 "data": {"text/plain": "<Figure>"}},
            ],
        },
        {"cell_type": "raw", "metadata": {}, "source": "raw body"},
    ], language="julia"), encoding="utf-8")

    res = nb_ex.extract(src, tmp_path / "assets")

    assert res.status == "processed"
    assert res.extractor == "notebook"
    assert "# Heading" in res.markdown
    # Code cells are fenced with the kernel language, raw cells bare.
    assert "```julia\nprint('x')\n```" in res.markdown
    assert "```\nraw body\n```" in res.markdown
    # Honesty: outputs are dropped, and the note says exactly how many.
    assert res.notes == [
        "omitted 2 cell output(s) (images/text/streams) — see source notebook for them"
    ]


def test_notebook_without_outputs_adds_no_omission_note(tmp_path: Path) -> None:
    from ingest_lib.extractors import notebook as nb_ex
    src = tmp_path / "clean.ipynb"
    src.write_text(_nb_json([
        {"cell_type": "code", "metadata": {}, "execution_count": None,
         "source": "y = 1", "outputs": []},
    ]), encoding="utf-8")

    res = nb_ex.extract(src, tmp_path / "assets")
    assert res.status == "processed"
    assert res.notes == []
    assert "```python\ny = 1\n```" in res.markdown


def test_notebook_v3_is_upgraded_in_memory(tmp_path: Path) -> None:
    # v3 exposes `worksheets`, not `cells`: without as_version=4 this file
    # would AttributeError instead of extracting.
    from ingest_lib.extractors import notebook as nb_ex
    src = tmp_path / "legacy.ipynb"
    src.write_text(json.dumps({
        "metadata": {"name": "legacy"},
        "nbformat": 3,
        "nbformat_minor": 0,
        "worksheets": [{"cells": [
            {"cell_type": "markdown", "source": ["old markdown"], "metadata": {}},
            {"cell_type": "code", "language": "python", "collapsed": False,
             "input": ["z = 2"], "outputs": [], "metadata": {}},
        ], "metadata": {}}],
    }), encoding="utf-8")

    res = nb_ex.extract(src, tmp_path / "assets")
    assert res.status == "processed"
    assert "old markdown" in res.markdown
    assert "z = 2" in res.markdown


def test_notebook_corrupt_file_is_manual_review(tmp_path: Path) -> None:
    from ingest_lib.extractors import notebook as nb_ex
    src = tmp_path / "broken.ipynb"
    src.write_text("{not json at all", encoding="utf-8")

    res = nb_ex.extract(src, tmp_path / "assets")
    assert res.status == "manual_review"
    assert res.extractor == "notebook"
    assert res.markdown == ""
    assert (res.error or "").startswith("open failed:")


def test_notebook_missing_nbformat_is_manual_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # nbformat is a declared dependency, but the extractor still has to
    # degrade honestly rather than crash the run if it is not importable.
    import builtins
    from collections.abc import Mapping, Sequence
    from types import ModuleType

    from ingest_lib.extractors import notebook as nb_ex
    real_import = builtins.__import__

    def no_nbformat(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "nbformat":
            raise ImportError("no module named nbformat")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", no_nbformat)
    src = tmp_path / "x.ipynb"
    src.write_text(_nb_json([]), encoding="utf-8")
    res = nb_ex.extract(src, tmp_path / "assets")
    assert res.status == "manual_review"
    assert "nbformat missing" in (res.error or "")


# --------------------------------------------------- registry / parquet stub

def test_notebook_and_dataset_extensions_are_registered() -> None:
    from ingest_lib.extractors import dispatch_extractor, registered_extensions
    from ingest_lib.extractors import notebook as nb_ex
    assert dispatch_extractor(Path("a.ipynb")) is nb_ex.extract
    assert dispatch_extractor(Path("A.IPYNB")) is nb_ex.extract
    assert dispatch_extractor(Path("rows.csv")) is ds.extract
    assert dispatch_extractor(Path("cube.parquet")) is ds.extract_parquet_stub
    # Nothing is registered for an unknown extension.
    assert dispatch_extractor(Path("x.unknownext")) is None

    exts = registered_extensions()
    assert exts == sorted(exts) and len(exts) == len(set(exts))
    assert {".ipynb", ".csv", ".tsv", ".jsonl", ".parquet", ".pdf"} <= set(exts)
    assert all(e.startswith(".") and e == e.lower() for e in exts)


def test_parquet_stub_is_honest_manual_review(tmp_path: Path) -> None:
    src = tmp_path / "cube.parquet"
    src.write_bytes(b"PAR1" + b"\0" * 12)
    res = ds.extract_parquet_stub(src, tmp_path / "assets")
    assert res.status == "manual_review"
    assert res.extractor == "dataset-parquet"
    assert res.error == "parquet extractor not implemented"
    assert res.notes == ["file size: 16 bytes"]
    # No invented content: the markdown says only that it is unimplemented.
    assert "not implemented yet" in res.markdown


def test_parquet_stub_reports_a_failed_stat(tmp_path: Path) -> None:
    res = ds.extract_parquet_stub(tmp_path / "gone.parquet", tmp_path / "assets")
    assert res.status == "manual_review"
    assert len(res.notes) == 1 and res.notes[0].startswith("stat failed:")


def test_dataset_extract_rejects_an_unrecognised_extension(tmp_path: Path) -> None:
    src = tmp_path / "rows.dat"
    src.write_text("a,b\n", encoding="utf-8")
    res = ds.extract(src, tmp_path / "assets")
    assert res.status == "manual_review"
    assert res.error == "unrecognised dataset extension: .dat"


# ------------------------------------------------------- docx (D8/F110)

# 1x1 transparent PNG for embedding.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def test_docx_extracts_table_and_flags_embedded_image(tmp_path: Path) -> None:
    from docx import Document
    from docx.oxml import OxmlElement
    doc = Document()
    doc.add_paragraph("Intro paragraph.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "price|usd"   # pipe must be escaped
    table.cell(0, 1).text = "name"
    table.cell(1, 0).text = "10"
    table.cell(1, 1).text = "widget"
    # Inject a w:drawing element (what the extractor counts) rather than a
    # real image, so the test doesn't depend on python-docx's image parser.
    run = doc.add_paragraph("figure below").add_run()
    run._r.append(OxmlElement("w:drawing"))
    src = tmp_path / "d.docx"
    doc.save(str(src))

    from ingest_lib.extractors import docx as docx_ex
    res = docx_ex.extract(src, tmp_path / "a")
    assert "Intro paragraph." in res.markdown
    assert "price\\|usd" in res.markdown            # table extracted + escaped
    assert "| widget |" in res.markdown
    # An embedded image can't be extracted -> honest partial + note.
    assert res.status == "partial"
    assert any("image" in n or "drawing" in n for n in res.notes)


# ------------------------------------------------------- pptx (D8/F036)

def test_pptx_extracts_table_and_flags_picture(tmp_path: Path) -> None:
    from pptx import Presentation
    from pptx.util import Inches
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    tb = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1))
    tb.text_frame.text = "A bullet of text"
    tbl = slide.shapes.add_table(2, 2, Inches(1), Inches(2), Inches(4), Inches(1)).table
    tbl.cell(0, 0).text = "col|a"
    tbl.cell(0, 1).text = "colb"
    tbl.cell(1, 0).text = "1"
    tbl.cell(1, 1).text = "2"
    import io
    slide.shapes.add_picture(io.BytesIO(_PNG), Inches(1), Inches(3))
    src = tmp_path / "p.pptx"
    prs.save(str(src))

    from ingest_lib.extractors import pptx as pptx_ex
    res = pptx_ex.extract(src, tmp_path / "a")
    assert "A bullet of text" in res.markdown
    assert "col\\|a" in res.markdown                # table cell extracted + escaped
    assert "| colb |" in res.markdown
    assert res.status == "partial"                  # picture present
    assert any("picture" in n for n in res.notes)


# ------------------------------------------------------------------ vlm.py

def _gemini_resp(text: str, finish_reason: FinishReason) -> SimpleNamespace:
    from types import SimpleNamespace
    return SimpleNamespace(
        text=text, candidates=[SimpleNamespace(finish_reason=finish_reason)]
    )


def _patch_gemini(monkeypatch: pytest.MonkeyPatch, resp: SimpleNamespace) -> None:
    from google import genai
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    class FakeModels:
        def generate_content(self, **_kw: object) -> SimpleNamespace:
            return resp

    class FakeClient:
        def __init__(self, **_kw: object) -> None:
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeClient)


def test_vlm_gemini_blank_page_is_success_not_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # Contract (stated three times in vlm.py): "" == blank page (success),
    # None only on failure. The gemini helper must not coerce "" to None.
    from google.genai import types
    from ingest_lib.extractors import vlm

    _patch_gemini(monkeypatch, _gemini_resp("", types.FinishReason.STOP))
    res = vlm._vision_gemini(png=b"png", model="gemini-2.5-flash")
    assert res is not None
    assert res.text == ""
    assert res.truncated is False


def test_vlm_gemini_reports_max_tokens_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.genai import types
    from ingest_lib.extractors import vlm

    _patch_gemini(
        monkeypatch, _gemini_resp("cut off mid", types.FinishReason.MAX_TOKENS)
    )
    res = vlm._vision_gemini(png=b"png", model="gemini-2.5-flash")
    assert res is not None
    assert res.truncated is True


def test_vlm_anthropic_reports_max_tokens_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    import anthropic
    from types import SimpleNamespace
    from ingest_lib.extractors import vlm

    block = anthropic.types.TextBlock.model_construct(type="text", text="cut off")
    resp = SimpleNamespace(content=[block], stop_reason="max_tokens")

    class FakeMessages:
        def create(self, **_kw: object) -> SimpleNamespace:
            return resp

    class FakeClient:
        def __init__(self, **_kw: object) -> None:
            self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    res = vlm._vision_anthropic(png=b"png", model="claude-sonnet-4-6")
    assert res is not None
    assert res.text == "cut off"
    assert res.truncated is True


def test_vlm_openai_reports_length_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai
    from types import SimpleNamespace
    from ingest_lib.extractors import vlm

    resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="cut off"), finish_reason="length"
            )
        ]
    )

    class FakeCompletions:
        def create(self, **_kw: object) -> SimpleNamespace:
            return resp

    class FakeClient:
        def __init__(self, **_kw: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    res = vlm._vision_openai_compatible(
        png=b"png", model="gpt-5-mini", provider="openai"
    )
    assert res is not None
    assert res.truncated is True


def test_vlm_extract_marks_truncated_pages_partial(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ingest_lib import summarize as _summ
    from ingest_lib.extractors import vlm

    monkeypatch.setattr(_summ, "_select_provider", lambda: "anthropic")
    monkeypatch.setattr(vlm, "_render_pages", lambda _src: [b"p1", b"p2"])

    def fake_transcribe(
        *, png: bytes, provider: str, model: str, page_no: int
    ) -> vlm._VisionText | vlm._PageFailure:
        if page_no == 1:
            return vlm._VisionText("dense page", truncated=True)
        return vlm._VisionText("fine page", truncated=False)

    monkeypatch.setattr(vlm, "_transcribe_page", fake_transcribe)
    res = vlm.extract(tmp_path / "doc.pdf", tmp_path / "assets")
    assert res.status == "partial"
    assert any("truncated" in n for n in res.notes)
    # Truncated is NOT failed: the all-pages-failed cleanup must not fire and
    # the truncated text must survive with a visible marker.
    assert not any("failed transcription" in n for n in res.notes)
    assert "dense page" in res.markdown
    assert "truncated at the model output cap" in res.markdown
    assert len(res.assets) == 2


def test_vlm_extract_blank_page_stays_processed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ingest_lib import summarize as _summ
    from ingest_lib.extractors import vlm

    monkeypatch.setattr(_summ, "_select_provider", lambda: "anthropic")
    monkeypatch.setattr(vlm, "_render_pages", lambda _src: [b"p1"])
    monkeypatch.setattr(
        vlm,
        "_transcribe_page",
        lambda **_kw: vlm._VisionText("", truncated=False),
    )
    res = vlm.extract(tmp_path / "doc.pdf", tmp_path / "assets")
    assert res.status == "processed"
    assert "_(blank page)_" in res.markdown


def test_vlm_extract_all_failed_reports_first_cause(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # When every page fails, the manual_review error must carry the first
    # page's underlying cause, not just a bare count — otherwise the failure
    # is undiagnosable from index.jsonl / the log.
    from ingest_lib import summarize as _summ
    from ingest_lib.extractors import vlm

    monkeypatch.setattr(_summ, "_select_provider", lambda: "anthropic")
    monkeypatch.setattr(vlm, "_render_pages", lambda _src: [b"p1", b"p2"])
    monkeypatch.setattr(
        vlm,
        "_transcribe_page",
        lambda **_kw: vlm._PageFailure("RuntimeError('vision api exploded')"),
    )
    assets = tmp_path / "assets"
    res = vlm.extract(tmp_path / "doc.pdf", assets)
    assert res.status == "manual_review"
    assert res.error is not None
    assert "all 2 page(s) failed transcription" in res.error
    assert "vision api exploded" in res.error   # the threaded cause
    # Existing behaviour preserved: assets cleaned up, none returned.
    assert not assets.exists()
    assert res.assets == []


def test_vlm_extract_local_model_env_read_at_call_time(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # BRAIN_LOCAL_MODEL is read when extract() runs, not at import — setting it
    # after the module is imported must still be honoured (matches summarize).
    from ingest_lib import summarize as _summ
    from ingest_lib.extractors import vlm

    monkeypatch.setattr(_summ, "_select_provider", lambda: "local")
    monkeypatch.setattr(vlm, "_render_pages", lambda _src: [b"p1"])
    monkeypatch.delenv("BRAIN_VLM_MODEL", raising=False)
    monkeypatch.setenv("BRAIN_LOCAL_MODEL", "my-local-vision:custom")

    seen: dict[str, str] = {}

    def fake_transcribe(*, png: bytes, provider: str, model: str, page_no: int) -> vlm._VisionText:
        seen["model"] = model
        return vlm._VisionText("ok", truncated=False)

    monkeypatch.setattr(vlm, "_transcribe_page", fake_transcribe)
    vlm.extract(tmp_path / "doc.pdf", tmp_path / "assets")
    assert seen["model"] == "my-local-vision:custom"


def test_vlm_anthropic_default_model_is_sonnet_5(monkeypatch: pytest.MonkeyPatch) -> None:
    # The anthropic vision default is pinned to the current Sonnet: same 1M
    # context as sonnet-4-6 at a lower price. An unknown provider falls back
    # to the same anthropic default.
    from ingest_lib.extractors import vlm

    monkeypatch.delenv("BRAIN_VLM_MODEL", raising=False)
    assert vlm._default_vlm_model("anthropic") == "claude-sonnet-5"
    assert vlm._default_vlm_model("nonesuch") == "claude-sonnet-5"


def test_mineru_output_handling_crash_degrades_to_manual_review(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An unexpected exception while gathering MinerU's outputs must degrade to
    # manual_review (so extract() falls back to pypdf), not abort ingestion.
    # Patched via the directly-imported `subprocess` module (the same module
    # object `pdf.py` calls through), rather than `pdf.subprocess`, since the
    # latter isn't an attribute `pdf` explicitly re-exports.
    import subprocess
    from types import SimpleNamespace
    from ingest_lib.extractors import pdf

    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    def boom(_tmp_root: Path, _stem: str) -> NoReturn:
        raise RuntimeError("locate exploded")

    monkeypatch.setattr(pdf, "_locate_mineru_outputs", boom)
    res = pdf._extract_with_mineru(tmp_path / "x.pdf", tmp_path / "assets")
    assert res.status == "manual_review"
    assert res.error is not None
    assert "output handling failed" in res.error
    assert "locate exploded" in res.error


# ------------------------------------------------------------------ image.py

def _png_bytes() -> bytes:
    from PIL import Image
    import io
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def test_image_extension_is_registered() -> None:
    from ingest_lib.extractors import dispatch_extractor
    from ingest_lib.extractors import image as image_ex
    from pathlib import Path
    assert dispatch_extractor(Path("photo.jpg")) is image_ex.extract
    assert dispatch_extractor(Path("scan.HEIC")) is image_ex.extract


def test_image_no_vision_provider_is_manual_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ingest_lib import summarize as _summ
    from ingest_lib.extractors import image as image_ex
    monkeypatch.setattr(_summ, "_select_provider", lambda: None)
    src = tmp_path / "x.png"
    src.write_bytes(_png_bytes())
    res = image_ex.extract(src, tmp_path / "a")
    assert res.status == "manual_review"
    assert "no vision" in (res.error or "").lower()


def test_image_transcription_becomes_processed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ingest_lib import summarize as _summ
    from ingest_lib.extractors import image as image_ex
    from ingest_lib.extractors import vlm
    monkeypatch.setattr(_summ, "_select_provider", lambda: "anthropic")
    monkeypatch.setattr(
        vlm, "_transcribe_page",
        lambda **_kw: vlm._VisionText("# Whiteboard\n\nSprint goals", truncated=False),
    )
    src = tmp_path / "board.jpg"
    src.write_bytes(_png_bytes())
    res = image_ex.extract(src, tmp_path / "a")
    assert res.status == "processed"
    assert "Sprint goals" in res.markdown


def test_image_transcription_failure_is_manual_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ingest_lib import summarize as _summ
    from ingest_lib.extractors import image as image_ex
    from ingest_lib.extractors import vlm
    monkeypatch.setattr(_summ, "_select_provider", lambda: "anthropic")
    monkeypatch.setattr(
        vlm, "_transcribe_page",
        lambda **_kw: vlm._PageFailure("rate limited"),
    )
    src = tmp_path / "x.png"
    src.write_bytes(_png_bytes())
    res = image_ex.extract(src, tmp_path / "a")
    assert res.status == "manual_review"
    assert "rate limited" in (res.error or "")


def test_image_undecodable_file_is_manual_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ingest_lib import summarize as _summ
    from ingest_lib.extractors import image as image_ex
    monkeypatch.setattr(_summ, "_select_provider", lambda: "anthropic")
    src = tmp_path / "corrupt.png"
    src.write_bytes(b"this is not a real png")
    res = image_ex.extract(src, tmp_path / "a")
    assert res.status == "manual_review"
    assert "decode" in (res.error or "").lower()


# ------------------------------------------------------------------ audio.py

def test_srt_transcript_extracted(tmp_path: Path) -> None:
    from ingest_lib.extractors import audio as audio_ex
    srt = (
        "1\n00:00:01,000 --> 00:00:03,000\nHello and welcome.\n\n"
        "2\n00:00:03,000 --> 00:00:05,000\nHello and welcome.\n\n"   # duplicate (rolling)
        "3\n00:01:10,000 --> 00:01:12,000\nNow the second minute.\n"
    )
    src = tmp_path / "talk.srt"
    src.write_text(srt, encoding="utf-8")
    res = audio_ex.extract(src, tmp_path / "a")
    assert res.status == "processed"
    assert "## Transcript" in res.markdown
    assert "Now the second minute." in res.markdown
    assert res.markdown.count("Hello and welcome.") == 1   # dedupe
    assert "**[0:01]**" in res.markdown and "**[1:10]**" in res.markdown  # segment markers


def test_vtt_with_tags_and_dot_times(tmp_path: Path) -> None:
    from ingest_lib.extractors import audio as audio_ex
    vtt = (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.000\n<v Alice>Hi <b>there</b></v>\n\n"
        "00:00:02.000 --> 00:00:04.000\nSecond line.\n"
    )
    src = tmp_path / "call.vtt"
    src.write_text(vtt, encoding="utf-8")
    res = audio_ex.extract(src, tmp_path / "a")
    assert res.status == "processed"
    assert "Hi there" in res.markdown        # inline tags stripped
    assert "<v" not in res.markdown and "<b>" not in res.markdown


def test_unparseable_subtitle_is_manual_review(tmp_path: Path) -> None:
    from ingest_lib.extractors import audio as audio_ex
    src = tmp_path / "bad.srt"
    src.write_text("this has no timing lines at all\n", encoding="utf-8")
    res = audio_ex.extract(src, tmp_path / "a")
    assert res.status == "manual_review"
    assert "no parseable subtitle cues" in (res.error or "")


def test_audio_without_asr_backend_is_manual_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # No faster-whisper installed -> honest manual_review with the install cmd.
    import builtins
    from collections.abc import Mapping, Sequence
    from types import ModuleType
    from ingest_lib.extractors import audio as audio_ex
    real_import = builtins.__import__

    def _no_whisper(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "faster_whisper":
            raise ImportError("no faster_whisper")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _no_whisper)
    src = tmp_path / "lecture.m4a"
    src.write_bytes(b"not really audio")
    res = audio_ex.extract(src, tmp_path / "a")
    assert res.status == "manual_review"
    assert "faster-whisper" in (res.error or "") and "ffmpeg" in (res.error or "")


def test_audio_extensions_registered() -> None:
    from ingest_lib.extractors import dispatch_extractor
    from ingest_lib.extractors import audio as audio_ex
    from pathlib import Path as P
    assert dispatch_extractor(P("x.srt")) is audio_ex.extract
    assert dispatch_extractor(P("x.m4a")) is audio_ex.extract
    assert dispatch_extractor(P("x.vtt")) is audio_ex.extract


def test_vtt_note_and_header_blocks_skipped(tmp_path: Path) -> None:
    from ingest_lib.extractors import audio as audio_ex
    vtt = (
        "WEBVTT\n\n"
        "NOTE this is a comment with a stray --> arrow 00:00:00,000\n\n"
        "00:00:01.000 --> 00:00:02.000\nReal line.\n"
    )
    src = tmp_path / "c.vtt"
    src.write_text(vtt, encoding="utf-8")
    res = audio_ex.extract(src, tmp_path / "a")
    assert res.status == "processed"
    assert "Real line." in res.markdown
    assert "this is a comment" not in res.markdown   # NOTE block not a cue


def test_subtitle_unparseable_timing_marks_partial(tmp_path: Path) -> None:
    from ingest_lib.extractors import audio as audio_ex
    srt = (
        "1\n00:00:01,000 --> 00:00:02,000\nGood cue.\n\n"
        "2\nBADTIME --> ALSOBAD\nOrphan cue text.\n"
    )
    src = tmp_path / "d.srt"
    src.write_text(srt, encoding="utf-8")
    res = audio_ex.extract(src, tmp_path / "a")
    assert res.status == "partial"                    # honest about the dropped cue
    assert any("dropped" in n for n in res.notes)
    assert "Good cue." in res.markdown


def test_subtitle_tag_strip_keeps_literal_angle_brackets(tmp_path: Path) -> None:
    from ingest_lib.extractors import audio as audio_ex
    srt = "1\n00:00:01,000 --> 00:00:02,000\nif x < 3 and y > 2 then done\n"
    src = tmp_path / "e.srt"
    src.write_text(srt, encoding="utf-8")
    res = audio_ex.extract(src, tmp_path / "a")
    assert "x < 3 and y > 2" in res.markdown           # not eaten as a tag


# --------------------------------------------- write_index_note frontmatter merge

def _note_content(*, status: str = "partial") -> NoteContent:
    return NoteContent(
        title="T",
        source_relative_path="a/b.txt",
        source_hash="h1",
        status=status,
        extracted_markdown="body",
        processing_notes=[],
        extractor="text",
    )


def test_reingest_over_unparseable_frontmatter_leaves_note_untouched(tmp_path: Path) -> None:
    # F5: a hand-corrupted YAML block (unclosed bracket) must never be
    # silently treated as "no user keys" and overwritten with generated
    # frontmatter only — that discards every user-added key with no trace.
    target = tmp_path / "note.md"
    write_index_note(target=target, content=_note_content())
    corrupted = target.read_text(encoding="utf-8").replace(
        "aliases: []", "aliases: []\nmy_key: important user note\nbroken: [unclosed"
    )
    target.write_text(corrupted, encoding="utf-8")

    write_index_note(target=target, content=_note_content())

    assert target.read_text(encoding="utf-8") == corrupted


def test_reingest_preserves_user_keys_when_frontmatter_is_valid(tmp_path: Path) -> None:
    # Sanity check alongside the corruption case above: a *valid* user block
    # is still merged in as before (rule 9's ordinary path is untouched).
    target = tmp_path / "note.md"
    write_index_note(target=target, content=_note_content())
    edited = target.read_text(encoding="utf-8").replace(
        "aliases: []", "aliases: []\nmy_key: important user note"
    )
    target.write_text(edited, encoding="utf-8")

    write_index_note(target=target, content=_note_content())

    assert "my_key: important user note" in target.read_text(encoding="utf-8")


def test_reingest_preserves_user_scalar_spelling_not_yaml11_coercion(tmp_path: Path) -> None:
    # F6: PyYAML is YAML 1.1 — round-tripping a user value through
    # safe_load/safe_dump silently rewrites `yes` -> `true`, `010` -> `8`,
    # `12:30` -> `750`, `0101` -> `65`. The user's keys are not owned by the
    # pipeline, so their original source text must survive verbatim.
    target = tmp_path / "note.md"
    write_index_note(target=target, content=_note_content())
    edited = target.read_text(encoding="utf-8").replace(
        "aliases: []",
        "aliases: []\ncourse: 0101\nmy_aliases: [yes, 010, 1e3, 12:30, 'Y']",
    )
    target.write_text(edited, encoding="utf-8")

    write_index_note(target=target, content=_note_content())

    text = target.read_text(encoding="utf-8")
    assert "course: 0101" in text
    assert "my_aliases: [yes, 010, 1e3, 12:30, 'Y']" in text
    # None of the YAML 1.1 coercions leaked through.
    assert "course: 65" not in text
    assert "750" not in text
