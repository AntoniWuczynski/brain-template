"""Tests for ``ingest_lib.extractors.ppt`` — the legacy .ppt -> .pptx
LibreOffice conversion shim.

No real ``soffice`` binary is invoked: PATH is pointed at small fake
executable scripts written per-test, so the suite stays hermetic and
fast while still exercising the real subprocess/PATH dispatch (not a
mock of ``subprocess.run``). The one real-LibreOffice check the task
requires is run manually against a live vault file, not from this suite
(see the session report).
"""
from __future__ import annotations

import os
import shutil
import stat
import tempfile
from pathlib import Path

import pytest

from ingest_lib.extractors import dispatch_extractor, registered_extensions
from ingest_lib.extractors import ppt


def _write_fake_soffice(bin_dir: Path, script: str) -> None:
    """Write an executable ``soffice`` to ``bin_dir`` and put it first on
    PATH for the duration of the test (via the caller's monkeypatch)."""
    path = bin_dir / "soffice"
    path.write_text(script)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _prepend_path(monkeypatch: pytest.MonkeyPatch, bin_dir: Path) -> None:
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")


# ------------------------------------------------------------- dispatch

def test_ppt_extension_is_registered() -> None:
    assert dispatch_extractor(Path("deck.ppt")) is ppt.extract
    assert dispatch_extractor(Path("DECK.PPT")) is ppt.extract
    assert ".ppt" in registered_extensions()


# ---------------------------------------------------------- absent soffice

def test_extract_is_manual_review_when_soffice_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    src = tmp_path / "deck.ppt"
    src.write_bytes(b"not a real ppt, irrelevant for this path")

    res = ppt.extract(src, tmp_path / "assets")

    assert res.status == "manual_review"
    assert res.extractor == "ppt-soffice"
    assert res.markdown == ""
    assert res.error is not None
    assert "soffice" in res.error.lower()
    assert "LibreOffice" in res.error


# ------------------------------------------------------------ failure path

def test_extract_is_manual_review_when_soffice_conversion_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_soffice(
        bin_dir,
        "#!/bin/sh\n"
        "echo 'boom: could not load document' 1>&2\n"
        "exit 1\n",
    )
    _prepend_path(monkeypatch, bin_dir)

    src = tmp_path / "deck.ppt"
    src.write_bytes(b"garbage")

    res = ppt.extract(src, tmp_path / "assets")

    assert res.status == "manual_review"
    assert res.extractor == "ppt-soffice"
    assert res.markdown == ""
    assert res.error is not None
    assert "exit=1" in res.error
    assert "boom: could not load document" in res.error


def test_extract_is_manual_review_when_soffice_produces_no_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Exits 0 (as soffice sometimes does on an unsupported/corrupt input)
    # but writes nothing to --outdir.
    _write_fake_soffice(bin_dir, "#!/bin/sh\nexit 0\n")
    _prepend_path(monkeypatch, bin_dir)

    src = tmp_path / "deck.ppt"
    src.write_bytes(b"garbage")

    res = ppt.extract(src, tmp_path / "assets")

    assert res.status == "manual_review"
    assert res.extractor == "ppt-soffice"
    assert res.error is not None
    assert "produced no" in res.error
    assert "deck.pptx" in res.error


def test_extract_reports_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ppt, "_SOFFICE_TIMEOUT_S", 0.01)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_soffice(bin_dir, "#!/bin/sh\nsleep 5\n")
    _prepend_path(monkeypatch, bin_dir)

    src = tmp_path / "deck.ppt"
    src.write_bytes(b"garbage")

    res = ppt.extract(src, tmp_path / "assets")

    assert res.status == "manual_review"
    assert res.error is not None
    assert "timed out" in res.error


# ------------------------------------------------------------ success path

def _write_real_pptx(path: Path) -> None:
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    tb = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1))
    tb.text_frame.text = "Converted slide text"
    prs.save(str(path))


# A fake soffice that scans its own argv for "--outdir <dir>" and treats
# the last argument as the source file, rather than hardcoding argument
# positions — so it keeps working if ppt.py's exact flag order changes.
_FAKE_SOFFICE_COPY_SCRIPT = (
    "#!/bin/sh\n"
    "outdir=\"\"\n"
    "prev=\"\"\n"
    "last=\"\"\n"
    "for arg in \"$@\"; do\n"
    "  if [ \"$prev\" = \"--outdir\" ]; then outdir=\"$arg\"; fi\n"
    "  prev=\"$arg\"\n"
    "  last=\"$arg\"\n"
    "done\n"
    "stem=$(basename \"$last\" .ppt)\n"
    "cp \"$FAKE_SOFFICE_SRC_PPTX\" \"$outdir/$stem.pptx\"\n"
    "exit 0\n"
)


def test_extract_delegates_to_pptx_extractor_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The fake soffice "conversion" just copies a real pptx (built with
    # python-pptx, a declared dependency) to where soffice would have
    # written its output, so the delegation to pptx.extract is real.
    fixture_pptx = tmp_path / "fixture.pptx"
    _write_real_pptx(fixture_pptx)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_soffice(bin_dir, _FAKE_SOFFICE_COPY_SCRIPT)
    _prepend_path(monkeypatch, bin_dir)
    monkeypatch.setenv("FAKE_SOFFICE_SRC_PPTX", str(fixture_pptx))

    src = tmp_path / "deck.ppt"
    src.write_bytes(b"legacy binary ppt bytes, irrelevant to the fake")

    res = ppt.extract(src, tmp_path / "assets")

    assert res.status == "processed"
    assert res.extractor == "ppt-soffice+pptx"
    assert "Converted slide text" in res.markdown
    assert res.error is None
    assert any("converted from legacy .ppt to .pptx via LibreOffice" in n for n in res.notes)


def test_conversion_note_omits_the_machine_specific_build_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """AGENTS.md rule 7 (determinism): ``soffice --version``'s trailing
    token is a per-build git hash, not stable across two installs of the
    same release, so it must not land in the note body — only the
    product name and release number may."""
    fixture_pptx = tmp_path / "fixture.pptx"
    _write_real_pptx(fixture_pptx)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_soffice(
        bin_dir,
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--version\" ]; then\n"
        "  echo 'LibreOffice 25.8.1.1 54047653041915e595ad4e45cccea684809c77b5'\n"
        "  exit 0\n"
        "fi\n" + _FAKE_SOFFICE_COPY_SCRIPT.removeprefix("#!/bin/sh\n"),
    )
    _prepend_path(monkeypatch, bin_dir)
    monkeypatch.setenv("FAKE_SOFFICE_SRC_PPTX", str(fixture_pptx))

    src = tmp_path / "deck.ppt"
    src.write_bytes(b"legacy binary ppt bytes, irrelevant to the fake")

    res = ppt.extract(src, tmp_path / "assets")

    assert res.status == "processed"
    conversion_note = next(n for n in res.notes if "converted from legacy .ppt" in n)
    assert "LibreOffice 25.8.1.1" in conversion_note
    assert "54047653041915e595ad4e45cccea684809c77b5" not in conversion_note


# -------------------------------------------------------------- cleanup

def test_temp_dir_is_removed_after_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture_pptx = tmp_path / "fixture.pptx"
    _write_real_pptx(fixture_pptx)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_soffice(bin_dir, _FAKE_SOFFICE_COPY_SCRIPT)
    _prepend_path(monkeypatch, bin_dir)
    monkeypatch.setenv("FAKE_SOFFICE_SRC_PPTX", str(fixture_pptx))

    created: list[Path] = []
    orig_mkdtemp = tempfile.mkdtemp

    def spy_mkdtemp(*, prefix: str | None = None) -> str:
        d = orig_mkdtemp(prefix=prefix)
        created.append(Path(d))
        return d

    monkeypatch.setattr(tempfile, "mkdtemp", spy_mkdtemp)

    src = tmp_path / "deck.ppt"
    src.write_bytes(b"irrelevant")
    res = ppt.extract(src, tmp_path / "assets")

    assert res.status == "processed"
    assert created, "mkdtemp was never called"
    assert not created[0].exists()


def test_temp_dir_is_removed_after_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_soffice(bin_dir, "#!/bin/sh\nexit 1\n")
    _prepend_path(monkeypatch, bin_dir)

    created: list[Path] = []
    orig_mkdtemp = tempfile.mkdtemp

    def spy_mkdtemp(*, prefix: str | None = None) -> str:
        d = orig_mkdtemp(prefix=prefix)
        created.append(Path(d))
        return d

    monkeypatch.setattr(tempfile, "mkdtemp", spy_mkdtemp)

    src = tmp_path / "deck.ppt"
    src.write_bytes(b"garbage")
    res = ppt.extract(src, tmp_path / "assets")

    assert res.status == "manual_review"
    assert created, "mkdtemp was never called"
    assert not created[0].exists()


# ------------------------------------------------------- never touches raw

def test_extract_never_writes_to_src_parent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Guards the archive/raw immutability rule: the source directory must
    contain exactly the one file we put there, before and after."""
    fixture_pptx = tmp_path / "fixture.pptx"
    _write_real_pptx(fixture_pptx)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    src = raw_dir / "deck.ppt"
    src.write_bytes(b"irrelevant")
    before = sorted(p.name for p in raw_dir.iterdir())

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_soffice(bin_dir, _FAKE_SOFFICE_COPY_SCRIPT)
    _prepend_path(monkeypatch, bin_dir)
    monkeypatch.setenv("FAKE_SOFFICE_SRC_PPTX", str(fixture_pptx))

    ppt.extract(src, tmp_path / "assets")

    after = sorted(p.name for p in raw_dir.iterdir())
    assert before == after == ["deck.ppt"]
