"""F16/F8 scan side: knowledge/assistant/archive/ holds promoted facts whose
content already lives in the entity note they were folded into, so the scan
must skip that subtree or the fact becomes double-retrievable. inbox/ and
PROFILE.md under assistant/ stay scanned. No embedding model is loaded."""
from __future__ import annotations

import logging
from pathlib import Path

from ingest_lib.config import VaultPaths, paths_for_root
from ingest_lib.knowledge import scan_knowledge

_LOG = logging.getLogger("test")

_NOTE = (
    "---\ntitle: fact\n---\n\n"
    "A consolidated fact note with enough body text to chunk and embed "
    "cleanly into the index for retrieval later on.\n"
)


def _vault(tmp_path: Path) -> VaultPaths:
    paths = paths_for_root(tmp_path)
    paths.ensure()
    return paths


def _write(paths: VaultPaths, rel: str, text: str) -> None:
    p = paths.root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_scan_knowledge_excludes_assistant_archive(tmp_path: Path) -> None:
    paths = _vault(tmp_path)
    # Live assistant notes ARE scanned; their archived twins are NOT.
    _write(paths, "knowledge/assistant/inbox/fact-001.md", _NOTE)
    _write(paths, "knowledge/assistant/PROFILE.md", _NOTE)
    _write(paths, "knowledge/assistant/digests/2026-06.md", _NOTE)
    _write(paths, "knowledge/assistant/archive/2026/fact-001.md", _NOTE)
    _write(paths, "knowledge/assistant/archive/fact-002.md", _NOTE)

    scan = scan_knowledge(paths, logger=_LOG)
    rels = {r.relative_path for r in scan.records}

    assert "knowledge/assistant/inbox/fact-001.md" in rels
    assert "knowledge/assistant/PROFILE.md" in rels
    assert "knowledge/assistant/digests/2026-06.md" in rels
    # The whole archive/ subtree is excluded from the record set.
    assert not any(r.startswith("knowledge/assistant/archive/") for r in rels)
    assert scan.read_failures == 0
