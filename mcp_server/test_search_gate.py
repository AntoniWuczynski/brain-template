"""Tests for the search-hit read-gate path mapping.

A search hit's content must only be returned if the agent could read the
backing artifact directly. Ingested sources are backed by their processed
markdown under ``archive/processed/``; knowledge notes are their own
backing artifact and must be gated on their own path.

Run with::

    uv run python -m mcp_server.test_search_gate
"""
from __future__ import annotations

from mcp_server.tools import _hit_gate_path


def main() -> int:
    failures: list[str] = []

    def expect(label: str, got: str, want: str) -> None:
        if got == want:
            print(f"  PASS  {label}")
        else:
            failures.append(label)
            print(f"  FAIL  {label}  (got {got!r}, want {want!r})")

    print("\n[search gate path]")
    # Ingested source: gate on its processed artifact.
    expect(
        "source hit maps to archive/processed twin",
        _hit_gate_path("university/COMP0023/04_error_coding.pdf", "pdf-mineru"),
        "archive/processed/university/COMP0023/04_error_coding.md",
    )
    # Knowledge note (origin says so): it IS the artifact, gate on it.
    expect(
        "knowledge-note hit gates on its own path",
        _hit_gate_path("knowledge/projects/brain/brain.md", "knowledge-note"),
        "knowledge/projects/brain/brain.md",
    )
    # The collision case: an ingested source dropped at inbox/knowledge/x.pdf
    # is LABELLED knowledge/x.pdf but is NOT a vault note — origin, not the
    # path prefix, decides; its backing artifact is the processed twin.
    expect(
        "ingested source labelled knowledge/ maps to processed twin",
        _hit_gate_path("knowledge/x.pdf", "pdf-pypdf-fallback"),
        "archive/processed/knowledge/x.md",
    )
    # Legacy index rows (built before origin existed) degrade to the old
    # prefix heuristic rather than breaking until the next rebuild.
    expect(
        "legacy row without origin falls back to prefix heuristic",
        _hit_gate_path("knowledge/projects/brain/brain.md", ""),
        "knowledge/projects/brain/brain.md",
    )

    print(f"\nresults: {4 - len(failures)} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
