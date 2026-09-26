"""The ``secret-leak`` sweep check: credentials already sitting in the vault.

Redaction at the write boundaries stops new credentials entering. It says
nothing about what predates it, and nothing about a note typed straight into
Obsidian, which never passes a writer at all. So this reads what is actually
on disk.

It reports a file, a line number and a kind, and never the matched value. A
report that quotes the secret has only moved it: `sweep-report.md` is written
on every maintenance run.
"""
from __future__ import annotations

from pathlib import Path

from .secrets import _VENDOR_REDACTIONS, _denylist
from .sweep_common import Finding, _read_text

# Where a credential would be both durable and reachable: notes, the processed
# copies of every ingested document, and the index whose summaries are
# git-tracked and readable over MCP. `archive/raw/` is deliberately absent —
# it is immutable by AGENTS.md and already in git history, so it is the
# retroactive scan's business to report on, not this check's to nag about
# every run.
_SCANNED = ("knowledge", "archive/processed", "metadata")

# Only the unambiguous shapes. The generic length-and-alphabet fallbacks would
# flag every git SHA in the vault on every maintenance run, which is how a
# check gets ignored.
_SUFFIXES = {".md", ".jsonl", ".txt"}


def secret_kinds_in_line(line: str, denylist: list[str]) -> list[str]:
    """The sorted, de-duplicated set of credential-shape labels found in
    one line: vendor pattern labels, plus ``"known-secret"`` when a
    denylist value appears."""
    kinds = {label for label, pat in _VENDOR_REDACTIONS if pat.search(line)}
    if any(value in line for value in denylist):
        kinds.add("known-secret")
    return sorted(kinds)


def _check_secret_leak(root: Path) -> list[Finding]:
    """secret-leak: an unambiguous credential shape sitting in a vault file."""
    findings: list[Finding] = []
    denylist = _denylist()

    for area in _SCANNED:
        base = root / area
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _SUFFIXES:
                continue
            text = _read_text(path)
            if text is None:
                continue
            rel = path.relative_to(root).as_posix()
            for lineno, line in enumerate(text.splitlines(), start=1):
                kinds = secret_kinds_in_line(line, denylist)
                if not kinds:
                    continue
                findings.append(Finding(
                    category="secret-leak",
                    path=rel,
                    detail=(
                        f"line {lineno}: {', '.join(kinds)} "
                        f"— rotate the credential, then redact the file"
                    ),
                ))
    return findings
