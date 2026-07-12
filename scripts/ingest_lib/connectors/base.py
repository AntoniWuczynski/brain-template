"""The connector contract: a source-native ``pull()`` that yields snapshots.

A *connector* is the one non-deterministic edge of the vault: it reaches an
external source (Granola, a mail server, a Git host) and turns each item into
a :class:`Snapshot` — the exact bytes to archive, plus the identity needed to
skip it next time. Everything downstream is the existing deterministic
pipeline: a snapshot lands in ``inbox/<source_class>/``, gets copied to the
immutable ``archive/raw/`` and extracted like any other source.

Connectors carry NO extraction logic — that lives in a matching extractor
registered by source-class prefix (see ``extractors.dispatch_extractor``), so
a re-ingest of an archived snapshot is fully offline and deterministic.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from collections.abc import Iterator

if TYPE_CHECKING:
    from .state import ConnectorState

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def meeting_filename(date: str, title: str, native_id: str) -> str:
    """Collision-safe inbox filename ``<date>-<slug>-<8hex(id)>.json``. The
    id hash guarantees two distinct meetings never share a filename even with
    the same date and title (which would otherwise silently overwrite one)."""
    slug = _SLUG_RE.sub("-", title.strip().lower()).strip("-") or "meeting"
    sid = hashlib.sha1(native_id.encode("utf-8")).hexdigest()[:8]  # noqa: S324 - filename tag, not security
    return f"{date or 'undated'}-{slug}-{sid}.json"


def normalize_attendees(raw: object) -> list[str]:
    """Attendee display names from a mixed list of strings / ``{'name': ...}``
    dicts. Entries without a usable name are dropped (never the string
    ``'None'``)."""
    out: list[str] = []
    if not isinstance(raw, list):
        return out
    for a in raw:
        name = a.get("name") if isinstance(a, dict) else a
        if isinstance(name, str) and name.strip():
            out.append(name.strip())
    return out


@dataclass(frozen=True)
class Snapshot:
    """One external item, captured as bytes ready for the archive.

    ``payload`` must contain EVERYTHING the extractor needs offline — the
    fetch is the only networked step and must not have to be repeated. The
    ``(native_id, content_hash)`` pair is the idempotency key: an unchanged
    item is skipped before it ever touches ``inbox/``.
    """

    source_class: str      # e.g. "meetings/granola" -> inbox/meetings/granola/
    native_id: str         # stable source-native id (meeting id, message id, ...)
    filename: str          # deterministic leaf name derived from the payload
    payload: bytes         # exact bytes to write; fully offline-extractable

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    @property
    def inbox_relpath(self) -> str:
        """Vault-relative destination, e.g. ``inbox/meetings/granola/x.json``."""
        cls = self.source_class.strip("/")
        return f"inbox/{cls}/{self.filename}"


@runtime_checkable
class Connector(Protocol):
    """A named source that yields snapshots. ``pull`` may read ``state`` (its
    own prior cursor/entries) to fetch incrementally, but correctness never
    depends on it — the runner re-checks every yielded snapshot against the
    state, so a connector that re-yields everything is merely slower, never
    wrong."""

    name: str

    def pull(self, state: ConnectorState) -> Iterator[Snapshot]: ...
