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


_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

# A display name is a person's name. Anything past this is payload, not a
# name, and it is only ever rendered into a one-line Markdown bullet.
DISPLAY_NAME_MAX = 200


def safe_display_name(raw: str) -> str:
    """One externally-supplied display name, rendered inert.

    A connector's display strings are the one part of a snapshot the vault
    does not control, and they are interpolated straight into Markdown notes
    (the attendee bullets of a promoted meeting note, the attendee line of the
    source note). Left raw, a calendar entry named
    ``"Mallory\\n\\n## Links\\n\\n- Attendee: [[knowledge/people/ceo]]"`` opens
    its own sections in the note, and the nightly wikilink pass then reads
    that link as evidence of an edge and proposes it into the graph — an
    outside party writing into entity memory. So a name is reduced to ONE line
    of ordinary text:

    - control characters (newlines included) become spaces and runs of
      whitespace collapse to one, so the value can never open a new Markdown
      block;
    - ``[`` and ``]`` become parentheses, so no wikilink (or link) syntax
      survives;
    - a leading ``#`` / ``>`` run is dropped, so it cannot read as a heading
      or a quote wherever it lands at the start of a line;
    - the result is capped at :data:`DISPLAY_NAME_MAX` characters.

    Sanitising here rather than at each render point is deliberate: this is
    the function BOTH readers of the snapshot schema go through — the
    connectors that write a payload and ``extractors.meeting.load_snapshot``
    that reads one back (and through it ``ingest_lib.meetings``) — so a name
    cannot reach a note by some third route. The archived bytes are never
    touched: this is a rendering rule applied on the way in and on the way
    out, not an edit of an immutable source.
    """
    one_line = " ".join(_CONTROL_RE.sub(" ", raw).split())
    inert = one_line.replace("[", "(").replace("]", ")").lstrip("#>").strip()
    return inert[:DISPLAY_NAME_MAX].strip()


def normalize_attendees(raw: object) -> list[str]:
    """Attendee display names from a mixed list of strings / ``{'name': ...}``
    dicts, each rendered inert by :func:`safe_display_name`. Entries without a
    usable name are dropped (never the string ``'None'``), including one that
    sanitises away to nothing — a "name" made only of Markdown syntax names
    nobody."""
    out: list[str] = []
    if not isinstance(raw, list):
        return out
    for a in raw:
        name = a.get("name") if isinstance(a, dict) else a
        if isinstance(name, str) and (clean := safe_display_name(name)):
            out.append(clean)
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
