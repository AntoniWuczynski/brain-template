"""Common types for extractors."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from collections.abc import Callable

_BACKTICK_RUN = re.compile(r"`+")


def fence(content: str, info: str = "") -> str:
    """Wrap ``content`` in a Markdown code fence long enough not to be
    closed early by backtick runs inside it.

    A fixed ```` ``` ```` fence breaks the moment the content contains its
    own ```` ``` ```` (guaranteed for most Markdown/notebook sources): the
    fence closes at the first inner run and the remainder spills out as raw
    Markdown. Per CommonMark, the fence must be longer than any backtick run
    it encloses, so use ``max(3, longest_run + 1)`` backticks.
    """
    longest = max((len(m.group(0)) for m in _BACKTICK_RUN.finditer(content)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{info}\n{content}\n{ticks}"


@dataclass(frozen=True)
class ExtractionResult:
    """The output of one extractor invocation.

    - ``status="processed"`` — full content extracted.
    - ``status="partial"``  — some content extracted, some lost.
    - ``status="manual_review"`` — extraction failed; ``markdown`` may be empty.
    """

    status: Literal["processed", "partial", "manual_review"]
    extractor: str
    markdown: str
    assets: list[Path] = field(default_factory=list)
    error: str | None = None
    notes: list[str] = field(default_factory=list)


# An extractor takes the source path plus a directory it may write
# auxiliary assets into (figures, tables...). It must never modify
# ``src`` or read outside of it.
Extractor = Callable[[Path, Path], ExtractionResult]
