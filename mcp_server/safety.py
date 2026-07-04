"""Path resolution and allow/deny enforcement.

Every tool that touches the filesystem goes through ``resolve_safe()``
or one of its specialised variants. The contract is:

- The resolved absolute path lies inside the vault root.
- The resolved path is not a symlink (or points only inside the vault).
- For writes, the path lies under one of the allow-prefixes.
- For reads, the path is not under a deny-prefix and the basename is
  not on the deny-names list.

Any violation raises ``SafetyError`` which the tool handler converts
to a user-visible MCP error. Stack traces never reach the agent.
"""
from __future__ import annotations

from pathlib import Path

from .config import (
    CONCEPT_WRITE_PREFIX,
    DENY_READ_NAMES,
    DENY_READ_PREFIXES,
    READ_ALLOW_PREFIXES,
    READ_ALLOW_ROOT_FILES,
    WRITE_ALLOW_PREFIXES,
)


class SafetyError(Exception):
    """Raised when a requested path violates the safety policy."""


def _resolve_inside_vault(vault_root: Path, rel: str) -> Path:
    """Resolve ``rel`` against ``vault_root`` and confirm it stays inside.

    Rejects absolute paths, ``..`` escapes, and symlinks that point outside
    the vault. The path doesn't need to exist; the parent does.
    """
    if not rel or rel.startswith("/"):
        raise SafetyError(f"path must be relative to the vault root, got: {rel!r}")
    # Reject ALL control characters, not just NUL. Newlines/tabs in a
    # path are valid POSIX but they survive into git commit messages
    # (forever, on a remote anyone can clone) and into log lines that
    # parsers can misinterpret. Also reject the Unicode separators that
    # git/web views render as line breaks (NEL, LINE/PARAGRAPH SEPARATOR)
    # and the BOM. The vault has no legitimate need for any of these.
    _unicode_line_breakers = {0x0085, 0x2028, 0x2029, 0xFEFF}
    for ch in rel:
        if ord(ch) < 0x20 or ord(ch) == 0x7F or ord(ch) in _unicode_line_breakers:
            raise SafetyError(
                f"path contains a control character (U+{ord(ch):04X})"
            )

    candidate = (vault_root / rel)
    try:
        # strict=False because the file may not exist yet (write tools)
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise SafetyError(f"path resolution failed: {exc}") from None

    # Reject any symlinks anywhere along the chain that leave the vault.
    # ``is_relative_to`` was added in Python 3.9; we require 3.12.
    if not resolved.is_relative_to(vault_root):
        raise SafetyError(f"path escapes the vault root: {rel!r}")

    # If any existing parent is a symlink, refuse — agents shouldn't have
    # to reason about symlink chasing, and neither should we.
    for parent in candidate.parents:
        if parent == vault_root:
            break
        if parent.is_symlink():
            raise SafetyError(f"path traverses a symlink at {parent.name!r}; refused")
    if candidate.exists() and candidate.is_symlink():
        raise SafetyError(f"target itself is a symlink: {rel!r}")

    return resolved


def resolve_read(vault_root: Path, rel: str) -> Path:
    """Resolve ``rel`` for a READ operation (allowlist + deny, defense in depth).

    The path must sit under one of READ_ALLOW_PREFIXES or be one of
    READ_ALLOW_ROOT_FILES; the DENY_* lists apply on top. Every refusal
    returns the SAME opaque message so a probing agent can't tell
    "exists but denied" from "absent" and map the filesystem.
    """
    resolved = _resolve_inside_vault(vault_root, rel)
    posix = resolved.relative_to(vault_root).as_posix()
    # Case-fold the DENY checks. On case-insensitive filesystems (macOS
    # APFS, Windows) ``Path.resolve()`` preserves the *requested* casing in
    # the name/path while still opening the real file, so an exact-case deny
    # test is bypassable (e.g. ``metadata/EMBEDDINGS_META.jsonl`` slipping
    # past ``embeddings_meta.jsonl``). The DENY_* constants are lowercase, so
    # casefolding the candidate closes the bypass. The ALLOW checks stay
    # case-sensitive so they fail closed on a case-sensitive FS (Linux).
    posix_cf = posix.casefold()
    name_cf = resolved.name.casefold()

    allowed = posix in READ_ALLOW_ROOT_FILES or any(
        posix == p or posix.startswith(p + "/") for p in READ_ALLOW_PREFIXES
    )
    if not allowed:
        raise SafetyError("not found or not readable")
    for deny in DENY_READ_PREFIXES:
        if posix_cf == deny or posix_cf.startswith(deny + "/"):
            raise SafetyError("not found or not readable")
    if name_cf in DENY_READ_NAMES:
        raise SafetyError("not found or not readable")
    return resolved


def resolve_write_under_allowlist(vault_root: Path, rel: str) -> Path:
    """Resolve ``rel`` for a WRITE operation under the standard allowlist."""
    resolved = _resolve_inside_vault(vault_root, rel)
    posix = resolved.relative_to(vault_root).as_posix()

    for allow in WRITE_ALLOW_PREFIXES:
        if posix == allow:
            raise SafetyError(
                f"refusing to write to bare directory {allow!r}; "
                "provide a file path under it"
            )
        if posix.startswith(allow + "/"):
            return resolved
    raise SafetyError(
        f"write denied: path must live under one of {list(WRITE_ALLOW_PREFIXES)}, "
        f"got {posix!r}"
    )


def resolve_write_concept(vault_root: Path, slug: str) -> Path:
    """Resolve a concept-note path from its slug (filename without .md)."""
    if not slug or "/" in slug or slug.startswith(".") or len(slug) > 128:
        raise SafetyError(f"invalid concept slug: {slug!r}")
    rel = f"{CONCEPT_WRITE_PREFIX}/{slug}.md"
    return _resolve_inside_vault(vault_root, rel)


def resolve_inbox(vault_root: Path, rel: str) -> Path:
    """Resolve a path under ``inbox/`` for the drop_inbox_file tool.

    Critically, the resolved path must stay *under* ``inbox/``. Without
    this check a path like ``../archive/processed/x`` resolves inside the
    vault (so ``_resolve_inside_vault`` accepts it) but lands in a
    ground-truth layer the agent must never write to.
    """
    if rel.startswith("inbox/"):
        rel = rel[len("inbox/"):]
    full_rel = f"inbox/{rel}"
    resolved = _resolve_inside_vault(vault_root, full_rel)
    posix = resolved.relative_to(vault_root).as_posix()
    if posix != "inbox" and not posix.startswith("inbox/"):
        raise SafetyError("inbox path must stay under inbox/")
    return resolved
