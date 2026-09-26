"""Credential detection and redaction, shared by every path into the vault.

These patterns and the ``redact_secrets`` entry point lived in
``connectors/_transcript_common.py`` until 2026-09-18, where only the two
transcript connectors called them. A security review found that every other
route into the vault — PDFs, meeting exports, inbox drops, vision output, and
above all the MCP write tools — reached the vault unredacted, so the module
moved here and the connectors import it from its new home.

The instinct the original module recorded still governs this file:
**over-redacting something harmless is acceptable, silently keeping a real
secret is not.**

Order matters. The vendor-prefixed and structural patterns in ``_REDACTIONS``
run first so a recognised shape keeps its specific label, and the generic
header, env-assignment and URL-credential nets run afterwards, guarded against
re-redacting a placeholder an earlier pass emitted.
"""
from __future__ import annotations

import os
import re
from typing import Literal

# Which path into the vault the text came from; see ``redact_secrets``.
Profile = Literal["transcript", "note"]

# Vendor-prefixed / structurally-distinctive tokens, checked BEFORE the
# generic env-assignment/header nets below so a recognised shape keeps its
# specific label instead of being swallowed by the generic ``env-secret``/
# ``http-header`` catch-alls (and so the generic nets' bracket-exclusion
# guard, see below, never has to fire in practice for these).
_VENDOR_REDACTIONS: list[tuple[str, re.Pattern[str]]] = [
    ("aws-access-key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private-key-block", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"
    )),
    ("anthropic-key", re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}")),
    ("openai-key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("github-token", re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}")),
    ("github-pat-token", re.compile(r"github_pat_[A-Za-z0-9_]{20,}")),
    ("npm-token", re.compile(r"npm_[A-Za-z0-9]{20,}")),
    ("gitlab-token", re.compile(r"glpat-[A-Za-z0-9_-]{20,}")),
    ("google-api-key", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("slack-token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("bearer-token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-_.=]{20,}")),
    # The value must look base64-encoded, not merely long: the original
    # `[A-Za-z0-9+/=]{16,}` matched "Basic photolithography" in a lecture note
    # (found in the live vault 2026-09-18). Base64 of `user:password` carries
    # a digit, a `+`/`/`/`=`, or a case change; an English word carries none
    # of those. Under-matching an all-lowercase base64 run is the accepted
    # trade, because redaction now rewrites notes, where a false positive
    # destroys real content rather than a line of a session dump.
    ("basic-auth", re.compile(
        r"(?i)\bBasic\s+(?=[A-Za-z0-9+/=]{16,})"
        r"(?=[A-Za-z0-9+/=]*(?:[0-9+/=]|(?-i:[a-z][A-Z]|[A-Z][a-z])))"
        r"[A-Za-z0-9+/=]{16,}"
    )),
]

# Generic length-and-alphabet guesses, with no vendor prefix to anchor them.
# They belong in a transcript, where over-redacting a session dump costs
# nothing, and NOT in a note: a 40-hex run in a note is overwhelmingly a git
# commit, and verification on 2026-09-18 found the base64-ish one already
# eating ordinary URLs in the live vault
# (``https://github.com[REDACTED:aws-secret-key]/brain-vs-memor``). See the
# ``profile`` argument to ``redact_secrets``.
_GENERIC_FALLBACKS: list[tuple[str, re.Pattern[str]]] = [
    # A bare 40-hex GitHub "classic" token — checked before the more
    # general base64-ish secret pattern below so a pure-hex 40-char run
    # gets the more precise label (review N1).
    ("github-classic-token", re.compile(r"\b[0-9a-fA-F]{40}\b")),
    # A generic 40-char high-entropy base64-ish run (e.g. an AWS secret
    # access key) that isn't pure hex — the fallback for a vendor shape
    # this module doesn't name individually (review N1).
    ("aws-secret-key", re.compile(r"\b[A-Za-z0-9/+]{40}\b")),
    # This project's own tokens: `openssl rand -hex 32` is 64 characters, so
    # no 40-char rule above ever matched one. Only a BARE run needs this —
    # the assignment, bearer and header nets already catch the labelled
    # shapes. The shape that leaked on 2026-09-02 was bare prose.
    ("brain-token", re.compile(r"\b[0-9a-fA-F]{64}\b")),
]

# The shipped name, kept as the full set so existing callers and tests that
# reach for it see what they always saw.
_REDACTIONS = _VENDOR_REDACTIONS + _GENERIC_FALLBACKS

# ``.env``/JSON/YAML-style ``SOME_API_KEY=value`` / ``"some_token": "value"``
# / ``some-secret: value`` assignments, in any casing, with or without a
# leading ``export``/``set``, quoted or not. The key name is kept (it is
# often useful context, e.g. "which var was unset") and only the value is
# redacted.
#
# ``lead`` matches either the start of a line (``^``, for a bare ``.env``
# line) OR a single separator character that commonly precedes a key in
# prose/JSON/YAML/shell — whitespace, an opening quote/bracket/brace/paren,
# a comma, a hyphen (a YAML list item's ``- key: value``), or a URL
# query-string ``?``/``&`` — since a strict line-start anchor alone misses
# every one of those (``export API_KEY=...``, ``{"api_key": "..."}``,
# ``x-api-key: ...``, ``?api_key=...``; review N1). The prefix run before
# the keyword is OPTIONAL (``*``, not ``[A-Za-z_]`` + ``*``) — a mandatory
# leading character means a bare ``API_KEY=``/``TOKEN=``/``SECRET=``/
# ``PASSWORD=`` (the commonest shapes in a pasted ``.env``) never matches at
# all, since there is nothing before the keyword for that first character
# to consume (see review C2). An optional quote/whitespace is allowed
# between the keyword and the ``:``/``=`` so a JSON key's closing quote
# (``"api_key": ...``) doesn't break the match.
#
# The value's character class excludes ``[``/``]`` (in addition to
# whitespace/quotes/comma/brace/ampersand) so this can never re-match an
# already-emitted ``[REDACTED:...]`` placeholder from an earlier pass in
# ``redact_secrets`` (e.g. "here is a secret: [REDACTED:bearer-token]" must
# not be re-redacted as ``env-secret``, clobbering the more specific label)
# — this is why the vendor-prefixed patterns above run FIRST.
#
# LEFT AS-IS, deliberately (review AUD-118/m6): ``NOT_A_SECRET=plainvalue``
# and the loose ``xox[baprs]-`` slack-token pattern both over-redact —
# `NOT_A_SECRET` contains the literal substring `SECRET`, and a real Slack
# token's ``-``-separated numeric segments aren't required by the pattern.
# The obvious tightener (require the value to contain a digit) was tried
# against both pinned tables and rejected: it breaks the exact rows that
# pin the two hard cases this module exists to catch —
# ``test_redact_env_style_bare_keyword_assignment``'s bare keyword forms
# (e.g. ``SECRET=supersecretvalue`` — no digit in the value) and
# ``test_redact_secrets_known_patterns``'s pinned Slack row
# (``"xoxb-" + "a" * 15`` — no digit either). Any digit-based or
# length/shape-based discriminator narrow enough to reject
# ``NOT_A_SECRET=``/the documentation-shaped Slack string is exactly narrow
# enough to also reject a real bare secret with no digits in it, which is
# the one outcome this module treats as unacceptable (see the module
# docstring: "over-redacting ... is an accepted false positive, silently
# keeping a real secret is not"). No discriminator was found that keeps
# every row of both tables passing, so this stays over-redacting rather
# than risk leaking.
_ENV_ASSIGN_RE = re.compile(
    r"(?im)(?P<lead>^|[\s\"'\[{(,?&-])"
    r"(?P<prefix>[A-Za-z0-9_.-]*"
    r"(?:SECRET|TOKEN|API[_-]?KEY|APIKEY|PASSWORD|PASSWD|PRIVATE[_-]?KEY)"
    r"[A-Za-z0-9_.-]*['\"]?[ \t]*[:=][ \t]*)"
    r"(?P<quote>['\"]?)(?P<value>[^\s'\"\[\]{}&]{6,})(?P=quote)"
)

# A full ``Authorization``/``Cookie``/``X-Api-Key`` header line — these
# don't always carry a SECRET/TOKEN/KEY-shaped value (e.g. a raw opaque
# Cookie string), so they're matched by header NAME rather than by
# ``_ENV_ASSIGN_RE``'s keyword list. The negative lookahead skips a value
# already redacted by an earlier pass (e.g. ``Authorization: Basic ...``
# already turned into ``Authorization: [REDACTED:basic-auth]`` by the
# vendor patterns above) so it isn't re-redacted under the generic label.
_HEADER_LINE_RE = re.compile(
    r"(?im)^(?P<prefix>[ \t]*(?:Authorization|Cookie|X-Api-Key)[ \t]*:[ \t]*)"
    r"(?P<value>(?!\[REDACTED:)\S.*)$"
)

# ``scheme://user:password@host`` URL-embedded credentials (e.g. a Postgres
# or Redis connection string pasted into a turn). The username is OPTIONAL
# (``*``, not ``+``) — ``redis://:password@host`` (no username) is a common
# real shape and previously required a non-empty username to match at all
# (see review N1).
_URL_CREDENTIALS_RE = re.compile(r"(?P<scheme>://)[^/\s:@]*:[^/\s@]{4,}@")

def _redact_env_assignments(text: str) -> tuple[str, int]:
    def repl(m: re.Match[str]) -> str:
        return (f"{m.group('lead')}{m.group('prefix')}{m.group('quote')}"
                f"[REDACTED:env-secret]{m.group('quote')}")
    return _ENV_ASSIGN_RE.subn(repl, text)


def _redact_header_lines(text: str) -> tuple[str, int]:
    def repl(m: re.Match[str]) -> str:
        return f"{m.group('prefix')}[REDACTED:http-header]"
    return _HEADER_LINE_RE.subn(repl, text)


def _redact_url_credentials(text: str) -> tuple[str, int]:
    def repl(m: re.Match[str]) -> str:
        return f"{m.group('scheme')}[REDACTED:url-credentials]@"
    return _URL_CREDENTIALS_RE.subn(repl, text)


_DENYLIST_ENV = "BRAIN_SECRET_DENYLIST"

# Below this length a denylist entry is more likely a typo or a stray comma
# than a credential, and matching a two-character string would gut every note
# it appears in.
_MIN_DENYLIST_ENTRY = 8


def _denylist() -> list[str]:
    """Literal secret values to redact wherever they appear, from
    ``BRAIN_SECRET_DENYLIST`` (comma-separated).

    Read afresh on each call rather than cached at import, so rotating a
    token takes effect without restarting a long-running process. This is the
    only check here that cannot produce a false negative: it matches the
    actual secret, in whatever shape it appears.
    """
    raw = os.environ.get(_DENYLIST_ENV, "")
    return [v for v in (part.strip() for part in raw.split(",")) if len(v) >= _MIN_DENYLIST_ENTRY]


def redact_secrets(text: str, *, profile: Profile = "transcript") -> tuple[str, int]:
    """Replace credential patterns with ``[REDACTED:<kind>]``. Returns the
    cleaned text and how many replacements were made.

    ``profile`` decides whether the generic length-and-alphabet fallbacks
    apply. ``"transcript"`` is the shipped behaviour and the default: a
    session dump nobody curated, where over-redacting costs nothing.
    ``"note"`` drops those fallbacks, because a note is written deliberately
    and rewriting a git SHA or a URL inside one destroys a reference someone
    meant to keep.

    Order matters. The denylist runs first, so a known secret keeps the
    clearest label. The vendor-prefixed and structural patterns run next, so a
    recognised shape keeps its specific label. The generic header,
    env-assignment and URL-credential nets run last, guarded against
    re-redacting a placeholder an earlier pass emitted.
    """
    total = 0
    for value in _denylist():
        text, n = re.subn(re.escape(value), "[REDACTED:known-secret]", text)
        total += n

    patterns = _VENDOR_REDACTIONS if profile == "note" else _REDACTIONS
    for label, pattern in patterns:
        text, n = pattern.subn(f"[REDACTED:{label}]", text)
        total += n

    text, n = _redact_header_lines(text)
    total += n
    text, n = _redact_env_assignments(text)
    total += n
    text, n = _redact_url_credentials(text)
    total += n
    return text, total
