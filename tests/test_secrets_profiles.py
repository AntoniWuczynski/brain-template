"""Redaction profiles: what each path into the vault is allowed to rewrite.

Transcripts and notes want different answers. A transcript is a dump of a
session nobody curated, so over-redacting it costs nothing and the generic
40-character fallbacks earn their place. A note is written deliberately, by a
person or an agent, and quietly rewriting a git SHA or a URL inside one
corrupts a reference someone meant to keep — so the note profile drops the
generic fallbacks and keeps the shapes that are unambiguously credentials.
"""
from __future__ import annotations

import pytest

from ingest_lib.secrets import Profile, redact_secrets

_TOKEN = "95b0471744ac6d3e8f2a1c5b9e7d4a6f0c8b2e5a1d7f3b9c4e6a8d0f2b5c7e91"
_GIT_SHA = "e83c5163316f89bfbde7d9ab23ca2e25604af290"


@pytest.mark.parametrize("profile", ["transcript", "note"])
@pytest.mark.parametrize(
    "text",
    [
        "AKIA1234567890ABCDEF",
        "sk-ant-api03-aaaaaaaaaaaaaaaaaaaaaaaa",
        "ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "Bearer " + _TOKEN,
        "BRAIN_MCP_TOKENS=" + _TOKEN,
        'x-api-key: ' + _TOKEN,
    ],
)
def test_unambiguous_credentials_are_redacted_in_both_profiles(
    profile: Profile, text: str
) -> None:
    out, n = redact_secrets(text, profile=profile)
    assert n >= 1, (profile, text)
    assert _TOKEN not in out
    assert "AKIA1234567890ABCDEF" not in out


def test_transcript_profile_redacts_a_bare_token_in_prose() -> None:
    """The 2026-09-02 leak shape: a token discussed rather than assigned."""
    out, n = redact_secrets(f"the token is {_TOKEN} and it works", profile="transcript")
    assert n == 1
    assert _TOKEN not in out


def test_note_profile_leaves_a_git_sha_alone() -> None:
    """A 40-hex run in a note is overwhelmingly a commit, not a credential."""
    text = f"Fixed in {_GIT_SHA}, see the push queue."
    out, n = redact_secrets(text, profile="note")
    assert out == text
    assert n == 0


def test_transcript_profile_still_redacts_a_40_hex_run() -> None:
    """Unchanged from the shipped behaviour — transcripts stay aggressive."""
    out, n = redact_secrets(f"key {_GIT_SHA}", profile="transcript")
    assert n == 1
    assert _GIT_SHA not in out


def test_note_profile_leaves_urls_alone() -> None:
    """Verification found the generic fallback eating real URLs in the vault."""
    text = "See https://github.com/AntoniWuczynski/brain-vs-memor for the comparison."
    out, n = redact_secrets(text, profile="note")
    assert out == text
    assert n == 0


def test_a_live_token_is_caught_by_value_in_both_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """The denylist is the one check with no false negatives: it matches the
    actual secret, whatever shape it appears in."""
    odd = "hunter2-not-a-recognised-shape"
    monkeypatch.setenv("BRAIN_SECRET_DENYLIST", odd)
    profiles: tuple[Profile, ...] = ("transcript", "note")
    for profile in profiles:
        out, n = redact_secrets(f"the value is {odd}.", profile=profile)
        assert n == 1, profile
        assert odd not in out


def test_denylist_ignores_blank_and_short_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stray empty entry must not turn into a match on every character."""
    monkeypatch.setenv("BRAIN_SECRET_DENYLIST", " , ,abc, ")
    out, n = redact_secrets("abc and some ordinary text", profile="note")
    assert out == "abc and some ordinary text"
    assert n == 0


def test_default_profile_is_the_shipped_transcript_behaviour() -> None:
    out_default, n_default = redact_secrets(f"key {_GIT_SHA}")
    out_explicit, n_explicit = redact_secrets(f"key {_GIT_SHA}", profile="transcript")
    assert (out_default, n_default) == (out_explicit, n_explicit)


def test_basic_auth_ignores_ordinary_prose() -> None:
    """Found in the live vault on 2026-09-18: the shipped pattern matched
    "Basic photolithography" in a lecture note, because the word after
    "Basic" is 16+ alphanumeric characters. Harmless while redaction only
    touched transcripts; now that every note write redacts, it would rewrite
    the lecture on its next ingestion."""
    for prose in (
        "Basic photolithography process is central to chip fabrication.",
        "Basic understanding of the material is assumed.",
        "basic instrumentation techniques",
    ):
        out, n = redact_secrets(prose, profile="note")
        assert (out, n) == (prose, 0), prose


def test_basic_auth_still_catches_a_real_header() -> None:
    """Base64 of user:password — mixed case, digits, or padding."""
    for header in (
        "Authorization: Basic dXNlcjpwYXNzd29yZA==",
        "Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==",
        "basic YWRtaW46czNjcjN0cGFzcw==",
    ):
        out, n = redact_secrets(header, profile="note")
        assert n >= 1, header
        assert "dXNlcjpwYXNzd29yZA" not in out
        assert "QWxhZGRpbjpvcGVuIHNlc2FtZQ" not in out
