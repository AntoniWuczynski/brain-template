"""``ingest_lib.connectors._transcript_common``: secret redaction and turn
finalisation (home-path scrubbing, harness-wrapper stripping/collapsing,
length capping).

Split out of ``tests/test_transcripts.py`` (that file's own 1000-line limit)
along its existing section boundary; see ``tests/test_transcripts.py`` and
its sibling ``test_transcripts_*`` modules for the rest of the split.
"""
from __future__ import annotations

import pytest

from ingest_lib.connectors import _transcript_common as tc


@pytest.mark.parametrize(
    "raw,kind",
    [
        ("AKIA" + "A" * 16, "aws-access-key"),
        ("sk-ant-" + "a" * 25, "anthropic-key"),
        ("sk-" + "a" * 25, "openai-key"),
        ("ghp_" + "a" * 40, "github-token"),
        ("xoxb-" + "a" * 15, "slack-token"),
        ("Bearer " + "a" * 25, "bearer-token"),
    ],
)
def test_redact_secrets_known_patterns(raw: str, kind: str) -> None:
    cleaned, n = tc.redact_secrets(f"here is a secret: {raw} end")
    assert n == 1
    assert f"[REDACTED:{kind}]" in cleaned
    assert raw not in cleaned


def test_redact_secrets_private_key_block() -> None:
    block = "-----BEGIN RSA PRIVATE KEY-----\nabc123\n-----END RSA PRIVATE KEY-----"
    cleaned, n = tc.redact_secrets(block)
    assert n == 1
    assert "[REDACTED:private-key-block]" in cleaned


def test_redact_secrets_jwt() -> None:
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dGhpc2lzYXNpZ25hdHVyZQ"
    # Not an env-style "token=..." assignment (which now correctly matches
    # _ENV_ASSIGN_RE first per the C2 fix) — plain prose containing a JWT,
    # to exercise the JWT pattern itself in isolation.
    cleaned, n = tc.redact_secrets(f"here is a raw jwt: {jwt}")
    assert n == 1
    assert "[REDACTED:jwt]" in cleaned


def test_redact_env_style_assignment_keeps_key_name() -> None:
    cleaned, n = tc.redact_secrets('MY_API_KEY="abcdef123456"')
    assert n == 1
    assert "MY_API_KEY=" in cleaned
    assert "[REDACTED:env-secret]" in cleaned
    assert "abcdef123456" not in cleaned


# The refuter's exact table (review C2): a bare canonical keyword — with
# NOTHING before it for a mandatory leading `[A-Za-z_]` to consume — used to
# pass through unredacted entirely.
@pytest.mark.parametrize(
    "raw",
    [
        "TOKEN=supersecretvalue",
        "SECRET=supersecretvalue",
        "PASSWORD=supersecretvalue",
        "APIKEY=supersecretvalue",
        "API_KEY=supersecretvalue",
        "password: supersecretvalue",
    ],
)
def test_redact_env_style_bare_keyword_assignment(raw: str) -> None:
    cleaned, n = tc.redact_secrets(raw)
    assert n == 1
    assert "[REDACTED:env-secret]" in cleaned
    assert "supersecretvalue" not in cleaned


@pytest.mark.parametrize("raw", ["XTOKEN=supersecretvalue", "X_TOKEN=supersecretvalue"])
def test_redact_env_style_prefixed_keyword_still_matches(raw: str) -> None:
    # The prefixed shape already worked before the C2 fix; must keep working.
    cleaned, n = tc.redact_secrets(raw)
    assert n == 1
    assert "[REDACTED:env-secret]" in cleaned


def test_redact_url_credentials() -> None:
    cleaned, n = tc.redact_secrets("DATABASE_URL=postgres://user:hunter2@host/db")
    assert n >= 1
    assert "hunter2" not in cleaned
    assert "[REDACTED:url-credentials]" in cleaned
    assert "postgres://" in cleaned  # scheme kept for context


# The refuter's round-two adversarial table (review N1) — shapes that
# leaked through the round-one redactor untouched because the env-assign
# regex couldn't cross a space/quote/hyphen and had no vendor pattern for
# these prefixes at all.
@pytest.mark.parametrize(
    "raw,secret",
    [
        ("export API_KEY=supersecretvalue", "supersecretvalue"),
        ("export ANTHROPIC_AUTH_TOKEN=supersecretvalue", "supersecretvalue"),
        ("  export SECRET=supersecretvalue", "supersecretvalue"),
        ("env API_KEY=supersecretvalue node app.js", "supersecretvalue"),
        ('{"api_key": "supersecretvalue"}', "supersecretvalue"),
        ('  "apiKey": "supersecretvalue",', "supersecretvalue"),
        ('  "password": "supersecretvalue",', "supersecretvalue"),
        ('"client_secret":"supersecretvalue"', "supersecretvalue"),
        ("x-api-key: supersecretvalue", "supersecretvalue"),
        ("-H 'x-api-key: supersecretvalue'", "supersecretvalue"),
        ("  - token: supersecretvalue", "supersecretvalue"),
        ("https://api.example.com/v1?api_key=supersecretvalue", "supersecretvalue"),
        ("redis://:hunter2pass@127.0.0.1:6379/0", "hunter2pass"),
        ("github_pat_11ABCDEFG0" + "b" * 60, "github_pat_11ABCDEFG0" + "b" * 60),
        ("npm_" + "A" * 36, "npm_" + "A" * 36),
        ("glpat-" + "A" * 20, "glpat-" + "A" * 20),
        ("AIzaSyA1234567890abcdefghijklmnopqrstuv", "AIzaSyA1234567890abcdefghijklmnopqrstuv"),
        ("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
        ("0123456789abcdef0123456789abcdef01234567", "0123456789abcdef0123456789abcdef01234567"),
    ],
)
def test_redact_secrets_adversarial_table(raw: str, secret: str) -> None:
    cleaned, n = tc.redact_secrets(raw)
    assert n >= 1
    assert "REDACTED" in cleaned
    assert secret not in cleaned


def test_redact_secrets_vendor_pattern_keeps_specific_label_over_generic_one() -> None:
    # A recognised vendor shape sitting right after a generic keyword must
    # keep ITS specific label, not get re-redacted/overwritten by the
    # generic env-secret/http-header net once the vendor pattern already
    # replaced it with a placeholder (review N1 — the double-redaction
    # trap the broadened generic patterns would otherwise fall into).
    cleaned, n = tc.redact_secrets("here is a secret: " + "AKIA" + "A" * 16 + " end")
    assert n == 1
    assert "[REDACTED:aws-access-key]" in cleaned


def test_redact_url_credentials_empty_username() -> None:
    cleaned, n = tc.redact_secrets("redis://:hunter2pass@127.0.0.1:6379/0")
    assert n == 1
    assert "hunter2pass" not in cleaned
    assert "[REDACTED:url-credentials]" in cleaned


def test_redact_basic_auth() -> None:
    cleaned, n = tc.redact_secrets("Authorization: Basic YWxhZGRpbjpvcGVuc2VzYW1l")
    assert n == 1
    assert "[REDACTED:basic-auth]" in cleaned
    assert "YWxhZGRpbjpvcGVuc2VzYW1l" not in cleaned


def test_home_relative_swaps_prefix() -> None:
    assert tc.home_relative("/Users/alice/brain/scripts", "/Users/alice") == "~/brain/scripts"
    assert tc.home_relative("/Users/alice", "/Users/alice") == "~"
    assert tc.home_relative("/var/other", "/Users/alice") == "/var/other"


def test_finalize_turn_text_strips_wrapper_tags_and_scrubs_home() -> None:
    raw = "<system-reminder>ignore this</system-reminder>Please look at /Users/alice/brain/x.py"
    text, n_redacted, truncated = tc.finalize_turn_text(raw, "/Users/alice")
    assert text is not None
    assert "system-reminder" not in text
    assert "~/brain/x.py" in text
    assert not truncated


def test_finalize_turn_text_collapses_bare_slash_command() -> None:
    raw = "<command-message>dream-pass</command-message>\n<command-name>/dream-pass</command-name>"
    text, _n, _t = tc.finalize_turn_text(raw, "")
    assert text == "_(ran `/dream-pass`)_"


def test_finalize_turn_text_collapses_bare_slash_command_name_first() -> None:
    # The real harness emits command-name FIRST for a typed slash command
    # (review C1) — the command-only match must not be order-dependent.
    raw = (
        "<command-name>/exit</command-name>\n"
        "            <command-message>exit</command-message>\n"
        "            <command-args></command-args>"
    )
    text, _n, _t = tc.finalize_turn_text(raw, "")
    assert text == "_(ran `/exit`)_"


def test_finalize_turn_text_collapses_skill_invocation_with_skill_format() -> None:
    # A skill invocation: command-message first, plus a trailing skill-format.
    raw = (
        "<command-message>dream-pass</command-message>\n"
        "<command-name>/dream-pass</command-name>\n"
        "<skill-format>markdown</skill-format>"
    )
    text, _n, _t = tc.finalize_turn_text(raw, "")
    assert text == "_(ran `/dream-pass`)_"


def test_finalize_turn_text_returns_none_for_pure_wrapper() -> None:
    raw = "<system-reminder>only harness content</system-reminder>"
    text, _n, _t = tc.finalize_turn_text(raw, "")
    assert text is None


@pytest.mark.parametrize(
    "tag",
    [
        "local-command-caveat",
        "task-notification",
        "bash-input",
        "bash-stdout",
        "bash-stderr",
    ],
)
def test_finalize_turn_text_strips_all_observed_wrapper_tags(tag: str) -> None:
    raw = f"<{tag}>harness furniture, not something the user said</{tag}>"
    text, _n, _t = tc.finalize_turn_text(raw, "")
    assert text is None, f"{tag} should be fully stripped, leaving nothing"


def test_finalize_turn_text_drops_unrecognised_harness_tag_via_allowlist() -> None:
    # A wrapper tag this module doesn't name yet, but still SHAPED LIKE
    # this harness's own tag vocabulary (a "system-*" tag here) — the
    # allowlist fallback must still drop it rather than keep it verbatim as
    # if it were real prose (review C1, restricted per N2).
    raw = "<system-future-event>new bookkeeping shape</system-future-event>"
    text, _n, _t = tc.finalize_turn_text(raw, "")
    assert text is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Not every leading "<word>" is harness — plain prose/markup a
        # developer pasted must survive, even when it happens to open with
        # a tag, as long as that tag isn't shaped like this harness's own
        # wrapper vocabulary (review N2).
        ("<system-reminder>drop me</system-reminder>but keep this part", "but keep this part"),
        (
            "<details>\n<summary>x</summary>\nreal prose\n</details>",
            "<details>\n<summary>x</summary>\nreal prose\n</details>",
        ),
        (
            "<div>  hello world, this is a component I am debugging </div>",
            "<div>  hello world, this is a component I am debugging </div>",
        ),
        (
            "<html><body>Fix this page please</body></html>",
            "<html><body>Fix this page please</body></html>",
        ),
        ("<svg>…</svg>\nwhy does this not render?", "<svg>…</svg>\nwhy does this not render?"),
        (
            "<configuration>…</configuration>\nwhat is wrong here?",
            "<configuration>…</configuration>\nwhat is wrong here?",
        ),
    ],
)
def test_finalize_turn_text_keeps_real_prose_starting_with_angle_bracket_word(
    raw: str, expected: str
) -> None:
    text, _n, _t = tc.finalize_turn_text(raw, "")
    assert text == expected


def test_finalize_turn_text_keeps_prose_after_a_leading_command_name_tag() -> None:
    # A <command-name> element mixed into otherwise-real prose (not a PURE
    # bare-command wrapper) must have just that element stripped, not make
    # the allowlist net mistake the remainder for an unrecognised wrapper
    # and drop the whole turn (review N2).
    raw = "<command-name>/foo</command-name>\nand here is my actual question about the build"
    text, _n, _t = tc.finalize_turn_text(raw, "")
    assert text == "and here is my actual question about the build"


def test_finalize_turn_text_truncates_long_turns() -> None:
    raw = "x" * (tc._MAX_TURN_CHARS + 500)
    text, _n, truncated = tc.finalize_turn_text(raw, "")
    assert truncated is True
    assert text is not None
    assert len(text) < len(raw)
    assert "truncated" in text
