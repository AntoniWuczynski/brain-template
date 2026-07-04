"""Frontmatter serialization: YAML-safe, byte-stable for the common case.

Pins the fix that replaced repr()/f-string interpolation (which mangled
dates and could emit invalid YAML that the next rebuild silently dropped)
with the shared fm_scalar/fm_list helpers.
"""
from __future__ import annotations

import datetime

import yaml

from ingest_lib.notes import fm_list, fm_scalar


def test_fm_scalar_leaves_plain_strings_unquoted():
    # Byte-stable: plain-safe strings render exactly as the old f-string did.
    for s in ["Beta", "Requirements Engineering", "Behaviour-Driven Development"]:
        assert fm_scalar(s) == s


def test_fm_scalar_quotes_yaml_hostile_strings_validly():
    for s in ["TCP: Congestion Control", "[bracket", "@handle", "a #b"]:
        rendered = fm_scalar(s)
        assert yaml.safe_load(f"title: {rendered}")["title"] == s


def test_fm_scalar_renders_dates_as_iso_not_repr():
    d = datetime.date(2026, 6, 9)
    assert fm_scalar(d) == "2026-06-09"
    assert "datetime.date" not in fm_scalar(d)
    assert yaml.safe_load(f"reviewed: {fm_scalar(d)}")["reviewed"] == d


def test_fm_list_empty_and_populated_and_coercion():
    assert fm_list([]) == "[]"                       # byte-stable with prior "[]"
    assert yaml.safe_load(f"aliases: {fm_list(['a', 'b'])}")["aliases"] == ["a", "b"]
    # A bare string coerces to a one-element list rather than being dropped.
    assert yaml.safe_load(f"aliases: {fm_list('solo')}")["aliases"] == ["solo"]
    # A non-list/str becomes [].
    assert fm_list(None) == "[]"
