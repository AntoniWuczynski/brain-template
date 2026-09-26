"""The vault's one slug rule: ``concepts.slugify`` folds diacritics to the
base letter (decided 2026-09-17 so Polish, and later Spanish, meeting notes
get readable ASCII ids) and every other slug site delegates to it."""

from __future__ import annotations

from ingest_lib.concepts import fold_ascii, slugify
from ingest_lib.connectors.base import meeting_filename
from ingest_lib.connectors.notion import _note_slug
from ingest_lib.extractors.meeting import attendee_slug


def test_fold_ascii_folds_decomposable_and_non_decomposable_letters() -> None:
    assert fold_ascii("Łódź") == "Lodz"
    assert fold_ascii("Zażółć gęślą jaźń") == "Zazolc gesla jazn"
    assert fold_ascii("Peña, Núñez, Müller, Øresund, Straße") == "Pena, Nunez, Muller, Oresund, Strasse"
    assert fold_ascii("日本語") == "日本語"   # no Latin base: untouched, not dropped


def test_slugify_folds_polish_and_spanish() -> None:
    assert slugify("Spotkanie zarządu") == "spotkanie-zarzadu"
    assert slugify("Umowa o pracę") == "umowa-o-prace"
    assert slugify("Łódź") == "lodz"
    assert slugify("Antoni Wuczyński") == "antoni-wuczynski"
    assert slugify("Reunión con Peña") == "reunion-con-pena"
    assert slugify("Quantum Mechanics and Schrödinger Equation") == (
        "quantum-mechanics-and-schrodinger-equation"
    )
    assert slugify("日本語") == ""


def test_slugify_still_collapses_case_and_punctuation_drift() -> None:
    assert slugify("Behaviour-Driven Development") == slugify("behaviour driven development")
    assert slugify("  --Kern--  ") == "kern"


def test_every_slug_site_shares_the_rule() -> None:
    assert attendee_slug("Łukasz Żółć") == "lukasz-zolc"
    assert meeting_filename("2026-09-17", "Spotkanie zarządu", "id-1").startswith(
        "2026-09-17-spotkanie-zarzadu-"
    )
    assert _note_slug("Notatki z Łodzi") == "notatki-z-lodzi"
