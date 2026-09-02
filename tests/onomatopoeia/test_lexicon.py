"""Tests for panelspeak.onomatopoeia.normalize_vocalization."""

from __future__ import annotations

import pytest

from panelspeak.onomatopoeia import normalize_vocalization as nv


@pytest.mark.parametrize("surface,canonical", [
    ("tsk", "tsk"), ("Tch", "tsk"), ("tut-tut", "tsk"),
    ("*sigh*", "sigh"), ("Sigh", "sigh"),
    ("gasp", "gasp"),
    ("ugh", "ugh"), ("Urgh!", "ugh"),
    ("hmph", "scoff"), ("pfft", "scoff"),
    ("hmm", "hmm"), ("Hmmmm", "hmm"),
    ("haha", "laugh"), ("heh heh", "laugh"), ("BWAHAHA", "laugh"),
    ("ahem", "ahem"),
    ("shh", "shush"), ("shhhh", "shush"),
    ("aaah", "scream"), ("AIEEE", "scream"), ("aaaaargh", "argh"),
    ("whew", "whew"), ("Phew", "whew"),
    ("psst", "psst"),
])
def test_known_vocalizations(surface, canonical):
    voc = nv(surface)
    assert voc is not None, surface
    assert voc.canonical == canonical


def test_known_carries_emotion_tag_and_narration():
    voc = nv("*sigh*")
    assert voc.emotion == "weary"
    assert voc.nonverbal_tag == "<sigh>"
    assert voc.narration == "sighed"          # verb phrase for a narrator beat


def test_clean_interjection_has_spoken_form():
    voc = nv("ugh")
    assert voc.spoken and "<" not in voc.spoken   # a plain engine can just read it
    assert voc.prefer_narration is False


def test_breathy_noises_prefer_narration_over_phonetic_spelling():
    for s in ("*sigh*", "gasp", "aaah", "sob", "yawn"):
        voc = nv(s)
        assert voc.spoken == ""
        assert voc.prefer_narration is True
        assert voc.narration


def test_elongation_increases_intensity():
    assert nv("aaaaaah").intensity > nv("aah").intensity


def test_exclamation_marks_increase_intensity():
    assert nv("ugh!!!").intensity > nv("ugh").intensity


def test_all_caps_bumps_intensity():
    assert nv("AAAH").intensity >= nv("aaah").intensity


def test_scream_is_narrated_not_spelled():
    voc = nv("aaaaaaah")
    assert voc.canonical == "scream"
    assert voc.spoken == "" and voc.narration == "screamed"


def test_unknown_but_vocalization_shaped_passes_through():
    voc = nv("blorp")
    assert voc is not None
    assert voc.canonical == ""       # unknown
    assert voc.is_known is False
    assert voc.spoken == "blorp"     # handed to the engine verbatim


def test_actual_sentence_is_not_a_vocalization():
    assert nv("You really thought that would work?") is None
    assert nv("I greet you, brother") is None


def test_empty_and_none():
    assert nv("") is None
    assert nv("   ") is None
    assert nv(None) is None


def test_double_blurt_recognised():
    assert nv("tsk tsk").canonical == "tsk"
    assert nv("ha ha").canonical == "laugh"
