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


def test_known_carries_emotion_and_fallback_text():
    voc = nv("*sigh*")
    assert voc.emotion == "weary"
    assert voc.nonverbal_tag == "<sigh>"
    assert voc.spoken and "<" not in voc.spoken  # plain text for tag-less engines


def test_elongation_increases_intensity():
    assert nv("aaaaaah").intensity > nv("aah").intensity


def test_exclamation_marks_increase_intensity():
    assert nv("ugh!!!").intensity > nv("ugh").intensity


def test_all_caps_bumps_intensity():
    assert nv("AAAH").intensity >= nv("aaah").intensity


def test_scream_fallback_scales_with_stretch():
    short = nv("aah").spoken
    long = nv("aaaaaaah").spoken
    assert long.count("a") > short.count("a")


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
