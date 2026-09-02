"""Tests for panelspeak.attribution.attribute_vocalization."""

from __future__ import annotations

from panelspeak.attribution import attribute_vocalization
from panelspeak.text_elements import BBox, Bubble, Character, TextElement


def _el(x, y, bubble_id=None):
    return TextElement(text="tsk", bbox=BBox(x, y, 20, 10), bubble_id=bubble_id)


def test_bubbled_vocalization_goes_to_bubble_owner():
    el = _el(10, 10, bubble_id="b1")
    bubbles = [Bubble("b1", owner="MERA", bbox=BBox(0, 0, 40, 40))]
    assert attribute_vocalization(el, [], bubbles) == "MERA"


def test_single_nearby_character_is_the_source():
    el = _el(100, 100)
    chars = [Character("MERA", bbox=BBox(110, 110, 30, 60))]
    assert attribute_vocalization(el, chars, panel_diagonal=500) == "MERA"


def test_far_character_is_not_the_source():
    el = _el(10, 10)
    chars = [Character("MERA", bbox=BBox(480, 480, 30, 60))]
    assert attribute_vocalization(el, chars, panel_diagonal=500) is None


def test_two_equidistant_characters_are_ambiguous():
    el = _el(100, 100)
    chars = [
        Character("MERA", bbox=BBox(70, 100, 20, 40)),
        Character("ARTHUR", bbox=BBox(130, 100, 20, 40)),
    ]
    assert attribute_vocalization(el, chars, panel_diagonal=400) is None


def test_open_mouth_breaks_the_tie():
    el = _el(100, 100)
    chars = [
        Character("MERA", bbox=BBox(70, 100, 20, 40), mouth_open=True),
        Character("ARTHUR", bbox=BBox(130, 100, 20, 40), mouth_open=False),
    ]
    assert attribute_vocalization(el, chars, panel_diagonal=400) == "MERA"


def test_no_characters_and_no_bubble_is_unattributable():
    assert attribute_vocalization(_el(10, 10), [], []) is None


def test_no_geometry_falls_back_to_lone_open_mouth():
    el = TextElement(text="gasp")  # no bbox
    chars = [
        Character("MERA", mouth_open=True),
        Character("ARTHUR", mouth_open=False),
    ]
    assert attribute_vocalization(el, chars, []) == "MERA"


def test_bubble_without_owner_returns_none_not_crash():
    el = _el(10, 10, bubble_id="b1")
    assert attribute_vocalization(el, [], [Bubble("b1", owner=None)]) is None
