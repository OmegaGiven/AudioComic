"""Characterization tests for ``parse_narrative`` + ``PLACEHOLDER_SPEAKERS``
in 03_narrative -- the deterministic guard rails around the model's output.
"""

from __future__ import annotations


def test_basic_speaker_lines(script):
    raw = "NARRATOR: The city burned.\nMERA: We have to move, now!"
    segs = script("narrative").parse_narrative(raw)
    assert segs == [
        {"speaker": "NARRATOR", "text": "The city burned."},
        {"speaker": "MERA", "text": "We have to move, now!"},
    ]


def test_placeholder_speaker_rewritten_to_narrator(script):
    for bad in ("SPEAKER", "VILLAIN", "ENTITY", "CAPTION"):
        segs = script("narrative").parse_narrative(f"{bad}: something ominous")
        assert segs == [{"speaker": "NARRATOR", "text": "something ominous"}]


def test_wrapped_continuation_line_appended(script):
    raw = "NARRATOR: The lantern rose into the sky\nand the light went out."
    segs = script("narrative").parse_narrative(raw)
    assert len(segs) == 1
    assert segs[0]["text"] == "The lantern rose into the sky and the light went out."


def test_leading_prose_before_any_speaker_is_dropped(script):
    raw = "Here is the narration:\nNARRATOR: It begins."
    segs = script("narrative").parse_narrative(raw)
    assert segs == [{"speaker": "NARRATOR", "text": "It begins."}]


def test_colon_inside_dialogue_is_kept(script):
    segs = script("narrative").parse_narrative("MERA: I'll say it once: run.")
    assert segs == [{"speaker": "MERA", "text": "I'll say it once: run."}]


def test_apostrophe_and_hyphen_in_name(script):
    segs = script("narrative").parse_narrative("BLACK HAND: I greet you.")
    assert segs[0]["speaker"] == "BLACK HAND"


def test_lowercase_line_is_treated_as_continuation_not_speaker(script):
    raw = "MERA: hold the line\neveryone, hold."
    segs = script("narrative").parse_narrative(raw)
    assert len(segs) == 1
