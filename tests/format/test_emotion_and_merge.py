"""Tests for panelspeak.emotion -- the emotion-hint line format and folding
standalone vocalization lines into a character's speech (design note 2).
"""

from __future__ import annotations

from panelspeak.emotion import (
    Segment,
    merge_vocalizations,
    parse_line,
    parse_script,
)

# --- line parsing -----------------------------------------------------------

def test_plain_line_still_parses_no_emotion():
    p = parse_line("MERA: We have to move.")
    assert (p.speaker, p.text, p.emotion) == ("MERA", "We have to move.", None)


def test_emotion_hint_parsed():
    p = parse_line("MERCURY (panicked): You don't understand!")
    assert p.speaker == "MERCURY"
    assert p.emotion == "panicked"
    assert p.text == "You don't understand!"


def test_multiword_emotion_hint():
    p = parse_line("ARTHUR (grim, resolute): So it ends here.")
    assert p.emotion == "grim, resolute"


def test_placeholder_speaker_still_rewritten_with_emotion_form():
    p = parse_line("VILLAIN (mocking): Pathetic.")
    assert p.speaker == "NARRATOR"


def test_narrator_line():
    assert parse_line("NARRATOR: The lantern went dark.").speaker == "NARRATOR"


def test_non_line_returns_none():
    assert parse_line("just some loose prose") is None


# --- whole-script parsing --------------------------------------------------

def test_parse_script_handles_wrapped_lines():
    raw = "NARRATOR: The ring rose into the sky\nand its light guttered out.\nMERA: No."
    segs = parse_script(raw)
    assert len(segs) == 2
    assert segs[0].text.endswith("guttered out.")


def test_parse_script_extracts_nonverbal_tags():
    segs = parse_script("MERA (weary): *sigh* Fine. Have it your way.")
    assert "<sigh>" in segs[0].nonverbal


# --- vocalization merging ------------------------------------------------

def test_standalone_vocalization_merges_forward_into_same_speaker():
    segs = [
        Segment("MERCURY", "Tsk."),
        Segment("MERCURY", "You really thought that would work?"),
    ]
    out = merge_vocalizations(segs)
    assert len(out) == 1
    assert out[0].text == "Tsk. You really thought that would work?"
    assert out[0].speaker == "MERCURY"


def test_standalone_vocalization_merges_backward_into_same_speaker():
    segs = [
        Segment("MERCURY", "You really thought that would work?"),
        Segment("MERCURY", "Tsk."),
    ]
    out = merge_vocalizations(segs)
    assert len(out) == 1
    assert out[0].text.startswith("You really thought")
    assert out[0].text.rstrip().endswith("Tsk.")


def test_vocalization_not_merged_across_different_speakers():
    segs = [
        Segment("MERCURY", "Tsk."),
        Segment("ARTHUR", "Say that again."),
    ]
    out = merge_vocalizations(segs)
    assert len(out) == 2


def test_narrator_vocalization_left_alone():
    segs = [
        Segment("NARRATOR", "Ugh."),
        Segment("NARRATOR", "The smell hit them first."),
    ]
    out = merge_vocalizations(segs)
    assert len(out) == 2


def test_emotion_carried_through_merge():
    segs = [
        Segment("MERCURY", "Tsk.", emotion="disdain"),
        Segment("MERCURY", "Amateurs."),
    ]
    out = merge_vocalizations(segs)
    assert out[0].emotion == "disdain"


def test_two_consecutive_vocalizations_not_collapsed_into_nothing():
    segs = [Segment("MERCURY", "Tsk."), Segment("MERCURY", "Ugh.")]
    out = merge_vocalizations(segs)
    # neither is a dialogue line to attach to; both survive
    assert [s.text for s in out] == ["Tsk.", "Ugh."]
