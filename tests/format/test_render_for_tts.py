"""Tests for panelspeak.emotion.render_for_tts -- cleaning onomatopoeia for a
plain (non-emotive) engine like Kokoro.
"""

from __future__ import annotations

from panelspeak.emotion import Segment, render_for_tts


def _texts(segs):
    return [(s.speaker, s.text) for s in segs]


def test_emotive_engine_passthrough():
    segs = [Segment("HAL JORDAN", "Aaaah!"), Segment("MERA", "*sigh* Fine.")]
    assert render_for_tts(segs, emotive_engine=True) == segs


def test_standalone_scream_becomes_narrator_beat():
    out = render_for_tts([Segment("HAL JORDAN", "Aaaah!")])
    assert _texts(out) == [("NARRATOR", "Hal Jordan screamed.")]


def test_standalone_tsk_becomes_narrator_beat_too():
    out = render_for_tts([Segment("BLACK HAND", "Tsk.")])
    assert out[0].speaker == "NARRATOR"
    assert out[0].text == "Black Hand clicked their tongue."


def test_breathy_lead_in_dialogue_is_lifted_out():
    out = render_for_tts([Segment("MERA", "*sigh* It's never just one of them.")])
    assert _texts(out) == [
        ("NARRATOR", "Mera sighed."),
        ("MERA", "It's never just one of them."),
    ]


def test_clean_interjection_lead_is_kept_inline():
    out = render_for_tts([Segment("BLACK HAND", "Tsk. You should be honored.")])
    assert len(out) == 1
    assert out[0].speaker == "BLACK HAND"
    assert out[0].text == "Tsk. You should be honored."


def test_narrator_lines_untouched():
    seg = Segment("NARRATOR", "The lantern went dark.")
    assert render_for_tts([seg]) == [seg]


def test_plain_dialogue_untouched():
    seg = Segment("MERA", "We have to move, now.")
    assert render_for_tts([seg]) == [seg]
