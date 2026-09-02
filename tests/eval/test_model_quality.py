"""Opt-in model-quality evaluations. NOT part of the CI gate.

Run explicitly against a live stack:

    PANELSPEAK_CHAT_ENDPOINT=http://localhost:11434 pytest -m llm tests/eval

These measure the *models*, not our code -- they're for catching a bad model
swap or a prompt regression, and they're allowed to be slow and flaky-ish, so
they never block a merge.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.llm


BANNED_NARRATION_PHRASES = [
    "this panel", "the panel", "the image shows", "the image depicts",
    "in this scene we see", "the panel focuses on", "we see", "depicts a",
]


@pytest.mark.skipif(not os.getenv("PANELSPEAK_CHAT_ENDPOINT"),
                    reason="PANELSPEAK_CHAT_ENDPOINT not set")
def test_narration_does_not_leak_art_description_framing():
    pytest.skip("wire panelspeak.adapters.chat, then: generate a page and "
                "assert none of BANNED_NARRATION_PHRASES appear")


@pytest.mark.skipif(not os.getenv("PANELSPEAK_CHAT_ENDPOINT"),
                    reason="PANELSPEAK_CHAT_ENDPOINT not set")
def test_narration_covers_every_panel_of_a_page():
    pytest.skip("wire adapter, then: feed a known N-panel page, assert "
                ">= N narrative segments (the early-stop regression)")


@pytest.mark.skipif(not os.getenv("PANELSPEAK_VISION_ENDPOINT"),
                    reason="PANELSPEAK_VISION_ENDPOINT not set")
def test_vision_classification_accuracy_on_corpus():
    pytest.skip("wire vision adapter, then: run real model on corpus panel "
                "images, assert kind-accuracy >= floor")


@pytest.mark.skipif(not os.getenv("PANELSPEAK_TTS_ENDPOINT"),
                    reason="PANELSPEAK_TTS_ENDPOINT not set")
def test_tts_smoke_produces_audio_of_plausible_length():
    pytest.skip("wire TTS adapter, then: synth a 12-word line, assert "
                "2s <= duration <= 12s and non-silent")


def test_banned_phrase_list_is_lowercase_and_nonempty():
    """A real assertion so this file isn't entirely skips: the list the other
    tests depend on stays sane."""
    assert BANNED_NARRATION_PHRASES
    assert all(p == p.lower() for p in BANNED_NARRATION_PHRASES)
