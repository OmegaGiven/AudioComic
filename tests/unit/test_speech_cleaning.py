"""Characterization tests for ``clean_for_speech`` in 04_tts_render.

These pin the Piper-specific workarounds (lowercasing to stop espeak-ng
spelling out short all-caps words, URL stripping). When a neural engine
replaces Piper, this function's contract will change and these tests are the
checklist of what to reconsider.
"""

from __future__ import annotations


def test_lowercases_to_avoid_initialism_bug(script):
    # "IT" would be read "I.T." by espeak-ng
    assert script("tts").clean_for_speech("IT is done") == "it is done"


def test_strips_urls(script):
    out = script("tts").clean_for_speech("see https://example.com/x for more")
    assert "http" not in out
    assert out.strip() == "see  for more".strip()


def test_trims_surrounding_whitespace(script):
    assert script("tts").clean_for_speech("  hello  ") == "hello"


def test_empty_stays_empty(script):
    assert script("tts").clean_for_speech("") == ""
    assert script("tts").clean_for_speech("   ") == ""
