"""The behaviour every provider adapter must honour.

``contract_*`` functions here are the shared checklist. They run against the
fakes on every CI run, and against a real endpoint when the matching env var
is set:

    PANELSPEAK_VISION_ENDPOINT / _MODEL
    PANELSPEAK_CHAT_ENDPOINT   / _MODEL
    PANELSPEAK_TTS_ENDPOINT    / _MODEL

When a new adapter is added (Chatterbox, VibeVoice, ...), wire it into the
``real_*`` fixtures below and it inherits the whole contract.
"""

from __future__ import annotations

import os
import wave

import pytest

from panelspeak.providers import TTSRequest
from tests.providers.fakes import FakeChatModel, FakeTTS, FakeVisionModel

# --------------------------------------------------------------------------
# contracts

def contract_vision(model, sample_image):
    res = model.analyze_panel(str(sample_image), prompt="describe this panel")
    assert res.error is None, res.error
    assert isinstance(res.text, str) and res.text.strip()


def contract_chat(model):
    out = model.complete("Reply with the single word: ping", max_tokens=16)
    assert isinstance(out, str) and out.strip()


def contract_tts(provider, tmp_path):
    res = provider.synthesize(TTSRequest(text="This is a test line.", speaker="NARRATOR"))
    assert res.sample_rate > 0
    assert res.duration_s > 0
    assert isinstance(res.audio, (bytes, bytearray)) and len(res.audio) > 0

    # audio must be valid PCM WAV *or* raw frames we can length-check;
    # the fake returns raw int16 frames, real adapters return WAV bytes.
    wav_path = tmp_path / "out.wav"
    wav_path.write_bytes(res.audio)
    try:
        with wave.open(str(wav_path), "rb") as w:
            assert w.getnframes() > 0
    except wave.Error:
        assert len(res.audio) >= res.sample_rate * 0.1 * 2  # >=0.1s of int16


def contract_tts_capability_flags_are_declared(provider):
    for flag in ("supports_voice_cloning", "supports_emotion", "supports_context"):
        assert isinstance(getattr(provider, flag), bool)
    assert isinstance(provider.nonverbal_tags, frozenset)


# --------------------------------------------------------------------------
# run the contracts against the fakes

def test_fake_vision_contract(tmp_path):
    img = tmp_path / "p.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0stub")
    contract_vision(FakeVisionModel(), img)


def test_fake_chat_contract():
    contract_chat(FakeChatModel())


def test_fake_tts_contract(tmp_path):
    contract_tts(FakeTTS(), tmp_path)
    contract_tts_capability_flags_are_declared(FakeTTS())


def test_fake_vision_reports_errors_not_raises():
    m = FakeVisionModel()
    m._healthy = False
    res = m.analyze_panel("x.jpg", prompt="p")
    assert res.error and res.text == ""


def test_fake_tts_passes_emotion_and_context_through():
    tts = FakeTTS()
    tts.synthesize(TTSRequest(text="hi", speaker="MERA", emotion="weary",
                              intensity=0.7, context_before="before",
                              context_after="after"))
    got = tts.requests[-1]
    assert got.emotion == "weary" and got.intensity == 0.7
    assert got.context_before == "before"


# --------------------------------------------------------------------------
# opt-in: same contracts against a real endpoint

_HAS_VISION = bool(os.getenv("PANELSPEAK_VISION_ENDPOINT"))
_HAS_CHAT = bool(os.getenv("PANELSPEAK_CHAT_ENDPOINT"))
_HAS_TTS = bool(os.getenv("PANELSPEAK_TTS_ENDPOINT"))


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_CHAT, reason="PANELSPEAK_CHAT_ENDPOINT not set")
def test_real_chat_contract():
    pytest.skip("adapter not implemented yet -- wire panelspeak.adapters here")


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_VISION, reason="PANELSPEAK_VISION_ENDPOINT not set")
def test_real_vision_contract():
    pytest.skip("adapter not implemented yet -- wire panelspeak.adapters here")


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_TTS, reason="PANELSPEAK_TTS_ENDPOINT not set")
def test_real_tts_contract():
    pytest.skip("adapter not implemented yet -- wire panelspeak.adapters here")
