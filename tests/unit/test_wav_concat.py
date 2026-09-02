"""Characterization tests for ``concat_wavs`` in 04_tts_render."""

from __future__ import annotations

import wave

import pytest


def _write_wav(path, *, frames=1000, rate=22050, nchannels=1, sampwidth=2, fill=b"\x01\x00"):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(nchannels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(fill * frames)


def test_empty_input_raises(script, tmp_path):
    with pytest.raises(ValueError):
        script("tts").concat_wavs([], tmp_path / "out.wav")


def test_concatenates_in_given_order(script, tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    _write_wav(a, frames=1000)
    _write_wav(b, frames=500)
    out = tmp_path / "out.wav"
    script("tts").concat_wavs([a, b], out)
    with wave.open(str(out), "rb") as w:
        assert w.getnframes() == 1500
        assert w.getframerate() == 22050


def test_single_segment_roundtrips(script, tmp_path):
    a = tmp_path / "a.wav"
    _write_wav(a, frames=777)
    out = tmp_path / "out.wav"
    script("tts").concat_wavs([a], out)
    with wave.open(str(out), "rb") as w:
        assert w.getnframes() == 777


def test_mismatched_sample_rate_is_currently_mis_stitched(script, tmp_path):
    """Characterization of a known limitation: concat_wavs takes the *first*
    file's params and writes every other file's frames under that header, so a
    24kHz segment after a 22.05kHz one comes out pitch-shifted. When a neural
    TTS engine lands (mixed rates become likely), fix concat to resample and
    change this test to assert correctness."""
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    _write_wav(a, rate=22050, frames=1000)
    _write_wav(b, rate=24000, frames=1000)
    out = tmp_path / "out.wav"
    script("tts").concat_wavs([a, b], out)
    with wave.open(str(out), "rb") as w:
        assert w.getframerate() == 22050          # first file's rate wins
        assert w.getnframes() == 2000             # frames concatenated as-is
