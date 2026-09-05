#!/usr/bin/env python3
"""04_tts_render_dia.py <work_dir> <output.wav>

Dia-1.6B (nari-labs) variant of stage 4 -- same narrative.json in, same
concatenated .wav out, so it's a swap-in alternative to
04_tts_render_kokoro.py at the render phase (see run.sh's TTS_ENGINE switch).

Dia's native format is a *dialogue* script tagged [S1]/[S2], not a per-voice
API -- we call it once per segment instead (mirrors Kokoro's per-line call
shape), with the [S1] tag every time. Voice CONSISTENCY across separate
calls needs Dia's audio-prompt cloning: the first line a speaker gets is
generated with no prompt (whatever voice Dia lands on), then cached as that
speaker's reference (audio + transcript) and fed as the audio_prompt for
every later line from them, per Dia's documented voice-clone pattern.

Known rough edges from a real smoke test on this pipeline's GPU, not just
theory -- read before turning this on for a full issue:
  * ~0.5-0.7x realtime generation (slower than audio playback) vs Kokoro's
    much-faster-than-realtime -- a full issue's worth of segments will take
    noticeably longer to render than the Kokoro path.
  * left at its default max_tokens=3072, Dia visibly over-generated on a
    2-sentence test (30s of audio for what should be ~5s) -- max_tokens is
    now estimated per line from word count (~86 tokens/sec of audio) with a
    floor and ceiling, but this hasn't been tuned against a full issue yet.

    pip install -e tools/dia  (needs torch 2.8 cu128 on a 5000-series GPU)
"""
from __future__ import annotations

import json
import re
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from panelspeak.emotion import Segment, render_for_tts  # noqa: E402

MODEL_ID = "nari-labs/Dia-1.6B-0626"
SAMPLE_RATE = 44100  # Dia's native output rate
TOKENS_PER_SEC = 86  # per Dia's own docs
MIN_TOKENS, MAX_TOKENS = 258, 2064  # floor ~3s, ceiling ~24s of audio


def _decaps(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters or sum(c.isupper() for c in letters) / len(letters) < 0.6:
        return text
    text = text.lower()
    text = re.sub(r"(^\s*|[.!?]\s+|[\"'“‘]\s*)([a-z])",
                  lambda m: m.group(1) + m.group(2).upper(), text)
    text = re.sub(r"\bi\b", "I", text)
    text = re.sub(r"\bi'(m|ll|ve|d)\b", lambda m: "I'" + m.group(1), text)
    return text


def clean_for_speech(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = _decaps(text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_segments(narrative: dict) -> list[Segment]:
    raw: list[Segment] = []
    for page_idx in sorted(narrative.keys(), key=int):
        for seg in narrative[page_idx]:
            raw.append(Segment(seg["speaker"], seg["text"]))
    return render_for_tts(raw, emotive_engine=False)


def _max_tokens_for(text: str) -> int:
    words = max(len(text.split()), 1)
    est_seconds = words / 2.3 + 1.0   # ~2.3 words/sec speech + a beat of pad
    return int(min(MAX_TOKENS, max(MIN_TOKENS, est_seconds * TOKENS_PER_SEC)))


def concat_wavs(wav_paths, output_path: Path):
    if not wav_paths:
        raise ValueError("No audio segments to concatenate")
    with wave.open(str(wav_paths[0]), "rb") as first:
        params = first.getparams()
    with wave.open(str(output_path), "wb") as out:
        out.setparams(params)
        for wp in wav_paths:
            with wave.open(str(wp), "rb") as w:
                out.writeframes(w.readframes(w.getnframes()))


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: 04_tts_render_dia.py <work_dir> <output.wav>", file=sys.stderr)
        sys.exit(2)

    work_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    narrative = json.load(open(work_dir / "narrative.json"))

    import soundfile as sf
    from dia.model import Dia

    segments = prepare_segments(narrative)
    print(f"{len(segments)} segments to synthesize (Dia).")

    chunks_dir = work_dir / "audio_chunks_dia"
    chunks_dir.mkdir(exist_ok=True)
    refs_dir = work_dir / "voice_refs_dia"
    refs_dir.mkdir(exist_ok=True)
    voice_map_path = work_dir / "voice_map_dia.json"
    voice_refs: dict[str, dict] = (json.load(open(voice_map_path))
                                   if voice_map_path.exists() else {})

    model = Dia.from_pretrained(MODEL_ID, compute_dtype="float16")

    wav_paths = []
    for i, seg in enumerate(segments):
        chunk_path = chunks_dir / f"seg{i:05d}.wav"
        if chunk_path.exists():
            wav_paths.append(chunk_path)
            continue
        text = clean_for_speech(seg.text)
        if not text:
            continue

        ref = voice_refs.get(seg.speaker)
        script = f"[S1] {text}"
        audio_prompt = None
        if ref:
            script = f"[S1] {ref['text']} {text}"
            audio_prompt = ref["audio"]

        max_tokens = _max_tokens_for(script)
        try:
            audio = model.generate(script, max_tokens=max_tokens, use_torch_compile=False,
                                   verbose=False, cfg_scale=3.0, temperature=1.2,
                                   top_p=0.95, cfg_filter_top_k=45, audio_prompt=audio_prompt)
        except Exception as e:
            print(f"[{i+1}/{len(segments)}] {seg.speaker}: FAILED ({e})", file=sys.stderr)
            continue

        sf.write(str(chunk_path), audio, SAMPLE_RATE)
        wav_paths.append(chunk_path)

        if seg.speaker not in voice_refs:
            # this take becomes the speaker's locked-in reference voice
            ref_path = refs_dir / f"{seg.speaker.replace(' ', '_')}.wav"
            sf.write(str(ref_path), audio, SAMPLE_RATE)
            voice_refs[seg.speaker] = {"audio": str(ref_path), "text": text}
            json.dump(voice_refs, open(voice_map_path, "w"), indent=2)

        print(f"[{i+1}/{len(segments)}] {seg.speaker}: OK ({max_tokens} max_tokens)")

    print(f"Concatenating {len(wav_paths)} chunks into {output_path}...")
    concat_wavs(wav_paths, output_path)
    print(f"Done. Audiobook: {output_path}")


if __name__ == "__main__":
    main()
