#!/usr/bin/env python3
"""04_tts_render_kokoro.py <work_dir> <output.wav>

Kokoro-82M variant of stage 4. Parallel to 04_tts_render.py (Piper) so the
two can be run against the same work dir and compared.

Differences from the Piper stage 4:
* Kokoro voices (24 kHz) instead of Piper voices (22.05 kHz).
* Segments are run through panelspeak's onomatopoeia cleanup first -- a bare
  "Aaaah!" becomes a narrator beat, "*sigh*" opening a line is lifted out,
  "Tsk." is kept inline (Kokoro has no emotion control, so wordless noises
  never sound right read phonetically).
* No lowercasing -- Kokoro's G2P handles normal casing (the lowercase hack
  was an espeak/Piper workaround).

voice_map.json is shared in format with the Piper script but holds Kokoro
voice ids, so keep separate work dirs (or delete voice_map.json) when
switching engines on the same issue.

    pip install kokoro soundfile
"""
import json
import re
import subprocess
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from panelspeak.emotion import Segment, render_for_tts  # noqa: E402

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
GENDER_MODEL = "devstral:24b"

SAMPLE_RATE = 24000
NARRATOR_VOICE = "af_heart"
# gender x age-tone -> a small pool of Kokoro voices, round-robined so a big
# cast still gets distinct voices. https://huggingface.co/hexgrad/Kokoro-82M
VOICE_POOLS = {
    "MALE-YOUNG":   ["am_fenrir", "am_puck", "am_liam", "am_adam"],
    "MALE-OLD":     ["am_onyx", "am_echo", "am_eric"],
    "FEMALE-YOUNG": ["af_bella", "af_nicole", "af_sarah", "af_aoede"],
    "FEMALE-OLD":   ["af_sky", "bf_alice", "bf_emma"],
}

CLASSIFY_BATCH_SIZE = 15  # a big single call times out and silently defaults
# every speaker to one bucket -- chunk it (real bug from the Piper era).


def classify_speakers(speakers: list) -> dict:
    """Chunked gender+age-tone classification so voice assignment is
    character-appropriate. An unparsed name is reported, never guessed."""
    names = [s for s in speakers if s != "NARRATOR"]
    if not names:
        return {}

    classes = {}
    for i in range(0, len(names), CLASSIFY_BATCH_SIZE):
        batch = names[i:i + CLASSIFY_BATCH_SIZE]
        prompt = (
            "For each of these comic book character names, classify BOTH gender and "
            "approximate age-tone, best guess from the name/context if you don't "
            "recognize the character. Respond with exactly one line per name, format "
            "'NAME: MALE-YOUNG' or 'NAME: MALE-OLD' or 'NAME: FEMALE-YOUNG' or "
            "'NAME: FEMALE-OLD' (use OLD only for a character clearly written/drawn "
            "as elderly or notably older, otherwise YOUNG), nothing else.\n\n"
            + "\n".join(batch)
        )
        payload = {"model": GENDER_MODEL, "prompt": prompt, "stream": False,
                   "options": {"num_predict": len(batch) * 15 + 50}}
        r = subprocess.run(
            ["curl", "-s", "-m", "180", OLLAMA_URL, "--data-binary", "@-"],
            input=json.dumps(payload), capture_output=True, text=True, timeout=185,
        )
        try:
            raw = json.loads(r.stdout.strip().split("\n")[0]).get("response", "")
        except Exception:
            raw = ""
        for line in raw.splitlines():
            m = re.match(r"^(.+?):\s*(MALE|FEMALE)-(YOUNG|OLD)\s*$", line.strip(), re.I)
            if m:
                classes[m.group(1).strip().upper()] = f"{m.group(2).upper()}-{m.group(3).upper()}"
    return classes


def clean_for_speech(text: str) -> str:
    """Kokoro's G2P handles normal casing fine (unlike espeak/Piper, which
    spelled short all-caps words as initialisms), so only strip URLs and
    collapse whitespace here."""
    text = re.sub(r"https?://\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_voice_map(work_dir: Path, speakers: set) -> dict:
    voice_map_path = work_dir / "voice_map.json"
    voice_map = json.load(open(voice_map_path)) if voice_map_path.exists() else {}

    new_speakers = [s for s in sorted(speakers) if s not in voice_map]
    if new_speakers:
        classes = classify_speakers(new_speakers)
        pool_idx = {k: sum(1 for v in voice_map.values() if v in pool)
                    for k, pool in VOICE_POOLS.items()}
        unclassified = []
        for speaker in new_speakers:
            if speaker == "NARRATOR":
                voice_map[speaker] = NARRATOR_VOICE
                continue
            cls = classes.get(speaker.upper())
            if cls is None:
                unclassified.append(speaker)
                cls = "MALE-YOUNG"
            pool = VOICE_POOLS[cls]
            voice_map[speaker] = pool[pool_idx[cls] % len(pool)]
            pool_idx[cls] += 1
        if unclassified:
            print(f"  WARNING: classification failed for {unclassified} -- "
                  f"defaulted to MALE-YOUNG, review voice_map.json manually.",
                  file=sys.stderr)

    json.dump(voice_map, open(voice_map_path, "w"), indent=2)
    return voice_map


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


def prepare_segments(narrative: dict) -> list[Segment]:
    """narrative.json -> flat Segment list, onomatopoeia cleaned for a plain
    (non-emotive) engine."""
    raw: list[Segment] = []
    for page_idx in sorted(narrative.keys(), key=int):
        for seg in narrative[page_idx]:
            raw.append(Segment(seg["speaker"], seg["text"]))
    return render_for_tts(raw, emotive_engine=False)


def main():
    if len(sys.argv) != 3:
        print("Usage: 04_tts_render_kokoro.py <work_dir> <output.wav>", file=sys.stderr)
        sys.exit(2)

    work_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    narrative = json.load(open(work_dir / "narrative.json"))

    from kokoro import KPipeline  # heavy import; keep it out of module load

    segments = prepare_segments(narrative)
    speakers = {s.speaker for s in segments}
    voice_map = load_voice_map(work_dir, speakers)
    print(f"Voice map: {voice_map}")
    print(f"{len(segments)} segments to synthesize.")

    chunks_dir = work_dir / "audio_chunks_kokoro"
    chunks_dir.mkdir(exist_ok=True)
    pipe = KPipeline(lang_code="a")

    import numpy as np
    import soundfile as sf

    wav_paths = []
    for i, seg in enumerate(segments):
        chunk_path = chunks_dir / f"seg{i:05d}.wav"
        if chunk_path.exists():
            wav_paths.append(chunk_path)
            continue
        text = clean_for_speech(seg.text)
        if not text:
            continue
        voice = voice_map.get(seg.speaker, NARRATOR_VOICE)
        audio = [a for _, _, a in pipe(text, voice=voice)]
        if not audio:
            print(f"[{i+1}/{len(segments)}] {seg.speaker}: FAILED (no audio)", file=sys.stderr)
            continue
        sf.write(str(chunk_path),
                 np.concatenate(audio) if len(audio) > 1 else audio[0], SAMPLE_RATE)
        wav_paths.append(chunk_path)
        print(f"[{i+1}/{len(segments)}] {seg.speaker} ({voice}): OK")

    print(f"Concatenating {len(wav_paths)} chunks into {output_path}...")
    concat_wavs(wav_paths, output_path)
    print(f"Done. Audiobook: {output_path}")


if __name__ == "__main__":
    main()
