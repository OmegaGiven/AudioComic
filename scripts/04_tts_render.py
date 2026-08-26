#!/usr/bin/env python3
"""04_tts_render.py <work_dir> <output.wav>

Multi-voice TTS rendering. Reads narrative.json (page -> list of
{"speaker","text"} segments), assigns each distinct non-narrator speaker a
consistent Piper voice for the whole issue (saved to voice_map.json so it's
stable across reruns and reviewable/editable by hand), synthesizes every
segment in reading order, and concatenates into one final audiobook WAV.

All voices confirmed at 22050Hz -- plain WAV frame concatenation, no
resampling needed.
"""
import json
import re
import subprocess
import sys
import wave
from pathlib import Path

PIPER_BIN = Path.home() / ".local" / "opt" / "piper" / "piper"
VOICES_DIR = Path.home() / ".local" / "opt" / "piper" / "voices"

NARRATOR_VOICE = "en_US-lessac-medium"
# Two axes (gender x age-tone) instead of just gender, for real per-character
# distinctiveness rather than everyone splitting into two buckets.
VOICE_POOLS = {
    "MALE-YOUNG": ["en_US-joe-medium", "en_US-ryan-medium", "en_US-sam-medium"],
    "MALE-OLD": ["en_US-norman-medium", "en_US-bryce-medium"],
    "FEMALE-YOUNG": ["en_US-amy-medium", "en_US-kristin-medium", "en_US-ljspeech-medium"],
    # Only one distinctly older-toned female voice available locally -- if
    # more than one older woman appears in a cast, they'll share this voice.
    # Real limitation of the current voice pool, not a classification bug.
    "FEMALE-OLD": ["en_US-hfc_female-medium"],
}
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
GENDER_MODEL = "devstral:24b"


CLASSIFY_BATCH_SIZE = 15  # real bug hit in testing: a 65-name single call
# exceeded a 60s timeout, returned nothing, and silently defaulted EVERY
# speaker to MALE-YOUNG (including clearly female characters). Smaller
# batches + a generous per-batch timeout instead of one huge fragile call.


def classify_speakers(speakers: list) -> dict:
    """One-shot (well, chunked) gender+age-tone classification so voice
    assignment is actually character-appropriate (real bugs hit in testing:
    John Stewart got a female voice, an older woman got a male voice -- the
    latter traced to a silent MALE default whenever classification failed
    to parse a name). No more silent defaults here -- an unparsed name is
    reported, not guessed wrong."""
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
            d = json.loads(r.stdout.strip().split("\n")[0])
            raw = d.get("response", "")
        except Exception:
            raw = ""

        for line in raw.splitlines():
            m = re.match(r"^(.+?):\s*(MALE|FEMALE)-(YOUNG|OLD)\s*$", line.strip(), re.I)
            if m:
                classes[m.group(1).strip().upper()] = f"{m.group(2).upper()}-{m.group(3).upper()}"

    return classes


def clean_for_speech(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    # espeak-ng (Piper's phonemizer) reads short ALL-CAPS words as
    # initialisms to spell out -- confirmed real bug: "it" in caps came out
    # "I.T." instead of the word "it". Lowercasing is the safe fix; normal
    # sentence-level TTS prosody doesn't depend on case.
    return text.strip().lower()


def load_voice_map(work_dir: Path, speakers: set) -> dict:
    voice_map_path = work_dir / "voice_map.json"
    voice_map = {}
    if voice_map_path.exists():
        voice_map = json.load(open(voice_map_path))

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
                cls = "MALE-YOUNG"  # only reached after the loud warning below
            pool = VOICE_POOLS[cls]
            voice_map[speaker] = pool[pool_idx[cls] % len(pool)]
            pool_idx[cls] += 1

        if unclassified:
            print(f"  WARNING: classification failed for {unclassified} -- "
                  f"defaulted to MALE-YOUNG, review voice_map.json manually.",
                  file=sys.stderr)

    json.dump(voice_map, open(voice_map_path, "w"), indent=2)
    return voice_map


def synthesize(text: str, voice: str, out_path: Path):
    voice_path = VOICES_DIR / f"{voice}.onnx"
    p = subprocess.run(
        [str(PIPER_BIN), "--model", str(voice_path), "--output_file", str(out_path)],
        input=text, capture_output=True, text=True, timeout=60,
    )
    return p.returncode == 0 and out_path.exists()


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


def main():
    if len(sys.argv) != 3:
        print("Usage: 04_tts_render.py <work_dir> <output.wav>", file=sys.stderr)
        sys.exit(2)

    work_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    narrative = json.load(open(work_dir / "narrative.json"))

    chunks_dir = work_dir / "audio_chunks"
    chunks_dir.mkdir(exist_ok=True)

    all_segments = []
    for page_idx in sorted(narrative.keys(), key=int):
        all_segments.extend(narrative[page_idx])

    speakers = {seg["speaker"] for seg in all_segments}
    voice_map = load_voice_map(work_dir, speakers)
    print(f"Voice map: {voice_map}")
    print(f"{len(all_segments)} total segments to synthesize.")

    wav_paths = []
    for i, seg in enumerate(all_segments):
        chunk_path = chunks_dir / f"seg{i:05d}.wav"
        if chunk_path.exists():
            wav_paths.append(chunk_path)
            continue

        text = clean_for_speech(seg["text"])
        if not text:
            continue
        voice = voice_map.get(seg["speaker"], NARRATOR_VOICE)
        ok = synthesize(text, voice, chunk_path)
        if ok:
            wav_paths.append(chunk_path)
            print(f"[{i+1}/{len(all_segments)}] {seg['speaker']} ({voice}): OK")
        else:
            print(f"[{i+1}/{len(all_segments)}] {seg['speaker']}: FAILED to synthesize", file=sys.stderr)

    print(f"Concatenating {len(wav_paths)} audio chunks into {output_path}...")
    concat_wavs(wav_paths, output_path)
    print(f"Done. Audiobook: {output_path}")


if __name__ == "__main__":
    main()
