"""VibeVoice-1.5B (Microsoft) -- MIT, long-form multi-speaker in one pass.

The continuity candidate: instead of synthesizing 8 isolated clips and gluing
them, VibeVoice generates the whole passage as one turn-taking conversation,
so intonation carries across lines. That's the single biggest fix for the
"concatenated clips sound flat" problem.

Needs reference voice samples (one wav per distinct speaker). Drop them in
bakeoff/refs/ named to match passage.json's `ref` fields. Without refs it
falls back to the model's built-in speaker set.

    pip install vibevoice-community    # community-maintained packaging
    # or: git clone https://github.com/microsoft/VibeVoice

~3 GB model. Slow -- a minute of audio can take a few minutes on the 5080.
"""

from __future__ import annotations

import soundfile as sf
from _common import REFS_DIR, engine_out, finish_full, load_passage

MODEL = "microsoft/VibeVoice-1.5B"
BUILTIN = ["Alice", "Frank", "Carter", "Maya"]  # fallback speaker names


def main() -> None:
    data, segs = load_passage()

    speakers = list(data["cast"].keys())
    sp_index = {name: i for i, name in enumerate(speakers)}
    refs = []
    for i, name in enumerate(speakers):
        r = REFS_DIR / data["cast"][name].get("ref", "")
        refs.append(str(r) if r.exists() else BUILTIN[i % len(BUILTIN)])

    # VibeVoice script format: "Speaker N: line"
    lines = [f"Speaker {sp_index[s.speaker]}: {s.speak_text}" for s in segs]
    script = "\n".join(lines)

    from vibevoice import VibeVoiceForConditionalGeneration, VibeVoiceProcessor  # type: ignore
    processor = VibeVoiceProcessor.from_pretrained(MODEL)
    model = VibeVoiceForConditionalGeneration.from_pretrained(MODEL, device_map="cuda")

    inputs = processor(text=[script], voice_samples=[refs], return_tensors="pt").to("cuda")
    out = model.generate(**inputs, tokenizer=processor.tokenizer,
                         cfg_scale=1.3, max_new_tokens=None)
    audio = out.speech_outputs[0].detach().cpu().numpy().squeeze()

    d = engine_out("vibevoice")
    full = d / "full.wav"
    sf.write(str(full), audio, 24000)
    finish_full("vibevoice",
                {"note": "MIT 1.5B; whole passage in one pass -> cross-line prosody",
                 "refs_used": [r if r in BUILTIN else "refs/" + r.split("/")[-1] for r in refs]},
                full, segs)


if __name__ == "__main__":
    main()
