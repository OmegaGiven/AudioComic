"""Chatterbox (Resemble AI) -- MIT, ~0.5B, the emotion-knob candidate.

Zero-shot voice cloning from a short reference clip, plus an `exaggeration`
control (unique among open TTS). If bakeoff/refs/<name>.wav exists for a
speaker it's cloned; otherwise the built-in voice is used for everyone.

    pip install chatterbox-tts

emotion -> (exaggeration, cfg_weight): higher exaggeration is more expressive
but speeds up delivery, so cfg_weight drops to keep pacing sane (per the
project's own audiobook guidance).
"""

from __future__ import annotations

import soundfile as sf
from _common import engine_out, finish, load_passage, ref_path
from chatterbox.tts import ChatterboxTTS

EMOTION_PARAMS = {
    "menacing":  (0.85, 0.3),
    "pain":      (1.30, 0.3),
    "urgent":    (0.75, 0.35),
    "commanding":(0.70, 0.4),
    "tense":     (0.65, 0.4),
    "ominous":   (0.60, 0.45),
    "weary":     (0.45, 0.5),
    "neutral":   (0.50, 0.5),
}


def main() -> None:
    _, segs = load_passage()
    model = ChatterboxTTS.from_pretrained(device="cuda")
    d = engine_out("chatterbox")
    wavs = []
    cloned = set()
    for seg in segs:
        out = d / f"seg{seg.idx:02d}.wav"
        exaggeration, cfg = EMOTION_PARAMS.get(seg.emotion, (0.5, 0.5))
        if seg.intensity:
            exaggeration = min(1.6, exaggeration + 0.4 * seg.intensity)
        ref = ref_path(seg)
        kwargs = {"exaggeration": exaggeration, "cfg_weight": cfg}
        if ref:
            kwargs["audio_prompt_path"] = str(ref)
            cloned.add(seg.speaker)
        wav = model.generate(seg.speak_text, **kwargs)
        sf.write(str(out), wav.squeeze(0).detach().cpu().numpy(), model.sr)
        wavs.append(out)
    finish("chatterbox",
           {"note": "MIT ~0.5B; exaggeration knob per emotion",
            "cloned_speakers": sorted(cloned) or "none (built-in voice for all)"},
           wavs, segs)


if __name__ == "__main__":
    main()
