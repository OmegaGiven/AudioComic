"""Chatterbox (Resemble AI) -- MIT, ~0.5B, the emotion-knob candidate.

Zero-shot voice cloning from a short reference clip, plus an `exaggeration`
control (unique among open TTS). Chatterbox ships only ONE built-in voice, so
distinct characters come from reference clips in bakeoff/refs/ -- generate
them with `make_refs.py` (Kokoro voices) or drop in your own recordings.

    pip install chatterbox-tts soundfile

Notes learned from the bake-off:
* exaggeration above ~1.0 degrades fast, and worst on very short inputs -- a
  1-word scream at exaggeration 1.5 came out as noise. Cap at 0.9, and treat
  vocalizations gently.
* lower cfg_weight = slower, more deliberate pacing (compensates for the
  speed-up that exaggeration causes).
* keep punctuation/case -- it's neural, it uses them.
"""

from __future__ import annotations

import re

import soundfile as sf
from _common import engine_out, finish, load_passage, ref_path
from chatterbox.tts import ChatterboxTTS

# emotion -> (exaggeration, cfg_weight). Nothing above 0.9.
EMOTION_PARAMS = {
    "menacing":   (0.80, 0.30),
    "urgent":     (0.75, 0.35),
    "commanding": (0.75, 0.40),
    "pain":       (0.85, 0.45),
    "tense":      (0.60, 0.40),
    "ominous":    (0.55, 0.45),
    "weary":      (0.45, 0.55),
    "neutral":    (0.50, 0.50),
}
# vocalizations ("Aaaah!", "*sigh*") are short and fragile -- fixed gentle knob
VOCALIZATION_PARAMS = (0.65, 0.50)


def _text_for(seg) -> str:
    """Keep case + punctuation. For a bare vocalization use the original
    written form ('Aaaah!' not 'aaaah'), and give a 1-2 char blurt a little
    more to chew on so the model doesn't choke."""
    if seg.kind == "vocalization":
        t = re.sub(r"[*_]", "", seg.raw_text).strip()
    else:
        t = seg.speak_text.strip()
    if len(t.rstrip(".!?")) < 4:
        t = (t.rstrip(".!?") + "... ") * 2
    return t


def main() -> None:
    _, segs = load_passage(emotive=True)
    model = ChatterboxTTS.from_pretrained(device="cuda")
    d = engine_out("chatterbox")
    wavs = []
    cloned = set()
    for seg in segs:
        out = d / f"seg{seg.idx:02d}.wav"
        if seg.kind == "vocalization":
            exaggeration, cfg = VOCALIZATION_PARAMS
        else:
            exaggeration, cfg = EMOTION_PARAMS.get(seg.emotion, (0.5, 0.5))
            exaggeration = min(0.9, exaggeration + 0.1 * seg.intensity)

        kwargs = {"exaggeration": exaggeration, "cfg_weight": cfg}
        ref = ref_path(seg)
        if ref:
            kwargs["audio_prompt_path"] = str(ref)
            cloned.add(seg.speaker)

        wav = model.generate(_text_for(seg), **kwargs)
        sf.write(str(out), wav.squeeze(0).detach().cpu().numpy(), model.sr)
        wavs.append(out)
    finish("chatterbox",
           {"note": "MIT ~0.5B; per-emotion exaggeration (<=0.9), voices cloned "
                    "from Kokoro reference clips",
            "cloned_speakers": sorted(cloned) or "none (built-in voice for all)"},
           wavs, segs)


if __name__ == "__main__":
    main()
