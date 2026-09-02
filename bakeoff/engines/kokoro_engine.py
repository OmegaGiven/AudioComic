"""Kokoro-82M -- the fast, tiny, permissive default candidate.

Apache-2.0, ~350 MB, runs in ~2-3 GB VRAM (or CPU). 54 fixed voices, no
emotion control, no cloning. The "draft mode / fast preview" tier.

    pip install kokoro soundfile
"""

from __future__ import annotations

import soundfile as sf
from _common import engine_out, finish, load_passage
from kokoro import KPipeline

# Kokoro voice ids: a=American. https://huggingface.co/hexgrad/Kokoro-82M
VOICE_FOR = {
    "NARRATOR":   "af_heart",
    "BLACK HAND": "am_onyx",
    "HAL JORDAN": "am_michael",
    "MERA":       "af_bella",
}
SR = 24000


def main() -> None:
    _, segs = load_passage()
    pipe = KPipeline(lang_code="a")
    d = engine_out("kokoro")
    wavs = []
    for seg in segs:
        out = d / f"seg{seg.idx:02d}.wav"
        voice = VOICE_FOR.get(seg.speaker, "af_heart")
        audio_chunks = [a for _, _, a in pipe(seg.speak_text, voice=voice)]
        audio = audio_chunks[0] if len(audio_chunks) == 1 else _cat(audio_chunks)
        sf.write(str(out), audio, SR)
        wavs.append(out)
    finish("kokoro", {"note": "82M, Apache-2.0, no emotion/cloning; fast-preview tier"},
           wavs, segs)


def _cat(chunks):
    import numpy as np
    return np.concatenate(chunks)


if __name__ == "__main__":
    main()
