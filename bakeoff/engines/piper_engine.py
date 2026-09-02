"""Baseline: Piper -- the engine the pipeline ships with today.

No emotion, no cloning; a fixed voice per speaker. This is the bar every other
engine has to clear. Uses the Piper install at ~/.local/opt/piper.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from _common import engine_out, finish, load_passage

PIPER = Path.home() / ".local" / "opt" / "piper" / "piper"
VOICES = Path.home() / ".local" / "opt" / "piper" / "voices"

VOICE_FOR = {
    "NARRATOR":   "en_US-lessac-medium",
    "BLACK HAND": "en_US-norman-medium",
    "HAL JORDAN": "en_US-joe-medium",
    "MERA":       "en_US-amy-medium",
}


def main() -> None:
    _, segs = load_passage()
    d = engine_out("piper")
    wavs = []
    for seg in segs:
        out = d / f"seg{seg.idx:02d}.wav"
        voice = VOICES / f"{VOICE_FOR.get(seg.speaker, 'en_US-lessac-medium')}.onnx"
        # matches 04_tts_render.clean_for_speech: lowercase for espeak-ng
        text = seg.speak_text.strip().lower()
        subprocess.run(
            [str(PIPER), "--model", str(voice), "--output_file", str(out)],
            input=text, text=True, check=True, capture_output=True,
        )
        wavs.append(out)
    finish("piper", {"note": "current pipeline baseline; fixed voice per speaker, no emotion"},
           wavs, segs)


if __name__ == "__main__":
    main()
