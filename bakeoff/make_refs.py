"""Generate reference voice clips for the cloning engines (Chatterbox,
VibeVoice) using Kokoro's voice bank.

Chatterbox ships only one built-in voice; it clones from a reference clip.
Kokoro has ~20 clean English voices under Apache-2.0. So: pick a distinct
Kokoro voice per character, read one fixed neutral paragraph (~12 s), and
save it to bakeoff/refs/<name>.wav. Chatterbox then clones each character
from its clip, giving distinct voices sourced from an engine we already
like the sound of.

Run with the kokoro venv:

    bakeoff/.venvs/kokoro/bin/python bakeoff/make_refs.py

Re-run to regenerate. Delete a refs/*.wav and re-run to refresh just the
missing ones. Drop in your own hand-recorded wavs with the same names to
override.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import soundfile as sf
from _common import BAKEOFF_DIR, REFS_DIR
from kokoro import KPipeline

SR = 24000

# a neutral, mid-paced paragraph -- no strong emotion, varied phonemes,
# long enough (~12 s) for a stable voice clone
REF_TEXT = (
    "The evening settled over the harbor, and the boats rocked quietly "
    "against the dock. Somewhere a bell was ringing, slow and even, the "
    "way it always did at that hour. She pulled her coat tighter and "
    "watched the water turn from grey to black."
)

# character -> Kokoro voice id. https://huggingface.co/hexgrad/Kokoro-82M
VOICE_FOR = {
    "narrator.wav":   "af_heart",    # warm female narrator
    "black_hand.wav": "am_onyx",     # deep, measured male
    "hal_jordan.wav": "am_fenrir",   # brighter, younger male
    "mera.wav":       "af_bella",    # firm female
}


def main() -> None:
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    # keep names aligned with passage.json's cast refs
    cast_refs = {c["ref"] for c in json.loads((BAKEOFF_DIR / "passage.json").read_text())["cast"].values()}
    unknown = cast_refs - set(VOICE_FOR)
    if unknown:
        print(f"note: passage.json wants refs with no Kokoro mapping: {sorted(unknown)}", file=sys.stderr)

    pipe = KPipeline(lang_code="a")
    for fname, voice in VOICE_FOR.items():
        out = REFS_DIR / fname
        if out.exists():
            print(f"keep  {fname}")
            continue
        chunks = [a for _, _, a in pipe(REF_TEXT, voice=voice)]
        audio = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        sf.write(str(out), audio, SR)
        print(f"wrote {fname}  ({voice}, {len(audio) / SR:.1f}s)")


if __name__ == "__main__":
    main()
