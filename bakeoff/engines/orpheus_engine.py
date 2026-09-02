"""Orpheus 3B (Canopy Labs) -- Apache-2.0, Llama-backbone speech LLM.

The non-verbal-tag candidate: understands inline <sigh>, <laugh>, <gasp>,
<groan>, <yawn>, <cough>, <sniffle>, <chuckle> -- which is exactly what
panelspeak.onomatopoeia emits. Zero-shot voice by name (tara/leo/dan/...),
no reference clip needed.

    pip install transformers snac torch soundfile

~7 GB model download on first run. Fits the 5080 comfortably.
"""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf
import torch
from _common import engine_out, finish, load_passage
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer

# canopylabs/orpheus-3b-0.1-ft is gated; this mirror is open. Override with
# ORPHEUS_MODEL=... (and set HF_TOKEN) to use the official weights.
MODEL = os.getenv("ORPHEUS_MODEL", "unsloth/orpheus-3b-0.1-ft")
SR = 24000

VOICE_FOR = {
    "NARRATOR":   "tara",
    "BLACK HAND": "dan",
    "HAL JORDAN": "leo",
    "MERA":       "jess",
}
# Orpheus only understands its own tag set; map ours onto it.
TAG_MAP = {"<sigh>": "<sigh>", "<groan>": "<groan>", "<gasp>": "<gasp>",
           "<laugh>": "<laugh>", "<chuckle>": "<chuckle>", "<yawn>": "<yawn>",
           "<cough>": "<cough>", "<sniff>": "<sniffle>"}


def _decode(codes: list[int], snac: SNAC) -> np.ndarray:
    # regroup the flat 7-per-frame code stream into SNAC's 3 tiers
    frames = len(codes) // 7
    l1, l2, l3 = [], [], []
    for i in range(frames):
        b = codes[7 * i: 7 * i + 7]
        l1.append(b[0])
        l2 += [b[1], b[4]]
        l3 += [b[2], b[3], b[5], b[6]]
    layers = [torch.tensor(x, device=snac.device).unsqueeze(0) for x in (l1, l2, l3)]
    with torch.inference_mode():
        return snac.decode(layers).squeeze().cpu().numpy()


def main() -> None:
    _, segs = load_passage()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="cuda")
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to("cuda")

    d = engine_out("orpheus")
    wavs = []
    for seg in segs:
        voice = VOICE_FOR.get(seg.speaker, "tara")
        text = seg.speak_text
        for ours, theirs in TAG_MAP.items():
            text = text.replace(ours, theirs)
        prompt = f"{voice}: {text}"

        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        # Orpheus special framing tokens
        start = torch.tensor([[128259]], device="cuda")
        end = torch.tensor([[128009, 128260]], device="cuda")
        ids = torch.cat([start, ids, end], dim=1)
        with torch.inference_mode():
            gen = model.generate(ids, max_new_tokens=1800, do_sample=True,
                                 temperature=0.6, top_p=0.9, repetition_penalty=1.1,
                                 eos_token_id=128258)
        out_tokens = gen[0][ids.shape[1]:].tolist()
        codes = [t - 128266 for t in out_tokens if 128266 <= t < 128266 + 4096]
        codes = codes[: (len(codes) // 7) * 7]
        audio = _decode(codes, snac) if codes else np.zeros(SR // 2, dtype=np.float32)

        out = d / f"seg{seg.idx:02d}.wav"
        sf.write(str(out), audio, SR)
        wavs.append(out)
    finish("orpheus", {"note": "Apache-2.0 3B; inline non-verbal tags, voice-by-name"},
           wavs, segs)


if __name__ == "__main__":
    main()
