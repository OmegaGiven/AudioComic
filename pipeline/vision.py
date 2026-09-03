"""Shared helpers for the two vision phases (transcribe, redescribe)."""
from __future__ import annotations

import base64
import json
import re
import subprocess

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
# qwen3-vl:8b regressed under Ollama -- with think:false it now returns an
# empty response and dumps everything in the `thinking` field, so transcribe
# got ~0 dialogue. qwen2.5vl:7b is the pre-reasoning version: no <think>, does
# the two-part task, reads all-caps comic lettering well.
VISION_MODEL = "qwen2.5vl:7b"

_THINK = re.compile(r"<think>.*?</think>\s*", re.S | re.I)
_OPEN_THINK = re.compile(r"<think>.*", re.S | re.I)
_ANSWER_LINE = re.compile(r"^\s*(?:CAPTION|SPEAKER)\s*:", re.M | re.I)
_REASONING = re.compile(
    r"<think>|let'?s (tackle|break this down|start|see|think)|the user wants|"
    r"^\s*(got it|okay|alright|first,|now,|wait,|hmm)", re.I | re.M)


def strip_think(text: str) -> str:
    """Remove qwen3-vl reasoning. It ignores think:false and frequently emits
    an unclosed <think> block; the real answer (CAPTION:/SPEAKER: lines) still
    follows it, so anchor on the first answer line when there's no close tag."""
    text = _THINK.sub("", text)
    if "<think>" in text.lower():
        m = _ANSWER_LINE.search(text)
        text = text[m.start():] if m else _OPEN_THINK.sub("", text)
    return text.strip()


def looks_like_reasoning(text: str) -> bool:
    return bool(_REASONING.search(text or ""))


def ask_vision(image_path: str, prompt: str, *, num_predict: int = 1200,
               timeout: int = 300) -> dict:
    img_b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    payload = {
        "model": VISION_MODEL, "prompt": prompt, "images": [img_b64],
        "stream": False, "think": False,
        # temp 0.15 was too low: ~18 near-identical panel prompts collapsed onto
        # one canned opening ("Rain lashed the graves...") for a whole page.
        "options": {"num_predict": num_predict, "num_ctx": 16384, "temperature": 0.35},
    }
    try:
        r = subprocess.run(
            ["curl", "-s", "-m", str(timeout), OLLAMA_URL, "--data-binary", "@-"],
            input=json.dumps(payload), capture_output=True, text=True, timeout=timeout + 10,
        )
        d = json.loads(r.stdout.strip().split("\n")[0])
    except Exception as e:
        return {"text": "", "error": str(e)}
    # never fall back to the `thinking` field -- that is the reasoning, not
    # the answer, and narrating it was the 147-minute bug.
    return {"text": strip_think((d.get("response") or "").strip()),
            "eval_count": d.get("eval_count")}
