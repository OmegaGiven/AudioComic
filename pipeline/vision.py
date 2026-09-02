"""Shared helpers for the two vision phases (transcribe, redescribe)."""
from __future__ import annotations

import base64
import json
import re
import subprocess

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
VISION_MODEL = "qwen3-vl:8b"

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_OPEN_THINK = re.compile(r"<think>.*", re.S | re.I)


def strip_think(text: str) -> str:
    text = _THINK.sub("", text)
    # unclosed <think> that never terminated: drop it, keep anything after a
    # blank line (the model usually answers after the reasoning)
    if "<think>" in text.lower():
        tail = _OPEN_THINK.sub("", text)
        text = tail if tail.strip() else text
    return text.strip()


def ask_vision(image_path: str, prompt: str, *, num_predict: int = 1200,
               timeout: int = 300) -> dict:
    img_b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    payload = {
        "model": VISION_MODEL, "prompt": prompt, "images": [img_b64],
        "stream": False, "think": False,
        "options": {"num_predict": num_predict, "num_ctx": 16384, "temperature": 0.15},
    }
    try:
        r = subprocess.run(
            ["curl", "-s", "-m", str(timeout), OLLAMA_URL, "--data-binary", "@-"],
            input=json.dumps(payload), capture_output=True, text=True, timeout=timeout + 10,
        )
        d = json.loads(r.stdout.strip().split("\n")[0])
    except Exception as e:
        return {"text": "", "error": str(e)}
    text = (d.get("response") or "").strip() or (d.get("thinking") or "").strip()
    return {"text": strip_think(text), "eval_count": d.get("eval_count")}
