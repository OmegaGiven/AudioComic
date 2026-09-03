"""Tiny text-only Ollama client, shared by the deterministic-plus-LLM phases."""
from __future__ import annotations

import json
import subprocess

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


def ask_llm(prompt: str, *, model: str = "devstral:24b", num_predict: int = 1200,
            timeout: int = 240, temperature: float = 0.3) -> str:
    payload = {
        "model": model, "prompt": prompt, "stream": False, "think": False,
        "options": {"num_predict": num_predict, "num_ctx": 16384,
                    "temperature": temperature},
    }
    try:
        r = subprocess.run(
            ["curl", "-s", "-m", str(timeout), OLLAMA_URL, "--data-binary", "@-"],
            input=json.dumps(payload), capture_output=True, text=True,
            timeout=timeout + 10,
        )
        return (json.loads(r.stdout.strip().split("\n")[0]).get("response") or "").strip()
    except Exception:
        return ""
