"""Thin Anthropic Messages API client for the vision phases.

Mirrors pipeline/vision.py's ask_vision() interface (curl + subprocess, same
as the Ollama client) so it's a drop-in swap, not a new pattern. Needs
ANTHROPIC_API_KEY in the environment -- this is Console/API billing, separate
from a claude.ai Pro/Max subscription's usage credits.

    from pipeline.claude_vision import ask_claude
    res = ask_claude(image_path, system=..., prompt=...)
    res["text"], res["usage"]  # {"input_tokens": n, "output_tokens": n}
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import subprocess

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

# model ids as of this build; override with CLAUDE_VISION_MODEL if these drift
DEFAULT_MODEL = os.environ.get("CLAUDE_VISION_MODEL", "claude-sonnet-5")

# https://www.anthropic.com/claude/fable and /sonnet -- $/million tokens.
# Used only to report an estimated running cost; never affects behavior.
PRICE_PER_MTOK = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (15.00, 75.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = PRICE_PER_MTOK.get(model, (3.00, 15.00))
    return input_tokens / 1e6 * in_rate + output_tokens / 1e6 * out_rate


def ask_claude(image_path: str, *, system: str, prompt: str,
               model: str = DEFAULT_MODEL, max_tokens: int = 2000,
               timeout: int = 120) -> dict:
    """One image + one text turn. Returns
    {"text", "usage": {"input_tokens","output_tokens"}, "cost", "error"}.

    No `temperature` param -- claude-sonnet-5 rejects it outright ("temperature
    is deprecated for this model"), which failed every call with $0 billed
    (Anthropic rejects at validation, before any generation happens)."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return {"text": "", "error": "ANTHROPIC_API_KEY not set"}

    media_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    img_b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                             "data": img_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    try:
        r = subprocess.run(
            ["curl", "-s", "-m", str(timeout), API_URL,
             "-H", f"x-api-key: {key}",
             "-H", f"anthropic-version: {API_VERSION}",
             "-H", "content-type: application/json",
             "--data-binary", "@-"],
            input=json.dumps(payload), capture_output=True, text=True, timeout=timeout + 10,
        )
        d = json.loads(r.stdout)
    except Exception as e:
        return {"text": "", "error": f"request failed: {e}"}

    if "error" in d:
        return {"text": "", "error": d["error"].get("message", str(d["error"]))}

    text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
    usage = d.get("usage", {})
    cost = estimate_cost(model, usage.get("input_tokens", 0), usage.get("output_tokens", 0))
    return {"text": text, "usage": usage, "cost": cost, "stop_reason": d.get("stop_reason")}
