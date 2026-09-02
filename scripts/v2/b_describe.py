#!/usr/bin/env python3
"""b_describe.py <work_dir>

Stage B of pipeline v2. Per-panel *visual description only* -- no character
names, no dialogue (Magi already has the text and the speakers). A small VLM
is fine for this; the job is "what does this panel look like", not "who is
this and what franchise".

Reads   <work_dir>/structure.json
Writes  <work_dir>/descriptions.json   {"<page>": {"<panel>": "text", ...}, ...}
Resumable -- checkpoints after every panel.
"""
import base64
import io
import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3-vl:8b"

PROMPT = """Describe what is visibly happening in this single comic panel in 2-3 plain sentences: the setting, how many figures are present and what they physically look like (clothing, posture, expression), and the action.

Hard rules:
- Do NOT name any character, hero, villain, franchise, or publisher. You do not know who anyone is. Refer to people only by appearance: "a hooded figure", "a woman in a red coat", "an armoured man".
- Do NOT transcribe or mention any speech, caption, sound effect, or lettering. Ignore all text in the image.
- Do NOT speculate about story, motivation, or what happens next. Only what is shown.
- No preamble, no reasoning. Just the description."""

THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def describe(img: Image.Image) -> dict:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "images": [base64.b64encode(buf.getvalue()).decode()],
        "stream": False,
        "think": False,
        "options": {"num_predict": 400, "num_ctx": 8192, "temperature": 0.2},
    }
    try:
        r = subprocess.run(
            ["curl", "-s", "-m", "180", OLLAMA_URL, "--data-binary", "@-"],
            input=json.dumps(payload), capture_output=True, text=True, timeout=190,
        )
        d = json.loads(r.stdout.strip().split("\n")[0])
    except Exception as e:
        return {"text": "", "error": str(e)}

    text = (d.get("response") or "").strip()
    if not text:
        text = (d.get("thinking") or "").strip()
    # strip any leaked reasoning and any line that still names something
    text = THINK_RE.sub("", text).strip()
    text = re.split(r"\n\s*\n", text)[0].strip()
    return {"text": text, "eval_count": d.get("eval_count")}


def main():
    if len(sys.argv) != 2:
        print("Usage: b_describe.py <work_dir>", file=sys.stderr)
        sys.exit(2)

    work_dir = Path(sys.argv[1])
    structure = json.load(open(work_dir / "structure.json"))
    out_path = work_dir / "descriptions.json"
    descriptions = json.load(open(out_path)) if out_path.exists() else {}

    todo = []
    for page in structure["pages"]:
        pi = str(page["page_index"])
        img_full = Image.open(page["image"])
        panels = page["panels"] or [{"panel_index": 0,
                                     "bbox": [0, 0, page["width"], page["height"]]}]
        for panel in panels:
            key = str(panel["panel_index"])
            if descriptions.get(pi, {}).get(key):
                continue
            todo.append((pi, key, img_full, panel["bbox"]))

    print(f"{len(todo)} panels to describe.")
    for n, (pi, key, img_full, bbox) in enumerate(todo):
        x1, y1, x2, y2 = bbox
        crop = img_full.crop((int(x1), int(y1), int(x2), int(y2)))
        res = describe(crop)
        descriptions.setdefault(pi, {})[key] = res.get("text", "")
        json.dump(descriptions, open(out_path, "w"), indent=2)
        status = "OK" if res.get("text") else f"FAILED {res.get('error')}"
        print(f"[{n+1}/{len(todo)}] page {pi} panel {key}: {status}")

    print(f"Done. Descriptions: {out_path}")


if __name__ == "__main__":
    main()
