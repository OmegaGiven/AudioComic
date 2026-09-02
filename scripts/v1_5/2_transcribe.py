#!/usr/bin/env python3
"""2_transcribe.py <work_dir>

Stage 2 of pipeline v1.5. One vision pass per panel, deliberately close to the
v1 prompt (which reliably read every bubble in order) with two changes:

  1. The description must NOT name any character -- appearance only. (v1's one
     real failure was naming: it called Black Hand "Batman".)
  2. Speech balloons are always tagged SPEAKER. Naming is a later job, done
     from the dialogue text itself, not guessed from the art.

Small models (qwen3-vl:8b) fall apart on heavily structured prompts, so this
keeps it plain: a short description, then one text line per bubble.

Reads   <work_dir>/manifest.json
Writes  <work_dir>/transcript.json
  { "<key>": {"lines": [{"kind": "CAPTION"|"SFX"|"DIALOGUE",
                         "speaker": None|"NAME", "text": "..."}],
              "scene": "..."} }
Resumable -- checkpoints after every panel.
"""
import base64
import json
import re
import subprocess
import sys
from pathlib import Path

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3-vl:8b"

PROMPT = """First, describe this comic panel in 1-2 sentences: the setting, and each person by appearance and action (clothing, posture, expression). Do NOT name any character, hero, villain, or franchise -- refer to people only by how they look ("a hooded man", "a woman in a red coat"). The only exception is a name actually printed on a costume or a nameplate in the panel.

Then transcribe EVERY piece of lettered text in the panel, in reading order, one item per line, each written as:
CAPTION: <exact text>   -- a narration box / caption (not spoken by a character)
SPEAKER: <exact text>   -- a speech balloon
SFX: <exact text>       -- sound-effect lettering

Copy the text exactly as lettered. Do not fix spelling, do not paraphrase, do not skip anything, do not explain your choices. If a panel has no text, transcribe nothing.

Respond with the description, then the transcription lines. Nothing else."""

THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
LINE_RE = re.compile(r"^\s*[-*]?\s*(CAPTION|SFX|SPEAKER|[A-Z][A-Z0-9 '.\-]{0,24})\s*:\s*(.+?)\s*$")
NON_NAMES = {"CAPTION", "SFX", "SPEAKER", "NAME", "CHARACTER", "NARRATOR",
             "DESCRIPTION", "SCENE", "PART", "NOTE"}


def analyze(image_path: str) -> dict:
    img_b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    payload = {
        "model": MODEL, "prompt": PROMPT, "images": [img_b64],
        "stream": False, "think": False,
        "options": {"num_predict": 1200, "num_ctx": 16384, "temperature": 0.1},
    }
    try:
        r = subprocess.run(
            ["curl", "-s", "-m", "300", OLLAMA_URL, "--data-binary", "@-"],
            input=json.dumps(payload), capture_output=True, text=True, timeout=310,
        )
        d = json.loads(r.stdout.strip().split("\n")[0])
    except Exception as e:
        return {"lines": [], "scene": "", "error": str(e)}

    text = (d.get("response") or "").strip() or (d.get("thinking") or "").strip()
    text = THINK_RE.sub("", text).strip()
    return parse(text)


def parse(text: str) -> dict:
    scene_lines, tx_lines = [], []
    for ln in text.splitlines():
        raw = ln.strip()
        if not raw:
            continue
        if LINE_RE.match(raw):
            tx_lines.append(raw)
        elif not tx_lines:  # description comes before the first tagged line
            scene_lines.append(raw)

    scene = " ".join(scene_lines)
    scene = re.sub(r"^(here('?s)?|description|scene)[:,\s-]+", "", scene, flags=re.I)
    scene = re.sub(r"^(the panel shows|this panel shows|in this panel)[:,]?\s*", "", scene, flags=re.I)
    scene = re.sub(r"\s+", " ", scene).strip()

    lines = []
    for raw in tx_lines:
        m = LINE_RE.match(raw)
        tag, body = m.group(1).strip().upper(), m.group(2).strip()
        body = body.strip('"“” ')
        if not body:
            continue
        if tag == "CAPTION":
            lines.append({"kind": "CAPTION", "speaker": None, "text": body})
        elif tag == "SFX":
            lines.append({"kind": "SFX", "speaker": None, "text": body})
        elif tag in NON_NAMES:
            lines.append({"kind": "DIALOGUE", "speaker": None, "text": body})
        else:
            lines.append({"kind": "DIALOGUE", "speaker": tag, "text": body})
    return {"lines": lines, "scene": scene}


def main():
    if len(sys.argv) != 2:
        print("Usage: 2_transcribe.py <work_dir>", file=sys.stderr)
        sys.exit(2)

    work_dir = Path(sys.argv[1])
    manifest = json.load(open(work_dir / "manifest.json"))
    out_path = work_dir / "transcript.json"
    transcript = json.load(open(out_path)) if out_path.exists() else {}

    todo = []
    for page in manifest["pages"]:
        for panel in page["panels"]:
            key = f"page{page['page_index']:03d}_panel{panel['panel_index']:02d}"
            if key not in transcript:
                todo.append((key, panel["file"]))
    print(f"{len(todo)} panels to transcribe.")

    for n, (key, file) in enumerate(todo):
        res = analyze(file)
        transcript[key] = res
        json.dump(transcript, open(out_path, "w"), indent=2)
        print(f"[{n+1}/{len(todo)}] {key}: {len(res.get('lines', []))} text lines"
              + (f"  ERR {res['error']}" if res.get("error") else ""))

    print(f"Done. Transcript: {out_path}")


if __name__ == "__main__":
    main()
