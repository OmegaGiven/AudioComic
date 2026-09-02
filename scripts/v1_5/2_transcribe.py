#!/usr/bin/env python3
"""2_transcribe.py <work_dir>

Stage 2 of pipeline v1.5. Per panel, one vision pass that does two clearly
separated jobs:

  PART 1 - TRANSCRIPTION: every lettered element, in reading order, verbatim,
           tagged CAPTION / <NAME> / SFX. This is authoritative -- the page's
           own words are the script.
  PART 2 - SCENE: 1-2 sentences on what the panel looks like. Appearance only,
           no names unless printed, no restating the text. This is the only
           part the model is allowed to be "creative" about, and it stays
           short so a weak description can't drown the real words.

The v1 lesson: the captions and dialogue on the page were always right and
should be used verbatim; only the model's scene guesses were shaky. So v1.5
leans hard on Part 1 and keeps Part 2 minimal.

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

PROMPT = """You are transcribing ONE comic panel for an audiobook. Do two parts.

PART 1 - TRANSCRIPTION
List every piece of lettered text in the panel, in natural reading order, one per line, each tagged:
  CAPTION: <exact text>     -- a narration / caption box (usually rectangular, often yellow). NOT spoken by a character.
  NAME: <exact text>        -- a speech balloon. For NAME use the character's name ONLY if it is printed in this panel (on a costume, a nameplate) OR the speaker is directly named in the dialogue. Otherwise write SPEAKER.
  SFX: <exact text>         -- sound-effect lettering (BOOM, KRA-KOW, a drawn-out AAAH).
Rules: copy the text EXACTLY as lettered. Do not correct spelling. Do not paraphrase, summarise, translate, or skip anything. Keep a balloon that continues another as its own line, in order. If the panel has no text, write: (none)

PART 2 - SCENE
Write 1-2 short sentences: the setting, and each person by appearance only (clothing, posture, expression, what they are doing). Do NOT name anyone unless a name is printed on them. Do NOT repeat any words from Part 1. Do NOT guess story, history, or motive. Do NOT write "the panel shows".

Respond with exactly:
PART 1
<lines>
PART 2
<sentences>

Nothing else. No reasoning, no preamble."""

THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
LINE_RE = re.compile(r"^\s*(CAPTION|SFX|[A-Z][A-Z0-9 '.\-]{0,30})\s*:\s*(.+?)\s*$")


def analyze(image_path: str) -> dict:
    img_b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    payload = {
        "model": MODEL, "prompt": PROMPT, "images": [img_b64],
        "stream": False, "think": False,
        "options": {"num_predict": 900, "num_ctx": 16384, "temperature": 0.1},
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
    # split PART 1 / PART 2
    parts = re.split(r"^\s*PART\s*2\s*$", text, maxsplit=1, flags=re.I | re.M)
    p1 = re.sub(r"^\s*PART\s*1\s*$", "", parts[0], flags=re.I | re.M).strip()
    scene = parts[1].strip() if len(parts) > 1 else ""
    scene = re.sub(r'^(the panel shows|this panel|in this panel)[:,]?\s*', "", scene, flags=re.I)
    scene = re.sub(r"\s+", " ", scene).strip()

    lines = []
    for ln in p1.splitlines():
        ln = ln.strip().lstrip("-*").strip()
        if not ln or ln.lower() in ("(none)", "none"):
            continue
        m = LINE_RE.match(ln)
        if not m:
            continue
        tag, body = m.group(1).strip(), m.group(2).strip()
        if tag.upper() == "CAPTION":
            lines.append({"kind": "CAPTION", "speaker": None, "text": body})
        elif tag.upper() == "SFX":
            lines.append({"kind": "SFX", "speaker": None, "text": body})
        else:
            spk = None if tag.upper() == "SPEAKER" else tag.upper()
            lines.append({"kind": "DIALOGUE", "speaker": spk, "text": body})
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
            if not transcript.get(key, {}).get("lines") and key not in transcript:
                todo.append((key, panel["file"]))
    print(f"{len(todo)} panels to transcribe.")

    for n, (key, file) in enumerate(todo):
        res = analyze(file)
        transcript[key] = res
        json.dump(transcript, open(out_path, "w"), indent=2)
        nl = len(res.get("lines", []))
        print(f"[{n+1}/{len(todo)}] {key}: {nl} text lines"
              + (f"  ERR {res['error']}" if res.get("error") else ""))

    print(f"Done. Transcript: {out_path}")


if __name__ == "__main__":
    main()
