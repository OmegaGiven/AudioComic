#!/usr/bin/env python3
"""c_narrative.py <work_dir>

Stage C of pipeline v2. Merges Magi's structure (dialogue + speakers) with the
VLM's panel descriptions into flowing audiobook narration, page by page, using
a prose-tuned model (Mistral-Nemo by default -- swap MODEL for another).

Reads   <work_dir>/structure.json, <work_dir>/descriptions.json
Writes  <work_dir>/narrative.json   {"<page>": [{"speaker","text"}, ...], ...}
  -- same format the Kokoro stage 4 (04_tts_render_kokoro.py) consumes.

Name handling: Magi gives speakers as real names only if a character bank was
supplied; otherwise "Character 1" etc. This stage is told it MAY replace a
"Character N" label with a real name **only when that name is stated in the
dialogue itself** (e.g. someone is addressed by name). It must never invent a
name from vibes -- an unresolved speaker stays a description ("a hooded man").
"""
import json
import re
import subprocess
import sys
from pathlib import Path

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "mistral-nemo:12b"

PLACEHOLDER_SPEAKERS = {
    "SPEAKER", "CHARACTER", "VILLAIN", "MAN", "WOMAN", "ENTITY", "VOICE",
    "PERSON", "FIGURE", "STRANGER", "UNKNOWN", "CAPTION",
}

PROMPT_TEMPLATE = """You are adapting one comic page into flowing audiobook narration.

Below, in reading order, each panel has:
  DESCRIPTION: what the panel looks like (no names -- the describer wasn't told who anyone is)
  DIALOGUE: lines actually lettered on the page, with the speaker Magi attributed

This page has {panel_count} panels. Cover every panel, in order, none skipped.

Write it as a novelist would narrate the scene happening -- NOT "this panel shows".
- Weave the descriptions into narration.
- Keep the dialogue lines as spoken lines, using the speaker labels given.
- A speaker labelled "Character 1", "Character 2" etc: keep that label UNLESS a
  real name for that person is spoken in the dialogue on this page (someone is
  addressed by name, or names themselves). Then use the real name. Never invent
  a name that isn't in the text -- if you don't know, write the narration around
  them as "the hooded man" / "the woman in the red coat" using the description.
- Caption / narration-box text (speaker NARRATOR) becomes narration.

Output ONLY lines in this exact format, nothing else:
NARRATOR: narration text
NAME: dialogue text

Panels:
{panels}"""


def build_page_prompt(page, descriptions) -> tuple[str, int]:
    pi = str(page["page_index"])
    panels = page["panels"] or [{"panel_index": 0}]
    texts_by_panel = {}
    for t in page["texts"]:
        if not t.get("essential", True):
            continue  # drop SFX / watermarks / garbled OCR outright
        texts_by_panel.setdefault(t.get("panel_index"), []).append(t)

    blocks = []
    for panel in panels:
        idx = panel["panel_index"]
        desc = descriptions.get(pi, {}).get(str(idx), "(no description)")
        lines = [f"[Panel {idx + 1}]", f"DESCRIPTION: {desc}"]
        for t in texts_by_panel.get(idx, []):
            spk = t.get("speaker") or "NARRATOR"
            lines.append(f"DIALOGUE {spk}: {t['text']}")
        blocks.append("\n".join(lines))
    # any essential text Magi couldn't place in a panel -> list under the page
    for t in texts_by_panel.get(None, []):
        blocks.append(f"[unplaced] DIALOGUE {t.get('speaker') or 'NARRATOR'}: {t['text']}")
    return "\n\n".join(blocks), len(panels)


def generate(prompt: str) -> str:
    payload = {"model": MODEL, "prompt": prompt, "stream": False,
               "options": {"num_predict": 2000, "num_ctx": 16384, "temperature": 0.7}}
    try:
        r = subprocess.run(
            ["curl", "-s", "-m", "300", OLLAMA_URL, "--data-binary", "@-"],
            input=json.dumps(payload), capture_output=True, text=True, timeout=310,
        )
        return (json.loads(r.stdout.strip().split("\n")[0]).get("response") or "").strip()
    except Exception:
        return ""


def parse(raw: str):
    segs = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([A-Z][A-Za-z0-9 '.\-]{1,40}):\s*(.+)$", line)
        if m:
            spk = m.group(1).strip()
            if spk.upper() in PLACEHOLDER_SPEAKERS:
                spk = "NARRATOR"
            segs.append({"speaker": spk.upper() if spk.isupper() or " " in spk else spk,
                         "text": m.group(2).strip()})
        elif segs:
            segs[-1]["text"] += " " + line
    return segs


def main():
    if len(sys.argv) != 2:
        print("Usage: c_narrative.py <work_dir>", file=sys.stderr)
        sys.exit(2)

    work_dir = Path(sys.argv[1])
    structure = json.load(open(work_dir / "structure.json"))
    descriptions = json.load(open(work_dir / "descriptions.json"))
    narr_path = work_dir / "narrative.json"
    narrative = json.load(open(narr_path)) if narr_path.exists() else {}

    pages = structure["pages"]
    todo = [p for p in pages if str(p["page_index"]) not in narrative]
    print(f"{len(pages)} pages, {len(todo)} to narrate.")

    for n, page in enumerate(todo):
        panels_text, panel_count = build_page_prompt(page, descriptions)
        prompt = PROMPT_TEMPLATE.format(panel_count=panel_count, panels=panels_text)
        segs = parse(generate(prompt))
        if len(segs) < panel_count:
            segs2 = parse(generate(prompt + "\n\nYour last attempt stopped early. "
                                            f"Cover all {panel_count} panels."))
            if len(segs2) > len(segs):
                segs = segs2
        narrative[str(page["page_index"])] = segs
        json.dump(narrative, open(narr_path, "w"), indent=2)
        print(f"[{n+1}/{len(todo)}] page {page['page_index']}: {len(segs)} segments")

    print(f"Done. Narrative: {narr_path}")


if __name__ == "__main__":
    main()
