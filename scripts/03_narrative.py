#!/usr/bin/env python3
"""03_narrative.py <work_dir>

Turns raw per-panel vision analysis into flowing audiobook-style narrative
prose, page by page (page = the natural narrative unit -- matches how the
comic itself is paced). Output format per line: "SPEAKER: text" where
SPEAKER is "NARRATOR" for descriptive prose or a character's name for
dialogue -- same simple format already proven in the vision-analysis phase,
picked over JSON because local models produce it far more reliably.

Resumable: checkpoints after every page to narrative.json.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "devstral:24b"

PROMPT_TEMPLATE = """You are adapting a comic book page into a flowing audiobook narration. Below is the raw panel-by-panel analysis of one page, in reading order (each panel has a scene description and any dialogue/captions found in it).

This page has exactly {panel_count} panels, numbered [Panel 1] through [Panel {panel_count}] below. You MUST produce narration/dialogue content derived from EVERY one of these {panel_count} panels, in order, with none skipped or merged away -- a real story beat (a death, a plot reveal, a new scene) must never be silently dropped just because it's inconvenient to transition into. Do not stop early. If you have covered fewer than all {panel_count} panels, you are not done.

Turn this into a natural, flowing narrative for an audiobook -- NOT a dry panel-by-panel recap. Weave the scene descriptions into narration and keep the actual dialogue lines as spoken lines. Use character names consistently (if a character is named anywhere in this page's data, use their real name every time they speak, not a generic label).

CRITICAL: the raw panel data below is written like art description ("this panel depicts...", "the image shows...") because that's how it was extracted -- do NOT carry that framing into your output. You are not describing a comic page to someone, you are narrating the STORY that is happening, the way a novelist would. Never write "this panel shows", "the image depicts", "in this scene we see", "the panel focuses on", or anything else that reminds the listener they're looking at a comic. Instead, just narrate what happens: characters act, speak, and things occur, directly, in-scene. For example, instead of "This panel depicts John Stewart flying through the air while dodging debris," write "John Stewart banked hard, debris screaming past him."

Output ONLY lines in this exact format, nothing else:
NARRATOR: narration text
CHARACTER NAME: dialogue text

Rules for the SPEAKER label:
- "NARRATOR" for all descriptive/narration text AND for the comic's own caption boxes (yellow/green narration boxes are still just narration -- do not invent a separate "CAPTION" speaker, fold that text into NARRATOR lines).
- The character's actual name in caps, ONLY for an actual spoken dialogue line from that character (a speech bubble).
- If a speaker truly can't be identified for a real spoken line, use "NARRATOR" and phrase it as reported speech, e.g. "NARRATOR: A voice cries out, 'text'". Never invent placeholder speaker labels like "SPEAKER", "CHARACTER", "VILLAIN", "MAN", or "ENTITY" -- if you don't have a real name, it's NARRATOR.
- Never include raw formatting artifacts from the source data (dashes used as line-break markers like "--", ellipses used the same way, etc.) -- write clean prose sentences instead.

Raw panel data for this page ({panel_count} panels total):
{panel_data}"""


def build_panel_data_text(page_entry, panel_analysis):
    lines = []
    for panel in page_entry["panels"]:
        key = f"page{page_entry['page_index']:03d}_panel{panel['panel_index']:02d}"
        text = panel_analysis.get(key, {}).get("text", "")
        if text:
            lines.append(f"[Panel {panel['panel_index'] + 1}]\n{text}")
    return "\n\n".join(lines), len(lines)


def generate_narrative(panel_data_text: str, panel_count: int) -> str:
    prompt = PROMPT_TEMPLATE.format(panel_data=panel_data_text, panel_count=panel_count)
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 2000, "num_ctx": 16384},
    }
    r = subprocess.run(
        ["curl", "-s", "-m", "300", OLLAMA_URL, "--data-binary", "@-"],
        input=json.dumps(payload), capture_output=True, text=True, timeout=310,
    )
    try:
        d = json.loads(r.stdout.strip().split("\n")[0])
    except Exception as e:
        return ""
    return d.get("response", "").strip()


# Placeholder labels the model invents despite being told not to -- caught
# deterministically here rather than relying on prompt compliance, which is
# real but imperfect (confirmed: "SPEAKER" still slipped through in testing
# even with an explicit "never use SPEAKER" instruction).
PLACEHOLDER_SPEAKERS = {
    "SPEAKER", "CHARACTER", "VILLAIN", "MAN", "WOMAN", "ENTITY", "VOICE",
    "PERSON", "FIGURE", "STRANGER", "UNKNOWN", "CAPTION",
}


def parse_narrative(raw_text: str):
    """Parses 'SPEAKER: text' lines into [{"speaker":..., "text":...}]."""
    segments = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([A-Z][A-Z0-9 '.\-]{1,40}):\s*(.+)$", line)
        if m:
            speaker = m.group(1).strip()
            if speaker in PLACEHOLDER_SPEAKERS:
                speaker = "NARRATOR"
            segments.append({"speaker": speaker, "text": m.group(2).strip()})
        elif segments:
            # Continuation of the previous line (model wrapped it).
            segments[-1]["text"] += " " + line
    return segments


def main():
    if len(sys.argv) != 2:
        print("Usage: 03_narrative.py <work_dir>", file=sys.stderr)
        sys.exit(2)

    work_dir = Path(sys.argv[1])
    manifest = json.load(open(work_dir / "manifest.json"))
    panel_analysis = json.load(open(work_dir / "panel_analysis.json"))
    narrative_path = work_dir / "narrative.json"

    narrative = {}
    if narrative_path.exists():
        narrative = json.load(open(narrative_path))

    pages = manifest["pages"]
    todo = [p for p in pages if str(p["page_index"]) not in narrative or not narrative[str(p["page_index"])]]
    print(f"{len(pages)} total pages, {len(todo)} remaining.")

    for i, page in enumerate(todo):
        panel_data_text, panel_count = build_panel_data_text(page, panel_analysis)
        if not panel_data_text:
            narrative[str(page["page_index"])] = []
            print(f"[{i+1}/{len(todo)}] page {page['page_index']}: no panel data, skipped")
            continue

        raw = generate_narrative(panel_data_text, panel_count)
        segments = parse_narrative(raw)

        # Real bug hit in testing: the model can silently stop after the
        # first panel or two instead of covering the whole page, dropping
        # real story beats. Heuristic: fewer segments than panels is a red
        # flag (even a sparse page should produce >=1 line per panel) --
        # retry once with a more forceful nudge before accepting it.
        if len(segments) < panel_count:
            print(f"  WARN: only {len(segments)} segments for {panel_count} panels, retrying with stronger prompt...")
            retry_prompt_suffix = (
                f"\n\nIMPORTANT: your previous attempt only covered part of this page and "
                f"stopped early. This page has {panel_count} panels -- you must cover ALL of them."
            )
            raw2 = generate_narrative(panel_data_text + retry_prompt_suffix, panel_count)
            segments2 = parse_narrative(raw2)
            if len(segments2) > len(segments):
                segments = segments2

        narrative[str(page["page_index"])] = segments

        json.dump(narrative, open(narrative_path, "w"), indent=2)
        if not segments:
            status = f"FAILED (raw len={len(raw)})"
        elif len(segments) < panel_count:
            status = f"WARN: only {len(segments)} segments for {panel_count} panels -- possible dropped content, review manually"
        else:
            status = "OK"
        print(f"[{i+1}/{len(todo)}] page {page['page_index']}: {len(segments)} segments {status}")

    print(f"All done. Narrative: {narrative_path}")


if __name__ == "__main__":
    main()
