#!/usr/bin/env python3
"""c_narrative.py <work_dir>

Stage C of pipeline v2. Builds narrative.json from Magi's structure + the
VLM's panel descriptions.

Design rule (learned the hard way): **the language model never writes a
spoken line.** Dialogue is copied verbatim from Magi's OCR with Magi's
speaker attribution. The model only writes NARRATOR connective prose from the
panel descriptions -- so it cannot invent dialogue, add "(whispering)" stage
directions, or drop real bubbled lines.

Reads   <work_dir>/structure.json, <work_dir>/descriptions.json
Writes  <work_dir>/narrative.json   {"<page>": [{"speaker","text"}, ...], ...}
        -- same format 04_tts_render_kokoro.py consumes.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from panelspeak.classify import refine_kind  # noqa: E402
from panelspeak.text_elements import ElementKind  # noqa: E402

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "mistral-nemo:12b"

# lines that are never story content, whatever Magi flagged them
JUNK_RE = re.compile(
    r"decomics\.com|dccomics\.com|conversion by|first issue of eight|"
    r"^\s*\[\d\d:\d\d\]|<[a-z]+>|warrts|good-ghs|^\s*if\s*$|"
    r"\b\d+%\s+of\s+\w+\s+seconds?\b|population:\s*[\d,]+|the city without fear",
    re.I,
)

NARR_PROMPT = """Write 1-2 sentences of audiobook narration for ONE comic panel, based only on this visual description:

{desc}

Rules:
- Narrate the scene as a novelist would ("Rain hammered the cemetery."), NOT "the panel shows".
- Do NOT write any dialogue or quoted speech. Someone else handles the spoken lines.
- Do NOT name any character, hero, or franchise. Use appearance only ("the hooded man").
- Do NOT add stage directions like "(whispering)".
- If the description is just cover art / a logo / nothing is happening, reply with exactly: SKIP
Output only the narration sentence(s), nothing else."""

VOCATIVE_RE = re.compile(r"[,\"']\s*([A-Z][a-z]{2,15})\.?[\"']?\s*$")
SAID_NAME_RE = re.compile(r'said,?\s*["\u201c][^"\u201d]*,\s*([A-Z][a-z]{2,15})\.?["\u201d]')


def clean_line(t: str) -> str:
    t = re.sub(r"\([^)]*\)", "", t)          # drop (stage directions)
    t = t.strip().strip('"\u201c\u201d ').strip()
    t = re.sub(r"\s+", " ", t)
    return t


def is_junk(t: str) -> bool:
    if JUNK_RE.search(t):
        return True
    letters = re.sub(r"[^A-Za-z]", "", t)
    if len(letters) < 2:
        return True
    # single repeated-token OCR gibberish ("Fish Fish Fish", "01/01/01...")
    toks = t.split()
    if len(toks) >= 3 and len(set(toks)) == 1:
        return True
    if re.fullmatch(r"[\d/ ]+", t):
        return True
    return False


def resolve_names(structure) -> dict:
    """Cluster label -> real name, if a name is spoken at/about that speaker."""
    names = {}
    for page in structure["pages"]:
        for t in page["texts"]:
            spk = t.get("speaker")
            if not spk or not spk.startswith("Character"):
                continue
            txt = t["text"]
            m = SAID_NAME_RE.search(txt) or VOCATIVE_RE.search(txt)
            if m:
                cand = m.group(1)
                if cand.lower() not in {"yes", "no", "sir", "please", "well"}:
                    names.setdefault(spk, cand)
    if names:
        print(f"resolved names: {names}")
    return names


def narrate_panel(desc: str) -> str:
    if not desc or desc == "(no description)":
        return ""
    payload = {"model": MODEL, "prompt": NARR_PROMPT.format(desc=desc),
               "stream": False,
               "options": {"num_predict": 160, "num_ctx": 4096, "temperature": 0.6}}
    try:
        r = subprocess.run(
            ["curl", "-s", "-m", "120", OLLAMA_URL, "--data-binary", "@-"],
            input=json.dumps(payload), capture_output=True, text=True, timeout=125,
        )
        out = (json.loads(r.stdout.strip().split("\n")[0]).get("response") or "").strip()
    except Exception:
        return ""
    out = re.split(r"\n\s*\n", out)[0].strip().strip('"')
    if out.upper().startswith("SKIP") or not out:
        return ""
    # belt-and-suspenders: no quotes/parens leaked in
    return re.sub(r"\([^)]*\)", "", out).replace('"', "").strip()


def page_texts(page, names):
    """Ordered (kind, speaker, text) for a page. kind in {DIALOGUE, NARRATION}."""
    items = []
    for t in page["texts"]:
        raw = t["text"]
        if is_junk(raw):
            continue
        txt = clean_line(raw)
        if not txt:
            continue
        spk = t.get("speaker")
        # Magi's is_essential flag is noisy: it drops real dialogue (keep those
        # -- they have a speaker) but it's right about junk fragments. So only
        # trust a False when the text is also short and unattributed.
        if not t.get("essential", True) and not spk and len(txt.split()) < 5:
            continue
        # SFX slipped through OCR -> drop
        if refine_kind("SFX" if not spk else "DIALOGUE", txt,
                       in_bubble=bool(spk)) is ElementKind.SFX:
            continue
        if spk and spk.startswith("Character"):
            spk = names.get(spk, spk)
        pidx = t.get("panel_index") if t.get("panel_index") is not None else 999
        if spk:
            items.append((pidx, "DIALOGUE", spk, txt))
        else:
            # unattributed -> narration box (location captions, off-panel voice-over)
            items.append((pidx, "NARRATION", "NARRATOR", txt))
    items.sort(key=lambda x: x[0])
    return items


def merge_runs(segs):
    out = []
    for s in segs:
        if out and out[-1]["speaker"] == s["speaker"]:
            out[-1]["text"] = f"{out[-1]['text']} {s['text']}".strip()
        else:
            out.append(dict(s))
    return out


def main():
    if len(sys.argv) != 2:
        print("Usage: c_narrative.py <work_dir>", file=sys.stderr)
        sys.exit(2)

    work_dir = Path(sys.argv[1])
    structure = json.load(open(work_dir / "structure.json"))
    descriptions = json.load(open(work_dir / "descriptions.json"))
    names = resolve_names(structure)

    narrative = {}
    for page in structure["pages"]:
        pi = str(page["page_index"])
        texts = page_texts(page, names)
        real_text = [t for t in texts if len(t[3]) > 3]
        # front matter: one "panel", basically no text -> skip
        if len(page["panels"]) <= 1 and len(real_text) <= 1:
            print(f"page {pi}: front matter / empty ({len(real_text)} texts) -- skipped")
            continue

        by_panel = {}
        for pidx, kind, spk, txt in texts:
            by_panel.setdefault(pidx, []).append((kind, spk, txt))

        segs = []
        panels = page["panels"] or [{"panel_index": 0}]
        seen_panels = [p["panel_index"] for p in panels]
        for pidx in seen_panels + [x for x in by_panel if x not in seen_panels]:
            desc = descriptions.get(pi, {}).get(str(pidx), "")
            narr = narrate_panel(desc)
            if narr:
                segs.append({"speaker": "NARRATOR", "text": narr})
            for _kind, spk, txt in by_panel.get(pidx, []):
                segs.append({"speaker": spk.upper() if spk == "NARRATOR" else spk,
                             "text": txt})

        segs = merge_runs(segs)
        narrative[pi] = segs
        json.dump(narrative, open(work_dir / "narrative.json", "w"), indent=2)
        print(f"page {pi}: {len(segs)} segments "
              f"({sum(1 for s in segs if s['speaker'] != 'NARRATOR')} dialogue)")

    print(f"Done. Narrative: {work_dir / 'narrative.json'}")


if __name__ == "__main__":
    main()
