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
    toks = t.split()
    # creator credits: mixed-case name list, e.g. "Geoff JOHNS man REIS oclair ALBERT"
    caps = [w for w in toks if len(w) >= 3 and w.isupper()]
    title = [w for w in toks if len(w) >= 3 and w[0].isupper() and not w.isupper()]
    if len(toks) <= 8 and len(caps) >= 2 and title:
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
        essential = t.get("essential", True)
        # Magi's is_essential flag is noisy in both directions. Drop a
        # non-essential line only when it's *also* short (a sign / label /
        # OCR fragment), whether or not Magi stuck a speaker on it.
        if not essential and len(txt.split()) < 5:
            continue
        # SFX lettering: all-caps blurt, or panelspeak says SFX
        letters = re.sub(r"[^A-Za-z]", "", txt)
        allcaps_blurt = (txt == txt.upper() and len(txt.split()) == 1
                         and 2 <= len(letters) <= 8)
        if allcaps_blurt or refine_kind(
                "SFX" if not spk else "DIALOGUE", txt, in_bubble=bool(spk)) is ElementKind.SFX:
            continue
        # a "dialogue" line that reads as third-person narration is a mis-
        # attributed caption -> send it to the narrator
        if spk and re.search(r"\b(their|his|her|the .+ of|unmarked grave)\b", txt) \
                and not re.search(r"\b(I|me|my|you|your|we|our)\b", txt):
            spk = None
        if spk and spk.startswith("Character"):
            spk = names.get(spk, spk)
        pidx = t.get("panel_index") if t.get("panel_index") is not None else 999
        if spk:
            items.append((pidx, "DIALOGUE", spk, txt, essential))
        else:
            items.append((pidx, "NARRATION", "NARRATOR", txt, essential))
    items.sort(key=lambda x: x[0])
    return [it[:4] for it in items]


_INCOMPLETE_TAIL = re.compile(
    r"\b(and|or|but|the|a|an|of|to|in|with|who|which|that|was|were|is|are|as|by|for)\s*$",
    re.I,
)


def looks_incomplete(txt: str) -> bool:
    """OCR fragment that trails off mid-phrase ('Thomas and his wife, who was')."""
    t = txt.strip()
    if t.endswith((".", "!", "?", '"', "”", "--")):
        return False
    return len(t.split()) < 7 and bool(_INCOMPLETE_TAIL.search(t))


def merge_runs(segs):
    """Merge only consecutive *generated* narration (two panels of description
    with no dialogue between). Caption boxes (verbatim OCR) and every spoken
    line stay their own segment."""
    out = []
    for s in segs:
        if (out and out[-1]["speaker"] == "NARRATOR" == s["speaker"]
                and out[-1].get("_gen") and s.get("_gen")):
            out[-1]["text"] = f"{out[-1]['text']} {s['text']}".strip()
        else:
            out.append(dict(s))
    for s in out:
        s.pop("_gen", None)
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
        dialogue = [t for t in texts if t[1] == "DIALOGUE"]
        narration = [t for t in texts if t[1] == "NARRATION" and len(t[3].split()) >= 4]
        # front matter (cover / credits / recap): no attributed dialogue and no
        # real narration-box text -- just logos and art.
        if not dialogue and not narration:
            print(f"page {pi}: front matter / no story text -- skipped")
            continue

        by_panel = {}
        for pidx, kind, spk, txt in texts:
            if kind == "NARRATION" and looks_incomplete(txt):
                continue  # drop trailing-off OCR fragments
            by_panel.setdefault(pidx, []).append((kind, spk, txt))

        segs = []
        panels = page["panels"] or [{"panel_index": 0}]
        seen_panels = [p["panel_index"] for p in panels]
        for pidx in seen_panels + [x for x in by_panel if x not in seen_panels]:
            desc = descriptions.get(pi, {}).get(str(pidx), "")
            narr = narrate_panel(desc)
            if narr:
                segs.append({"speaker": "NARRATOR", "text": narr, "_gen": True})
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
