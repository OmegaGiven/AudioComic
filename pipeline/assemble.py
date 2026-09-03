"""Phase 6 -- deterministic assembly of comic.json -> narrative.json.

Nothing is paraphrased or invented. Per page (front matter skipped), per
panel in reading order:
  1. NARRATOR: the panel's scene description (Pass 2 if it ran)
  2. every block in order:
       CAPTION  -> NARRATOR: <verbatim>
       DIALOGUE -> <entity name or appearance>: <verbatim>   (reported speech
                   if the speaker is unknown and has no usable description)
       SFX      -> panelspeak: a vocalization folds into the prior speaker,
                   ambient SFX is dropped
Consecutive generated-narration lines merge; every spoken line stays its own
segment.

    python -m pipeline.assemble <work_dir>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from panelspeak.onomatopoeia import normalize_vocalization  # noqa: E402
from pipeline.comicdb import ComicDB  # noqa: E402

JUNK = re.compile(r"decomics\.com|dccomics\.com|conversion by|^\s*\[\d\d:\d\d\]|"
                  r"first issue of eight", re.I)


def clean(t: str) -> str:
    t = re.sub(r"\([^)]*\)", "", t)
    t = re.sub(r"\s+", " ", t).strip().strip('"“” ')
    return t


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def dedupe(segs: list[dict]) -> list[dict]:
    """Drop a segment whose text repeats one already emitted (kumiko panel
    splits re-OCR the same caption; recap pages repeat lines)."""
    seen: set[str] = set()
    out = []
    for s in segs:
        key = _norm(s["text"])
        if len(key) > 8 and key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def looks_third_person(t: str) -> bool:
    return bool(re.search(r"\b(their|his|her)\b", t) and
               not re.search(r"\b(I|me|my|you|your|we|our)\b", t))


def sfx_seg(text: str, prev_speaker: str):
    voc = normalize_vocalization(text)
    if voc and voc.is_known and prev_speaker and prev_speaker != "NARRATOR":
        if voc.prefer_narration:
            return ("NARRATOR", f"{prev_speaker} {voc.narration}.")
        return (prev_speaker, f"{voc.spoken}.")
    return None


def merge(segs: list[dict]) -> list[dict]:
    out: list[dict] = []
    for s in segs:
        same = out and out[-1]["speaker"] == s["speaker"]
        gen_run = same and out[-1].get("_gen") and s.get("_gen")
        voice_run = same and s["speaker"] == "A VOICE"
        if gen_run or voice_run:
            joiner = "" if out[-1]["text"].endswith((".", "!", "?", '"')) else "."
            out[-1]["text"] = f"{out[-1]['text']}{joiner} {s['text']}".strip()
        else:
            out.append(dict(s))
    for s in out:
        s.pop("_gen", None)
    return out


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m pipeline.assemble <work_dir>", file=sys.stderr)
        sys.exit(2)
    work_dir = Path(sys.argv[1])
    db = ComicDB.load(work_dir)
    front = {p.index for p in db.pages() if p.is_front_matter}

    narrative: dict[str, list[dict]] = {}
    for page in sorted(db.pages(), key=lambda p: p.index):
        if page.index in front:
            continue
        panels = sorted((p for p in db.panels() if p.page == page.index),
                        key=lambda p: p.index)
        blocks_here = [b for pn in panels for b in db.blocks(panel=pn.id)
                       if b.kind in ("CAPTION", "DIALOGUE") and not JUNK.search(b.text_raw)]
        if not blocks_here:
            continue

        segs: list[dict] = []
        for pn in panels:
            if pn.scene and len(pn.scene.split()) >= 4:
                segs.append({"speaker": "NARRATOR", "text": clean(pn.scene), "_gen": True})
            prev = segs[-1]["speaker"] if segs else "NARRATOR"
            for b in db.blocks(panel=pn.id):
                txt = clean(b.text_clean or b.text_raw)
                if not txt or JUNK.search(txt):
                    continue
                if b.kind == "CAPTION":
                    segs.append({"speaker": "NARRATOR", "text": txt})
                    prev = "NARRATOR"
                elif b.kind == "SFX":
                    added = sfx_seg(txt, prev)
                    if added:
                        segs.append({"speaker": added[0], "text": added[1]})
                else:  # DIALOGUE
                    ent = db.entity(b.entity) if b.entity else None
                    name = ent.name if ent and ent.name else None
                    if name:
                        segs.append({"speaker": name, "text": txt})
                        prev = name
                    elif looks_third_person(txt):
                        segs.append({"speaker": "NARRATOR", "text": txt})
                        prev = "NARRATOR"
                    else:
                        # unknown speaker -> a distinct voice, line spoken
                        # verbatim (no "a voice says" wrapper read by the narrator)
                        segs.append({"speaker": "A VOICE", "text": txt.strip('"“” ')})
                        prev = "A VOICE"

        narrative[str(page.index)] = merge(dedupe(segs))

    (work_dir / "narrative.json").write_text(json.dumps(narrative, indent=2))
    pages = len(narrative)
    lines = sum(len(v) for v in narrative.values())
    dlg = sum(1 for v in narrative.values() for s in v if s["speaker"] != "NARRATOR")
    print(f"assemble: {pages} pages, {lines} segments ({dlg} dialogue)")
    print(f"done -> {work_dir / 'narrative.json'}")


if __name__ == "__main__":
    main()
