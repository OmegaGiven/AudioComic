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

JUNK = re.compile(r"decomics\.com|dccomics\.com|conversion by|wildstorm|^\s*\[\d\d:\d\d\]|"
                  r"first issue of eight|issue \w+ of \w+|sep '?\d\d|^\s*\*?\s*20\d\d\s*$", re.I)

# cover / credits pages: OCR'd title treatment + creator names, no story text.
_FRAG = re.compile(r"^[^a-z]{0,40}$")  # no lowercase letters at all


def looks_like_credits(texts: list[str]) -> bool:
    txts = [t.strip() for t in texts if t.strip()]
    if len(txts) < 2:
        return False
    frag = sum(1 for t in txts
               if len(t.split()) <= 3 and _FRAG.match(t) and not t.rstrip()[-1:] in ".!?")
    return frag / len(txts) >= 0.7


_LEADIN = re.compile(r"^(in this (comic ?book )?(panel|image|scene),?|the (panel|image) shows|"
                     r"this (comic ?book )?panel( shows)?)\s*", re.I)


def clean(t: str) -> str:
    t = re.sub(r"\([^)]*\)", "", t)
    t = re.sub(r"\s+", " ", t).strip().strip('"“” ')
    t = _LEADIN.sub("", t)
    return t[:1].upper() + t[1:] if t else t


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


_DANGLING = re.compile(
    r"(?:[,:;]|--|—|\b(?:and|the|of|to|an?|in|on|at|with|but|or|for|as|so|"
    r"who|that|which|what|when|while|because|their|his|her|its|my|your|our|"
    r"you|we|i|he|she|it|they|me|him|us|them|this|these|can|will|would|could|should|"
    r"is|are|was|were|been|be|am|has|have|had|do|does|did|not|no|"
    r"about|from|into|than|then|like))$", re.I)


def _unfinished(t: str) -> bool:
    """The text ends mid-thought -- a comma / dash, a dangling function word,
    or (in a normal-case caption) a lowercase letter. A complete short line
    like "SPACE SECTOR 2814" is NOT unfinished even with no period."""
    t = t.strip().rstrip("\"'”’) ")
    return bool(_DANGLING.search(t) or (t[-1:].islower()))


def coalesce_blocks(blocks: list):
    """Per-panel OCR splits one caption box / one speech balloon across its
    lettered lines. Re-join a run of same-kind blocks into one, but only
    while the running text is clearly unfinished -- so two separate caption
    boxes, or two people's balloons, stay apart."""
    out = []
    for b in blocks:
        j = out[-1] if out else None
        # join fragments of one box/balloon while the text is still unfinished.
        # entity may disagree only because Magi clustered the split fragments
        # differently -- tolerate that unless BOTH carry a (different) id.
        entity_ok = j is not None and (j.entity == b.entity or not j.entity or not b.entity)
        joinable = (
            j is not None and j.kind == b.kind and j.kind in ("CAPTION", "DIALOGUE")
            and entity_ok and _unfinished(j.text_raw)
        )
        if joinable:
            left = re.sub(r"[-—]+\s*$", "", j.text_raw.rstrip())
            right = re.sub(r"^\s*[-—]+", "", b.text_raw.lstrip())
            j.text_raw = re.sub(r"\s+", " ", f"{left} {right}").strip()
            j.text_clean = j.text_raw
            j.entity = j.entity or b.entity
        else:
            out.append(b)
    return out


def looks_third_person(t: str) -> bool:
    return bool(re.search(r"\b(their|his|her)\b", t) and
               not re.search(r"\b(I|me|my|you|your|we|our)\b", t))


def sounds_spoken(t: str) -> bool:
    """A box the model tagged CAPTION but that is really a line of dialogue --
    2nd person plus a question or a direct address. Narration boxes are 3rd
    person and past tense; this catches a phone call lettered in caption
    boxes ("HOW CAN YOU EVEN THINK ABOUT IT, RAY?")."""
    first_second = re.search(r"\b(you|your|you're|you'?ll|i|i'?m|me|my|we|us)\b", t, re.I)
    if not first_second:
        return False
    vocative = re.search(r",\s*[A-Za-z]{2,}[.!?]*$", t)     # "..., Ray?" / "..., CARTER."
    return bool(t.rstrip().endswith(("?", "!")) or vocative
                or re.match(r"\s*(please|don'?t|do not|listen|look|wait|stop|get|give|tell|"
                            r"forget|hold)\b", t, re.I))


_SAFE_CAPS = set("""A An The And But Or So If Then As At By For From In Into Of On To Up With
He She It They We You His Her Its Their Our My Your This That There Here Now When While After
Before Behind Above Below Beside Between Near Over Under Through Across Amid Amidst Among Around
Against Atop Within Without Beyond Beneath Toward Towards Suddenly Meanwhile Later Nearby Outside
Inside Somewhere Nothing Something Someone Everyone Nobody One Two Three Four Five Six Seven
Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June
July August September October November December Several Many Both Each Every All Some Two""".split())
_PROPER = re.compile(r"\b([A-Z][a-z]{2,}|[A-Z]{3,})\b")


def scene_allowed(db: ComicDB) -> set[str]:
    """Proper nouns a scene line is allowed to name: resolved character names,
    plus any proper noun already lettered in a caption or a balloon."""
    ok = set(w.lower() for w in _SAFE_CAPS)
    for e in db.entities():
        if e.name:
            ok.update(w.lower() for w in re.findall(r"[A-Za-z]+", e.name))
    for b in db.blocks():
        if b.kind in ("CAPTION", "DIALOGUE"):
            ok.update(w.lower() for w in _PROPER.findall(b.text_raw))
    return ok


def strip_unknown_names(text: str, allowed: set[str]) -> str:
    """Remove a proper noun the vision model invented for a character
    ("...the figure, Abin Sur, lies...") -- keep the sentence, drop the name."""
    def repl(m: re.Match) -> str:
        tok = m.group(1)
        if tok.lower() in allowed:
            return tok
        pre = text[:m.start()].rstrip()
        if pre and pre[-1] not in ".!?:\"":          # mid-sentence -> a name
            return ""
        nxt = text[m.end():].lstrip()[:1]
        return "" if nxt.isupper() else tok          # "Gotham City" phrase
    out = _PROPER.sub(repl, text)
    out = re.sub(r"\s*,\s*,", ",", out)
    out = re.sub(r"\(\s*\)|\bthe\s*,", "the", out)
    out = re.sub(r"\s{2,}", " ", out).replace(" ,", ",").replace(" .", ".")
    out = re.sub(r",\s*(and\b|lies\b|stands\b|is\b|was\b|walks\b)", r" \1", out)
    # a phrase left dangling by a removed name
    out = re.sub(r"\b(bearing the name|which reads|labell?ed|named|marked|the words?)\s*[.,]?\s*$",
                 "", out, flags=re.I).strip(" ,")
    out = re.sub(r",?\s*(with|and)\s*$", "", out, flags=re.I)
    if out and not out.endswith((".", "!", "?", '"')):
        out += "."
    return out


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
        # merge only consecutive generated scene lines. Dialogue -- including
        # an unknown "A VOICE" -- stays one segment per line: joining two
        # different speakers' balloons was producing run-on gibberish.
        if same and out[-1].get("_gen") and s.get("_gen"):
            joiner = "" if out[-1]["text"].endswith((".", "!", "?", '"')) else "."
            out[-1]["text"] = f"{out[-1]['text']}{joiner} {s['text']}".strip()
        else:
            out.append(dict(s))
    # keep a persistent marker: narrate rewrites only generated scene lines,
    # never a verbatim caption or a line of dialogue
    for s in out:
        s["gen"] = bool(s.pop("_gen", False))
    return out


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m pipeline.assemble <work_dir>", file=sys.stderr)
        sys.exit(2)
    work_dir = Path(sys.argv[1])
    db = ComicDB.load(work_dir)
    front = {p.index for p in db.pages() if p.is_front_matter}
    allowed = scene_allowed(db)

    _VOICE = ["A VOICE", "A SECOND VOICE", "A THIRD VOICE", "A FOURTH VOICE"]
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
        if page.index <= 3 and looks_like_credits([b.text_raw for b in blocks_here]):
            print(f"  page {page.index}: skipped (looks like cover/credits)")
            continue

        segs: list[dict] = []
        voice_ord: dict[str, int] = {}
        for pn in panels:
            if pn.scene and len(pn.scene.split()) >= 4:
                sc = strip_unknown_names(clean(pn.scene), allowed)
                if len(sc.split()) >= 4:
                    segs.append({"speaker": "NARRATOR", "text": sc, "_gen": True})
            prev = segs[-1]["speaker"] if segs else "NARRATOR"
            for b in coalesce_blocks(db.blocks(panel=pn.id)):
                txt = clean(b.text_clean or b.text_raw)
                if not txt or JUNK.search(txt):
                    continue
                kind = b.kind
                if kind == "CAPTION" and sounds_spoken(txt):
                    kind = "DIALOGUE"     # a phone call lettered in caption boxes
                if kind == "CAPTION":
                    segs.append({"speaker": "NARRATOR", "text": txt})
                    prev = "NARRATOR"
                elif kind == "SFX":
                    added = sfx_seg(txt, prev)
                    if added:
                        segs.append({"speaker": added[0], "text": added[1]})
                else:  # DIALOGUE (incl. a reclassified caption)
                    ent = db.entity(b.entity) if b.entity else None
                    if b.kind == "CAPTION":
                        ent = None
                    name = ent.name if ent and ent.name else None
                    if name:
                        segs.append({"speaker": name, "text": txt})
                        prev = name
                    elif looks_third_person(txt):
                        segs.append({"speaker": "NARRATOR", "text": txt})
                        prev = "NARRATOR"
                    else:
                        # unknown speaker -> a distinct voice per clustered
                        # character, so a back-and-forth doesn't collapse to one
                        if b.entity:
                            n = voice_ord.setdefault(b.entity, len(voice_ord))
                            spk = _VOICE[n] if n < len(_VOICE) else "ANOTHER VOICE"
                        else:
                            spk = "A VOICE"
                        segs.append({"speaker": spk, "text": txt.strip('"“” ')})
                        prev = spk

        narrative[str(page.index)] = merge(dedupe(segs))

    (work_dir / "narrative.json").write_text(json.dumps(narrative, indent=2))
    pages = len(narrative)
    lines = sum(len(v) for v in narrative.values())
    dlg = sum(1 for v in narrative.values() for s in v if s["speaker"] != "NARRATOR")
    print(f"assemble: {pages} pages, {lines} segments ({dlg} dialogue)")
    print(f"done -> {work_dir / 'narrative.json'}")


if __name__ == "__main__":
    main()
