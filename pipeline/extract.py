"""Phase 2 -- page-level extraction (replaces the old per-panel transcribe).

One vision call per page. The model enumerates the panels itself and, for
each, gives a description plus every piece of lettering in reading order with
a speaker-continuity tag. Keeping a whole page in one call preserves the
intra-page coherence (bubble order, who is talking to whom) that a
panel-at-a-time sweep loses.

Kumiko boxes (from segment) are only used to crop panels for the review UI,
and only when their count matches the model's panel count.

Idempotent: keyed by (page, model, PROMPT_V). A page whose cached vision
still matches is re-parsed from the stored raw response -- no inference.

    python -m pipeline.extract <work_dir>
"""
from __future__ import annotations

import datetime as dt
import re
import sys

from pipeline.comicdb import Block, ComicDB, Panel, Vision, panel_id
from pipeline.vision import VISION_MODEL, ask_vision, looks_like_reasoning, strip_think

PROMPT_V = 2
PROMPT = """You are transcribing ONE page of a comic book for an audio adaptation.

Go panel by panel in reading order (left to right, top to bottom; a full-width strip is one panel). This page has roughly {n_panels} panels.

For EVERY panel write a block in exactly this shape:

PANEL 1
DESC: One or two sentences describing ONLY what is visible in THIS panel -- the people (by appearance only: clothing, build, posture; never a character, hero, or franchise name), their actions and expressions, and the setting. Do not invent weather, lighting, time of day, or mood that is not shown. Do not carry anything over from another panel.
TEXT:
- CAPTION: text of a rectangular narration or caption box
- man in white shirt: words in that character's speech balloon
- SFX: a sound-effect word that nobody speaks

Rules for the TEXT lines:
- Put ALL the words from ONE balloon or ONE caption box on a SINGLE line, even if they are lettered across several rows. Never split one balloon into multiple lines.
- Start every line with "- " then the label, then ": ", then the exact words.
- Label a speech balloon with a SHORT appearance label for who is speaking ("man in white shirt", "cloaked figure"). Reuse the identical label every time the same person speaks.
- A rectangular caption / narration box is always CAPTION, even when it is a character's inner monologue.
- List the balloons and captions in reading order. Copy the words exactly -- do not paraphrase, fix spelling, translate, or skip anything.
- If the panel has no lettering, write exactly: TEXT: none
- Never repeat a line. If you find yourself repeating, stop that panel.

Output only PANEL blocks. No commentary, no summary, no reasoning.
"""

_PANEL_RE = re.compile(r"^\s*PANEL\s+(\d+)\b", re.I | re.M)
_TAG_RE = re.compile(r"^\s*[-*]?\s*([^:|]{1,40}?)\s*[|:]\s*(.+?)\s*$")
_SFX_SHAPE = re.compile(r"^[A-Z][A-Z'\-]{1,10}[!?.]*$")
_MAX_LINES_PER_PANEL = 24


class ParsedPanel:
    __slots__ = ("desc", "lines")

    def __init__(self, desc: str, lines: list[tuple[str, str]]):
        self.desc = desc
        self.lines = lines  # [(tag, text)]  tag in {CAPTION, SFX, <appearance>}


def is_runaway(raw: str) -> bool:
    """A dense page can send the model into a repeat loop ("FLASH | FLASH"
    x500). Flag it: many lines, few of them distinct."""
    body = [ln.strip() for ln in raw.splitlines()
            if ln.strip().startswith(("-", "*")) or "|" in ln]
    if len(body) < 40:
        return False
    return len(set(b.lower() for b in body)) / len(body) < 0.5


def parse(raw: str) -> list[ParsedPanel]:
    text = strip_think(raw)
    marks = list(_PANEL_RE.finditer(text))
    if not marks:
        return []
    out: list[ParsedPanel] = []
    for i, m in enumerate(marks):
        chunk = text[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        desc, lines = _parse_chunk(chunk)
        out.append(ParsedPanel(desc, lines))
    return out


def _parse_chunk(chunk: str) -> tuple[str, list[tuple[str, str]]]:
    desc_parts: list[str] = []
    lines: list[tuple[str, str]] = []
    mode = "desc"
    for ln in chunk.splitlines():
        raw = ln.strip()
        if not raw:
            continue
        low = raw.lower()
        if low.startswith("desc:"):
            mode = "desc"
            desc_parts.append(raw[5:].strip())
            continue
        if low.startswith("text:"):
            mode = "text"
            rest = raw[5:].strip()
            if rest and rest.lower() != "none":
                _add_line(lines, rest)
            continue
        if mode == "desc":
            desc_parts.append(raw)
        else:
            _add_line(lines, raw)
        if len(lines) > _MAX_LINES_PER_PANEL:
            break
    desc = re.sub(r"\s+", " ", " ".join(desc_parts)).strip()
    desc = re.sub(r"^(here'?s?|description|scene|the panel shows|this panel( shows)?|"
                  r"in this panel,?)[:,\s-]+", "", desc, flags=re.I)
    desc = desc[:1].upper() + desc[1:] if desc else desc
    if looks_like_reasoning(desc) or len(desc.split()) > 90:
        desc = "" if looks_like_reasoning(desc) else " ".join(desc.split()[:90])
    return desc, lines


def _add_line(lines: list, raw: str) -> None:
    m = _TAG_RE.match(raw)
    if not m:
        return
    tag = re.sub(r"[<>]", "", m.group(1)).strip()
    txt = m.group(2).strip().strip('"“” ')
    if not txt:
        return
    # drop an exact repeat of the previous line (runaway loop guard)
    if lines and lines[-1][1].lower() == txt.lower():
        return
    lines.append((tag, txt))


def to_blocks(db: ComicDB, pid: str, lines: list[tuple[str, str]]) -> list[Block]:
    out = []
    for order, (tag, text) in enumerate(lines):
        tl = tag.lower()
        if tl == "caption":
            kind, hint = "CAPTION", ""
        elif tl == "sfx" or (_SFX_SHAPE.match(text) and len(text.split()) == 1
                             and tl not in ("man", "woman", "figure")):
            kind, hint = "SFX", ""
        else:
            kind, hint = "DIALOGUE", tag
        out.append(Block(id=db.next_block_id(), panel=pid, order=order, kind=kind,
                         text_raw=text, text_clean=re.sub(r"\s+", " ", text).strip(),
                         speaker_hint=hint))
    return out


def _crop_boxes(page, n_panels: int) -> list[list[float]]:
    """Kumiko boxes only if the count agrees; else no crops (use full page)."""
    if page.panel_boxes and len(page.panel_boxes) == n_panels:
        return page.panel_boxes
    return []


def _page_summary(panels: list[ParsedPanel]) -> str:
    descs = [p.desc for p in panels if p.desc]
    return re.sub(r"\s+", " ", " ".join(descs))[:400]


def _rebuild_page(db: ComicDB, page, panels: list[ParsedPanel]) -> None:
    from PIL import Image
    db.remove_panels_for_page(page.index)
    boxes = _crop_boxes(page, len(panels))
    im = Image.open(page.image) if boxes else None
    for pi, pp in enumerate(panels):
        pid = panel_id(page.index, pi)
        bbox: list[float] = []
        img_path = page.image
        if boxes:
            x, y, x2, y2 = boxes[pi]
            bbox = [x, y, x2, y2]
            crop_path = db.path.parent / "panels" / f"{pid}.jpg"
            im.crop((int(x), int(y), int(x2), int(y2))).save(crop_path, quality=92)
            img_path = str(crop_path)
        db.add_panel(Panel(id=pid, page=page.index, index=pi, image=img_path,
                           bbox=bbox, scene=pp.desc, scene_source="extract"))
        db.replace_blocks_for_panel(pid, to_blocks(db, pid, pp.lines))
    if im:
        im.close()
    db.set_page_summary(page.index, _page_summary(panels))


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m pipeline.extract <work_dir>", file=sys.stderr)
        sys.exit(2)
    db = ComicDB.load(sys.argv[1])
    pages = [p for p in db.pages() if not p.is_front_matter]

    todo, cached = [], 0
    for p in pages:
        v = p.vision
        if v.raw and v.model == VISION_MODEL and v.prompt_v == PROMPT_V:
            parsed = parse(v.raw)
            if parsed:
                _rebuild_page(db, p, parsed)
                cached += 1
                continue
        todo.append(p)
    if cached:
        print(f"{cached} pages re-parsed from cache")
    print(f"{len(todo)} pages to extract")

    opts = {"repeat_penalty": 1.3, "repeat_last_n": 320}
    for n, p in enumerate(todo):
        prompt = PROMPT.replace("{n_panels}", str(max(len(p.panel_boxes), 1)))
        res = ask_vision(p.image, prompt, num_predict=1500, timeout=420, extra_options=opts)
        raw = res.get("text", "")
        note = ""
        if is_runaway(raw):
            note = " [runaway -> retry]"
            res = ask_vision(p.image, prompt, num_predict=900, timeout=300,
                             extra_options={"repeat_penalty": 1.5, "repeat_last_n": 512,
                                            "temperature": 0.2})
            raw = res.get("text", "")
        parsed = parse(raw)
        v = Vision(model=VISION_MODEL, prompt_v=PROMPT_V, raw=raw,
                   at=dt.datetime.now().isoformat(timespec="seconds"))
        db.set_page_vision(p.index, v)
        if parsed:
            _rebuild_page(db, p, parsed)
        db.save()
        nlines = sum(len(pp.lines) for pp in parsed)
        print(f"[{n+1}/{len(todo)}] page {p.index}: {len(parsed)} panels, {nlines} text lines"
              + note + (f"  ERR {res['error']}" if res.get("error") else ""))

    db.save()
    print(f"done -> {db.path}")


if __name__ == "__main__":
    main()
