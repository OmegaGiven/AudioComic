"""Phase 2 (Pass 1 vision) -- per panel: verbatim text + a rough description.

Deliberately close to the v1 prompt, which reliably read every bubble in
order. One change: the description must not name any character. Naming is a
later job (resolve), done from the dialogue text, never guessed from the art.

Idempotent: keyed by (panel, model, PROMPT_V). A panel whose cached vision
matches is re-parsed from the stored raw response -- no inference.

    python -m pipeline.transcribe <work_dir>
"""
from __future__ import annotations

import datetime as dt
import re
import sys

from pipeline.comicdb import Block, ComicDB, Vision
from pipeline.vision import VISION_MODEL, ask_vision, looks_like_reasoning, strip_think

PROMPT_V = 9
PROMPT = """Describe THIS panel in 1-2 sentences: the scene, the people present and their actions and expressions. Open with the concrete subject and action you actually see. Describe only what is visible in this panel -- do not invent weather, lighting, time of day, or mood that is not clearly shown, and do not carry over details from other panels. Refer to each person ONLY by appearance (clothing, build, posture) -- do not use any character name, hero name, villain name, or franchise, even one you recognise.

Then transcribe every piece of dialogue or caption text visible in the panel, in reading order, formatted exactly as:
SPEAKER: text
or
CAPTION: text
(CAPTION for narration boxes with no speaker)

Copy the text exactly as lettered. Do not fix spelling, do not paraphrase, do not skip anything.

Respond directly with the description then the transcription. Do not explain your reasoning process."""

LINE_RE = re.compile(r"^\s*[-*]?\s*(SPEAKER|CAPTION)\s*:\s*(.+?)\s*$", re.I)
SFX_SHAPE = re.compile(r"^[A-Z][A-Z'\-]{1,10}[!?.]*$")


def parse(text: str) -> tuple[str, list[tuple[str, str]]]:
    """-> (scene, [(kind, text)])  kind in CAPTION|SPEAKER"""
    text = strip_think(text)
    scene_parts, lines = [], []
    for ln in text.splitlines():
        raw = ln.strip()
        if not raw:
            continue
        m = LINE_RE.match(raw)
        if m:
            lines.append((m.group(1).upper(), m.group(2).strip().strip('"“” ')))
        elif not lines:
            scene_parts.append(raw)
    scene = " ".join(scene_parts)
    scene = re.sub(r"^(here'?s?|description|scene|the panel shows|this panel( shows)?)[:,\s-]+",
                   "", scene, flags=re.I)
    scene = re.sub(r"\s+", " ", scene).strip()
    # a scene that still reads as model reasoning is worse than none
    if looks_like_reasoning(scene) or len(scene.split()) > 80:
        scene = "" if looks_like_reasoning(scene) else " ".join(scene.split()[:80])
    return scene, [(k, t) for k, t in lines if t]


def to_blocks(db: ComicDB, panel_id: str, lines: list[tuple[str, str]]) -> list[Block]:
    out = []
    for order, (kind, text) in enumerate(lines):
        if kind == "CAPTION":
            b_kind = "CAPTION"
        elif SFX_SHAPE.match(text) and len(text.split()) == 1:
            b_kind = "SFX"
        else:
            b_kind = "DIALOGUE"
        out.append(Block(id=db.next_block_id(), panel=panel_id, order=order,
                         kind=b_kind, text_raw=text,
                         text_clean=re.sub(r"\s+", " ", text).strip()))
    return out


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m pipeline.transcribe <work_dir>", file=sys.stderr)
        sys.exit(2)
    db = ComicDB.load(sys.argv[1])
    # transcribe every panel; assemble is what skips front matter later
    todo, cached = [], 0
    for p in db.panels():
        v = p.vision
        if v.raw and v.model == VISION_MODEL and v.prompt_v == PROMPT_V:
            # re-parse from cache, no inference
            scene, lines = parse(v.raw)
            db.set_transcribe(p.id, scene, v)
            db.replace_blocks_for_panel(p.id, to_blocks(db, p.id, lines))
            cached += 1
        else:
            todo.append(p)
    if cached:
        print(f"{cached} panels re-parsed from cache")

    print(f"{len(todo)} panels to transcribe")
    for n, p in enumerate(todo):
        res = ask_vision(p.image, PROMPT)
        raw = res.get("text", "")
        scene, lines = parse(raw)
        v = Vision(model=VISION_MODEL, prompt_v=PROMPT_V, raw=raw,
                   at=dt.datetime.now().isoformat(timespec="seconds"))
        db.set_transcribe(p.id, scene, v)
        db.replace_blocks_for_panel(p.id, to_blocks(db, p.id, lines))
        db.save()
        print(f"[{n+1}/{len(todo)}] {p.id}: {len(lines)} text lines"
              + (f"  ERR {res['error']}" if res.get("error") else ""))

    db.save()
    print(f"done -> {db.path}")


if __name__ == "__main__":
    main()
