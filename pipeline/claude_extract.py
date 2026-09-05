"""Claude vision path -- replaces transcribe+identify+resolve+redescribe with
one page-level pass per page.

Unlike the local qwen2.5vl pipeline, this asks for structured JSON directly
(Claude follows it reliably, unlike the small local model) and asks the model
to name a speaker itself, under a confidence-tiered rule set, instead of
inferring names afterward from evidence the way resolve.py does:

    tier              example                                   confidence
    explicit          self-ID, addressed by name, named in caption   ~0.9+
    visual signature  a costume/design already confirmed earlier
                       in THIS issue (fed back in as "known cast")    ~0.6-0.85
    (none)            no page evidence -- appearance tag only, no
                       name, regardless of what the model "knows"     n/a

A name below NAME_MIN_CONFIDENCE is kept as an appearance tag, never shown as
a name -- this is the guard against the "Batman" class of hallucination,
enforced structurally rather than hoped for from the prompt alone.

Idempotent and cost-aware: each page's raw response is cached (Page.vision)
keyed by (model, PROMPT_V), so a re-run with no change re-parses for free.

    ANTHROPIC_API_KEY=... python -m pipeline.claude_extract <work_dir>
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys

from pipeline.claude_vision import DEFAULT_MODEL, ask_claude
from pipeline.comicdb import Block, ComicDB, Entity, Panel, Vision, panel_id

PROMPT_V = 1
NAME_MIN_CONFIDENCE = 0.6

SYSTEM = """You are transcribing one page of a comic book for an audio adaptation for blind and low-vision readers. Accuracy and honesty matter more than a satisfying answer.

Hard rules:
1. Describe only what is visibly drawn on THIS page -- people by build/clothing/posture, action, setting, expression. Never invent weather, mood, or lighting that isn't shown. Never carry a detail over from another panel.
2. Do NOT name a character from franchise knowledge or pattern-matching to a comic you may recognize. A name is only allowed when justified by evidence ON THE PAGES OF THIS ISSUE:
   - HIGH confidence (0.85-1.0): the name is explicitly lettered -- a self-identification ("I'm ___"), another character addressing them by name, or a caption naming them.
   - MEDIUM confidence (0.6-0.85): you recognize a costume/visual design that was already confirmed by name on an earlier page of THIS issue (given to you as "known cast" below) reappearing here with no new name lettered.
   - Anything else: no name. Use a short neutral appearance tag instead (e.g. "the man in the green cloak") and give it confidence 0.
3. Transcribe every piece of lettering exactly as printed, in reading order, unmodified (no spelling fixes, no paraphrase).
4. Reply with ONLY a JSON object, no commentary, matching this shape exactly:

{
  "panels": [
    {
      "desc": "one to two sentence objective description",
      "lines": [
        {"kind": "caption|dialogue|sfx", "speaker": "name or appearance tag or null for a caption/SFX",
         "confidence": 0.0, "evidence": "short reason, e.g. self-id / addressed by X / recurring costume",
         "text": "exact lettering"}
      ]
    }
  ],
  "cast_updates": [
    {"name": "string or null if unnamed", "appearance": "short visual description",
     "confidence": 0.0, "evidence": "short reason"}
  ]
}

cast_updates lists every distinct person seen on this page, named or not, so their appearance can be tracked across pages -- it is not filtered by confidence."""


def _prompt(page_num: int, total: int, cast: str) -> str:
    known = cast if cast else "(none yet -- this is early in the issue)"
    return (f"This is page {page_num} of {total} of the issue, in order.\n\n"
            f"Known cast so far, from earlier pages of THIS issue:\n{known}\n\n"
            f"Process this page now. Reply with only the JSON object.")


def _parse(raw: str) -> dict | None:
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _cast_summary(cast: dict) -> str:
    lines = []
    for name, info in cast.items():
        conf = info["confidence"]
        tag = "confirmed" if conf >= 0.85 else "tentative"
        lines.append(f"- {name} ({tag}, {info['appearance']})")
    return "\n".join(lines)


def _update_cast(cast: dict, updates: list) -> None:
    for u in updates or []:
        name = (u.get("name") or "").strip()
        if not name:
            continue
        conf = float(u.get("confidence") or 0)
        existing = cast.get(name)
        if not existing or conf > existing["confidence"]:
            cast[name] = {"appearance": u.get("appearance", ""), "confidence": conf}


def _rebuild_page(db: ComicDB, page, data: dict, cast: dict) -> None:
    db.remove_panels_for_page(page.index)
    for pi, pn in enumerate(data.get("panels", [])):
        pid = panel_id(page.index, pi)
        db.add_panel(Panel(id=pid, page=page.index, index=pi, image=page.image,
                           scene=(pn.get("desc") or "").strip(), scene_source="claude"))
        blocks = []
        for order, ln in enumerate(pn.get("lines", [])):
            text = (ln.get("text") or "").strip()
            if not text:
                continue
            kind = {"caption": "CAPTION", "sfx": "SFX"}.get(
                (ln.get("kind") or "").lower(), "DIALOGUE")
            speaker = (ln.get("speaker") or "").strip() or None
            conf = float(ln.get("confidence") or 0)
            ent_id = None
            if kind == "DIALOGUE" and speaker:
                named = conf >= NAME_MIN_CONFIDENCE and speaker in cast
                ent_id = _entity_for(db, speaker, named, conf)
            blocks.append(Block(
                id=db.next_block_id(), panel=pid, order=order, kind=kind,
                text_raw=text, text_clean=re.sub(r"\s+", " ", text).strip(),
                entity=ent_id, speaker_raw=speaker if conf >= NAME_MIN_CONFIDENCE else None,
                speaker_confidence=conf, speaker_evidence=(ln.get("evidence") or "")[:120],
            ))
        db.replace_blocks_for_panel(pid, blocks)


_entity_by_name: dict[str, str] = {}


def _entity_for(db: ComicDB, label: str, named: bool, confidence: float) -> str:
    """One entity per distinct label (name if confident, else appearance tag),
    reused across the whole issue so the same person keeps the same voice."""
    key = label.lower()
    if key in _entity_by_name:
        eid = _entity_by_name[key]
        if named:
            db.bind_name(eid, label, round(confidence, 2))
        return eid
    eid = db.next_entity_id()
    ent = Entity(id=eid, appearance="" if named else label,
                name=label if named else None,
                name_confidence=round(confidence, 2) if named else 0.0)
    db.set_entities([*db.entities(), ent])
    _entity_by_name[key] = eid
    return eid


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: ANTHROPIC_API_KEY=... python -m pipeline.claude_extract <work_dir>",
              file=sys.stderr)
        sys.exit(2)
    db = ComicDB.load(sys.argv[1])
    pages = [p for p in db.pages() if not p.is_front_matter]
    _entity_by_name.clear()

    cast: dict = {}
    total_cost = 0.0
    todo, cached = [], 0
    for p in pages:
        v = p.vision
        if v.raw and v.model == DEFAULT_MODEL and v.prompt_v == PROMPT_V:
            data = _parse(v.raw)
            if data:
                cached += 1
                continue
        todo.append(p)
    if cached:
        print(f"{cached} pages already extracted (cached, no API call)")
    print(f"{len(todo)} pages to send to {DEFAULT_MODEL}")

    for n, p in enumerate(todo):
        res = ask_claude(p.image, system=SYSTEM,
                         prompt=_prompt(p.index + 1, len(pages), _cast_summary(cast)),
                         max_tokens=2500)
        if res.get("error"):
            print(f"[{n+1}/{len(todo)}] page {p.index}: ERROR {res['error']}", file=sys.stderr)
            continue
        data = _parse(res["text"])
        v = Vision(model=DEFAULT_MODEL, prompt_v=PROMPT_V, raw=res["text"],
                   at=dt.datetime.now().isoformat(timespec="seconds"))
        db.set_page_vision(p.index, v)
        if data:
            _update_cast(cast, data.get("cast_updates"))
            _rebuild_page(db, p, data, cast)
        db.save()
        cost = res.get("cost", 0.0)
        total_cost += cost
        npan = len(data.get("panels", [])) if data else 0
        print(f"[{n+1}/{len(todo)}] page {p.index}: {npan} panels, "
              f"${cost:.4f} (running total ${total_cost:.2f})")

    db.save()
    named = [(e.id, e.name, e.name_confidence) for e in db.entities() if e.name]
    print(f"\n{len(named)} named characters:")
    for eid, name, conf in named:
        print(f"  {eid} = {name}  ({conf})")
    print(f"total estimated cost this run: ${total_cost:.2f}")
    print(f"done -> {db.path}")


if __name__ == "__main__":
    main()
