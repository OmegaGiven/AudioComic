"""Phase 4 -- bind character entities to real names, with all the evidence.

Deterministic. Walk every text block for naming signals, weigh them, and bind
an entity only when the evidence clears a bar (a self-identification, or two
independent references). Merge aliases. Apply user overrides. Backfill is
automatic -- assemble reads whatever name is bound now.

    python -m pipeline.resolve <work_dir>
"""
from __future__ import annotations

import re
import sys

from pipeline.comicdb import ComicDB, NameEvidence

NAME = r"([A-Z][a-z]{2,15}(?:\s[A-Z][a-z]{2,15})?)"
SELF_ID = re.compile(rf"\b(?:i am|i'm|call me|my name is|name's)\s+{NAME}", re.I)
VOCATIVE = re.compile(rf"[,:]\s*{NAME}[.!?\"'”]*\s*$")
NARRATION = re.compile(rf"^{NAME}\s+(?:said|knelt|stood|turned|raised|whispered|shouted|"
                       r"replied|thought|watched|walked|ran|stared)")
STOPWORDS = {"Yes", "No", "Sir", "Please", "Well", "Okay", "Now", "Hey", "Wait",
             "The", "And", "But", "Not", "Look", "Stop", "Come", "Here", "There",
             "Death", "God", "Space", "Sector", "First", "Some", "Thing", "Things"}
WEIGHT = {"self_id": 1.0, "narration": 0.5, "vocative": 0.35, "printed": 0.8}
BIND_THRESHOLD = 0.7


def _ok(name: str) -> bool:
    return name.split()[0] not in STOPWORDS


def collect(db: ComicDB) -> list[NameEvidence]:
    ev: list[NameEvidence] = []
    for b in db.blocks():
        if b.kind == "SFX":
            continue
        text = b.text_clean or b.text_raw
        panel = b.panel

        if b.kind == "DIALOGUE" and b.entity:
            m = SELF_ID.search(text)
            if m and _ok(m.group(1)):
                ev.append(NameEvidence(b.entity, panel, "self_id", text, m.group(1), WEIGHT["self_id"]))
            if b.speaker_raw and b.speaker_raw.istitle() and _ok(b.speaker_raw):
                ev.append(NameEvidence(b.entity, panel, "printed", text, b.speaker_raw, WEIGHT["printed"]))

        # vocative / narration: names the *addressee* or an actor. Bind only
        # when we can point to exactly one candidate: another speaker in this
        # panel, or -- for a small cast -- the sole other unbound entity.
        others = _present_entities(db, panel, exclude=b.entity)
        if not others:
            others = [e.id for e in db.entities() if e.id != b.entity and not e.name]
        for rx, kind in ((VOCATIVE, "vocative"), (NARRATION, "narration")):
            m = rx.search(text)
            if m and _ok(m.group(1)) and len(others) == 1:
                ev.append(NameEvidence(others[0], panel, kind, text, m.group(1), WEIGHT[kind]))
    return ev


def _present_entities(db: ComicDB, panel_id: str, exclude: str | None) -> list[str]:
    """Entities that speak in this panel (proxy for 'present in frame')."""
    return sorted({b.entity for b in db.blocks(panel=panel_id)
                   if b.entity and b.entity != exclude})


def bind(db: ComicDB, evidence: list[NameEvidence]) -> None:
    by_entity: dict[str, list[NameEvidence]] = {}
    for e in evidence:
        by_entity.setdefault(e.entity, []).append(e)

    for ent in db.entities():
        if ent.id in db.overrides:
            db.bind_name(ent.id, db.overrides[ent.id], 1.0)
            continue
        evs = by_entity.get(ent.id, [])
        if not evs:
            continue
        # score per candidate name
        scores: dict[str, float] = {}
        kinds: dict[str, set] = {}
        for e in evs:
            scores[e.name] = scores.get(e.name, 0.0) + e.weight
            kinds.setdefault(e.name, set()).add(e.kind)
        name, score = max(scores.items(), key=lambda kv: kv[1])
        strong = "self_id" in kinds[name] or "printed" in kinds[name]
        independent = len({e.panel for e in evs if e.name == name}) >= 2
        if score >= BIND_THRESHOLD and (strong or independent):
            db.bind_name(ent.id, name, min(1.0, score))

    _merge_aliases(db)


def _merge_aliases(db: ComicDB) -> None:
    ents = [e for e in db.entities() if e.name]
    for a in ents:
        for b in ents:
            if a is b or not a.name or not b.name:
                continue
            an, bn = a.name.lower(), b.name.lower()
            if an != bn and (an in bn or bn in an):
                keep = a if len(a.name) >= len(b.name) else b
                drop = b if keep is a else a
                # relabel drop's blocks to keep, mark drop unnamed
                for blk in db.blocks(entity=drop.id):
                    db.link_block_entity(blk.id, keep.id)
                db.bind_name(drop.id, None, 0.0)


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m pipeline.resolve <work_dir>", file=sys.stderr)
        sys.exit(2)
    db = ComicDB.load(sys.argv[1])
    ev = collect(db)
    db.set_name_evidence(ev)
    bind(db, ev)
    db.save()
    named = [(e.id, e.name, e.name_confidence) for e in db.entities() if e.name]
    print(f"resolve: {len(ev)} evidence items -> {len(named)} named entities")
    for eid, name, conf in named:
        print(f"  {eid} = {name}  ({conf})")
    print(f"done -> {db.path}")


if __name__ == "__main__":
    main()
