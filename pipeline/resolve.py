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

# Comic dialogue is lettered ALL CAPS, so case can't tell a name from a word.
# Match case-insensitively, then reject candidates that are common words.
NAME = r"([A-Za-z][A-Za-z']{1,15}(?:\s[A-Za-z][A-Za-z']{1,15})?)"
SELF_ID = re.compile(rf"\b(?:i am|i'?m|call me|my name is|name's|they call me)\s+{NAME}", re.I)
VOCATIVE = re.compile(rf"[,:]\s+{NAME}[.!?\"'”]*\s*$")
NARRATION = re.compile(rf"^\s*{NAME}\s+(?:said|knelt|stood|turned|raised|whispered|shouted|"
                       r"replied|thought|watched|walked|ran|stared|grabbed|screamed)", re.I)

_COMMON = set("""
a an the and or but not so if then than as at by for from in into of on to up with
i me my you your he she it we they him her them us our his hers its their this that
these those here there now back away down out off over under again just only very
yes no ok okay sure fine well hey wait stop look listen come go get got give take
sorry please thanks thank sir maam man dude kid son dad mom mother father brother
sister boss chief doctor captain sergeant lieutenant colonel general everyone
everybody someone somebody nobody people friend buddy pal boy girl lady god hell
damn christ lord jesus death dead alive alone lost cold hot warm dark light hungry
tired scared afraid ready done gone home real true good bad better worse right wrong
late early sick hurt fighting trying doing saying telling asking talking working
looking waiting leaving staying calling dying running thinking coming going holding
standing sitting moving turning falling rising warning begging hoping praying human
family blood power willing able glad happy proud curious aware part what who where
why how when which whose everything nothing something anything all none more most
less least first last next each every any some both few many much such other same
""".split())


def _ok(name: str) -> bool:
    words = [w.lower().strip(".'\"") for w in name.split()]
    return bool(words) and not all(w in _COMMON for w in words)


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
                ev.append(NameEvidence(b.entity, panel, "self_id", text, _title(m.group(1)), 1.0))
            if b.speaker_raw and _ok(b.speaker_raw):
                ev.append(NameEvidence(b.entity, panel, "printed", text, _title(b.speaker_raw), 1.0))

        others = _present_entities(db, panel, exclude=b.entity)
        if not others:
            others = [e.id for e in db.entities() if e.id != b.entity and not e.name]
        for rx, kind in ((VOCATIVE, "vocative"), (NARRATION, "narration")):
            m = rx.search(text)
            if m and _ok(m.group(1)) and len(others) == 1:
                ev.append(NameEvidence(others[0], panel, kind, text, _title(m.group(1)),
                                       0.5 if kind == "narration" else 0.35))
    return ev


def _title(name: str) -> str:
    return " ".join(w.capitalize() for w in name.split())


def _present_entities(db: ComicDB, panel_id: str, exclude: str | None) -> list[str]:
    """Entities that speak in this panel (proxy for 'present in frame')."""
    return sorted({b.entity for b in db.blocks(panel=panel_id)
                   if b.entity and b.entity != exclude})


def bind(db: ComicDB, evidence: list[NameEvidence]) -> None:
    by_entity: dict[str, list[NameEvidence]] = {}
    for e in evidence:
        by_entity.setdefault(e.entity, []).append(e)

    # how often each candidate name-token shows up anywhere in dialogue --
    # supporting signal that it's a real proper noun, not a one-off word
    mentions: dict[str, int] = {}
    for b in db.blocks():
        if b.kind == "DIALOGUE":
            for tok in re.findall(r"[A-Za-z']{3,}", b.text_raw):
                mentions[tok.lower()] = mentions.get(tok.lower(), 0) + 1

    for ent in db.entities():
        if ent.id in db.overrides:
            db.bind_name(ent.id, db.overrides[ent.id], 1.0)
            continue
        evs = by_entity.get(ent.id, [])
        if not evs:
            continue
        scores: dict[str, float] = {}
        kinds: dict[str, set] = {}
        panels: dict[str, set] = {}
        for e in evs:
            scores[e.name] = scores.get(e.name, 0.0) + e.weight
            kinds.setdefault(e.name, set()).add(e.kind)
            panels.setdefault(e.name, set()).add(e.panel)
        name, score = max(scores.items(), key=lambda kv: kv[1])

        n_ev = len([e for e in evs if e.name == name])
        n_panels = len(panels[name])
        multiword = len(name.split()) >= 2
        # supporting: every token of the name shows up in >=3 dialogue blocks
        supported = all(mentions.get(w.lower(), 0) >= 3 for w in name.split())
        strong = "printed" in kinds[name] or "self_id" in kinds[name]

        # never bind on a lone weak signal. A self-identification or printed
        # name counts if it's a full name or echoed elsewhere; otherwise need
        # two independent references.
        if (n_panels >= 2 and n_ev >= 2) or (strong and (multiword or supported)):
            db.bind_name(ent.id, name,
                         round(min(1.0, 0.4 + 0.2 * n_ev + 0.15 * n_panels), 2))

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
