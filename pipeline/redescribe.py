"""Phase 5 (Pass 2 vision) -- re-describe each panel, this time with the
resolved identities fed into the prompt.

The Pass 1 description was appearance-only and often generic/guessed. Now the
describer is told who and what is in frame, so it produces accurate,
consistent scene-setting.

Idempotent: stores its result in panel.scene with a Vision block tagged
prompt_v = REDESC_V + a hash of the entity context, so a panel is only
re-done when its identities changed.

    python -m pipeline.redescribe <work_dir>
"""
from __future__ import annotations

import hashlib
import re
import sys

from pipeline.comicdb import ComicDB
from pipeline.vision import ask_vision


def _context(db: ComicDB, panel_id: str) -> list[str]:
    """Human labels for entities that speak in this panel."""
    labels = []
    for b in db.blocks(panel=panel_id):
        if b.entity:
            e = db.entity(b.entity)
            if e:
                labels.append(e.name or e.appearance or "a figure")
    seen, out = set(), []
    for x in labels:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _prompt(context: list[str]) -> str:
    if context:
        who = "; ".join(context)
        lead = (f"This panel contains: {who}. Use these names/descriptions. "
                f"Do not introduce any other name.")
    else:
        lead = "Refer to any person only by appearance -- do not use any character or franchise name."
    return (f"{lead}\n\nDescribe THIS panel in 1-3 sentences for an audiobook: "
            f"the setting, what each person is doing, and their expression. "
            f"Open with the concrete subject and action you actually see in this panel "
            f"(for example 'A man in a military uniform holds a child.'). "
            f"Describe only what is visible in this panel -- do NOT invent weather, "
            f"lighting, time of day, or mood that is not clearly shown, and do NOT carry "
            f"over details from other panels. Write it as prose narration, not 'the panel shows'. "
            f"Do NOT transcribe or mention any lettering, dialogue, caption, or sound effect. "
            f"Reply with only the description.")


REDESC_V = 5  # bump to force Pass 2 to re-run every panel


def _sig(context: list[str]) -> str:
    return hashlib.sha1(f"v{REDESC_V}|{'|'.join(context)}".encode()).hexdigest()[:8]


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m pipeline.redescribe <work_dir>", file=sys.stderr)
        sys.exit(2)
    db = ComicDB.load(sys.argv[1])
    front = {p.index for p in db.pages() if p.is_front_matter}

    todo = []
    for p in db.panels():
        if p.page in front:
            continue
        ctx = _context(db, p.id)
        sig = _sig(ctx)
        if p.scene_source == "pass2" and p.scene_sig == sig and p.scene:
            continue
        todo.append((p, ctx, sig))

    print(f"redescribe: {len(todo)} panels")
    for n, (p, ctx, sig) in enumerate(todo):
        res = ask_vision(p.image, _prompt(ctx), num_predict=300)
        scene = res.get("text", "")
        # the model sometimes appends CAPTION:/SPEAKER: lines anyway -- cut them
        scene = re.split(r"\b(?:CAPTION|SPEAKER)\s*:", scene)[0]
        scene = re.sub(r"\s+", " ", scene).strip().strip('"“”')
        scene = re.sub(r"^(in this (comic ?book )?(panel|image|scene),?|"
                       r"the (panel|image) shows|this (comic ?book )?panel( shows)?|"
                       r"here is [^:]*:?)\s*", "", scene, flags=re.I)
        scene = scene[:1].upper() + scene[1:] if scene else scene
        db.set_redescribe(p.id, scene, sig)
        db.save()
        print(f"[{n+1}/{len(todo)}] {p.id}  ctx={ctx or '-'}")

    db.save()
    print(f"done -> {db.path}")


if __name__ == "__main__":
    main()
