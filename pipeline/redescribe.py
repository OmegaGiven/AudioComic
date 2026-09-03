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


class Ctx:
    __slots__ = ("who", "hints", "page_sum", "prev_sum", "next_sum", "siblings")

    def __init__(self, who, hints, page_sum, prev_sum, next_sum, siblings):
        self.who = who              # resolved entity names/appearances speaking here
        self.hints = hints          # extract speaker-appearance tags
        self.page_sum = page_sum
        self.prev_sum = prev_sum
        self.next_sum = next_sum
        self.siblings = siblings    # other panel descriptions on this page

    def key(self) -> list[str]:
        return [*self.who, "|", *self.hints, "|", self.page_sum,
                self.prev_sum, self.next_sum, *self.siblings]


def _context(db: ComicDB, panel) -> Ctx:
    who, hints = [], []
    for b in db.blocks(panel=panel.id):
        if b.entity:
            e = db.entity(b.entity)
            if e and (lbl := e.name or e.appearance):
                who.append(lbl)
        if b.speaker_hint:
            hints.append(b.speaker_hint)
    who = list(dict.fromkeys(who))
    hints = list(dict.fromkeys(h for h in hints if h))
    pages = {p.index: p for p in db.pages()}
    page = pages.get(panel.page)
    siblings = [pn.scene for pn in db.panels()
                if pn.page == panel.page and pn.id != panel.id and pn.scene]
    return Ctx(who, hints,
               page.summary if page else "",
               pages[panel.page - 1].summary if panel.page - 1 in pages else "",
               pages[panel.page + 1].summary if panel.page + 1 in pages else "",
               siblings[:6])


def _prompt(c: Ctx) -> str:
    if c.who:
        lead = (f"This panel contains: {'; '.join(c.who)}. Use these names/descriptions "
                f"for those people; do not introduce any other name.")
    elif c.hints:
        lead = (f"People seen elsewhere on this page: {'; '.join(c.hints)}. "
                f"Refer to anyone present only by appearance -- no character or franchise name.")
    else:
        lead = "Refer to any person only by appearance -- no character or franchise name."

    bg = ""
    if c.page_sum:
        bg += f"\n\nFor context only (do NOT repeat it), this page so far: {c.page_sum}"
    if c.prev_sum:
        bg += f"\nThe previous page: {c.prev_sum}"

    return (f"{lead}{bg}\n\nNow describe ONLY THIS panel in 1-3 sentences for an audiobook: "
            f"the setting, what each person is doing, and their expression. "
            f"Open with the concrete subject and action you actually see in THIS panel "
            f"(e.g. 'A man in a military uniform holds a child.'). "
            f"Describe only what is visible in this panel -- do NOT invent weather, lighting, "
            f"time of day, or mood that is not shown, and do NOT carry over details from other "
            f"panels or from the context above. Write it as prose narration, not 'the panel shows'. "
            f"Do NOT transcribe or mention any lettering, dialogue, caption, or sound effect. "
            f"Reply with only the description.")


REDESC_V = 6  # bump to force Pass 2 to re-run every panel


def _sig(c: Ctx) -> str:
    return hashlib.sha1(f"v{REDESC_V}|{'|'.join(c.key())}".encode()).hexdigest()[:8]


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
        ctx = _context(db, p)
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
        scene = re.sub(r"^(in this panel,?|the panel shows|this panel( shows)?|"
                       r"here is [^:]*:?)\s*", "", scene, flags=re.I)
        scene = scene[:1].upper() + scene[1:] if scene else scene
        db.set_redescribe(p.id, scene, sig)
        db.save()
        print(f"[{n+1}/{len(todo)}] {p.id}  who={ctx.who or '-'}")

    db.save()
    print(f"done -> {db.path}")


if __name__ == "__main__":
    main()
