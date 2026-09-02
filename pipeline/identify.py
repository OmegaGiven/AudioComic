"""Phase 3 -- cluster characters across the whole issue (Magi v2, identity
only). Links each DIALOGUE block to a character entity.

Magi runs chapter-wide and returns, per page: character boxes, a cluster
label per box (same person -> same label across the whole issue), and its own
OCR + text->character links. We ignore Magi's OCR as the final text; we use it
only to match Magi's attributed line to OUR transcribed block on the same
page, then link that block to the entity for the cluster.

    MAGI_PY=... python -m pipeline.identify <work_dir>

Needs the Magi venv (transformers + torch 2.8 + torchvision 0.23). Run with
that interpreter, or set MAGI_PYTHON and it re-execs itself.
"""
from __future__ import annotations

import difflib
import os
import re
import sys
from pathlib import Path

from pipeline.comicdb import ComicDB, Entity, Observation


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _best_match(magi_text: str, candidates: list, cutoff: float = 0.55):
    mt = _norm(magi_text)
    best, best_r = None, cutoff
    for b in candidates:
        r = difflib.SequenceMatcher(None, mt, _norm(b.text_clean or b.text_raw)).ratio()
        if r > best_r:
            best, best_r = b, r
    return best


def run(work_dir: str) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel

    db = ComicDB.load(work_dir)
    pages = sorted(db.pages(), key=lambda p: p.index)
    imgs = [np.array(Image.open(p.image).convert("L").convert("RGB")) for p in pages]

    model = AutoModel.from_pretrained("ragavsachdeva/magiv2", trust_remote_code=True)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    with torch.no_grad():
        results = model.do_chapter_wide_prediction(
            imgs, {"images": [], "names": []}, use_tqdm=True, do_ocr=True)

    # entity per global cluster label
    entities: dict[int, Entity] = {}
    linked = 0
    for page, res in zip(pages, results, strict=False):
        clusters = res.get("character_cluster_labels", [])
        char_boxes = res.get("characters", [])
        ocr = res.get("ocr", [])
        t2c = dict(res.get("text_character_associations", []))
        page_blocks = [b for b in db.blocks()
                       if db.panel(b.panel) and db.panel(b.panel).page == page.index
                       and b.kind == "DIALOGUE"]

        for ci, box in enumerate(char_boxes):
            label = clusters[ci] if ci < len(clusters) else ci
            ent = entities.get(label)
            if ent is None:
                ent = Entity(id=f"e{label + 1}", appearance="")
                entities[label] = ent
            ent.observations.append(Observation(panel="", bbox=[round(float(v), 1) for v in box]))

        for ti, text in enumerate(ocr):
            ci = t2c.get(ti)
            if ci is None:
                continue
            label = clusters[ci] if ci < len(clusters) else ci
            ent = entities.get(label)
            if ent is None:
                continue
            blk = _best_match(text, page_blocks)
            if blk:
                db.link_block_entity(blk.id, ent.id)
                linked += 1

    db.set_entities(sorted(entities.values(), key=lambda e: int(e.id[1:])))
    db.save()
    print(f"identify: {len(entities)} character clusters, {linked} dialogue blocks linked")
    print(f"done -> {db.path}")


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m pipeline.identify <work_dir>", file=sys.stderr)
        sys.exit(2)
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        alt = os.environ.get("MAGI_PYTHON")
        if alt and Path(alt).exists():
            os.execv(alt, [alt, "-m", "pipeline.identify", sys.argv[1]])
        print("identify needs the Magi venv (transformers + torch). "
              "Run with that python or set MAGI_PYTHON.", file=sys.stderr)
        sys.exit(3)
    run(sys.argv[1])


if __name__ == "__main__":
    main()
