#!/usr/bin/env python3
"""a_structure_magi.py <cbr/cbz file> <work_dir> [char_bank_dir]

Stage A of pipeline v2. Replaces stages 1 + the structural half of stage 2.

Magi v2 (ragavsachdeva/magiv2) -- a model purpose-built for comic
transcription -- gives per page:
  * panels, in reading order, with bboxes
  * text regions with OCR and an essential-vs-junk flag (SFX, watermarks,
    garbled OCR come back non-essential)
  * speaker attribution: which character-cluster said which text
  * character clusters (same person across pages gets one identity)

Magi does NOT hallucinate character *names* the way an LLM VLM does. Without a
character bank every speaker is "Character 0/1/2..."; with a bank
(<Name>.png crops) it assigns the real name.

Output: <work_dir>/structure.json
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def extract(archive_path: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    sfx = archive_path.suffix.lower()
    if sfx == ".cbr":
        subprocess.run(["unrar", "x", "-y", str(archive_path), str(dest) + "/"],
                       check=True, capture_output=True, text=True)
    elif sfx in (".cbz", ".zip"):
        subprocess.run(["unzip", "-o", str(archive_path), "-d", str(dest)],
                       check=True, capture_output=True, text=True)
    else:
        raise ValueError(f"Unsupported archive type: {sfx}")


def find_page_files(extracted_dir: Path):
    imgs = (sorted(extracted_dir.rglob("*.jpg"))
            + sorted(extracted_dir.rglob("*.jpeg"))
            + sorted(extracted_dir.rglob("*.png")))
    numbered = [f for f in imgs if re.search(r"(\d+)\.\w+$", f.name)]
    numbered.sort(key=lambda f: int(re.search(r"(\d+)\.\w+$", f.name).group(1)))
    return numbered or imgs


def read_image(path) -> np.ndarray:
    with open(path, "rb") as fh:
        return np.array(Image.open(fh).convert("L").convert("RGB"))


def load_character_bank(bank_dir: Path | None) -> dict:
    if not bank_dir or not bank_dir.is_dir():
        return {"images": [], "names": []}
    images, names = [], []
    for f in sorted(bank_dir.iterdir()):
        if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        images.append(read_image(f))
        names.append(re.sub(r"[-_ ]?\d+$", "", f.stem).strip())
    print(f"character bank: {len(images)} crops for {sorted(set(names))}")
    return {"images": images, "names": names}


def _bbox(x):
    return [round(float(v), 1) for v in x]


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _contains(panel, pt):
    return panel[0] <= pt[0] <= panel[2] and panel[1] <= pt[1] <= panel[3]


def _panel_for(text_bbox, panels):
    """Index of the panel whose box contains the text centre; else the panel
    whose centre is nearest; else None."""
    if not panels:
        return None
    c = _center(text_bbox)
    for i, p in enumerate(panels):
        if _contains(p, c):
            return i
    dists = [((_center(p)[0] - c[0]) ** 2 + (_center(p)[1] - c[1]) ** 2, i)
             for i, p in enumerate(panels)]
    return min(dists)[1]


def speaker_label(char_idx, cluster_labels, names):
    """char_idx -> a stable label. Real name if the bank matched, else a
    cluster id shared across the whole chapter."""
    if char_idx is None or char_idx >= len(cluster_labels):
        return None
    nm = names[char_idx] if char_idx < len(names) else "Other"
    if nm and nm != "Other":
        return nm
    return f"Character {cluster_labels[char_idx]}"


def main():
    if not (3 <= len(sys.argv) <= 4):
        print("Usage: a_structure_magi.py <issue.cbz> <work_dir> [char_bank_dir]",
              file=sys.stderr)
        sys.exit(2)

    archive_path = Path(sys.argv[1])
    work_dir = Path(sys.argv[2])
    bank_dir = Path(sys.argv[3]) if len(sys.argv) == 4 else None
    extracted = work_dir / "extracted"

    print(f"Extracting {archive_path.name}...")
    extract(archive_path, extracted)
    pages = find_page_files(extracted)
    print(f"{len(pages)} pages.")

    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained("ragavsachdeva/magiv2", trust_remote_code=True)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    bank = load_character_bank(bank_dir)
    imgs = [read_image(p) for p in pages]
    with torch.no_grad():
        results = model.do_chapter_wide_prediction(imgs, bank, use_tqdm=True, do_ocr=True)

    out = {"source": str(archive_path), "pages": []}
    for i, (page_path, res) in enumerate(zip(pages, results, strict=False)):
        w, h = Image.open(page_path).size
        panels_raw = [list(map(float, b)) for b in res.get("panels", [])]
        panels = [{"panel_index": pi, "bbox": _bbox(b)} for pi, b in enumerate(panels_raw)]

        cluster_labels = res.get("character_cluster_labels", [])
        names = res.get("character_names", [])
        char_boxes = res.get("characters", [])
        characters = [
            {"cluster": cluster_labels[ci] if ci < len(cluster_labels) else ci,
             "id": speaker_label(ci, cluster_labels, names),
             "bbox": _bbox(cb)}
            for ci, cb in enumerate(char_boxes)
        ]

        ocr = res.get("ocr", [])
        tboxes = [list(map(float, b)) for b in res.get("texts", [])]
        essential = res.get("is_essential_text", [])
        t2c = {t: c for t, c in res.get("text_character_associations", [])}

        texts = []
        for ti, t in enumerate(ocr):
            tb = tboxes[ti] if ti < len(tboxes) else None
            texts.append({
                "text": t,
                "bbox": _bbox(tb) if tb else None,
                "essential": bool(essential[ti]) if ti < len(essential) else True,
                "speaker": speaker_label(t2c.get(ti), cluster_labels, names),
                "panel_index": _panel_for(tb, panels_raw) if tb else None,
            })

        out["pages"].append({
            "page_index": i, "image": str(page_path), "width": w, "height": h,
            "panels": panels, "characters": characters, "texts": texts,
        })
        ess = sum(1 for t in texts if t["essential"])
        print(f"[{i+1}/{len(pages)}] {len(panels)} panels, {ess}/{len(texts)} essential texts, "
              f"{len(set(cluster_labels))} distinct characters")

    (work_dir / "structure.json").write_text(json.dumps(out, indent=2))
    print(f"Done. Structure: {work_dir / 'structure.json'}")


if __name__ == "__main__":
    main()
