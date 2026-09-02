#!/usr/bin/env python3
"""a_structure_magi.py <cbr/cbz file> <work_dir> [char_bank_dir]

Stage A of pipeline v2. Replaces stages 1 + the structural half of stage 2.

Uses Magi v2 (ragavsachdeva/magiv2) -- a model purpose-built for comic
transcription -- to get, per page:
  * panels, in reading order, with bboxes
  * text blocks: OCR'd text, essential-vs-junk flag
  * speaker attribution (which character said which text, via bubble tails)
  * consistent character identities across the whole chapter

Crucially Magi does NOT hallucinate character *names* the way an LLM VLM does
-- it clusters characters by appearance and only assigns a real name when you
give it a character bank (reference crops). Without a bank, speakers come out
as "Character 0", "Character 1", ... and later stages resolve names from the
dialogue text.

char_bank_dir (optional): a folder of <Name>.png / <Name>.jpg crops, one or
more per character. Filename stem (minus any trailing digits) is the name.

Output: <work_dir>/structure.json
  {
    "source": "...",
    "pages": [
      {"page_index": 0, "image": "extracted/....jpg", "width": W, "height": H,
       "panels": [{"panel_index": 0, "bbox": [x1,y1,x2,y2]}, ...],
       "texts": [
         {"text": "...", "bbox": [...], "essential": true,
          "speaker": "Character 1" | "William" | null,
          "panel_index": 0 | null}
       ],
       "characters": [{"id": "Character 1", "bbox": [...]}]}
    ]
  }
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
    suffix = archive_path.suffix.lower()
    if suffix == ".cbr":
        subprocess.run(["unrar", "x", "-y", str(archive_path), str(dest) + "/"],
                       check=True, capture_output=True, text=True)
    elif suffix in (".cbz", ".zip"):
        subprocess.run(["unzip", "-o", str(archive_path), "-d", str(dest)],
                       check=True, capture_output=True, text=True)
    else:
        raise ValueError(f"Unsupported archive type: {suffix}")


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

    model = AutoModel.from_pretrained("ragavsachdeva/magiv2",
                                      trust_remote_code=True)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    bank = load_character_bank(bank_dir)
    imgs = [read_image(p) for p in pages]

    with torch.no_grad():
        results = model.do_chapter_wide_prediction(
            imgs, bank, use_tqdm=True, do_ocr=True)

    out = {"source": str(archive_path), "pages": []}
    for i, (page_path, res) in enumerate(zip(pages, results)):
        w, h = Image.open(page_path).size
        panels = [
            {"panel_index": pi, "bbox": _bbox(b)}
            for pi, b in enumerate(res.get("panels", []))
        ]
        char_boxes = res.get("characters", [])
        char_names = res.get("character_names", [])
        characters = [
            {"id": char_names[ci] if ci < len(char_names) else f"Character {ci}",
             "bbox": _bbox(cb)}
            for ci, cb in enumerate(char_boxes)
        ]
        texts = []
        ocr = res.get("ocr", [])
        tboxes = res.get("texts", res.get("text_bboxes", []))
        essential = res.get("is_essential_text", [])
        assoc = dict(res.get("text_character_associations", []))
        for ti, t in enumerate(ocr):
            ci = assoc.get(ti)
            speaker = None
            if ci is not None:
                speaker = (char_names[ci] if ci < len(char_names)
                           else f"Character {ci}")
            texts.append({
                "text": t,
                "bbox": _bbox(tboxes[ti]) if ti < len(tboxes) else None,
                "essential": bool(essential[ti]) if ti < len(essential) else True,
                "speaker": speaker,
            })
        out["pages"].append({
            "page_index": i,
            "image": str(page_path),
            "width": w, "height": h,
            "panels": panels,
            "characters": characters,
            "texts": texts,
        })
        print(f"[{i+1}/{len(pages)}] {len(panels)} panels, {len(texts)} texts, "
              f"{len(characters)} characters")

    (work_dir / "structure.json").write_text(json.dumps(out, indent=2))
    print(f"Done. Structure: {work_dir / 'structure.json'}")


if __name__ == "__main__":
    main()
