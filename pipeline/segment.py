"""Phase 1 -- extract archive, segment panels (Kumiko), seed the DB.

    python -m pipeline.segment <issue.cbz> <work_dir>
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from pipeline.comicdb import ComicDB, Page, Panel, panel_id

REPO_ROOT = Path(__file__).resolve().parent.parent
KUMIKO_DIR = REPO_ROOT / "tools" / "kumiko"


def extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    sfx = archive.suffix.lower()
    if sfx == ".cbr":
        subprocess.run(["unrar", "x", "-y", str(archive), str(dest) + "/"],
                       check=True, capture_output=True, text=True)
    elif sfx in (".cbz", ".zip"):
        subprocess.run(["unzip", "-o", str(archive), "-d", str(dest)],
                       check=True, capture_output=True, text=True)
    else:
        raise ValueError(f"unsupported archive: {sfx}")


def find_pages(extracted: Path) -> list[Path]:
    imgs = (sorted(extracted.rglob("*.jpg")) + sorted(extracted.rglob("*.jpeg"))
            + sorted(extracted.rglob("*.png")))
    numbered = [f for f in imgs if re.search(r"(\d+)\.\w+$", f.name)]
    numbered.sort(key=lambda f: int(re.search(r"(\d+)\.\w+$", f.name).group(1)))
    return numbered or imgs


def run_kumiko(image_path: Path) -> list[list[float]] | None:
    try:
        r = subprocess.run(
            [sys.executable, "kumiko", "-i", str(image_path.resolve())],
            cwd=str(KUMIKO_DIR), capture_output=True, text=True, timeout=60,
        )
        return json.loads(r.stdout)[0]["panels"]
    except Exception as e:
        print(f"  kumiko failed on {image_path.name}: {e}", file=sys.stderr)
        return None


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: python -m pipeline.segment <issue.cbz> <work_dir>", file=sys.stderr)
        sys.exit(2)
    archive, work_dir = Path(sys.argv[1]), Path(sys.argv[2])
    from PIL import Image

    m = (re.match(r"^(.+?)[\s_#-]+0*(\d{1,4})\s*$", archive.stem)
         or re.match(r"^([A-Za-z]{2,}?)0*(\d{1,4})\s*$", archive.stem))
    series = m.group(1).strip() if m else archive.stem
    number = int(m.group(2)) if m else None
    db = ComicDB.load_or_new(work_dir, source=archive.name, series=series, number=number)
    extracted = work_dir / "extracted"
    panels_dir = work_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    print(f"extracting {archive.name}")
    extract(archive, extracted)
    pages = find_pages(extracted)
    print(f"{len(pages)} pages")

    for i, f in enumerate(pages):
        im = Image.open(f)
        w, h = im.size
        panels = run_kumiko(f) or [[0, 0, w, h]]
        # front matter heuristic: a spread-ish single-panel page near the
        # start, or a page Kumiko couldn't split with almost nothing on it
        front = i < 4 and len(panels) <= 1
        db.add_page(Page(index=i, image=str(f), w=w, h=h, is_front_matter=front))
        for pi, (x, y, pw, ph) in enumerate(panels):
            pid = panel_id(i, pi)
            crop_path = panels_dir / f"{pid}.jpg"
            im.crop((int(x), int(y), int(x + pw), int(y + ph))).save(crop_path, quality=92)
            db.add_panel(Panel(id=pid, page=i, index=pi, image=str(crop_path),
                               bbox=[float(x), float(y), float(x + pw), float(y + ph)]))
        im.close()
        print(f"[{i+1}/{len(pages)}] {f.name}: {len(panels)} panels"
              + ("  (front matter)" if front else ""))

    db.save()
    print(f"done -> {db.path}")


if __name__ == "__main__":
    main()
