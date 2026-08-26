#!/usr/bin/env python3
"""01_extract_and_segment.py <cbr/cbz file> <work_dir>

Deterministic, fast phase of the comic-audiobook pipeline:
1. Extracts the archive (unrar for .cbr, unzip for .cbz).
2. Classifies pages: filters non-page junk (release-group credit images --
   detected by filename not matching the comic's own numbering scheme),
   flags double-page spreads (aspect ratio ~2x the modal single-page ratio).
3. Runs Kumiko panel segmentation on every real page (spreads included --
   Kumiko just segments whatever image bounds it's given).
4. Crops every panel to its own image file, in reading order.
5. Writes manifest.json: the full page/panel structure for later phases.

Usage: 01_extract_and_segment.py <path/to/issue.cbr> <work_dir>
work_dir gets: extracted/, panels/, manifest.json
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KUMIKO_DIR = REPO_ROOT / "tools" / "kumiko"
VENV_PYTHON = Path(sys.executable)  # run this script with your venv's python and Kumiko inherits it


def extract(archive_path: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    suffix = archive_path.suffix.lower()
    if suffix == ".cbr":
        subprocess.run(["unrar", "x", "-y", str(archive_path), str(dest) + "/"],
                        check=True, capture_output=True, text=True)
    elif suffix == ".cbz":
        subprocess.run(["unzip", "-o", str(archive_path), "-d", str(dest)],
                        check=True, capture_output=True, text=True)
    else:
        raise ValueError(f"Unsupported archive type: {suffix}")


def find_page_files(extracted_dir: Path):
    """Real comic pages share one consistent naming scheme with a numeric
    index (e.g. 'Blackest Night 01-003.jpg'). Anything that doesn't match
    that scheme (credit/ad images dropped in by the scanning group, e.g.
    'Kingpin8.jpg') is junk, not a page -- filtered out here rather than
    guessed at downstream."""
    all_images = sorted(extracted_dir.rglob("*.jpg")) + sorted(extracted_dir.rglob("*.png")) + sorted(extracted_dir.rglob("*.jpeg"))
    numbered = [f for f in all_images if re.search(r"-(\d+)\.\w+$", f.name)]
    numbered.sort(key=lambda f: int(re.search(r"-(\d+)\.\w+$", f.name).group(1)))
    skipped = [f for f in all_images if f not in numbered]
    return numbered, skipped


def classify_pages(page_files):
    """Returns list of dicts: {path, index, width, height, is_spread}.
    Spread = aspect ratio roughly double the modal single-page ratio."""
    from PIL import Image
    pages = []
    ratios = []
    for f in page_files:
        with Image.open(f) as im:
            w, h = im.size
        pages.append({"path": str(f), "width": w, "height": h, "ratio": w / h})
        ratios.append(round(w / h, 2))
    modal_ratio = max(set(ratios), key=ratios.count)
    for p in pages:
        p["is_spread"] = p["ratio"] > modal_ratio * 1.5
    return pages


def run_kumiko(image_path: str):
    result = subprocess.run(
        [str(VENV_PYTHON), "kumiko", "-i", image_path],
        cwd=str(KUMIKO_DIR), capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        print(f"  WARN: kumiko failed on {image_path}: {result.stderr[-300:]}", file=sys.stderr)
        return None
    try:
        return json.loads(result.stdout)[0]
    except Exception as e:
        print(f"  WARN: could not parse kumiko output for {image_path}: {e}", file=sys.stderr)
        return None


def main():
    if len(sys.argv) != 3:
        print("Usage: 01_extract_and_segment.py <issue.cbr> <work_dir>", file=sys.stderr)
        sys.exit(2)

    archive_path = Path(sys.argv[1])
    work_dir = Path(sys.argv[2])
    extracted_dir = work_dir / "extracted"
    panels_dir = work_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {archive_path.name}...")
    extract(archive_path, extracted_dir)

    page_files, skipped = find_page_files(extracted_dir)
    print(f"Found {len(page_files)} real pages, skipped {len(skipped)} non-page file(s): {[f.name for f in skipped]}")

    pages = classify_pages(page_files)
    spread_count = sum(1 for p in pages if p["is_spread"])
    print(f"Detected {spread_count} double-page spread(s).")

    from PIL import Image
    manifest = {"source": str(archive_path), "pages": []}

    for i, page in enumerate(pages):
        print(f"[{i+1}/{len(pages)}] Segmenting {Path(page['path']).name} "
              f"({'SPREAD' if page['is_spread'] else 'single'})...")
        kumiko_result = run_kumiko(page["path"])
        panels = kumiko_result["panels"] if kumiko_result else [[0, 0, page["width"], page["height"]]]

        page_entry = {
            "page_index": i,
            "source_file": page["path"],
            "is_spread": page["is_spread"],
            "panels": [],
        }

        im = Image.open(page["path"])
        for pi, (x, y, w, h) in enumerate(panels):
            panel_filename = f"page{i:03d}_panel{pi:02d}.jpg"
            panel_path = panels_dir / panel_filename
            im.crop((x, y, x + w, y + h)).save(panel_path, quality=92)
            page_entry["panels"].append({
                "panel_index": pi,
                "file": str(panel_path),
                "bbox": [x, y, w, h],
            })
        im.close()
        manifest["pages"].append(page_entry)

    manifest_path = work_dir / "manifest.json"
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    total_panels = sum(len(p["panels"]) for p in manifest["pages"])
    print(f"Done. {len(manifest['pages'])} pages, {total_panels} panels. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
