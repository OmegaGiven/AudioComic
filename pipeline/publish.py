"""Phase 7b -- encode the render and copy it to the media library.

    python -m pipeline.publish <work_dir> [<render.wav>]

Reads the issue metadata from comic.json, encodes the wav to a compact mp3,
and copies it to the Audiobookshelf library so it shows up on the phone
without anyone moving files by hand.

Config (env, with defaults for this tailnet):
    MEDIA_HOST   go
    MEDIA_ROOT   /mnt/media/BOOKS/Audiobooks/AudioComics
    MP3_BITRATE  64k
    ABS_URL      (optional) e.g. http://go:13378  -- triggers a library scan
    ABS_TOKEN    (optional) Audiobookshelf API token
    ABS_LIBRARY  (optional) library id to scan
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

from pipeline.comicdb import ComicDB

MEDIA_HOST = os.environ.get("MEDIA_HOST", "go")
MEDIA_ROOT = os.environ.get("MEDIA_ROOT", "/mnt/media/BOOKS/Audiobooks/AudioComics")
MP3_BITRATE = os.environ.get("MP3_BITRATE", "64k")


def _slug(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w .'-]", "", s)).strip() or "Untitled"


def title_for(db: ComicDB) -> tuple[str, str]:
    """-> (series, issue title)"""
    issue = db.issue
    raw_series = os.environ.get("COMIC_SERIES") or issue.get("series") or ""
    series = _slug(raw_series) if raw_series.strip() else ""
    number = os.environ.get("COMIC_NUMBER") or issue.get("number")
    number = int(number) if number not in (None, "") else None
    src = Path(issue.get("source", "")).stem
    if not series:
        # guess "Series NN" from the archive name
        m = (re.match(r"^(.+?)[\s_#-]+0*(\d{1,4})\s*$", src)
             or re.match(r"^([A-Za-z]{2,}?)0*(\d{1,4})\s*$", src))
        series = _slug(m.group(1)) if m else (_slug(src) or "AudioComics")
        number = number or (int(m.group(2)) if m else None)
    name = f"{series} {int(number):02d}" if number is not None else (_slug(src) or series)
    return series, _slug(name)


def encode(wav: Path, mp3: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
         "-c:a", "libmp3lame", "-b:a", MP3_BITRATE, "-ac", "1", str(mp3)],
        check=True,
    )


def copy_to_media(mp3: Path, series: str, name: str) -> str:
    dest_dir = f"{MEDIA_ROOT}/{series}"
    dest = f"{dest_dir}/{name}.mp3"
    subprocess.run(["ssh", MEDIA_HOST, f"mkdir -p {json.dumps(dest_dir)}"], check=True)
    subprocess.run(["scp", "-q", str(mp3), f"{MEDIA_HOST}:{json.dumps(dest)}"], check=True)
    return dest


def trigger_abs_scan() -> None:
    url, token, lib = (os.environ.get("ABS_URL"), os.environ.get("ABS_TOKEN"),
                       os.environ.get("ABS_LIBRARY"))
    if not (url and token and lib):
        print("  (set ABS_URL / ABS_TOKEN / ABS_LIBRARY to auto-scan the library)")
        return
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/libraries/{lib}/scan", method="POST",
        headers={"Authorization": f"Bearer {token}"})
    try:
        urllib.request.urlopen(req, timeout=15)
        print("  Audiobookshelf scan triggered")
    except Exception as e:
        print(f"  ABS scan failed: {e}")


def main() -> None:
    if not (2 <= len(sys.argv) <= 3):
        print("usage: python -m pipeline.publish <work_dir> [<render.wav>]", file=sys.stderr)
        sys.exit(2)
    work_dir = Path(sys.argv[1])
    db = ComicDB.load(work_dir)
    wav = Path(sys.argv[2]) if len(sys.argv) == 3 else next(
        (p for p in work_dir.glob("*.wav")), None)
    if not wav or not wav.exists():
        print(f"no render wav found (looked in {work_dir})", file=sys.stderr)
        sys.exit(1)

    series, name = title_for(db)
    mp3 = work_dir / f"{name}.mp3"
    print(f"encoding {wav.name} -> {mp3.name} ({MP3_BITRATE})")
    encode(wav, mp3)
    print(f"copying to {MEDIA_HOST}:{MEDIA_ROOT}/{series}/")
    dest = copy_to_media(mp3, series, name)
    trigger_abs_scan()
    print(f"published -> {MEDIA_HOST}:{dest}  ({mp3.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
