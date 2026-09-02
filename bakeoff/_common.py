"""Shared helpers for the TTS bake-off engine scripts.

Each engine script imports this, gets the passage as a list of prepared
segments, synthesizes each one to a wav, and calls :func:`finish` to stitch
the full clip and drop a manifest. :func:`build_page` then turns every
engine's output into one comparison page.
"""

from __future__ import annotations

import json
import subprocess
import sys
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

BAKEOFF_DIR = Path(__file__).resolve().parent
REPO_ROOT = BAKEOFF_DIR.parent
OUT_DIR = BAKEOFF_DIR / "out"
REFS_DIR = BAKEOFF_DIR / "refs"

sys.path.insert(0, str(REPO_ROOT))


@dataclass
class Seg:
    idx: int
    speaker: str
    emotion: str
    kind: str
    #: text as written in passage.json
    raw_text: str
    #: text after onomatopoeia normalization -- what the engine should speak
    speak_text: str
    #: non-verbal tags recognised (e.g. "<sigh>"), for tag-capable engines
    nonverbal: list[str] = field(default_factory=list)
    #: 0..1 intensity hint from stretched letters / punctuation
    intensity: float = 0.0
    ref: str | None = None


def load_passage() -> tuple[dict, list[Seg]]:
    data = json.loads((BAKEOFF_DIR / "passage.json").read_text())
    try:
        from panelspeak.onomatopoeia import normalize_vocalization
    except Exception:
        normalize_vocalization = lambda s: None  # noqa: E731

    cast = data["cast"]
    segs: list[Seg] = []
    for i, s in enumerate(data["segments"]):
        raw = s["text"]
        speak = raw
        tags: list[str] = []
        intensity = 0.0

        # replace a leading/standalone *marker* or bare vocalization token
        voc = normalize_vocalization(raw.strip().strip("*"))
        if voc is not None:
            intensity = voc.intensity
            if voc.nonverbal_tag:
                tags.append(voc.nonverbal_tag)
            # whole segment is just the noise -> use the spoken fallback
            if voc.is_known and len(raw.split()) <= 2:
                speak = voc.spoken
        else:
            # inline "*sigh* rest of line" -> pull the marker out
            import re
            m = re.match(r"\s*\*([a-z]+)\*\s*(.*)", raw, re.I)
            if m:
                inner = normalize_vocalization(m.group(1))
                if inner is not None:
                    intensity = inner.intensity
                    if inner.nonverbal_tag:
                        tags.append(inner.nonverbal_tag)
                    speak = (inner.spoken + ". " + m.group(2)).strip()

        segs.append(Seg(
            idx=i,
            speaker=s["speaker"],
            emotion=s.get("emotion", "neutral"),
            kind=s.get("kind", "dialogue"),
            raw_text=raw,
            speak_text=speak,
            nonverbal=tags,
            intensity=intensity,
            ref=cast.get(s["speaker"], {}).get("ref"),
        ))
    return data, segs


def engine_out(engine: str) -> Path:
    d = OUT_DIR / engine
    d.mkdir(parents=True, exist_ok=True)
    return d


def ref_path(seg: Seg) -> Path | None:
    if not seg.ref:
        return None
    p = REFS_DIR / seg.ref
    return p if p.exists() else None


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate() or 1)


def concat_wavs(paths: list[Path], out_path: Path, gap_s: float = 0.35) -> None:
    """Stitch wavs with a short silence between segments. Resamples nothing --
    engine scripts must emit a consistent rate."""
    paths = [p for p in paths if p and p.exists()]
    if not paths:
        raise SystemExit("no segment wavs to concatenate")
    with wave.open(str(paths[0]), "rb") as first:
        params = first.getparams()
    gap_frames = int(gap_s * params.framerate)
    silence = b"\x00" * (gap_frames * params.sampwidth * params.nchannels)
    with wave.open(str(out_path), "wb") as out:
        out.setparams(params)
        for i, p in enumerate(paths):
            with wave.open(str(p), "rb") as w:
                if w.getframerate() != params.framerate:
                    raise SystemExit(
                        f"{p.name}: {w.getframerate()} Hz != {params.framerate} Hz; "
                        "engine must emit one rate")
                out.writeframes(w.readframes(w.getnframes()))
            if i != len(paths) - 1:
                out.writeframes(silence)


def to_opus(wav_path: Path) -> Path | None:
    """Compress to opus for the web page (needs ffmpeg). Returns None on failure."""
    opus = wav_path.with_suffix(".opus")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
             "-c:a", "libopus", "-b:a", "48k", "-ac", "1", str(opus)],
            check=True, capture_output=True,
        )
        return opus
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def finish(engine: str, meta: dict, seg_wavs: list[Path], segs: list[Seg]) -> None:
    """Concatenate, compress, and write out/<engine>/manifest.json."""
    d = engine_out(engine)
    full = d / "full.wav"
    concat_wavs(seg_wavs, full)

    entries = []
    for seg, wav in zip(segs, seg_wavs, strict=False):
        opus = to_opus(wav) if wav and wav.exists() else None
        entries.append({
            **asdict(seg),
            "wav": wav.name if wav and wav.exists() else None,
            "opus": opus.name if opus else None,
            "duration_s": round(wav_duration(wav), 2) if wav and wav.exists() else None,
        })
    full_opus = to_opus(full)
    manifest = {
        "engine": engine,
        "meta": meta,
        "full_wav": full.name,
        "full_opus": full_opus.name if full_opus else None,
        "full_duration_s": round(wav_duration(full), 2),
        "segments": entries,
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[{engine}] wrote {full}  ({manifest['full_duration_s']}s)")


def finish_full(engine: str, meta: dict, full_wav: Path, segs: list[Seg]) -> None:
    """For engines (VibeVoice) that generate the whole passage in one pass and
    can't cleanly hand back per-segment wavs."""
    d = engine_out(engine)
    dst = d / "full.wav"
    if full_wav.resolve() != dst.resolve():
        dst.write_bytes(full_wav.read_bytes())
    full_opus = to_opus(dst)
    manifest = {
        "engine": engine,
        "meta": meta,
        "full_wav": dst.name,
        "full_opus": full_opus.name if full_opus else None,
        "full_duration_s": round(wav_duration(dst), 2),
        "segments": [{**asdict(s), "wav": None, "opus": None, "duration_s": None} for s in segs],
        "per_segment_available": False,
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[{engine}] wrote {dst}  ({manifest['full_duration_s']}s, full-only)")
