#!/usr/bin/env python3
"""3_assemble.py <work_dir>

Stage 3 of pipeline v1.5. Deterministic assembly -- no LLM. The page's words
are the script; nothing is paraphrased or invented.

Per page, per panel in reading order:
  1. NARRATOR: <scene sentence>            (from the vision pass; skipped if
                                            empty or basically a repeat)
  2. every transcribed line, in order:
       CAPTION  -> NARRATOR: <verbatim>
       DIALOGUE -> <SPEAKER>: <verbatim>
       SFX      -> handled by panelspeak: a character vocalization folds into
                   that character's speech; ambient SFX is dropped
Consecutive same-speaker lines are merged. Front-matter pages (no caption and
no dialogue anywhere) are skipped.

Reads   <work_dir>/manifest.json, <work_dir>/transcript.json
Writes  <work_dir>/narrative.json   {"<page>": [{"speaker","text"}, ...], ...}
"""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from panelspeak.onomatopoeia import normalize_vocalization  # noqa: E402

# a lettered word that's clearly a sound effect, not speech
SFX_SHAPE = re.compile(r"^[A-Z][A-Z'\-]{1,}[!?.]*$")
JUNK_RE = re.compile(r"decomics\.com|dccomics\.com|conversion by|^\s*\[\d\d:\d\d\]", re.I)
VOCATIVE_RE = re.compile(r"[,\"']\s*([A-Z][a-z]{2,15})\.?[\"']?\s*$")
SAID_NAME_RE = re.compile(r'said,?\s*["\u201c][^"\u201d]*,\s*([A-Z][a-z]{2,15})')
STOPWORDS = {"yes", "no", "sir", "please", "well", "okay", "now", "hey", "wait"}


def clean(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip().strip('"\u201c\u201d')
    return t


def resolve_names(transcript) -> dict:
    """SPEAKER-cluster -> real name, if named in the dialogue anywhere."""
    names = {}
    for entry in transcript.values():
        for ln in entry.get("lines", []):
            if ln["kind"] != "DIALOGUE":
                continue
            m = SAID_NAME_RE.search(ln["text"]) or VOCATIVE_RE.search(ln["text"])
            if m and m.group(1).lower() not in STOPWORDS:
                names.setdefault(ln.get("speaker") or "SPEAKER", m.group(1))
    if names:
        print(f"resolved names: {names}")
    return names


def is_scene_useful(scene: str, said: set) -> bool:
    if not scene or len(scene.split()) < 4:
        return False
    low = scene.lower()
    if any(low == s or low in s for s in said):
        return False
    return True


def sfx_segment(text, prev_speaker):
    """Return (speaker, text) to append for an SFX line, or None to drop.
    A recognised vocalization attaches to whoever just spoke; anything else
    (ambient BOOM/KRAKKA) is dropped -- no character 'says' it."""
    voc = normalize_vocalization(text)
    if voc and voc.is_known and prev_speaker and prev_speaker != "NARRATOR":
        if voc.prefer_narration:
            return ("NARRATOR", f"{prev_speaker.title()} {voc.narration}.")
        return (prev_speaker, f"{voc.spoken}.")
    return None


def merge_runs(segs):
    out = []
    for s in segs:
        if out and out[-1]["speaker"] == s["speaker"]:
            joiner = " " if out[-1]["text"].endswith((".", "!", "?", '"')) else ". "
            out[-1]["text"] = f"{out[-1]['text']}{joiner}{s['text']}".strip()
        else:
            out.append(dict(s))
    return out


def main():
    if len(sys.argv) != 2:
        print("Usage: 3_assemble.py <work_dir>", file=sys.stderr)
        sys.exit(2)

    work_dir = Path(sys.argv[1])
    manifest = json.load(open(work_dir / "manifest.json"))
    transcript = json.load(open(work_dir / "transcript.json"))
    names = resolve_names(transcript)

    narrative = {}
    for page in manifest["pages"]:
        pi = page["page_index"]
        panel_keys = [
            f"page{pi:03d}_panel{p['panel_index']:02d}" for p in page["panels"]
        ]
        has_words = any(
            ln["kind"] in ("CAPTION", "DIALOGUE")
            for k in panel_keys for ln in transcript.get(k, {}).get("lines", [])
        )
        if not has_words:
            print(f"page {pi}: front matter / no words -- skipped")
            continue

        segs = []
        for key in panel_keys:
            entry = transcript.get(key, {})
            said = {ln["text"].lower() for ln in entry.get("lines", [])}
            scene = entry.get("scene", "")
            if is_scene_useful(scene, said):
                segs.append({"speaker": "NARRATOR", "text": clean(scene)})

            prev_speaker = segs[-1]["speaker"] if segs else "NARRATOR"
            for ln in entry.get("lines", []):
                txt = clean(ln["text"])
                if not txt or JUNK_RE.search(txt):
                    continue
                if ln["kind"] == "CAPTION":
                    segs.append({"speaker": "NARRATOR", "text": txt})
                    prev_speaker = "NARRATOR"
                elif ln["kind"] == "SFX" or (not ln.get("speaker") and SFX_SHAPE.match(txt) and len(txt.split()) == 1):
                    added = sfx_segment(txt, prev_speaker)
                    if added:
                        segs.append({"speaker": added[0], "text": added[1]})
                else:  # DIALOGUE
                    spk = ln.get("speaker") or "SPEAKER"
                    spk = names.get(spk, spk)
                    if spk == "SPEAKER":
                        # unnamed speaker with no resolution -> reported speech
                        segs.append({"speaker": "NARRATOR",
                                     "text": f'A voice says, "{txt}"'})
                        prev_speaker = "NARRATOR"
                    else:
                        segs.append({"speaker": spk, "text": txt})
                        prev_speaker = spk

        segs = merge_runs(segs)
        narrative[str(pi)] = segs
        json.dump(narrative, open(work_dir / "narrative.json", "w"), indent=2)
        dlg = sum(1 for s in segs if s["speaker"] != "NARRATOR")
        print(f"page {pi}: {len(segs)} segments ({dlg} dialogue)")

    print(f"Done. Narrative: {work_dir / 'narrative.json'}")


if __name__ == "__main__":
    main()
