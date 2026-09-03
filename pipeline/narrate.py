"""Phase 6b -- polish the narrator's connective tissue with a constrained LLM.

`assemble` places every verbatim line (dialogue, captions) deterministically
and marks the generated scene-setting lines with "gen": true. Those generated
lines read as a flat list of panel descriptions. Here an LLM rewrites ONLY
those lines into smooth narration, one page at a time, under a hard contract:

  * dialogue and caption lines are passed as read-only context and must come
    back byte-for-byte, in the same order
  * the rewrite may not introduce a name or place not already in the script
  * the rewrite may not add events -- only rephrase what the scene lines say
  * total narrator wordcount may not balloon

Any page whose rewrite breaks the contract keeps its deterministic version.
If Ollama is unreachable the phase is a no-op.

    python -m pipeline.narrate <work_dir>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pipeline.llm import ask_llm

MODEL = "devstral:24b"
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]+")
_PROPER = re.compile(r"\b([A-Z][a-z]{2,}|[A-Z]{3,})\b")
# capitalised words that are not names -- allowed to appear anywhere
_SAFE_CAPS = set("""
A An The And But Or So If Then As At By For From In Into Of On To Up With He She It
They We You His Her Its Their Our My Your This That These Those There Here Now When
While After Before Behind Above Below Beside Between Near Over Under Through Across
Suddenly Meanwhile Later Nearby Outside Inside Somewhere Everywhere Nothing Something
Someone Everyone Nobody One Two Three Four Five Six Seven Eight Nine Ten
Monday Tuesday Wednesday Thursday Friday Saturday Sunday
January February March April May June July August September October November December
""".split())


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", t.lower())).strip()


def _allowed_names(narr: dict, db: dict) -> set[str]:
    ok: set[str] = set(_SAFE_CAPS)
    for e in db.get("entities", []):
        if e.get("name"):
            ok.update(_WORD.findall(e["name"]))
    # any proper noun the reader will already have heard in a verbatim line
    for segs in narr.values():
        for s in segs:
            if not s.get("gen"):
                ok.update(_PROPER.findall(s["text"]))
    return {w.lower() for w in ok}


def _new_proper_nouns(text: str, allowed: set[str]) -> list[str]:
    bad = []
    for m in _PROPER.finditer(text):
        tok = m.group(1)
        if tok.lower() in allowed:
            continue
        after = text[m.end():m.end() + 3]
        nxt = text[m.end():].lstrip()[:1]
        pre = text[:m.start()].rstrip()
        sentence_initial = (not pre) or pre[-1] in ".!?:\""
        if tok.isupper() and len(tok) >= 3:          # SHOUTED NAME / place caption
            bad.append(tok)
        elif after[:2] in ("'s", "’s"):          # possessive -> a character
            bad.append(tok)
        elif not sentence_initial:                     # capitalised mid-sentence
            bad.append(tok)
        elif nxt.isupper():                            # "Gotham City" proper phrase
            bad.append(tok)
    return bad


def _render_script(segs: list[dict]) -> str:
    rows = []
    for i, s in enumerate(segs):
        if s.get("gen"):
            role = "SCENE"
        elif s["speaker"] == "NARRATOR":
            role = "CAPTION"
        else:
            role = s["speaker"]
        rows.append(f"[{i}] {role}: {s['text']}")
    return "\n".join(rows)


_ROW_RE = re.compile(r"^\s*\[(\d+)\]\s*([^:]+):\s*(.*)$")


def _parse_script(text: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for ln in text.splitlines():
        m = _ROW_RE.match(ln)
        if m:
            out[int(m.group(1))] = m.group(3).strip()
    return out


PROMPT = """You are editing the narration of an audio comic, one page at a time.

Below is the page's script. Each line is [index] ROLE: text.
- ROLE "SCENE" is rough scene-setting written by a machine -- REWRITE these into smooth, vivid narrator prose.
- ROLE "CAPTION" is a narration box and every other ROLE is spoken dialogue -- copy these lines back EXACTLY, unchanged, in the same place.

Rules for the SCENE lines you rewrite:
- Only rephrase what the SCENE line already says. Do not add events, actions, dialogue, weather, sounds, or thoughts that are not there.
- Do not use any name, place, or proper noun that does not already appear somewhere in this script.
- Keep it about the same length -- tighten, do not expand.
- You may merge two adjacent SCENE lines into one (leave the freed index blank after its number) but never merge a SCENE line into a CAPTION or dialogue line.

Output the full script back, every [index] in order, same ROLE labels. No commentary.

SCRIPT:
{script}
"""


def _polish_page(segs: list[dict], allowed: set[str]) -> tuple[list[dict], str]:
    gen_idx = [i for i, s in enumerate(segs) if s.get("gen")]
    if not gen_idx:
        return segs, "no scene lines"

    resp = ask_llm(PROMPT.replace("{script}", _render_script(segs)),
                   model=MODEL, num_predict=1600, timeout=300)
    if not resp:
        return segs, "llm unavailable"
    rewritten = _parse_script(resp)

    # contract check 1: every non-SCENE line returned unchanged
    for i, s in enumerate(segs):
        if s.get("gen"):
            continue
        if _norm(rewritten.get(i, "")) != _norm(s["text"]):
            return segs, f"verbatim line [{i}] altered"

    # contract check 2 + 3: no invented proper nouns, no ballooning
    old_words = sum(len(_WORD.findall(segs[i]["text"])) for i in gen_idx)
    new_words = 0
    for i in gen_idx:
        nt = rewritten.get(i, "").strip()
        new_words += len(_WORD.findall(nt))
        bad = _new_proper_nouns(nt, allowed)
        if bad:
            return segs, f"introduced {bad}"
    if old_words and new_words > 1.7 * old_words:
        return segs, f"expanded {old_words}->{new_words} words"

    out: list[dict] = []
    for i, s in enumerate(segs):
        if not s.get("gen"):
            out.append(s)
            continue
        nt = rewritten.get(i, "").strip().strip('"“” ')
        if nt:
            out.append({**s, "text": nt})
        # empty -> the model merged it upward; drop the line
    return out, "ok"


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m pipeline.narrate <work_dir>", file=sys.stderr)
        sys.exit(2)
    work = Path(sys.argv[1])
    npath = work / "narrative.json"
    narr = json.loads(npath.read_text())
    try:
        db = json.loads((work / "comic.json").read_text())
    except OSError:
        db = {}
    allowed = _allowed_names(narr, db)

    kept, polished = 0, 0
    for page in sorted(narr, key=int):
        new_segs, why = _polish_page(narr[page], allowed)
        if why == "ok":
            narr[page] = new_segs
            polished += 1
            print(f"[{page}] polished")
        else:
            kept += 1
            if why not in ("no scene lines",):
                print(f"[{page}] kept deterministic ({why})")

    npath.write_text(json.dumps(narr, indent=2))
    print(f"narrate: {polished} pages polished, {kept} kept as-is")
    print(f"done -> {npath}")


if __name__ == "__main__":
    main()
