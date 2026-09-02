"""Deterministic correction of the vision model's element labels.

Stage 2 gives every text run a first-guess label. It is frequently wrong at
exactly the boundary we care about: it calls a big ``BOOM`` "DIALOGUE", or it
calls a bubbled ``tsk`` an "SFX". :func:`refine_kind` applies cheap geometric
and lexical rules to fix the obvious cases, the same way ``03_narrative.py``
deterministically rewrites invented placeholder speakers instead of trusting
the prompt.
"""

from __future__ import annotations

import re

from panelspeak.onomatopoeia import normalize_vocalization
from panelspeak.text_elements import ElementKind

# a token in ALL CAPS, no lowercase, mostly consonants, optional !/? -- the
# classic sound-effect shape (KRAKKA, THWIP, SKREEE, WHUMP)
_SFX_SHAPE = re.compile(r"^[A-Z][A-Z'’\-]{1,}[!?]*$")

_DICTIONARY_STOPWORDS = {
    "THE", "AND", "YOU", "WHAT", "NO", "YES", "STOP", "HELP", "RUN", "GO",
    "NOW", "HERE", "THERE", "WAIT", "OKAY", "OK",
}


def _is_sfx_shape(text: str) -> bool:
    t = text.strip()
    if " " in t:
        return False
    if not _SFX_SHAPE.match(t):
        return False
    core = re.sub(r"[^A-Za-z]", "", t)
    if core.upper() in _DICTIONARY_STOPWORDS:
        return False
    vowels = sum(c in "AEIOU" for c in core.upper())
    return len(core) >= 3 and (vowels <= 2 or len(set(core.upper())) <= 3)


def refine_kind(
    raw_label: str | ElementKind,
    text: str,
    *,
    in_bubble: bool = False,
    lettering: str = "normal",
    area_ratio: float = 0.0,
) -> ElementKind:
    """Return the corrected :class:`ElementKind` for one text element.

    Parameters
    ----------
    raw_label   the vision model's guess ("DIALOGUE" / "CAPTION" / ...).
    text        the transcribed text.
    in_bubble   is the text inside a located speech/thought bubble.
    lettering   "normal" for bubble/caption lettering, "display" for big
                stylised sound-effect art.
    area_ratio  element bbox area / panel bbox area (0 if unknown).
    """
    try:
        raw = ElementKind(str(raw_label).strip().upper())
    except ValueError:
        raw = ElementKind.UNKNOWN

    stripped = text.strip()
    if not stripped:
        return ElementKind.UNKNOWN

    voc = normalize_vocalization(stripped)
    big_display = lettering == "display" or area_ratio >= 0.14

    # 1. Big stylised lettering outside any bubble is ambient SFX, full stop --
    #    even if the model called it dialogue.
    if big_display and not in_bubble:
        return ElementKind.SFX

    # 2. Inside a bubble: it's something a character produces.
    if in_bubble:
        if voc is not None and (voc.is_known or len(stripped.split()) == 1):
            return ElementKind.VOCALIZATION
        return ElementKind.DIALOGUE

    # 3. Not in a bubble, normal lettering.
    if raw is ElementKind.CAPTION:
        return ElementKind.CAPTION
    if voc is not None and voc.is_known:
        # a recognised noise floating by a character -- attribution decides
        # later whether it actually lands on someone
        return ElementKind.VOCALIZATION
    if _is_sfx_shape(stripped):
        return ElementKind.SFX
    if voc is not None and voc.canonical == "" and len(stripped.split()) == 1:
        # unknown blurt, lowercase-ish, single token -> lean SFX when free-floating
        return ElementKind.SFX if not stripped.islower() else ElementKind.VOCALIZATION

    if raw in (ElementKind.DIALOGUE, ElementKind.VOCALIZATION, ElementKind.SFX):
        return raw
    return ElementKind.CAPTION if raw is ElementKind.UNKNOWN and len(stripped.split()) > 3 else raw
