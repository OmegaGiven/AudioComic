"""panelspeak -- deterministic post-processing helpers for the AudioComic pipeline.

The four numbered scripts in ``scripts/`` do the model-driven work (panel
segmentation, vision analysis, narrative adaptation, TTS). Those models are
imperfect and non-deterministic. This package is the deterministic layer that
sits *around* them: lexicons, classifiers, and attribution rules that turn
messy model output into something stable enough to test and to feed forward.

Nothing here calls a model or the network. Everything is pure functions over
plain data, so it is fast and fully unit-testable. New pipeline features
(emotion-aware TTS, onomatopoeia handling, ...) land here first as tested
logic, then get wired into the scripts.
"""

from panelspeak.attribution import attribute_vocalization
from panelspeak.classify import refine_kind
from panelspeak.emotion import ParsedLine, Segment, merge_vocalizations, parse_line
from panelspeak.onomatopoeia import Vocalization, normalize_vocalization
from panelspeak.text_elements import BBox, Bubble, Character, ElementKind, TextElement

__all__ = [
    "BBox",
    "Bubble",
    "Character",
    "ElementKind",
    "TextElement",
    "Vocalization",
    "normalize_vocalization",
    "refine_kind",
    "attribute_vocalization",
    "ParsedLine",
    "Segment",
    "merge_vocalizations",
    "parse_line",
]
