"""Attribute a free-floating vocalization to the character who made it.

A ``tsk`` inside a speech bubble belongs to the bubble owner -- easy. A ``tsk``
lettered next to a character with no bubble needs a rule. This module is that
rule, kept deterministic and conservative: when it can't be reasonably sure, it
returns ``None`` and the caller treats the noise as ambient rather than putting
words in the wrong character's mouth (the same failure ``04_tts_render.py``
already guards against for voice assignment).
"""

from __future__ import annotations

from panelspeak.text_elements import Bubble, Character, TextElement

#: a candidate character must be closer than this fraction of the panel
#: diagonal to the vocalization's centre to be considered its source.
_NEAR_FRACTION = 0.33


def attribute_vocalization(
    element: TextElement,
    characters: list[Character],
    bubbles: list[Bubble] | None = None,
    *,
    panel_diagonal: float | None = None,
) -> str | None:
    """Return the name of the character who made ``element``, or ``None``.

    Resolution order:

    1. If the element is inside a bubble, the bubble owner said it.
    2. Otherwise, if exactly one character is *near* it, they said it -- a
       character with an open mouth wins ties and satisfies the check on its
       own.
    3. Otherwise ``None`` (ambient / unattributable).
    """
    bubbles = bubbles or []

    # 1. bubbled -> owner
    if element.bubble_id:
        for b in bubbles:
            if b.bubble_id == element.bubble_id:
                return b.owner or None

    if not characters or element.bbox is None:
        # no geometry to reason about; fall back to an open-mouthed loner
        open_mouthed = [c for c in characters if c.mouth_open]
        if len(open_mouthed) == 1:
            return open_mouthed[0].name
        return None

    if panel_diagonal is None:
        # derive from the spread of everything we know about
        xs = [element.bbox.center[0]] + [c.bbox.center[0] for c in characters if c.bbox]
        ys = [element.bbox.center[1]] + [c.bbox.center[1] for c in characters if c.bbox]
        panel_diagonal = (((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5) or 1.0

    threshold = _NEAR_FRACTION * panel_diagonal
    scored: list[tuple[float, bool, str]] = []
    for c in characters:
        if c.bbox is None:
            continue
        d = element.bbox.distance_to(c.bbox)
        if d <= threshold:
            scored.append((d, c.mouth_open, c.name))

    if not scored:
        return None
    if len(scored) == 1:
        return scored[0][2]

    # multiple nearby: an open mouth breaks the tie if it's unique
    open_ones = [s for s in scored if s[1]]
    if len(open_ones) == 1:
        return open_ones[0][2]

    # still ambiguous -> don't guess
    return None
