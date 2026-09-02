"""Shared data types for panel text elements.

Stage 2 (vision) is expected to emit, per panel, a list of text elements with
a rough geometry and a first-guess label. Everything downstream operates on
these types.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class ElementKind(enum.StrEnum):
    """What a piece of text in a panel *is*, for narration purposes."""

    DIALOGUE = "DIALOGUE"          # a character speaking, in a speech bubble
    CAPTION = "CAPTION"            # narration box (yellow/green), no speaker
    VOCALIZATION = "VOCALIZATION"  # a noise a character makes: "tsk", "aaah", *sigh*
    SFX = "SFX"                    # ambient sound-effect lettering: BOOM, KRAKKA
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in panel pixel space. ``(x, y)`` is the top-left."""

    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def distance_to(self, other: BBox) -> float:
        cx, cy = self.center
        ox, oy = other.center
        return ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5


@dataclass(frozen=True)
class TextElement:
    """One transcribed run of text within a panel."""

    text: str
    kind: ElementKind = ElementKind.UNKNOWN
    bbox: BBox | None = None
    #: "normal" = ordinary bubble/caption lettering; "display" = big stylised
    #: sound-effect art. Vision model reports this from the lettering style.
    lettering: str = "normal"
    #: id of the speech bubble this text sits inside, if any.
    bubble_id: str | None = None
    #: raw speaker label the vision model guessed, verbatim (may be junk).
    raw_speaker: str | None = None


@dataclass(frozen=True)
class Character:
    """A character the vision model located in the panel."""

    name: str
    bbox: BBox | None = None
    mouth_open: bool = False


@dataclass(frozen=True)
class Bubble:
    """A speech/thought bubble located in the panel."""

    bubble_id: str
    owner: str | None = None      # character name the tail points to
    bbox: BBox | None = None
    kind: str = "speech"          # "speech" | "thought" | "shout" | "whisper"


@dataclass
class Panel:
    """Everything stage 2 knows about one panel."""

    key: str
    bbox: BBox | None = None
    elements: list[TextElement] = field(default_factory=list)
    characters: list[Character] = field(default_factory=list)
    bubbles: list[Bubble] = field(default_factory=list)
