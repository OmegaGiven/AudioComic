"""Line format with emotion hints, and vocalization merging.

Stage 3 currently emits ``SPEAKER: text``. To let the emotion-aware TTS
engines act, it will emit an optional parenthetical hint:

    MERCURY (panicked): You don't understand what you've done!
    NARRATOR: The lantern flickered out.
    MERCURY: Tsk.

:func:`parse_line` reads both the old and the new form (old form => no
emotion). :func:`merge_vocalizations` folds a standalone vocalization line into
an adjacent dialogue line from the *same* speaker, so ``"Tsk."`` followed by
``"You really thought that would work?"`` becomes one spoken line -- which is
what note 2 of the design asks for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from panelspeak.onomatopoeia import normalize_vocalization

# SPEAKER, optional "(emotion)", colon, text. SPEAKER is caps/space/'.- like
# the existing parser in 03_narrative.py, kept compatible.
_LINE = re.compile(
    r"^\s*(?P<speaker>[A-Z][A-Z0-9 '.\-]{0,40}?)"
    r"(?:\s*\((?P<emotion>[a-zA-Z ,\-]{2,30})\))?"
    r"\s*:\s*(?P<text>.+?)\s*$"
)

_PLACEHOLDER_SPEAKERS = {
    "SPEAKER", "CHARACTER", "VILLAIN", "MAN", "WOMAN", "ENTITY", "VOICE",
    "PERSON", "FIGURE", "STRANGER", "UNKNOWN", "CAPTION",
}


@dataclass(frozen=True)
class ParsedLine:
    speaker: str
    text: str
    emotion: str | None = None


@dataclass
class Segment:
    speaker: str
    text: str
    emotion: str | None = None
    #: nonverbal tags recognised in this segment, in order ("<sigh>", ...).
    nonverbal: list[str] = field(default_factory=list)


def parse_line(line: str) -> ParsedLine | None:
    """Parse one ``SPEAKER: text`` / ``SPEAKER (emotion): text`` line."""
    m = _LINE.match(line)
    if not m:
        return None
    speaker = m.group("speaker").strip()
    if speaker in _PLACEHOLDER_SPEAKERS:
        speaker = "NARRATOR"
    emotion = m.group("emotion")
    emotion = emotion.strip().lower() if emotion else None
    return ParsedLine(speaker=speaker, text=m.group("text").strip(), emotion=emotion)


def parse_script(raw_text: str) -> list[Segment]:
    """Parse a whole model response into segments, handling wrapped
    continuation lines the same way ``03_narrative.parse_narrative`` does."""
    segments: list[Segment] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = parse_line(line)
        if parsed:
            segments.append(Segment(parsed.speaker, parsed.text, parsed.emotion))
        elif segments:
            segments[-1].text += " " + line
    for seg in segments:
        seg.nonverbal = _nonverbal_tags(seg.text)
    return segments


def _nonverbal_tags(text: str) -> list[str]:
    tags: list[str] = []
    for token in re.findall(r"\*([a-z]+)\*|<([a-z]+)>", text.lower()):
        word = token[0] or token[1]
        voc = normalize_vocalization(word)
        if voc and voc.nonverbal_tag:
            tags.append(voc.nonverbal_tag)
    return tags


def _is_pure_vocalization(text: str) -> bool:
    """True if the line is essentially nothing but a recognised noise, allowing
    a doubled blurt like ``"tsk tsk"`` or ``"ha ha"``."""
    stripped = text.strip().strip("\"'*_()").strip()
    if not stripped:
        return False
    parts = re.sub(r"[.!?,\-]+", " ", stripped.lower()).split()
    if not parts or len(parts) > 2:
        return False
    if len(parts) == 2 and parts[0] != parts[1]:
        return False
    voc = normalize_vocalization(parts[0])
    return voc is not None and voc.is_known


def merge_vocalizations(segments: list[Segment]) -> list[Segment]:
    """Fold standalone vocalization lines into an adjacent same-speaker
    dialogue line. NARRATOR vocalizations and unattributable ones are left
    alone."""
    out: list[Segment] = []
    i = 0
    n = len(segments)
    while i < n:
        seg = segments[i]
        if (
            seg.speaker != "NARRATOR"
            and _is_pure_vocalization(seg.text)
            and (i + 1) < n
            and segments[i + 1].speaker == seg.speaker
            and not _is_pure_vocalization(segments[i + 1].text)
        ):
            nxt = segments[i + 1]
            merged_text = f"{seg.text.rstrip('.!? ')}. {nxt.text}".strip()
            out.append(Segment(
                speaker=seg.speaker,
                text=merged_text,
                emotion=seg.emotion or nxt.emotion,
                nonverbal=_nonverbal_tags(merged_text),
            ))
            i += 2
            continue
        if (
            seg.speaker != "NARRATOR"
            and _is_pure_vocalization(seg.text)
            and out
            and out[-1].speaker == seg.speaker
            and not _is_pure_vocalization(out[-1].text)
        ):
            prev = out[-1]
            prev.text = f"{prev.text.rstrip()} {seg.text.strip()}".strip()
            prev.emotion = prev.emotion or seg.emotion
            prev.nonverbal = _nonverbal_tags(prev.text)
            i += 1
            continue
        out.append(seg)
        i += 1
    return out
