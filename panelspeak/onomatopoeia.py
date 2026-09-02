"""Onomatopoeia / vocalization lexicon.

Comics are full of lettered noises. Two kinds matter to us:

* **Vocalizations** -- a noise a *character* makes: ``tsk``, ``aaah``, ``ugh``,
  ``*sigh*``, ``heh``. These should come out of that character's mouth in the
  audio, with appropriate emotion.
* **Ambient SFX** -- ``BOOM``, ``KRAKKA``, ``THWIP``. Nobody "says" these.
  Handled elsewhere (narrator prose now, a real SFX layer later).

This module only concerns the first kind. :func:`normalize_vocalization` takes
a raw surface string and, if it recognises a vocalization, returns a
:class:`Vocalization` describing how to render it:

* ``nonverbal_tag`` -- an inline tag for engines that support them
  (Orpheus: ``<sigh>``, ``<laugh>`` ...). ``None`` if there isn't a good one.
* ``spoken`` -- a plain-text fallback the TTS can just read, for engines
  without tags (Piper, Chatterbox, VibeVoice).
* ``emotion`` -- a hint the emotion-aware engines can act on.
* ``intensity`` -- 0..1, lifted from letter-stretching and ``!`` count
  (``AAAAAHHH!!!`` is more intense than ``aah``).

An unrecognised token that still *looks* like a vocalization (a short,
mostly-consonant blurt, not a real word, not a sentence) is returned with
``canonical == ""`` and ``spoken == surface`` so it is passed through to the
neural engine verbatim rather than dropped. Anything that looks like an actual
sentence returns ``None``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Vocalization:
    surface: str            # exactly what was in the panel
    canonical: str          # lexicon key, or "" for a recognised-shape unknown
    emotion: str | None
    nonverbal_tag: str | None
    spoken: str             # clean interjection for a plain TTS engine; "" = don't
                            # try to voice it, narrate it instead
    narration: str          # verb phrase for a narrator beat: "{subject} {narration}."
    intensity: float        # 0.0 .. 1.0

    @property
    def is_known(self) -> bool:
        return self.canonical != ""

    @property
    def prefer_narration(self) -> bool:
        """True when a plain (non-emotive) engine should turn this into a
        narrator action beat rather than read it out. Breathy / wordless
        noises never sound right spelled phonetically."""
        return not self.spoken


# canonical -> (emotion, nonverbal_tag, spoken_interjection, narration_verb_phrase)
# spoken == "" means: there's no clean way to voice this; narrate it.
_LEXICON: dict[str, tuple[str | None, str | None, str, str]] = {
    "tsk":    ("disapproval",    None,        "Tsk",   "clicked their tongue"),
    "sigh":   ("weary",          "<sigh>",    "",      "sighed"),
    "gasp":   ("shock",          "<gasp>",    "",      "gasped"),
    "ugh":    ("disgust",        "<groan>",   "Ugh",   "groaned"),
    "argh":   ("frustration",    "<groan>",   "Argh",  "let out a frustrated groan"),
    "grr":    ("anger",          None,        "Grrr",  "growled"),
    "laugh":  ("amused",         "<laugh>",   "Ha, ha","laughed"),
    "chuckle":("amused",         "<chuckle>", "Heh",   "chuckled"),
    "scoff":  ("scorn",          None,        "Hmph",  "scoffed"),
    "hmm":    ("thoughtful",     None,        "Hmm",   "hummed thoughtfully"),
    "huh":    ("confused",       None,        "Huh",   "grunted, confused"),
    "ahem":   ("pointed",        "<cough>",   "Ahem",  "cleared their throat"),
    "cough":  ("neutral",        "<cough>",   "",      "coughed"),
    "shush":  ("urgent",         None,        "Shh",   "hissed for quiet"),
    "scream": ("fear",           None,        "",      "screamed"),
    "whimper":("distress",       None,        "",      "whimpered"),
    "oof":    ("pain",           None,        "Oof",   "grunted"),
    "whew":   ("relief",         None,        "Whew",  "let out a breath"),
    "mmm":    ("pleased",        None,        "Mmm",   "made a low, pleased sound"),
    "psst":   ("conspiratorial", None,        "Psst",  "whispered"),
    "boo":    ("mocking",        None,        "Boo",   "jeered"),
    "yawn":   ("bored",          "<yawn>",    "",      "yawned"),
    "sniff":  ("tearful",        "<sniff>",   "",      "sniffled"),
    "sob":    ("crying",         None,        "",      "sobbed"),
    "gulp":   ("nervous",        None,        "",      "swallowed hard"),
    "pant":   ("winded",         None,        "",      "panted"),
}

# surface spelling (already lowercased + de-stretched) -> canonical key
_ALIASES: dict[str, str] = {
    "tsk": "tsk", "tch": "tsk", "tut": "tsk", "tsktsk": "tsk", "tuttut": "tsk",
    "sigh": "sigh", "haah": "sigh", "hah": "sigh",  # bare "hah" exhale; "haha" -> laugh below
    "gasp": "gasp", "hgh": "gasp",
    "ugh": "ugh", "urgh": "ugh", "eugh": "ugh", "bleh": "ugh",
    "argh": "argh", "aargh": "argh", "aaargh": "argh", "gah": "argh", "grah": "argh",
    "grr": "grr", "grrr": "grr", "rrr": "grr",
    "haha": "laugh", "haha ": "laugh", "hahaha": "laugh", "heh": "laugh",
    "hehe": "laugh", "hehheh": "laugh", "bwaha": "laugh", "mwaha": "laugh",
    "hyuk": "laugh", "haw": "laugh",
    "chuckle": "chuckle", "hmf": "scoff", "hmph": "scoff", "humph": "scoff",
    "pff": "scoff", "pfft": "scoff", "pssh": "scoff", "tch ": "scoff",
    "hmm": "hmm", "hm": "hmm", "mmh": "hmm", "hmmm": "hmm",
    "huh": "huh", "wha": "huh", "buh": "huh",
    "ahem": "ahem", "ehem": "ahem",
    "cough": "cough", "koff": "cough", "hack": "cough",
    "shh": "shush", "sh": "shush", "shhh": "shush", "hush": "shush",
    "aaah": "scream", "aah": "scream", "aiee": "scream", "aieee": "scream",
    "eeek": "scream", "eek": "scream", "yaaa": "scream", "waaa": "scream",
    "noo": "scream", "nooo": "scream", "aieeee": "scream",
    "nnh": "whimper", "nng": "whimper", "mmf": "whimper", "hnn": "whimper",
    "oof": "oof", "oaf": "oof", "ooof": "oof", "urk": "oof", "gurk": "oof",
    "whew": "whew", "phew": "whew", "pew": "whew",
    "mmm": "mmm", "mm": "mmm", "yum": "mmm", "mmmm": "mmm",
    "psst": "psst", "pst": "psst",
    "boo": "boo",
    "yawn": "yawn", "haaw": "yawn",
    "sniff": "sniff", "snf": "sniff", "snif": "sniff",
    "sob": "sob", "waah": "sob", "boohoo": "sob",
    "gulp": "gulp", "glp": "gulp",
    "pant": "pant", "huff": "pant", "puff": "pant", "wheeze": "pant",
}

# collapse a run of 3+ identical chars down to 2 ("aaaah" -> "aah")
_RUN = re.compile(r"(.)\1{2,}", re.IGNORECASE)
# strip surrounding markup/whitespace/punct but remember the punctuation
_EDGE_PUNCT = re.compile(r"^[\s*_~\"'`(<\[-]+|[\s*_~\"'`)>\]-]+$")


def _max_run_len(s: str) -> int:
    best = 1
    run = 1
    for a, b in zip(s, s[1:], strict=False):
        if a.lower() == b.lower():
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def _looks_like_sentence(s: str) -> bool:
    words = s.split()
    if len(words) >= 4:
        return True
    # 2-3 words with a real vowel-containing word that isn't in our alias table
    real_words = [w for w in words if re.search(r"[aeiou]{1,2}", w) and len(w) >= 4]
    return len(real_words) >= 2


def _looks_like_vocalization_shape(token: str) -> bool:
    """A single blurt: short, no digits, not obviously a dictionary word."""
    if not token or " " in token:
        return False
    if any(ch.isdigit() for ch in token):
        return False
    if len(token) > 12:
        return False
    letters = [c for c in token if c.isalpha()]
    if not letters:
        return False
    vowels = sum(c in "aeiou" for c in (c.lower() for c in letters))
    # mostly-consonant blurts (grr, pfft, hmph) or vowel screams (aaah)
    return vowels <= 2 or vowels / len(letters) >= 0.7


def normalize_vocalization(surface: str | None) -> Vocalization | None:
    """Return a :class:`Vocalization` for ``surface``, or ``None`` if it is not
    a vocalization at all (e.g. an ordinary sentence)."""
    if surface is None:
        return None
    raw = surface.strip()
    if not raw:
        return None

    # intensity signal from the *original* (before we de-stretch)
    bang = raw.count("!") + raw.count("?")
    stretch = _max_run_len(re.sub(r"[^A-Za-z]", "", raw))
    intensity = min(1.0, 0.15 * bang + max(0.0, (stretch - 1)) / 6.0)
    if raw.isupper() and len(raw) >= 3:
        intensity = min(1.0, intensity + 0.15)

    # normalise: drop edge markup, lowercase, collapse long runs, drop inner punct
    core = _EDGE_PUNCT.sub("", raw)
    core = _EDGE_PUNCT.sub("", core)  # once more for "**sigh**" style
    core = core.lower().strip()
    core_nospace = re.sub(r"[^a-z]", "", core)
    destretched = _RUN.sub(r"\1\1", core_nospace)

    if not destretched:
        return None

    def _mk(canonical: str, *, spoken_override: str | None = None) -> Vocalization:
        emotion, tag, spoken, narration = _LEXICON[canonical]
        return Vocalization(raw, canonical, emotion, tag,
                            spoken if spoken_override is None else spoken_override,
                            narration, round(intensity, 3))

    # 1a. laughter family: (bw|mw|w)? a? (ha|hah|heh|hyuk)+
    if re.fullmatch(r"(?:bw|mw|w)?a?(?:ha|hah|heh|huk|hyuk|hah){2,}h?", core_nospace):
        return _mk("laugh")

    # 1b. the open-vowel scream family: a+ h* / e+ k+ / etc.
    if re.fullmatch(r"a{2,}h*|a+h{2,}|e{3,}k*|y?a{3,}", core_nospace):
        return _mk("scream")

    # 1c. exact alias hit (try de-stretched, then a single-char-run variant)
    for key in (destretched, _RUN.sub(r"\1", core_nospace)):
        canonical = _ALIASES.get(key)
        if canonical:
            return _mk(canonical)

    # 2. sentence? then it's not a vocalization
    if _looks_like_sentence(core):
        return None

    # 3. recognised *shape* but unknown token -> pass through to the engine
    if _looks_like_vocalization_shape(destretched):
        return Vocalization(raw, "", None, None, core.strip() or raw,
                            "made a sound", round(intensity, 3))

    return None
