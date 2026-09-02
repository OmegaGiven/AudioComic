"""Loader for the regression corpus.

Each comic that surfaces a bug gets a directory under ``tests/corpus/``:

    tests/corpus/<slug>/
        case.md          -- human notes: what this comic broke, why it's here
        elements.jsonl    -- one JSON object per line, a labelled text element

``elements.jsonl`` record schema (all fields optional except ``text`` and
``expect_kind``)::

    {
      "panel": "page012_panel03",
      "text": "tsk",
      "raw_label": "SFX",            # what the vision model called it
      "in_bubble": true,
      "lettering": "normal",         # "normal" | "display"
      "area_ratio": 0.02,
      "expect_kind": "VOCALIZATION", # DIALOGUE | CAPTION | VOCALIZATION | SFX
      "characters": [                # optional, for attribution checks
        {"name": "MERA", "bbox": [10,10,30,60], "mouth_open": true}
      ],
      "bubbles": [{"id": "b1", "owner": "MERA", "bbox": [0,0,40,40]}],
      "bubble_id": "b1",
      "bbox": [12, 8, 20, 10],
      "expect_speaker": "MERA"       # optional; null = should be unattributable
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent


@dataclass
class CorpusElement:
    slug: str
    text: str
    expect_kind: str
    panel: str = ""
    raw_label: str = "UNKNOWN"
    in_bubble: bool = False
    lettering: str = "normal"
    area_ratio: float = 0.0
    bbox: list[float] | None = None
    bubble_id: str | None = None
    panel_diagonal: float | None = None
    characters: list[dict] = field(default_factory=list)
    bubbles: list[dict] = field(default_factory=list)
    expect_speaker: str | None = None
    has_speaker_expectation: bool = False

    @property
    def id(self) -> str:
        return f"{self.slug}:{self.panel or '?'}:{self.text[:20]}"


def load_corpus() -> list[CorpusElement]:
    out: list[CorpusElement] = []
    for case_dir in sorted(p for p in CORPUS_DIR.iterdir() if p.is_dir()):
        jsonl = case_dir / "elements.jsonl"
        if not jsonl.exists():
            continue
        for line in jsonl.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            d = json.loads(line)
            out.append(CorpusElement(
                slug=case_dir.name,
                text=d["text"],
                expect_kind=d["expect_kind"],
                panel=d.get("panel", ""),
                raw_label=d.get("raw_label", "UNKNOWN"),
                in_bubble=d.get("in_bubble", False),
                lettering=d.get("lettering", "normal"),
                area_ratio=d.get("area_ratio", 0.0),
                bbox=d.get("bbox"),
                bubble_id=d.get("bubble_id"),
                panel_diagonal=d.get("panel_diagonal"),
                characters=d.get("characters", []),
                bubbles=d.get("bubbles", []),
                expect_speaker=d.get("expect_speaker"),
                has_speaker_expectation="expect_speaker" in d,
            ))
    return out
