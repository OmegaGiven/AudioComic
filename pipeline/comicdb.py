"""ComicDB -- the accumulating per-issue record.

`comic.json` in the work dir. Held fully in memory (an issue is well under a
megabyte). Every phase reads it, mutates it, saves it. Nothing is final until
`resolve` runs; `assemble` is a pure function of the finished DB.

    db = ComicDB.load(work_dir)          # or ComicDB.new(work_dir, issue=...)
    db.add_panel(page=3, index=0, bbox=..., image=...)
    db.save()

IDs are stable strings so cross-phase links survive re-runs:
    panel   p<page:03d>_<index:02d>          p003_00
    block   b<seq:05d>                       b00042
    entity  e<seq>                           e2
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DB_NAME = "comic.json"
SCHEMA_VERSION = 1


@dataclass
class Vision:
    model: str = ""
    prompt_v: int = 0
    raw: str = ""
    at: str = ""


@dataclass
class Page:
    index: int
    image: str
    w: int = 0
    h: int = 0
    is_front_matter: bool = False


@dataclass
class Panel:
    id: str
    page: int
    index: int
    image: str
    bbox: list[float] = field(default_factory=list)
    scene: str = ""
    #: which phase owns .scene ("pass1" transcribe, "pass2" redescribe) and a
    #: signature of the redescribe context, so redescribe only re-runs panels
    #: whose identities changed.
    scene_source: str = "pass1"
    scene_sig: str = ""
    #: the transcribe vision call -- raw response cached for re-parse
    vision: Vision = field(default_factory=Vision)


@dataclass
class Block:
    id: str
    panel: str
    order: int
    kind: str            # CAPTION | DIALOGUE | SFX
    text_raw: str
    text_clean: str = ""
    entity: str | None = None       # set by identify
    speaker_raw: str | None = None  # a name the vision model printed, if any
    essential: bool = True


@dataclass
class Observation:
    panel: str
    bbox: list[float] = field(default_factory=list)


@dataclass
class Entity:
    id: str
    appearance: str = ""
    name: str | None = None
    name_confidence: float = 0.0
    voice: str | None = None
    observations: list[Observation] = field(default_factory=list)

    @property
    def display(self) -> str:
        """Name if bound, else a short appearance label, else the id."""
        if self.name:
            return self.name
        if self.appearance:
            return self.appearance
        return self.id


@dataclass
class NameEvidence:
    entity: str
    panel: str
    kind: str            # self_id | narration | vocative | printed
    quote: str
    name: str
    weight: float


class ComicDB:
    def __init__(self, path: Path, data: dict[str, Any]):
        self.path = path
        self._d = data

    # -- lifecycle --------------------------------------------------------
    @classmethod
    def new(cls, work_dir: str | Path, *, series: str = "", number: int | None = None,
            source: str = "") -> ComicDB:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "schema": SCHEMA_VERSION,
            "issue": {"series": series, "number": number, "source": source},
            "pages": [], "panels": [], "blocks": [], "entities": [],
            "name_evidence": [], "overrides": {}, "_seq": {"block": 0, "entity": 0},
        }
        return cls(work_dir / DB_NAME, data)

    @classmethod
    def load(cls, work_dir: str | Path) -> ComicDB:
        path = Path(work_dir) / DB_NAME
        return cls(path, json.loads(path.read_text()))

    @classmethod
    def load_or_new(cls, work_dir: str | Path, **kw) -> ComicDB:
        path = Path(work_dir) / DB_NAME
        return cls.load(work_dir) if path.exists() else cls.new(work_dir, **kw)

    def save(self) -> None:
        self.path.write_text(json.dumps(self._d, indent=2))

    # -- typed accessors (views; mutate via helpers below) --------------
    @property
    def issue(self) -> dict:
        return self._d["issue"]

    @property
    def overrides(self) -> dict:
        return self._d["overrides"]

    def pages(self) -> list[Page]:
        return [Page(**p) for p in self._d["pages"]]

    def panels(self) -> list[Panel]:
        return [_panel(p) for p in self._d["panels"]]

    def panel(self, panel_id: str) -> Panel | None:
        raw = next((p for p in self._d["panels"] if p["id"] == panel_id), None)
        return _panel(raw) if raw else None

    def blocks(self, *, panel: str | None = None, entity: str | None = None) -> list[Block]:
        out = [Block(**b) for b in self._d["blocks"]]
        if panel is not None:
            out = [b for b in out if b.panel == panel]
        if entity is not None:
            out = [b for b in out if b.entity == entity]
        return out

    def entities(self) -> list[Entity]:
        return [_entity(e) for e in self._d["entities"]]

    def entity(self, entity_id: str) -> Entity | None:
        raw = next((e for e in self._d["entities"] if e["id"] == entity_id), None)
        return _entity(raw) if raw else None

    def evidence(self, *, entity: str | None = None) -> list[NameEvidence]:
        out = [NameEvidence(**e) for e in self._d["name_evidence"]]
        return [e for e in out if e.entity == entity] if entity else out

    # -- mutators -------------------------------------------------------
    def add_page(self, page: Page) -> None:
        self._d["pages"] = [p for p in self._d["pages"] if p["index"] != page.index]
        self._d["pages"].append(asdict(page))
        self._d["pages"].sort(key=lambda p: p["index"])

    def add_panel(self, panel: Panel) -> None:
        self._d["panels"] = [p for p in self._d["panels"] if p["id"] != panel.id]
        self._d["panels"].append(_panel_dict(panel))
        self._d["panels"].sort(key=lambda p: (p["page"], p["index"]))

    def set_transcribe(self, panel_id: str, scene: str, vision: Vision) -> None:
        """Store the Pass-1 vision. Only sets .scene if Pass 2 hasn't claimed it."""
        for p in self._d["panels"]:
            if p["id"] == panel_id:
                p["vision"] = asdict(vision)
                if p.get("scene_source", "pass1") == "pass1":
                    p["scene"] = scene
                return

    def set_redescribe(self, panel_id: str, scene: str, sig: str) -> None:
        for p in self._d["panels"]:
            if p["id"] == panel_id:
                p["scene"] = scene
                p["scene_source"] = "pass2"
                p["scene_sig"] = sig
                return

    def replace_blocks_for_panel(self, panel_id: str, blocks: list[Block]) -> None:
        self._d["blocks"] = [b for b in self._d["blocks"] if b["panel"] != panel_id]
        for b in blocks:
            self._d["blocks"].append(asdict(b))
        self._d["blocks"].sort(key=lambda b: (_page_of(self, b["panel"]),
                                              _idx_of(self, b["panel"]), b["order"]))

    def next_block_id(self) -> str:
        self._d["_seq"]["block"] += 1
        return f"b{self._d['_seq']['block']:05d}"

    def next_entity_id(self) -> str:
        self._d["_seq"]["entity"] += 1
        return f"e{self._d['_seq']['entity']}"

    def set_entities(self, entities: list[Entity]) -> None:
        self._d["entities"] = [_entity_dict(e) for e in entities]

    def link_block_entity(self, block_id: str, entity_id: str | None) -> None:
        for b in self._d["blocks"]:
            if b["id"] == block_id:
                b["entity"] = entity_id
                return

    def set_name_evidence(self, evidence: list[NameEvidence]) -> None:
        self._d["name_evidence"] = [asdict(e) for e in evidence]

    def bind_name(self, entity_id: str, name: str | None, confidence: float) -> None:
        for e in self._d["entities"]:
            if e["id"] == entity_id:
                e["name"] = name
                e["name_confidence"] = round(confidence, 3)
                return

    def set_voice(self, entity_id: str, voice: str) -> None:
        for e in self._d["entities"]:
            if e["id"] == entity_id:
                e["voice"] = voice
                return

    # -- review helpers ------------------------------------------------
    def unbound_entities(self, min_lines: int = 1) -> list[tuple[Entity, int]]:
        out = []
        for e in self.entities():
            if e.name:
                continue
            n = len(self.blocks(entity=e.id))
            if n >= min_lines:
                out.append((e, n))
        return sorted(out, key=lambda t: -t[1])

    def low_confidence_names(self, threshold: float = 0.6) -> list[Entity]:
        return [e for e in self.entities()
                if e.name and e.name_confidence < threshold]


# -- (de)serialisation helpers -----------------------------------------
def _panel(raw: dict) -> Panel:
    raw = dict(raw)
    raw["vision"] = Vision(**raw.get("vision", {}))
    return Panel(**raw)


def _panel_dict(p: Panel) -> dict:
    d = asdict(p)
    return d


def _entity(raw: dict) -> Entity:
    raw = dict(raw)
    raw["observations"] = [Observation(**o) for o in raw.get("observations", [])]
    return Entity(**raw)


def _entity_dict(e: Entity) -> dict:
    return asdict(e)


def _page_of(db: ComicDB, panel_id: str) -> int:
    p = db.panel(panel_id)
    return p.page if p else 0


def _idx_of(db: ComicDB, panel_id: str) -> int:
    p = db.panel(panel_id)
    return p.index if p else 0


def panel_id(page: int, index: int) -> str:
    return f"p{page:03d}_{index:02d}"
