"""Tests for pipeline.comicdb -- the accumulating per-issue record."""

from __future__ import annotations

from pipeline.comicdb import (
    Block,
    ComicDB,
    Entity,
    NameEvidence,
    Observation,
    Page,
    Panel,
    Vision,
    panel_id,
)


def _seed(tmp_path):
    db = ComicDB.new(tmp_path, series="Blackest Night", number=1, source="bn01.cbz")
    db.add_page(Page(index=0, image="x/000.jpg", w=100, h=150, is_front_matter=True))
    db.add_page(Page(index=3, image="x/003.jpg", w=100, h=150))
    for i in range(2):
        pid = panel_id(3, i)
        db.add_panel(Panel(id=pid, page=3, index=i, image=f"panels/{pid}.jpg", bbox=[0, 0, 10, 10]))
    return db


def test_new_load_roundtrip(tmp_path):
    _seed(tmp_path).save()
    db = ComicDB.load(tmp_path)
    assert db.issue["series"] == "Blackest Night"
    assert [p.index for p in db.pages()] == [0, 3]
    assert len(db.panels()) == 2
    assert db.pages()[0].is_front_matter is True


def test_panel_ids_stable_and_sorted(tmp_path):
    db = _seed(tmp_path)
    assert panel_id(3, 1) == "p003_01"
    db.add_panel(Panel(id=panel_id(3, 0), page=3, index=0, image="re.jpg"))  # replace
    assert len(db.panels()) == 2
    assert db.panels()[0].image == "re.jpg"


def test_scene_and_vision_cached(tmp_path):
    db = _seed(tmp_path)
    v = Vision(model="qwen3-vl:8b", prompt_v=2, raw="RAW RESPONSE", at="2026-01-01")
    db.set_transcribe("p003_00", "A dark room.", v)
    db.save()
    p = ComicDB.load(tmp_path).panel("p003_00")
    assert p.scene == "A dark room."
    assert p.vision.raw == "RAW RESPONSE" and p.vision.prompt_v == 2


def test_redescribe_owns_scene_after_pass2(tmp_path):
    db = _seed(tmp_path)
    db.set_transcribe("p003_00", "pass 1 guess", Vision(raw="r"))
    db.set_redescribe("p003_00", "pass 2 accurate", sig="abc123")
    # a later transcribe re-parse must not clobber the pass-2 scene
    db.set_transcribe("p003_00", "pass 1 again", Vision(raw="r2"))
    p = db.panel("p003_00")
    assert p.scene == "pass 2 accurate" and p.scene_source == "pass2"
    assert p.vision.raw == "r2"


def test_blocks_replace_and_query(tmp_path):
    db = _seed(tmp_path)
    b1 = Block(id=db.next_block_id(), panel="p003_00", order=0, kind="CAPTION",
               text_raw="SOME THINGS ARE WORSE THAN DEATH")
    b2 = Block(id=db.next_block_id(), panel="p003_00", order=1, kind="DIALOGUE",
               text_raw="I hear you")
    db.replace_blocks_for_panel("p003_00", [b1, b2])
    assert [b.id for b in db.blocks(panel="p003_00")] == ["b00001", "b00002"]
    # re-transcribe the same panel -> old blocks gone, not duplicated
    b3 = Block(id=db.next_block_id(), panel="p003_00", order=0, kind="CAPTION", text_raw="new")
    db.replace_blocks_for_panel("p003_00", [b3])
    assert [b.text_raw for b in db.blocks(panel="p003_00")] == ["new"]


def test_entity_linking_and_display(tmp_path):
    db = _seed(tmp_path)
    b = Block(id=db.next_block_id(), panel="p003_01", order=0, kind="DIALOGUE",
              text_raw="Everyone dies, William.")
    db.replace_blocks_for_panel("p003_01", [b])
    e = Entity(id=db.next_entity_id(), appearance="a hooded figure in a studded helmet",
               observations=[Observation(panel="p003_01", bbox=[1, 1, 2, 2])])
    db.set_entities([e])
    db.link_block_entity(b.id, e.id)
    assert db.blocks(entity="e1")[0].text_raw == "Everyone dies, William."
    assert db.entity("e1").display == "a hooded figure in a studded helmet"
    db.bind_name("e1", "William", 0.7)
    assert db.entity("e1").display == "William"


def test_name_evidence_and_binding(tmp_path):
    db = _seed(tmp_path)
    db.set_entities([Entity(id="e1")])
    db.set_name_evidence([
        NameEvidence(entity="e1", panel="p003_01", kind="vocative",
                     quote="Everyone dies, William.", name="William", weight=0.4),
    ])
    assert db.evidence(entity="e1")[0].name == "William"
    db.bind_name("e1", "William", 0.4)
    db.save()
    assert ComicDB.load(tmp_path).entity("e1").name_confidence == 0.4


def test_review_helpers(tmp_path):
    db = _seed(tmp_path)
    db.set_entities([Entity(id="e1"), Entity(id="e2", name="Mera", name_confidence=0.4)])
    for i, eid in enumerate(["e1", "e1", "e1", "e2"]):
        b = Block(id=db.next_block_id(), panel="p003_00", order=i, kind="DIALOGUE",
                  text_raw=f"line {i}", entity=eid)
        db._d["blocks"].append(_as_dict(b))
    assert db.unbound_entities(min_lines=2)[0][0].id == "e1"
    assert [e.id for e in db.low_confidence_names(0.6)] == ["e2"]


def _as_dict(b: Block) -> dict:
    from dataclasses import asdict
    return asdict(b)
