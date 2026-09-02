"""Tests for the deterministic pipeline phases (parse, resolve, assemble)."""

from __future__ import annotations

import json

from pipeline import assemble, resolve, transcribe
from pipeline.comicdb import Block, ComicDB, Entity, Page, Panel, panel_id

# --- transcribe.parse -----------------------------------------------------

def test_parse_scene_then_lines():
    text = ("A hooded figure kneels in the rain among gravestones.\n"
            "CAPTION: SPACE SECTOR 2814. GOTHAM CITY.\n"
            "SPEAKER: I hear you out there in deep space.\n"
            "SPEAKER: KKK.")
    scene, lines = transcribe.parse(text)
    assert scene.startswith("A hooded figure kneels")
    assert lines == [
        ("CAPTION", "SPACE SECTOR 2814. GOTHAM CITY."),
        ("SPEAKER", "I hear you out there in deep space."),
        ("SPEAKER", "KKK."),
    ]


def test_parse_strips_leaked_reasoning_prefix():
    scene, lines = transcribe.parse("Description: a dark room.\nCAPTION: LATER.")
    assert scene == "a dark room."


def test_to_blocks_kinds(tmp_path):
    db = ComicDB.new(tmp_path)
    db.add_panel(Panel(id="p000_00", page=0, index=0, image="x"))
    blocks = transcribe.to_blocks(db, "p000_00", [
        ("CAPTION", "MEANWHILE"), ("SPEAKER", "Hello there friend"), ("SPEAKER", "KRAKKA")])
    assert [b.kind for b in blocks] == ["CAPTION", "DIALOGUE", "SFX"]


# --- resolve ------------------------------------------------------------

def _mini(tmp_path):
    db = ComicDB.new(tmp_path)
    db.add_page(Page(index=3, image="x"))
    db.add_panel(Panel(id="p003_00", page=3, index=0, image="x"))
    db.set_entities([Entity(id="e1"), Entity(id="e2")])
    return db


def test_resolve_self_id_binds(tmp_path):
    db = _mini(tmp_path)
    b = Block(id="b1", panel="p003_00", order=0, kind="DIALOGUE",
              text_raw="I am William Hand, and I have come home.", entity="e1")
    db.replace_blocks_for_panel("p003_00", [b])
    ev = resolve.collect(db)
    db.set_name_evidence(ev)
    resolve.bind(db, ev)
    assert db.entity("e1").name == "William Hand"


def test_resolve_single_vocative_is_not_enough(tmp_path):
    db = _mini(tmp_path)
    b = Block(id="b1", panel="p003_00", order=0, kind="DIALOGUE",
              text_raw="Get down, Mera.", entity="e1")
    db.replace_blocks_for_panel("p003_00", [b])
    ev = resolve.collect(db)
    resolve.bind(db, ev)
    # one weak vocative, no second reference -> stays unbound
    assert db.entity("e2").name is None


def test_resolve_two_vocatives_bind(tmp_path):
    db = _mini(tmp_path)
    db.add_panel(Panel(id="p003_01", page=3, index=1, image="x"))
    db.replace_blocks_for_panel("p003_00", [
        Block(id="b1", panel="p003_00", order=0, kind="DIALOGUE",
              text_raw="Run, Barry.", entity="e1")])
    db.replace_blocks_for_panel("p003_01", [
        Block(id="b2", panel="p003_01", order=0, kind="DIALOGUE",
              text_raw="You always were fast, Barry.", entity="e1")])
    ev = resolve.collect(db)
    resolve.bind(db, ev)
    assert db.entity("e2").name == "Barry"


def test_resolve_override_wins(tmp_path):
    db = _mini(tmp_path)
    db.overrides["e1"] = "Black Hand"
    resolve.bind(db, [])
    assert db.entity("e1").name == "Black Hand"
    assert db.entity("e1").name_confidence == 1.0


def test_resolve_stopword_name_rejected(tmp_path):
    db = _mini(tmp_path)
    db.replace_blocks_for_panel("p003_00", [
        Block(id="b1", panel="p003_00", order=0, kind="DIALOGUE",
              text_raw="I'm Death.", entity="e1")])
    ev = resolve.collect(db)
    resolve.bind(db, ev)
    assert db.entity("e1").name is None


# --- assemble ---------------------------------------------------------

def _assemble_db(tmp_path):
    db = ComicDB.new(tmp_path)
    db.add_page(Page(index=0, image="x", is_front_matter=True))
    db.add_page(Page(index=3, image="x"))
    db.add_panel(Panel(id="p000_00", page=0, index=0, image="x", scene="cover art"))
    db.add_panel(Panel(id=panel_id(3, 0), page=3, index=0, image="x",
                       scene="Rain lashed the graves as a hooded figure knelt."))
    db.set_entities([Entity(id="e1", name="William")])
    db.replace_blocks_for_panel("p003_00", [
        Block(id="b1", panel="p003_00", order=0, kind="CAPTION",
              text_raw="SPACE SECTOR 2814."),
        Block(id="b2", panel="p003_00", order=1, kind="DIALOGUE",
              text_raw="I hear you out there.", entity="e1"),
        Block(id="b3", panel="p003_00", order=2, kind="SFX", text_raw="KKK"),
    ])
    return db


def test_assemble_skips_front_matter_and_uses_verbatim(tmp_path):
    _assemble_db(tmp_path).save()
    assemble.main.__wrapped__ if hasattr(assemble.main, "__wrapped__") else None
    import sys
    sys.argv = ["assemble", str(tmp_path)]
    assemble.main()
    nar = json.loads((tmp_path / "narrative.json").read_text())
    assert "0" not in nar  # front matter skipped
    segs = nar["3"]
    assert segs[0] == {"speaker": "NARRATOR", "text": "Rain lashed the graves as a hooded figure knelt."}
    assert segs[1] == {"speaker": "NARRATOR", "text": "SPACE SECTOR 2814."}
    assert segs[2] == {"speaker": "William", "text": "I hear you out there."}
    # ambient SFX "KKK" dropped
    assert all("KKK" not in s["text"] for s in segs)


def test_assemble_unknown_speaker_is_reported_speech(tmp_path):
    db = _assemble_db(tmp_path)
    db.replace_blocks_for_panel("p003_00", [
        Block(id="b9", panel="p003_00", order=0, kind="DIALOGUE",
              text_raw="Who goes there?", entity=None)])
    db.save()
    import sys
    sys.argv = ["assemble", str(tmp_path)]
    assemble.main()
    nar = json.loads((tmp_path / "narrative.json").read_text())
    assert nar["3"][-1]["speaker"] == "NARRATOR"
    assert 'A voice says, "Who goes there?"' in nar["3"][-1]["text"]


# --- regression: bugs the full-issue run exposed ------------------------

def test_strip_think_unclosed_keeps_the_answer():
    from pipeline.vision import strip_think
    raw = ("<think> Got it, let's tackle this. The user wants a description...\n"
           "the panel has a skull.\n"
           "CAPTION: SOME THINGS ARE WORSE THAN DEATH\n"
           "SPEAKER: I hear you.")
    out = strip_think(raw)
    assert "<think>" not in out and "let's tackle" not in out
    assert out.startswith("CAPTION:")


def test_parse_blanks_a_reasoning_scene():
    scene, lines = transcribe.parse(
        "Got it, let's break this down. The user wants...\nCAPTION: LATER.")
    assert scene == "" and lines == [("CAPTION", "LATER.")]


def test_resolve_rejects_allcaps_common_word_self_id(tmp_path):
    db = _mini(tmp_path)
    for i, txt in enumerate(["I AM HUNGRY.", "I'M SORRY--", "I AM TRYING--"]):
        db.add_panel(Panel(id=f"p003_{i:02d}", page=3, index=i, image="x"))
        db.replace_blocks_for_panel(f"p003_{i:02d}", [
            Block(id=f"b{i}", panel=f"p003_{i:02d}", order=0, kind="DIALOGUE",
                  text_raw=txt, entity="e1")])
    ev = resolve.collect(db)
    resolve.bind(db, ev)
    assert db.entity("e1").name is None


def test_resolve_full_name_self_id_still_binds(tmp_path):
    db = _mini(tmp_path)
    db.replace_blocks_for_panel("p003_00", [
        Block(id="b1", panel="p003_00", order=0, kind="DIALOGUE",
              text_raw="MY FATHER SAID, EVERYONE DIES. I AM WILLIAM HAND.", entity="e1")])
    ev = resolve.collect(db)
    resolve.bind(db, ev)
    assert db.entity("e1").name == "William Hand"
