"""Characterization tests for ``load_voice_map`` in 04_tts_render.

Key behaviours being pinned:
* a persisted ``voice_map.json`` is authoritative across reruns (stability);
* voices are handed out round-robin within a gender/age pool;
* NARRATOR always gets the fixed narrator voice;
* a name the classifier can't place is reported loudly, never silently
  defaulted (the regression from the code comments).
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def tts(script):
    return script("tts")


@pytest.fixture
def fixed_classifier(tts, monkeypatch):
    """Replace the LLM call with a lookup table."""
    table = {}

    def fake(names):
        return {n.upper(): table.get(n.upper()) for n in names if table.get(n.upper())}

    monkeypatch.setattr(tts, "classify_speakers", fake)
    return table


def test_narrator_gets_narrator_voice(tts, tmp_path, fixed_classifier):
    vm = tts.load_voice_map(tmp_path, {"NARRATOR"})
    assert vm["NARRATOR"] == tts.NARRATOR_VOICE


def test_round_robin_within_pool(tts, tmp_path, fixed_classifier):
    fixed_classifier.update({"A": "MALE-YOUNG", "B": "MALE-YOUNG", "C": "MALE-YOUNG"})
    vm = tts.load_voice_map(tmp_path, {"A", "B", "C"})
    assigned = {vm["A"], vm["B"], vm["C"]}
    assert assigned == set(tts.VOICE_POOLS["MALE-YOUNG"])  # 3 distinct voices


def test_persisted_map_is_authoritative(tts, tmp_path, fixed_classifier):
    (tmp_path / "voice_map.json").write_text(json.dumps({"MERA": "en_US-amy-medium"}))
    fixed_classifier["MERA"] = "MALE-OLD"  # would pick a different voice
    vm = tts.load_voice_map(tmp_path, {"MERA"})
    assert vm["MERA"] == "en_US-amy-medium"


def test_unclassified_name_warns_and_does_not_silently_default(tts, tmp_path, fixed_classifier, capsys):
    # classifier returns nothing for this name
    vm = tts.load_voice_map(tmp_path, {"ZZZ"})
    err = capsys.readouterr().err
    assert "ZZZ" in err and "review" in err.lower()
    # it still gets *a* voice so the run completes, but the warning is the point
    assert vm["ZZZ"] in sum(tts.VOICE_POOLS.values(), [])


def test_map_written_to_disk(tts, tmp_path, fixed_classifier):
    fixed_classifier["MERA"] = "FEMALE-YOUNG"
    tts.load_voice_map(tmp_path, {"MERA"})
    on_disk = json.loads((tmp_path / "voice_map.json").read_text())
    assert on_disk["MERA"] in tts.VOICE_POOLS["FEMALE-YOUNG"]
