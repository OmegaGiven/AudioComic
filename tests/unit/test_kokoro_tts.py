"""Tests for the Kokoro stage-4 variant (04_tts_render_kokoro.py).

Runs alongside the Piper stage-4 tests; the two scripts are kept in parallel
so output can be compared.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def k(script):
    return script("tts_kokoro")


# --- clean_for_speech: no lowercasing (that was a Piper/espeak workaround) ---

def test_keeps_case(k):
    assert k.clean_for_speech("IT is DONE") == "IT is DONE"


def test_strips_urls_and_collapses_whitespace(k):
    assert k.clean_for_speech("see  https://example.com/x   now") == "see now"


def test_empty(k):
    assert k.clean_for_speech("   ") == ""


# --- voice map: Kokoro voice ids, same stability guarantees ----------------

@pytest.fixture
def fixed_classifier(k, monkeypatch):
    table: dict[str, str] = {}
    monkeypatch.setattr(k, "classify_speakers",
                        lambda names: {n.upper(): table[n.upper()]
                                       for n in names if n.upper() in table})
    return table


def test_narrator_voice_is_kokoro(k, tmp_path, fixed_classifier):
    vm = k.load_voice_map(tmp_path, {"NARRATOR"})
    assert vm["NARRATOR"] == k.NARRATOR_VOICE == "af_heart"


def test_round_robin_within_pool(k, tmp_path, fixed_classifier):
    fixed_classifier.update({"A": "FEMALE-YOUNG", "B": "FEMALE-YOUNG", "C": "FEMALE-YOUNG"})
    vm = k.load_voice_map(tmp_path, {"A", "B", "C"})
    assert len({vm["A"], vm["B"], vm["C"]}) == 3
    assert all(v in k.VOICE_POOLS["FEMALE-YOUNG"] for v in (vm["A"], vm["B"], vm["C"]))


def test_persisted_map_wins(k, tmp_path, fixed_classifier):
    (tmp_path / "voice_map.json").write_text(json.dumps({"MERA": "af_bella"}))
    fixed_classifier["MERA"] = "MALE-OLD"
    assert k.load_voice_map(tmp_path, {"MERA"})["MERA"] == "af_bella"


def test_unclassified_warns_not_silent(k, tmp_path, fixed_classifier, capsys):
    k.load_voice_map(tmp_path, {"ZZZ"})
    assert "ZZZ" in capsys.readouterr().err


# --- prepare_segments: onomatopoeia cleaned via panelspeak ----------------

def test_prepare_segments_narrates_bare_scream(k):
    narrative = {"0": [
        {"speaker": "NARRATOR", "text": "The ground split open."},
        {"speaker": "HAL JORDAN", "text": "Aaaah!"},
    ]}
    segs = k.prepare_segments(narrative)
    assert [(s.speaker, s.text) for s in segs] == [
        ("NARRATOR", "The ground split open."),
        ("NARRATOR", "Hal Jordan screamed."),
    ]


def test_prepare_segments_lifts_breathy_lead(k):
    narrative = {"0": [{"speaker": "MERA", "text": "*sigh* It's never just one."}]}
    segs = k.prepare_segments(narrative)
    assert [(s.speaker, s.text) for s in segs] == [
        ("NARRATOR", "Mera sighed."),
        ("MERA", "It's never just one."),
    ]


def test_prepare_segments_keeps_clean_interjection(k):
    narrative = {"0": [{"speaker": "BLACK HAND", "text": "Tsk. You should be honored."}]}
    segs = k.prepare_segments(narrative)
    assert len(segs) == 1 and segs[0].text == "Tsk. You should be honored."


def test_prepare_segments_orders_pages_numerically(k):
    narrative = {"10": [{"speaker": "NARRATOR", "text": "ten"}],
                 "2": [{"speaker": "NARRATOR", "text": "two"}]}
    assert [s.text for s in k.prepare_segments(narrative)] == ["two", "ten"]
