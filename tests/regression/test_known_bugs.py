"""One test per bug we've already been bitten by, so it can't come back.

Each references the code comment that records the original incident. When a
new comic turns up a new bug: fix it, then add a test here (or a corpus row
if it's a classification/attribution miss).
"""

from __future__ import annotations

# --- 02_vision_analyze -----------------------------------------------------

def test_empty_result_is_retried_not_treated_as_done(script, tmp_path):
    """Regression: 6 panels saved as {"text": "", "error": ...} were never
    retried because the key existed in the results dict."""
    results = {
        "page000_panel00": {"text": "a real description"},
        "page000_panel01": {"text": "", "error": "subprocess timeout"},
    }
    # reproduce the 'todo' computation from main()
    manifest = {"pages": [{"page_index": 0, "panels": [
        {"panel_index": 0, "file": "a.jpg"},
        {"panel_index": 1, "file": "b.jpg"},
    ]}]}
    all_panels = [(f"page{p['page_index']:03d}_panel{pn['panel_index']:02d}", pn["file"])
                  for p in manifest["pages"] for pn in p["panels"]]
    todo = [(k, f) for k, f in all_panels if not results.get(k, {}).get("text")]
    assert [k for k, _ in todo] == ["page000_panel01"]


def test_num_ctx_is_bounded_not_the_262k_default(script):
    """Regression: the 262144 default context grew the KV cache and caused an
    18x throughput collapse over a long run."""
    import inspect
    text = inspect.getsource(script("vision"))
    assert '"num_ctx": 16384' in text
    assert '"num_ctx": 262144' not in text
    assert '"num_ctx": 262_144' not in text


# --- 03_narrative --------------------------------------------------------

def test_invented_placeholder_speakers_never_survive(script):
    """Regression: the model emits 'SPEAKER:' / 'VILLAIN:' despite the prompt
    forbidding it."""
    narrative = script("narrative")
    raw = "\n".join(f"{p}: line {i}" for i, p in enumerate(narrative.PLACEHOLDER_SPEAKERS))
    segs = narrative.parse_narrative(raw)
    assert all(s["speaker"] == "NARRATOR" for s in segs)


# --- 04_tts_render -----------------------------------------------------

def test_short_caps_word_not_spelled_as_initialism(script):
    """Regression: 'IT' -> 'I.T.' in espeak-ng."""
    assert script("tts").clean_for_speech("IT") == "it"


def test_classification_failure_is_never_a_silent_male_default(script, tmp_path, monkeypatch, capsys):
    """Regression: a 65-name classify call timed out and every speaker
    silently became MALE-YOUNG, including women."""
    tts = script("tts")
    monkeypatch.setattr(tts, "classify_speakers", lambda names: {})  # total failure
    tts.load_voice_map(tmp_path, {"MERA", "ARTHUR"})
    err = capsys.readouterr().err
    assert "MERA" in err and "ARTHUR" in err
    assert "review" in err.lower()
