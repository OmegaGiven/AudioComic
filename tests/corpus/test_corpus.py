"""Run the deterministic classifiers/attribution against the labelled corpus.

This is the file that grows. Every time a new comic reveals a
mis-classification or a mis-attribution, add a line to that comic's
``elements.jsonl`` and this suite covers it forever.

Two modes:
* per-element assertions (parametrized, so failures name the exact panel);
* an aggregate accuracy floor, so a broad regression that nudges many
  elements at once still trips even if no single case is "required".
"""

from __future__ import annotations

import pytest

from panelspeak.attribution import attribute_vocalization
from panelspeak.classify import refine_kind
from panelspeak.text_elements import BBox, Bubble, Character, ElementKind, TextElement
from tests.corpus._schema import load_corpus

CORPUS = load_corpus()

# raise these as the corpus matures; they exist so a big regression fails CI
KIND_ACCURACY_FLOOR = 0.80
SPEAKER_ACCURACY_FLOOR = 0.75


def _predict_kind(el):
    return refine_kind(el.raw_label, el.text, in_bubble=el.in_bubble,
                       lettering=el.lettering, area_ratio=el.area_ratio)


def _predict_speaker(el):
    bbox = BBox(*el.bbox) if el.bbox else None
    element = TextElement(text=el.text, bbox=bbox, bubble_id=el.bubble_id)
    chars = [Character(c["name"],
                       BBox(*c["bbox"]) if c.get("bbox") else None,
                       c.get("mouth_open", False)) for c in el.characters]
    bubbles = [Bubble(b["id"], b.get("owner"),
                      BBox(*b["bbox"]) if b.get("bbox") else None) for b in el.bubbles]
    return attribute_vocalization(element, chars, bubbles,
                                  panel_diagonal=el.panel_diagonal)


@pytest.mark.skipif(not CORPUS, reason="corpus is empty")
@pytest.mark.parametrize("el", CORPUS, ids=[e.id for e in CORPUS])
def test_element_kind(el):
    predicted = _predict_kind(el)
    assert predicted == ElementKind(el.expect_kind), (
        f"{el.id}: got {predicted}, expected {el.expect_kind}"
    )


@pytest.mark.skipif(not CORPUS, reason="corpus is empty")
@pytest.mark.parametrize(
    "el",
    [e for e in CORPUS if e.has_speaker_expectation],
    ids=[e.id for e in CORPUS if e.has_speaker_expectation] or ["none"],
)
def test_element_speaker(el):
    assert _predict_speaker(el) == el.expect_speaker


@pytest.mark.skipif(len(CORPUS) < 5, reason="not enough corpus for an aggregate floor")
def test_kind_accuracy_floor():
    correct = sum(_predict_kind(e) == ElementKind(e.expect_kind) for e in CORPUS)
    acc = correct / len(CORPUS)
    assert acc >= KIND_ACCURACY_FLOOR, f"kind accuracy {acc:.2%} < {KIND_ACCURACY_FLOOR:.0%}"


@pytest.mark.skipif(
    len([e for e in CORPUS if e.has_speaker_expectation]) < 5,
    reason="not enough speaker-labelled corpus for an aggregate floor",
)
def test_speaker_accuracy_floor():
    labelled = [e for e in CORPUS if e.has_speaker_expectation]
    correct = sum(_predict_speaker(e) == e.expect_speaker for e in labelled)
    acc = correct / len(labelled)
    assert acc >= SPEAKER_ACCURACY_FLOOR, f"speaker accuracy {acc:.2%} < {SPEAKER_ACCURACY_FLOOR:.0%}"
