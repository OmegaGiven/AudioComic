"""Tests for panelspeak.classify.refine_kind -- correcting the vision model's
element labels using geometry + the lexicon.
"""

from __future__ import annotations

from panelspeak.classify import refine_kind
from panelspeak.text_elements import ElementKind as K


def test_big_display_sfx_outside_bubble_is_sfx_even_if_labeled_dialogue():
    assert refine_kind("DIALOGUE", "KRAKKA-THOOM", in_bubble=False,
                       lettering="display", area_ratio=0.3) is K.SFX


def test_large_area_alone_is_enough_for_sfx():
    assert refine_kind("UNKNOWN", "WHUMP", in_bubble=False, area_ratio=0.2) is K.SFX


def test_bubbled_interjection_is_vocalization_not_sfx():
    assert refine_kind("SFX", "tsk", in_bubble=True) is K.VOCALIZATION


def test_bubbled_sentence_is_dialogue():
    assert refine_kind("DIALOGUE", "You have no idea what's coming.",
                       in_bubble=True) is K.DIALOGUE


def test_bubbled_single_unknown_word_treated_as_vocalization():
    assert refine_kind("DIALOGUE", "hrrk", in_bubble=True) is K.VOCALIZATION


def test_caption_box_stays_caption():
    assert refine_kind("CAPTION", "Meanwhile, in Gotham...",
                       in_bubble=False) is K.CAPTION


def test_freefloating_known_noise_is_vocalization_pending_attribution():
    # a lettered "ugh" near a character, normal lettering, not huge
    assert refine_kind("SFX", "ugh", in_bubble=False, lettering="normal",
                       area_ratio=0.03) is K.VOCALIZATION


def test_allcaps_consonant_blurt_freefloating_is_sfx():
    assert refine_kind("UNKNOWN", "THWIP", in_bubble=False) is K.SFX
    assert refine_kind("UNKNOWN", "SKREEE", in_bubble=False) is K.SFX


def test_common_word_in_caps_is_not_sfx():
    # "NO!" shouted is dialogue-ish, never a sound effect
    assert refine_kind("DIALOGUE", "NO", in_bubble=True) is not K.SFX


def test_unparseable_raw_label_does_not_crash():
    assert refine_kind("gobbledegook", "BOOM", in_bubble=False,
                       lettering="display") is K.SFX
