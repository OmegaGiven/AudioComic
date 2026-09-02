"""Characterization tests for ``classify_pages`` -- double-page-spread detection
by aspect ratio vs. the modal single-page ratio.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PIL")


def test_single_pages_are_not_spreads(script, make_image):
    pages = [make_image(660, 1000) for _ in range(4)]
    out = script("extract").classify_pages(pages)
    assert [p["is_spread"] for p in out] == [False] * 4


def test_double_width_page_flagged_as_spread(script, make_image):
    pages = [make_image(660, 1000) for _ in range(4)]
    pages.append(make_image(1320, 1000))  # ~2x the modal ratio
    out = script("extract").classify_pages(pages)
    assert [p["is_spread"] for p in out] == [False, False, False, False, True]


def test_modal_ratio_wins_even_if_spreads_are_common(script, make_image):
    # 3 singles, 2 spreads: modal ratio is still the single-page one
    pages = [make_image(660, 1000) for _ in range(3)]
    pages += [make_image(1320, 1000) for _ in range(2)]
    out = script("extract").classify_pages(pages)
    assert sum(p["is_spread"] for p in out) == 2


def test_reports_dimensions_and_ratio(script, make_image):
    out = script("extract").classify_pages([make_image(600, 900)])
    assert out[0]["width"] == 600 and out[0]["height"] == 900
    assert out[0]["ratio"] == pytest.approx(600 / 900)
