"""Characterization tests for ``find_page_files`` in 01_extract_and_segment.

Locks in the current junk-filtering behaviour: real pages match the comic's
``-NNN`` numbering scheme, everything else (scanner-group credit images) is
skipped.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def extracted(tmp_path):
    d = tmp_path / "extracted" / "Some Comic 01"
    d.mkdir(parents=True)
    return d


def _touch(d, *names):
    for n in names:
        (d / n).write_bytes(b"\xff\xd8\xff\xe0stub")  # tiny jpeg-ish stub


def test_keeps_numbered_pages_drops_credit_image(script, extracted):
    _touch(extracted,
           "Some Comic 01-000.jpg", "Some Comic 01-001.jpg",
           "Some Comic 01-002.jpg", "Kingpin8.jpg")
    numbered, skipped = script("extract").find_page_files(extracted)

    assert [p.name for p in numbered] == [
        "Some Comic 01-000.jpg", "Some Comic 01-001.jpg", "Some Comic 01-002.jpg",
    ]
    assert [p.name for p in skipped] == ["Kingpin8.jpg"]


def test_orders_by_numeric_index_not_lexically(script, extracted):
    _touch(extracted,
           "C 01-2.jpg", "C 01-10.jpg", "C 01-1.jpg")
    numbered, _ = script("extract").find_page_files(extracted)
    assert [p.name for p in numbered] == ["C 01-1.jpg", "C 01-2.jpg", "C 01-10.jpg"]


def test_mixed_extensions_all_considered(script, extracted):
    _touch(extracted, "C 01-000.jpg", "C 01-001.png", "C 01-002.jpeg")
    numbered, skipped = script("extract").find_page_files(extracted)
    assert len(numbered) == 3
    assert skipped == []


def test_no_numbered_files_everything_is_skipped(script, extracted):
    """A scan with an unusual naming scheme yields zero pages -- documents the
    current (fragile) behaviour so a future fix is a deliberate change."""
    _touch(extracted, "cover.jpg", "page_a.jpg", "page_b.jpg")
    numbered, skipped = script("extract").find_page_files(extracted)
    assert numbered == []
    assert len(skipped) == 3
