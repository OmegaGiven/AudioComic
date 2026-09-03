"""Tests for pipeline.publish title/path derivation."""
from __future__ import annotations

from pipeline.comicdb import ComicDB
from pipeline.publish import title_for


def test_title_from_issue_metadata(tmp_path):
    db = ComicDB.new(tmp_path, series="Blackest Night", number=1, source="bn01.cbz")
    assert title_for(db) == ("Blackest Night", "Blackest Night 01")


def test_title_falls_back_to_archive_name(tmp_path):
    db = ComicDB.new(tmp_path, source="Saga 007.cbz")
    assert title_for(db) == ("Saga", "Saga 07")


def test_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("COMIC_SERIES", "Blackest Night")
    monkeypatch.setenv("COMIC_NUMBER", "1")
    db = ComicDB.new(tmp_path, source="bn01-full.cbz")
    assert title_for(db) == ("Blackest Night", "Blackest Night 01")


def test_slug_strips_unsafe_chars(tmp_path):
    db = ComicDB.new(tmp_path, series="X/Men: Blue", number=3, source="x.cbz")
    series, name = title_for(db)
    assert "/" not in series and ":" not in name
