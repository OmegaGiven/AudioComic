"""Tests for webui.jobs progress parsing + reconcile."""
from __future__ import annotations

from webui import jobs


def test_phase_and_progress_regexes():
    assert jobs._PHASE_RE.match("== transcribe ==").group(1) == "transcribe"
    m = jobs._PROG_RE.search("[40/130] p012_03: 3 text lines")
    assert (m.group(1), m.group(2)) == ("40", "130")
    assert jobs._PHASE_RE.match("Done -> x.wav") is None


def test_every_phase_has_a_label():
    assert set(jobs.PHASES) <= set(jobs.PHASE_LABEL)
    assert all(jobs.PHASE_LABEL[p] for p in jobs.PHASES)


def test_reconcile_marks_orphan_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path)
    store = jobs.JobStore()
    job = store.create("x.cbz", b"data")
    job.status = "running"
    job.save()
    store.reconcile()
    assert store.get(job.id).status == "failed"
