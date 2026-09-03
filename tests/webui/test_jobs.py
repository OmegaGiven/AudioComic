"""Tests for webui.jobs -- progress parsing, queue, reconcile."""
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


def test_enqueue_numbers_the_line(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path)
    store = jobs.JobStore()
    runner = jobs.Runner.__new__(jobs.Runner)  # no worker thread
    import queue as _q
    runner.store, runner._q, runner._current, runner._procs = store, _q.Queue(), None, {}
    a = store.create("a.cbz", b"x")
    b = store.create("b.cbz", b"x")
    runner.enqueue(a.id)
    runner.enqueue(b.id)
    assert store.get(a.id).queue_pos == 1
    assert store.get(b.id).queue_pos == 2


def test_cancel_queued_job(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path)
    store = jobs.JobStore()
    runner = jobs.Runner.__new__(jobs.Runner)
    import queue as _q
    runner.store, runner._q, runner._current, runner._procs = store, _q.Queue(), None, {}
    j = store.create("a.cbz", b"x")
    runner.enqueue(j.id)
    assert runner.cancel(j.id) is True
    assert store.get(j.id).status == "cancelled"


def test_reconcile_finalizes_dead_orphan(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(jobs, "_pipeline_alive", lambda jid: False)
    store = jobs.JobStore()
    runner = jobs.Runner.__new__(jobs.Runner)
    import queue as _q
    runner.store, runner._q, runner._current, runner._procs = store, _q.Queue(), None, {}
    job = store.create("x.cbz", b"data")
    job.status = "running"
    job.save()
    store.reconcile(runner)
    assert store.get(job.id).status == "failed"
