"""Job store + background runner.

One directory per job: jobs/<id>/
    job.json          state
    upload/<file>     the comic the user uploaded
    work/             the pipeline work dir (comic.json, panels/, narrative.json)
    <Series NN>.mp3   the finished audio (in work/, symlinked/copied here)

The runner shells out to pipeline/run.sh and parses its stdout for phase and
progress markers, so it inherits all the venv / GPU-juggling logic.
"""
from __future__ import annotations

import json
import queue
import re
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = Path(__file__).resolve().parent / "jobs"

PHASES = ["segment", "transcribe", "identify", "resolve", "redescribe",
          "assemble", "render", "publish"]
PHASE_LABEL = {
    "segment": "Splitting pages into panels",
    "transcribe": "Reading every panel",
    "identify": "Identifying characters",
    "resolve": "Working out character names",
    "redescribe": "Describing each scene",
    "assemble": "Writing the script",
    "render": "Recording the voices",
    "publish": "Publishing to the library",
}


@dataclass
class Job:
    id: str
    filename: str
    series: str = ""
    number: int | None = None
    status: str = "queued"          # queued | running | done | failed | cancelled
    queue_pos: int = 0              # 1-based place in line while queued
    phase: str = ""                 # current phase key
    phase_label: str = ""
    progress: str = ""              # human line, e.g. "panel 40 of 130"
    percent: int = 0
    created: float = field(default_factory=time.time)
    finished: float | None = None
    error: str = ""
    mp3: str = ""                   # basename of the finished file
    duration_s: float | None = None

    @property
    def dir(self) -> Path:
        return JOBS_DIR / self.id

    def save(self) -> None:
        (self.dir / "job.json").write_text(json.dumps(asdict(self), indent=2))


class JobStore:
    def __init__(self) -> None:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def create(self, upload_name: str, data: bytes, *, series: str = "",
               number: int | None = None) -> Job:
        jid = uuid.uuid4().hex[:12]
        job = Job(id=jid, filename=upload_name, series=series, number=number)
        (job.dir / "upload").mkdir(parents=True, exist_ok=True)
        (job.dir / "upload" / upload_name).write_bytes(data)
        job.save()
        return job

    def get(self, jid: str) -> Job | None:
        p = JOBS_DIR / jid / "job.json"
        if not p.exists():
            return None
        return Job(**json.loads(p.read_text()))

    def list(self) -> list[Job]:
        out = [self.get(d.name) for d in JOBS_DIR.iterdir() if d.is_dir()]
        out = [j for j in out if j]
        out.sort(key=lambda j: j.created, reverse=True)  # newest first
        return out

    def reconcile(self, runner: Runner) -> None:
        """On startup: re-attach a pipeline that's still running, re-queue jobs
        that were waiting, finalize what finished while we were down."""
        for job in reversed(self.list()):  # oldest first -> queue order kept
            if job.status == "running":
                if _pipeline_alive(job.id):
                    runner.reattach(job.id)
                else:
                    _finalize(job, "The server restarted while this comic was generating.")
            elif job.status == "queued":
                runner.enqueue(job.id)


def _pipeline_alive(jid: str) -> bool:
    r = subprocess.run(["pgrep", "-f", f"run.sh .*jobs/{jid}/"],
                       capture_output=True, text=True)
    return r.returncode == 0


def _finalize(job: Job | None, fail_reason: str) -> None:
    if not job:
        return
    wav = job.dir / "out.wav"
    mp3 = next((job.dir / "work").glob("*.mp3"), None)
    if wav.exists() and mp3:
        (job.dir / mp3.name).write_bytes(mp3.read_bytes())
        job.mp3, job.status, job.percent = mp3.name, "done", 100
        job.phase, job.phase_label = "done", "Finished"
        try:
            import wave
            with wave.open(str(wav), "rb") as w:
                job.duration_s = round(w.getnframes() / w.getframerate(), 1)
        except Exception:
            pass
    elif job.status not in ("cancelled",):
        job.status, job.error = "failed", fail_reason
    else:
        job.error = job.error or "Cancelled."
    job.finished = time.time()
    job.save()


_PHASE_RE = re.compile(r"^==\s*([a-z]+)")
_PROG_RE = re.compile(r"\[(\d+)/(\d+)\]")


class Runner:
    """FIFO queue, one job processed at a time (single GPU). A background
    worker pulls job ids and runs the pipeline; the rest wait as 'queued'."""

    def __init__(self, store: JobStore) -> None:
        self.store = store
        self._q: queue.Queue[str] = queue.Queue()
        self._current: str | None = None
        self._procs: dict[str, subprocess.Popen] = {}
        threading.Thread(target=self._worker, daemon=True).start()

    @property
    def current(self) -> str | None:
        return self._current

    def enqueue(self, jid: str) -> None:
        self._q.put(jid)
        self._renumber()

    def reattach(self, jid: str) -> None:
        threading.Thread(target=self._wait_out, args=(jid,), daemon=True).start()

    def cancel(self, jid: str) -> bool:
        job = self.store.get(jid)
        if not job:
            return False
        if job.status == "queued":
            job.status, job.error = "cancelled", "Cancelled before it started."
            job.finished = time.time()
            job.save()
            self._renumber()
            return True
        if job.status == "running" and jid in self._procs:
            try:
                self._procs[jid].send_signal(signal.SIGTERM)
            except Exception:
                pass
            return True
        return False

    def _renumber(self) -> None:
        waiting = [j for j in reversed(self.store.list()) if j.status == "queued"]
        for i, j in enumerate(waiting, 1):
            if j.queue_pos != i:
                j.queue_pos = i
                j.save()

    def _worker(self) -> None:
        while True:
            jid = self._q.get()
            job = self.store.get(jid)
            if not job or job.status != "queued":
                continue
            self._current = jid
            try:
                self._execute(jid)
            finally:
                self._current = None
                self._renumber()

    def _wait_out(self, jid: str) -> None:
        self._current = jid
        while _pipeline_alive(jid):
            time.sleep(5)
        job = self.store.get(jid)
        if job and job.status == "running":
            _finalize(job, "The pipeline stopped before finishing.")
        self._current = None

    def _execute(self, jid: str) -> None:
        job = self.store.get(jid)
        if not job:
            return
        job.status = "running"
        job.queue_pos = 0
        job.save()

        upload = next((job.dir / "upload").iterdir())
        work = job.dir / "work"
        out_wav = job.dir / "out.wav"
        env = {
            "COMIC_SERIES": job.series or "",
            "COMIC_NUMBER": str(job.number) if job.number else "",
            "PATH": "/home/omegagiven/.local/bin:/usr/bin:/bin",
            "HOME": str(Path.home()),
        }
        proc = subprocess.Popen(
            ["bash", str(REPO_ROOT / "pipeline" / "run.sh"),
             str(upload), str(work), str(out_wav)],
            cwd=str(REPO_ROOT), env={**env}, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True,  # survive a web-server restart
        )
        self._procs[jid] = proc
        total_phases = len(PHASES)
        for line in proc.stdout or []:
            line = line.rstrip()
            m = _PHASE_RE.match(line)
            if m and m.group(1) in PHASE_LABEL:
                job.phase = m.group(1)
                job.phase_label = PHASE_LABEL[job.phase]
                job.progress = ""
                job.percent = int(100 * PHASES.index(job.phase) / total_phases)
                job.save()
                continue
            pm = _PROG_RE.search(line)
            if pm:
                done, tot = int(pm.group(1)), int(pm.group(2))
                job.progress = f"{done} of {tot}"
                if job.phase in PHASES:
                    base = PHASES.index(job.phase) / total_phases
                    job.percent = int(100 * (base + (done / max(tot, 1)) / total_phases))
                job.save()
        proc.wait()
        self._procs.pop(jid, None)
        job = self.store.get(jid)
        if job and job.status == "running" and proc.returncode not in (0,):
            job.status = "cancelled" if proc.returncode in (-15, -2, 143) else "failed"
        _finalize(job or self.store.get(jid),
                  "The pipeline stopped before finishing. Check the server log.")
