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
import re
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
    status: str = "queued"          # queued | running | done | failed
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
        out = []
        for d in sorted(JOBS_DIR.iterdir(), reverse=True):
            j = self.get(d.name) if d.is_dir() else None
            if j:
                out.append(j)
        return out

    def reconcile(self, runner: Runner | None = None) -> None:
        """On startup: a job left 'running' by a server restart is orphaned.
        If its pipeline process is still alive, re-attach a monitor; else
        finish it from disk, or mark it failed."""
        for job in self.list():
            if job.status != "running":
                continue
            if _pipeline_alive(job.id):
                if runner:
                    runner.reattach(job.id)
                continue
            _finalize(job, "The server restarted while this comic was generating.")


def _pipeline_alive(jid: str) -> bool:
    r = subprocess.run(["pgrep", "-f", f"run.sh .*jobs/{jid}/"],
                       capture_output=True, text=True)
    return r.returncode == 0


def _finalize(job: Job, fail_reason: str) -> None:
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
    else:
        job.status, job.error = "failed", fail_reason
    job.finished = time.time()
    job.save()


_PHASE_RE = re.compile(r"^==\s*([a-z]+)")
_PROG_RE = re.compile(r"\[(\d+)/(\d+)\]")
_TXT_PROG_RE = re.compile(r"^(\d+)\s+panels?\s+to\s+(transcribe|describe)", re.I)


class Runner:
    """Runs one job at a time in a background thread."""

    def __init__(self, store: JobStore) -> None:
        self.store = store
        self._lock = threading.Lock()
        self._current: str | None = None

    @property
    def busy(self) -> bool:
        return self._current is not None

    def start(self, job: Job) -> None:
        threading.Thread(target=self._run, args=(job.id, self._execute),
                         daemon=True).start()

    def reattach(self, jid: str) -> None:
        """A pipeline still running after a server restart -- wait it out and
        finalize from disk (progress won't update, but the job completes)."""
        threading.Thread(target=self._run, args=(jid, self._wait_out),
                         daemon=True).start()

    def _run(self, jid: str, fn) -> None:
        with self._lock:
            if self._current:
                return
            self._current = jid
        try:
            fn(jid)
        finally:
            with self._lock:
                self._current = None

    def _wait_out(self, jid: str) -> None:
        while _pipeline_alive(jid):
            time.sleep(5)
        job = self.store.get(jid)
        if job and job.status == "running":
            _finalize(job, "The pipeline stopped before finishing.")

    def _execute(self, jid: str) -> None:
        job = self.store.get(jid)
        if not job:
            return
        job.status = "running"
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
        )
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
        _finalize(job, "The pipeline stopped before finishing. Check the server log.")
