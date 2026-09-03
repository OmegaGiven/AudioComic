"""AudioComic Studio -- FastAPI app.

    uvicorn webui.app:app --host 0.0.0.0 --port 8971
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from webui.jobs import PHASES, JobStore, Runner

STATIC = Path(__file__).resolve().parent / "static"
ALLOWED = {".cbz", ".cbr", ".pdf", ".zip"}

app = FastAPI(title="AudioComic Studio")
store = JobStore()
runner = Runner(store)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text()


@app.get("/api/phases")
def phases() -> list[dict]:
    from webui.jobs import PHASE_LABEL
    return [{"key": k, "label": PHASE_LABEL[k]} for k in PHASES]


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [asdict(j) for j in store.list()]


@app.post("/api/jobs")
async def create_job(file: UploadFile, series: str = Form(""),
                     number: str = Form("")) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(
            400, f"That file type isn't supported. Upload one of: "
                 f"{', '.join(sorted(ALLOWED))}.")
    if runner.busy:
        raise HTTPException(
            409, "A comic is already being generated. Please wait for it to "
                 "finish, then try again.")
    data = await file.read()
    num = int(number) if number.strip().isdigit() else None
    job = store.create(file.filename or f"comic{ext}", data,
                       series=series.strip(), number=num)
    runner.start(job)
    return asdict(job)


@app.get("/api/jobs/{jid}")
def get_job(jid: str) -> dict:
    job = store.get(jid)
    if not job:
        raise HTTPException(404, "No job with that id.")
    return asdict(job)


@app.get("/api/jobs/{jid}/events")
async def job_events(jid: str) -> StreamingResponse:
    if not store.get(jid):
        raise HTTPException(404, "No job with that id.")

    async def gen():
        last = None
        while True:
            job = store.get(jid)
            if not job:
                break
            payload = json.dumps(asdict(job))
            if payload != last:
                yield f"data: {payload}\n\n"
                last = payload
            if job.status in ("done", "failed"):
                break
            await asyncio.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.get("/api/jobs/{jid}/download")
def download(jid: str) -> FileResponse:
    job = store.get(jid)
    if not job or not job.mp3:
        raise HTTPException(404, "That audio comic isn't ready yet.")
    path = job.dir / job.mp3
    if not path.exists():
        raise HTTPException(404, "The audio file is missing.")
    return FileResponse(path, media_type="audio/mpeg", filename=job.mp3)


@app.get("/api/jobs/{jid}/db")
def get_db(jid: str) -> dict:
    job = store.get(jid)
    if not job:
        raise HTTPException(404, "No job with that id.")
    p = job.dir / "work" / "comic.json"
    if not p.exists():
        raise HTTPException(404, "The database isn't built yet.")
    return json.loads(p.read_text())


app.mount("/static", StaticFiles(directory=STATIC), name="static")
