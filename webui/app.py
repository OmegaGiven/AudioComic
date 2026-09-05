"""AudioComic Studio -- FastAPI app.

    uvicorn webui.app:app --host 0.0.0.0 --port 8971
"""
from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from webui.jobs import CLAUDE_PHASES, PHASES, JobStore, Runner

STATIC = Path(__file__).resolve().parent / "static"
ALLOWED = {".cbz", ".cbr", ".pdf", ".zip"}

app = FastAPI(title="AudioComic Studio")
store = JobStore()
runner = Runner(store)
store.reconcile(runner)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text()


@app.get("/api/phases")
def phases() -> dict:
    from webui.jobs import PHASE_LABEL
    return {
        "local": [{"key": k, "label": PHASE_LABEL[k]} for k in PHASES],
        "claude": [{"key": k, "label": PHASE_LABEL[k]} for k in CLAUDE_PHASES],
    }


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [asdict(j) for j in store.list()]


@app.post("/api/jobs")
async def create_job(file: UploadFile, series: str = Form(""),
                     number: str = Form(""), vision: str = Form("local")) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(
            400, f"That file type isn't supported. Upload one of: "
                 f"{', '.join(sorted(ALLOWED))}.")
    data = await file.read()
    num = int(number) if number.strip().isdigit() else None
    job = store.create(file.filename or f"comic{ext}", data,
                       series=series.strip(), number=num, vision=vision)
    runner.enqueue(job.id)
    return asdict(store.get(job.id))


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str) -> dict:
    if not runner.cancel(jid):
        raise HTTPException(409, "That job can't be cancelled (already finished?).")
    return asdict(store.get(jid))


@app.post("/api/jobs/{jid}/rerun")
def rerun_job(jid: str, phase: str = Form(...)) -> dict:
    ok, why = runner.rerun(jid, phase)
    if not ok:
        raise HTTPException(409, why)
    return asdict(store.get(jid))


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str) -> dict:
    job = store.get(jid)
    if not job:
        raise HTTPException(404, "No job with that id.")
    if job.status in ("running", "queued"):
        raise HTTPException(409, "Cancel the job before removing it.")
    shutil.rmtree(job.dir, ignore_errors=True)
    return {"removed": jid}


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


def _work_file(jid: str, name: str) -> Path:
    job = store.get(jid)
    if not job:
        raise HTTPException(404, "No job with that id.")
    p = job.dir / "work" / name
    if not p.exists():
        raise HTTPException(404, f"{name} isn't available yet.")
    return p


@app.get("/api/jobs/{jid}/db")
def get_db(jid: str) -> dict:
    return json.loads(_work_file(jid, "comic.json").read_text())


@app.get("/api/jobs/{jid}/narrative")
def get_narrative(jid: str) -> dict:
    try:
        return json.loads(_work_file(jid, "narrative.json").read_text())
    except HTTPException:
        return {}


@app.get("/api/jobs/{jid}/pages/{idx}")
def get_page_image(jid: str, idx: int) -> FileResponse:
    db = json.loads(_work_file(jid, "comic.json").read_text())
    page = next((p for p in db["pages"] if p["index"] == idx), None)
    if not page or not Path(page["image"]).exists():
        raise HTTPException(404, "No such page.")
    return FileResponse(page["image"])


@app.get("/api/jobs/{jid}/panels/{pid}")
def get_panel_image(jid: str, pid: str) -> FileResponse:
    db = json.loads(_work_file(jid, "comic.json").read_text())
    panel = next((p for p in db["panels"] if p["id"] == pid), None)
    if not panel or not Path(panel["image"]).exists():
        raise HTTPException(404, "No such panel.")
    return FileResponse(panel["image"])


app.mount("/static", StaticFiles(directory=STATIC), name="static")
