"use strict";

const $ = (s, r = document) => r.querySelector(s);
const api = (p, o) => fetch(p, o).then(async r => {
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `Request failed (${r.status})`);
  return body;
});

/* ---------- tabs ---------- */
const tabs = [...document.querySelectorAll('[role="tab"]')];
function selectTab(tab) {
  tabs.forEach(t => {
    const on = t === tab;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
    $("#" + t.getAttribute("aria-controls")).hidden = !on;
  });
  tab.focus();
  if (tab.id === "tab-review") loadReviewJobs();
}
tabs.forEach((t, i) => {
  t.addEventListener("click", () => selectTab(t));
  t.addEventListener("keydown", e => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      selectTab(tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length]);
    }
  });
});

/* ---------- generate ---------- */
const form = $("#gen-form");
const fileInput = $("#file");
const chosen = $("#file-chosen");
const dropzone = $("#dropzone");
const errBox = $("#form-error");
const submit = $("#submit");
let PHASES = [];

api("/api/phases").then(p => { PHASES = p; });

fileInput.addEventListener("change", () => {
  chosen.textContent = fileInput.files.length
    ? `Chosen: ${fileInput.files[0].name}` : "No file chosen";
  const stem = (fileInput.files[0]?.name || "").replace(/\.[^.]+$/, "");
  const m = stem.match(/^(.+?)[\s_#-]+0*(\d{1,4})\s*$/);
  if (m && !$("#series").value) $("#series").value = m[1].trim();
  if (m && !$("#number").value) $("#number").value = m[2];
});

["dragenter", "dragover"].forEach(ev => dropzone.addEventListener(ev, e => {
  e.preventDefault(); dropzone.classList.add("drag");
}));
["dragleave", "drop"].forEach(ev => dropzone.addEventListener(ev, e => {
  e.preventDefault(); dropzone.classList.remove("drag");
}));
dropzone.addEventListener("drop", e => {
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    fileInput.dispatchEvent(new Event("change"));
  }
});

function showError(msg) {
  errBox.textContent = msg;
  errBox.hidden = false;
  errBox.focus?.();
}

form.addEventListener("submit", async e => {
  e.preventDefault();
  errBox.hidden = true;
  if (!fileInput.files.length) return showError("Please choose a comic file first.");
  submit.disabled = true;
  submit.textContent = "Starting…";
  const fd = new FormData();
  fd.append("file", fileInput.files[0]);
  fd.append("series", $("#series").value);
  fd.append("number", $("#number").value);
  try {
    const job = await api("/api/jobs", { method: "POST", body: fd });
    startProgress(job.id);
  } catch (err) {
    showError(err.message);
    submit.disabled = false;
    submit.textContent = "Start generating";
  }
});

/* ---------- progress ---------- */
const progressBox = $("#progress");
const statusLine = $("#progress-status");
const barFill = $("#bar-fill");
const phaseList = $("#phase-list");
const resultBox = $("#result");

function renderPhaseList(job) {
  const cur = PHASES.findIndex(p => p.key === job.phase);
  phaseList.innerHTML = PHASES.map((p, i) => {
    let state = "pending";
    if (job.status === "failed" && i === cur) state = "failed";
    else if (i < cur || job.status === "done") state = "done";
    else if (i === cur) state = "running";
    const detail = state === "running" && job.progress ? job.progress : "";
    return `<li data-state="${state}">
      <span class="icon" aria-hidden="true"></span>
      <span class="name">${p.label}</span>
      <span class="detail">${detail}</span>
    </li>`;
  }).join("");
}

function startProgress(jid) {
  progressBox.hidden = false;
  resultBox.hidden = true;
  statusLine.textContent = "Starting…";
  progressBox.scrollIntoView({ block: "nearest" });
  $("#progress-h").tabIndex = -1;
  $("#progress-h").focus();

  const es = new EventSource(`/api/jobs/${jid}/events`);
  es.onmessage = ev => {
    const job = JSON.parse(ev.data);
    renderPhaseList(job);
    barFill.style.width = job.percent + "%";
    document.title = job.status === "done"
      ? "Done — AudioComic Studio"
      : `${job.percent}% — AudioComic Studio`;

    if (job.status === "running") {
      statusLine.textContent = job.progress
        ? `${job.phase_label}: ${job.progress}`
        : job.phase_label + "…";
    } else if (job.status === "done") {
      statusLine.textContent = "Finished.";
      es.close();
      showResult(job);
    } else if (job.status === "failed") {
      statusLine.textContent = "Something went wrong: " + (job.error || "unknown error");
      es.close();
      submit.disabled = false;
      submit.textContent = "Start generating";
    }
  };
  es.onerror = () => { statusLine.textContent = "Lost connection to the server. Reload to check status."; };
}

function fmtDuration(s) {
  if (!s) return "";
  const m = Math.round(s / 60);
  return `, ${m} minute${m === 1 ? "" : "s"}`;
}

function showResult(job) {
  resultBox.hidden = false;
  const title = job.series && job.number
    ? `${job.series} ${String(job.number).padStart(2, "0")}` : job.filename;
  const dl = $("#download");
  dl.href = `/api/jobs/${job.id}/download`;
  dl.setAttribute("aria-label", `Download ${title}, MP3${fmtDuration(job.duration_s)}`);
  dl.textContent = `Download ${title}`;
  $("#player").src = `/api/jobs/${job.id}/download`;
  $("#open-review").onclick = e => {
    e.preventDefault();
    $("#review-job").value = job.id;
    selectTab($("#tab-review"));
  };
  resultBox.scrollIntoView({ block: "nearest" });
  $("#result-h").tabIndex = -1;
  $("#result-h").focus();
  try { new Audio("data:audio/wav;base64,UklGRl9vT19XQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=").play(); } catch (e) {}
}

/* ---------- review (stub) ---------- */
async function loadReviewJobs() {
  const sel = $("#review-job");
  try {
    const jobs = await api("/api/jobs");
    const done = jobs.filter(j => j.status === "done");
    sel.innerHTML = `<option value="">${done.length ? "Choose a comic…" : "No finished comics yet"}</option>`
      + done.map(j => `<option value="${j.id}">${j.series || j.filename} ${j.number || ""}</option>`).join("");
  } catch (e) {
    sel.innerHTML = `<option value="">Couldn't load jobs</option>`;
  }
}
