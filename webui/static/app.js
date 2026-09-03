"use strict";

const $ = (s, r = document) => r.querySelector(s);
const api = (p, o) => fetch(p, o).then(async r => {
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `Request failed (${r.status})`);
  return body;
});
let PHASES = [];
api("/api/phases").then(p => { PHASES = p; refreshQueue(); });

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

/* ---------- upload ---------- */
const form = $("#gen-form");
const fileInput = $("#file");
const errBox = $("#form-error");
const addOk = $("#add-ok");
const submit = $("#submit");

fileInput.addEventListener("change", () => {
  const stem = (fileInput.files[0]?.name || "").replace(/\.[^.]+$/, "");
  const m = stem.match(/^(.+?)[\s_#-]+0*(\d{1,4})\b/);
  if (m && !$("#series").value) $("#series").value = m[1].trim();
  if (m && !$("#number").value) $("#number").value = m[2];
});
["dragenter", "dragover"].forEach(ev => form.addEventListener(ev, e => {
  e.preventDefault(); form.classList.add("drag");
}));
["dragleave", "drop"].forEach(ev => form.addEventListener(ev, e => {
  e.preventDefault(); form.classList.remove("drag");
}));
form.addEventListener("drop", e => {
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    fileInput.dispatchEvent(new Event("change"));
  }
});
function showError(msg) { errBox.textContent = msg; errBox.hidden = false; }

form.addEventListener("submit", async e => {
  e.preventDefault();
  errBox.hidden = true; addOk.hidden = true;
  if (!fileInput.files.length) return showError("Choose a comic file first.");
  submit.disabled = true; submit.textContent = "Adding…";
  const fd = new FormData();
  fd.append("file", fileInput.files[0]);
  fd.append("series", $("#series").value);
  fd.append("number", $("#number").value);
  try {
    const job = await api("/api/jobs", { method: "POST", body: fd });
    addOk.textContent = job.status === "running"
      ? `Added "${jobTitle(job)}" — generating now.`
      : `Added "${jobTitle(job)}" — position ${job.queue_pos} in line.`;
    addOk.hidden = false;
    form.reset();
    refreshQueue();
  } catch (err) {
    showError(err.message);
  } finally {
    submit.disabled = false; submit.textContent = "Add";
  }
});

/* ---------- persistent player ---------- */
const np = $("#nowplaying");
const player = $("#player");
const npName = $("#np-name");
let nowPlayingId = null;

function playAudio(url, name, id) {
  npName.textContent = name;
  player.src = url;
  np.hidden = false;
  nowPlayingId = id || url;
  player.play().catch(() => {});
}
$("#np-close").addEventListener("click", () => {
  player.pause();
  player.removeAttribute("src");
  player.load();
  np.hidden = true;
  nowPlayingId = null;
  refreshQueue();
});
player.addEventListener("play", markPlaying);
player.addEventListener("pause", markPlaying);
function markPlaying() {
  document.querySelectorAll("[data-play-id]").forEach(b => {
    const on = b.dataset.playId === String(nowPlayingId) && !player.paused;
    b.textContent = on ? "Pause" : "Play";
    b.setAttribute("aria-pressed", String(on));
  });
}

/* ---------- queue ---------- */
const queueList = $("#queue-list");
const queueSummary = $("#queue-summary");
const queueCount = $("#queue-count");
setInterval(refreshQueue, 2500);

function jobTitle(j) {
  return j.series && j.number
    ? `${j.series} ${String(j.number).padStart(2, "0")}`
    : (j.filename || j.id);
}
function fmtDur(s) {
  if (!s) return "";
  const m = Math.round(s / 60);
  return ` · ${m} min`;
}
function statusText(j) {
  if (j.status === "queued") return `Waiting — position ${j.queue_pos} in line`;
  if (j.status === "running") return j.progress ? `${j.phase_label}: ${j.progress}` : (j.phase_label || "Starting") + "…";
  if (j.status === "done") return "Finished" + fmtDur(j.duration_s);
  if (j.status === "cancelled") return "Cancelled";
  return "Failed — " + (j.error || "unknown error");
}

function track(j) {
  if (!PHASES.length) return "";
  const cur = PHASES.findIndex(p => p.key === j.phase);
  const fill = j.status === "done" ? 100 : (j.status === "running" ? j.percent : 0);
  const dots = PHASES.map((p, i) => {
    let st = "pending";
    if (j.status === "done" || i < cur) st = "done";
    else if (i === cur && j.status === "running") st = "current";
    const left = PHASES.length === 1 ? 50 : (i / (PHASES.length - 1)) * 100;
    return `<span class="dot" data-state="${st}" style="left:${left}%" title="${p.label}"></span>`;
  }).join("");
  const legend = PHASES.map((p, i) => {
    let st = "pending";
    if (j.status === "done" || i < cur) st = "done";
    else if (i === cur && j.status === "running") st = "current";
    return `<span data-state="${st}">${p.label}</span>`;
  }).join("");
  return `<div class="track" role="img" aria-label="Phase ${Math.max(cur + 1, 0)} of ${PHASES.length}: ${statusText(j)}">
      <span class="fill" style="width:${fill}%"></span>${dots}
    </div>
    <div class="phase-legend" aria-hidden="true">${legend}</div>`;
}

async function refreshQueue() {
  let jobs;
  try { jobs = await api("/api/jobs"); }
  catch { queueSummary.textContent = "Couldn't reach the server."; return; }

  const active = jobs.filter(j => j.status === "queued" || j.status === "running");
  queueSummary.textContent = active.length
    ? `${active.length} being generated or waiting.`
    : (jobs.length ? "Nothing generating right now." : "No comics yet — add one above.");
  queueCount.hidden = active.length === 0;
  queueCount.textContent = active.length;

  queueList.innerHTML = jobs.map(j => {
    const actions = [];
    if (j.status === "queued" || j.status === "running")
      actions.push(`<button data-act="cancel" data-id="${j.id}">${j.status === "running" ? "Stop" : "Cancel"}</button>`);
    if (j.status === "done") {
      const nm = jobTitle(j).replace(/"/g, "&quot;");
      actions.push(`<button data-act="play" data-id="${j.id}" data-play-id="${j.id}" data-name="${nm}" aria-pressed="false">Play</button>`);
      actions.push(`<a class="btnlink" href="/api/jobs/${j.id}/download" download>Download</a>`);
    }
    if (["done", "failed", "cancelled"].includes(j.status))
      actions.push(`<button data-act="remove" data-id="${j.id}">Remove</button>`);
    const showTrack = j.status === "running" || j.status === "done";
    return `<li class="jobrow" data-status="${j.status}">
      <div class="jobmain">
        <span class="jobtitle">${jobTitle(j)}</span>
        <span class="jobstatus">${statusText(j)}</span>
      </div>
      ${showTrack ? track(j) : ""}
      <div class="jobactions">${actions.join(" ")}</div>
    </li>`;
  }).join("");
  markPlaying();
}

queueList.addEventListener("click", async e => {
  const btn = e.target.closest("button[data-act]");
  if (!btn) return;
  const { act, id, name } = btn.dataset;
  if (act === "play") {
    if (nowPlayingId === id && !player.paused) player.pause();
    else if (nowPlayingId === id) player.play();
    else playAudio(`/api/jobs/${id}/download`, name, id);
    return;
  }
  btn.disabled = true;
  try {
    if (act === "cancel") await api(`/api/jobs/${id}/cancel`, { method: "POST" });
    if (act === "remove") {
      if (nowPlayingId === id) $("#np-close").click();
      await api(`/api/jobs/${id}`, { method: "DELETE" });
    }
  } catch (err) { alert(err.message); }
  refreshQueue();
});

/* ---------- review (stub) ---------- */
async function loadReviewJobs() {
  const sel = $("#review-job");
  try {
    const jobs = await api("/api/jobs");
    const done = jobs.filter(j => j.status === "done");
    sel.innerHTML = `<option value="">${done.length ? "Choose a comic…" : "No finished comics yet"}</option>`
      + done.map(j => `<option value="${j.id}">${jobTitle(j)}</option>`).join("");
  } catch {
    sel.innerHTML = `<option value="">Couldn't load jobs</option>`;
  }
}
