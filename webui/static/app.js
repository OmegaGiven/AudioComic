"use strict";

const $ = (s, r = document) => r.querySelector(s);
const api = (p, o) => fetch(p, o).then(async r => {
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `Request failed (${r.status})`);
  return body;
});
let PHASES = [];
api("/api/phases").then(p => { PHASES = p; });

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
  if (tab.id === "tab-queue") { refreshQueue(); startQueuePoll(); }
  else stopQueuePoll();
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
const addOk = $("#add-ok");

fileInput.addEventListener("change", () => {
  chosen.textContent = fileInput.files.length
    ? `Chosen: ${fileInput.files[0].name}` : "No file chosen";
  const stem = (fileInput.files[0]?.name || "").replace(/\.[^.]+$/, "");
  const m = stem.match(/^(.+?)[\s_#-]+0*(\d{1,4})\b/);
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
function showError(msg) { errBox.textContent = msg; errBox.hidden = false; }

form.addEventListener("submit", async e => {
  e.preventDefault();
  errBox.hidden = true; addOk.hidden = true;
  if (!fileInput.files.length) return showError("Please choose a comic file first.");
  submit.disabled = true; submit.textContent = "Adding…";
  const fd = new FormData();
  fd.append("file", fileInput.files[0]);
  fd.append("series", $("#series").value);
  fd.append("number", $("#number").value);
  try {
    const job = await api("/api/jobs", { method: "POST", body: fd });
    addOk.textContent = job.status === "running"
      ? "Added — now generating. See the Queue tab."
      : `Added to the queue (position ${job.queue_pos}). See the Queue tab.`;
    addOk.hidden = false;
    form.reset(); chosen.textContent = "No file chosen";
    refreshQueue();
  } catch (err) {
    showError(err.message);
  } finally {
    submit.disabled = false; submit.textContent = "Add to queue";
  }
});

/* ---------- queue ---------- */
const queueList = $("#queue-list");
const queueSummary = $("#queue-summary");
const queueCount = $("#queue-count");
let queueTimer = null;

function startQueuePoll() { if (!queueTimer) queueTimer = setInterval(refreshQueue, 2500); }
function stopQueuePoll() { clearInterval(queueTimer); queueTimer = null; }

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

async function refreshQueue() {
  let jobs;
  try { jobs = await api("/api/jobs"); }
  catch { queueSummary.textContent = "Couldn't reach the server."; return; }

  const active = jobs.filter(j => j.status === "queued" || j.status === "running");
  queueSummary.textContent = active.length
    ? `${active.length} in the queue.`
    : (jobs.length ? "Queue is empty." : "Nothing here yet — add a comic on the Generate tab.");
  queueCount.hidden = active.length === 0;
  queueCount.textContent = active.length;

  queueList.innerHTML = jobs.map(j => {
    const pct = j.status === "running" ? j.percent : (j.status === "done" ? 100 : 0);
    const actions = [];
    if (j.status === "queued" || j.status === "running")
      actions.push(`<button data-act="cancel" data-id="${j.id}">${j.status === "running" ? "Stop" : "Cancel"}</button>`);
    if (j.status === "done")
      actions.push(`<a class="btnlink" href="/api/jobs/${j.id}/download" download>Download</a>`);
    if (["done", "failed", "cancelled"].includes(j.status))
      actions.push(`<button data-act="remove" data-id="${j.id}">Remove</button>`);
    return `<li class="jobrow" data-status="${j.status}">
      <div class="jobmain">
        <span class="jobtitle">${jobTitle(j)}</span>
        <span class="jobstatus">${statusText(j)}</span>
      </div>
      <div class="jobbar" role="img" aria-label="${pct}% done"><span style="width:${pct}%"></span></div>
      <div class="jobactions">${actions.join(" ")}</div>
    </li>`;
  }).join("");
}

queueList.addEventListener("click", async e => {
  const btn = e.target.closest("button[data-act]");
  if (!btn) return;
  const { act, id } = btn.dataset;
  btn.disabled = true;
  try {
    if (act === "cancel") await api(`/api/jobs/${id}/cancel`, { method: "POST" });
    if (act === "remove") await api(`/api/jobs/${id}`, { method: "DELETE" });
  } catch (err) { alert(err.message); }
  refreshQueue();
});

/* keep the badge current even when the tab isn't open */
setInterval(async () => {
  try {
    const jobs = await api("/api/jobs");
    const n = jobs.filter(j => j.status === "queued" || j.status === "running").length;
    queueCount.hidden = n === 0;
    queueCount.textContent = n;
  } catch {}
}, 5000);

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
