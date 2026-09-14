// Role-play cast builder: agent cards, Markdown agent files (upload and download), AI
// drafting, and cast checks. Loaded by new_experiment.html, which defines PREFIX and
// syncJSON() before any of these run.
//
// Field names match agent_spec.py on the server, which owns the file format; this file
// only moves agents between cards and the server's JSON.

const CAST_FIELDS = [
  // [key, label, control, placeholder, group]
  ["name", "Name", "input", "Dana Reyes"],
  ["role", "Role", "input", "Lead counsel for the Provider"],
  ["goal", "Objective", "area", "What is this person trying to achieve?"],
  ["backstory", "Background", "area", "Where they come from, and the experience that shapes how they act."],
  ["demeanor", "Demeanor", "input", "Calm and precise; never raises their voice."],
  ["tendencies", "Tendencies", "list", "Anchors hard early\nReframes every risk as a dollar figure"],
  ["priorities", "Priorities", "area", "The interests behind their position: what they care about most, and why."],
  ["bottom_line", "Bottom line", "area", "The point past which they would rather walk away than agree, and what they would do instead.", "private"],
  ["confidential", "Confidential information", "area", "Facts only this person knows.", "private"],
  ["notes", "Additional notes", "area", "Anything else about this person.", "more"],
  ["id", "Short id", "input", "made from the name automatically", "more"],
  ["system_prompt", "Full prompt (replaces everything above)", "area", "Only if you want to write the whole prompt yourself.", "more"],
];

function att(s) { return (s || "").replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;"); }

// Mirrors agent_spec.slugify, so ids made here match ids made from uploaded files.
function slugify(s) {
  return (s || "").normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase()
    .replace(/[^a-z0-9_]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40).replace(/-+$/, "") || "agent";
}

function field(card, key) { return card.querySelector(`[data-field="${key}"]`); }

function fieldHTML([key, label, control, placeholder, group]) {
  const tag = group === "private" ? ' <span class="private-tag">private</span>'
    : control === "list" ? ' <span class="hint-inline">(one per line)</span>' : "";
  const input = control === "input"
    ? `<input type="text" data-field="${key}" placeholder="${att(placeholder)}">`
    : `<textarea class="short" data-field="${key}" placeholder="${att(placeholder)}"></textarea>`;
  return `<label>${label}${tag}</label>${input}`;
}

function addAgent(agent) {
  agent = agent || {};
  const card = document.createElement("div");
  card.className = "agent-card";
  card.innerHTML = `<div class="agent-card-head">
      <span class="agent-num"></span>
      <span class="agent-actions">
        <button type="button" class="small-btn" onclick="downloadAgent(this.closest('.agent-card'))">Download</button>
        <button type="button" class="remove-agent" onclick="removeAgent(this.closest('.agent-card'))">Remove</button>
      </span>
    </div>
    ${CAST_FIELDS.filter(f => f[4] !== "more").map(fieldHTML).join("")}
    <details class="agent-more"><summary>More options</summary>
      ${CAST_FIELDS.filter(f => f[4] === "more").map(fieldHTML).join("")}
    </details>`;
  card.querySelectorAll("[data-field]").forEach(el => {
    const v = agent[el.dataset.field];
    el.value = Array.isArray(v) ? v.join("\n") : (v || "");
    el.addEventListener("input", () => onCardInput(card, el));
  });
  // The id follows the name until someone types an id of their own.
  card.dataset.autoId = !agent.id || agent.id === slugify(agent.name) ? "1" : "0";
  if (!agent.id && agent.name) field(card, "id").value = slugify(agent.name);
  document.getElementById("agents").appendChild(card);
  renumber();
  syncJSON();
  return card;
}

function onCardInput(card, el) {
  const key = el.dataset.field;
  if (key === "id") card.dataset.autoId = el.value.trim() ? "0" : "1";
  if (key === "name") {
    if (card.dataset.autoId === "1") field(card, "id").value = el.value.trim() ? slugify(el.value) : "";
    renumber();
  }
  card.classList.remove("has-error", "has-warning");
  syncJSON();
}

function renumber() {
  document.querySelectorAll(".agent-card").forEach((c, i) => {
    const name = field(c, "name").value.trim();
    c.querySelector(".agent-num").textContent = `Agent ${i + 1}` + (name ? ` · ${name}` : "");
  });
}

function removeAgent(card) { card.remove(); renumber(); syncJSON(); }

function readCard(card) {
  const a = {};
  CAST_FIELDS.forEach(([key, , control]) => {
    const raw = field(card, key).value;
    if (control === "list") {
      const items = raw.split("\n")
        .map(s => s.replace(/^\s*(?:[-*+•]|\d{1,3}[.)])\s+/, "").trim()).filter(Boolean);
      if (items.length) a[key] = items;
    } else if (raw.trim()) {
      a[key] = raw.trim();
    }
  });
  if (!a.id && a.name) a.id = slugify(a.name);
  return a.id ? {id: a.id, ...a} : a;
}

function collectAgents() {
  return [...document.querySelectorAll(".agent-card")].map(readCard).filter(a => a.id || a.name);
}

// Problems that would make the experiment impossible to run; checked before creating it.
function castErrors() {
  const agents = collectAgents();
  const errors = agents.length < 2 ? ["A role-play needs at least two agents."] : [];
  const seen = new Set();
  agents.forEach(a => {
    if (!a.name) errors.push(`Agent "${a.id}" needs a name.`);
    if (seen.has(a.id)) errors.push(`Two agents share the id "${a.id}". Change one of their names or short ids.`);
    seen.add(a.id);
  });
  return errors;
}

function showMessages(items, heading) {
  const box = document.getElementById("cast-messages");
  box.hidden = !items.length && !heading;
  box.innerHTML = (heading ? '<p class="cast-msg-head"></p>' : "")
    + `<ul class="issue-list">${items.map(() => "<li></li>").join("")}</ul>`;
  if (heading) box.querySelector(".cast-msg-head").textContent = heading;
  box.querySelectorAll("li").forEach((li, i) => {
    const level = items[i].level || "warning";
    li.className = `issue ${level}`;
    li.innerHTML = '<span class="issue-level"></span> <span class="issue-text"></span>';
    li.querySelector(".issue-level").textContent = level;
    li.querySelector(".issue-text").textContent = items[i].message;
  });
}

async function errorText(r) {
  try { return (await r.json()).detail || r.statusText; } catch (e) { return r.statusText; }
}

async function postJSON(path, body) {
  const r = await fetch(`${PREFIX}${path}`, {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  if (!r.ok) throw new Error(await errorText(r));
  return r;
}

function plural(n, word) { return `${n} ${word}${n === 1 ? "" : "s"}`; }

// ------------------------------------------------------------------ files

async function uploadAgentFiles(fileList) {
  const files = [...fileList];
  if (!files.length) return;
  const form = new FormData();
  files.forEach(f => form.append("files", f, f.name));
  form.append("taken", JSON.stringify(collectAgents().map(a => a.id)));
  showMessages([], `Reading ${plural(files.length, "file")}…`);
  try {
    const r = await fetch(`${PREFIX}/api/agents/upload`, {method: "POST", body: form});
    if (!r.ok) throw new Error(await errorText(r));
    const d = await r.json();
    d.agents.forEach(addAgent);
    showMessages(d.warnings.map(message => ({level: "warning", message})),
      `Added ${plural(d.agents.length, "agent")} from ${plural(files.length, "file")}.`);
  } catch (e) {
    showMessages([{level: "error", message: String(e.message || e)}], "Upload failed.");
  }
  document.getElementById("agent-files").value = "";
}

async function downloadMarkdown(agents, filename) {
  try {
    const r = await postJSON("/api/agents/download", {agents, filename});
    const url = URL.createObjectURL(await r.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = `${filename}.md`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (e) {
    showMessages([{level: "error", message: String(e.message || e)}], "Download failed.");
  }
}

function downloadCast() {
  const agents = collectAgents();
  if (!agents.length) {
    showMessages([{level: "error", message: "There are no agents to download yet."}]);
    return;
  }
  return downloadMarkdown(agents, `${slugify(document.getElementById("name").value || "role-play")}-cast`);
}

function downloadAgent(card) {
  const a = readCard(card);
  if (!a.id) {
    showMessages([{level: "error", message: "Give this agent a name before downloading it."}]);
    return;
  }
  return downloadMarkdown([a], a.id);
}

function enableDropZone() {
  const zone = document.getElementById("cast-drop");
  ["dragenter", "dragover"].forEach(type => zone.addEventListener(type, e => {
    e.preventDefault();
    zone.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach(type => zone.addEventListener(type, e => {
    e.preventDefault();
    zone.classList.remove("dragging");
  }));
  zone.addEventListener("drop", e => uploadAgentFiles(e.dataTransfer.files));
}

// ------------------------------------------------------------------ AI help

function scenarioText() { return document.getElementById("rp_scenario").value.trim(); }

// A source read by /api/source/read (title, kind, url, site, published, words, text, ...),
// kept in the page until a cast is drafted from it.
let currentSource = null;
// What the experiment records about the source its cast was drafted from (no text).
let draftedSource = null;

function draftFrom() {
  const picked = document.querySelector('input[name="draft-from"]:checked');
  return picked ? picked.value : "scenario";
}

function showDraftSource() {
  const from = draftFrom();
  document.querySelectorAll(".source-input").forEach(el => { el.hidden = el.dataset.from !== from; });
  document.getElementById("source-preview").hidden = from === "scenario" || !currentSource;
  document.getElementById("draft-scenario-option").hidden = from === "scenario";
}

async function readSource(from) {
  let request;
  if (from === "link") {
    const url = document.getElementById("source-url").value.trim();
    if (!url) { showMessages([{level: "error", message: "Paste a link first."}]); return; }
    request = postJSON("/api/source/read", {url});
  } else if (from === "file") {
    const file = document.getElementById("source-file").files[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file, file.name);
    request = fetch(`${PREFIX}/api/source/read`, {method: "POST", body: form})
      .then(async r => { if (!r.ok) throw new Error(await errorText(r)); return r; });
  } else {
    const text = document.getElementById("source-text").value;
    if (!text.trim()) { showMessages([{level: "error", message: "Paste some text first."}]); return; }
    request = postJSON("/api/source/read", {text});
  }
  currentSource = null;
  document.getElementById("source-preview").hidden = true;
  showMessages([], from === "link" ? "Reading the link…" : "Reading the source…");
  try {
    currentSource = await (await request).json();
    renderSourcePreview(currentSource);
    showMessages(currentSource.warnings.map(message => ({level: "warning", message})),
      `Read ${plural(currentSource.words, "word")}. Check the preview is the right material, then draft the cast.`);
  } catch (e) {
    showMessages([{level: "error", message: String(e.message || e)}], "The source could not be read.");
  }
}

function renderSourcePreview(s) {
  const box = document.getElementById("source-preview");
  box.innerHTML = '<div class="source-title"></div><div class="source-meta"></div><p class="source-excerpt"></p>';
  box.querySelector(".source-title").textContent = s.title;
  const meta = box.querySelector(".source-meta");
  meta.textContent = [s.kind, s.site, (s.published || "").slice(0, 10),
    `${s.words.toLocaleString()} words${s.truncated ? " (shortened)" : ""}`].filter(Boolean).join(" · ");
  if (s.url) {
    const link = document.createElement("a");
    link.href = s.url;
    link.target = "_blank";
    link.rel = "noopener";
    link.className = "small-link";
    link.textContent = "open original";
    meta.append(" · ", link);
  }
  const words = s.text.split(/\s+/);
  box.querySelector(".source-excerpt").textContent = words.slice(0, 80).join(" ") + (words.length > 80 ? " …" : "");
  box.hidden = false;
}

async function draftCast() {
  const from = draftFrom();
  const scenario = scenarioText();
  if (from === "scenario" && !scenario) {
    showMessages([{level: "error", message: "Describe the scenario first. The AI drafts the cast from it."}]);
    document.getElementById("rp_scenario").focus();
    return;
  }
  if (from !== "scenario" && !currentSource) {
    showMessages([{level: "error", message: "Read the source first, so you can check what was found before drafting."}]);
    return;
  }
  const btn = document.getElementById("draft-btn");
  const replace = document.getElementById("draft-replace").checked;
  const body = {
    count: parseInt(document.getElementById("draft-count").value, 10),
    notes: document.getElementById("draft-notes").value,
    taken: replace ? [] : collectAgents().map(a => a.id),
  };
  if (from === "scenario") body.scenario = scenario; else body.source = currentSource;
  btn.disabled = true;
  showMessages([], from === "scenario"
    ? "Drafting the cast with gemma4. This usually takes one to three minutes…"
    : "Drafting a scenario and cast from the source with gemma4. This usually takes two to four minutes…");
  try {
    const d = await (await postJSON("/api/agents/draft", body)).json();
    if (!d.agents.length) throw new Error(d.warnings[0] || "The AI did not produce any agents. Try again.");
    if (replace) document.querySelectorAll(".agent-card").forEach(c => c.remove());
    d.agents.forEach(addAgent);
    if (d.scenario && (document.getElementById("draft-scenario").checked || !scenario)) {
      document.getElementById("rp_scenario").value = d.scenario;
    }
    if (d.source) draftedSource = d.source;
    else if (replace) draftedSource = null;
    syncJSON();
    const notes = d.warnings.map(message => ({level: "warning", message}));
    if (d.renamed && d.renamed.length) {
      notes.unshift({level: "note", message: "Real names in the source were replaced before drafting: "
        + d.renamed.map(r => `${r.real} → ${r.invented}`).join("; ") + "."});
    }
    showMessages(notes,
      `Drafted ${d.scenario ? "a scenario and " : ""}${plural(d.agents.length, "agent")}. Read them through and change anything that does not fit before creating the experiment.`);
  } catch (e) {
    showMessages([{level: "error", message: String(e.message || e)}], "Drafting failed.");
  }
  btn.disabled = false;
}

async function checkCast() {
  const ai = document.getElementById("check-ai").checked;
  const scenario = scenarioText();
  const btn = document.getElementById("check-btn");
  const cards = [...document.querySelectorAll(".agent-card")];
  cards.forEach(c => c.classList.remove("has-error", "has-warning"));
  btn.disabled = true;
  showMessages([], ai && scenario ? "Checking the cast, including an AI review against the scenario…"
                                  : "Checking the cast…");
  try {
    const r = await postJSON("/api/agents/check", {agents: collectAgents(), scenario, ai});
    const issues = (await r.json()).issues;
    if (ai && !scenario) issues.push({level: "suggestion", message: "Add a scenario to include the AI review."});
    issues.forEach(issue => {
      if (!issue.agent || issue.level === "suggestion") return;
      const card = cards.find(c => readCard(c).id === issue.agent);
      if (card) card.classList.add(issue.level === "error" ? "has-error" : "has-warning");
    });
    showMessages(issues, issues.length ? `Found ${plural(issues.length, "thing")} to look at.`
                                       : "No problems found.");
  } catch (e) {
    showMessages([{level: "error", message: String(e.message || e)}], "The check could not run.");
  }
  btn.disabled = false;
}
