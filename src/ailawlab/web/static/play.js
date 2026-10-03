// A person's seat in a live role-play (play.html). Polls the seat every two seconds and
// redraws what changed; never touches what the person is typing.

const $ = id => document.getElementById(id);
const AGENT = ROLES[0];
let last = null, shownTurns = -1, shownExhibits = -1, profileFilled = false, busy = false;

function esc(s) { const d = document.createElement("div"); d.textContent = s ?? ""; return d.innerHTML; }

// Paragraphs, with [S1] and [E1] markers picked out. `markers` turns the speaker's own
// numbers into the exhibits they became, so [S7] reads as [E1].
function prose(text, markers) {
  return esc(text).split(/\n{2,}/).map(p => `<p>${p.replace(/\n/g, "<br>")
    .replace(/\[(S|E)(\d+)\]/g, (m, kind, n) => {
      const shown = (markers || {})[kind + n] || kind + n;
      return `<span class="cite">[${shown}]</span>`;
    })}</p>`).join("");
}

function words(text) { return (text.trim().match(/\S+/g) || []).length; }

async function api(path, body) {
  const r = await fetch(`${PREFIX}/api/runs/${RUN_ID}/seat${path}`, body === undefined
    ? {cache: "no-store"}
    : {method: "POST", headers: {"Content-Type": "application/json"},
       body: JSON.stringify({agent: AGENT, ...body})});
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `The server answered ${r.status}.`);
  return data;
}

function colours(v) {
  const order = [v.agent_id, ...v.others.map(o => o.id)];
  return id => (order.indexOf(id) % 8) + 1;
}

function drawStatus(v) {
  const box = $("status");
  let text = "", cls = "";
  if (v.phase === "starting") text = "The run is starting…";
  else if (v.phase === "joining") {
    const waiting = v.others.filter(o => o.person && !o.ready).map(o => o.name);
    text = v.ready
      ? (waiting.length ? `You are ready. Waiting for ${waiting.join(", ")}.` : "Everyone is ready. Starting…")
      : "Fill in your profile, then press “I'm ready”. The exchange starts when everyone is ready.";
  } else if (v.phase === "running") {
    if (v.your_turn) { text = "It is your turn."; cls = "yours"; }
    else if (v.speaking) text = `${v.speaking} is speaking…`;
    else text = "The moderator is choosing who speaks next…";
  } else if (v.phase === "assessing") text = "The exchange is over. The assessor is writing up the outcome…";
  else if (v.phase === "done") {
    box.innerHTML = `The run has ended. <a href="${PREFIX}/runs/${RUN_ID}">Open the full record</a>, with everyone's private notes and the assessment.`;
    box.className = "seat-status done";
    document.querySelector(".stop-form")?.remove();
    return;
  }
  if (v.stopped && v.phase !== "done") text = "The role-play has been ended. " + text;
  box.textContent = text;
  box.className = "seat-status " + cls;
}

function drawProfile(v) {
  const box = $("profile-box");
  box.hidden = v.phase !== "joining";
  if (box.hidden) return;
  if (!profileFilled) {
    document.querySelectorAll("[data-pf]").forEach(el => { el.value = v.profile[el.dataset.pf] || ""; });
    $("pf-files").textContent = v.case_files.length
      ? `Your case files: ${v.case_files.join(", ")}. Only you can search them.` : "";
    profileFilled = true;
  }
  document.querySelectorAll("[data-pf]").forEach(el => { el.disabled = v.ready; });
  $("ready-btn").hidden = v.ready;
  $("ready-note").textContent = v.ready ? "Your profile is set." : "";
}

function drawSetup(v) {
  $("scenario").textContent = v.scenario;
  const c = colours(v);
  const me = {id: v.agent_id, name: v.profile.name || "You", role: v.profile.role, person: true};
  $("roster").innerHTML = [me, ...v.others].map(o => `<li class="c${c(o.id)}"><span class="turn-name">${esc(o.name)}</span>
    ${o.role ? `<span class="hint-inline">${esc(o.role)}</span>` : ""}
    <span class="lib-tag">${o.id === v.agent_id ? "you" : o.person ? "a person" : "AI"}</span></li>`).join("");
}

function drawTranscript(v) {
  if (v.transcript.length === shownTurns) return;
  shownTurns = v.transcript.length;
  const c = colours(v);
  $("transcript").innerHTML = v.transcript.map(t => t.agent_id === "moderator"
    ? `<div class="turn moderator"><div class="turn-head"><span class="turn-name">Moderator</span>
         <span class="turn-meta">${esc(t.role)}</span></div><div class="turn-body md">${prose(t.content)}</div></div>`
    : `<article class="turn c${c(t.agent_id)}"><div class="turn-head">
         <span class="turn-name">${esc(t.name)}${t.agent_id === v.agent_id ? " (you)" : ""}</span>
         <span class="turn-meta">${t.role ? esc(t.role) + " · " : ""}turn ${t.turn} · ${words(t.content)} words</span></div>
         <div class="turn-body md">${prose(t.content, t.markers)}</div>
         ${(t.disclosed || []).length ? `<p class="disclosed-note">Disclosed ${t.disclosed.map(m => `[${m}]`).join(", ")}.</p>` : ""}
       </article>`).join("");
  $("transcript-empty").hidden = v.transcript.length > 0;
  if (v.transcript.length) $("transcript").lastElementChild.scrollIntoView({block: "nearest", behavior: "smooth"});
}

function drawExhibits(v) {
  if (v.exhibits.length === shownExhibits) return;
  shownExhibits = v.exhibits.length;
  $("exhibits-head").hidden = $("exhibits").hidden = !v.exhibits.length;
  $("exhibits").innerHTML = v.exhibits.map(e => `<li><span class="cite">[${esc(e.marker)}]</span> ${esc(e.label)}
    <span class="hint">disclosed by ${esc(e.name)} in turn ${e.turn}</span>
    <details><summary>Passage</summary><div class="thinking">${esc(e.content)}</div></details></li>`).join("");
}

const PLACES = {own: "My case file", shared: "Shared legal sources"};

function drawTurn(v) {
  const box = $("turn-box");
  const was = !box.hidden;
  box.hidden = !v.your_turn;
  if (!v.your_turn) return;
  $("turn-no").textContent = `turn ${v.turn}`;
  $("directive").hidden = !v.directive;
  $("directive").textContent = v.directive ? `From the moderator: ${v.directive}` : "";
  $("search-box").hidden = !v.can_search;
  const where = $("search-where");
  if (where.options.length !== v.places.length)
    where.innerHTML = v.places.map(p => `<option value="${p}">${PLACES[p]}</option>`).join("");
  drawFound(v.found);
  countWords(v);
  if (!was) { box.scrollIntoView({block: "start", behavior: "smooth"}); $("reply").focus(); }
}

function drawFound(found) {
  $("found").innerHTML = found.map(f => `<li><span class="cite">[${esc(f.marker)}]</span> ${esc(f.label)}
    ${f.private ? '<span class="private-tag">your case file</span>' : ""}
    <button type="button" class="small-btn" data-cite="${esc(f.marker)}">Cite</button>
    <div class="thinking">${esc(f.content)}</div></li>`).join("");
}

function countWords(v) {
  const n = words($("reply").value), limit = (v || last)?.word_limit || 0;
  $("word-count").textContent = limit ? `${n} of about ${limit} words` : `${n} words`;
  $("word-count").classList.toggle("over", limit && n > limit * 1.15);
}

function drawDeadline() {
  const v = last;
  if (!v || !v.your_turn || !v.deadline) { $("deadline").textContent = ""; return; }
  const left = Math.max(0, Math.round(v.deadline - Date.now() / 1000));
  $("deadline").textContent = `· ${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")} left to reply`;
}

function draw(v) {
  last = v;
  drawStatus(v);
  if (v.phase === "starting" || v.phase === "done") return;
  drawProfile(v);
  drawSetup(v);
  drawTranscript(v);
  drawExhibits(v);
  drawTurn(v);
  drawDeadline();
}

async function poll() {
  try {
    const v = await api("");
    draw(v);
    if (v.phase === "done") return;
  } catch (e) {
    $("status").textContent = e.message;
  }
  setTimeout(poll, 2000);
}

function profile() {
  const out = {};
  document.querySelectorAll("[data-pf]").forEach(el => { out[el.dataset.pf] = el.value; });
  return out;
}

$("ready-btn").addEventListener("click", async () => {
  if (!$("pf-name").value.trim()) { $("ready-note").textContent = "Give your character a name first."; return; }
  try { draw(await api("/ready", {profile: profile()})); }
  catch (e) { $("ready-note").textContent = e.message; }
});

$("reply").addEventListener("input", () => countWords());

$("speak-btn").addEventListener("click", async () => {
  if (busy) return;
  const text = $("reply").value.trim();
  if (!text) { $("word-count").textContent = "Write what you want to say first."; return; }
  busy = true; $("speak-btn").disabled = true;
  try {
    draw(await api("/speak", {text}));
    $("reply").value = "";
  } catch (e) {
    $("word-count").textContent = e.message;
  } finally {
    busy = false; $("speak-btn").disabled = false;
  }
});

async function search() {
  const query = $("search-q").value.trim();
  if (!query) return;
  $("search-btn").disabled = true;
  try {
    const r = await api("/search", {query, where: $("search-where").value});
    drawFound(r.found);
    const note = r.hits.length ? `Found ${r.hits.join(", ")}.` : "Nothing relevant found there.";
    $("word-count").textContent = r.exhibits.length ? `${note} ${r.exhibits.join(", ")} already on the record.` : note;
  } catch (e) {
    $("word-count").textContent = e.message;
  } finally {
    $("search-btn").disabled = false;
  }
}

$("search-btn").addEventListener("click", search);
$("search-q").addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); search(); } });

// "Cite" puts the marker where the person is typing.
$("found").addEventListener("click", e => {
  const marker = e.target.dataset?.cite;
  if (!marker) return;
  const box = $("reply"), at = box.selectionStart ?? box.value.length;
  box.value = box.value.slice(0, at) + `[${marker}]` + box.value.slice(box.selectionEnd ?? at);
  box.focus();
  countWords();
});

setInterval(drawDeadline, 1000);
poll();
