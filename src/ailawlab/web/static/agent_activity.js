// Agent activity: an agentic run followed step by step on its run page. Each model call is
// a step (what the agent decided, how long it took, its reasoning on demand); each tool call
// says what came back; a meter shows how full the agent's context is, since the run
// answers once it passes the wrap-up share. Fed the events already written, then the
// trace stream (see run.html).
(function () {
  const root = document.getElementById("activity");
  if (!root) return;
  const CONTEXT = parseInt(root.dataset.context, 10) || 131072;
  const WRAP = parseFloat(root.dataset.wrap) || 0.85;
  const list = root.querySelector(".activity-list");
  const els = {
    step: root.querySelector("[data-step]"), elapsed: root.querySelector("[data-elapsed]"),
    status: root.querySelector("[data-status]"), fill: root.querySelector(".ctx-fill"),
    ctxText: root.querySelector("[data-ctx]"),
  };
  let live = root.dataset.live === "1";
  let steps = 0, started = null, lastAt = null, last = null, wrote = false;
  const agentSteps = {};          // research agent (Q1, Q2, … or lead) -> its steps
  const seen = new Set();

  const TOOL_NAMES = {search_libraries: "searched the libraries", search_online: "searched online",
                      read_online: "read online", search_databases: "searched the legal databases",
                      search_web: "searched the web", read: "read", calculate: "calculated"};
  let DATABASES = {};
  try { DATABASES = JSON.parse(root.dataset.databases || "{}"); } catch (e) { DATABASES = {}; }

  function when(s) { return s ? new Date(String(s).replace(" ", "T")) : null; }
  function secs(ms) { return ms >= 60000 ? `${Math.floor(ms / 60000)} min ${Math.round(ms % 60000 / 1000)} s`
                                         : `${(ms / 1000).toFixed(ms < 10000 ? 1 : 0)} s`; }
  function clock(ms) { const s = Math.max(0, Math.round(ms / 1000));
                       return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; }
  function el(tag, cls, text) { const n = document.createElement(tag); if (cls) n.className = cls;
                                if (text !== undefined) n.textContent = text; return n; }
  function quote(s) { return `“${String(s).trim()}”`; }

  // One line for a tool the agent asked for: what it is doing, with what.
  function intent(call) {
    const fn = call.function || {}, a = typeof fn.arguments === "string" ? safeJSON(fn.arguments) : (fn.arguments || {});
    switch (fn.name) {
      case "search_libraries":
        return `search ${a.library ? `the library ${quote(a.library)}` : "its libraries"} for ${quote(a.query || "")}`;
      case "search_web":
        return `search the web for ${quote(a.query || "")}`;
      case "search_databases":
      case "search_online":
        return `search ${DATABASES[a.database] || a.database || "online"} for ${quote(a.query || "")}`;
      case "read":
      case "read_online":
        return `read ${a.source || "a source"}` + (a.look_for ? `, looking for ${quote(a.look_for)}` : "");
      case "calculate":
        return `calculate ${a.expression || ""}`;
      default:
        return `use ${fn.name || "a tool"}`;
    }
  }
  function safeJSON(s) { try { return JSON.parse(s); } catch (e) { return {}; } }

  // What came back, in a line, with the titles it found.
  function outcome(tool, result) {
    const text = String(result || "");
    const firsts = re => [...text.matchAll(re)].map(m => m[1].trim());
    if (tool === "search_online" || tool === "search_databases" || tool === "search_web") {
      const hits = firsts(/^\[W\d+\]\s*(.+)$/gm);
      return {line: hits.length ? `${hits.length} result${hits.length === 1 ? "" : "s"}` : text.split("\n")[0],
              items: hits};
    }
    if (tool === "search_libraries") {
      const hits = firsts(/^\[\d+\]\s*(.+)$/gm);
      return {line: hits.length ? `${hits.length} passage${hits.length === 1 ? "" : "s"}` : text.split("\n")[0],
              items: [...new Set(hits.map(h => h.replace(/\s*\(library .*$/, "")))]};
    }
    if (tool === "read_online" || tool === "read") {
      const read = firsts(/Read (“[^”]+”)/g);
      return {line: read.length ? `read ${read.length} document${read.length === 1 ? "" : "s"} and saved ${read.length === 1 ? "it" : "them"}`
                                : text.split("\n")[0].slice(0, 200), items: read};
    }
    return {line: text.split("\n")[0].slice(0, 200), items: []};
  }

  function details(label, body, cls) {
    const d = el("details", cls);
    d.appendChild(el("summary", null, label));
    d.appendChild(el("div", "activity-more", body));
    return d;
  }

  function add(e) {
    if (seen.has(e.seq)) return;
    seen.add(e.seq);
    const at = when(e.created_at);
    if (!started && at) started = at;
    if (at) lastAt = at;
    last = e;
    const p = e.payload || {};
    let item = null;

    if (e.event_type === "llm_call") {
      steps += 1;
      const who = e.agent_id || "";
      if (who) agentSteps[who] = (agentSteps[who] || 0) + 1;
      const took = (e.eval_ms || 0) + (e.queue_wait_ms || 0);
      item = el("li", "activity-step");
      const head = el("div", "activity-head-line");
      head.appendChild(el("span", "activity-num", who ? `${who === "lead" ? "Lead" : who} · step ${agentSteps[who]}` : `Step ${steps}`));
      const calls = p.tool_calls || [];
      const what = calls.length ? "Decided to " + calls.map(intent).join("; then ")
                 : p.response ? `Wrote the answer (${p.response.length.toLocaleString()} characters)`
                 : "Produced no answer";
      if (!calls.length && p.response) wrote = true;
      head.appendChild(el("span", "activity-what", what));
      const meta = [took ? secs(took) : "", e.prompt_tokens ? `${Math.round(100 * e.prompt_tokens / CONTEXT)}% of context` : ""]
        .filter(Boolean).join(" · ");
      if (meta) head.appendChild(el("span", "activity-meta", meta));
      item.appendChild(head);
      if (e.thinking) item.appendChild(details("Reasoning", e.thinking, "activity-reasoning"));
      if (e.prompt_tokens) meter(e.prompt_tokens);
    } else if (e.event_type === "tool_call") {
      const o = outcome(p.tool, p.result);
      item = el("li", "activity-result");
      const head = el("div", "activity-head-line");
      head.appendChild(el("span", "activity-tool", (e.agent_id ? e.agent_id + " · " : "") + (TOOL_NAMES[p.tool] || p.tool || "tool")));
      head.appendChild(el("span", "activity-what", o.line));
      if (e.eval_ms) head.appendChild(el("span", "activity-meta", secs(e.eval_ms)));
      item.appendChild(head);
      if (o.items.length) {
        const ul = el("ul", "activity-items");
        o.items.slice(0, 12).forEach(t => ul.appendChild(el("li", null, t)));
        if (o.items.length > 12) ul.appendChild(el("li", "muted", `and ${o.items.length - 12} more`));
        item.appendChild(ul);
      }
      if (p.result) item.appendChild(details("What the agent was shown", String(p.result), "activity-reasoning"));
    } else if (e.event_type === "error") {
      item = el("li", "activity-note bad", p.message || "error");
    } else if (p.message && !/^run (started|completed)/.test(p.message)) {
      const who = e.agent_id && !p.message.startsWith(e.agent_id) ? `${e.agent_id}: ` : "";
      item = el("li", "activity-note", who + p.message);
    }
    if (item) list.appendChild(item);
    refresh();
  }

  function meter(tokens) {
    const share = tokens / CONTEXT;
    els.fill.style.width = `${Math.min(100, share * 100).toFixed(1)}%`;
    els.fill.classList.toggle("near", share >= WRAP * 0.8);
    els.ctxText.textContent = `${Math.round(share * 100)}% of the context used (${tokens.toLocaleString()} of `
      + `${CONTEXT.toLocaleString()} tokens). Past ${Math.round(WRAP * 100)}%, its next step answers.`;
  }

  function waiting() {
    if (!last) return "Starting…";
    if (last.event_type === "llm_call" && (last.payload || {}).tool_calls?.length) return "Running its tools";
    if (last.event_type === "note" && /read .* online/.test((last.payload || {}).message || "")) return "Reading sources";
    return wrote ? "Checking the answer" : "Thinking";
  }

  function refresh() {
    const lanes = Object.entries(agentSteps).filter(([k]) => k !== "lead");
    els.step.textContent = !steps ? "No steps yet"
      : lanes.length ? `${steps} steps · ` + lanes.map(([k, n]) => `${k}: ${n}`).join(" · ")
      : `Step ${steps}`;
    const now = live ? new Date() : lastAt;
    if (started && now) els.elapsed.textContent = clock(now - started);
    if (live) els.status.textContent = lastAt ? `${waiting()}… ${secs(Date.now() - lastAt)}` : waiting() + "…";
    else els.status.textContent = wrote ? "Answered" : "Finished";
  }

  window.agentActivity = {
    add,
    finish() { live = false; refresh(); },
  };
  (JSON.parse(document.getElementById("activity-data").textContent) || []).forEach(add);
  refresh();
  if (live) setInterval(refresh, 1000);
})();
