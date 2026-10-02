// Research options for an agentic study (templates/_research_opts.html): show the library list
// and the save library when they apply, say whether the save library exists, and read the
// settings back.
function resShow(p) {
  const only = document.querySelector(`input[name="${p}_libmode"][value="only"]`).checked;
  document.getElementById(`${p}_libs_box`).hidden = !only;
  const outside = document.getElementById(`${p}_db`).checked || document.getElementById(`${p}_web`).checked;
  document.getElementById(`${p}_save`).hidden = !outside;
  netStatus(p);
}

function netFetchName(p) {
  return document.getElementById(`${p}_fetch`).value.trim().replace(/\s+/g, " ");
}

function netStatus(p) {
  const box = document.getElementById(`${p}_fetch`), note = document.getElementById(`${p}_fetch_status`);
  if (!box || !note) return;
  const libs = window.NET_LIBS || {};
  const typed = netFetchName(p);
  const name = typed || box.dataset.own || "";
  const docs = libs[name];
  if (!name) note.textContent = "Left empty, it gets a library of its own, named for the experiment.";
  else if (docs !== undefined)
    note.textContent = `Adds to the existing library “${name}” (${docs} document${docs === 1 ? "" : "s"}).`;
  else
    note.textContent = (typed ? "Starts" : "Left empty, it starts")
      + ` a new library, “${name}”, with the first document it reads.`;
}

// {choose_libraries, libraries, use_databases, use_web, fetch_library, time_limit_minutes}
function resValues(p) {
  const choose = document.querySelector(`input[name="${p}_libmode"][value="choose"]`).checked;
  const libs = [...document.querySelectorAll(`#${p}_libs input[type=checkbox]:checked`)].map(i => i.value);
  const n = parseInt(document.getElementById(`${p}_minutes`).value, 10);
  return {choose_libraries: choose, libraries: choose ? [] : libs,
          use_databases: document.getElementById(`${p}_db`).checked,
          use_web: document.getElementById(`${p}_web`).checked,
          fetch_library: netFetchName(p),
          time_limit_minutes: Math.max(5, Math.min(1440, isNaN(n) ? 120 : n))};
}
