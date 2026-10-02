// Network tools for an agentic run (templates/_network_opts.html): show the options when
// allowed, and say whether the save library already exists or will be started.
function netShow(p) {
  document.getElementById(`${p}_network_opts`).hidden = !document.getElementById(`${p}_network`).checked;
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

// The run settings the options describe: {allow_network, network_sources, fetch_library}.
function netValues(p) {
  const allowed = document.getElementById(`${p}_network`).checked;
  const n = parseInt(document.getElementById(`${p}_sources`).value, 10);
  return {allow_network: allowed,
          network_sources: Math.max(1, Math.min(20, isNaN(n) ? 5 : n)),
          fetch_library: netFetchName(p)};
}
