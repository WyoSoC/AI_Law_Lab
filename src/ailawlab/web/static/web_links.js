// Web links in a library: add pages by address, and check them later for changes.
// Used by the Legal Sources page and each library page; expects PREFIX to be defined.

const LinkUI = (() => {
  const esc = s => String(s ?? "").replace(/[&<>"]/g,
    c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
  const MARK = {added: '✓', updated: '↻', unchanged: '=', exists: '=', error: '✕', not_a_link: '–'};
  const WORD = {added: 'added', updated: 'updated', unchanged: 'no change', exists: 'already there',
                error: 'not added', not_a_link: 'not a link'};

  async function post(url, body) {
    const r = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'},
                                body: body ? JSON.stringify(body) : undefined});
    let d = {};
    try { d = await r.json(); } catch (e) { /* non-JSON error page */ }
    if (!r.ok) throw new Error(d.detail || `The server answered ${r.status}.`);
    return d;
  }

  function resultList(results) {
    return `<ul class="link-results">${results.map(r => `
      <li class="link-result ${esc(r.status)}">
        <span class="hit-mark" aria-hidden="true">${MARK[r.status] || '?'}</span>
        <span>
          ${r.document_id ? `<a href="${PREFIX}/sources/documents/${r.document_id}">${esc(r.title || r.link)}</a>`
                          : `<span class="link-url">${esc(r.link)}</span>`}
          <span class="link-status">${esc(WORD[r.status] || r.status)}</span>
          ${r.detail ? `<span class="hint">${esc(r.detail)}</span>` : ''}
        </span>
      </li>`).join('')}</ul>`;
  }

  // Read the links in `text` into `corpus`; show a per-link report in `box`.
  async function add(corpus, text, box, button) {
    if (!text.trim()) { box.innerHTML = '<p class="hint">Paste at least one web link.</p>'; return null; }
    if (!corpus) { box.innerHTML = '<p class="hint">Choose a library, or name a new one.</p>'; return null; }
    if (button) button.disabled = true;
    box.innerHTML = '<p class="hint">Reading the pages and building passages… a long page can take a minute.</p>';
    try {
      const d = await post(`${PREFIX}/api/sources/links`, {corpus, text});
      box.innerHTML = `<p class="cast-msg-head">${d.added} added to
        <a href="${PREFIX}/sources/corpora/${encodeURIComponent(d.corpus)}"><strong>${esc(d.corpus)}</strong></a>${d.version ? `, now version ${d.version}` : ''}.
        ${d.note ? esc(d.note) : ''}</p>` + resultList(d.results);
      return d;
    } catch (e) {
      box.innerHTML = `<p class="provider-note warn">${esc(e.message)}</p>`;
      return null;
    } finally {
      if (button) button.disabled = false;
    }
  }

  // Check one link document for changes; `cell` shows the outcome.
  async function check(documentId, cell, button) {
    if (button) button.disabled = true;
    cell.innerHTML = '<span class="hint">checking…</span>';
    try {
      const r = await post(`${PREFIX}/api/sources/documents/${documentId}/refresh`);
      cell.innerHTML = `<span class="link-result ${esc(r.status)}"><span class="hit-mark">${MARK[r.status] || '?'}</span>
        ${r.status === 'updated'
          ? `updated just now (<a href="${PREFIX}/sources/documents/${r.document_id}">new copy</a>${r.version ? `, library v${r.version}` : ''})`
          : esc(r.status === 'unchanged' ? 'checked just now: no change' : r.detail)}</span>`;
      return r;
    } catch (e) {
      cell.innerHTML = `<span class="provider-note warn">${esc(e.message)}</span>`;
      return null;
    } finally {
      if (button) button.disabled = false;
    }
  }

  // Check every link in a corpus; summary and per-link results go in `box`.
  async function checkAll(corpus, box, button) {
    if (button) button.disabled = true;
    box.innerHTML = '<p class="hint">Checking each link in turn…</p>';
    try {
      const d = await post(`${PREFIX}/api/sources/corpora/${encodeURIComponent(corpus)}/refresh`);
      const c = d.counts;
      const parts = [c.updated && `${c.updated} updated`, c.unchanged && `${c.unchanged} unchanged`,
                     c.error && `${c.error} could not be checked`].filter(Boolean);
      const version = d.results.map(r => r.version).filter(Boolean).pop();
      box.innerHTML = `<p class="cast-msg-head">Checked ${d.results.length} link${d.results.length === 1 ? '' : 's'}: ${parts.join(', ') || 'nothing to check'}.
        ${version ? `The library is now version ${version}; <a href="">reload</a> to see it.` : ''}</p>`
                      + resultList(d.results);
      return d;
    } catch (e) {
      box.innerHTML = `<p class="provider-note warn">${esc(e.message)}</p>`;
      return null;
    } finally {
      if (button) button.disabled = false;
    }
  }

  return {add, check, checkAll};
})();
