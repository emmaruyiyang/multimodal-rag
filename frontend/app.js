// ── Config ──────────────────────────────────────────────────────────────────
const API = '';  // same origin — FastAPI serves both API and frontend
// Register at https://developer.adobe.com/document-services/apis/pdf-embed/
const PDF_CLIENT_ID = 'b9a034b86e5c41ce806f246b734bfffc';

// ── State ────────────────────────────────────────────────────────────────────
let currentDoc = null;
let pdfAPIs    = null;

// ── Adobe SDK ready ──────────────────────────────────────────────────────────
function whenSDKReady(cb) {
  if (window.AdobeDC) { cb(); }
  else { document.addEventListener('adobe_dc_view_sdk.ready', cb, { once: true }); }
}

function loadPDF(docName) {
  currentDoc = docName;
  pdfAPIs = null;

  document.getElementById('pdf-empty').style.display = 'none';

  // Replace the viewer div so AdobeDC.View can re-init cleanly
  const panel = document.getElementById('pdf-panel');
  const old = document.getElementById('pdf-viewer');
  const fresh = document.createElement('div');
  fresh.id = 'pdf-viewer';
  panel.replaceChild(fresh, old);

  whenSDKReady(() => {
    const view = new AdobeDC.View({ clientId: PDF_CLIENT_ID, divId: 'pdf-viewer' });
    view.previewFile(
      {
        content:  { location: { url: `${API}/api/pdf/${docName}` } },
        metaData: { fileName: `${docName}.pdf` },
      },
      {
        embedMode: 'FULL_WINDOW',
        defaultViewMode: 'FIT_PAGE',
        showAnnotationTools: false,
        showLeftHandPanel: false,
        enableSearchAPIs: true,
      }
    ).then(viewer => {
      viewer.getAPIs().then(apis => {
        pdfAPIs = apis;
        window.pdfAPIs = apis;  // for manual console testing
      });
    });
  });
}

let lastSearch = null;

// Locate a source in the viewer: highlight its text if we have a snippet,
// otherwise just jump to its page (used for figures).
async function locateSource(pageNum, snippet) {
  if (!pdfAPIs) return;
  try {
    if (lastSearch) { await lastSearch.clear(); lastSearch = null; }
  } catch (_) {}

  if (snippet) {
    try {
      lastSearch = await pdfAPIs.search(snippet);
      return;
    } catch (e) {
      console.warn('[PDF] search failed, falling back to page jump:', e);
    }
  }
  if (pageNum) {
    try { await pdfAPIs.gotoLocation(Number(pageNum)); }
    catch (e) { console.error('[PDF] navigation error:', e); }
  }
}

// ── Document list ────────────────────────────────────────────────────────────
async function loadDocList() {
  const docs = await fetch(`${API}/api/documents`).then(r => r.json());
  const sel = document.getElementById('doc-select');
  const prev = sel.value;
  sel.innerHTML = '<option value="">Select document…</option>';
  docs.forEach(d => {
    const o = document.createElement('option');
    o.value = o.textContent = d;
    sel.appendChild(o);
  });
  if (prev && docs.includes(prev)) sel.value = prev;
}

document.getElementById('doc-select').addEventListener('change', e => {
  if (e.target.value) loadPDF(e.target.value);
});

// ── Upload & index ────────────────────────────────────────────────────────────
document.getElementById('pdf-upload').addEventListener('change', async e => {
  const file = e.target.files[0];
  if (!file) return;
  e.target.value = '';

  const status = document.getElementById('index-status');
  status.textContent = 'Indexing… (may take 1-2 min)';
  status.className = 'busy';

  const form = new FormData();
  form.append('file', file);

  try {
    const res  = await fetch(`${API}/api/documents/index`, { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Index failed');

    status.textContent = `✓ ${data.text_chunks} text + ${data.image_chunks} image chunks`;
    status.className = 'ok';
    await loadDocList();
    document.getElementById('doc-select').value = data.doc_name;
    loadPDF(data.doc_name);
  } catch (err) {
    status.textContent = `✗ ${err.message}`;
    status.className = 'err';
  }
});

// ── Chat ──────────────────────────────────────────────────────────────────────
function addMsg(role, text, sources) {
  const box = document.getElementById('messages');
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  wrap.appendChild(bubble);

  if (sources && sources.length) {
    const row = document.createElement('div');
    row.className = 'sources';
    sources.forEach(s => {
      const page = s.pages && s.pages[0];
      const clickable = page || s.snippet;
      const chip = document.createElement('button');
      chip.className = 'src-chip' + (clickable ? '' : ' no-page');
      chip.title = s.preview || '';
      chip.textContent = [
        page ? `p.${page}` : null,
        s.type,
        `${s.score}`,
      ].filter(Boolean).join(' · ');
      if (clickable) chip.addEventListener('click', () => locateSource(page, s.snippet));
      row.appendChild(chip);
    });
    wrap.appendChild(row);
  }

  box.appendChild(wrap);
  box.scrollTop = box.scrollHeight;
  return wrap;
}

async function askQuestion() {
  const input = document.getElementById('q-input');
  const q = input.value.trim();
  if (!q) return;
  if (!currentDoc) { alert('Select a document first.'); return; }

  input.value = '';
  document.getElementById('ask-btn').disabled = true;

  addMsg('user', q);
  const loading = addMsg('assistant', '…');
  loading.querySelector('.bubble').classList.add('loading');

  try {
    const res  = await fetch(`${API}/api/documents/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, doc_name: currentDoc, top_k: 5 }),
    });
    const data = await res.json();
    loading.remove();
    if (!res.ok) throw new Error(data.detail || 'Query failed');
    addMsg('assistant', data.answer, data.sources);
  } catch (err) {
    loading.remove();
    addMsg('assistant', `Error: ${err.message}`);
  } finally {
    document.getElementById('ask-btn').disabled = false;
    input.focus();
  }
}

document.getElementById('ask-btn').addEventListener('click', askQuestion);
document.getElementById('q-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); askQuestion(); }
});

// ── Init ──────────────────────────────────────────────────────────────────────
loadDocList();
