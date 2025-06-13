// main.js -- Instrumented PDF.js loader and telemetry bridge
// WebSocket is now opened automatically on PDF load and will auto-reconnect unless the PDF is manually reloaded.
let ws = null;
let sessionArxivId = null;

// Status helper
function setStatus(msg, cls) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = cls || '';
}

// Patch PDFViewerApplication to guarantee highlight event handler is present
document.addEventListener("DOMContentLoaded", function() {
  if (window.PDFViewerApplication) {
    const originalHandler = window.PDFViewerApplication.handleAnnotationEditorStatesChanged?.bind(window.PDFViewerApplication);
    window.PDFViewerApplication.handleAnnotationEditorStatesChanged = function(event) {
      // Preserve original PDF.js behavior
      if (typeof originalHandler === 'function') {
        originalHandler(event);
      }
      if (
        event &&
        event.source &&
        event.source.annotationType === 8 &&
        event.states &&
        event.states.includes('persistent')
      ) {
        const value = event.source;
        const highlightData = {
          type: 'highlight',
          page: value.pageIndex + 1,
          color: value.color,
          quadPoints: value.quadPoints,
          text: value.contents,
          annotationId: value.id
        };
        console.log('Persistent Highlight Created Event:', highlightData);
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify(highlightData));
        }
      }
    };
  }
});

function connectWS(arxiv_id) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.close(1000, 'reloading');
  }
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  function doConnect() {
    console.log('[WS] Attempting connection for', arxiv_id);
    ws = new WebSocket(`${proto}://${window.location.host}/ws?arxiv_id=${encodeURIComponent(arxiv_id)}`);
    ws.onopen = () => {
      console.log('[WS] Connected');
      document.getElementById('status').textContent = 'WebSocket connected';
    };
    ws.onclose = (event) => {
      console.log('[WS] Closed', event);
      document.getElementById('status').textContent = 'WebSocket closed';
      // If not a manual reload, auto-reconnect after 1s
      if (event.code !== 1000 || event.reason !== 'reloading') {
        console.log('[WS] Reconnecting in 1s...');
        setTimeout(doConnect, 1000);
      }
    };
    ws.onerror = (e) => {
      console.error('[WS] Error', e);
      document.getElementById('status').textContent = 'WebSocket error';
    };
  }
  doConnect();
}

// Store paper progress globally
let paperProgress = {};

// Populate paper dropdown (refactored for reuse)
async function refreshPaperDropdown() {
  const select = document.getElementById('paperSelect');
  select.innerHTML = '<option>Loading...</option>';
  try {
    const [papersResp, progressResp] = await Promise.all([
      fetch('/papers/list'),
      fetch('/papers/progress')
    ]);
    const papers = await papersResp.json();
    const progressList = await progressResp.json();
    // Build progress lookup: arxiv_id -> progress object
    paperProgress = {};
    for (const p of progressList) {
      paperProgress[p.arxiv_id] = p;
    }
    // Optionally sort papers by completion percentage (incomplete first)
    papers.sort((a, b) => {
      const compA = paperProgress[a.arxiv_id]?.completion ?? 0;
      const compB = paperProgress[b.arxiv_id]?.completion ?? 0;
      return compA - compB;
    });
    select.innerHTML = '';
    for (const paper of papers) {
      const prog = paperProgress[paper.arxiv_id];
      const opt = document.createElement('option');
      opt.value = paper.arxiv_id;
      opt.textContent = paper.title + ' [' + paper.arxiv_id + ']';
      if (prog && prog.completion !== null && prog.completion !== undefined) {
        opt.textContent += ` (${prog.completion}% read)`;
      }
      select.appendChild(opt);
    }
  } catch (e) {
    select.innerHTML = '<option>Error loading papers</option>';
  }
}
window.addEventListener('DOMContentLoaded', refreshPaperDropdown);

// Add Paper logic
const addArxivBtn = document.getElementById('addArxivBtn');
const addArxivInput = document.getElementById('addArxivInput');
addArxivBtn.addEventListener('click', async function() {
  const arxivId = addArxivInput.value.trim();
  if (!arxivId) {
    setStatus('Please enter an arXiv ID.', 'error');
    return;
  }
  setStatus('Adding paper...');
  try {
    // Assume backend: GET /metadata/{arxiv_id} triggers fetch/add if not present
    const resp = await fetch(`/metadata/${arxivId}`);
    if (resp.ok) {
      setStatus('Paper added or already exists.');
      await refreshPaperDropdown();
      addArxivInput.value = '';
    } else {
      setStatus('Failed to add paper: ' + (await resp.text()), 'error');
    }
  } catch (e) {
    setStatus('Error adding paper: ' + e, 'error');
  }
});

// Load PDF by selected paper
// Use radio buttons to determine mode
// Remove old checkbox/resume button logic

document.getElementById('loadBtn').addEventListener('click', async function() {
  const select = document.getElementById('paperSelect');
  const arxivId = select.value;
  if (!arxivId) {
    setStatus('Please select a paper.', 'error');
    return;
  }
  const resumeRadio = document.getElementById('resumeRadio');
  const prog = paperProgress[arxivId];
  if (resumeRadio.checked && prog && prog.last_page) {
    await loadPDF(arxivId, prog.last_page);
    setStatus(`Resumed at page ${prog.last_page}`);
  } else {
    await loadPDF(arxivId);
    setStatus('Loaded from first page');
  }
});


async function loadPDF(arxivId, page=null) {
  sessionArxivId = arxivId;
  // Always reload iframe
  document.getElementById('viewer').src = `/static/pdfjs/viewer.html?file=/pdfs/${arxivId}.pdf`;
  // If page provided, jump after iframe loads
  if (page) {
    document.getElementById('viewer').addEventListener('load', function handler() {
      document.getElementById('viewer').removeEventListener('load', handler);
      // PDF.js expects 1-based page numbers
      document.getElementById('viewer').contentWindow.postMessage(JSON.stringify({event_type: 'jump_to_page', payload: {page}}), '*');
    });
  }
}

document.getElementById('viewer').addEventListener('load', () => {
  if (sessionArxivId) {
    connectWS(sessionArxivId);
  }
});

document.getElementById('finishBtn').onclick = async () => {
  if (!sessionArxivId || !ws || ws.readyState !== WebSocket.OPEN) {
    alert('No active session to finish.');
    return;
  }
  // Prompt for metrics
  const comprehension = window.prompt('Self-rated comprehension (0-5)?');
  if (comprehension === null) return;
  const relevance = window.prompt('Self-rated relevance (0-5)?');
  if (relevance === null) return;
  const novelty = window.prompt('Self-rated novelty personal (0-1)?');
  if (novelty === null) return;
  const timeSpent = window.prompt('Actual time spent (minutes)?');
  if (timeSpent === null) return;
  const metrics = {
    self_rated_comprehension: parseFloat(comprehension),
    self_rated_relevance: parseFloat(relevance),
    self_rated_novelty_personal: parseFloat(novelty),
    actual_time_spent_minutes: parseFloat(timeSpent)
  };
  // POST to backend
  try {
    const resp = await fetch('/finish_session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ arxiv_id: sessionArxivId, metrics })
    });
    if (resp.ok) {
      ws.close(1000, 'finished');
      document.getElementById('status').textContent = 'Session finished';
      alert('Session finished and metrics logged.');
    } else {
      alert('Failed to finish session.');
    }
  } catch (e) {
    alert('Error finishing session: ' + e);
  }
};

// Telemetry bridge: inject into PDF.js viewer iframe
window.addEventListener('message', (event) => {
  // Debug: log all messages from iframe
  console.log('parent received message:', event.data);
  if (!ws || ws.readyState !== 1) return;
  // Expect PDF.js events as {event_type, payload}
  try {
    const msg = JSON.parse(event.data);
    if (msg.event_type) {
      console.log('sending to ws:', msg);
      ws.send(JSON.stringify(msg));
    }
  } catch (e) { console.warn('Failed to parse event.data', event.data, e); }
}, false);
