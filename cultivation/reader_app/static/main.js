// main.js -- Instrumented PDF.js loader and telemetry bridge
// WebSocket is now opened automatically on PDF load and will auto-reconnect unless the PDF is manually reloaded.
let ws = null;
let sessionArxivId = null;

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

// Populate paper dropdown on load
window.addEventListener('DOMContentLoaded', async function() {
  const select = document.getElementById('paperSelect');
  select.innerHTML = '<option>Loading...</option>';
  try {
    const resp = await fetch('/papers/list');
    const papers = await resp.json();
    select.innerHTML = '';
    for (const paper of papers) {
      const opt = document.createElement('option');
      opt.value = paper.arxiv_id;
      opt.textContent = paper.title + ' [' + paper.arxiv_id + ']';
      select.appendChild(opt);
    }
  } catch (e) {
    select.innerHTML = '<option>Error loading papers</option>';
  }
});

// Load PDF by selected paper
document.getElementById('loadBtn').addEventListener('click', async function() {
  const select = document.getElementById('paperSelect');
  const arxivId = select.value;
  if (!arxivId) {
    setStatus('Please select a paper.', 'error');
    return;
  }
  await loadPDF(arxivId);
});

async function loadPDF(arxivId) {
  sessionArxivId = arxivId;
  document.getElementById('viewer').src = `/static/pdfjs/viewer.html?file=/pdfs/${arxivId}.pdf`;
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
