// main.js -- Instrumented PDF.js loader and telemetry bridge
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
  ws = new WebSocket(`${proto}://${window.location.host}/ws?arxiv_id=${encodeURIComponent(arxiv_id)}`);
  ws.onopen = () => {
    document.getElementById('status').textContent = 'WebSocket connected';
  };
  ws.onclose = () => {
    document.getElementById('status').textContent = 'WebSocket closed';
  };
  ws.onerror = (e) => {
    document.getElementById('status').textContent = 'WebSocket error';
  };
}

document.getElementById('loadBtn').onclick = () => {
  const arxiv_id = document.getElementById('arxiv_id').value.trim();
  if (!arxiv_id) return;
  sessionArxivId = arxiv_id;
  document.getElementById('viewer').src = `/static/pdfjs/viewer.html?file=/pdfs/${arxiv_id}.pdf`;
  connectWS(arxiv_id);
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
