// main.js -- Instrumented PDF.js loader and telemetry bridge
let ws = null;
let sessionArxivId = null;

// Patch PDFViewerApplication to guarantee highlight event handler is present
document.addEventListener("DOMContentLoaded", function() {
  if (window.PDFViewerApplication) {
    window.PDFViewerApplication.handleAnnotationEditorStatesChanged = async function(event) {
      const { AnnotationType } = globalThis.pdfjsLib || {};
      const { type, value } = event.detail || {};
      if (type === 'add' && value && AnnotationType && value.annotationType === AnnotationType.HIGHLIGHT) {
        const highlightData = {
          event_type: 'highlight_created',
          timestamp: new Date().toISOString(),
          pageNumber: (typeof value.pageIndex === 'number') ? value.pageIndex + 1 : undefined,
          textContent: value.textContent || '',
          color: value.color,
          rects: value.rects,
          quadPoints: value.quadPoints,
          annotationId: value.id
        };
        console.log('Persistent Highlight Created Event:', highlightData);
        // Backend send placeholder
      }
    };
  }
});

function connectWS(arxiv_id) {
  ws = new WebSocket(`ws://${window.location.host}/ws?arxiv_id=${arxiv_id}`);
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
