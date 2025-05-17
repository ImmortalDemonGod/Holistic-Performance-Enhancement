// Minimal PDF.js Viewer Bridge module stub for testing

function handleTextLayerRendered(event) {
  // Simulate the postMessage API as expected by tests
  if (typeof window !== 'undefined' && window.parent && window.parent.postMessage) {
    const detail = event.detail || {};
    window.parent.postMessage({
      type: 'textlayerrendered',
      page: detail.pageNumber || 1
    }, '*');
  }
}

module.exports = {
  handleTextLayerRendered
};
