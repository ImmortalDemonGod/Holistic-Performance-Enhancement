/**
 * Sends a message to the parent window when a PDF.js text layer is rendered.
 *
 * Used in testing environments to simulate the PDF.js viewer bridge by posting a message containing the rendered page number.
 *
 * @param {Object} event - The event object from the text layer rendered event, expected to have a `detail.pageNumber` property.
 *
 * @remark
 * The message is only sent if `window.parent.postMessage` is available. The page number defaults to 1 if not specified in the event detail.
 */

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
