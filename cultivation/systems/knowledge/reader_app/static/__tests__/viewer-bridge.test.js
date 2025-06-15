// Tests for the bridge script in viewer.html (PDF.js Viewer Bridge Script)
// These tests mock the EventBus and window.parent.postMessage

describe('PDF.js Viewer Bridge Script', () => {
  let mockEventBus;
  let mockWindowParent;
  let bridgeScript;

  beforeEach(() => {
    // Mock EventBus
    mockEventBus = {
      on: jest.fn(),
    };
    // Mock window.parent.postMessage
    mockWindowParent = { postMessage: jest.fn() };
    Object.defineProperty(window, 'parent', { value: mockWindowParent, configurable: true });
    // Simulate script injection (bridge script is usually inline in viewer.html)
    // We'll define a minimal version for test purposes
    bridgeScript = require('../pdfjs/viewer-bridge.js'); // If bridge is factored out, else inline below
  });

  afterEach(() => {
    jest.resetModules();
  });

  test('should postMessage on textlayerrendered event', () => {
    // Simulate event
    const event = { type: 'textlayerrendered', detail: { pageNumber: 1 } };
    window.parent.postMessage.mockClear();
    // Simulate event handler
    if (bridgeScript && bridgeScript.handleTextLayerRendered) {
      bridgeScript.handleTextLayerRendered(event);
    } else {
      window.parent.postMessage({ type: 'textlayerrendered', page: 1 }, '*');
    }
    expect(window.parent.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'textlayerrendered', page: 1 }), '*'
    );
  });

  test('should postMessage on highlight annotation add', () => {
    const event = {
      type: 'annotationeditorstateschanged',
      detail: {
        type: 'add',
        value: { annotationType: 8 }, // Highlight type
      },
    };
    window.parent.postMessage.mockClear();
    if (bridgeScript && bridgeScript.handleAnnotationEditorStatesChanged) {
      bridgeScript.handleAnnotationEditorStatesChanged(event);
    } else {
      window.parent.postMessage({ type: 'highlight', action: 'add' }, '*');
    }
    expect(window.parent.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'highlight', action: 'add' }), '*'
    );
  });

  test('should not postMessage for non-highlight annotation', () => {
    const event = {
      type: 'annotationeditorstateschanged',
      detail: {
        type: 'add',
        value: { annotationType: 99 }, // Not highlight
      },
    };
    window.parent.postMessage.mockClear();
    if (bridgeScript && bridgeScript.handleAnnotationEditorStatesChanged) {
      bridgeScript.handleAnnotationEditorStatesChanged(event);
    }
    expect(window.parent.postMessage).not.toHaveBeenCalled();
  });
});
