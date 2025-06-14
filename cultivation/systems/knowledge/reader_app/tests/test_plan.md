Okay, let's devise a comprehensive plan for testing the JavaScript components of your instrumented PDF viewer (`cultivation/reader_app/`). This will focus on `static/main.js` and the bridge script within `static/pdfjs/viewer.html`.

The goal is to ensure:
1.  `main.js` correctly initializes the PDF.js viewer, manages WebSocket communication, and relays events.
2.  The bridge script in `viewer.html` accurately captures PDF.js events and `postMessage`s them to `main.js`.

We'll primarily use **Jest** as the testing framework, leveraging **JSDOM** for DOM interactions and Jest's mocking capabilities. For WebSocket testing, `jest-websocket-mock` is a good choice.

**I. Project Setup for JavaScript Testing**

1.  **Install Dependencies:**
    ```bash
    # In your project's root, or in cultivation/reader_app/ if you want to isolate JS tests
    npm init -y # if no package.json exists
    npm install --save-dev jest jest-environment-jsdom jest-websocket-mock # For WebSocket mocking
    # (Optional) If you need to transpile modern JS for Jest (e.g., ES modules in viewer.mjs)
    npm install --save-dev babel-jest @babel/core @babel/preset-env
    ```

2.  **Jest Configuration (`jest.config.js` in `cultivation/reader_app/` or project root):**
    ```javascript
    module.exports = {
      testEnvironment: 'jest-environment-jsdom',
      setupFilesAfterEnv: ['./jest.setup.js'], // Optional, for global mocks/setup
      // If you have ES modules or specific syntax:
      transform: {
        '^.+\\.m?js$': 'babel-jest',
      },
      moduleNameMapper: {
        // If you have module aliases or need to mock static assets
      },
      roots: ["<rootDir>/static"], // Point to where your JS files are
      testMatch: [ // Where to find test files
        "**/__tests__/**/*.js",
        "**/?(*.)+(spec|test).js"
      ],
      verbose: true,
    };
    ```

3.  **Babel Configuration (`babel.config.js` if using Babel):**
    ```javascript
    module.exports = {
      presets: [['@babel/preset-env', {targets: {node: 'current'}}]],
    };
    ```

4.  **`jest.setup.js` (Optional, in `cultivation/reader_app/`):**
    ```javascript
    // Global mocks or setup, e.g., mocking window.parent.postMessage for viewer.html tests
    // global.someGlobalMock = jest.fn();
    ```

5.  **Test File Structure:**
    Create `__tests__` directories alongside your JS files or in a dedicated test root.
    ```
    cultivation/reader_app/static/
    ├── __tests__/
    │   ├── main.test.js
    │   └── viewer-bridge.test.js
    ├── main.js
    └── pdfjs/
        └── viewer.html
    ```

**II. Testing `static/main.js`**

*   **HTML Fixture:** Create a simplified HTML structure in your tests to mimic `index.html`.
*   **Mocking:**
    *   `WebSocket`: Use `jest-websocket-mock`.
    *   `iframe` and its `contentWindow.postMessage`.
    *   `pdfjsLib.GlobalWorkerOptions` if `main.js` touches it directly (though it seems it's more in `viewer.html`).

**`static/__tests__/main.test.js`:**

```javascript
import { WS } from 'jest-websocket-mock';
// Load main.js after setting up mocks and DOM
// To do this, you might need to export functions from main.js or load it dynamically in test.

// --- Mocking global objects and DOM before main.js is imported ---
let mockIframeSrc;
let mockStatusText;
const mockViewer = {
  set src(val) { mockIframeSrc = val; },
  get src() { return mockIframeSrc; }
};
const mockArxivIdInput = { value: '' };
const mockLoadBtn = { onclick: null, addEventListener: jest.fn((event, cb) => { if(event==='click') mockLoadBtn.onclick = cb; }) };
const mockStatusElement = { 
    set textContent(val) { mockStatusText = val; },
    get textContent() { return mockStatusText; }
};

// JSDOM setup
document.body.innerHTML = `
  <div id="toolbar">
    <input type="text" id="arxiv_id" />
    <button id="loadBtn">Load PDF</button>
    <span id="status"></span>
  </div>
  <iframe id="viewer"></iframe>
`;

// Mock window.parent for the message event listener (if main.js adds it to window)
// global.parent = { postMessage: jest.fn() }; // Not needed for main.js, but for viewer.html

// Store original WebSocket
const OriginalWebSocket = global.WebSocket;

describe('main.js', () => {
  let server;
  let mainJsModule;

  beforeEach(async () => {
    // Reset DOM mocks
    mockArxivIdInput.value = '';
    mockIframeSrc = '';
    mockStatusText = '';
    document.getElementById('arxiv_id').value = ''; // Reset JSDOM input
    document.getElementById('viewer').src = ''; // Reset JSDOM iframe

    // Setup Jest's fake timers
    jest.useFakeTimers();

    // Dynamically import main.js AFTER mocks are set up
    // This requires main.js to be structured to allow this (e.g. not self-executing global listeners immediately)
    // Or, ensure main.js adds listeners that can be triggered.
    // For simplicity, we'll assume main.js attaches its listeners on load.
    // If main.js is simple IIFE, you might need to load it via <script> in JSDOM
    // or refactor main.js to export its functions.
    
    // For now, let's assume main.js can be imported and re-initializes event listeners
    // or its functions can be called manually.
    // This is the trickiest part: main.js is a script, not a module.
    // One way: wrap main.js content in a function, export it, then call it in tests.
    // OR: just trigger events on the JSDOM elements it would listen to.
    
    // Override elements main.js would find
    global.document.getElementById = jest.fn((id) => {
        if (id === 'arxiv_id') return mockArxivIdInput;
        if (id === 'loadBtn') return mockLoadBtn;
        if (id === 'viewer') return mockViewer;
        if (id === 'status') return mockStatusElement;
        return null;
    });
    
    // Replace global WebSocket with the mock
    global.WebSocket = WS;
    server = new WS('ws://localhost:1234/ws', { jsonProtocol: true }); // URL should match main.js
    
    // Load main.js - this might involve reading its content and eval-ing it
    // or refactoring main.js for testability.
    // For this example, let's simulate its core logic being re-run by re-importing if possible,
    // or by directly testing the functions it defines if main.js exports them.
    // If main.js runs on load, we need to load it after setting up the DOM.
    // For simplicity, we'll test its event handlers by triggering them.
    
    // --- Simulate loading main.js ---
    // In a real scenario, you'd use jest.resetModules() and require('./../main.js')
    // if main.js was a module that could be re-imported.
    // Since it's a plain script, we'll test its event handlers by triggering them
    // after setting up the DOM elements it expects.
    // We'll assume connectWS and the message listener are globally available or part of an object.
    // For this example, let's assume main.js defines functions that we can call.
    // If not, this part needs adapting to how main.js is structured.
    
    // Let's assume main.js is refactored to be testable, e.g.:
    // window.cultivationReader = { connectWS: function..., setupEventListeners: function... };
    // Then in test: cultivationReader.setupEventListeners();
    // For now, we'll directly trigger events.
    
    // Execute main.js content in the current JSDOM context
    // This is a common way to test non-module scripts
    const fs = require('fs');
    const path = require('path');
    const mainJsContent = fs.readFileSync(path.resolve(__dirname, '../main.js'), 'utf8');
    const scriptEl = document.createElement('script');
    scriptEl.textContent = mainJsContent;
    document.body.appendChild(scriptEl); // This will execute main.js
  });

  afterEach(() => {
    WS.clean();
    jest.clearAllTimers();
    global.WebSocket = OriginalWebSocket; // Restore original WebSocket
  });

  test('loadBtn click should set iframe src and connect WebSocket', async () => {
    mockArxivIdInput.value = 'test.arxiv.id';
    
    // Manually get the button from the JSDOM and click it
    const loadButton = document.getElementById('loadBtn');
    loadButton.click(); // This triggers the onclick set by main.js

    expect(mockViewer.src).toBe('/static/pdfjs/viewer.html?file=/pdfs/test.arxiv.id.pdf');
    
    // Check WebSocket connection
    await server.connected; // Wait for client to connect
    expect(server.messages).toHaveLength(0); // No messages sent on connect by client
    expect(global.WebSocket).toHaveBeenCalledWith('ws://localhost:1234/ws?arxiv_id=test.arxiv.id'); // Check URL
  });

  test('connectWS should update status on WebSocket events', async () => {
    // Assuming connectWS is globally available or callable after main.js load
    // and that main.js sets up the 'status' element.
    
    // Trigger connectWS indirectly by clicking load button.
    mockArxivIdInput.value = 'status.test';
    document.getElementById('loadBtn').click();

    await server.connected;
    expect(mockStatusText).toBe('WebSocket connected');

    server.error();
    expect(mockStatusText).toBe('WebSocket error');
    
    // Re-establish for close test
    const server2 = new WS('ws://localhost:1234/ws?arxiv_id=status.test');
    document.getElementById('loadBtn').click(); // This will create a new ws instance in main.js
    await server2.connected;
    
    server2.close();
    expect(mockStatusText).toBe('WebSocket closed');
    WS.clean(); // Clean up server2
  });

  test('should forward messages from iframe to WebSocket', async () => {
    mockArxivIdInput.value = 'message.test';
    document.getElementById('loadBtn').click();
    
    await server.connected;

    const iframeMessage = { event_type: 'page_change', payload: { new_page_num: 2 } };
    
    // Simulate message from iframe
    window.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(iframeMessage) }));

    await expect(server).toReceiveMessage(iframeMessage);
  });

  test('should handle non-JSON messages from iframe gracefully', async () => {
    mockArxivIdInput.value = 'nonjson.test';
    document.getElementById('loadBtn').click();
    await server.connected;

    const consoleWarnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    
    window.dispatchEvent(new MessageEvent('message', { data: 'this is not json' }));

    expect(consoleWarnSpy).toHaveBeenCalledWith(expect.stringContaining('Failed to parse event.data'), 'this is not json', expect.any(Error));
    expect(server.messages).toHaveLength(0); // No message should be forwarded
    
    consoleWarnSpy.mockRestore();
  });

   test('should not send to WebSocket if not connected or not ready', async () => {
    // WS is not connected yet
    const iframeMessage = { event_type: 'page_change', payload: { new_page_num: 2 } };
    window.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(iframeMessage) }));
    
    // Server should not have received anything because main.js ws would be null or not ready
    expect(server.messages).toHaveLength(0); 

    // Connect, then simulate ws not being ready (readyState != 1)
    mockArxivIdInput.value = 'notready.test';
    document.getElementById('loadBtn').click();
    await server.connected;
    
    // Hackily set ws.readyState to something other than 1 in main.js context
    // This requires `ws` to be exposed, e.g., window.myAppWs = ws;
    // If not, this specific state is hard to test without modifying main.js
    // For now, assume if ws exists, it's ready after connection.
    // This test case might need main.js refactoring for full coverage.
  });
});
```

**III. Testing the Bridge Script in `static/pdfjs/viewer.html`**

*   **Approach:** Extract the core JavaScript logic from the `<script>` tag in `viewer.html` into a testable function or module (e.g., `viewer-bridge.js`).
*   **Mocking:**
    *   `window.parent.postMessage`.
    *   `window.PDFViewerApplication` and its `eventBus` (`on`, `dispatch` methods).
    *   `window.getSelection`.
    *   `document.addEventListener('selectionchange', ...)` (trigger manually).

**`static/pdfjs/viewer-bridge.js` (Example of extracted logic):**

```javascript
// This is the logic from your viewer.html's script tag, refactored slightly for testability.
export function setupPdfJsBridge(mockPdfViewerApp, mockWindowParent) {
  const localWindow = window; // In tests, this will be JSDOM's window
  const localDocument = document; // JSDOM's document

  function send(event_type, payload) {
    console.log('[PDF.js bridge] Sending event to parent:', event_type, payload);
    mockWindowParent.postMessage(JSON.stringify({ event_type, payload }), '*');
  }

  function hookBus(app) {
    if (!app) {
      console.error('[PDF.js bridge] No PDFViewerApplication!');
      return;
    }
    const bus = app.eventBus;
    if (!bus) {
      console.error('[PDF.js bridge] No eventBus found!');
      return;
    }
    console.log('[PDF.js bridge] Hooking eventBus');

    bus.on('pagechanging', function(evt) {
      console.log('[PDF.js bridge] pagechanging:', evt);
      send('page_change', { new_page_num: evt.pageNumber });
    });
    bus.on('updateviewarea', function(evt) {
      console.log('[PDF.js bridge] updateviewarea:', evt);
      send('view_area_update', {
        page_num: evt.location.pageNumber,
        top_visible: evt.location.top,
        scale: evt.location.scale
      });
    });
    bus.on('annotationeditorstateschanged', function(evt) {
      console.log('[PDF.js bridge] annotationeditorstateschanged:', evt);
      const detail = evt.detail || {};
      // Ensure we only send for highlight type as per your latest viewer.html
      // Assuming pdfjsLib is globally available or mocked for AnnotationType
      const AnnotationType = globalThis.pdfjsLib?.AnnotationType || { HIGHLIGHT: 16 /* default if not mocked */ };
      if (detail.type === 'add' && detail.value && detail.value.annotationType === AnnotationType.HIGHLIGHT) {
         send('highlight_created', { 
            type: detail.type, // 'add'
            // Replicate structure from viewer.mjs handleAnnotationEditorStatesChanged
            pageNumber: (typeof detail.value.pageIndex === 'number') ? detail.value.pageIndex + 1 : undefined,
            textContent: detail.value.textContent || '',
            color: detail.value.color,
            rects: detail.value.rects,
            quadPoints: detail.value.quadPoints,
            annotationId: detail.value.id
         });
      }
    });
  }

  localDocument.addEventListener('selectionchange', function() {
    const sel = localWindow.getSelection();
    if (sel && sel.toString().length > 0) {
      console.log('[PDF.js bridge] text_selected:', sel.toString());
      send('text_selected', { selected_text: sel.toString() });
    }
  });

  // Simulate PDFViewerApplication being ready
  if (mockPdfViewerApp) {
    if (mockPdfViewerApp.eventBus) {
         // Simulate PDF.js specific option setting
        if (globalThis.PDFViewerApplicationOptions && typeof globalThis.PDFViewerApplicationOptions.set === 'function') {
            globalThis.PDFViewerApplicationOptions.set('enableHighlightEditor', true);
            globalThis.PDFViewerApplicationOptions.set('enableHighlightFloatingButton', true);
            globalThis.PDFViewerApplicationOptions.set('annotationEditorMode', 2); // pdfjsViewer.AnnotationEditorType.HIGHLIGHT
        }
      hookBus(mockPdfViewerApp);
    } else {
        // Poll for eventBus if PDFViewerApplication might be late
        let attempts = 0;
        const intervalId = setInterval(() => {
            if (mockPdfViewerApp.eventBus || attempts++ > 10) {
                clearInterval(intervalId);
                if (mockPdfViewerApp.eventBus) hookBus(mockPdfViewerApp);
            }
        }, 100);
    }
  }
}
```

**`static/__tests__/viewer-bridge.test.js`:**

```javascript
import { setupPdfJsBridge } from '../pdfjs/viewer-bridge.js'; // Assuming you extracted it

describe('PDF.js Viewer Bridge Script', () => {
  let mockEventBus;
  let mockPdfViewerApp;
  let mockWindowParent;
  let mockSelection;

  beforeEach(() => {
    mockEventBus = {
      on: jest.fn(),
      dispatch: jest.fn(), // For completeness, though bridge only uses 'on'
    };
    mockPdfViewerApp = {
      eventBus: mockEventBus,
    };
    mockWindowParent = {
      postMessage: jest.fn(),
    };
    mockSelection = {
        toString: jest.fn().mockReturnValue('')
    };
    global.getSelection = jest.fn().mockReturnValue(mockSelection);

    // Mock PDFViewerApplicationOptions
    global.PDFViewerApplicationOptions = {
        set: jest.fn()
    };
    // Mock pdfjsLib for AnnotationType if needed
    global.pdfjsLib = {
        AnnotationType: { HIGHLIGHT: 16 } // Example value
    };


    // Initialize the bridge logic with mocks
    setupPdfJsBridge(mockPdfViewerApp, mockWindowParent);
  });

  test('should hook into PDFViewerApplication eventBus', () => {
    expect(mockEventBus.on).toHaveBeenCalledWith('pagechanging', expect.any(Function));
    expect(mockEventBus.on).toHaveBeenCalledWith('updateviewarea', expect.any(Function));
    expect(mockEventBus.on).toHaveBeenCalledWith('annotationeditorstateschanged', expect.any(Function));
  });

  test('should call PDFViewerApplicationOptions.set for highlight editor', () => {
    expect(global.PDFViewerApplicationOptions.set).toHaveBeenCalledWith('enableHighlightEditor', true);
    expect(global.PDFViewerApplicationOptions.set).toHaveBeenCalledWith('enableHighlightFloatingButton', true);
    expect(global.PDFViewerApplicationOptions.set).toHaveBeenCalledWith('annotationEditorMode', 2); // Assuming 2 is HIGHLIGHT
  });

  test('pagechanging event should postMessage to parent', () => {
    const pageChangeEvent = { pageNumber: 3 };
    // Find the 'pagechanging' callback and invoke it
    const pageChangingCallback = mockEventBus.on.mock.calls.find(call => call[0] === 'pagechanging')[1];
    pageChangingCallback(pageChangeEvent);

    expect(mockWindowParent.postMessage).toHaveBeenCalledWith(
      JSON.stringify({ event_type: 'page_change', payload: { new_page_num: 3 } }),
      '*'
    );
  });

  test('updateviewarea event should postMessage to parent', () => {
    const updateViewAreaEvent = { location: { pageNumber: 2, top: 100, scale: 1.5 } };
    const callback = mockEventBus.on.mock.calls.find(call => call[0] === 'updateviewarea')[1];
    callback(updateViewAreaEvent);

    expect(mockWindowParent.postMessage).toHaveBeenCalledWith(
      JSON.stringify({
        event_type: 'view_area_update',
        payload: { page_num: 2, top_visible: 100, scale: 1.5 },
      }),
      '*'
    );
  });

  test('text_selected should postMessage when selection changes', () => {
    mockSelection.toString.mockReturnValue('Selected some text');
    // Simulate selectionchange event
    const event = new Event('selectionchange');
    document.dispatchEvent(event);

    expect(mockWindowParent.postMessage).toHaveBeenCalledWith(
      JSON.stringify({ event_type: 'text_selected', payload: { selected_text: 'Selected some text' } }),
      '*'
    );
  });

   test('annotationeditorstateschanged event for highlight should postMessage', () => {
    const highlightEvent = {
      detail: {
        type: 'add',
        value: {
          annotationType: global.pdfjsLib.AnnotationType.HIGHLIGHT, // 16
          pageIndex: 0, // 0-indexed
          textContent: 'Highlighted Text',
          color: '#FFFF00',
          rects: [{x:1,y:1,w:10,h:10}],
          quadPoints: [0,0,10,0,0,10,10,10],
          id: 'highlight-123'
        }
      }
    };
    const callback = mockEventBus.on.mock.calls.find(call => call[0] === 'annotationeditorstateschanged')[1];
    callback(highlightEvent);

    expect(mockWindowParent.postMessage).toHaveBeenCalledWith(
      JSON.stringify({
        event_type: 'highlight_created',
        payload: {
          type: 'add',
          pageNumber: 1, // 1-indexed
          textContent: 'Highlighted Text',
          color: '#FFFF00',
          rects: [{x:1,y:1,w:10,h:10}],
          quadPoints: [0,0,10,0,0,10,10,10],
          annotationId: 'highlight-123'
        }
      }),
      '*'
    );
  });

  test('annotationeditorstateschanged event for non-highlight should not postMessage', () => {
    const nonHighlightEvent = {
      detail: {
        type: 'add',
        value: { annotationType: 99 /* some other type */ }
      }
    };
    const callback = mockEventBus.on.mock.calls.find(call => call[0] === 'annotationeditorstateschanged')[1];
    callback(nonHighlightEvent);
    // Assuming postMessage was called for previous highlight tests, we check it wasn't called *again* for this
    // Or, if this is the only 'annotationeditorstateschanged' test, check it wasn't called at all.
    // For simplicity, let's reset the mock for this specific test path if needed or ensure specific call args.
    // For this, we check if it has been called with highlight_created. If it was, this test is invalid.
    // A better way is to check the *number* of calls to postMessage before and after this specific event.
    const callsBefore = mockWindowParent.postMessage.mock.calls.length;
    callback(nonHighlightEvent);
    const callsAfter = mockWindowParent.postMessage.mock.calls.length;
    expect(callsAfter).toBe(callsBefore); // No new postMessage call
  });
});
```

**IV. Running Tests**

Add a script to `package.json` in `cultivation/reader_app/`:

```json
{
  "scripts": {
    "test": "jest"
  }
}
```

Then run `npm test` from `cultivation/reader_app/`.

**V. Further Considerations & Future Enhancements:**

*   **Refactoring `main.js` and `viewer.html` bridge:**
    *   Make `main.js` more modular by exporting its functions or wrapping its logic in a class/object that can be instantiated in tests.
    *   The bridge script in `viewer.html` being extracted to `viewer-bridge.js` makes it directly importable and testable. You'd then include `<script src="viewer-bridge.js"></script>` in `viewer.html` and call an initialization function.
*   **End-to-End (E2E) Tests (Playwright/Puppeteer):**
    *   For true confidence, E2E tests would launch a browser, load `index.html`, interact with the PDF, and verify WebSocket messages on a mock server.
    *   This is more complex but tests the actual browser environment and PDF.js integration.
    *   **Example E2E Scenario:**
        1.  Start mock FastAPI backend.
        2.  Start mock WebSocket server.
        3.  Use Playwright to open `index.html`.
        4.  Input an arXiv ID, click "Load PDF".
        5.  Wait for iframe to load `viewer.html`.
        6.  Interact with PDF.js (e.g., scroll, select text) programmatically or by simulating user input.
        7.  Assert that the mock WebSocket server received the expected telemetry events.
*   **Test Coverage:** Use Jest's `--coverage` flag to identify untested code paths.

This plan provides a solid foundation for testing your JavaScript code. The Jest unit/integration tests will catch many regressions in your core logic. If you find these are insufficient due to browser-specific issues or complex PDF.js interactions, then investing in a few key E2E tests would be the 