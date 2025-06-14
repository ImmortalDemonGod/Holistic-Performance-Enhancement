// Tests for main.js using Jest and JSDOM, as per test_plan.md
import { fireEvent } from '@testing-library/dom';
import WS from 'jest-websocket-mock';

// We'll import main.js as a module if possible, or simulate its effects otherwise
let server;
let mainModule;

beforeEach(async () => {
  // Set up DOM
  document.body.innerHTML = `
    <button id="load">Load</button>
    <span id="status"></span>
    <iframe id="pdf-viewer"></iframe>
  `;
  // Create a mock WebSocket server
  server = new WS('ws://localhost:8765/');
  // Import main.js (if it exports anything)
  jest.resetModules();
  mainModule = await import('../main.js');
});

afterEach(() => {
  WS.clean();
  jest.resetModules();
});

describe('main.js', () => {
  test('initializes PDF.js viewer and sets up event listeners', () => {
    const loadBtn = document.getElementById('load');
    expect(loadBtn).not.toBeNull();
    // Simulate click
    fireEvent.click(loadBtn);
    // Should update status to 'Connecting...'
    expect(document.getElementById('status').textContent).toMatch(/Connecting/);
  });

  test('connectWS should update status on WebSocket events', async () => {
    const loadBtn = document.getElementById('load');
    fireEvent.click(loadBtn);
    // Simulate server connection
    await server.connected;
    server.send('connected');
    // Wait for DOM update
    await new Promise(r => setTimeout(r, 10));
    expect(document.getElementById('status').textContent).toMatch(/connected/i);
  });

  test('relays postMessage events from iframe to WebSocket', async () => {
    const loadBtn = document.getElementById('load');
    fireEvent.click(loadBtn);
    await server.connected;
    // Simulate message event from iframe
    window.dispatchEvent(new MessageEvent('message', { data: { type: 'test', payload: 42 }, source: window }));
    // Should send to WebSocket
    await expect(server).toReceiveMessage(JSON.stringify({ type: 'test', payload: 42 }));
  });
});
