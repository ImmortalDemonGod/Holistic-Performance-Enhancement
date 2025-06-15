// e2e-server.js: Serves static files and WebSocket endpoint for end-to-end tests
const http = require('http');
const handler = require('serve-handler');
const WebSocket = require('ws');

const PORT = 8000;
const server = http.createServer((request, response) => {
  return handler(request, response, { public: 'static' });
});

const wss = new WebSocket.Server({ noServer: true });
server.on('upgrade', (request, socket, head) => {
  if (request.url.startsWith('/ws')) {
    wss.handleUpgrade(request, socket, head, ws => {
      wss.emit('connection', ws, request);
    });
  } else {
    socket.destroy();
  }
});

wss.on('connection', ws => {
  ws.on('message', message => {
    // echo messages back
    ws.send(message);
  });
});

server.listen(PORT, () => {
  console.log(`E2E server listening at http://localhost:${PORT}`);
});
