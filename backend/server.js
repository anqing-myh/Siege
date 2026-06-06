const { WebSocketServer } = require('ws');
const http = require('http');

const server = http.createServer();
const wss = new WebSocketServer({ server });

const rooms = new Map();
const CHARS = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';

function generateRoomCode() {
  let code = '';
  for (let i = 0; i < 5; i++) {
    code += CHARS[Math.floor(Math.random() * CHARS.length)];
  }
  return code;
}

function send(ws, data) {
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify(data));
  }
}

function broadcast(room, data, exceptWs = null) {
  for (const p of room.players) {
    if (p.ws !== exceptWs) send(p.ws, data);
  }
}

function leaveRoom(ws) {
  const code = ws.currentRoom;
  if (!code || !rooms.has(code)) return;

  const room = rooms.get(code);

  if (ws.playerIndex === 0) {
    broadcast(room, { type: 'room_closed', reason: 'host_left' }, ws);
    for (const p of room.players) {
      p.ws.currentRoom = null;
      p.ws.playerIndex = null;
    }
    rooms.delete(code);
    return;
  }

  room.players = room.players.filter(p => p.ws !== ws);

  const host = room.players.find(p => p.playerIndex === 0);
  if (host) send(host.ws, { type: 'opponent_left' });

  ws.currentRoom = null;
  ws.playerIndex = null;
}

wss.on('connection', (ws) => {
  ws.currentRoom = null;
  ws.playerIndex = null;

  ws.on('message', (raw) => {
    try {
      const msg = JSON.parse(raw.toString());

      if (msg.type === 'create') {
        let code;
        do {
          code = generateRoomCode();
        } while (rooms.has(code));

        rooms.set(code, {
          players: [{ ws, playerIndex: 0 }],
          state: msg.state || null
        });

        ws.currentRoom = code;
        ws.playerIndex = 0;

        send(ws, {
          type: 'created',
          roomCode: code,
          playerIndex: 0
        });

        return;
      }

      if (msg.type === 'join') {
        const code = msg.roomCode?.toUpperCase();

        if (!code || !rooms.has(code)) {
          send(ws, { type: 'error', message: '房间不存在' });
          return;
        }

        const room = rooms.get(code);

        if (room.players.length >= 2) {
          send(ws, { type: 'error', message: '房间已满' });
          return;
        }

        room.players.push({ ws, playerIndex: 1 });

        ws.currentRoom = code;
        ws.playerIndex = 1;

        send(ws, {
          type: 'joined',
          roomCode: code,
          playerIndex: 1
        });

        const host = room.players.find(p => p.playerIndex === 0);
        if (host) {
          send(host.ws, { type: 'opponent_joined' });
        }

        if (room.state) {
          send(ws, {
            type: 'sync',
            from: 0,
            state: room.state
          });
        }

        return;
      }

      if (msg.type === 'leave') {
        leaveRoom(ws);
        return;
      }

      if (!ws.currentRoom || !rooms.has(ws.currentRoom)) return;

      const room = rooms.get(ws.currentRoom);

      if (msg.type === 'sync') {
        if (msg.state) {
          room.state = msg.state;
        }

        broadcast(room, {
          type: 'sync',
          from: ws.playerIndex,
          state: room.state
        }, ws);

        return;
      }

      if (msg.type === 'reset_request') {
        const host = room.players.find(p => p.playerIndex === 0);
        if (host) {
          send(host.ws, {
            type: 'reset_request',
            from: ws.playerIndex
          });
        }
        return;
      }

    } catch (e) {
      console.error('消息解析错误:', e);
    }
  });

  ws.on('close', () => {
    leaveRoom(ws);
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => console.log('服务器已启动，端口:', PORT));