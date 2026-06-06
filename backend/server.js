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

wss.on('connection', (ws) => {
  let currentRoom = null;
  let playerIndex = null;

  ws.on('message', (raw) => {
    try {
      const msg = JSON.parse(raw.toString());

      if (msg.type === 'create') {
        let code;
        do { code = generateRoomCode(); } while (rooms.has(code));
        rooms.set(code, [{ ws, playerIndex: 0 }]);
        currentRoom = code;
        playerIndex = 0;
        ws.send(JSON.stringify({ type: 'created', roomCode: code, playerIndex: 0 }));
      }
      else if (msg.type === 'join') {
        const code = msg.roomCode.toUpperCase();
        const room = rooms.get(code);
        if (!room) {
          ws.send(JSON.stringify({ type: 'error', message: '房间不存在' }));
          return;
        }
        if (room.length >= 2) {
          ws.send(JSON.stringify({ type: 'error', message: '房间已满' }));
          return;
        }
        room.push({ ws, playerIndex: 1 });
        currentRoom = code;
        playerIndex = 1;
        ws.send(JSON.stringify({ type: 'joined', roomCode: code, playerIndex: 1 }));
        // 通知房主有玩家加入
        room[0].ws.send(JSON.stringify({ type: 'opponent_joined' }));
      }
      else if (msg.type === 'action' || msg.type === 'sync') {
        if (currentRoom) {
          const room = rooms.get(currentRoom);
          if (room) {
            const target = room.find(p => p.playerIndex !== playerIndex);
            if (target && target.ws.readyState === 1) {
              target.ws.send(JSON.stringify({ ...msg, from: playerIndex }));
            }
          }
        }
      }
    } catch (e) {
      // 忽略解析错误
    }
  });

  ws.on('close', () => {
    if (currentRoom) {
      const room = rooms.get(currentRoom);
      if (room) {
        const other = room.find(p => p.playerIndex !== playerIndex);
        if (other && other.ws.readyState === 1) {
          other.ws.send(JSON.stringify({ type: 'opponent_left' }));
        }
        rooms.delete(currentRoom);
      }
    }
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log('围城服务器已启动，端口:', PORT);
});