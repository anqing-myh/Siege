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
        const code = msg.roomCode?.toUpperCase();
        if (!code || !rooms.has(code)) {
          ws.send(JSON.stringify({ type: 'error', message: '房间不存在' }));
          return;
        }
        const room = rooms.get(code);
        if (room.length >= 2) {
          ws.send(JSON.stringify({ type: 'error', message: '房间已满' }));
          return;
        }
        room.push({ ws, playerIndex: 1 });
        currentRoom = code;
        playerIndex = 1;
        ws.send(JSON.stringify({ type: 'joined', roomCode: code, playerIndex: 1 }));
        // 通知房主对手加入
        room[0].ws.send(JSON.stringify({ type: 'opponent_joined' }));
      }
      else if (msg.type === 'action' || msg.type === 'sync' || msg.type === 'restart') {
        // 所有游戏消息都严格按房间广播
        if (!currentRoom || !rooms.has(currentRoom)) return;
        const room = rooms.get(currentRoom);
        for (const p of room) {
          if (p.ws !== ws && p.ws.readyState === 1) {
            p.ws.send(JSON.stringify({ ...msg, from: playerIndex }));
          }
        }
      }
      else if (msg.type === 'reset_request') {
        if (!currentRoom || !rooms.has(currentRoom)) return;
        const room = rooms.get(currentRoom);
        // 房主处理重置请求
        if (playerIndex === 0) {
          for (const p of room) {
            if (p.ws.readyState === 1) {
              p.ws.send(JSON.stringify({ type: 'restart' }));
            }
          }
        }
      }
    } catch (e) {
      console.error('消息解析错误:', e);
    }
  });

  ws.on('close', () => {
    if (currentRoom && rooms.has(currentRoom)) {
      const room = rooms.get(currentRoom);
      const other = room.find(p => p.playerIndex !== playerIndex);
      if (other && other.ws.readyState === 1) {
        other.ws.send(JSON.stringify({ type: 'opponent_left' }));
      }
      // 移除当前玩家
      const remaining = room.filter(p => p.ws !== ws);
      if (remaining.length > 0) {
        rooms.set(currentRoom, remaining);
      } else {
        rooms.delete(currentRoom);
      }
    }
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => console.log('围城服务器已启动，端口:', PORT));