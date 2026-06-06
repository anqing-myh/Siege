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

// 核心离开房间函数
function leaveRoom(ws) {
  const currentRoom = ws.currentRoom;
  const playerIndex = ws.playerIndex;

  if (!currentRoom || !rooms.has(currentRoom)) return;

  const room = rooms.get(currentRoom);

  // 房主离开 → 直接解散房间
  if (playerIndex === 0) {
    room.forEach(p => {
      if (p.ws.readyState === 1 && p.ws !== ws) {
        p.ws.send(JSON.stringify({ type: 'room_closed', reason: 'host_left' }));
      }
      p.ws.currentRoom = null;
      p.ws.playerIndex = null;
    });
    rooms.delete(currentRoom);
    console.log(`Room ${currentRoom} closed (host left)`);
    return;
  }

  // 非房主离开 → 通知房主
  const host = room.find(p => p.playerIndex === 0);
  if (host && host.ws.readyState === 1) {
    host.ws.send(JSON.stringify({ type: 'opponent_left' }));
  }

  // 移除离开的玩家
  const remaining = room.filter(p => p.ws !== ws);
  if (remaining.length > 0) {
    rooms.set(currentRoom, remaining);
  } else {
    rooms.delete(currentRoom);
  }

  ws.currentRoom = null;
  ws.playerIndex = null;
  console.log(`Player left room ${currentRoom}`);
}

wss.on('connection', (ws) => {
  ws.currentRoom = null;
  ws.playerIndex = null;

  ws.on('message', (raw) => {
    try {
      const msg = JSON.parse(raw.toString());

      if (msg.type === 'create') {
        let code;
        do { code = generateRoomCode(); } while (rooms.has(code));
        rooms.set(code, [{ ws, playerIndex: 0 }]);
        ws.currentRoom = code;
        ws.playerIndex = 0;
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
        ws.currentRoom = code;
        ws.playerIndex = 1;
        ws.send(JSON.stringify({ type: 'joined', roomCode: code, playerIndex: 1 }));
        // 通知房主对手加入
        const host = room.find(p => p.playerIndex === 0);
        if (host && host.ws.readyState === 1) {
          host.ws.send(JSON.stringify({ type: 'opponent_joined' }));
        }
      }
      else if (msg.type === 'leave') {
        leaveRoom(ws);
      }
      else if (msg.type === 'action' || msg.type === 'sync' || msg.type === 'restart') {
        if (!ws.currentRoom || !rooms.has(ws.currentRoom)) return;
        const room = rooms.get(ws.currentRoom);
        for (const p of room) {
          if (p.ws !== ws && p.ws.readyState === 1) {
            p.ws.send(JSON.stringify({ ...msg, from: ws.playerIndex }));
          }
        }
      }
      else if (msg.type === 'reset_request') {
        if (!ws.currentRoom || !rooms.has(ws.currentRoom)) return;
        const room = rooms.get(ws.currentRoom);
        if (ws.playerIndex === 0) {
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
    leaveRoom(ws);
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => console.log('围城服务器已启动，端口:', PORT));