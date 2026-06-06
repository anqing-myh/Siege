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

// 离开房间逻辑
function leaveRoom(ws) {
  const roomCode = ws.currentRoom;
  const playerIndex = ws.playerIndex;

  if (!roomCode || !rooms.has(roomCode)) return;
  const room = rooms.get(roomCode);

  // 房主离开 → 通知其他玩家，但不切换模式
  if (playerIndex === 0) {
    room.forEach(p => {
      if (p.ws.readyState === 1 && p.ws !== ws) {
        p.ws.send(JSON.stringify({ type: 'room_closed', reason: 'host_left' }));
      }
      p.ws.currentRoom = null;
      p.ws.playerIndex = null;
    });
    rooms.delete(roomCode);
    console.log(`Room ${roomCode} closed (host left)`);
    return;
  }

  // 非房主离开 → 移除玩家并通知房主
  const host = room.find(p => p.playerIndex === 0);
  if (host && host.ws.readyState === 1) {
    host.ws.send(JSON.stringify({ type: 'opponent_left' }));
  }

  const remaining = room.filter(p => p.ws !== ws);
  if (remaining.length > 0) {
    rooms.set(roomCode, remaining);
  } else {
    rooms.delete(roomCode);
  }

  ws.currentRoom = null;
  ws.playerIndex = null;
  console.log(`Player left room ${roomCode}`);
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
      rooms.set(code, [{ ws, playerIndex: 0, state: {} }]);
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

      // 加入房间
      room.push({ ws, playerIndex: 1, state: {} });
      ws.currentRoom = code;
      ws.playerIndex = 1;
      ws.send(JSON.stringify({ type: 'joined', roomCode: code, playerIndex: 1 }));

      // 通知房主对手加入，并同步游戏状态给新加入玩家
      const host = room.find(p => p.playerIndex === 0);
      if (host && host.ws.readyState === 1) {
        host.ws.send(JSON.stringify({ type: 'opponent_joined' }));
        ws.send(JSON.stringify({
          type: 'sync',
          from: 0,
          state: host.state // 房主状态统一存放在 state
        }));
      }
    }
    else if (msg.type === 'leave') {
      leaveRoom(ws);
    }
    else if (msg.type === 'action' || msg.type === 'sync' || msg.type === 'restart') {
      if (!ws.currentRoom || !rooms.has(ws.currentRoom)) return;
      const room = rooms.get(ws.currentRoom);

      // 保存房主状态在 state 中
      if (ws.playerIndex === 0 && msg.state) {
        room[0].state = msg.state; // 统一存放在 state
      }

      // 广播给房间内其他玩家
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
server.listen(PORT, () => console.log('服务器已启动，端口:', PORT));