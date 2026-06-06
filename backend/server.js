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
        // 生成唯一房间码
        let code;
        do { code = generateRoomCode(); } while (rooms.has(code));
        rooms.set(code, [{ ws, playerIndex: 0 }]);
        currentRoom = code;
        playerIndex = 0;
        ws.send(JSON.stringify({ type: 'created', roomCode: code, playerIndex: 0 }));
      }
      else if (msg.type === 'join') {
        const code = (msg.roomCode || '').toUpperCase();
        if (!code) {
          ws.send(JSON.stringify({ type: 'error', message: '房间码不能为空' }));
          return;
        }
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
        // 通知房主和对方对手加入
        room.forEach(p => {
          if (p.ws.readyState === 1 && p.ws !== ws) {
            p.ws.send(JSON.stringify({ type: 'opponent_joined', roomCode: code }));
          }
        });
      }
      else if (msg.type === 'reset_request') {
        // 加入者请求房主重置游戏
        if (currentRoom) {
          const room = rooms.get(currentRoom);
          if (!room) return;
          // 只能加入者请求，房主忽略
          if (playerIndex !== 1) return;
          const owner = room.find(p => p.playerIndex === 0);
          if (owner && owner.ws.readyState === 1) {
            owner.ws.send(JSON.stringify({ type: 'reset_request', roomCode: currentRoom }));
          }
        }
      }
      else if (msg.type === 'action' || msg.type === 'sync') {
        if (currentRoom) {
          const room = rooms.get(currentRoom);
          if (!room) return;
          // 对手连接
          const target = room.find(p => p.playerIndex !== playerIndex);
          if (target && target.ws.readyState === 1) {
            // 携带来源和房间码，中继给对方
            const forwardMsg = {
              ...msg,
              from: playerIndex,
              roomCode: currentRoom,
            };
            target.ws.send(JSON.stringify(forwardMsg));
          }
        }
      }
      else {
        // 无效或忽略其它类型消息
      }
    }
    catch (e) {
      // 解析失败忽略
    }
  });

  ws.on('close', () => {
    if (!currentRoom) return;
    const room = rooms.get(currentRoom);
    if (!room) return;

    // 移除当前连接
    const idx = room.findIndex(p => p.ws === ws);
    if (idx !== -1) room.splice(idx, 1);

    // 通知对手有人离开
    if (room.length === 1) {
      const other = room[0];
      if (other.ws.readyState === 1) {
        other.ws.send(JSON.stringify({ type: 'opponent_left', roomCode: currentRoom }));
      }
    }

    // 房间无人后销毁房间
    if (room.length === 0) {
      rooms.delete(currentRoom);
    }

    currentRoom = null;
    playerIndex = null;
  });

  ws.on('error', () => {
    // 可选处理错误，关闭连接时自动调用 close
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log('围城服务器已启动，端口:', PORT);
});