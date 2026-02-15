// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  VoxRoom — Signaling Server
//  WhatsApp-style calling with low-bandwidth optimization
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const { v4: uuidv4 } = require("uuid");
const path = require("path");
const cors = require("cors");

const app = express();
const server = http.createServer(app);

const io = new Server(server, {
  cors: { origin: "*" },
  pingInterval: 8000,
  pingTimeout: 25000,
  maxHttpBufferSize: 1e6,
  transports: ["websocket", "polling"],
});

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

const PORT = process.env.PORT || 8080;
const BOOT = Date.now();

// ━━ DATA STORES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const rooms = new Map();
const socketToUser = new Map();

// ━━ HELPERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function genCode() {
  const c = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  let code = "";
  for (let i = 0; i < 6; i++) code += c[Math.floor(Math.random() * c.length)];
  return code;
}

function genAvatar(name) {
  const colors = [
    "#25D366","#128C7E","#075E54","#34B7F1",
    "#E91E63","#9C27B0","#673AB7","#2196F3",
    "#00BCD4","#009688","#4CAF50","#FF9800",
    "#FF5722","#795548","#607D8B","#F44336",
  ];
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = name.charCodeAt(i) + ((hash << 5) - hash);
  return colors[Math.abs(hash) % colors.length];
}

function broadcastRoom(roomCode) {
  const room = rooms.get(roomCode);
  if (!room) return;

  const data = {
    code: room.code,
    name: room.name,
    maxCall: room.maxCall,
    creatorId: room.creatorId,
    createdAt: room.createdAt,
    callActive: room.callActive,
    participants: [],
    callMembers: [...room.callMembers],
    messages: room.messages.slice(-200),
  };

  for (const [uid, u] of room.users) {
    data.participants.push({
      id: u.id,
      name: u.name,
      color: u.color,
      online: u.online,
      inCall: room.callMembers.includes(uid),
      joinedAt: u.joinedAt,
    });
  }

  io.to(roomCode).emit("room:update", data);
}

function systemMsg(room, text) {
  const msg = {
    id: uuidv4(),
    type: "system",
    text,
    time: Date.now(),
  };
  room.messages.push(msg);
  io.to(room.code).emit("chat:message", msg);
}

// ━━ API ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    uptime: Math.floor((Date.now() - BOOT) / 1000),
    rooms: rooms.size,
    connections: io.engine.clientsCount,
  });
});

// ━━ SOCKET.IO SIGNALING ━━━━━━━━━━━━━━━━━━━━━━━━
io.on("connection", (socket) => {
  console.log(`🔌 ${socket.id} connected`);

  let myRoom = null;
  let myUserId = null;

  // ── CREATE ROOM ──────────────────────────────
  socket.on("room:create", ({ name, userName }, cb) => {
    const code = genCode();
    const userId = uuidv4();
    myUserId = userId;
    myRoom = code;

    const room = {
      code,
      name: name || `Room ${code}`,
      maxCall: 2,
      creatorId: userId,
      createdAt: Date.now(),
      callActive: false,
      callMembers: [],
      users: new Map(),
      messages: [],
    };

    const user = {
      id: userId,
      name: userName || "Anonymous",
      color: genAvatar(userName || "Anonymous"),
      socketId: socket.id,
      online: true,
      joinedAt: Date.now(),
    };

    room.users.set(userId, user);
    rooms.set(code, room);
    socketToUser.set(socket.id, { roomCode: code, userId });

    socket.join(code);
    systemMsg(room, `${user.name} created the room`);

    console.log(`  🏠 Room ${code} created by ${user.name}`);

    if (cb) cb({ ok: true, code, userId, room: { code, name: room.name } });
    broadcastRoom(code);
  });

  // ── JOIN ROOM ────────────────────────────────
  socket.on("room:join", ({ code, userName }, cb) => {
    const room = rooms.get(code);
    if (!room) {
      if (cb) cb({ ok: false, error: "Room not found. Check the code." });
      return;
    }

    const userId = uuidv4();
    myUserId = userId;
    myRoom = code;

    const user = {
      id: userId,
      name: userName || "Anonymous",
      color: genAvatar(userName || "Anonymous"),
      socketId: socket.id,
      online: true,
      joinedAt: Date.now(),
    };

    room.users.set(userId, user);
    socketToUser.set(socket.id, { roomCode: code, userId });

    socket.join(code);
    systemMsg(room, `${user.name} joined the room`);

    console.log(`  👤 ${user.name} joined ${code}`);

    if (cb) cb({ ok: true, code, userId, room: { code, name: room.name } });
    broadcastRoom(code);
  });

  // ── CHAT MESSAGE ─────────────────────────────
  socket.on("chat:send", ({ text }) => {
    if (!myRoom || !myUserId) return;
    const room = rooms.get(myRoom);
    if (!room) return;
    const user = room.users.get(myUserId);
    if (!user) return;

    const msg = {
      id: uuidv4(),
      type: "user",
      userId: myUserId,
      userName: user.name,
      userColor: user.color,
      text: text.slice(0, 2000),
      time: Date.now(),
    };

    room.messages.push(msg);
    io.to(myRoom).emit("chat:message", msg);
  });

  // ── ROOM SETTINGS (creator only) ────────────
  socket.on("room:settings", ({ maxCall }) => {
    if (!myRoom || !myUserId) return;
    const room = rooms.get(myRoom);
    if (!room || room.creatorId !== myUserId) return;

    if (maxCall >= 2 && maxCall <= 8) {
      room.maxCall = maxCall;
      systemMsg(room, `Max call participants changed to ${maxCall}`);
      broadcastRoom(myRoom);
    }
  });

  // ── CALL: JOIN ───────────────────────────────
  socket.on("call:join", (_, cb) => {
    if (!myRoom || !myUserId) return;
    const room = rooms.get(myRoom);
    if (!room) return;

    if (room.callMembers.length >= room.maxCall) {
      if (cb) cb({ ok: false, error: `Call is full (max ${room.maxCall})` });
      return;
    }

    if (!room.callMembers.includes(myUserId)) {
      room.callMembers.push(myUserId);
    }
    room.callActive = room.callMembers.length > 0;

    const user = room.users.get(myUserId);
    systemMsg(room, `${user.name} joined the call`);

    // Tell existing call members about new peer
    for (const uid of room.callMembers) {
      if (uid === myUserId) continue;
      const other = room.users.get(uid);
      if (!other) continue;
      // Tell the new peer about existing members
      socket.emit("call:peer-joined", {
        peerId: uid,
        peerName: other.name,
        peerColor: other.color,
        shouldOffer: true,
      });
      // Tell existing members about new peer
      io.to(other.socketId).emit("call:peer-joined", {
        peerId: myUserId,
        peerName: user.name,
        peerColor: user.color,
        shouldOffer: false,
      });
    }

    console.log(`  📞 ${user.name} joined call in ${myRoom} (${room.callMembers.length}/${room.maxCall})`);

    if (cb) cb({ ok: true, members: room.callMembers.length });
    broadcastRoom(myRoom);
  });

  // ── CALL: LEAVE ──────────────────────────────
  socket.on("call:leave", () => {
    handleCallLeave();
  });

  function handleCallLeave() {
    if (!myRoom || !myUserId) return;
    const room = rooms.get(myRoom);
    if (!room) return;

    const idx = room.callMembers.indexOf(myUserId);
    if (idx === -1) return;

    room.callMembers.splice(idx, 1);
    room.callActive = room.callMembers.length > 0;

    const user = room.users.get(myUserId);
    systemMsg(room, `${user?.name || "Someone"} left the call`);

    // Tell remaining members
    for (const uid of room.callMembers) {
      const other = room.users.get(uid);
      if (other) {
        io.to(other.socketId).emit("call:peer-left", { peerId: myUserId });
      }
    }

    console.log(`  📵 ${user?.name} left call in ${myRoom}`);
    broadcastRoom(myRoom);
  }

  // ── WebRTC SIGNALING ─────────────────────────
  socket.on("rtc:offer", ({ targetUserId, offer }) => {
    if (!myRoom) return;
    const room = rooms.get(myRoom);
    if (!room) return;
    const target = room.users.get(targetUserId);
    if (target) {
      io.to(target.socketId).emit("rtc:offer", {
        fromUserId: myUserId,
        offer,
      });
    }
  });

  socket.on("rtc:answer", ({ targetUserId, answer }) => {
    if (!myRoom) return;
    const room = rooms.get(myRoom);
    if (!room) return;
    const target = room.users.get(targetUserId);
    if (target) {
      io.to(target.socketId).emit("rtc:answer", {
        fromUserId: myUserId,
        answer,
      });
    }
  });

  socket.on("rtc:ice", ({ targetUserId, candidate }) => {
    if (!myRoom) return;
    const room = rooms.get(myRoom);
    if (!room) return;
    const target = room.users.get(targetUserId);
    if (target) {
      io.to(target.socketId).emit("rtc:ice", {
        fromUserId: myUserId,
        candidate,
      });
    }
  });

  // ── NETWORK QUALITY REPORT ───────────────────
  socket.on("net:quality", ({ quality }) => {
    if (!myRoom || !myUserId) return;
    const room = rooms.get(myRoom);
    if (!room) return;
    // Relay quality info to call peers so they can adapt
    for (const uid of room.callMembers) {
      if (uid === myUserId) continue;
      const other = room.users.get(uid);
      if (other) {
        io.to(other.socketId).emit("net:peer-quality", {
          peerId: myUserId,
          quality,
        });
      }
    }
  });

  // ── TYPING INDICATOR ─────────────────────────
  socket.on("chat:typing", () => {
    if (!myRoom || !myUserId) return;
    const room = rooms.get(myRoom);
    if (!room) return;
    const user = room.users.get(myUserId);
    if (user) {
      socket.to(myRoom).emit("chat:typing", {
        userId: myUserId,
        name: user.name,
      });
    }
  });

  // ── DISCONNECT ───────────────────────────────
  socket.on("disconnect", () => {
    console.log(`🔌 ${socket.id} disconnected`);

    handleCallLeave();

    if (myRoom && myUserId) {
      const room = rooms.get(myRoom);
      if (room) {
        const user = room.users.get(myUserId);
        if (user) {
          user.online = false;
          user.socketId = null;
          systemMsg(room, `${user.name} went offline`);
        }
        broadcastRoom(myRoom);

        // Clean up empty rooms after 30 min
        const onlineCount = [...room.users.values()].filter((u) => u.online).length;
        if (onlineCount === 0) {
          setTimeout(() => {
            const r = rooms.get(myRoom);
            if (r) {
              const still = [...r.users.values()].filter((u) => u.online).length;
              if (still === 0) {
                rooms.delete(myRoom);
                console.log(`  🗑️ Room ${myRoom} cleaned up`);
              }
            }
          }, 1800000);
        }
      }
    }

    socketToUser.delete(socket.id);
  });
});

// ━━ BOOT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
server.listen(PORT, () => {
  console.log(`
╔══════════════════════════════════════╗
║   📱 VoxRoom v1.0                    ║
║   Listening on port ${PORT}             ║
║   ${new Date().toISOString()}     ║
╚══════════════════════════════════════╝
  `);
});
