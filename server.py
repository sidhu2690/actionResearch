// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  VoxRoom — Signaling Server v1.1
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

const express = require("express");
const http = require("http");
const path = require("path");

let Server, uuidv4, cors;

try {
  Server = require("socket.io").Server;
} catch (e) {
  console.error("❌ socket.io not found:", e.message);
  process.exit(1);
}

try {
  uuidv4 = require("uuid").v4;
} catch (e) {
  // Fallback UUID generator if uuid package fails
  console.warn("⚠ uuid package not found, using fallback");
  uuidv4 = () => {
    return "xxxx-xxxx-xxxx-xxxx".replace(/x/g, () =>
      Math.floor(Math.random() * 16).toString(16)
    );
  };
}

try {
  cors = require("cors");
} catch (e) {
  console.warn("⚠ cors package not found, using manual headers");
  cors = null;
}

const app = express();
const server = http.createServer(app);

const io = new Server(server, {
  cors: { origin: "*", methods: ["GET", "POST"] },
  pingInterval: 8000,
  pingTimeout: 25000,
  maxHttpBufferSize: 1e6,
  transports: ["websocket", "polling"],
});

if (cors) {
  app.use(cors());
} else {
  app.use((req, res, next) => {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Headers", "Content-Type");
    next();
  });
}

app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

const PORT = process.env.PORT || 8080;
const BOOT = Date.now();

// ━━ DATA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const rooms = new Map();
const socketToUser = new Map();

// ━━ HELPERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function genCode() {
  const c = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  let code = "";
  for (let i = 0; i < 6; i++) code += c[Math.floor(Math.random() * c.length)];
  return code;
}

function genColor(name) {
  const colors = [
    "#25D366","#128C7E","#075E54","#34B7F1",
    "#E91E63","#9C27B0","#673AB7","#2196F3",
    "#00BCD4","#009688","#4CAF50","#FF9800",
    "#FF5722","#795548","#607D8B","#F44336",
  ];
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash);
  }
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
    text: text,
    time: Date.now(),
  };
  room.messages.push(msg);
  io.to(room.code).emit("chat:message", msg);
}

// ━━ ROUTES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    uptime: Math.floor((Date.now() - BOOT) / 1000),
    rooms: rooms.size,
    connections: io.engine ? io.engine.clientsCount : 0,
  });
});

// Serve index.html for root and any unmatched routes
app.get("/", (req, res) => {
  const indexPath = path.join(__dirname, "public", "index.html");
  res.sendFile(indexPath, (err) => {
    if (err) {
      console.error("Error serving index.html:", err.message);
      res.status(200).send(`
        <!DOCTYPE html>
        <html><head><title>VoxRoom</title></head>
        <body style="background:#111;color:#eee;display:flex;align-items:center;justify-content:center;height:100vh;font-family:sans-serif">
          <div style="text-align:center">
            <h1>📱 VoxRoom</h1>
            <p>Server is running but index.html not found.</p>
            <p>Check that public/index.html exists.</p>
          </div>
        </body></html>
      `);
    }
  });
});

// ━━ SOCKET.IO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
io.on("connection", (socket) => {
  console.log(`🔌 ${socket.id} connected (total: ${io.engine.clientsCount})`);

  let myRoom = null;
  let myUserId = null;

  // ── CREATE ROOM ──────────────────────────────
  socket.on("room:create", (payload, cb) => {
    try {
      const { name, userName } = payload || {};
      const code = genCode();
      const userId = uuidv4();
      myUserId = userId;
      myRoom = code;

      const uName = (userName || "Anonymous").slice(0, 30);

      const room = {
        code: code,
        name: name || `${uName}'s Room`,
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
        name: uName,
        color: genColor(uName),
        socketId: socket.id,
        online: true,
        joinedAt: Date.now(),
      };

      room.users.set(userId, user);
      rooms.set(code, room);
      socketToUser.set(socket.id, { roomCode: code, userId: userId });

      socket.join(code);
      systemMsg(room, `${user.name} created the room`);

      console.log(`  🏠 Room ${code} created by ${user.name}`);

      if (typeof cb === "function") {
        cb({ ok: true, code: code, userId: userId, room: { code: code, name: room.name } });
      }
      broadcastRoom(code);
    } catch (e) {
      console.error("room:create error:", e);
      if (typeof cb === "function") cb({ ok: false, error: "Server error" });
    }
  });

  // ── JOIN ROOM ────────────────────────────────
  socket.on("room:join", (payload, cb) => {
    try {
      const { code, userName } = payload || {};
      const roomCode = (code || "").toUpperCase().trim();
      const room = rooms.get(roomCode);

      if (!room) {
        if (typeof cb === "function") cb({ ok: false, error: "Room not found. Check the code." });
        return;
      }

      const userId = uuidv4();
      myUserId = userId;
      myRoom = roomCode;

      const uName = (userName || "Anonymous").slice(0, 30);

      const user = {
        id: userId,
        name: uName,
        color: genColor(uName),
        socketId: socket.id,
        online: true,
        joinedAt: Date.now(),
      };

      room.users.set(userId, user);
      socketToUser.set(socket.id, { roomCode: roomCode, userId: userId });

      socket.join(roomCode);
      systemMsg(room, `${user.name} joined the room`);

      console.log(`  👤 ${user.name} joined ${roomCode} (${room.users.size} users)`);

      if (typeof cb === "function") {
        cb({ ok: true, code: roomCode, userId: userId, room: { code: roomCode, name: room.name } });
      }
      broadcastRoom(roomCode);
    } catch (e) {
      console.error("room:join error:", e);
      if (typeof cb === "function") cb({ ok: false, error: "Server error" });
    }
  });

  // ── CHAT MESSAGE ─────────────────────────────
  socket.on("chat:send", (payload) => {
    try {
      if (!myRoom || !myUserId) return;
      const room = rooms.get(myRoom);
      if (!room) return;
      const user = room.users.get(myUserId);
      if (!user) return;

      const text = ((payload && payload.text) || "").slice(0, 2000).trim();
      if (!text) return;

      const msg = {
        id: uuidv4(),
        type: "user",
        userId: myUserId,
        userName: user.name,
        userColor: user.color,
        text: text,
        time: Date.now(),
      };

      room.messages.push(msg);

      // Keep messages bounded
      if (room.messages.length > 500) {
        room.messages = room.messages.slice(-300);
      }

      io.to(myRoom).emit("chat:message", msg);
    } catch (e) {
      console.error("chat:send error:", e);
    }
  });

  // ── ROOM SETTINGS ────────────────────────────
  socket.on("room:settings", (payload) => {
    try {
      if (!myRoom || !myUserId) return;
      const room = rooms.get(myRoom);
      if (!room || room.creatorId !== myUserId) return;

      const maxCall = payload && payload.maxCall;
      if (maxCall >= 2 && maxCall <= 8) {
        room.maxCall = maxCall;
        systemMsg(room, `Max call participants changed to ${maxCall}`);
        broadcastRoom(myRoom);
      }
    } catch (e) {
      console.error("room:settings error:", e);
    }
  });

  // ── CALL JOIN ────────────────────────────────
  socket.on("call:join", (_, cb) => {
    try {
      if (!myRoom || !myUserId) return;
      const room = rooms.get(myRoom);
      if (!room) return;

      if (room.callMembers.length >= room.maxCall) {
        if (typeof cb === "function") cb({ ok: false, error: `Call is full (max ${room.maxCall})` });
        return;
      }

      if (!room.callMembers.includes(myUserId)) {
        room.callMembers.push(myUserId);
      }
      room.callActive = room.callMembers.length > 0;

      const user = room.users.get(myUserId);
      systemMsg(room, `${user.name} joined the call`);

      // Notify existing call members about new peer
      for (const uid of room.callMembers) {
        if (uid === myUserId) continue;
        const other = room.users.get(uid);
        if (!other || !other.socketId) continue;

        socket.emit("call:peer-joined", {
          peerId: uid,
          peerName: other.name,
          peerColor: other.color,
          shouldOffer: true,
        });

        io.to(other.socketId).emit("call:peer-joined", {
          peerId: myUserId,
          peerName: user.name,
          peerColor: user.color,
          shouldOffer: false,
        });
      }

      console.log(`  📞 ${user.name} joined call in ${myRoom} (${room.callMembers.length}/${room.maxCall})`);

      if (typeof cb === "function") cb({ ok: true, members: room.callMembers.length });
      broadcastRoom(myRoom);
    } catch (e) {
      console.error("call:join error:", e);
      if (typeof cb === "function") cb({ ok: false, error: "Server error" });
    }
  });

  // ── CALL LEAVE ───────────────────────────────
  socket.on("call:leave", () => {
    handleCallLeave();
  });

  function handleCallLeave() {
    try {
      if (!myRoom || !myUserId) return;
      const room = rooms.get(myRoom);
      if (!room) return;

      const idx = room.callMembers.indexOf(myUserId);
      if (idx === -1) return;

      room.callMembers.splice(idx, 1);
      room.callActive = room.callMembers.length > 0;

      const user = room.users.get(myUserId);
      const userName = user ? user.name : "Someone";
      systemMsg(room, `${userName} left the call`);

      for (const uid of room.callMembers) {
        const other = room.users.get(uid);
        if (other && other.socketId) {
          io.to(other.socketId).emit("call:peer-left", { peerId: myUserId });
        }
      }

      console.log(`  📵 ${userName} left call in ${myRoom}`);
      broadcastRoom(myRoom);
    } catch (e) {
      console.error("handleCallLeave error:", e);
    }
  }

  // ── WebRTC SIGNALING ─────────────────────────
  socket.on("rtc:offer", (payload) => {
    try {
      if (!myRoom) return;
      const room = rooms.get(myRoom);
      if (!room) return;
      const target = room.users.get(payload.targetUserId);
      if (target && target.socketId) {
        io.to(target.socketId).emit("rtc:offer", {
          fromUserId: myUserId,
          offer: payload.offer,
        });
      }
    } catch (e) {
      console.error("rtc:offer error:", e);
    }
  });

  socket.on("rtc:answer", (payload) => {
    try {
      if (!myRoom) return;
      const room = rooms.get(myRoom);
      if (!room) return;
      const target = room.users.get(payload.targetUserId);
      if (target && target.socketId) {
        io.to(target.socketId).emit("rtc:answer", {
          fromUserId: myUserId,
          answer: payload.answer,
        });
      }
    } catch (e) {
      console.error("rtc:answer error:", e);
    }
  });

  socket.on("rtc:ice", (payload) => {
    try {
      if (!myRoom) return;
      const room = rooms.get(myRoom);
      if (!room) return;
      const target = room.users.get(payload.targetUserId);
      if (target && target.socketId) {
        io.to(target.socketId).emit("rtc:ice", {
          fromUserId: myUserId,
          candidate: payload.candidate,
        });
      }
    } catch (e) {
      console.error("rtc:ice error:", e);
    }
  });

  // ── NETWORK QUALITY ──────────────────────────
  socket.on("net:quality", (payload) => {
    try {
      if (!myRoom || !myUserId) return;
      const room = rooms.get(myRoom);
      if (!room) return;
      for (const uid of room.callMembers) {
        if (uid === myUserId) continue;
        const other = room.users.get(uid);
        if (other && other.socketId) {
          io.to(other.socketId).emit("net:peer-quality", {
            peerId: myUserId,
            quality: payload.quality,
          });
        }
      }
    } catch (e) {
      console.error("net:quality error:", e);
    }
  });

  // ── TYPING ───────────────────────────────────
  socket.on("chat:typing", () => {
    try {
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
    } catch (e) {}
  });

  // ── DISCONNECT ───────────────────────────────
  socket.on("disconnect", (reason) => {
    console.log(`🔌 ${socket.id} disconnected (${reason})`);

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
          const roomToClean = myRoom;
          setTimeout(() => {
            const r = rooms.get(roomToClean);
            if (r) {
              const still = [...r.users.values()].filter((u) => u.online).length;
              if (still === 0) {
                rooms.delete(roomToClean);
                console.log(`  🗑️ Room ${roomToClean} cleaned up`);
              }
            }
          }, 1800000);
        }
      }
    }

    socketToUser.delete(socket.id);
  });

  // ── ERROR HANDLER ────────────────────────────
  socket.on("error", (err) => {
    console.error(`Socket error for ${socket.id}:`, err);
  });
});

// ━━ GLOBAL ERROR HANDLERS ━━━━━━━━━━━━━━━━━━━━━
process.on("uncaughtException", (err) => {
  console.error("❌ Uncaught exception:", err);
});

process.on("unhandledRejection", (err) => {
  console.error("❌ Unhandled rejection:", err);
});

// ━━ START ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
server.listen(PORT, "0.0.0.0", () => {
  console.log("");
  console.log("╔══════════════════════════════════════╗");
  console.log("║   📱 VoxRoom v1.1                    ║");
  console.log(`║   Listening on 0.0.0.0:${PORT}          ║`);
  console.log(`║   ${new Date().toISOString()}  ║`);
  console.log("╚══════════════════════════════════════╝");
  console.log("");
});

server.on("error", (err) => {
  console.error("❌ Server error:", err);
  if (err.code === "EADDRINUSE") {
    console.error(`Port ${PORT} is already in use!`);
    process.exit(1);
  }
});
