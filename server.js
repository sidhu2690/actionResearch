const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const path = require("path");

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "*" },
  transports: ["websocket", "polling"],
});

app.use(express.static(path.join(__dirname, "public")));
const PORT = process.env.PORT || 8080;

// Track rooms: code → Set of socket ids
const rooms = new Map();

io.on("connection", (socket) => {
  let myRoom = null;

  socket.on("join", (code, cb) => {
    code = (code || "").toUpperCase().trim();
    if (!code || code.length < 3) {
      cb({ ok: false, error: "Code too short" });
      return;
    }

    if (!rooms.has(code)) rooms.set(code, new Set());
    const room = rooms.get(code);

    if (room.size >= 2) {
      cb({ ok: false, error: "Room full" });
      return;
    }

    room.add(socket.id);
    socket.join(code);
    myRoom = code;

    console.log(`📞 ${socket.id} joined ${code} (${room.size}/2)`);
    cb({ ok: true, count: room.size });

    // Tell everyone in room the count
    io.to(code).emit("room-count", room.size);

    // If 2 people, tell the joiner to make the offer
    if (room.size === 2) {
      socket.emit("start-call", true); // you offer
    }
  });

  // WebRTC signaling — just relay
  socket.on("signal", (data) => {
    if (myRoom) socket.to(myRoom).emit("signal", data);
  });

  socket.on("disconnect", () => {
    if (myRoom && rooms.has(myRoom)) {
      const room = rooms.get(myRoom);
      room.delete(socket.id);
      io.to(myRoom).emit("room-count", room.size);
      io.to(myRoom).emit("peer-left");
      console.log(`📵 ${socket.id} left ${myRoom} (${room.size}/2)`);
      if (room.size === 0) rooms.delete(myRoom);
    }
  });
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`📞 VoxCall running on port ${PORT}`);
});
