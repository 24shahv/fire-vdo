/* ============================================================================
 * SAFEX AI — browser client
 *
 * Owns the camera, ships frames to the backend, and paints the results back
 * over the live video. The server never touches a capture device.
 *
 * Frame path:
 *   getUserMedia -> <video> -> hidden 640x480 canvas (mirrored) -> JPEG blob
 *   -> WebSocket -> YOLO -> JSON -> overlay canvas + minimap + speech
 *
 * Sending is strictly request/response: the next frame is only grabbed once
 * the previous verdict lands. That single rule is what keeps latency bounded
 * on a small Render instance instead of building a queue that never drains.
 * ========================================================================== */

(() => {
  "use strict";

  // ── config ───────────────────────────────────────────────────────────────
  const FRAME_W = 640;
  const FRAME_H = 480;
  const JPEG_QUALITY = 0.6;
  const RECONNECT_MS = 1500;

  const COLORS = {
    exit:   "#00E06A",
    fire:   "#FF3B21",
    smoke:  "#FFB020",
    person: "#00E06A",
    route:  "#FFB020",
    dot:    "#4DA3FF",
    wall:   "#3A4A45",
    free:   "#16211D",
    ink:    "#E9F1ED",
  };

  // ── element refs ─────────────────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);

  const video       = $("video");
  const overlay     = $("overlay");
  const minimap     = $("minimap");
  const gate        = $("gate");
  const gateNote    = $("gateNote");
  const startBtn    = $("startBtn");
  const stopBtn     = $("stopBtn");
  const copyBtn     = $("copyBtn");
  const deviceSel   = $("deviceSelect");
  const camIdSel    = $("camIdSelect");
  const rateSel     = $("rateSelect");
  const voiceToggle = $("voiceToggle");
  const shareLink   = $("shareLink");

  const linkDot   = $("linkDot");
  const linkLabel = $("linkLabel");
  const directive = $("directive");
  const dirEyebrow= $("directiveEyebrow");
  const dirText   = $("directiveText");

  const elTotal   = $("totalPeople");
  const elCamPpl  = $("camPeople");
  const elCamCnt  = $("camCount");
  const elFps     = $("fps");
  const elLoadA   = $("loadA");
  const elLoadB   = $("loadB");
  const elOverride= $("overrideNote");
  const elSmokePx = $("smokePixels");
  const elLatency = $("latency");
  const elTransport = $("transport");
  const camBadge  = $("camBadge");
  const fireBadge = $("fireBadge");
  const smokeBadge= $("smokeBadge");

  const octx = overlay.getContext("2d");
  const mctx = minimap.getContext("2d");

  // Off-screen capture surface. Reused forever — allocating a canvas per frame
  // is the classic way to make a tab's memory climb until it dies.
  const capture = document.createElement("canvas");
  capture.width = FRAME_W;
  capture.height = FRAME_H;
  const cctx = capture.getContext("2d", { alpha: false, willReadFrequently: false });

  // ── state ────────────────────────────────────────────────────────────────
  const params = new URLSearchParams(location.search);

  let sessionId = params.get("session") || randomId();
  let camId = parseInt(params.get("cam") || "0", 10) || 0;

  let stream = null;
  let socket = null;
  let running = false;
  let inFlight = false;
  let reconnectTimer = null;
  let loopHandle = null;
  let useRestFallback = false;
  let wsFailures = 0;

  let lastSendAt = 0;
  let frameTimes = [];
  let lastSpoken = 0;

  // ── boot ─────────────────────────────────────────────────────────────────
  camIdSel.value = String(camId);
  camBadge.textContent = `CAM ${camId}`;
  updateShareLink();
  drawEmptyMinimap();
  syncUrl();

  startBtn.addEventListener("click", start);
  stopBtn.addEventListener("click", stop);

  camIdSel.addEventListener("change", () => {
    camId = parseInt(camIdSel.value, 10) || 0;
    camBadge.textContent = `CAM ${camId}`;
    syncUrl();
    updateShareLink();
    if (running) { closeSocket(); openSocket(); }
  });

  deviceSel.addEventListener("change", () => { if (running) restartStream(); });

  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(shareLink.value);
      copyBtn.textContent = "Copied";
      setTimeout(() => (copyBtn.textContent = "Copy"), 1400);
    } catch {
      shareLink.select();
    }
  });

  document.addEventListener("visibilitychange", () => {
    // Throttling a hidden tab keeps a backgrounded phone from burning the
    // server's CPU on frames nobody is looking at.
    if (document.hidden) setStatus("busy", "Paused");
    else if (running) setStatus("live", "Live");
  });

  window.addEventListener("beforeunload", stop);

  // ── camera ───────────────────────────────────────────────────────────────
  async function start() {
    gateNote.textContent = "";

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      gateNote.textContent =
        "This browser has no camera API. Try a current Chrome, Edge, Firefox or Safari.";
      return;
    }

    if (!window.isSecureContext) {
      gateNote.textContent =
        "Camera access needs HTTPS. Open the site over https:// (or localhost).";
      return;
    }

    startBtn.disabled = true;
    startBtn.textContent = "Requesting…";

    try {
      await openStream();
    } catch (err) {
      startBtn.disabled = false;
      startBtn.textContent = "Start camera";
      gateNote.textContent = describeCameraError(err);
      setStatus("error", "No camera");
      return;
    }

    gate.hidden = true;
    stopBtn.disabled = false;
    running = true;

    // Speech synthesis unlocks on a user gesture — prime it here.
    primeSpeech();

    openSocket();
    scheduleNextFrame();
    await listDevices();
  }

  async function openStream() {
    const deviceId = deviceSel.value;

    const constraints = {
      audio: false,
      video: deviceId
        ? { deviceId: { exact: deviceId }, width: { ideal: FRAME_W }, height: { ideal: FRAME_H } }
        : { facingMode: "environment", width: { ideal: FRAME_W }, height: { ideal: FRAME_H } },
    };

    let media;
    try {
      media = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err) {
      // Desktops have no "environment" camera; retry without the hint.
      if (!deviceId && (err.name === "OverconstrainedError" || err.name === "NotFoundError")) {
        media = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { width: { ideal: FRAME_W }, height: { ideal: FRAME_H } },
        });
      } else {
        throw err;
      }
    }

    releaseStream();
    stream = media;
    video.srcObject = stream;

    await video.play().catch(() => {});
    await waitForVideo();
  }

  async function restartStream() {
    try {
      await openStream();
    } catch (err) {
      gateNote.textContent = describeCameraError(err);
    }
  }

  function waitForVideo() {
    if (video.readyState >= 2 && video.videoWidth) return Promise.resolve();
    return new Promise((resolve) => {
      const done = () => { video.removeEventListener("loadeddata", done); resolve(); };
      video.addEventListener("loadeddata", done);
      setTimeout(resolve, 3000);
    });
  }

  async function listDevices() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cams = devices.filter((d) => d.kind === "videoinput");
      const current = deviceSel.value;

      deviceSel.innerHTML = '<option value="">Default</option>';
      cams.forEach((d, i) => {
        const opt = document.createElement("option");
        opt.value = d.deviceId;
        opt.textContent = d.label || `Camera ${i + 1}`;
        deviceSel.appendChild(opt);
      });
      deviceSel.value = current;
    } catch { /* label enumeration is best-effort */ }
  }

  function releaseStream() {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
  }

  function stop() {
    running = false;
    inFlight = false;

    if (loopHandle) { clearTimeout(loopHandle); loopHandle = null; }
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }

    closeSocket();
    releaseStream();

    video.srcObject = null;
    octx.clearRect(0, 0, overlay.width, overlay.height);

    gate.hidden = false;
    startBtn.disabled = false;
    startBtn.textContent = "Start camera";
    stopBtn.disabled = true;

    setStatus("idle", "Standby");
    setDirective(null);
    drawEmptyMinimap();
  }

  // ── transport ────────────────────────────────────────────────────────────
  function wsUrl() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${location.host}/ws?session=${encodeURIComponent(sessionId)}&cam=${camId}`;
  }

  function openSocket() {
    if (useRestFallback) { elTransport.textContent = "HTTP"; return; }

    try {
      socket = new WebSocket(wsUrl());
    } catch {
      useRestFallback = true;
      elTransport.textContent = "HTTP";
      return;
    }

    socket.binaryType = "arraybuffer";

    socket.addEventListener("open", () => {
      wsFailures = 0;
      inFlight = false;
      elTransport.textContent = "WebSocket";
      setStatus("live", "Live");
    });

    socket.addEventListener("message", (ev) => {
      inFlight = false;
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }

      if (msg.type === "ready") { sessionId = msg.session_id; syncUrl(); updateShareLink(); return; }
      if (msg.type === "pong") return;
      if (msg.type === "error") { console.warn("SAFEX:", msg.error); return; }

      handleResult(msg);
    });

    socket.addEventListener("close", () => {
      inFlight = false;
      if (!running) return;

      wsFailures += 1;
      if (wsFailures >= 3) {
        // Some proxies and corporate networks kill WebSockets. Degrade to
        // plain POSTs rather than leaving the operator with a dead console.
        useRestFallback = true;
        elTransport.textContent = "HTTP";
        setStatus("busy", "Live (HTTP)");
        return;
      }

      setStatus("busy", "Reconnecting");
      reconnectTimer = setTimeout(() => { if (running) openSocket(); }, RECONNECT_MS);
    });

    socket.addEventListener("error", () => { /* close handler does the work */ });
  }

  function closeSocket() {
    if (socket) {
      try { socket.onclose = null; socket.close(); } catch { /* already gone */ }
      socket = null;
    }
  }

  // ── capture loop ─────────────────────────────────────────────────────────
  function targetInterval() {
    const fps = parseInt(rateSel.value, 10);
    return fps > 0 ? 1000 / fps : 0;
  }

  function scheduleNextFrame() {
    if (!running) return;

    const wait = Math.max(0, targetInterval() - (performance.now() - lastSendAt));
    loopHandle = setTimeout(tick, wait);
  }

  async function tick() {
    if (!running) return;

    if (document.hidden || inFlight || !video.videoWidth) {
      scheduleNextFrame();
      return;
    }

    lastSendAt = performance.now();

    try {
      const blob = await grabFrame();
      if (blob) await sendFrame(blob);
    } catch (err) {
      console.warn("SAFEX capture error", err);
      inFlight = false;
    }

    scheduleNextFrame();
  }

  function grabFrame() {
    // Mirror while drawing. This reproduces the cv2.flip(frame, 1) the old
    // CameraManager did, so detections land in the same space the user sees.
    cctx.save();
    cctx.translate(FRAME_W, 0);
    cctx.scale(-1, 1);
    cctx.drawImage(video, 0, 0, FRAME_W, FRAME_H);
    cctx.restore();

    return new Promise((resolve) => {
      capture.toBlob((blob) => resolve(blob), "image/jpeg", JPEG_QUALITY);
    });
  }

  async function sendFrame(blob) {
    if (!useRestFallback && socket && socket.readyState === WebSocket.OPEN) {
      inFlight = true;
      const buf = await blob.arrayBuffer();
      socket.send(buf);
      return;
    }

    if (!useRestFallback) return; // socket still connecting

    inFlight = true;
    try {
      const form = new FormData();
      form.append("file", blob, "frame.jpg");
      form.append("session_id", sessionId);
      form.append("cam_id", String(camId));

      const res = await fetch("/api/detect", { method: "POST", body: form });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      handleResult(await res.json());
    } catch (err) {
      console.warn("SAFEX POST failed", err);
      setStatus("error", "Server unreachable");
    } finally {
      inFlight = false;
    }
  }

  // ── results ──────────────────────────────────────────────────────────────
  function handleResult(data) {
    if (!data || !data.ok) return;

    if (data.session_id && data.session_id !== sessionId) {
      sessionId = data.session_id;
      syncUrl();
      updateShareLink();
    }

    trackFps();
    drawOverlay(data);
    drawMinimap(data);
    updateTelemetry(data);

    if (data.announce) speak(data.announce);
  }

  function trackFps() {
    const now = performance.now();
    frameTimes.push(now);
    while (frameTimes.length && now - frameTimes[0] > 1000) frameTimes.shift();
    elFps.innerHTML = `${frameTimes.length}<small>fps</small>`;
  }

  function drawOverlay(data) {
    const w = (data.frame && data.frame.w) || FRAME_W;
    const h = (data.frame && data.frame.h) || FRAME_H;

    if (overlay.width !== w || overlay.height !== h) {
      overlay.width = w;
      overlay.height = h;
    }

    octx.clearRect(0, 0, w, h);
    octx.lineJoin = "round";

    // people
    octx.strokeStyle = COLORS.person;
    octx.lineWidth = 2;
    for (const p of data.detections.people) {
      const [x1, y1, x2, y2] = p.box;
      octx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      tag(`PERSON ${(p.conf * 100).toFixed(0)}%`, x1, y1, COLORS.person, "#04140C");
    }

    // fire / smoke
    octx.strokeStyle = COLORS.fire;
    octx.lineWidth = 3;
    for (const f of data.detections.fire) {
      const [x1, y1, x2, y2] = f.box;
      octx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      tag(`${f.label.toUpperCase()} ${(f.conf * 100).toFixed(0)}%`, x1, y1, COLORS.fire, "#fff");
    }

    // evacuation arrows
    octx.strokeStyle = COLORS.exit;
    octx.fillStyle = COLORS.exit;
    octx.lineWidth = 3;
    for (const a of data.arrows) arrow(a.from[0], a.from[1], a.to[0], a.to[1]);
  }

  function tag(text, x, y, bg, fg) {
    octx.font = "600 13px 'IBM Plex Mono', monospace";
    const pad = 5;
    const tw = octx.measureText(text).width;
    const th = 17;
    const ty = y - th < 0 ? y + 2 : y - th - 2;

    const prevStroke = octx.strokeStyle;
    octx.fillStyle = bg;
    octx.fillRect(x, ty, tw + pad * 2, th);
    octx.fillStyle = fg;
    octx.fillText(text, x + pad, ty + 12.5);
    octx.strokeStyle = prevStroke;
    octx.fillStyle = bg;
  }

  function arrow(x1, y1, x2, y2) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    if (!dx && !dy) return;

    const ang = Math.atan2(dy, dx);
    const head = 11;

    octx.beginPath();
    octx.moveTo(x1, y1);
    octx.lineTo(x2, y2);
    octx.stroke();

    octx.beginPath();
    octx.moveTo(x2, y2);
    octx.lineTo(x2 - head * Math.cos(ang - Math.PI / 7), y2 - head * Math.sin(ang - Math.PI / 7));
    octx.lineTo(x2 - head * Math.cos(ang + Math.PI / 7), y2 - head * Math.sin(ang + Math.PI / 7));
    octx.closePath();
    octx.fill();
  }

  // ── minimap ──────────────────────────────────────────────────────────────
  function drawEmptyMinimap() {
    mctx.fillStyle = "#060D0A";
    mctx.fillRect(0, 0, minimap.width, minimap.height);
  }

  function drawMinimap(data) {
    const grid = data.grid;
    if (!grid || !grid.length) return;

    const rows = grid.length;
    const cols = grid[0].length;
    const size = minimap.width;
    const cell = size / cols;
    const cellH = minimap.height / rows;

    mctx.clearRect(0, 0, minimap.width, minimap.height);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = grid[r][c];
        mctx.fillStyle = v === 1 ? COLORS.wall : v === 2 ? COLORS.fire : COLORS.free;
        mctx.fillRect(c * cell, r * cellH, cell - 1, cellH - 1);
      }
    }

    // routes
    if (data.routes && data.routes.length) {
      mctx.fillStyle = COLORS.route;
      for (const route of data.routes) {
        for (const [r, c] of route) {
          mctx.fillRect(c * cell + cell * 0.38, r * cellH + cellH * 0.38, cell * 0.24, cellH * 0.24);
        }
      }
    }

    // exits — chosen one is filled, the other outlined
    const chosen = data.exit && data.exit.chosen;
    for (const [er, ec] of (data.exit && data.exit.all) || []) {
      const isChosen = chosen && er === chosen[0] && ec === chosen[1];
      mctx.strokeStyle = COLORS.exit;
      mctx.lineWidth = 2;
      if (isChosen) {
        mctx.fillStyle = COLORS.exit;
        mctx.fillRect(ec * cell, er * cellH, cell - 1, cellH - 1);
      } else {
        mctx.strokeRect(ec * cell + 1, er * cellH + 1, cell - 3, cellH - 3);
      }
    }

    // people
    mctx.fillStyle = COLORS.dot;
    for (const [r, c] of data.grid_people || []) {
      mctx.beginPath();
      mctx.arc(c * cell + cell / 2, r * cellH + cellH / 2, Math.min(cell, cellH) * 0.22, 0, Math.PI * 2);
      mctx.fill();
    }
  }

  // ── telemetry + directive ────────────────────────────────────────────────
  function updateTelemetry(data) {
    const fire = data.fire.any_camera;
    const smoke = data.smoke.any_camera;

    elTotal.textContent = data.counts.total_people;
    elCamPpl.textContent = data.counts.camera_people;
    elCamCnt.textContent = data.counts.cameras;

    elTotal.dataset.alarm = fire ? "1" : "0";

    elSmokePx.textContent = `${data.smoke.pixels.toLocaleString()} / ${data.smoke.threshold.toLocaleString()}`;
    elLatency.textContent = `${data.timing_ms} ms`;

    fireBadge.hidden = !fire;
    smokeBadge.hidden = !smoke;

    const load = data.exit.load || {};
    const exits = data.exit.all || [];
    if (exits[0]) elLoadA.textContent = fmtLoad(load[`${exits[0][0]},${exits[0][1]}`]);
    if (exits[1]) elLoadB.textContent = fmtLoad(load[`${exits[1][0]},${exits[1][1]}`]);

    elOverride.hidden = !data.exit.override;

    setDirective(data);
  }

  function fmtLoad(v) {
    if (v === undefined || v === null) return "—";
    return v >= 999 ? "blocked" : String(v);
  }

  function setDirective(data) {
    if (!data) {
      directive.dataset.mode = "idle";
      directive.dataset.alarm = "0";
      dirEyebrow.textContent = "Awaiting camera";
      dirText.textContent = "SYSTEM IDLE";
      return;
    }

    const fire = data.fire.any_camera;
    directive.dataset.mode = data.exit.name === "A" ? "a" : "b";
    directive.dataset.alarm = fire ? "1" : "0";

    dirText.textContent = data.exit.text;

    if (fire) {
      dirEyebrow.textContent = "Fire detected — evacuate now";
    } else if (data.exit.override) {
      dirEyebrow.textContent = `Crowd routing · ${data.counts.total_people} people tracked`;
    } else if (data.smoke.any_camera) {
      dirEyebrow.textContent = "Smoke signature present";
    } else {
      dirEyebrow.textContent = "All clear · routing by exit load";
    }
  }

  function setStatus(state, label) {
    linkDot.dataset.state = state;
    linkLabel.textContent = label;
  }

  // ── speech (replaces the server-side pyttsx3 engine) ─────────────────────
  function primeSpeech() {
    if (!("speechSynthesis" in window)) return;
    try {
      const u = new SpeechSynthesisUtterance("");
      u.volume = 0;
      window.speechSynthesis.speak(u);
    } catch { /* not fatal */ }
  }

  function speak(message) {
    if (!voiceToggle.checked) return;
    if (!("speechSynthesis" in window)) return;

    // The server already rate-limits announcements; this is belt and braces
    // for the case where several cameras report at once.
    const now = Date.now();
    if (now - lastSpoken < 4000) return;
    lastSpoken = now;

    try {
      window.speechSynthesis.cancel();
      const utter = new SpeechSynthesisUtterance(message);
      utter.rate = 1.0;
      utter.pitch = 1.0;
      utter.volume = 1.0;
      window.speechSynthesis.speak(utter);
    } catch (err) {
      console.warn("SAFEX speech failed", err);
    }
  }

  // ── helpers ──────────────────────────────────────────────────────────────
  function randomId() {
    if (window.crypto && crypto.randomUUID) return crypto.randomUUID().replace(/-/g, "").slice(0, 16);
    return Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-6);
  }

  function syncUrl() {
    const url = new URL(location.href);
    url.searchParams.set("session", sessionId);
    url.searchParams.set("cam", String(camId));
    history.replaceState(null, "", url);
  }

  function updateShareLink() {
    const url = new URL(location.href);
    url.searchParams.set("session", sessionId);
    url.searchParams.set("cam", String(camId + 1));
    shareLink.value = url.toString();
  }

  function describeCameraError(err) {
    switch (err && err.name) {
      case "NotAllowedError":
      case "SecurityError":
        return "Camera permission was blocked. Allow it in the address-bar icon, then start again.";
      case "NotFoundError":
      case "OverconstrainedError":
        return "No camera found on this device.";
      case "NotReadableError":
        return "Another app is already using the camera. Close it and try again.";
      default:
        return `Camera failed to start: ${(err && err.message) || "unknown error"}`;
    }
  }
})();
