// static/js/monitor.js
// Owner: Sarthak
// Watches the camera preview and reports OBSERVABLE events only
// (covered camera, face not visible, multiple faces, face outside frame).
// Posture / head position is feedback only: it never creates a warning.
// Face detection runs inside the browser (MediaPipe). No video is uploaded or stored.

const CameraMonitor = (() => {
  const MP_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
  const MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";

  // ---------- settings (configurable, as the project requires) ----------
  const config = {
    checkIntervalMs: 500,
    coveredBrightnessMax: 25,   // average brightness (0-255) below this = covered
    flatFrameStdMax: 6,         // almost no variation in the image = covered/blank
    sustainSec: {               // how long a problem must last before a warning
      camera_covered: 3,
      face_not_visible: 3,
      multiple_faces: 2,
      face_outside_frame: 3,
    },
    tiltDeg: 15,                // head tilt angle for feedback
    turnRatio: 0.35,            // nose offset vs eye distance for "head turned"
    tooCloseWidth: 0.55,        // face width as share of frame width
    tooFarWidth: 0.15,
    centerMin: 0.3,
    centerMax: 0.7,
  };

  const MESSAGES = {
    camera_covered: "Camera appears covered or blank",
    face_not_visible: "Face not visible",
    multiple_faces: "Multiple faces detected",
    face_outside_frame: "Face is partly outside the camera frame",
  };
  const POSTURE_MESSAGES = {
    too_close: "You are very close to the camera. Sit a little further back.",
    too_far: "You are far from the camera. Move a little closer.",
    off_center: "Move to the centre of the frame.",
    head_tilted: "Your head looks tilted. Try to keep it level.",
    head_turned: "Your head is turned to the side. Face the screen when possible.",
  };

  // ---------- internal state ----------
  let detector = null;
  let video = null;
  let timer = null;
  let running = false;
  let paused = false;
  let starting = false;
  let startMs = 0;                 // first-ever monitoring start for the session
  let endMs = null;
  let activeStartedMs = null;      // current active segment; null while paused/stopped
  let activeAccumMs = 0;            // active monitoring time excluding pauses
  let samples = 0;
  let faceSamples = 0;
  const postureCounts = {};
  const state = {};       // per warning type: { since, record }
  const events = [];      // every warning raised (used by the report)
  const listeners = {};

  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 48;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  function on(eventName, fn) {
    (listeners[eventName] = listeners[eventName] || []).push(fn);
  }
  function emit(eventName, data) {
    (listeners[eventName] || []).forEach((fn) => fn(data));
  }

  // ---------- load the face detection model (once) ----------
  async function loadModel() {
    if (detector) return;
    const { FaceDetector, FilesetResolver } = await import(MP_BASE + "/+esm");
    const vision = await FilesetResolver.forVisionTasks(MP_BASE + "/wasm");
    detector = await FaceDetector.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
      runningMode: "VIDEO",
      minDetectionConfidence: 0.5,
    });
  }

  // ---------- brightness + contrast of the current frame ----------
  function frameStats() {
    ctx.drawImage(video, 0, 0, 64, 48);
    const d = ctx.getImageData(0, 0, 64, 48).data;
    const n = d.length / 4;
    let sum = 0, sumSq = 0;
    for (let i = 0; i < d.length; i += 4) {
      const g = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      sum += g;
      sumSq += g * g;
    }
    const mean = sum / n;
    const std = Math.sqrt(Math.max(0, sumSq / n - mean * mean));
    return { mean, std };
  }

  // ---------- warning bookkeeping ----------
  function track(type, condition, now) {
    const s = state[type] || (state[type] = { since: null, record: null });
    if (condition) {
      if (s.since === null) s.since = now;
      if (!s.record && (now - s.since) / 1000 >= config.sustainSec[type]) {
        s.record = { type, message: MESSAGES[type], startMs: s.since, endMs: null };
        events.push(s.record);
        emit("warning", {
          type,
          message: MESSAGES[type],
          time: new Date(s.since).toLocaleTimeString(),
        });
      }
    } else {
      closeWarning(type, now);
    }
  }
  function closeWarning(type, now) {
    const s = state[type];
    if (!s) return;
    if (s.record) {
      s.record.endMs = now;
      emit("warning-cleared", { type });
    }
    s.since = null;
    s.record = null;
  }
  function closeAll(now) {
    Object.keys(state).forEach((t) => closeWarning(t, now));
  }

  // ---------- one check (runs every checkIntervalMs) ----------
  function check() {
    if (!running || paused || !video || video.readyState < 2 || !video.videoWidth) return;
    const now = Date.now();
    const W = video.videoWidth;
    const H = video.videoHeight;
    samples++;

    // 1) covered / blank camera
    const { mean, std } = frameStats();
    const covered = mean < config.coveredBrightnessMax || std < config.flatFrameStdMax;
    track("camera_covered", covered, now);

    // 2) faces (skipped while the camera is covered)
    let faceOk = !!detector && !covered;
    let faces = [];
    if (faceOk) {
      try {
        faces = detector.detectForVideo(video, performance.now()).detections || [];
      } catch (e) {
        faceOk = false;     // detection failed this round: do not raise a false warning
      }
    }
    const count = faces.length;
    track("face_not_visible", faceOk && count === 0, now);
    track("multiple_faces", faceOk && count > 1, now);

    // 3) face position + posture feedback (largest face only)
    let outside = false;
    const hints = [];
    if (faceOk && count > 0) {
      faceSamples++;
      const primary = faces.reduce((a, b) =>
        b.boundingBox.width > a.boundingBox.width ? b : a
      );
      const b = primary.boundingBox;
      const marginX = 0.02 * W;
      const marginY = 0.02 * H;
      outside =
        b.originX < -marginX ||
        b.originY < -marginY ||
        b.originX + b.width > W + marginX ||
        b.originY + b.height > H + marginY;

      const widthShare = b.width / W;
      const centerX = (b.originX + b.width / 2) / W;
      if (widthShare > config.tooCloseWidth) hints.push("too_close");
      if (widthShare < config.tooFarWidth) hints.push("too_far");
      if (centerX < config.centerMin || centerX > config.centerMax) hints.push("off_center");

      // keypoints: 0 and 1 are the eyes, 2 is the nose tip
      const kp = primary.keypoints;
      if (kp && kp.length >= 3) {
        const nx = (k) => (k.x > 1 ? k.x / W : k.x);   // normalise just in case
        const ny = (k) => (k.y > 1 ? k.y / H : k.y);
        const eyes = [kp[0], kp[1]].sort((p, q) => nx(p) - nx(q));
        const dx = (nx(eyes[1]) - nx(eyes[0])) * W;
        const dy = (ny(eyes[1]) - ny(eyes[0])) * H;
        const angle = Math.abs((Math.atan2(dy, dx) * 180) / Math.PI);
        if (angle > config.tiltDeg) hints.push("head_tilted");
        const eyeDist = nx(eyes[1]) - nx(eyes[0]);
        const midX = (nx(eyes[0]) + nx(eyes[1])) / 2;
        if (eyeDist > 0 && Math.abs(nx(kp[2]) - midX) / eyeDist > config.turnRatio) {
          hints.push("head_turned");
        }
      }
      hints.forEach((h) => (postureCounts[h] = (postureCounts[h] || 0) + 1));
    }
    track("face_outside_frame", outside, now);

    emit("status", {
      faces: faceOk ? count : null,
      covered,
      brightness: Math.round(mean),
      contrast: Math.round(std),
      hints,
      messages: hints.map((h) => POSTURE_MESSAGES[h]),
    });
  }

  // ---------- public controls ----------
  async function start(videoEl) {
    if (running || starting) return;
    starting = true;
    video = videoEl;
    try {
      emit("loading", {});
      try {
        await loadModel();
      } catch (e) {
        // Still run the covered-camera check even without face detection
        emit("error", {
          message: "Could not load the face detection model (internet needed). Only the covered-camera check will run.",
        });
      }
      // Preserve event history and counters across stop/restart so the final
      // report represents the whole interview monitoring session.
      if (!startMs) startMs = Date.now();
      endMs = null;
      paused = false;
      activeStartedMs = Date.now();
      running = true;
      timer = setInterval(check, config.checkIntervalMs);
      emit("started", { faceDetection: !!detector });
    } finally {
      starting = false;
    }
  }

  function stop() {
    if (!running) return;
    clearInterval(timer);
    timer = null;
    const now = Date.now();
    closeAll(now);
    if (activeStartedMs !== null) {
      activeAccumMs += Math.max(0, now - activeStartedMs);
      activeStartedMs = null;
    }
    running = false;
    paused = false;
    endMs = now;
    emit("stopped", {});
  }

  // Used when a device disconnects (Phase 4): no warnings while paused
  function pause() {
    if (!running || paused) return;
    const now = Date.now();
    closeAll(now);
    if (activeStartedMs !== null) {
      activeAccumMs += Math.max(0, now - activeStartedMs);
      activeStartedMs = null;
    }
    paused = true;
  }
  function resume() {
    if (!running || !paused) return;
    paused = false;
    activeStartedMs = Date.now();
  }

  // Data for the final report page
  function getSummary() {
    const now = Date.now();
    const pct = (n, d) => (d ? Math.round((100 * n) / d) : 0);
    const posture = {};
    Object.keys(POSTURE_MESSAGES).forEach((h) => {
      posture[h] = pct(postureCounts[h] || 0, faceSamples);
    });
    const activeMs = activeAccumMs + (activeStartedMs !== null ? Math.max(0, now - activeStartedMs) : 0);
    return {
      monitoredSec: Math.round(activeMs / 1000),
      faceDetectionUsed: !!detector,
      faceVisiblePct: detector ? pct(faceSamples, samples) : null,
      warnings: events.map((e) => ({
        type: e.type,
        message: e.message,
        time: new Date(e.startMs).toLocaleTimeString(),
        durationSec: Math.round(((e.endMs || now) - e.startMs) / 1000),
      })),
      posturePctOfTime: posture,
      note: "These are observations only. They do not prove misconduct.",
    };
  }

  return { on, start, stop, pause, resume, getSummary, config, isRunning: () => running };
})();