// static/js/camera.js
// Owner: Sarthak
// Handles camera + microphone permission, live preview, mic level meter,
// device-lost detection, and clean shutdown.

const CameraModule = (() => {
  let stream = null;
  let audioContext = null;
  let analyser = null;
  let meterFrame = null;
  let stopping = false;      // true while WE are stopping (not a disconnect)
  let starting = false;      // true while permission/preview is being started
  const listeners = {};

  // ---------- tiny event system ----------
  function on(eventName, fn) {
    (listeners[eventName] = listeners[eventName] || []).push(fn);
  }
  function emit(eventName, data) {
    (listeners[eventName] || []).forEach((fn) => fn(data));
  }

  // ---------- friendly error messages ----------
  function friendlyError(err) {
    switch (err.name) {
      case "NotAllowedError":
        return "Permission denied. Click the lock icon in the address bar, allow camera and microphone, then try again.";
      case "NotFoundError":
        return "No camera or microphone was found. Please connect one and try again.";
      case "NotReadableError":
        return "Your camera or microphone is being used by another app. Close it and try again.";
      default:
        return "Could not access camera/microphone: " + err.message;
    }
  }

  // ---------- release any existing stream (used before starting and when stopping) ----------
  function releaseStream() {
    if (meterFrame) cancelAnimationFrame(meterFrame);
    meterFrame = null;
    if (stream) stream.getTracks().forEach((t) => t.stop());   // camera light turns off
    stream = null;
    if (audioContext) audioContext.close();
    audioContext = null;
    analyser = null;
  }

  // ---------- main function: ask permission and start preview ----------
  async function requestPermissions(videoEl) {
    if (starting) return { camera: isActive(), microphone: isActive() };  // ignore extra clicks
    starting = true;

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("This browser does not support camera access. Please use Google Chrome.");
      }

      stopping = true;        // stopping the OLD stream must not count as a disconnect
      releaseStream();        // never keep two streams open

      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" },
          audio: { echoCancellation: true, noiseSuppression: true },
        });
      } catch (err) {
        throw new Error(friendlyError(err));
      }

      stopping = false;
      videoEl.srcObject = stream;
      videoEl.muted = true;   // prevents hearing yourself (echo)
      try {
        await videoEl.play();
      } catch (err) {
        if (err.name !== "AbortError") throw err;   // AbortError is harmless
      }

      watchTracks();
      startMicMeter();
      emit("ready", { camera: true, microphone: true });
      return { camera: true, microphone: true };
    } catch (err) {
      emit("error", { message: err.message || String(err) });
      throw err;
    } finally {
      starting = false;
    }
  }

  // ---------- detect unplugged / lost devices ----------
  function watchTracks() {
    stream.getTracks().forEach((track) => {
      track.addEventListener("ended", () => {
        if (!stopping) emit("device-lost", { kind: track.kind });   // "video" or "audio"
      });
      track.addEventListener("mute", () => {
        if (!stopping) emit("device-muted", { kind: track.kind });
      });
      track.addEventListener("unmute", () => {
        if (!stopping) emit("device-unmuted", { kind: track.kind });
      });
    });
  }

  // ---------- microphone level meter (0-100) ----------
  function startMicMeter() {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(stream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);

    const data = new Uint8Array(analyser.fftSize);
    function tick() {
      if (!analyser) return;
      analyser.getByteTimeDomainData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i++) {
        const v = (data[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / data.length);
      emit("mic-level", Math.min(100, Math.round(rms * 300)));
      meterFrame = requestAnimationFrame(tick);
    }
    tick();
  }

  // ---------- stop EVERYTHING (call when interview ends) ----------
  function stopAll(videoEl) {
    stopping = true;
    releaseStream();
    if (videoEl) videoEl.srcObject = null;
    emit("stopped", {});
  }

  function isActive() {
    return !!stream && stream.getTracks().every((t) => t.readyState === "live");
  }
  function getStream() {
    return stream;
  }

  return { on, requestPermissions, stopAll, isActive, getStream };
})();