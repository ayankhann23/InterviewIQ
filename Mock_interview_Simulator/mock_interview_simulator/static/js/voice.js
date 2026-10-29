// static/js/voice.js
// Owner: Sarthak
// Text-to-speech (AI interviewer speaks) + speech-to-text (candidate answers).
// Uses browser built-in features. Works best in Google Chrome.

const VoiceModule = (() => {
  const synth = window.speechSynthesis;
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  let voices = [];
  let recognition = null;
  let listening = false;
  let manualStop = false;      // true when WE stop it (button/silence/pause)
  let finalText = "";          // all confirmed words of the current answer
  let discardCurrent = false;
  let silenceTimer = null;
  let resolveListen = null;
  let options = {};
  const listeners = {};

  // ---------- tiny event system ----------
  function on(eventName, fn) {
    (listeners[eventName] = listeners[eventName] || []).push(fn);
  }
  function emit(eventName, data) {
    (listeners[eventName] || []).forEach((fn) => fn(data));
  }

  function isSupported() {
    return { speech: !!synth, recognition: !!SpeechRecognition };
  }

  // ---------- voices (Chrome loads them a moment after the page) ----------
  function loadVoices() {
    voices = synth ? synth.getVoices() : [];
  }
  if (synth) {
    loadVoices();
    synth.addEventListener("voiceschanged", loadVoices);
  }
  function pickVoice() {
    return (
      voices.find((v) => v.lang === "en-IN") ||
      voices.find((v) => v.lang === "en-US") ||
      voices.find((v) => v.lang.startsWith("en")) ||
      null
    );
  }

  // =====================================================
  //  PART 1: AI INTERVIEWER SPEAKS
  // =====================================================
  // Returns a Promise that finishes when the sentence has been spoken.
  // Usage:  await VoiceModule.speak("Tell me about yourself.");
  function speak(text, { rate = 0.95 } = {}) {
    return new Promise((resolve, reject) => {
      if (!synth) {
        reject(new Error("Speech output is not supported in this browser. Please use Google Chrome."));
        return;
      }
      synth.cancel();   // stop anything already being spoken

      const utterance = new SpeechSynthesisUtterance(text);
      const voice = pickVoice();
      if (voice) {
        utterance.voice = voice;
        utterance.lang = voice.lang;
      } else {
        utterance.lang = "en-IN";
      }
      utterance.rate = rate;

      utterance.onstart = () => emit("speaking-start", { text });
      utterance.onend = () => {
        emit("speaking-end", { text });
        resolve();
      };
      utterance.onerror = (e) => {
        if (e.error === "canceled" || e.error === "interrupted") {
          resolve();     // we stopped it on purpose
          return;
        }
        emit("error", { type: e.error, message: "Could not speak the question (" + e.error + ")." });
        reject(new Error("Speech error: " + e.error));
      };

      synth.speak(utterance);
    });
  }

  // =====================================================
  //  PART 2: CANDIDATE ANSWERS BY VOICE
  // =====================================================
  function speechErrorMessage(code) {
    switch (code) {
      case "not-allowed":
      case "service-not-allowed":
        return "Microphone permission is blocked. Allow the microphone in Chrome's address bar and try again.";
      case "audio-capture":
        return "No microphone was found. Please connect one.";
      case "network":
        return "Speech recognition needs an internet connection. Please check your network.";
      default:
        return "Speech recognition error: " + code;
    }
  }

  function resetSilenceTimer() {
    clearTimeout(silenceTimer);
    if (options.silenceMs > 0) {
      silenceTimer = setTimeout(stopListening, options.silenceMs);
    }
  }

  function finish() {
    clearTimeout(silenceTimer);
    listening = false;
    recognition = null;
    if (resolveListen) {
      const text = discardCurrent ? null : finalText.trim();
      discardCurrent = false;
      const done = resolveListen;
      resolveListen = null;
      emit("listening-end", { text });
      done(text);
    }
  }

  // Starts listening. Returns a Promise that gives the full answer text
  // when the candidate stops (Done button, or silence for `silenceMs`).
  // Usage:  const answer = await VoiceModule.startListening({ onInterim: showText });
  function startListening({ onInterim = null, silenceMs = 6000, lang = "en-IN" } = {}) {
    return new Promise((resolve, reject) => {
      if (!SpeechRecognition) {
        reject(new Error("Speech recognition is not supported in this browser. Please use Google Chrome."));
        return;
      }
      if (listening || recognition) {
        reject(new Error("Already listening."));
        return;
      }

      finalText = "";
      discardCurrent = false;
      manualStop = false;
      options = { onInterim, silenceMs };
      resolveListen = resolve;

      recognition = new SpeechRecognition();
      recognition.lang = lang;
      recognition.continuous = true;        // keep listening through pauses
      recognition.interimResults = true;    // show words while the candidate is speaking

      recognition.onstart = () => {
        if (!listening) {
          listening = true;
          emit("listening-start", {});
        }
      };

      recognition.onresult = (event) => {
        let interim = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const result = event.results[i];
          if (result.isFinal) finalText += result[0].transcript + " ";
          else interim += result[0].transcript;
        }
        const fullText = (finalText + interim).trim();
        emit("interim", fullText);
        if (options.onInterim) options.onInterim(fullText);
        resetSilenceTimer();   // silence countdown starts only after the first words
      };

      recognition.onerror = (event) => {
        // "no-speech" and "aborted" are normal; onend will restart listening
        if (event.error === "no-speech" || event.error === "aborted") return;
        emit("error", { type: event.error, message: speechErrorMessage(event.error) });
        discardCurrent = true; // never save partial text from a failed recognition session
        manualStop = true;     // real problem: do not restart
      };

      recognition.onend = () => {
        // Chrome sometimes ends a session by itself. Restart unless we stopped it.
        if (!manualStop && recognition) {
          try {
            recognition.start();
            return;
          } catch (e) { /* fall through to finish */ }
        }
        finish();
      };

      try {
        recognition.start();
      } catch (err) {
        recognition = null;
        resolveListen = null;
        reject(err);
      }
    });
  }

  // Candidate finished answering (Done button / silence / interview paused).
  function stopListening() {
    manualStop = true;
    clearTimeout(silenceTimer);
    if (recognition) {
      try { recognition.stop(); } catch (e) { /* ignore */ }
    }
  }

  // Stop talking AND listening (use when the interview ends or pauses).
  function cancelAll({ discardAnswer = true } = {}) {
    if (synth) synth.cancel();
    if (listening || recognition) discardCurrent = discardAnswer;
    stopListening();
  }

  function isListening() {
    return listening;
  }

  return { on, isSupported, speak, startListening, stopListening, cancelAll, isListening };
})();