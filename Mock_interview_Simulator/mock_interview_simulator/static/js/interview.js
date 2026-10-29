(() => {
  const $ = (id) => document.getElementById(id);
  const video = $('camera-preview');
  const cfg = window.INTERVIEW_CONFIG || {};
  // The server sends question objects {id, text, topic, difficulty, type, kind}. The rest of this file keeps
  // working with the plain text array; questionMeta is spliced in step with it when a follow-up is added.
  const questionMeta = (cfg.questions && cfg.questions.length ? cfg.questions : [
    'Tell me about yourself.', 'Describe a project you built and your role.', 'What would you improve in a project you worked on?'
  ]).map((q) => (typeof q === 'string' ? { id: null, text: q, kind: 'main' } : q));
  const questions = questionMeta.map((q) => q.text);
  let questionIndex = 0;
  let answers = [];
  const evaluations = {};       // question id -> evaluation returned by the server (AI)
  const evaluationErrors = {};  // question id -> safe error message
  let evaluating = false;
  // Hard cap: follow-ups are decided by the server, and the browser enforces the same limit.
  const INITIAL_QUESTION_COUNT = questions.length;
  const MAX_FOLLOW_UPS = Number.isInteger(cfg.maxFollowUps) ? cfg.maxFollowUps : 2;
  const MAX_TOTAL_QUESTIONS = INITIAL_QUESTION_COUNT + MAX_FOLLOW_UPS;
  const normalizeQuestion = (value) => String(value || '').toLowerCase()
    .replace(/^follow-up:\s*/, '').replace(/[^a-z0-9]+/g, ' ').trim();
  const questionKeys = new Set(questions.map(normalizeQuestion));
  let followUpsAdded = 0;
  let monitoringStarted = false;
  let monitoringEverStarted = false;
  let interviewFinished = false;
  let devicesReady = false;
  let speakingQuestion = false;
  const events = [];

  function log(message) {
    const line = `${new Date().toLocaleTimeString()} — ${message}`;
    events.push(line);
    $('event-log').textContent = events.slice(-80).join('\n');
  }
  function status(message) { $('device-status').textContent = message; }
  function updateNext() {
    $('next-question').disabled = evaluating || !answers[questionIndex]?.answer || questionIndex >= questions.length - 1;
  }
  function syncControls() {
    const ready = devicesReady && CameraModule.isActive() && monitoringStarted && !interviewFinished;
    $('speak-question').disabled = !ready || speakingQuestion || VoiceModule.isListening();
    $('start-answer').disabled = !ready || VoiceModule.isListening() || evaluating;
    $('done-answer').disabled = !VoiceModule.isListening();
    $('start-monitor').disabled = !devicesReady || !CameraModule.isActive() || monitoringStarted || interviewFinished;
    $('stop-monitor').disabled = !monitoringStarted || interviewFinished;
    $('start-devices').disabled = interviewFinished;
    updateNext();
  }

  // ---- AI feedback panel (all text inserted with textContent, never innerHTML) ----
  const feedbackBox = () => $('ai-feedback');
  function node(tag, className, text) {
    const n = document.createElement(tag);
    if (className) n.className = className;
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function feedbackMessage(message, isError, withRetry) {
    const box = feedbackBox();
    if (!box) return;
    box.textContent = '';
    box.appendChild(node('p', isError ? 'ai-status ai-error' : 'ai-status', message));
    if (withRetry) {
      const btn = node('button', 'btn-copy', '↻ Retry evaluation');
      btn.type = 'button';
      btn.addEventListener('click', () => evaluateAnswer(questionIndex));
      box.appendChild(btn);
    }
  }
  function addEvalList(box, title, items) {
    if (!items || !items.length) return;
    box.appendChild(node('h4', null, title));
    const ul = node('ul', 'ai-list');
    items.forEach((t) => ul.appendChild(node('li', null, t)));
    box.appendChild(ul);
  }
  function renderEvaluation(ev) {
    const box = feedbackBox();
    if (!box) return;
    box.textContent = '';
    if (!ev) { box.appendChild(node('p', 'ai-status', 'Your AI feedback appears here after you answer.')); return; }
    const head = node('div', 'metric');
    head.appendChild(node('strong', null, ev.score + '/100'));
    head.appendChild(document.createTextNode(ev.status === 'no_answer' ? 'No answer' : 'AI score'));
    box.appendChild(head);
    if (ev.criterion_scores) {
      Object.keys(ev.criterion_scores).forEach((key) => {
        const row = node('div', 'score-row');
        row.appendChild(node('span', null, key.charAt(0).toUpperCase() + key.slice(1)
          + (ev.rubric_weights && ev.rubric_weights[key] ? ' (' + ev.rubric_weights[key] + '%)' : '')));
        const bar = node('div', 'bar');
        const fill = node('span');
        fill.style.width = Math.max(0, Math.min(100, Number(ev.criterion_scores[key]) || 0)) + '%';
        bar.appendChild(fill);
        row.append(bar, node('strong', null, String(ev.criterion_scores[key])));
        box.appendChild(row);
      });
    }
    if (ev.explanation) box.appendChild(node('p', 'feedback', ev.explanation));
    if (ev.correctness) box.appendChild(node('p', 'ai-text', 'Correctness: ' + ev.correctness));
    if (ev.relevance) box.appendChild(node('p', 'ai-text', 'Relevance: ' + ev.relevance));
    addEvalList(box, 'What you did well', ev.strengths);
    addEvalList(box, 'Missing concepts', ev.missing_concepts);
    addEvalList(box, 'Mistakes', ev.mistakes);
    if (ev.model_answer) { box.appendChild(node('h4', null, 'Sample answer')); box.appendChild(node('p', 'ai-text', ev.model_answer)); }
    if (ev.improvement_tip) { box.appendChild(node('h4', null, 'Tip to improve')); box.appendChild(node('p', 'ai-text', ev.improvement_tip)); }
    box.appendChild(node('p', 'ai-note', 'AI-generated practice feedback; it can be wrong and is not a hiring decision.'));
  }
  function showFeedbackFor(index) {
    const id = questionMeta[index] && questionMeta[index].id;
    if (id && evaluations[id]) renderEvaluation(evaluations[id]);
    else if (id && evaluationErrors[id]) feedbackMessage(evaluationErrors[id] + ' Your answer is saved.', true, !!answers[index]?.answer);
    else renderEvaluation(null);
  }
  function showQuestion() {
    const meta = questionMeta[questionIndex] || {};
    const kind = meta.kind === 'follow_up' ? ' · Follow-up' : '';
    $('question-count').textContent = `Question ${questionIndex + 1} of ${questions.length}${kind}`;
    $('question-text').textContent = questions[questionIndex];
    $('live-answer').textContent = 'Your spoken answer will appear here.';
    $('saved-answer').textContent = answers[questionIndex]?.answer || 'No answer saved yet.';
    showFeedbackFor(questionIndex);
    syncControls();
  }
  async function evaluateAnswer(index) {
    const meta = questionMeta[index];
    const entry = answers[index];
    if (!meta || !meta.id || !entry || evaluating || interviewFinished) return;
    evaluating = true;
    delete evaluationErrors[meta.id];
    feedbackMessage('Evaluating your answer with NVIDIA Nemotron… you can read your saved answer meanwhile.', false, false);
    syncControls();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 100000);
    try {
      const response = await fetch('/api/answer/evaluate', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question_id: meta.id, answer: entry.answer }), signal: controller.signal,
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.ok) throw new Error(data.error || 'The AI evaluation is unavailable right now.');
      evaluations[meta.id] = data.evaluation;
      log(`Answer evaluated (${data.evaluation.score}/100)`);
      const fu = data.follow_up;
      if (fu && fu.text && index === questionIndex && followUpsAdded < MAX_FOLLOW_UPS
          && questions.length < MAX_TOTAL_QUESTIONS && !questionKeys.has(normalizeQuestion(fu.text))) {
        questionKeys.add(normalizeQuestion(fu.text));
        followUpsAdded += 1;
        questions.splice(index + 1, 0, fu.text);
        questionMeta.splice(index + 1, 0, fu);
        answers.splice(index + 1, 0, null);
        log(`A follow-up question was added (${followUpsAdded}/${MAX_FOLLOW_UPS})`);
      }
      if (index === questionIndex) {
        renderEvaluation(evaluations[meta.id]);
        $('question-count').textContent = `Question ${questionIndex + 1} of ${questions.length}`
          + (questionMeta[questionIndex].kind === 'follow_up' ? ' · Follow-up' : '');
      }
    } catch (err) {
      const message = err.name === 'AbortError' ? 'The AI evaluation took too long.' : (err.message || 'The AI evaluation failed.');
      evaluationErrors[meta.id] = message;
      log('Evaluation unavailable: ' + message);
      if (index === questionIndex) feedbackMessage(message + ' Your answer is saved and the interview can continue.', true, true);
    } finally {
      clearTimeout(timer);
      evaluating = false;
      syncControls();
    }
  }
  function setPaused(message) {
    devicesReady = false;
    $('mic-level').textContent = 'Microphone level: unavailable';
    if (monitoringStarted) CameraMonitor.pause();
    VoiceModule.cancelAll({ discardAnswer: true });
    status(`${message} Interview paused. Restore the device and click “Allow camera & microphone” to reconnect.`);
    $('monitor-status').textContent = 'Monitoring paused until devices reconnect.';
    log(message);
    syncControls();
  }
  async function startMonitoring() {
    if (!CameraModule.isActive()) throw new Error('Start camera and microphone first.');
    if (CameraMonitor.isRunning()) CameraMonitor.resume();
    else await CameraMonitor.start(video);
    monitoringStarted = true;
    monitoringEverStarted = true;
    log('Monitoring active');
    syncControls();
  }

  $('start-devices').addEventListener('click', async () => {
    try {
      await CameraModule.requestPermissions(video);
      devicesReady = true;
      status('Camera and microphone are ready.');
      $('start-devices').textContent = 'Reconnect / refresh devices';
      $('mic-level').textContent = 'Microphone level: waiting for input';
      if (monitoringEverStarted && CameraMonitor.isRunning()) {
        CameraMonitor.resume();
        monitoringStarted = true;
        $('monitor-status').textContent = 'Monitoring resumed after device reconnection.';
        log('Monitoring resumed after reconnection');
      } else {
        await startMonitoring();
      }
      log('Camera and microphone permissions granted');
      syncControls();
    } catch (err) {
      devicesReady = false;
      status(err.message || String(err));
      log('Device setup failed: ' + (err.message || String(err)));
      syncControls();
    }
  });

  CameraModule.on('mic-level', (level) => {
    $('mic-level').textContent = `Microphone level: ${level}%`;
  });
  CameraModule.on('device-lost', (device) => {
    const kind = device && device.kind;
    setPaused(`${kind === 'video' ? 'Camera' : kind === 'audio' ? 'Microphone' : 'Device'} disconnected.`);
  });
  CameraModule.on('device-muted', (device) => log(`${device.kind || 'Device'} device muted`));
  CameraModule.on('device-unmuted', (device) => log(`${device.kind || 'Device'} device unmuted`));
  CameraModule.on('error', (err) => log('Camera module error: ' + (err.message || err)));

  CameraMonitor.on('loading', () => { $('monitor-status').textContent = 'Loading face-detection model. Internet may be required.'; log('Loading face-detection model'); });
  CameraMonitor.on('started', (data) => { $('monitor-status').textContent = data.faceDetection ? 'Monitoring active; face detection available.' : 'Monitoring active; face detection unavailable, covered-camera check only.'; log('Monitoring started; face detection: ' + data.faceDetection); });
  CameraMonitor.on('status', (data) => {
    $('monitor-status').textContent = `Faces: ${data.faces === null ? 'n/a' : data.faces} | Brightness: ${data.brightness} | Contrast: ${data.contrast}\n${(data.messages || []).join('\n') || (data.covered ? 'Camera may be covered or blank.' : 'Posture tips: looks good')}`;
  });
  CameraMonitor.on('warning', (w) => log('WARNING [' + w.type + ']: ' + w.message));
  CameraMonitor.on('warning-cleared', (w) => log('Warning cleared: ' + w.type));
  CameraMonitor.on('error', (e) => log('Monitor note: ' + e.message));
  CameraMonitor.on('stopped', () => log('Monitoring stopped'));

  $('start-monitor').addEventListener('click', async () => {
    try { await startMonitoring(); }
    catch (err) { status(err.message || String(err)); log('Could not start monitoring: ' + err.message); }
  });
  $('stop-monitor').addEventListener('click', () => {
    CameraMonitor.stop();
    monitoringStarted = false;
    $('monitor-status').textContent = 'Monitoring stopped. Restart monitoring to continue the interview.';
    log('Monitoring stopped by candidate');
    syncControls();
  });

  $('speak-question').addEventListener('click', async () => {
    if (!devicesReady || !monitoringStarted || VoiceModule.isListening() || speakingQuestion) return;
    speakingQuestion = true;
    syncControls();
    try { await VoiceModule.speak(questions[questionIndex]); log('Question spoken'); }
    catch (err) { status(err.message || String(err)); log('Speech output issue: ' + err.message); }
    finally { speakingQuestion = false; syncControls(); }
  });
  $('start-answer').addEventListener('click', async () => {
    if (!devicesReady || !CameraModule.isActive() || !monitoringStarted || interviewFinished) {
      status('Camera, microphone and monitoring must be ready before answering.'); return;
    }
    $('start-answer').disabled = true;
    $('speak-question').disabled = true;
    $('done-answer').disabled = false;
    $('live-answer').textContent = 'Listening…';
    log('Answer listening started');
    syncControls();
    try {
      const answer = await VoiceModule.startListening({
        onInterim: (text) => { $('live-answer').textContent = text || 'Listening…'; },
        silenceMs: 6000, lang: 'en-IN'
      });
      if (interviewFinished || answer === null) {
        $('live-answer').textContent = 'Answer discarded because the interview was paused or ended. Please answer again.';
        log('Incomplete answer discarded');
        return;
      }
      const savedIndex = questionIndex;
      const previousId = questionMeta[savedIndex].id;
      if (previousId) { delete evaluations[previousId]; delete evaluationErrors[previousId]; }
      answers[savedIndex] = { question: questions[savedIndex], answer: (answer || '').trim() };
      $('saved-answer').textContent = answers[savedIndex].answer || 'No speech was captured. You can try again.';
      log('Answer saved (' + answers[savedIndex].answer.length + ' characters)');
      updateNext();
      if (answers[savedIndex].answer) {
        syncControls();
        evaluateAnswer(savedIndex);  // runs in the background; Next stays locked until it finishes or fails
      } else {
        renderEvaluation(null);
      }
    } catch (err) {
      if (!interviewFinished) { status(err.message || String(err)); log('Speech recognition issue: ' + err.message); }
    } finally {
      syncControls();
    }
  });
  $('done-answer').addEventListener('click', () => { VoiceModule.stopListening(); });
  $('next-question').addEventListener('click', () => {
    if (questionIndex < questions.length - 1 && answers[questionIndex]?.answer) {
      questionIndex++; showQuestion();
    }
  });

  $('finish-interview').addEventListener('click', async () => {
    if (interviewFinished) return;
    if (evaluating) { status('Please wait for the current answer to finish evaluating.'); return; }
    interviewFinished = true;
    $('finish-interview').disabled = true;
    $('finish-interview').textContent = 'Building your report…';
    VoiceModule.cancelAll({ discardAnswer: true });
    // Capture the summary before stopping the monitor, even if the candidate stopped monitoring manually.
    const cameraSummary = monitoringEverStarted ? CameraMonitor.getSummary() : null;
    CameraMonitor.stop();
    CameraModule.stopAll(video);
    const completeAnswers = questions.map((question, index) => ({
      question_id: questionMeta[index].id, question,
      answer: (answers[index] && answers[index].answer) ? answers[index].answer : '',
    }));
    const payload = { role: cfg.role, answers: completeAnswers, cameraSummary, events, finishedAt: new Date().toISOString() };
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 180000);
      const response = await fetch('/api/report/generate', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
        body: JSON.stringify({ answers: completeAnswers.filter((a) => a.question_id).map((a) => ({ question_id: a.question_id, answer: a.answer })) })
      });
      clearTimeout(timer);
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.ok) throw new Error(data.error || 'Report generation failed.');
      payload.report = data.report;
    } catch (err) {
      payload.reportError = err.name === 'AbortError' ? 'Report generation took too long.' : (err.message || 'The AI report is temporarily unavailable.');
      log('Report unavailable: ' + payload.reportError);
    }
    sessionStorage.setItem('mockInterviewReport', JSON.stringify(payload));
    window.location.href = '/report';
  });
  window.addEventListener('beforeunload', () => {
    VoiceModule.cancelAll({ discardAnswer: true });
    CameraMonitor.stop();
    CameraModule.stopAll(video);
  });
  showQuestion();
  if (!VoiceModule.isSupported().speech) log('Speech output is not supported in this browser.');
  if (!VoiceModule.isSupported().recognition) log('Speech recognition is not supported in this browser. Use Chrome or Edge.');
})();
