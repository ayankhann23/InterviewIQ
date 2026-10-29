// jsdom harness: loads the REAL rendered interview page + the REAL static/js/interview.js, stubs the
// camera/voice modules (no hardware in a test), and talks to a live Flask server with a fake AI.
// Usage: node interview_harness.js <baseUrl> <projectRoot>
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');
const [base, root] = process.argv.slice(2);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const jar = {};
async function http(url, opts = {}) {
  const headers = Object.assign({}, opts.headers || {}, { Cookie: Object.entries(jar).map(([k, v]) => k + '=' + v).join('; ') });
  const res = await fetch(new URL(url, base), Object.assign({}, opts, { headers }));
  (res.headers.getSetCookie ? res.headers.getSetCookie() : []).forEach((c) => { const [kv] = c.split(';'); const i = kv.indexOf('='); jar[kv.slice(0, i)] = kv.slice(i + 1); });
  return res;
}
(async () => {
  const out = { checks: {} };
  // 1) upload a resume, prepare AI questions, fetch the rendered interview page
  const form = new FormData();
  form.append('resume', new Blob([fs.readFileSync(path.join(root, 'tests/samples/sample_resume_python_developer.pdf'))]), 'r.pdf');
  await http('/', { method: 'POST', body: form });
  await http('/api/questions/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ role: 'Python Developer', difficulty: 'intermediate' }) });
  const html = await (await http('/interview?role=Python%20Developer&difficulty=intermediate')).text();

  const dom = new JSDOM(html.replace(/<script[\s\S]*?<\/script>/g, ''), { url: base + '/interview', runScripts: 'outside-only', pretendToBeVisual: true });
  const w = dom.window;
  const inline = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]).join('\n');
  // jsdom's AbortSignal is not accepted by Node's fetch, so bridge it to a Node AbortController.
  w.fetch = (u, o) => {
    o = o || {};
    const opts = { method: o.method, headers: o.headers, body: o.body };
    if (o.signal) {
      const ac = new AbortController();
      if (o.signal.aborted) ac.abort();
      else o.signal.addEventListener('abort', () => ac.abort(), { once: true });
      opts.signal = ac.signal;
    }
    return http(u, opts);
  };
  let pendingAnswer = null; let listening = false;
  w.VoiceModule = {
    isSupported: () => ({ speech: true, recognition: true }), isListening: () => listening, cancelAll() { listening = false; },
    speak: async () => {}, stopListening() {},
    startListening: () => { listening = true; return new Promise((res) => { pendingAnswer = (t) => { listening = false; res(t); }; }); },
  };
  const handlers = {};
  w.CameraModule = { isActive: () => true, on() {}, requestPermissions: async () => ({}), stopAll() {} };
  w.CameraMonitor = { on() {}, start: async () => {}, stop() {}, pause() {}, resume() {}, isRunning: () => false, getSummary: () => ({}) };
  w.eval(inline);
  w.eval(fs.readFileSync(path.join(root, 'static/js/interview.js'), 'utf8'));
  const $ = (id) => w.document.getElementById(id);
  const click = (id) => $(id).dispatchEvent(new w.Event('click', { bubbles: true }));

  click('start-devices'); await sleep(50);
  out.initialQuestionCount = w.INTERVIEW_CONFIG.questions.length;
  out.checks.firstScreen = $('question-count').textContent;
  const answerText = 'I use try and except blocks to catch specific exceptions, log the error, and use finally for cleanup.';
  const counts = [];
  let doubleSubmitBlocked = true, nextLockedWhileEvaluating = true;

  async function answerCurrent() {
    click('start-answer'); await sleep(20);
    pendingAnswer(answerText);
    await sleep(40);                                   // evaluation is now running (fake AI is slow)
    if (!$('start-answer').disabled) doubleSubmitBlocked = false;
    if (!$('next-question').disabled) nextLockedWhileEvaluating = false;
    click('start-answer');                              // a second click during evaluation must do nothing
    for (let i = 0; i < 100 && $('next-question').disabled; i++) await sleep(50);
    counts.push($('question-count').textContent);
  }
  for (let i = 0; i < 4; i++) { await answerCurrent(); if (!$('next-question').disabled) { click('next-question'); await sleep(20); } }
  out.progress = counts;
  out.feedbackShown = $('ai-feedback').textContent.slice(0, 80);
  out.doubleSubmitBlocked = doubleSubmitBlocked;
  out.nextLockedWhileEvaluating = nextLockedWhileEvaluating;
  out.finalCount = $('question-count').textContent;

  click('finish-interview');
  for (let i = 0; i < 100 && !w.sessionStorage.getItem('mockInterviewReport'); i++) await sleep(50);
  const payload = JSON.parse(w.sessionStorage.getItem('mockInterviewReport') || '{}');
  out.reportOverall = payload.report && payload.report.summary.overall_score;
  out.reportRecordIds = payload.report && payload.report.records.map((r) => r.id);
  out.reportError = payload.reportError || null;
  console.log(JSON.stringify(out));
  process.exit(0);
})().catch((e) => { console.error('HARNESS ERROR', e); process.exit(1); });
