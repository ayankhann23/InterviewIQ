// Setup page: prepares the interview questions (AI-generated when possible) before opening the interview.
(() => {
  const link = document.getElementById('continue-interview');
  const select = document.getElementById('difficulty-select');
  const note = document.getElementById('question-source-note');
  if (!link || !select) return;
  const label = link.querySelector('span');
  let busy = false;

  link.addEventListener('click', async (event) => {
    event.preventDefault();
    if (busy) return;
    busy = true;
    link.setAttribute('aria-disabled', 'true');
    link.style.pointerEvents = 'none';
    link.style.opacity = '0.7';
    const role = link.dataset.role;
    const difficulty = select.value;
    if (label) label.textContent = 'Preparing your questions…';
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 120000);
    try {
      const response = await fetch('/api/questions/generate', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role, difficulty }), signal: controller.signal,
      });
      const data = await response.json().catch(() => ({}));
      if (note && data.note) note.textContent = data.note;
    } catch (err) {
      // Never block the interview: the server falls back to the built-in questions.
      if (note) note.textContent = 'Question preparation failed; using the built-in questions.';
    } finally {
      clearTimeout(timer);
    }
    const target = new URL(link.dataset.interviewUrl, window.location.origin);
    target.searchParams.set('role', role);
    target.searchParams.set('difficulty', difficulty);
    window.location.href = target.toString();
  });
})();
