# InterviewIQ — Integrated Voice and Camera Prototype

## Included
- Existing PDF resume parsing and role/skill matching (`app.py`, `templates/index.html`).
- Setup page and live interview page.
- Browser speech output / speech recognition (`static/js/voice.js`).
- Camera and microphone permissions (`static/js/camera.js`).
- Camera monitoring (`static/js/monitor.js`).
- Interview flow and session report (`static/js/interview.js`, `templates/report.html`).

## Run on Windows / VS Code
1. Extract this ZIP into a new folder.
2. Open the extracted `mock_interview_simulator` folder in VS Code.
3. Open Terminal in VS Code and run:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python app.py
   ```

4. Open `http://127.0.0.1:5000` in Chrome.
5. Upload a text-based PDF resume. When the results appear, click **Start Voice & Camera Interview**.
6. On setup, click **Continue to interview**. On the interview page, click **Allow camera & microphone** and grant browser permission.
7. Test **Speak question**, **Start answering**, **Done answering**, **Start monitoring**, and **Finish interview & view report**.

## Important notes
- Keep the terminal running while using the app. Press Ctrl+C in the terminal to stop Flask.
- `localhost` camera/microphone access requires browser permission and a working device. If the laptop camera is faulty, test on another computer with a webcam.
- Browser speech recognition works best in Chrome/Edge and may need internet access.
- Face detection may need internet access to load its model. If unavailable, the app should still show the available basic camera-monitoring status.
- The Chrome DevTools message about Content Security Policy and `eval` may come from DevTools or an extension. This app does not use `eval`; if it persists, test in an Incognito window with extensions disabled.
- The report uses `sessionStorage` in the same browser tab; it is not a permanent database report.
- Camera/posture observations are not proof of misconduct. Test every flow before presenting or handing the project to teammates.

## Folder map
- `app.py`: Flask routes and PDF resume parsing
- `templates/`: home, setup, interview and report pages
- `static/css/style.css`: main homepage styles
- `static/js/`: voice, camera, monitoring and interview flow scripts
- `modules/`: reserved for shared Python modules to be added during integration
- `uploads/`, `reports/`: placeholders for future server-side persistence
- `tests/`: reserved for automated tests


## Bug-fix pass (2026-10-15)

- Monitoring starts after camera and microphone permissions are granted; interview actions require active devices and monitoring.
- Device disconnects pause the interview and monitoring; reconnecting devices resumes/restarts monitoring.
- Microphone level updates use a dedicated element so they no longer overwrite pause/status messages.
- Device disconnect messages identify the camera or microphone correctly.
- Interrupted answers are discarded instead of being saved as completed answers.
- Question playback is disabled while recording an answer.
- Camera summary is captured before monitor shutdown and is formatted as readable report text.
- Resume skill and role keyword matching uses whole-term matching (avoids `Java` matching `JavaScript` and `Excel` matching `excellent`).
- PyMuPDF minimum version raised to 1.24.3.
- Sample resume PDFs moved to `tests/samples/`.
- Setup/privacy notes explain speech-recognition audio processing and the third-party face-model CDN.

Validation performed: `node --check` passes for all five JavaScript files and Python syntax compilation passes. Flask route/PDF runtime tests could not be run in this environment because Flask is not installed; real browser camera, microphone, and speech recognition still need manual testing in Chrome or Edge.


## Follow-up audit fixes (2026-10-15)

- Corrected regex word-boundary escaping for role and skill detection.
- Kept camera-monitoring warning history and posture/sample counters across monitoring restarts.
- Monitoring duration now counts active monitoring time and excludes paused/disconnected periods.
- Speech-recognition errors now discard partial answers instead of returning them as complete answers.
- Invalid resume selections clear stale file details and keep the submit button disabled.
- Report camera summary now uses actual line breaks and human-readable posture percentages.
- Resume processing errors show a safe user-facing message while technical details remain in the server log.

Validation: all five JavaScript files pass `node --check`; `app.py` passes Python compilation; keyword matching was checked against Java/JavaScript and Excel/excellent false-positive examples. Real camera/microphone/browser behavior and Flask/PDF runtime tests still need to be run locally.


## Initial answer scoring (2026-10-15)

- Added `modules/evaluator.py` with a transparent heuristic score for relevance, detected technical concepts, completeness, and clarity.
- Added `POST /api/evaluate`; the interview finish flow sends recorded answers to this endpoint.
- Report now shows an overall score, per-question breakdown, detected concepts and coaching suggestions.
- Scores are heuristic and do not verify factual correctness; they are for practice feedback only and must not be used as the sole basis for hiring decisions.
- Run unit tests from the project root with `python -m unittest discover -s tests`.

---

# AI integration (NVIDIA Nemotron)

## Setup
1. Create a file named `.env` next to `app.py` (copy `.env.example`) and set your key.
   The file must use plain `KEY=value` lines - **not** Python code:
   ```
   NVIDIA_API_KEY=nvapi-your-real-key
   NVIDIA_MODEL=nvidia/nemotron-3.5-super-vl-120b-a12b
   NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
   ```
   The model and endpoint default to the supplied configuration and can be overridden with
   `NVIDIA_MODEL` / `NVIDIA_BASE_URL`. The key is read on the server only; it is never sent to
   the browser, HTML, JavaScript or reports. `.env` is git-ignored.
2. Install and run from the project folder:
   ```powershell
   python -m pip install -r requirements.txt
   python app.py
   ```
   `app.py` loads the `.env` next to itself even if you start it from another directory.
3. Check the connection (prints PASS/FAIL, never the key): `python scripts/check_nemotron.py`
   For a full live test of resume analysis, question generation, evaluation and report summary:
   `python scripts/check_nemotron.py --full`
4. Run the automated tests: `python -m unittest discover -s tests -t .`
   (the browser-level test needs `npm i jsdom`; it is skipped otherwise).
   On Windows, set `JSDOM_NODE_PATH` to the installed `node_modules` folder if jsdom is not
   found automatically.

## The integrated model
* Model: `nvidia/nemotron-3.5-super-vl-120b-a12b`
* Endpoint: `https://integrate.api.nvidia.com/v1` (OpenAI-compatible)
* The model is a vision-language model, but this project sends **text only** (resume text,
  answers and questions). Camera video never leaves the browser.
* `NVIDIA_ENABLE_THINKING=false` is the default: it disables the slower reasoning pass so
  normal JSON answers come back quickly. Set it to `true` only if you want to experiment.
* `NVIDIA_MAX_RETRIES=2` lets the OpenAI SDK retry transient NVIDIA 429/5xx/network errors
  with backoff. Authentication and "model not found" errors are not retried.
* The app adds one more retry after the SDK's retries for a transient 5xx/timeout/empty reply
  (`NVIDIA_RETRY_BACKOFF` controls the delay). A permanently invalid key or model is never retried.

## How it works
| Step | Where | What |
|---|---|---|
| Resume upload | `app.py` `home()` | Same validation + keyword analysis as before; text is also kept **server-side** for the AI. |
| Resume analysis | `/api/resume/analyze`, `static/js/resume_ai.js` | Nemotron returns structured findings shown in a new card. Email/phone/links are removed before sending. |
| Questions | `/api/questions/generate`, `static/js/setup.js` | Setup page picks difficulty; Nemotron writes `INTERVIEW_QUESTION_COUNT` questions. Falls back to the built-in bank on any failure. |
| Evaluation | `/api/answer/evaluate`, `static/js/interview.js` | One Nemotron call per answer → rubric scores, explanation, missing concepts, mistakes, sample answer, tip. |
| Follow-ups | same call | At most `INTERVIEW_MAX_FOLLOW_UPS` per interview, one per main question, duplicates dropped, enforced on server **and** browser. |
| Report | `/api/report/generate`, `templates/report.html` | Overall score is computed from stored evaluations; Nemotron only writes the qualitative summary. |

Files: `modules/nemotron_service.py` (the only code that calls NVIDIA), `modules/ai_features.py` (prompts + validation),
`modules/evaluator.py` (rubric + scoring policy), `modules/question_bank.py`, `modules/interview_store.py`, `modules/sanitize.py`.

## Scoring (explain this in the viva)
* Rubric weights live in `modules/evaluator.py`. Technical: correctness 40, relevance 20, completeness 20, reasoning 10, clarity 10.
  Project and behavioural questions use adjusted rubrics.
* The model scores each criterion 0–100; **the server** computes the weighted final score (the model's own total is ignored).
* Overall score = average of the **main** questions. Unanswered = 0. Answered but AI evaluation failed = excluded and shown as
  "evaluation unavailable" (no fake score). Follow-ups are shown separately and never change the overall score.

## Security & privacy
* The key is read from `.env` on the server only; it is never in HTML/JS/logs/reports. `.env` is git-ignored.
* Resume text and answers are treated as untrusted: wrapped in delimiter tags, delimiter look-alikes stripped, and the system prompt
  forbids following instructions inside them. All AI text is rendered with `textContent` (no HTML injection).
* Not removed: your **name** and street address (cannot be redacted reliably by rules) are still sent with the resume text.
* Resume text and evaluations are held in server memory only (4 h expiry) – never written to disk or logs. A server restart clears them;
  the app then uses the built-in questions and the report says the session expired.

## Known limits
* Camera, microphone and speech recognition were **not** changed. They need a real browser and device to verify.
* The old heuristic `/api/evaluate` route was removed on purpose: it produced scores that were not AI scores.
* The interview page shows AI feedback after every answer and keeps "Next" locked until that evaluation finishes or fails.

## Troubleshooting the NVIDIA connection
* `AI features are not configured`: `.env` is missing, is in the wrong folder, or still uses Python
  syntax. The file must contain `NVIDIA_API_KEY=nvapi-...` on its own line. Restart `python app.py`
  after editing it.
* `The NVIDIA API key was rejected` (401): the key is wrong, expired, or copied with extra text.
  Create a new key on NVIDIA's site and put only the `nvapi-...` value in `.env`.
* `The configured NVIDIA model or endpoint was not found` (404): check `NVIDIA_MODEL` and
  `NVIDIA_BASE_URL`; the model must be available to your NVIDIA account.
* `The AI service returned an error (500/502/503)`: NVIDIA had a transient failure. The service
  already retries; click Retry in the UI or run `python scripts/check_nemotron.py` again.
* `The AI service took too long`: check your internet connection or increase
  `NVIDIA_TIMEOUT_SECONDS` in `.env`.
* If you shared a `.env` or ZIP containing a real key (for example in a chat or repository),
  rotate/revoke that key on NVIDIA's site and put the new key in `.env`.

## AI integration validation (2026-10-20)
* Fixed the supplied `.env`: it contained Python code (`api_key = os.environ[...]`) instead of
  dotenv syntax, so no key or model was being loaded. It now uses `NVIDIA_API_KEY=...`,
  `NVIDIA_MODEL=nvidia/nemotron-3.5-super-vl-120b-a12b` and the OpenAI-compatible NVIDIA endpoint.
* `app.py` and `scripts/check_nemotron.py` now load the `.env` next to `app.py` explicitly, so
  starting the app from another directory still works.
* Verified with the real key: `python scripts/check_nemotron.py --full` passed resume analysis,
  question generation, answer evaluation and report summary against the requested model.
* `python -m unittest discover -s tests -t .` passes 68 tests, including the jsdom browser flow.
* `app.run()` now honours `FLASK_DEBUG` instead of always enabling the debugger.
* Transient NVIDIA 5xx/timeout/empty responses are retried (SDK retries + one app-level retry);
  authentication and invalid-model errors are never retried.
* Not tested automatically: a real camera, microphone and Chrome speech recognition. Use a real
  browser and device for those flows.
