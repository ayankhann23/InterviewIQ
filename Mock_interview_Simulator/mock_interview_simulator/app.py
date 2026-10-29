import os
import re
import secrets
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load NVIDIA_API_KEY etc. from the .env next to this file BEFORE the AI service
# reads the environment. Using an explicit path means the app also works when it
# is started from another working directory (e.g. an IDE run configuration).
load_dotenv(Path(__file__).with_name(".env"))

import pymupdf
from flask import Flask, render_template, request, jsonify, session

from modules import ai_features, evaluator, interview_store
from modules.nemotron_service import AIServiceError, NemotronService
from modules.question_bank import build_fallback_questions

app = Flask(__name__)

# Maximum upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
# The signed session cookie only carries a random id; resume text stays on the server.
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

MAX_PDF_PAGES = 15
MAX_STORED_RESUME_CHARS = 60000


def _env_int(name, default, low, high):
    try:
        value = int(os.environ.get(name, "").strip() or default)
    except ValueError:
        value = default
    return max(low, min(high, value))


def _env_bool(name, default=False):
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


# Configurable limits (see .env.example)
QUESTION_COUNT = _env_int("INTERVIEW_QUESTION_COUNT", 7, 3, 15)
MAX_FOLLOW_UPS = _env_int("INTERVIEW_MAX_FOLLOW_UPS", 2, 0, 5)
DEFAULT_DIFFICULTY = "intermediate"


def get_ai():
    """One shared AI service. Tests replace it via app.config['AI_SERVICE']."""
    service = app.config.get("AI_SERVICE")
    if service is None:
        service = NemotronService()
        app.config["AI_SERVICE"] = service
    return service


def analyse_resume(pdf_bytes):
    # Open the uploaded PDF from memory
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    try:
        # Extract text from the first pages only (limits work on huge files)
        resume_text = "\n".join(
            document.load_page(i).get_text()
            for i in range(min(document.page_count, MAX_PDF_PAGES))
        )
    finally:
        document.close()

    text_lower = resume_text.lower()

    # Suggest a profession using simple keyword matching
    roles = {
        "AI/ML Engineer": [
            "machine learning", "deep learning", "tensorflow"
        ],
        "Python Developer": [
            "python", "flask", "django"
        ],
        "Web Developer": [
            "html", "css", "javascript"
        ],
        "Data Analyst": [
            "power bi", "excel", "data analysis"
        ],
    }

    suggested_role = "General / Entry-Level Role"

    def contains_term(term):
        return re.search(rf"\b{re.escape(term.lower())}\b", text_lower) is not None

    for role, keywords in roles.items():
        if any(contains_term(keyword) for keyword in keywords):
            suggested_role = role
            break

    # Identify skills mentioned in the resume
    possible_skills = [
        "Python", "Java", "HTML", "CSS", "JavaScript",
        "SQL", "Machine Learning", "Deep Learning",
        "TensorFlow", "Flask", "Power BI", "Excel"
    ]

    found_skills = [
        skill for skill in possible_skills
        if re.search(rf"(?<!\w){re.escape(skill.lower())}(?!\w)", text_lower)
    ]

    return resume_text, suggested_role, found_skills


# ----------------------------------------------------------------------------- helpers
def _state():
    """The server-side state for this browser (or None)."""
    return interview_store.get(session.get("sid"))


def _ai_error(exc):
    transient = ("not_configured", "auth", "network", "timeout", "rate_limit", "server", "empty")
    status = 503 if exc.kind in transient else 502
    return jsonify({"ok": False, "error": exc.user_message, "kind": exc.kind, "retryable": exc.retryable}), status


def _public_question(q):
    """What the browser may see: no expected-concept hints."""
    return {k: q.get(k) for k in ("id", "text", "topic", "difficulty", "type", "kind", "parent_id")}


def _difficulty(value):
    value = str(value or "").lower()
    return value if value in ai_features.DIFFICULTIES else DEFAULT_DIFFICULTY


@app.route("/", methods=["GET", "POST"])
def home():
    # Always initialise the template variables
    resume_text = ""
    suggested_role = ""
    skills = []
    error = ""
    resume_ready = False

    if request.method == "POST":
        file = request.files.get("resume")

        if not file or not file.filename:
            error = "Please select a PDF resume."

        elif not file.filename.lower().endswith(".pdf"):
            error = "Please upload a PDF file."

        else:
            try:
                pdf_bytes = file.read()

                if not pdf_bytes:
                    raise ValueError("The uploaded file is empty.")

                if pdf_bytes.lstrip()[:5] != b"%PDF-":
                    error = "That file does not look like a valid PDF. Please upload a text-based PDF resume."
                else:
                    resume_text, suggested_role, skills = analyse_resume(
                        pdf_bytes
                    )

                    if not resume_text.strip():
                        error = (
                            "No readable text was found. "
                            "Try a text-based PDF resume."
                        )
                        suggested_role = ""
                        skills = []
                        resume_text = ""
                    else:
                        # Fresh server-side state for this resume (clears any earlier interview).
                        sid = interview_store.new_session()
                        session["sid"] = sid
                        state = interview_store.get(sid)
                        state["resume_text"] = resume_text[:MAX_STORED_RESUME_CHARS]
                        state["suggested_role"] = suggested_role
                        resume_ready = True

            except Exception:
                # Log technical details on the server; show a safe message to users.
                app.logger.exception("Resume processing failed")
                error = "Could not process this PDF. Please check that it is a valid, text-based PDF and try again."

    # Render the HTML page on both GET and POST requests
    return render_template(
        "index.html",
        resume_text=resume_text,
        suggested_role=suggested_role,
        skills=skills,
        error=error,
        ai_configured=get_ai().configured,
        resume_ready=resume_ready,
    )


@app.route("/api/resume/analyze", methods=["POST"])
def api_resume_analyze():
    """AI analysis of the resume uploaded in this browser session."""
    state = _state()
    if not state or not state.get("resume_text"):
        return jsonify({"ok": False, "error": "Upload a resume first."}), 409
    payload = request.get_json(silent=True) or {}
    role = state.get("suggested_role") or ""
    if role.startswith("General"):
        role = ""  # no specific target role was detected
    if state.get("resume_analysis") and not payload.get("force"):
        return jsonify({"ok": True, "analysis": state["resume_analysis"], "target_role": role, "cached": True})
    try:
        analysis = ai_features.analyze_resume(get_ai(), state["resume_text"], role)
    except AIServiceError as exc:
        return _ai_error(exc)
    except ValueError:
        return jsonify({"ok": False, "error": "The resume had no usable text to analyse."}), 400
    with interview_store.lock():
        state["resume_analysis"] = analysis
        state["resume_analysis_role"] = role
    return jsonify({"ok": True, "analysis": analysis, "target_role": role})


@app.route("/setup")
def setup():
    role = request.args.get("role", "General / Entry-Level Role")[:100]
    state = _state()
    return render_template(
        "setup.html", role=role,
        ai_configured=get_ai().configured,
        resume_ready=bool(state and state.get("resume_text")),
        question_count=QUESTION_COUNT,
    )


@app.route("/api/questions/generate", methods=["POST"])
def api_generate_questions():
    """Prepare the question list for the interview (AI when possible, otherwise the built-in bank)."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "error": "Send a JSON object with role and difficulty."}), 400
    role = str(payload.get("role") or "General / Entry-Level Role")[:100]
    difficulty = _difficulty(payload.get("difficulty"))
    state = _state()
    if state is None:
        return jsonify({"ok": True, "source": "fallback", "note": "No resume found; using the built-in questions."})

    note = ""
    source = "fallback"
    questions = None
    padded = 0
    if not state.get("resume_text"):
        note = "No resume found; using the built-in questions."
    elif not get_ai().configured:
        note = "AI is not configured; using the built-in questions."
    else:
        try:
            raw, padded = ai_features.generate_questions(get_ai(), state["resume_text"], role, difficulty, QUESTION_COUNT)
            questions = [
                {"id": f"q{i}", "text": q["text"], "topic": q["topic"], "difficulty": q["difficulty"],
                 "type": q["type"], "kind": "main",
                 "criteria": q["criteria"] or evaluator.reference_concepts(q["text"], role)}
                for i, q in enumerate(raw, start=1)
            ]
            source = "ai"
        except AIServiceError as exc:
            note = f"{exc.user_message} Using the built-in questions instead."
        except ValueError:
            note = "The AI could not produce a valid question list. Using the built-in questions instead."
    if questions is None:
        questions = build_fallback_questions(role, difficulty)
    with interview_store.lock():
        state["generated"] = {"role": role, "difficulty": difficulty, "questions": questions,
                              "source": source, "padded": padded, "note": note}
    return jsonify({"ok": True, "source": source, "count": len(questions), "padded": padded, "note": note})


@app.route("/interview")
def interview():
    role = request.args.get("role", "General / Entry-Level Role")[:100]
    difficulty = _difficulty(request.args.get("difficulty"))
    sid, state = interview_store.ensure(session.get("sid"))
    session["sid"] = sid
    generated = state.get("generated")
    if generated and generated["role"] == role and generated["difficulty"] == difficulty:
        questions, source = generated["questions"], generated["source"]
    else:
        questions, source = build_fallback_questions(role, difficulty), "fallback"
    interview_store.start_interview(state, role, difficulty, questions, source)
    return render_template(
        "interview.html", role=role,
        questions=[_public_question(q) for q in questions],
        question_source=source, difficulty=difficulty,
        max_follow_ups=MAX_FOLLOW_UPS, ai_configured=get_ai().configured,
    )


@app.route("/report")
def report():
    return render_template("report.html")


# ----------------------------------------------------------------------------- answer evaluation
def _context_lines(interview_state):
    lines = []
    for q in interview_state["questions"]:
        ev = interview_state["evaluations"].get(q["id"])
        if ev and ev.get("status") == "evaluated":
            lines.append(f"{q.get('topic')}: scored {ev['score']}/100 ({q.get('kind')} question)")
    return lines


@app.route("/api/answer/evaluate", methods=["POST"])
def api_evaluate_answer():
    """Evaluate ONE answer with Nemotron; may also return one bounded follow-up question."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("question_id"), str) \
            or not isinstance(payload.get("answer", ""), str):
        return jsonify({"ok": False, "error": "Send JSON with question_id and answer."}), 400
    qid = payload["question_id"][:20]
    answer = payload.get("answer", "").strip()[:ai_features.ANSWER_MAX_CHARS]

    state = _state()
    inter = state.get("interview") if state else None
    if not inter:
        return jsonify({"ok": False, "error": "The interview session expired (the server restarted or timed out). "
                        "Your answers are still saved in the browser.", "kind": "session"}), 409

    with interview_store.lock():
        question = next((q for q in inter["questions"] if q["id"] == qid), None)
        if question is None:
            return jsonify({"ok": False, "error": "Unknown question."}), 404
        question = dict(question)
        changed = inter["answers"].get(qid) != answer
        inter["answers"][qid] = answer
        if changed:
            inter["evaluations"].pop(qid, None)  # a changed answer invalidates the old evaluation
        cached = inter["evaluations"].get(qid)
        if cached and cached.get("status") in ("evaluated", "no_answer"):
            return jsonify({"ok": True, "evaluation": cached, "follow_up": None, "cached": True})
        if not answer:
            ev = evaluator.no_answer_evaluation(question.get("type"))
            inter["evaluations"][qid] = ev
            return jsonify({"ok": True, "evaluation": ev, "follow_up": None})
        allow_follow_up = (
            question["kind"] == "main"
            and qid not in inter["followed_parents"]
            and inter["follow_ups_used"] < MAX_FOLLOW_UPS
            and len(answer.split()) >= 8
        )
        existing = [q["text"] for q in inter["questions"]]
        previous = _context_lines(inter)
        role, difficulty = inter["role"], inter["difficulty"]

    try:
        result = ai_features.evaluate_answer(
            get_ai(), role, difficulty, question, answer, previous=previous,
            allow_follow_up=allow_follow_up, existing_questions=existing)
    except AIServiceError as exc:
        return _ai_error(exc)

    follow_up_public = None
    with interview_store.lock():
        if state.get("interview") is inter:  # the interview was not restarted while the AI was thinking
            inter["evaluations"][qid] = result["evaluation"]
            if allow_follow_up and qid not in inter["followed_parents"]:
                inter["followed_parents"].add(qid)  # decision made: a parent never gets a second chance
                fu = result["follow_up"]
                if fu and inter["follow_ups_used"] < MAX_FOLLOW_UPS:
                    inter["follow_ups_used"] += 1
                    follow = {"id": f"f{inter['follow_ups_used']}", "text": fu["text"],
                              "topic": fu["topic"] or question.get("topic"), "difficulty": question.get("difficulty"),
                              "type": "behavioral" if question.get("type") == "behavioral" else "technical",
                              "kind": "follow_up", "parent_id": qid, "criteria": []}
                    position = next(i for i, q in enumerate(inter["questions"]) if q["id"] == qid)
                    inter["questions"].insert(position + 1, follow)
                    follow_up_public = _public_question(follow)
    return jsonify({"ok": True, "evaluation": result["evaluation"], "follow_up": follow_up_public})


# ----------------------------------------------------------------------------- final report
def _evaluate_missing(inter, role, difficulty, targets):
    """Retry evaluation (no follow-ups) for answered questions that still lack a stored evaluation."""
    service = get_ai()
    if not service.configured or not targets:
        return

    def work(q):
        try:
            result = ai_features.evaluate_answer(service, role, difficulty, q, inter["answers"][q["id"]],
                                                 previous=[], allow_follow_up=False, existing_questions=[])
            return q["id"], result["evaluation"], None
        except AIServiceError as exc:
            return q["id"], None, exc.user_message

    with ThreadPoolExecutor(max_workers=3) as pool:
        for qid, evaluation, problem in pool.map(work, targets):
            with interview_store.lock():
                if evaluation:
                    inter["evaluations"][qid] = evaluation
                else:
                    inter.setdefault("eval_errors", {})[qid] = problem


@app.route("/api/report/generate", methods=["POST"])
def api_generate_report():
    """Build the final report from the stored per-question evaluations (+ an AI qualitative summary)."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("answers"), list):
        return jsonify({"ok": False, "error": "Send JSON with an answers list."}), 400
    state = _state()
    inter = state.get("interview") if state else None
    if not inter:
        return jsonify({"ok": False, "error": "The interview session expired, so scores could not be assembled.",
                        "kind": "session"}), 409

    with interview_store.lock():
        known = {q["id"] for q in inter["questions"]}
        for item in payload["answers"][:40]:
            if isinstance(item, dict) and item.get("question_id") in known:
                text = str(item.get("answer") or "").strip()[:ai_features.ANSWER_MAX_CHARS]
                if inter["answers"].get(item["question_id"]) != text:
                    inter["evaluations"].pop(item["question_id"], None)
                inter["answers"][item["question_id"]] = text
        role, difficulty = inter["role"], inter["difficulty"]
        pending = [dict(q) for q in inter["questions"]
                   if inter["answers"].get(q["id"], "").strip()
                   and (inter["evaluations"].get(q["id"]) or {}).get("status") != "evaluated"]

    _evaluate_missing(inter, role, difficulty, pending)

    records = []
    with interview_store.lock():
        for q in inter["questions"]:
            answer = inter["answers"].get(q["id"], "")
            ev = inter["evaluations"].get(q["id"])
            if not answer.strip():
                ev = evaluator.no_answer_evaluation(q.get("type"))
            elif not ev:
                reason = (inter.get("eval_errors") or {}).get(q["id"]) or "AI evaluation was not completed."
                ev = evaluator.unavailable_evaluation(reason, q["text"], role)
            records.append({
                "id": q["id"], "question": q["text"], "kind": q["kind"], "topic": q.get("topic"),
                "difficulty": q.get("difficulty"), "type": q.get("type"), "answer": answer, "evaluation": ev,
            })
        resume_analysis = state.get("resume_analysis")
        interview_id = inter["id"]

    summary = evaluator.summarize_interview(role, records)
    ai_summary, ai_summary_error = None, None
    if any((r["evaluation"] or {}).get("status") == "evaluated" for r in records):
        try:
            ai_summary = ai_features.summarize_report(get_ai(), role, summary, records, resume_analysis)
        except AIServiceError as exc:
            app.logger.warning("AI report summary failed (%s); building structured coaching summary", exc.user_message)
            ai_summary = evaluator.build_fallback_coaching_summary(role, summary, records, resume_analysis)
    elif records:
        ai_summary = evaluator.build_fallback_coaching_summary(role, summary, records, resume_analysis)
    else:
        ai_summary_error = "No answer could be evaluated, so no AI summary was generated."

    return jsonify({"ok": True, "report": {
        "interview_id": interview_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "difficulty": difficulty,
        "summary": summary,
        "records": records,
        "ai_summary": ai_summary,
        "ai_summary_error": ai_summary_error,
        "resume_analysis": resume_analysis,
        "criterion_labels": evaluator.CRITERION_LABELS,
        "disclaimer": "AI-generated practice feedback. Scores are approximate coaching estimates, not an objective "
                      "measure of ability and not a hiring decision.",
    }})


@app.route("/api/report/retry-summary", methods=["POST"])
def api_retry_report_summary():
    """Allows the report page to regenerate the AI coaching summary if desired."""
    payload = request.get_json(silent=True) or {}
    state = _state()
    inter = state.get("interview") if state else None

    role = str(payload.get("role") or (inter.get("role") if inter else "General / Entry-Level Role"))[:100]
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        if inter and inter.get("questions"):
            records = []
            for q in inter["questions"]:
                answer = inter["answers"].get(q["id"], "")
                ev = inter["evaluations"].get(q["id"])
                records.append({
                    "id": q["id"], "question": q["text"], "kind": q["kind"], "topic": q.get("topic"),
                    "difficulty": q.get("difficulty"), "type": q.get("type"), "answer": answer, "evaluation": ev,
                })
        else:
            records = []

    summary = payload.get("summary") or evaluator.summarize_interview(role, records)
    resume_analysis = payload.get("resume_analysis") or (state.get("resume_analysis") if state else None)

    service = get_ai()
    if service.configured and any((r.get("evaluation") or {}).get("status") == "evaluated" for r in records):
        try:
            ai_summary = ai_features.summarize_report(service, role, summary, records, resume_analysis)
            return jsonify({"ok": True, "ai_summary": ai_summary, "source": "ai"})
        except AIServiceError as exc:
            app.logger.warning("Live AI summary retry failed (%s); providing structured summary", exc.user_message)

    ai_summary = evaluator.build_fallback_coaching_summary(role, summary, records, resume_analysis)
    return jsonify({"ok": True, "ai_summary": ai_summary, "source": "structured"})


@app.errorhandler(413)
def file_too_large(error):
    return "The PDF is too large. Please use a file under 10 MB.", 413


if __name__ == "__main__":
    # FLASK_DEBUG is documented in .env.example. Default to off so a copied
    # project does not expose the interactive debugger/console by accident.
    app.run(debug=_env_bool("FLASK_DEBUG", False))
