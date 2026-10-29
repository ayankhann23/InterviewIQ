"""Smoke-test the NVIDIA Nemotron connection using the same service the app uses.

Run from the project folder:
    python scripts/check_nemotron.py          # one tiny JSON request (fast)
    python scripts/check_nemotron.py --full   # exercises all four AI features

Prints PASS/FAIL and never prints the API key or resume/answer content.
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# Always read the .env next to app.py, no matter where the script is run from.
load_dotenv(PROJECT_ROOT / ".env")

from modules import ai_features, evaluator  # noqa: E402
from modules.nemotron_service import AIServiceError, NemotronService  # noqa: E402

SAMPLE_RESUME = """
Avery Chen
Junior Python Developer
Skills: Python, Flask, SQL, Git, HTML, CSS, JavaScript.
Projects:
- Task Tracker: a Flask web app with routes to add, view and update tasks,
  form validation and SQLite storage.
- Library Record Manager: a small Django app to search book records by title
  and author.
Education: Diploma in Information Technology (2024-2027).
"""
SAMPLE_ANSWER = (
    "A list is mutable, so I can append, remove or change items after creation; it uses square brackets. "
    "A tuple is immutable, so its contents cannot change after creation; it uses parentheses. "
    "I use a list for collections that change and a tuple for fixed records such as coordinates."
)


def _step(label, fn):
    started = time.monotonic()
    try:
        result = fn()
    except (AIServiceError, ValueError) as exc:
        detail = getattr(exc, "user_message", str(exc))
        print(f"FAIL {label}: {detail}")
        return None
    print(f"PASS {label} ({time.monotonic() - started:.1f}s)")
    return result


def full_check(service):
    role = "Python Developer"
    analysis = _step("resume analysis", lambda: ai_features.analyze_resume(service, SAMPLE_RESUME, role))
    if analysis is None:
        return 1
    generated = _step("question generation",
                      lambda: ai_features.generate_questions(service, SAMPLE_RESUME, role, "intermediate", 3))
    if generated is None:
        return 1
    questions, _padded = generated
    question = questions[0]
    result = _step("answer evaluation",
                   lambda: ai_features.evaluate_answer(
                       service, role, "intermediate", question, SAMPLE_ANSWER,
                       previous=[], allow_follow_up=True,
                       existing_questions=[q["text"] for q in questions]))
    if result is None:
        return 1
    records = [{
        "id": question.get("id", "q1"), "question": question["text"], "kind": "main",
        "topic": question.get("topic"), "difficulty": question.get("difficulty"),
        "type": question.get("type"), "answer": SAMPLE_ANSWER, "evaluation": result["evaluation"],
    }]
    summary = evaluator.summarize_interview(role, records)
    report = _step("report summary",
                   lambda: ai_features.summarize_report(service, role, summary, records, analysis))
    if report is None:
        return 1
    print(f"PASS full integration: {len(questions)} questions, evaluation and report summary "
          f"returned valid structured output")
    return 0


def main():
    service = NemotronService()
    print(f"Model    : {service.model}")
    print(f"Endpoint : {service.base_url}")
    print(f"Thinking : {service.enable_thinking}")
    print(f"Retries  : {service.max_retries}")
    print(f"API key  : {'set (hidden)' if service.api_key else 'MISSING - add NVIDIA_API_KEY to .env'}")
    if not service.api_key:
        return 2
    try:
        model, seconds, _ = service.ping()
    except AIServiceError as exc:
        print(f"FAIL [{exc.kind}]: {exc.user_message}")
        return 1
    print(f"PASS ping: {model} answered a JSON request in {seconds:.1f}s")
    if "--full" in sys.argv[1:]:
        return full_check(service)
    return 0


if __name__ == "__main__":
    sys.exit(main())
