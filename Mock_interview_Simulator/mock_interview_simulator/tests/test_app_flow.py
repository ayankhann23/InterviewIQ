"""End-to-end backend tests: upload -> analysis -> questions -> evaluation -> follow-ups -> report.

The AI is a scripted fake (tests/fakes.py), so these tests prove the application logic, NOT the
live NVIDIA service. A real NVIDIA call is covered separately by scripts/check_nemotron.py.
"""
import copy
import io
import json
import os
import re
import unittest
from pathlib import Path

import httpx
import openai
import pymupdf

import app as app_module
from modules import interview_store
from modules.nemotron_service import NemotronService
from tests.fakes import (FakeClient, GOOD_EVAL, GOOD_QUESTIONS, GOOD_RESUME, GOOD_SUMMARY, eval_payload,
                         prompt_of, smart_responder)

SAMPLES = Path(__file__).parent / "samples"
SECRET = "nvapi-FLOW-TEST-SECRET"
FOLLOW_UPS = [
    "How would you verify that your error handling actually works under failure conditions?",
    "Which trade-offs did you weigh when choosing the evaluation metrics for that model?",
    "What would you change about the architecture if traffic grew by a factor of ten?",
    "Can you describe a bug that taught you something about debugging discipline?",
]
LONG_ANSWER = "I use try and except blocks to catch specific exceptions, log the error, and use finally for cleanup."


def make_responder(counter=None, fail_on=None):
    """Smart responder whose follow-up questions are unique per call, optionally failing a task."""
    counter = counter if counter is not None else {"n": 0}

    def responder(kwargs):
        p = prompt_of(kwargs)
        if fail_on and fail_on in p:
            request = httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions")
            raise openai.AuthenticationError(f"bad key {SECRET}", response=httpx.Response(401, request=request), body=None)
        if "criterion_scores" in p and "overall_comment" not in p:
            counter["n"] += 1
            ev = eval_payload(p)
            ev["follow_up"] = {"question": FOLLOW_UPS[(counter["n"] - 1) % len(FOLLOW_UPS)], "topic": "Testing"}
            return ev
        return smart_responder(kwargs)
    return responder


class FlowBase(unittest.TestCase):
    def setUp(self):
        interview_store.clear_all()
        self.fake = FakeClient(make_responder())
        app_module.app.config["AI_SERVICE"] = NemotronService(api_key=SECRET, client=self.fake)
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def tearDown(self):
        app_module.app.config.pop("AI_SERVICE", None)

    def upload(self, name="sample_resume_python_developer.pdf", data=None, filename=None):
        data = data if data is not None else (SAMPLES / name).read_bytes()
        return self.client.post("/", data={"resume": (io.BytesIO(data), filename or name)},
                                content_type="multipart/form-data")

    def post(self, url, payload=None):
        return self.client.post(url, json=payload if payload is not None else {})

    def start_interview(self, role="Python Developer"):
        self.upload()
        self.assertEqual(self.post("/api/questions/generate", {"role": role, "difficulty": "intermediate"}).status_code, 200)
        page = self.client.get(f"/interview?role={role}&difficulty=intermediate")
        self.assertEqual(page.status_code, 200)
        return page.get_data(as_text=True)


class ResumeTests(FlowBase):
    def test_valid_pdf_upload_keeps_existing_keyword_analysis(self):
        html = self.upload().get_data(as_text=True)
        self.assertIn("Python Developer", html)
        self.assertIn("Key Competencies", html)

    def test_invalid_extension_fake_pdf_and_empty_file_are_rejected(self):
        self.assertIn("Please upload a PDF file", self.upload(data=b"hello", filename="resume.txt").get_data(as_text=True))
        self.assertIn("does not look like a valid PDF", self.upload(data=b"not a pdf at all", filename="r.pdf").get_data(as_text=True))
        self.assertIn("Could not process", self.upload(data=b"", filename="r.pdf").get_data(as_text=True))

    def test_pdf_without_text_is_handled_gracefully(self):
        doc = pymupdf.open()
        doc.new_page()
        blank = doc.tobytes()
        self.assertIn("No readable text", self.upload(data=blank, filename="scan.pdf").get_data(as_text=True))
        self.assertEqual(self.post("/api/resume/analyze").status_code, 409)  # nothing stored for AI

    def test_ai_resume_analysis_and_caching(self):
        self.upload()
        first = self.post("/api/resume/analyze").get_json()
        self.assertTrue(first["ok"])
        self.assertEqual(first["analysis"]["role_match_score"], 72)
        calls = len(self.fake.calls)
        again = self.post("/api/resume/analyze").get_json()
        self.assertTrue(again["cached"])
        self.assertEqual(len(self.fake.calls), calls)  # no repeated model call

    def test_resume_analysis_without_upload_and_when_ai_down(self):
        self.assertEqual(self.post("/api/resume/analyze").status_code, 409)
        self.upload()
        app_module.app.config["AI_SERVICE"] = NemotronService(api_key=SECRET, client=FakeClient(make_responder(fail_on="<resume>")))
        resp = self.post("/api/resume/analyze")
        self.assertEqual(resp.status_code, 503)
        self.assertFalse(resp.get_json()["ok"])
        self.assertNotIn(SECRET, resp.get_data(as_text=True))
        # the existing keyword-based page still works while AI is down
        self.assertIn("Python Developer", self.upload().get_data(as_text=True))

    def test_not_configured_ai_does_not_break_the_page(self):
        app_module.app.config["AI_SERVICE"] = NemotronService(api_key="")
        html = self.upload().get_data(as_text=True)
        self.assertIn("Python Developer", html)
        self.assertEqual(self.post("/api/resume/analyze").get_json()["kind"], "not_configured")


class QuestionTests(FlowBase):
    def questions_in(self, html):
        match = re.search(r"questions: (\[.*?\]),\n", html, re.S)
        return json.loads(match.group(1))

    def test_ai_questions_reach_the_interview_without_hints_and_with_stable_count(self):
        html = self.start_interview()
        questions = self.questions_in(html)
        self.assertEqual(len(questions), 7)
        self.assertTrue(any("Traffic Sign CNN" in q["text"] for q in questions))
        self.assertEqual([q["id"] for q in questions], [f"q{i}" for i in range(1, 8)])
        self.assertNotIn("criteria", questions[0])  # expected concepts stay on the server
        self.assertEqual(len({q["text"] for q in questions}), 7)

    def test_fallback_to_builtin_questions_when_generation_fails(self):
        self.upload()
        app_module.app.config["AI_SERVICE"] = NemotronService(api_key=SECRET, client=FakeClient(make_responder(fail_on='"questions"')))
        result = self.post("/api/questions/generate", {"role": "Python Developer", "difficulty": "beginner"}).get_json()
        self.assertEqual(result["source"], "fallback")
        self.assertIn("built-in", result["note"])
        html = self.client.get("/interview?role=Python Developer&difficulty=beginner").get_data(as_text=True)
        self.assertEqual(len(self.questions_in(html)), 7)
        self.assertIn("What is the difference between a list and a tuple in Python?", html)

    def test_direct_visit_without_resume_uses_builtin_questions(self):
        html = self.client.get("/interview?role=Data Analyst").get_data(as_text=True)
        self.assertEqual(len(self.questions_in(html)), 7)


class EvaluationTests(FlowBase):
    def setUp(self):
        super().setUp()
        self.start_interview()

    def evaluate(self, qid, answer=LONG_ANSWER):
        return self.post("/api/answer/evaluate", {"question_id": qid, "answer": answer})

    def test_evaluation_has_all_fields_and_computed_score(self):
        data = self.evaluate("q3").get_json()
        ev = data["evaluation"]
        self.assertTrue(data["ok"])
        self.assertEqual(ev["score"], 82)
        for field in ("correctness", "relevance", "strengths", "missing_concepts", "mistakes", "explanation",
                      "model_answer", "improvement_tip", "criterion_scores"):
            self.assertIn(field, ev)

    def test_same_answer_is_not_evaluated_twice(self):
        self.evaluate("q3")
        calls = len(self.fake.calls)
        again = self.evaluate("q3").get_json()
        self.assertTrue(again["cached"])
        self.assertEqual(len(self.fake.calls), calls)

    def test_empty_answer_scores_zero_without_calling_the_model(self):
        calls = len(self.fake.calls)
        ev = self.evaluate("q4", "   ").get_json()["evaluation"]
        self.assertEqual((ev["score"], ev["status"]), (0, "no_answer"))
        self.assertEqual(len(self.fake.calls), calls)

    def test_follow_ups_are_bounded_unique_and_inserted_after_their_parent(self):
        first = self.evaluate("q1").get_json()["follow_up"]
        second = self.evaluate("q2").get_json()["follow_up"]
        third = self.evaluate("q3").get_json()["follow_up"]  # limit of 2 reached
        self.assertEqual((first["id"], first["parent_id"], first["kind"]), ("f1", "q1", "follow_up"))
        self.assertEqual(second["id"], "f2")
        self.assertIsNone(third)
        self.assertNotEqual(first["text"], second["text"])
        # a follow-up question itself never spawns another follow-up
        self.assertIsNone(self.evaluate("f1").get_json()["follow_up"])
        # a re-answered parent never gets a second follow-up
        self.assertIsNone(self.evaluate("q1", LONG_ANSWER + " I also write unit tests for each handler.").get_json()["follow_up"])
        inter = interview_store.get(next(iter(interview_store._store)))["interview"]
        self.assertEqual([q["id"] for q in inter["questions"]][:4], ["q1", "f1", "q2", "f2"])
        self.assertEqual(len(inter["questions"]), 9)  # 7 main + at most 2 follow-ups

    def test_follow_up_that_duplicates_an_existing_question_is_dropped(self):
        def dup_responder(kwargs):
            ev = eval_payload(prompt_of(kwargs))
            ev["follow_up"] = {"question": "What is the difference between a list and a tuple in Python?", "topic": "x"}
            return ev
        app_module.app.config["AI_SERVICE"] = NemotronService(api_key=SECRET, client=FakeClient(dup_responder))
        self.assertIsNone(self.evaluate("q1").get_json()["follow_up"])

    def test_short_answers_do_not_trigger_follow_ups(self):
        self.assertIsNone(self.evaluate("q1", "I use try except.").get_json()["follow_up"])

    def test_api_failure_returns_safe_error_and_keeps_progress(self):
        self.evaluate("q1")
        app_module.app.config["AI_SERVICE"] = NemotronService(api_key=SECRET, client=FakeClient(make_responder(fail_on="criterion_scores")))
        resp = self.evaluate("q2")
        self.assertEqual(resp.status_code, 503)
        self.assertFalse(resp.get_json()["ok"])
        self.assertNotIn(SECRET, resp.get_data(as_text=True))
        # earlier evaluation and interview state are intact; retrying later works
        app_module.app.config["AI_SERVICE"] = NemotronService(api_key=SECRET, client=FakeClient(make_responder()))
        self.assertTrue(self.evaluate("q2").get_json()["ok"])
        self.assertTrue(self.evaluate("q1").get_json()["cached"])

    def test_bad_requests(self):
        self.assertEqual(self.post("/api/answer/evaluate", {"answer": "x"}).status_code, 400)
        self.assertEqual(self.evaluate("zzz").status_code, 404)
        fresh = app_module.app.test_client()
        self.assertEqual(fresh.post("/api/answer/evaluate", json={"question_id": "q1", "answer": "x"}).status_code, 409)


class ReportTests(FlowBase):
    def setUp(self):
        super().setUp()
        self.start_interview()
        self.post("/api/resume/analyze")

    def answers(self, texts):
        return [{"question_id": qid, "answer": text} for qid, text in texts.items()]

    def test_overall_score_matches_stored_evaluations_and_policy(self):
        self.post("/api/answer/evaluate", {"question_id": "q1", "answer": LONG_ANSWER})
        self.post("/api/answer/evaluate", {"question_id": "q2", "answer": LONG_ANSWER})
        payload = {"answers": self.answers({"q1": LONG_ANSWER, "q2": LONG_ANSWER, "q3": "", "q4": "", "q5": "", "q6": "", "q7": ""})}
        report = self.post("/api/report/generate", payload).get_json()["report"]
        main_scores = [r["evaluation"]["score"] for r in report["records"] if r["kind"] == "main"]
        self.assertEqual(main_scores, [79, 81, 0, 0, 0, 0, 0])  # behavioral, project, then 5 unanswered
        self.assertEqual(report["summary"]["overall_score"], round(sum(main_scores) / 7))
        self.assertEqual(report["summary"]["answered_main_count"], 2)
        # follow-ups (f1, f2) are listed but unanswered and do not change the score
        self.assertEqual([r["id"] for r in report["records"]][:4], ["q1", "f1", "q2", "f2"])
        self.assertTrue(report["ai_summary"]["overall_comment"])
        self.assertEqual(report["resume_analysis"]["role_match_score"], 72)
        self.assertNotIn(SECRET, json.dumps(report))
        self.assertNotIn("criteria", json.dumps(report["records"]))

    def test_report_still_loads_when_some_evaluations_failed(self):
        self.post("/api/answer/evaluate", {"question_id": "q1", "answer": LONG_ANSWER})
        # AI goes down before the candidate finishes: q2 was answered but never evaluated
        app_module.app.config["AI_SERVICE"] = NemotronService(api_key=SECRET, client=FakeClient(make_responder(fail_on="<context>")))
        payload = {"answers": self.answers({"q1": LONG_ANSWER, "q2": LONG_ANSWER})}
        report = self.post("/api/report/generate", payload).get_json()["report"]
        by_id = {r["id"]: r for r in report["records"]}
        self.assertEqual(by_id["q2"]["evaluation"]["status"], "unavailable")
        self.assertIsNone(by_id["q2"]["evaluation"]["score"])
        self.assertTrue(by_id["q2"]["evaluation"]["static_reference_answer"])
        self.assertEqual(report["summary"]["not_evaluated_count"], 1)
        self.assertEqual(report["summary"]["overall_score"], round((79 + 0 * 5) / 6))  # q2 excluded; 5 unanswered = 0
        self.assertIsNotNone(report["ai_summary_error"])  # summary call failed -> reported, not faked
        self.assertIsNone(report["ai_summary"])

    def test_report_retries_missing_evaluations_once_ai_is_back(self):
        app_module.app.config["AI_SERVICE"] = NemotronService(api_key=SECRET, client=FakeClient(make_responder(fail_on="criterion_scores")))
        self.post("/api/answer/evaluate", {"question_id": "q1", "answer": LONG_ANSWER})
        app_module.app.config["AI_SERVICE"] = NemotronService(api_key=SECRET, client=FakeClient(make_responder()))
        report = self.post("/api/report/generate", {"answers": self.answers({"q1": LONG_ANSWER})}).get_json()["report"]
        self.assertEqual({r["id"]: r for r in report["records"]}["q1"]["evaluation"]["status"], "evaluated")

    def test_report_without_session_is_a_clean_error(self):
        fresh = app_module.app.test_client()
        resp = fresh.post("/api/report/generate", json={"answers": []})
        self.assertEqual(resp.status_code, 409)
        self.assertFalse(resp.get_json()["ok"])


class ConfigTests(unittest.TestCase):
    def test_no_secret_is_hard_coded_in_source_or_templates(self):
        root = Path(__file__).resolve().parent.parent
        for path in list(root.glob("*.py")) + list((root / "modules").glob("*.py")) + list((root / "static").rglob("*.js")) \
                + list((root / "templates").glob("*.html")):
            text = path.read_text(encoding="utf-8")
            self.assertNotRegex(text, r"nvapi-[A-Za-z0-9_-]{10,}", msg=str(path))
        # Browser code must never read the key; templates may only name the variable in help text.
        for path in (root / "static").rglob("*.js"):
            self.assertNotIn("NVIDIA_API_KEY", path.read_text(encoding="utf-8"), msg=str(path))
        for path in (root / "templates").glob("*.html"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("api_key", text.lower().replace("nvidia_api_key", ""), msg=str(path))

    def test_env_example_uses_dotenv_syntax_and_requested_model(self):
        text = (Path(__file__).resolve().parent.parent / ".env.example").read_text(encoding="utf-8")
        self.assertIn("NVIDIA_MODEL=nvidia/nemotron-3.5-super-vl-120b-a12b", text)
        self.assertIn("NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1", text)
        # The original hand-written .env contained Python code, which dotenv cannot parse.
        code_lines = [line.strip() for line in text.splitlines()
                      if line.strip() and not line.strip().startswith("#")]
        self.assertNotIn("os.environ", "\n".join(code_lines))
        for stripped in code_lines:
            self.assertIn("=", stripped, msg=stripped)


if __name__ == "__main__":
    unittest.main()
