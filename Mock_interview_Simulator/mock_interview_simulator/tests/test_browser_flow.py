"""Browser-level test of static/js/interview.js in jsdom against a live Flask server with a FAKE AI.

Camera / microphone / speech are stubbed (no hardware in CI), so this proves the interview LOGIC
(evaluation lock, no double submit, bounded follow-ups, report hand-off) - not real voice or camera
behaviour. Skipped automatically when node or jsdom is not installed.
"""
import json
import os
import shutil
import subprocess
import threading
import time
import unittest
from pathlib import Path

from werkzeug.serving import make_server

import app as app_module
from modules import interview_store
from modules.nemotron_service import NemotronService
from tests.fakes import FakeClient
from tests.test_app_flow import make_responder

ROOT = Path(__file__).resolve().parent.parent
NODE_PATH = os.environ.get("JSDOM_NODE_PATH", "/tmp/jsd/node_modules")


def _jsdom_available():
    if not shutil.which("node"):
        return False
    return subprocess.run(["node", "-e", "require('jsdom')"], env={**os.environ, "NODE_PATH": NODE_PATH},
                          capture_output=True).returncode == 0


@unittest.skipUnless(_jsdom_available(), "node + jsdom required (npm i jsdom; set JSDOM_NODE_PATH)")
class InterviewBrowserFlow(unittest.TestCase):
    def test_interview_logic_in_jsdom(self):
        interview_store.clear_all()
        base_responder = make_responder()

        def slow(kwargs):
            time.sleep(0.4)  # make the "evaluating" window observable
            return base_responder(kwargs)
        fake = FakeClient(slow)
        app_module.app.config["AI_SERVICE"] = NemotronService(api_key="k", client=fake)
        server = make_server("127.0.0.1", 5057, app_module.app, threaded=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            proc = subprocess.run(["node", str(ROOT / "tests/browser/interview_harness.js"), "http://127.0.0.1:5057", str(ROOT)],
                                  env={**os.environ, "NODE_PATH": NODE_PATH}, capture_output=True, text=True,
                                  encoding="utf-8", timeout=120)
        finally:
            server.shutdown()
            app_module.app.config.pop("AI_SERVICE", None)
        self.assertEqual(proc.returncode, 0, proc.stderr[-800:])
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        print("\nbrowser harness:", json.dumps(out, indent=1))
        self.assertEqual(out["initialQuestionCount"], 7)
        self.assertTrue(out["doubleSubmitBlocked"])
        self.assertTrue(out["nextLockedWhileEvaluating"])
        # follow-ups were added (max 2) so the visible total grew from 7 to at most 9, never more
        self.assertRegex(out["finalCount"], r"of (8|9)")
        self.assertIsNone(out["reportError"])
        self.assertIsNotNone(out["reportOverall"])
        self.assertLessEqual(len(out["reportRecordIds"]), 9)
        # one evaluation call per answered question (4) + generate (1) + report summary (1) - no duplicates
        eval_calls = [c for c in fake.calls if "criterion_scores" in json.dumps(c["messages"]) and "overall_comment" not in json.dumps(c["messages"])]
        self.assertEqual(len(eval_calls), 4)


if __name__ == "__main__":
    unittest.main()
