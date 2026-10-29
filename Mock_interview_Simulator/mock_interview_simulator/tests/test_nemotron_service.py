import os
import unittest
from unittest import mock

import httpx
import openai

from modules.nemotron_service import (AIServiceError, DEFAULT_BASE_URL, DEFAULT_MODEL, NemotronService,
                                      extract_json_object)
from tests.fakes import FakeClient

SECRET = "nvapi-TEST-SECRET-DO-NOT-LEAK"


def _status_error(cls, status):
    request = httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions")
    response = httpx.Response(status, request=request, text=f"boom {SECRET}")
    return cls(f"error with {SECRET}", response=response, body=None)


def _service(responder, **env):
    settings = {"NVIDIA_RETRY_BACKOFF": "0", **env}
    with mock.patch.dict(os.environ, settings, clear=False):
        svc = NemotronService(api_key=SECRET, client=FakeClient(responder))
    return svc


class JsonExtractionTests(unittest.TestCase):
    def test_plain_fenced_and_wrapped_json(self):
        self.assertEqual(extract_json_object('{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json_object('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json_object('Sure! Here it is: {"a": {"b": "}"}} hope that helps'), {"a": {"b": "}"}})

    def test_rejects_garbage(self):
        for bad in ["", "   ", "no json here", "[1, 2, 3]", '{"a": ', None]:
            with self.assertRaises(ValueError):
                extract_json_object(bad)


class RequestShapeTests(unittest.TestCase):
    def test_uses_supplied_model_endpoint_and_sampling_settings(self):
        clean = {k: v for k, v in os.environ.items() if not k.startswith("NVIDIA_")}
        with mock.patch.dict(os.environ, clean, clear=True):  # nothing set: the documented defaults apply
            svc = NemotronService(api_key=SECRET, client=FakeClient(lambda kw: {"status": "ok"}))
            svc.chat_json("sys", "user")
            call = svc._client.calls[0]
            self.assertEqual(call["model"], "nvidia/nemotron-3.5-super-vl-120b-a12b")
            self.assertEqual(DEFAULT_MODEL, "nvidia/nemotron-3.5-super-vl-120b-a12b")
            self.assertEqual(DEFAULT_BASE_URL, "https://integrate.api.nvidia.com/v1")
            self.assertEqual(call["top_p"], 0.95)
            self.assertEqual(call["seed"], 42)
            self.assertIs(call["stream"], False)
            self.assertEqual(call["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})
            self.assertEqual(call["messages"][1]["content"][0]["type"], "text")
            self.assertEqual(svc.max_retries, 2)

    def test_retry_count_is_configurable(self):
        with mock.patch.dict(os.environ, {"NVIDIA_MAX_RETRIES": "4"}):
            svc = NemotronService(api_key=SECRET, client=FakeClient(lambda kw: {"status": "ok"}))
            self.assertEqual(svc.max_retries, 4)
        with mock.patch.dict(os.environ, {"NVIDIA_MAX_RETRIES": "not-a-number"}):
            svc = NemotronService(api_key=SECRET, client=FakeClient(lambda kw: {"status": "ok"}))
            self.assertEqual(svc.max_retries, 2)

    def test_thinking_switch_and_model_override_come_from_env(self):
        with mock.patch.dict(os.environ, {"NVIDIA_ENABLE_THINKING": "true", "NVIDIA_MODEL": "nvidia/other-model"}):
            svc = NemotronService(api_key=SECRET, client=FakeClient(lambda kw: {"status": "ok"}))
            svc.chat_json("s", "u")
            call = svc._client.calls[0]
            self.assertEqual(call["model"], "nvidia/other-model")
            self.assertTrue(call["extra_body"]["chat_template_kwargs"]["enable_thinking"])

    def test_real_client_is_built_with_supplied_base_url_and_key_never_in_prompt(self):
        with mock.patch.dict(os.environ, {"NVIDIA_BASE_URL": ""}):
            svc = NemotronService(api_key=SECRET)
            client = svc._get_client()
            self.assertEqual(str(client.base_url).rstrip("/"), "https://integrate.api.nvidia.com/v1")
            self.assertEqual(client.api_key, SECRET)

    def test_not_configured_without_key(self):
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": ""}):
            svc = NemotronService(api_key="")
            self.assertFalse(svc.configured)
            with self.assertRaises(AIServiceError) as ctx:
                svc.chat_json("s", "u")
            self.assertEqual(ctx.exception.kind, "not_configured")


class ErrorHandlingTests(unittest.TestCase):
    def _kind(self, exc):
        def responder(kwargs):
            raise exc
        svc = _service(responder)
        with self.assertRaises(AIServiceError) as ctx:
            svc.chat_json("s", "u")
        # Secrets must never reach the user-facing message or the exception text.
        self.assertNotIn(SECRET, ctx.exception.user_message)
        self.assertNotIn(SECRET, str(ctx.exception))
        return ctx.exception

    def test_auth_error(self):
        err = self._kind(_status_error(openai.AuthenticationError, 401))
        self.assertEqual(err.kind, "auth")
        self.assertFalse(err.retryable)

    def test_rate_limit(self):
        err = self._kind(_status_error(openai.RateLimitError, 429))
        self.assertEqual((err.kind, err.retryable), ("rate_limit", True))

    def test_server_error(self):
        err = self._kind(_status_error(openai.InternalServerError, 503))
        self.assertEqual((err.kind, err.retryable), ("server", True))

    def test_model_not_found(self):
        self.assertEqual(self._kind(_status_error(openai.NotFoundError, 404)).kind, "model")

    def test_timeout_and_network(self):
        request = httpx.Request("POST", "https://x")
        self.assertEqual(self._kind(openai.APITimeoutError(request=request)).kind, "timeout")
        self.assertEqual(self._kind(openai.APIConnectionError(request=request)).kind, "network")

    def test_unexpected_exception_is_wrapped(self):
        self.assertEqual(self._kind(RuntimeError(f"secret {SECRET}")).kind, "unknown")

    def test_empty_and_missing_content(self):
        for payload in ["", "   ", None]:
            svc = _service(lambda kw, p=payload: p)
            with self.assertRaises(AIServiceError) as ctx:
                svc.chat_json("s", "u")
            self.assertEqual(ctx.exception.kind, "empty")


class TransientRetryTests(unittest.TestCase):
    def test_transient_server_error_is_retried_and_then_succeeds(self):
        calls = {"n": 0}

        def responder(kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _status_error(openai.InternalServerError, 500)
            return {"status": "ok"}

        svc = _service(responder)
        self.assertEqual(svc.chat_json("s", "u"), {"status": "ok"})
        self.assertEqual(calls["n"], 2)

    def test_auth_error_is_not_retried(self):
        calls = {"n": 0}

        def responder(kwargs):
            calls["n"] += 1
            raise _status_error(openai.AuthenticationError, 401)

        svc = _service(responder)
        with self.assertRaises(AIServiceError) as ctx:
            svc.chat_json("s", "u")
        self.assertEqual(ctx.exception.kind, "auth")
        self.assertEqual(calls["n"], 1)


class RetryOnInvalidOutputTests(unittest.TestCase):
    def test_malformed_then_valid_json_succeeds_after_one_reask(self):
        replies = iter(["this is not json", '{"status": "ok"}'])
        svc = _service(lambda kw: next(replies))
        self.assertEqual(svc.chat_json("s", "u"), {"status": "ok"})
        self.assertEqual(len(svc._client.calls), 2)
        self.assertIn("not valid", svc._client.calls[1]["messages"][1]["content"][0]["text"])

    def test_always_malformed_raises_after_two_attempts(self):
        svc = _service(lambda kw: "nope")
        with self.assertRaises(AIServiceError) as ctx:
            svc.chat_json("s", "u")
        self.assertEqual(ctx.exception.kind, "invalid_output")
        self.assertEqual(len(svc._client.calls), 2)

    def test_validator_failure_triggers_reask(self):
        replies = iter([{"x": 1}, {"ok": True}])

        def validator(d):
            if "ok" not in d:
                raise ValueError("missing ok")
            return d
        svc = _service(lambda kw: next(replies))
        self.assertEqual(svc.chat_json("s", "u", validator=validator), {"ok": True})

    def test_ping(self):
        svc = _service(lambda kw: {"status": "ok"})
        model, seconds, data = svc.ping()
        self.assertEqual(model, DEFAULT_MODEL)
        self.assertEqual(data, {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
