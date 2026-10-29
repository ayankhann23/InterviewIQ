"""Reusable NVIDIA Nemotron service for InterviewIQ.

This is the ONLY module that talks to the NVIDIA API. Every AI feature (resume
analysis, question generation, answer evaluation, report summary) calls
``NemotronService.chat_json`` so that authentication, timeouts, error handling and
JSON validation live in one place.

Configuration (all from environment variables / .env, never hard-coded):
    NVIDIA_API_KEY          required - stays on the server, never sent to the browser
    NVIDIA_MODEL            default nvidia/nemotron-3.5-super-vl-120b-a12b
    NVIDIA_BASE_URL         default https://integrate.api.nvidia.com/v1 (OpenAI-compatible)
    NVIDIA_TIMEOUT_SECONDS  default 45
    NVIDIA_MAX_TOKENS       default 4096
    NVIDIA_TEMPERATURE      default 0.3
    NVIDIA_TOP_P            default 0.95
    NVIDIA_SEED             default 42 (set to empty to disable)
    NVIDIA_ENABLE_THINKING  default false (true = Nemotron "thinking" mode, slower)
    NVIDIA_MAX_RETRIES      default 2 (SDK retries for transient 429/5xx/network errors)
    NVIDIA_RETRY_BACKOFF    default 0.8 seconds (app-level retry delay for transient errors)

The model id, base URL, sampling values and ``enable_thinking`` switch come from the
configuration snippet supplied for this project.
"""
import json
import logging
import os
import re
import time

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


class AIServiceError(Exception):
    """Raised for any AI failure. ``user_message`` is always safe to show in the UI."""

    def __init__(self, kind, user_message, retryable=False):
        super().__init__(f"{kind}: {user_message}")
        self.kind = kind
        self.user_message = user_message
        self.retryable = retryable


def _env_float(name, default):
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return float(default)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return int(default)


def _env_bool(name, default=False):
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def extract_json_object(text):
    """Return the first JSON object found in ``text`` (tolerates ```json fences and
    surrounding prose). Raises ValueError when no valid JSON object is present."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty model response")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    # Scan for a balanced {...} block, respecting strings and escapes.
    start = cleaned.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(cleaned[start:i + 1])
                        if isinstance(value, dict):
                            return value
                    except json.JSONDecodeError:
                        pass
                    break
        start = cleaned.find("{", start + 1)
    raise ValueError("no valid JSON object in model response")


class NemotronService:
    def __init__(self, api_key=None, model=None, base_url=None, client=None):
        self.api_key = (api_key if api_key is not None else os.environ.get("NVIDIA_API_KEY", "")).strip()
        self.model = (model or os.environ.get("NVIDIA_MODEL", "").strip() or DEFAULT_MODEL)
        self.base_url = (base_url or os.environ.get("NVIDIA_BASE_URL", "").strip() or DEFAULT_BASE_URL)
        self.timeout = _env_float("NVIDIA_TIMEOUT_SECONDS", 45)
        self.max_tokens = _env_int("NVIDIA_MAX_TOKENS", 4096)
        self.temperature = _env_float("NVIDIA_TEMPERATURE", 0.3)
        self.top_p = _env_float("NVIDIA_TOP_P", 0.95)
        seed_raw = os.environ.get("NVIDIA_SEED", "42").strip()
        self.seed = int(seed_raw) if seed_raw.lstrip("-").isdigit() else None
        self.enable_thinking = _env_bool("NVIDIA_ENABLE_THINKING", False)
        # NVIDIA's hosted endpoint occasionally returns a transient 500 on the first
        # request after an idle period. The SDK retries 429/5xx/timeouts with backoff.
        self.max_retries = max(0, min(5, _env_int("NVIDIA_MAX_RETRIES", 2)))
        self.retry_backoff = max(0.0, min(10.0, _env_float("NVIDIA_RETRY_BACKOFF", 0.8)))
        self._client = client  # injectable for tests

    @property
    def configured(self):
        return bool(self.api_key) or self._client is not None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise AIServiceError(
                    "not_configured",
                    "AI features are not configured. Add NVIDIA_API_KEY to the .env file and restart the app.",
                )
            try:
                from openai import OpenAI  # imported lazily so the app still starts without the package
            except ImportError:
                raise AIServiceError(
                    "not_configured",
                    "The 'openai' package is not installed. Run: python -m pip install -r requirements.txt",
                ) from None
            # max_retries lets the SDK retry transient 429/5xx/connection errors.
            self._client = OpenAI(
                base_url=self.base_url, api_key=self.api_key,
                timeout=self.timeout, max_retries=self.max_retries,
            )
        return self._client

    def _request_text(self, system_prompt, user_prompt, max_tokens, temperature):
        """One chat-completion call. Returns the assistant text or raises AIServiceError."""
        client = self._get_client()
        import openai  # SDK exception classes
        messages = [
            {"role": "system", "content": system_prompt},
            # Same typed-content-part format as the supplied configuration snippet.
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": False,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        try:
            completion = client.chat.completions.create(**kwargs)
        except openai.AuthenticationError:
            raise AIServiceError("auth", "The NVIDIA API key was rejected. Check NVIDIA_API_KEY in .env.") from None
        except openai.PermissionDeniedError:
            raise AIServiceError("auth", "The NVIDIA API key does not have access to this model.") from None
        except openai.RateLimitError:
            raise AIServiceError("rate_limit", "The AI service is busy (rate limit). Please retry in a moment.", True) from None
        except openai.APITimeoutError:
            raise AIServiceError("timeout", "The AI service took too long to respond. Please retry.", True) from None
        except openai.APIConnectionError:
            raise AIServiceError("network", "Could not reach the AI service. Check your internet connection.", True) from None
        except openai.NotFoundError:
            raise AIServiceError("model", "The configured NVIDIA model or endpoint was not found. Check NVIDIA_MODEL.") from None
        except openai.BadRequestError:
            raise AIServiceError("bad_request", "The AI service rejected the request.") from None
        except openai.APIStatusError as exc:
            status = getattr(exc, "status_code", 0)
            # Log only status + request id. Never log prompts, answers or the API key.
            logger.warning("NVIDIA API error status=%s request_id=%s",
                           status, getattr(exc, "request_id", None))
            raise AIServiceError("server", f"The AI service returned an error ({status}). Please retry.", status >= 500) from None
        except AIServiceError:
            raise
        except Exception as exc:  # never leak internals / secrets
            logger.warning("Unexpected AI client failure: %s", type(exc).__name__)
            raise AIServiceError("unknown", "The AI service failed unexpectedly. Please retry.", True) from None

        try:
            text = completion.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            text = None
        if not isinstance(text, str) or not text.strip():
            raise AIServiceError("empty", "The AI service returned an empty response. Please retry.", True)
        return text

    def chat_json(self, system_prompt, user_prompt, validator=None, max_tokens=None,
                  temperature=None, attempts=2, task="ai"):
        """Call the model and return a validated dict.

        ``validator(raw_dict)`` must return the cleaned dict or raise ValueError. Malformed
        JSON or a failed validation triggers ONE re-ask (``attempts=2``); a transient
        service error (5xx/timeout/rate limit) also gets one extra attempt. Network/auth
        errors are not retried indefinitely.
        """
        last_problem = "invalid output"
        total_attempts = max(1, attempts)
        for attempt in range(1, total_attempts + 1):
            started = time.monotonic()
            try:
                text = self._request_text(system_prompt, user_prompt, max_tokens, temperature)
            except AIServiceError as exc:
                # Retry one more time for genuinely transient failures (5xx, timeout,
                # rate limit, network, empty reply). Never retry auth/model/validation.
                if not exc.retryable or attempt >= total_attempts:
                    raise
                delay = min(self.retry_backoff * (2 ** (attempt - 1)), 5.0)
                logger.warning("AI %s transient failure (attempt %d/%d, %s); retrying in %.1fs",
                               task, attempt, total_attempts, exc.kind, delay)
                if delay:
                    time.sleep(delay)
                continue
            try:
                data = extract_json_object(text)
                result = validator(data) if validator else data
                logger.info("AI %s ok (attempt %d, %.1fs, model=%s)", task, attempt, time.monotonic() - started, self.model)
                return result
            except ValueError as exc:
                # Log only the reason, never the model text or user content.
                last_problem = str(exc)[:120]
                logger.warning("AI %s output rejected (attempt %d): %s", task, attempt, last_problem)
                user_prompt = (
                    user_prompt
                    + "\n\nYour previous reply was not valid. Reply again with ONLY one JSON object that follows "
                      "the required schema exactly - no markdown, no commentary."
                )
        raise AIServiceError("invalid_output", "The AI returned an unusable response. Please retry.", True)

    def ping(self):
        """Minimal request used by scripts/check_nemotron.py. Returns (model, seconds, data)."""
        def _check(data):
            if str(data.get("status", "")).lower() != "ok":
                raise ValueError("unexpected ping reply")
            return data

        started = time.monotonic()
        data = self.chat_json(
            "You reply with JSON only.",
            'Reply with exactly this JSON object: {"status": "ok"}',
            validator=_check, max_tokens=64, temperature=0, task="ping",
        )
        return self.model, time.monotonic() - started, data
