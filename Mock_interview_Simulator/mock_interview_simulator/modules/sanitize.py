"""Helpers for handling untrusted text (resumes, spoken answers) before it reaches the AI."""
import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
# Phone numbers: optional +country code, then 8-13 digits possibly separated by spaces, dots or dashes.
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,16}\d)(?!\w)")
LONG_DIGITS_RE = re.compile(r"\b\d{9,}\b")
# Tags used as delimiters in prompts; stripped from user content so it cannot "close" a block.
DELIMITER_RE = re.compile(r"</?\s*(?:resume|candidate_answer|question|context|data|instructions|system)\s*>", re.IGNORECASE)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def redact_pii(text):
    """Remove e-mail addresses, phone numbers and links so they are not sent to the API.

    Names and street addresses cannot be removed reliably with rules, so the README
    documents this limitation honestly.
    """
    text = EMAIL_RE.sub("[email removed]", text or "")
    text = URL_RE.sub("[link removed]", text)
    text = PHONE_RE.sub("[phone removed]", text)
    return LONG_DIGITS_RE.sub("[number removed]", text)


def clean_untrusted(text, max_chars):
    """Strip control characters and prompt-delimiter tags, collapse blank runs and cap length."""
    text = CONTROL_RE.sub(" ", str(text or ""))
    text = DELIMITER_RE.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:max_chars]


def normalize_question(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower().replace("follow-up:", "")).strip()


def is_duplicate_question(candidate, existing, threshold=0.72):
    """True when ``candidate`` equals or is very similar (token overlap) to any existing question."""
    key = normalize_question(candidate)
    if not key:
        return True
    tokens = set(key.split())
    for other in existing:
        other_key = normalize_question(other)
        if key == other_key:
            return True
        other_tokens = set(other_key.split())
        if tokens and other_tokens:
            overlap = len(tokens & other_tokens) / len(tokens | other_tokens)
            if overlap >= threshold:
                return True
    return False
