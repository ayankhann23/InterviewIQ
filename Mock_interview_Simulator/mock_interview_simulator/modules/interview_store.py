"""Small in-memory, per-browser store for resume text, generated questions and evaluations.

* The browser only holds a signed Flask session cookie containing a random id; resume text
  and evaluations stay on the server (they are never written to disk or logs).
* Entries expire after ``TTL_SECONDS`` and the store is capped at ``MAX_SESSIONS`` so memory
  use stays bounded. State is lost when the Flask process restarts - the app then falls back
  to the built-in questions, and the report explains that evaluations expired.
"""
import secrets
import threading
import time

TTL_SECONDS = 4 * 60 * 60
MAX_SESSIONS = 200

_lock = threading.RLock()
_store = {}


def _new_state():
    return {
        "created": time.time(), "touched": time.time(),
        "resume_text": "", "resume_analysis": None, "resume_analysis_role": None,
        "generated": None,   # {"role", "difficulty", "questions", "source", "padded", "note"}
        "interview": None,   # see start_interview()
    }


def _purge(now):
    expired = [sid for sid, s in _store.items() if now - s["touched"] > TTL_SECONDS]
    for sid in expired:
        _store.pop(sid, None)
    while len(_store) > MAX_SESSIONS:  # drop the least recently used
        oldest = min(_store, key=lambda k: _store[k]["touched"])
        _store.pop(oldest, None)


def new_session():
    with _lock:
        _purge(time.time())
        sid = secrets.token_urlsafe(16)
        _store[sid] = _new_state()
        return sid


def get(sid):
    """Return the live state dict for ``sid`` (or None). Callers mutate it while holding ``lock``."""
    with _lock:
        state = _store.get(sid) if sid else None
        if state is None:
            return None
        if time.time() - state["touched"] > TTL_SECONDS:
            _store.pop(sid, None)
            return None
        state["touched"] = time.time()
        return state


def ensure(sid):
    """Return (sid, state), creating a fresh session when ``sid`` is missing or expired."""
    state = get(sid)
    if state is None:
        sid = new_session()
        state = get(sid)
    return sid, state


def lock():
    return _lock


def start_interview(state, role, difficulty, questions, source):
    """Register the question list for a new interview run (replaces any previous run)."""
    with _lock:
        state["interview"] = {
            "id": secrets.token_hex(6),
            "role": role, "difficulty": difficulty, "source": source,
            "questions": [dict(q) for q in questions],
            "answers": {}, "evaluations": {},
            "followed_parents": set(), "follow_ups_used": 0,
        }
        return state["interview"]


def clear_all():
    with _lock:
        _store.clear()
