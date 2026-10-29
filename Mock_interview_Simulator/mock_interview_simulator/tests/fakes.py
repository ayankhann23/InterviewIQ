"""Test doubles: a fake OpenAI-compatible client so tests never call the real NVIDIA API."""
import json
from types import SimpleNamespace


def completion(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FakeClient:
    """Mimics ``OpenAI(...).chat.completions.create``.

    ``responder(kwargs)`` returns the assistant text (str / dict -> JSON) or raises an exception.
    All calls are recorded in ``self.calls`` (the kwargs), so tests can assert which model,
    sampling settings and prompts were used.
    """

    def __init__(self, responder):
        self.responder = responder
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.responder(kwargs)
        if isinstance(result, dict):
            result = json.dumps(result)
        return completion(result)


def prompt_of(kwargs):
    """The user prompt text from recorded kwargs."""
    content = kwargs["messages"][-1]["content"]
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content)
    return content


GOOD_EVAL = {
    "criterion_scores": {"correctness": 85, "relevance": 90, "completeness": 70, "reasoning": 75, "clarity": 80},
    "correctness": "Mostly correct.", "relevance": "Directly answers the question.",
    "strengths": ["Names try/except and logging"], "missing_concepts": ["finally for cleanup"], "mistakes": [],
    "explanation": "Solid answer that omits cleanup handling.",
    "model_answer": "Use try/except with specific exceptions, finally for cleanup, and log errors.",
    "improvement_tip": "Mention finally blocks and custom exceptions.",
    "follow_up": {"question": "How would you test that your error handling works correctly?", "topic": "Testing"},
}

GOOD_RESUME = {
    "summary": "Final-year AI-ML diploma student with Python and Flask projects.",
    "detected_skills": ["Python", "Flask", "SQL"],
    "education": ["Diploma in AI-ML"],
    "experience_summary": "None listed",
    "project_summary": ["Traffic sign CNN: combined code and compared models"],
    "certifications_achievements": [],
    "strengths": ["Hands-on Python projects"],
    "skill_gaps": [{"skill": "Docker", "status": "not_mentioned", "note": "Not mentioned in the resume."}],
    "recommended_roles": ["Python Developer"],
    "resume_improvements": ["Add measurable results to project bullets"],
    "role_match_score": 72, "role_match_note": "Good Python base, limited deployment evidence.",
}

GOOD_QUESTIONS = {"questions": [
    {"question": "Tell me about yourself and your interest in Python development.", "topic": "Background", "difficulty": "intermediate", "type": "behavioral", "evaluation_criteria": ["background", "skills", "goal"]},
    {"question": "In your Traffic Sign CNN project, how did you compare the different models?", "topic": "Projects", "difficulty": "intermediate", "type": "project", "evaluation_criteria": ["metrics", "validation"]},
    {"question": "How do you handle exceptions in a Python application?", "topic": "Python", "difficulty": "intermediate", "type": "technical", "evaluation_criteria": ["try", "except", "finally"]},
    {"question": "What is the difference between a list and a tuple in Python?", "topic": "Python", "difficulty": "intermediate", "type": "technical", "evaluation_criteria": ["mutable", "immutable"]},
    {"question": "How would you structure a Flask application that has several routes?", "topic": "Flask", "difficulty": "intermediate", "type": "technical", "evaluation_criteria": ["blueprints", "config"]},
    {"question": "How would you protect an API key used by a Flask backend?", "topic": "Security", "difficulty": "intermediate", "type": "technical", "evaluation_criteria": ["environment variables"]},
    {"question": "How would you write unit tests for a function that parses a PDF?", "topic": "Testing", "difficulty": "intermediate", "type": "technical", "evaluation_criteria": ["unit tests", "fixtures"]},
]}

GOOD_SUMMARY = {
    "overall_comment": "A reasonable practice interview with solid fundamentals.",
    "technical_strengths": ["Clear Python basics"], "improvement_areas": ["Testing depth"],
    "common_mistakes": ["Omitted cleanup handling"], "resume_skill_gaps": ["Docker not shown"],
    "study_recommendations": ["Practise pytest"],
    "improvement_plan": [{"step": "Testing", "detail": "Write tests for two small projects."}],
}


def eval_payload(prompt):
    """A valid evaluation whose criterion_scores match the rubric named in the prompt (like a real model would)."""
    import copy
    import re
    from modules.evaluator import rubric_for
    qtype = (re.search(r"Question type: (\w+)", prompt) or [None, "technical"])[1]
    payload = copy.deepcopy(GOOD_EVAL)
    payload["criterion_scores"] = {name: GOOD_EVAL["criterion_scores"].get(name, 80) for name in rubric_for(qtype)}
    return payload


def smart_responder(kwargs):
    """Routes by prompt content, returning a valid response for each AI task."""
    p = prompt_of(kwargs)
    if "criterion_scores" in p and "overall_comment" not in p:
        return eval_payload(p)
    if '"questions"' in p:
        return GOOD_QUESTIONS
    if "overall_comment" in p:
        return GOOD_SUMMARY
    if '"detected_skills"' in p:
        return GOOD_RESUME
    return {"status": "ok"}
