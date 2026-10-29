"""Predefined interview questions (used as the fallback when AI generation is unavailable).

The questions are the same ones that used to live inside app.py. Each question is now a
dict with an id, topic, type and expected concepts so the AI evaluator has context.
"""
from modules.evaluator import reference_concepts

QUESTION_BANK = {
    "AI/ML Engineer": [
        "Tell me about yourself and your interest in AI/ML.",
        "Describe an AI/ML project you built and explain your role.",
        "How would you reduce overfitting in a machine-learning model?",
        "How would you explain model performance to a non-technical stakeholder?",
        "How do you split data into training, validation, and test sets, and why?",
        "Which metrics would you choose for an imbalanced classification problem?",
        "How would you monitor a machine-learning model after deployment?",
    ],
    "Python Developer": [
        "Tell me about yourself and your Python experience.",
        "Describe a Python project you built and your role.",
        "What is the difference between a list and a tuple in Python?",
        "How do you handle exceptions in a Python application?",
        "What is the difference between a class and an object?",
        "How would you test and debug a Python function?",
        "How would you call an external API safely from a Python application?",
    ],
    "Web Developer": [
        "Tell me about yourself and your web development experience.",
        "Describe a website you built and your role.",
        "What is the difference between HTML, CSS, and JavaScript?",
        "How would you make a web page responsive?",
        "What happens when a browser requests a web page from a server?",
        "How do you improve web accessibility?",
        "How would you diagnose a slow-loading website?",
    ],
    "Data Analyst": [
        "Tell me about yourself and your interest in data analysis.",
        "Describe a data project you worked on and your role.",
        "How do you handle missing or inconsistent data?",
        "How would you explain a chart or insight to a non-technical audience?",
        "What is the difference between an inner join and a left join in SQL?",
        "How do you choose a useful KPI for a business question?",
        "How would you check whether a data trend is meaningful rather than noise?",
    ],
}
GENERIC_QUESTIONS = [
    "Tell me about yourself.",
    "Describe a project you built and explain your role.",
    "What was a challenge you faced in a project, and how did you solve it?",
    "What skills would you like to improve next?",
    "How do you prioritize tasks when you have competing deadlines?",
    "Tell me about a time you received feedback and acted on it.",
    "What would you like to learn in your next role?",
]

BEHAVIOURAL_HINTS = ("tell me about yourself", "tell me about a time", "prioritize", "would you like to learn",
                     "skills would you like to improve", "challenge you faced")
PROJECT_HINTS = ("describe a", "project", "website you built")


def _question_type(text):
    low = text.lower()
    if any(h in low for h in BEHAVIOURAL_HINTS):
        return "behavioral"
    if any(h in low for h in PROJECT_HINTS):
        return "project"
    return "technical"


def build_fallback_questions(role, difficulty="intermediate"):
    """Return the predefined questions for ``role`` as question dicts (stable ids q1..qN)."""
    texts = QUESTION_BANK.get(role, GENERIC_QUESTIONS)
    questions = []
    for index, text in enumerate(texts, start=1):
        questions.append({
            "id": f"q{index}",
            "text": text,
            "topic": "General" if _question_type(text) != "technical" else role,
            "difficulty": difficulty,
            "type": _question_type(text),
            "kind": "main",
            "criteria": reference_concepts(text, role),
        })
    return questions
