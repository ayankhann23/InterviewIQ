"""Interview scoring rules for InterviewIQ.

Scores now come from NVIDIA Nemotron (see ai_features.evaluate_answer). This module keeps
everything that must stay deterministic and auditable:

* the scoring rubric (criteria and weights per question type),
* the weighted final-score calculation (the model's own "score" is never trusted),
* the overall-interview scoring policy,
* the static reference concepts / reference answers for the built-in question bank
  (these were the heuristic scorer's data; they are now only context for the AI and an
  honestly-labelled "static reference answer" when the AI is unavailable).

There is NO heuristic fallback score: if the AI evaluation fails, the answer is reported
as "evaluation unavailable" instead of showing a made-up number.
"""
import math
import re

# ---------------------------------------------------------------------------------------
# Rubric: criterion -> weight (each set sums to 1.0). Edit here to change the scoring.
# ---------------------------------------------------------------------------------------
RUBRICS = {
    "technical": {"correctness": 0.40, "relevance": 0.20, "completeness": 0.20, "reasoning": 0.10, "clarity": 0.10},
    "project":   {"correctness": 0.30, "relevance": 0.20, "completeness": 0.20, "reasoning": 0.15, "clarity": 0.15},
    "behavioral": {"relevance": 0.25, "completeness": 0.25, "specificity": 0.20, "reasoning": 0.15, "clarity": 0.15},
}
CRITERION_LABELS = {
    "correctness": "Correctness", "relevance": "Relevance", "completeness": "Completeness",
    "reasoning": "Reasoning", "clarity": "Clarity", "specificity": "Specificity / evidence",
}
DEFAULT_TYPE = "technical"


def rubric_for(question_type):
    return RUBRICS.get(question_type, RUBRICS[DEFAULT_TYPE])


def weighted_score(criterion_scores, question_type):
    """Final 0-100 score from the criterion scores and the rubric weights.

    Raises ValueError if a rubric criterion is missing or not a number.
    """
    rubric = rubric_for(question_type)
    total = 0.0
    for name, weight in rubric.items():
        value = criterion_scores.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"missing criterion score: {name}")
        total += max(0, min(100, float(value))) * weight
    # Round half up (not Python's banker's rounding) so 81.5 always becomes 82.
    return int(math.floor(max(0, min(100, total)) + 0.5 + 1e-9))


# ---------------------------------------------------------------------------------------
# Static reference data kept from the original heuristic evaluator (context only).
# ---------------------------------------------------------------------------------------
# Question-specific concepts are deliberately explicit and easy to customize.
QUESTION_TERMS = {
    "overfitting": ["overfitting", "regularization", "cross-validation", "dropout", "early stopping", "training data", "validation data", "data augmentation", "simpler model"],
    "list and a tuple": ["mutable", "immutable", "list", "tuple", "change", "modify", "parentheses", "square brackets"],
    "exceptions": ["try", "except", "finally", "raise", "error", "exception", "logging", "handle"],
    "responsive": ["responsive", "media query", "flexbox", "grid", "viewport", "mobile", "breakpoint", "relative units"],
    "html, css, and javascript": ["html", "structure", "css", "style", "javascript", "behavior", "interaction", "dom"],
    "missing or inconsistent data": ["missing", "null", "impute", "remove", "median", "mean", "validate", "standardize", "deduplicate", "investigate"],
    "non-technical": ["audience", "simple language", "visual", "context", "impact", "recommendation", "avoid jargon", "business"],
    "model performance": ["accuracy", "precision", "recall", "f1", "confusion matrix", "baseline", "validation", "metrics", "stakeholder"],
    "machine-learning model": ["training", "validation", "test", "features", "model", "data", "metric", "deployment", "prediction"],
    "project": ["goal", "role", "challenge", "solution", "result", "impact", "team", "tools", "learned"],
    "tell me about yourself": ["experience", "skills", "background", "project", "interest", "goal", "learn", "role"],
    "challenge": ["challenge", "problem", "analyze", "solution", "steps", "result", "learned", "improve"],
    "skills would you like to improve": ["improve", "learn", "practice", "feedback", "goal", "course", "project", "skill"],
}


def reference_concepts(question, role):
    q = (question or "").lower()
    for phrase, terms in QUESTION_TERMS.items():
        if phrase in q:
            return terms
    role_terms = {
        "AI/ML Engineer": ["data", "model", "training", "validation", "metric", "features", "prediction", "experiment", "deployment"],
        "Python Developer": ["python", "function", "class", "data", "test", "exception", "library", "api", "debug"],
        "Web Developer": ["html", "css", "javascript", "responsive", "browser", "api", "accessibility", "performance", "testing"],
        "Data Analyst": ["data", "clean", "query", "sql", "visualization", "metric", "insight", "trend", "recommendation"],
    }
    return role_terms.get(role, ["goal", "example", "steps", "result", "reason", "impact", "learned"])


def reference_answer(question, role):
    q = (question or "").lower()
    answers = [
        ("overfitting", "Overfitting happens when a model learns noise or details specific to the training set and performs poorly on unseen data. I would use a validation set or cross-validation, apply regularization or dropout where appropriate, simplify the model, and use early stopping. I would compare validation performance rather than relying only on training accuracy."),
        ("list and a tuple", "A list is mutable, so its elements can be added, removed, or changed; it is written with square brackets. A tuple is immutable after creation and is usually written with parentheses. I choose a list for data that changes and a tuple for a fixed group of values."),
        ("exceptions", "I handle expected failures with try and except blocks, catch specific exception types, and use finally for cleanup that must always run. I log useful context without exposing secrets, validate inputs, and raise an exception when the caller needs to handle the failure. I avoid silently swallowing errors."),
        ("responsive", "I build responsive pages with flexible layouts such as Flexbox or CSS Grid, relative units, and media queries at appropriate breakpoints. I use the viewport meta tag, test on different screen sizes, and check keyboard access and readability as well as layout."),
        ("html, css, and javascript", "HTML defines the structure and meaning of page content, CSS controls presentation and layout, and JavaScript adds behavior and interaction, often by responding to events and updating the DOM. Keeping these responsibilities separated makes the site easier to maintain."),
        ("missing or inconsistent data", "I first profile the data to understand how much is missing and whether the pattern is meaningful. Depending on the context, I may correct source issues, remove unusable records, impute values using an appropriate method such as a median, standardize formats, and validate the results. I document assumptions and check how treatment affects conclusions."),
        ("non-technical", "I start with the audience's business question, explain the key finding in plain language, and use a clear visual or example. I describe the impact and uncertainty, avoid unnecessary jargon, and finish with a practical recommendation or next step."),
        ("training, validation, and test", "The training set is used to fit model parameters, the validation set helps choose features and tune hyperparameters, and the test set gives a final estimate on data not used during development. I keep the test set separate until the end to reduce leakage and over-optimistic results."),
        ("imbalanced classification", "For imbalanced classes, accuracy can be misleading. I would inspect the confusion matrix and consider precision, recall, F1, and PR-AUC based on the cost of false positives and false negatives. I would compare with a baseline and select a threshold using validation data."),
        ("monitor a machine-learning model", "After deployment I would monitor input data quality, feature and prediction drift, latency, errors, and business or model metrics when labels become available. I would define alerts, log versions and experiments, and have a process to investigate, roll back, or retrain safely."),
        ("class and an object", "A class defines a type and the attributes and methods its instances can have. An object is a particular instance of that class with its own state. For example, a Car class can define a drive method, while a specific car object has its own color and model."),
        ("test and debug", "I reproduce the issue with a small example, inspect the traceback and relevant values, and isolate the cause. I write unit tests for expected and edge cases, use a debugger or logging when useful, fix the root cause, and rerun the test suite to check for regressions."),
        ("external api", "I make requests with timeouts, handle HTTP errors and rate limits, validate the response schema, and avoid hard-coding credentials. I store secrets in environment variables, log safely, and use retries with backoff only when retrying is appropriate."),
        ("browser requests", "The browser resolves the domain and establishes a connection, then sends an HTTP request. The server or application processes it and returns a response with status, headers, and content. The browser parses HTML, fetches related assets, applies CSS, runs JavaScript, and renders the page."),
        ("accessibility", "I use semantic HTML, labels for form controls, meaningful alternative text, sufficient color contrast, keyboard navigation, visible focus states, and accessible error messages. I test with keyboard-only navigation and automated checks, while remembering that automated tools cannot catch every issue."),
        ("slow-loading", "I measure before changing anything using browser performance tools and network traces. I check large images and scripts, caching, server response time, render-blocking resources, and unnecessary requests. I optimize the largest bottlenecks and compare performance metrics after each change."),
        ("inner join and a left join", "An INNER JOIN returns rows where the join condition matches in both tables. A LEFT JOIN returns every row from the left table and matching rows from the right; unmatched right-side columns are NULL. I choose based on whether unmatched left records must be retained."),
        ("kpi", "I start from the business objective, define a measurable outcome, and choose a KPI that is actionable and clearly defined. I specify its formula, data source, time window, owner, and segmentation, then check that it reflects the goal rather than encouraging unintended behavior."),
        ("meaningful rather than noise", "I check data quality, sample size, time period, and possible confounders, then compare the trend with a baseline and expected variability. Depending on the question, I may use confidence intervals or a statistical test and validate the finding on another period or segment before making a recommendation."),
        ("tell me about yourself", "A strong response briefly connects my background, relevant skills, one concrete project or achievement, and why those experiences fit this role. I keep it focused on the position and end with what I hope to contribute or learn."),
        ("project", "I explain the project's goal, my specific responsibility, the tools and approach I used, the main challenge, and the result. I distinguish my contribution from the team's work and include measurable impact where I can support it."),
        ("challenge", "I describe the situation and the specific problem, explain how I investigated it, outline the actions I took and why, and state the result. I finish with what I learned and what I would improve next time."),
    ]
    for key, answer in answers:
        if key in q:
            return answer
    if role == "AI/ML Engineer":
        return "I would clarify the objective, inspect and validate the data, establish a simple baseline, choose an appropriate model and metric, and evaluate on data held out from training. I would document assumptions, compare results, and consider deployment and monitoring requirements."
    if role == "Python Developer":
        return "I would break the problem into small functions, validate inputs, handle expected errors explicitly, and write tests for normal and edge cases. I would keep the code readable, use appropriate standard-library or third-party tools, and document important decisions."
    if role == "Web Developer":
        return "I would clarify user and accessibility requirements, build semantic HTML and maintainable CSS and JavaScript, and test across screen sizes and browsers. I would measure performance, handle errors, and verify the result with practical user-focused tests."
    if role == "Data Analyst":
        return "I would clarify the business question, validate and clean the data, use suitable analysis or SQL, and choose a visualization that makes the finding easy to understand. I would check assumptions and communicate limitations with a practical recommendation."
    return "A strong answer explains the situation or goal, the actions taken and why, the result, and what was learned. Use a specific example and connect it to the requirements of the role."


# ---------------------------------------------------------------------------------------
# Evaluation records and overall-interview scoring policy
# ---------------------------------------------------------------------------------------
def unavailable_evaluation(reason, question_text="", role=""):
    """Record for an answer whose AI evaluation could not be produced. No score is invented."""
    return {
        "status": "unavailable",
        "score": None,
        "reason": str(reason)[:200],
        "static_reference_answer": reference_answer(question_text, role) if question_text else "",
    }


def no_answer_evaluation(question_type):
    """Deterministic record for an empty answer (no API call is needed to score 0)."""
    rubric = rubric_for(question_type)
    return {
        "status": "no_answer",
        "score": 0,
        "criterion_scores": {name: 0 for name in rubric},
        "correctness": "No answer was captured.",
        "relevance": "No answer was captured.",
        "strengths": [],
        "missing_concepts": [],
        "mistakes": [],
        "explanation": "No answer was recorded for this question, so it scores 0.",
        "model_answer": "",
        "improvement_tip": "Answer in your own words, even if you are unsure; explain your reasoning.",
    }


def summarize_interview(role, records):
    """Overall score and topic breakdown from stored per-question records.

    Scoring policy (documented in the report and README):
    * Overall score = average of the MAIN questions' scores.
    * A main question with no answer counts as 0 (skipping must not help the score).
    * A main question that was answered but whose AI evaluation FAILED is excluded from
      the average and flagged "not evaluated" (a technical failure must not hurt the candidate).
    * Follow-up questions are optional extras: they are listed and averaged separately but
      never change the overall score.
    """
    main = [r for r in records if r.get("kind") == "main"]
    follow = [r for r in records if r.get("kind") != "main"]

    def counted(r):
        ev = r.get("evaluation") or {}
        return ev.get("status") in ("evaluated", "no_answer") and isinstance(ev.get("score"), int)

    scored_main = [r for r in main if counted(r)]
    overall = round(sum(r["evaluation"]["score"] for r in scored_main) / len(scored_main)) if scored_main else None

    follow_scored = [r for r in follow if counted(r) and (r["evaluation"].get("status") == "evaluated")]
    follow_avg = round(sum(r["evaluation"]["score"] for r in follow_scored) / len(follow_scored)) if follow_scored else None

    topics = {}
    for r in records:
        if r.get("kind") == "main" and counted(r):
            topics.setdefault(r.get("topic") or "General", []).append(r["evaluation"]["score"])
    topic_performance = [
        {"topic": name, "average_score": round(sum(v) / len(v)), "questions": len(v)}
        for name, v in sorted(topics.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
    ]
    return {
        "role": str(role or "Not specified")[:100],
        "overall_score": overall,
        "main_question_count": len(main),
        "answered_main_count": sum(1 for r in main if (r.get("answer") or "").strip()),
        "not_evaluated_count": sum(1 for r in main if (r.get("answer") or "").strip() and not counted(r)),
        "follow_up_count": len(follow),
        "follow_up_average": follow_avg,
        "topic_performance": topic_performance,
        "scoring_method": (
            "Overall score = average of main-question scores. Unanswered main questions count as 0; "
            "answered questions whose AI evaluation failed are excluded and marked 'not evaluated'. "
            "Follow-up questions are shown separately and do not change the overall score."
        ),
    }


def build_fallback_coaching_summary(role, summary, records, resume_analysis=None):
    """Deterministic, structured coaching summary generated from evaluated question records.

    Ensures the final report always provides high-value coaching feedback even if
    the live LLM call is temporarily unavailable or returns a transient error.
    """
    overall_score = summary.get("overall_score")
    role_name = role or summary.get("role") or "Target Role"

    strengths = []
    missing_concepts = []
    mistakes = []
    improvement_tips = []

    for r in records:
        ev = r.get("evaluation") or {}
        if ev.get("status") in ("evaluated", "no_answer"):
            for s in ev.get("strengths") or []:
                if s and s not in strengths:
                    strengths.append(s)
            for m in ev.get("missing_concepts") or []:
                if m and m not in missing_concepts:
                    missing_concepts.append(m)
            for e in ev.get("mistakes") or []:
                if e and e not in mistakes:
                    mistakes.append(e)
            tip = ev.get("improvement_tip")
            if tip and tip not in improvement_tips:
                improvement_tips.append(tip)

    if overall_score is not None:
        if overall_score >= 80:
            comment = (
                f"Candidate demonstrated strong competence for the {role_name} role, "
                f"achieving an overall score of {overall_score}/100. Answers were structured, technically "
                f"sound, and showed good domain knowledge. Continuing to refine concise delivery will make "
                f"your interview performance outstanding."
            )
        elif overall_score >= 60:
            comment = (
                f"Candidate showed solid potential and good foundational understanding for {role_name} "
                f"with an overall score of {overall_score}/100. While core concepts were addressed, some "
                f"explanations would benefit from greater technical depth and specific real-world examples. "
                f"Targeted review of key concepts will help elevate your readiness."
            )
        elif overall_score >= 40:
            comment = (
                f"Candidate completed the practice interview for {role_name} scoring {overall_score}/100. "
                f"Fundamental awareness is present, but several technical criteria and detailed problem-solving steps "
                f"need reinforcement. Focus on structured answering frameworks and technical precision."
            )
        else:
            comment = (
                f"Candidate completed the practice interview for {role_name} with an overall score of {overall_score}/100. "
                f"Core concepts require additional preparation and hands-on practice. Consistent review of the "
                f"sample answers and reference materials below will build confidence and accuracy."
            )
    else:
        comment = (
            f"Candidate completed the practice interview for {role_name}. Review the topic-by-topic breakdowns "
            f"and recommendations below to guide your next study phase."
        )

    if not strengths:
        strengths = [
            f"Demonstrated clear enthusiasm and engagement throughout the {role_name} interview process.",
            "Communicated thoughts and reasoning in an organized manner.",
            "Approached scenario-based and technical questions with practical intent."
        ]

    low_topics = [t["topic"] for t in summary.get("topic_performance", []) if t.get("average_score", 100) < 75]
    improvement_areas = []
    if low_topics:
        improvement_areas.append(f"Strengthen understanding in key topics: {', '.join(low_topics[:3])}.")
    if missing_concepts:
        improvement_areas.extend([f"Deepen knowledge on: {c}" for c in missing_concepts[:4]])
    if not improvement_areas:
        improvement_areas = [
            "Incorporate more quantitative metrics and specific outcomes in your project narratives.",
            "Structure answers clearly with context, actions taken, and measurable results."
        ]

    common_mistakes = mistakes[:5]
    if not common_mistakes:
        if missing_concepts:
            common_mistakes = [f"Omitted key concept: {c}" for c in missing_concepts[:3]]
        else:
            common_mistakes = ["No major conceptual errors were identified across evaluated answers."]

    resume_gaps = []
    if resume_analysis:
        for g in resume_analysis.get("skill_gaps", [])[:5]:
            if isinstance(g, dict) and g.get("skill"):
                note = f" ({g['note']})" if g.get("note") else ""
                resume_gaps.append(f"{g['skill']}{note}")
            elif isinstance(g, str):
                resume_gaps.append(g)
    if not resume_gaps:
        resume_gaps = [f"Align resume project bullet points more closely with required skills for {role_name}."]

    study_recs = improvement_tips[:5]
    if not study_recs:
        study_recs = [
            f"Review standard architecture and best practices commonly tested for {role_name}.",
            "Practice explaining technical decisions aloud within 60-90 second time frames.",
            "Consult reference answers provided in each question card to benchmark depth."
        ]

    improvement_plan = [
        "Week 1: Core Fundamentals Review - Study key missing concepts and revisit basic definitions.",
        f"Week 2: Role-Specific Deep Dive - Practice scenario-based questions tailored to {role_name}.",
        "Week 3: Mock Interview Rehearsal - Record your answers aloud and refine delivery clarity and pacing.",
        "Week 4: Final Polish - Re-take this interview simulation to measure progress and achieve target scores."
    ]

    return {
        "overall_comment": comment,
        "technical_strengths": strengths[:6],
        "improvement_areas": improvement_areas[:6],
        "common_mistakes": common_mistakes[:6],
        "resume_skill_gaps": resume_gaps[:6],
        "study_recommendations": study_recs[:6],
        "improvement_plan": improvement_plan[:6],
    }

