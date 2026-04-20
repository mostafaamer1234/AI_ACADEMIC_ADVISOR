from __future__ import annotations

import json
import re
import time as _dbg_time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# #region debug log helper (debug session 97c192)
_DBG_LOG_PATH = "/Users/mostafa/newFianlLLMRepo/.cursor/debug-97c192.log"
def _dbg(hyp: str, location: str, message: str, data: Dict[str, Any]) -> None:
    try:
        with open(_DBG_LOG_PATH, "a") as _f:
            _f.write(json.dumps({
                "sessionId": "97c192",
                "runId": "r1",
                "hypothesisId": hyp,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(_dbg_time.time() * 1000),
            }) + "\n")
    except Exception:
        pass
# #endregion

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except Exception:  # pragma: no cover - fallback for reduced test environments
    @dataclass
    class SystemMessage:
        content: str

    @dataclass
    class HumanMessage:
        content: str


COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,}\s+\d{3}[A-Z]?\b", re.IGNORECASE)
SEMESTER_BLOCK_RE = re.compile(
    r"((?:[-*]\s*)?(?:\*\*)?Semester\s+\+?\d+:(?:\*\*)?\s*\n.*?(?:[-*]\s*)?(?:\*\*)?Total Credits:(?:\*\*)?\s*\d+)",
    flags=re.IGNORECASE | re.DOTALL,
)

VALID_DOMAINS = {
    "transcript",
    "course",
    "major",
    "policy",
    "general_academic_chat",
}

VALID_DELIVERABLES = {
    "direct_answer",
    "audit",
    "single_semester_plan",
    "multi_semester_plan",
    "full_graduation_plan",
    "course_lookup",
    "major_lookup",
    "policy_lookup",
}

VALID_PLAN_SCOPES = {
    "single_semester",
    "multi_semester",
    "full_graduation",
    "audit",
    "direct_answer",
}

COMMON_CORRECTIONS = {
    "sophmore": "sophomore",
    "freshamn": "freshman",
    "junor": "junior",
    "senoir": "senior",
    "managment": "management",
    "semster": "semester",
    "semseter": "semester",
    "converstation": "conversation",
    "gramttic": "grammar",
    "reasnong": "reasoning",
    "repsonse": "response",
    "frinedly": "friendly",
    "proffesional": "professional",
}


@dataclass
class OrchestrationPlan:
    original_question: str
    normalized_question: str
    corrected_question: str
    resolved_user_need: str
    domain: str
    deliverable: str
    plan_scope: str
    semester_horizon: int
    use_tools: bool
    allowed_tools: List[str] = field(default_factory=list)
    executor_path: str = "general_academic_chat"
    must_not_do: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    # Planner-driven retrieval contract.
    #
    # ``retrieval_plan`` is an ordered list of tool calls the executor MUST make
    # before synthesizing. Each step is a dict with keys ``tool``, ``query``,
    # ``reason``. The executor dispatches them via the ``EXECUTOR_TOOLS``
    # registry; nothing else decides which tools to call. ``synthesis_brief``
    # is the planner's natural-language instruction for what to do with the
    # retrieved bundle (e.g. "build a 15-credit first-semester schedule that
    # respects every course's enrollment prerequisites").
    retrieval_plan: List[Dict[str, str]] = field(default_factory=list)
    synthesis_brief: str = ""
    # Multi-major awareness. The orchestrator LLM is asked to extract every
    # major it sees in the question (e.g. "art and accounting double major"
    # → ``["Art", "Accounting"]``). When ``len(majors) > 1`` the canonical
    # retrieval plan fans out per major and the synthesizer is told to
    # cover requirements from EACH major. Empty/single-element list is the
    # normal single-major case.
    majors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutorResult:
    response: str
    tools_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response": self.response,
            "tools_used": list(self.tools_used),
            "metadata": dict(self.metadata),
            "notes": list(self.notes),
        }


def _safe_llm_content(llm: Any, messages: Sequence[Any]) -> str:
    if llm is None:
        return ""
    try:
        response = llm.invoke(list(messages))
        return str(getattr(response, "content", response) or "").strip()
    except Exception:
        return ""


def _extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    raw = raw_text.strip()
    if "```" in raw:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
    else:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
    try:
        loaded = json.loads(raw)
    except Exception:
        return None
    return loaded if isinstance(loaded, dict) else None


def _normalize_question_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _current_datetime_context() -> str:
    now = datetime.now()
    return now.strftime("Current local date/time: %A, %B %d, %Y at %I:%M %p.")


def _apply_common_corrections(text: str) -> str:
    corrected = _normalize_question_text(text)
    if not corrected:
        return corrected
    for wrong, right in COMMON_CORRECTIONS.items():
        corrected = re.sub(
            rf"\b{re.escape(wrong)}\b",
            right,
            corrected,
            flags=re.IGNORECASE,
        )
    corrected = re.sub(r"\bi\b", "I", corrected)
    if corrected:
        corrected = corrected[0].upper() + corrected[1:]
    if corrected and corrected[-1] not in ".!?":
        corrected += "?"
    return corrected


def _extract_requested_horizon(text: str) -> int:
    q = (text or "").lower()
    numeric = re.search(r"\b(?:next\s+)?(\d+)\s+semesters?\b", q)
    if numeric:
        try:
            return max(1, min(8, int(numeric.group(1))))
        except Exception:
            pass

    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
    }
    for word, count in words.items():
        if re.search(rf"\b(?:next\s+)?{word}\s+semesters?\b", q):
            return count

    if "next semester" in q or "upcoming semester" in q:
        return 1
    return 0


def _extract_semester_blocks(plan_text: str) -> List[str]:
    if not plan_text:
        return []
    matches = list(SEMESTER_BLOCK_RE.finditer(plan_text))
    blocks: List[str] = []
    for match in matches:
        block = match.group(1).strip()
        if block:
            blocks.append(block)
    return blocks


def _is_profile_recall_question(text: str) -> bool:
    lowered = (text or "").lower()
    recall_patterns = [
        r"\bwhat(?:'s| is)?\s+my\s+major\b",
        r"\bwhat(?:'s| is)?\s+my\s+year\b",
        r"\bwhat\s+year\s+and\s+major\s+am\s+i\b",
        r"\bwhat\s+major\s+and\s+year\s+am\s+i\b",
        r"\bremind\s+me\s+what\s+my\s+major\s+is\b",
        r"\bremind\s+me\s+what\s+year\s+i\s+am\b",
        r"\bwhich\s+major\s+am\s+i\b",
        r"\bwhat\s+did\s+i\s+say\s+my\s+major\s+was\b",
        r"\bwhat\s+did\s+i\s+say\s+my\s+year\s+was\b",
    ]
    return any(re.search(pattern, lowered) for pattern in recall_patterns)


def _find_recent_assistant_plan(conversation_history: Optional[List[dict]]) -> Optional[str]:
    if not conversation_history:
        return None
    for message in reversed(conversation_history):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content", "") or "")
        if len(_extract_semester_blocks(content)) >= 1:
            return content
    return None


def _is_policy_question(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        token in lowered
        for token in [
            "deadline",
            "late drop",
            "withdraw",
            "registration",
            "advisor",
            "advising",
            "lionpath",
            "policy",
            "entry to major",
        ]
    )


def _is_major_question(text: str) -> bool:
    lowered = (text or "").lower()
    explicit_major_tokens = [
            "major requirement",
            "major requirements",
            "what courses do i need for",
            "what classes do i need for",
            "prescribed courses",
            "option courses",
        ]
    if any(token in lowered for token in explicit_major_tokens):
        return True

    recommendation_triggers = [
        "courses for",
        "classes for",
        "courses to take",
        "classes to take",
        "recommend courses",
        "recommend classes",
        "recommended courses",
        "recommended classes",
        "first semester courses",
        "first semester classes",
        "first semester schedule",
        "first semester plan",
        "freshman semester",
        "freshman year",
        "what should i take",
        "what classes should i take",
        "what courses should i take",
        "suggest courses",
        "suggest classes",
    ]
    if any(trigger in lowered for trigger in recommendation_triggers):
        return True

    return False


def _looks_like_generic_research_question(text: str) -> bool:
    lowered = (text or "").lower().strip()
    if not lowered:
        return False
    if _looks_like_plan_request(lowered) or _is_profile_recall_question(lowered):
        return False
    advisory_markers = [
        "should i", "can you help me", "recommend", "plan for me", "what should i take",
        "give me a graduation plan", "audit my transcript",
    ]
    if any(marker in lowered for marker in advisory_markers):
        return False
    factual_starters = [
        "who", "what", "when", "where", "which", "how much", "how many", "find", "look up",
    ]
    if any(lowered.startswith(starter) for starter in factual_starters):
        return True
    return any(
        token in lowered
        for token in [
            "current", "currently", "latest", "today", "deadline", "exact date",
            "release date", "ceo", "president", "chancellor", "registrar page",
            "calendar page", "website", "office page", "unemployment rate",
        ]
    )


def _looks_like_plan_request(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        token in lowered
        for token in [
            "graduation plan",
            "semester-by-semester",
            "semester by semester",
            "full plan",
            "course plan",
            "plan for next semester",
            "next semester plan",
            "next 2 semesters",
            "next two semesters",
        ]
    )


# ---------------------------------------------------------------------------
# Advisor persona.
#
# Single source of truth for the agent's voice and behavior contract. Imported
# by the executor drafters in main_copy.py so the *draft* arrives with the
# right voice, and used in the Finalizer system prompt below so the polish
# layer applies the same contract end-to-end.
#
# The four traits encoded here are intentionally testable: any future change
# that drops a trait (encouragement, "Things to consider next", soft-ask, or
# semester-block preservation) trips PersonalityRegressionTests.
# ---------------------------------------------------------------------------

ADVISOR_PERSONA = (
    "You are an academic advisor for Penn State Abington students. "
    "Voice: professional, warm, encouraging, and genuinely curious about the student's goals. "
    "Behavior contract:\n"
    "- Help maximally: always include related details the student didn't ask for but would benefit from "
    "(prerequisite reminders, gen-ed tags, registration windows, advising office hours, related courses, "
    "scholarships/programs that fit). Add these as a short 'Things to consider next' section after the "
    "main answer (3-5 bullets, only items grounded in the draft or retrieved context).\n"
    "- Encourage and motivate: acknowledge what the student is working toward and reinforce realistic "
    "next steps. Avoid hedging that sounds discouraging.\n"
    "- Soft-ask when info is missing: when a critical piece (transcript, major, year, course code) would "
    "meaningfully change the answer, still give the best response possible with what you have, then end "
    "with ONE specific clarifying question. Skip the question if the same one was already asked recently.\n"
    "- Stay grounded: only mention specific course codes, deadlines, or policy text that appear in the "
    "draft or the retrieved context. Never invent.\n"
    "- Keep machine-readable blocks intact: any 'Semester N: ... Total Credits: N' block must be quoted "
    "verbatim and never wrapped, split, or paraphrased.\n"
)


# ---------------------------------------------------------------------------
# Structural routing helpers (domain-driven, not keyword-driven).
#
# These are the single source of truth for how a *domain* maps to an executor
# *path* and a *tool set*. The override pipeline uses them to repair the
# orchestrator LLM's plan without discarding its domain-level decision, and
# the finalizer uses _response_claims_specificity to catch hallucinated
# course codes on responses that ran without any retrieval.
# ---------------------------------------------------------------------------

# domain -> {deliverable (or "*") -> executor_path}
_CANONICAL_PATH_BY_DOMAIN: Dict[str, Dict[str, str]] = {
    "major": {"*": "major_lookup"},
    "course": {"*": "course_lookup"},
    "policy": {"*": "policy_lookup"},
    "transcript": {
        "audit": "transcript_audit",
        "*": "deterministic_plan",
    },
    "general_academic_chat": {"*": "general_academic_chat"},
}

# executor_path -> canonical tool list (in preference order)
_TOOLS_BY_PATH: Dict[str, List[str]] = {
    "major_lookup": [
        "majors_retriever",
        "course_retriever",
        "prerequisite_graph",
        "deterministic_planner",
        "transcript_state",
        "web_search",
    ],
    "course_lookup": ["prerequisite_graph", "course_retriever", "web_search"],
    "policy_lookup": ["policy_retriever", "web_search"],
    "deterministic_plan": ["transcript_state", "deterministic_planner", "conversation_history"],
    "transcript_audit": ["transcript_state", "transcript_retriever"],
    "research_lookup": ["conversation_history", "web_search"],
    "conversation_plan_slice": [],
    "conversation_profile_recall": [],
    "general_academic_chat": [
        "conversation_history",
        "course_retriever",
        "policy_retriever",
        "majors_retriever",
        "web_search",
    ],
}

_GROUNDED_DOMAINS = {"course", "major", "policy", "transcript"}
_GROUNDED_DELIVERABLES = {
    "course_lookup",
    "major_lookup",
    "policy_lookup",
    "audit",
    "single_semester_plan",
    "multi_semester_plan",
    "full_graduation_plan",
}
_CONVERSATION_ONLY_PATHS = {"conversation_plan_slice", "conversation_profile_recall"}


def _canonical_path_for_domain(domain: str, deliverable: str) -> str:
    mapping = _CANONICAL_PATH_BY_DOMAIN.get(domain or "", {})
    return mapping.get(deliverable or "", mapping.get("*", "general_academic_chat"))


def _canonical_tools_for_path(path: str, available_tools: Sequence[str]) -> List[str]:
    return _filter_allowed_tools(_TOOLS_BY_PATH.get(path, []), available_tools, use_tools=True)


def _requires_grounded_answer(domain: str, deliverable: str) -> bool:
    return (domain or "") in _GROUNDED_DOMAINS or (deliverable or "") in _GROUNDED_DELIVERABLES


def _llm_plan_is_coherent(plan: "OrchestrationPlan") -> bool:
    """Whether the orchestrator LLM produced a self-consistent grounded plan.

    The override layer's Phase 2 keyword rewriting was originally intended to
    *fix* hallucinated/empty LLM plans (e.g. ``domain=""``), but its keyword
    matching also fires on perfectly coherent plans and downgrades them
    (Trace 1: a coherent ``major + major_lookup + 5-step retrieval_plan`` got
    rewritten to ``transcript + deterministic_plan`` because the LLM's
    ``corrected_question`` happened to contain the words ``"course plan"``).

    A plan is "coherent" when *all* of these structural properties hold:

      1. ``domain`` is one of the four grounded values.
      2. ``executor_path`` is the canonical path for that domain.
      3. ``use_tools`` is true.
      4. ``retrieval_plan`` is non-empty (the LLM committed to a retrieval
         strategy for this question).

    When all four hold, the LLM has reasoned through the question end to
    end. Phase 2's keyword-driven domain rewrite must NOT touch it.
    """
    if plan.domain not in {"course", "major", "policy", "transcript"}:
        return False
    if not _path_is_compatible_with_domain(plan.executor_path, plan.domain):
        return False
    if not plan.use_tools:
        return False
    if not getattr(plan, "retrieval_plan", None):
        return False
    return True


def _path_is_compatible_with_domain(path: str, domain: str) -> bool:
    """Whether executor_path can legitimately serve this domain."""
    if not path or not domain:
        return True
    if path in _CONVERSATION_ONLY_PATHS:
        # slice/recall may be set for any domain in phase 1; other phases override.
        return False
    canonical = _canonical_path_for_domain(domain, "*")
    if path == canonical:
        return True
    if domain == "general_academic_chat" and path in {"research_lookup", "general_academic_chat"}:
        return True
    if domain == "transcript" and path in {"deterministic_plan", "transcript_audit"}:
        return True
    return False


def _response_claims_specificity(text: str) -> bool:
    """Detect unsourced specific claims (PSU course codes) that suggest hallucination
    when the executor reports zero tools used."""
    if not text:
        return False
    return bool(COURSE_CODE_RE.search(text))


# Deliverables where the executor will recommend or schedule specific courses.
# The orchestrator's retrieval_plan MUST include a prerequisite_graph step for
# these so the synthesizer can refuse to schedule a course whose prereqs aren't
# satisfied. This is the structural property that prevents bugs like
# "CMPSC 131 and CMPSC 132 in the same semester".
_COURSE_RECOMMENDATION_DELIVERABLES = {
    "single_semester_plan",
    "multi_semester_plan",
    "full_graduation_plan",
    "major_lookup",
}


def _is_course_recommendation_deliverable(deliverable: str, executor_path: str) -> bool:
    if (deliverable or "") in _COURSE_RECOMMENDATION_DELIVERABLES:
        return True
    if (executor_path or "") == "major_lookup":
        return True
    return False


def _canonical_retrieval_plan(
    executor_path: str,
    corrected_question: str,
    available_tools: Sequence[str],
    majors: Optional[Sequence[str]] = None,
) -> List[Dict[str, str]]:
    """The minimum retrieval plan the executor needs to safely serve a path.

    This is the "planner can be wrong" floor for the retrieval contract,
    mirroring ``_TOOLS_BY_PATH`` for ``allowed_tools``. The query strings are
    intentionally generic and re-use the user's question — the planner's own
    queries are usually richer and the override layer prefers them when
    present.

    When ``majors`` has more than one entry (double / triple major), the
    canonical major_lookup plan fans out one ``majors_retriever``,
    ``course_retriever``, and ``prerequisite_graph`` step per major so each
    catalog actually gets hit. Without fan-out a single combined query like
    ``"art and accounting double major"`` retrieves only the Art catalog
    and the Accounting requirements never reach the synthesizer.
    """
    available_set = {str(t) for t in available_tools}
    question = (corrected_question or "").strip() or "Penn State Abington"
    major_list = [m for m in (majors or []) if m and m.strip()]

    def step(tool: str, query: str, reason: str) -> Optional[Dict[str, str]]:
        if tool not in available_set:
            return None
        return {"tool": tool, "query": query, "reason": reason}

    candidates: List[Optional[Dict[str, str]]] = []
    if executor_path == "major_lookup":
        if len(major_list) > 1:
            # Fan out per major so each catalog is actually retrieved.
            for m in major_list:
                candidates.append(step(
                    "majors_retriever",
                    f"{m} major requirements at Penn State Abington",
                    f"Pull the official {m} catalog so the plan covers its required courses.",
                ))
            for m in major_list:
                candidates.append(step(
                    "course_retriever",
                    f"{m} foundational and required courses",
                    f"Get titles, credits, and descriptions for {m} courses.",
                ))
            for m in major_list:
                candidates.append(step(
                    "prerequisite_graph",
                    f"{m} prerequisite chains",
                    f"Get {m} enrollment and concurrent prereqs so scheduling never violates them.",
                ))
        else:
            candidates = [
                step(
                    "majors_retriever",
                    question,
                    "Get the official major requirements and the prescribed course sequence.",
                ),
                step(
                    "course_retriever",
                    question,
                    "Get titles, credit values, and descriptions for candidate courses.",
                ),
                step(
                    "prerequisite_graph",
                    question,
                    "Get enrollment and concurrent prereqs so the synthesis avoids scheduling violations.",
                ),
            ]
    elif executor_path == "course_lookup":
        candidates = [
            step("course_retriever", question, "Get the course title, credits, and description."),
            step("prerequisite_graph", question, "Get the course's prerequisites for the answer."),
        ]
    elif executor_path == "policy_lookup":
        candidates = [
            step("policy_retriever", question, "Get the official policy text."),
        ]
    elif executor_path == "research_lookup":
        candidates = [
            step("web_search", question, "Look up the answer from authoritative web sources."),
        ]
    return [c for c in candidates if c is not None]


def _ensure_retrieval_plan_has_tool(
    plan_steps: List[Dict[str, str]],
    tool: str,
    fallback_step: Dict[str, str],
) -> List[Dict[str, str]]:
    """Append ``fallback_step`` if no existing step targets ``tool``."""
    for step in plan_steps:
        if str(step.get("tool", "")).strip() == tool:
            return plan_steps
    return list(plan_steps) + [fallback_step]


def _ensure_per_major_catalog_steps(
    plan_steps: List[Dict[str, str]],
    majors: Sequence[str],
    available_tools: Sequence[str],
) -> List[Dict[str, str]]:
    """Guarantee one *clean, single-major* majors_retriever step per major.

    Why this exists: runtime logs (debug session 97c192) showed the planner
    emitting compound multi-major queries like
    ``"Computer Science B.S. ... and Psychology B.A./B.S. ..."`` as its
    first majors_retriever step. Chroma's vector search returned a single
    nearest-neighbor document (PSS) for *both* that compound query and a
    Psychology-only follow-up, so Computer Science catalog data was never
    retrieved and the synthesizer built a psychology-shaped plan.

    The fix is a "floor" just like :func:`_ensure_retrieval_plan_has_tool`:
    for every major in ``majors`` that is NOT already covered by a clean
    single-major query, prepend one. A query counts as "clean single-major"
    if the major's name appears in it and no other major's name does. We
    preserve the planner's original steps (they may be richer); we just
    guarantee each major will at least get a focused catalog hit so the
    bundle can't end up lopsided.
    """
    if "majors_retriever" not in set(available_tools or []):
        return plan_steps
    clean_majors = [m for m in (majors or []) if m and m.strip()]
    if len(clean_majors) <= 1:
        return plan_steps

    lowered_others_by_major: Dict[str, List[str]] = {
        m: [o.strip().lower() for o in clean_majors if o != m]
        for m in clean_majors
    }
    already_covered: set = set()
    for step in plan_steps:
        if str(step.get("tool", "")).strip() != "majors_retriever":
            continue
        q = str(step.get("query", "")).lower()
        for m in clean_majors:
            if m in already_covered:
                continue
            if m.strip().lower() not in q:
                continue
            others = lowered_others_by_major[m]
            if any(o in q for o in others):
                continue  # compound query, not a clean per-major hit
            already_covered.add(m)

    missing = [m for m in clean_majors if m not in already_covered]
    if not missing:
        return plan_steps

    prepend: List[Dict[str, str]] = [
        {
            "tool": "majors_retriever",
            "query": f"{m} major requirements at Penn State Abington",
            "reason": (
                f"Guaranteed single-major catalog retrieval for {m} "
                "(planner's own query did not isolate this major, which "
                "caused the double-major bundle to collapse onto one catalog)."
            ),
        }
        for m in missing
    ]
    return prepend + list(plan_steps)


def _build_success_criteria(plan_scope: str, semester_horizon: int, question: str) -> List[str]:
    criteria: List[str] = []
    if plan_scope == "single_semester":
        criteria.append("Return exactly one semester block.")
    elif plan_scope == "multi_semester":
        criteria.append(f"Return exactly {max(1, semester_horizon)} semester blocks.")
    elif plan_scope == "full_graduation":
        criteria.append("Return the full remaining graduation plan.")
    elif plan_scope == "audit":
        criteria.append("Summarize the student record and unmet requirements clearly.")
    else:
        criteria.append("Answer the user's request directly and accurately.")

    lowered = (question or "").lower()
    if "next semester" in lowered or "only" in lowered:
        criteria.append("Do not include additional semesters beyond the requested scope.")
    if "deadline" in lowered or "date" in lowered:
        criteria.append("Include a concrete date if one is required to answer.")
    return criteria


def _build_must_not_do(plan_scope: str, semester_horizon: int, use_tools: bool) -> List[str]:
    must_not_do: List[str] = []
    if plan_scope == "single_semester":
        must_not_do.append("Do not return more than one semester.")
    elif plan_scope == "multi_semester":
        must_not_do.append(f"Do not return more than {max(1, semester_horizon)} semesters.")
    if not use_tools:
        must_not_do.append("Do not call any tools for this response.")
    must_not_do.append("Do not ignore the conversation history when interpreting the request.")
    return must_not_do


def _filter_allowed_tools(allowed_tools: Sequence[str], available_tools: Sequence[str], use_tools: bool) -> List[str]:
    if not use_tools:
        return []
    available = list(dict.fromkeys(str(tool) for tool in available_tools))
    allowed = [str(tool) for tool in allowed_tools if str(tool) in available]
    return list(dict.fromkeys(allowed))


def _fallback_plan(
    user_question: str,
    conversation_history: Optional[List[dict]],
    has_transcript: bool,
    available_tools: Sequence[str],
) -> OrchestrationPlan:
    normalized_question = _normalize_question_text(user_question)
    corrected_question = _apply_common_corrections(normalized_question)
    lowered = corrected_question.lower()
    recent_plan = _find_recent_assistant_plan(conversation_history)
    requested_horizon = _extract_requested_horizon(corrected_question)

    if _is_profile_recall_question(corrected_question):
        use_tools = False
        return OrchestrationPlan(
            original_question=user_question,
            normalized_question=normalized_question,
            corrected_question=corrected_question,
            resolved_user_need="Recall the student's year and/or major from earlier conversation context.",
            domain="general_academic_chat",
            deliverable="direct_answer",
            plan_scope="direct_answer",
            semester_horizon=0,
            use_tools=use_tools,
            allowed_tools=[],
            executor_path="conversation_profile_recall",
            must_not_do=_build_must_not_do("direct_answer", 0, use_tools),
            success_criteria=_build_success_criteria("direct_answer", 0, corrected_question),
        )

    if recent_plan and requested_horizon > 0:
        plan_scope = "single_semester" if requested_horizon == 1 else "multi_semester"
        deliverable = "single_semester_plan" if requested_horizon == 1 else "multi_semester_plan"
        allowed_tools: List[str] = []
        use_tools = False
        executor_path = "conversation_plan_slice"
        resolved_need = f"Narrow the previously generated plan to the next {requested_horizon} semester(s)."
        return OrchestrationPlan(
            original_question=user_question,
            normalized_question=normalized_question,
            corrected_question=corrected_question,
            resolved_user_need=resolved_need,
            domain="transcript",
            deliverable=deliverable,
            plan_scope=plan_scope,
            semester_horizon=requested_horizon,
            use_tools=use_tools,
            allowed_tools=allowed_tools,
            executor_path=executor_path,
            must_not_do=_build_must_not_do(plan_scope, requested_horizon, use_tools),
            success_criteria=_build_success_criteria(plan_scope, requested_horizon, corrected_question),
        )

    if _looks_like_plan_request(corrected_question) or requested_horizon > 0:
        if requested_horizon <= 0:
            requested_horizon = 8
        if requested_horizon == 1:
            plan_scope = "single_semester"
            deliverable = "single_semester_plan"
        elif requested_horizon < 8:
            plan_scope = "multi_semester"
            deliverable = "multi_semester_plan"
        else:
            plan_scope = "full_graduation"
            deliverable = "full_graduation_plan"
        use_tools = True
        allowed_tools = _filter_allowed_tools(
            ["transcript_state", "deterministic_planner", "conversation_history"],
            available_tools,
            use_tools=True,
        )
        resolved_need = "Generate a semester plan that matches the requested planning horizon."
        return OrchestrationPlan(
            original_question=user_question,
            normalized_question=normalized_question,
            corrected_question=corrected_question,
            resolved_user_need=resolved_need,
            domain="transcript",
            deliverable=deliverable,
            plan_scope=plan_scope,
            semester_horizon=requested_horizon,
            use_tools=use_tools,
            allowed_tools=allowed_tools,
            executor_path="deterministic_plan",
            must_not_do=_build_must_not_do(plan_scope, requested_horizon, use_tools),
            success_criteria=_build_success_criteria(plan_scope, requested_horizon, corrected_question),
        )

    if "audit" in lowered or "review my transcript" in lowered or "where do i stand" in lowered:
        use_tools = True
        return OrchestrationPlan(
            original_question=user_question,
            normalized_question=normalized_question,
            corrected_question=corrected_question,
            resolved_user_need="Summarize transcript progress and unmet requirements.",
            domain="transcript",
            deliverable="audit",
            plan_scope="audit",
            semester_horizon=0,
            use_tools=use_tools,
            allowed_tools=_filter_allowed_tools(
                ["transcript_state", "transcript_retriever"],
                available_tools,
                use_tools=True,
            ),
            executor_path="transcript_audit",
            must_not_do=_build_must_not_do("audit", 0, use_tools),
            success_criteria=_build_success_criteria("audit", 0, corrected_question),
        )

    if COURSE_CODE_RE.search(corrected_question):
        use_tools = True
        return OrchestrationPlan(
            original_question=user_question,
            normalized_question=normalized_question,
            corrected_question=corrected_question,
            resolved_user_need="Answer a specific course lookup question.",
            domain="course",
            deliverable="course_lookup",
            plan_scope="direct_answer",
            semester_horizon=0,
            use_tools=use_tools,
            allowed_tools=_filter_allowed_tools(
                ["prerequisite_graph", "course_retriever", "web_search"],
                available_tools,
                use_tools=True,
            ),
            executor_path="course_lookup",
            must_not_do=_build_must_not_do("direct_answer", 0, use_tools),
            success_criteria=_build_success_criteria("direct_answer", 0, corrected_question),
        )

    if _is_policy_question(corrected_question):
        use_tools = True
        return OrchestrationPlan(
            original_question=user_question,
            normalized_question=normalized_question,
            corrected_question=corrected_question,
            resolved_user_need="Answer an academic policy or deadline question.",
            domain="policy",
            deliverable="policy_lookup",
            plan_scope="direct_answer",
            semester_horizon=0,
            use_tools=use_tools,
            allowed_tools=_filter_allowed_tools(
                ["policy_retriever", "web_search"],
                available_tools,
                use_tools=True,
            ),
            executor_path="policy_lookup",
            must_not_do=_build_must_not_do("direct_answer", 0, use_tools),
            success_criteria=_build_success_criteria("direct_answer", 0, corrected_question),
        )

    if _is_major_question(corrected_question):
        use_tools = True
        return OrchestrationPlan(
            original_question=user_question,
            normalized_question=normalized_question,
            corrected_question=corrected_question,
            resolved_user_need="Answer a major requirements question.",
            domain="major",
            deliverable="major_lookup",
            plan_scope="direct_answer",
            semester_horizon=0,
            use_tools=use_tools,
            allowed_tools=_filter_allowed_tools(
                ["majors_retriever", "web_search"],
                available_tools,
                use_tools=True,
            ),
            executor_path="major_lookup",
            must_not_do=_build_must_not_do("direct_answer", 0, use_tools),
            success_criteria=_build_success_criteria("direct_answer", 0, corrected_question),
        )

    if _looks_like_generic_research_question(corrected_question):
        use_tools = True
        return OrchestrationPlan(
            original_question=user_question,
            normalized_question=normalized_question,
            corrected_question=corrected_question,
            resolved_user_need="Research and answer the user's factual question with cited evidence.",
            domain="general_academic_chat",
            deliverable="direct_answer",
            plan_scope="direct_answer",
            semester_horizon=0,
            use_tools=use_tools,
            allowed_tools=_filter_allowed_tools(
                ["conversation_history", "web_search"],
                available_tools,
                use_tools=True,
            ),
            executor_path="research_lookup",
            must_not_do=_build_must_not_do("direct_answer", 0, use_tools),
            success_criteria=_build_success_criteria("direct_answer", 0, corrected_question),
        )

    use_tools = False
    return OrchestrationPlan(
        original_question=user_question,
        normalized_question=normalized_question,
        corrected_question=corrected_question,
        resolved_user_need="Provide a direct academic advising response.",
        domain="general_academic_chat",
        deliverable="direct_answer",
        plan_scope="direct_answer",
        semester_horizon=0,
        use_tools=use_tools,
        allowed_tools=[],
        executor_path="general_academic_chat",
        must_not_do=_build_must_not_do("direct_answer", 0, use_tools),
        success_criteria=_build_success_criteria("direct_answer", 0, corrected_question),
    )


_PLAN_SCOPE_DEFAULT_HORIZON: Dict[str, int] = {
    "full_graduation": 8,
    "multi_semester": 4,
    "single_semester": 1,
    "direct_answer": 0,
    "audit": 0,
}


def _coerce_semester_horizon(raw: Any, plan_scope: str, fallback: int) -> int:
    """Tolerantly parse the planner's ``semester_horizon`` value.

    The planner LLM is allowed to express the horizon in any natural shape it
    likes — ``8``, ``"8"``, ``"8 semesters"``, ``"8-semester"``, ``"next 4"``,
    ``"all_remaining_semesters"``, ``"until graduation"``, ``"full graduation"``
    — and we still arrive at a coherent integer in [0, 8].

    Resolution order:
      1. If ``raw`` is an int (or stringified int) in range, use it.
      2. If ``raw`` is a string with an embedded number 1-8, use that.
      3. If ``raw`` is a recognized semantic phrase, map it.
      4. Otherwise, fall back to the scope's default
         (``_PLAN_SCOPE_DEFAULT_HORIZON``), then to the planner's fallback.

    The previous implementation just did ``int(raw)`` and any non-numeric
    string silently became ``0``. That is what produced
    ``plan_scope=multi_semester`` paired with ``semester_horizon=0``, which
    in turn produced the contradictory ``Do not return more than 1 semesters``
    must_not_do directive.
    """
    if isinstance(raw, bool):
        raw = int(raw)
    if isinstance(raw, int):
        return max(0, min(8, raw))
    if isinstance(raw, float):
        try:
            return max(0, min(8, int(raw)))
        except Exception:
            pass

    text = str(raw or "").strip().lower()
    if text:
        # First number in the string wins (handles "8 semesters", "8-semester",
        # "next 4", "the next 2 semesters", etc.).
        m = re.search(r"\b([1-8])\b", text)
        if m:
            try:
                return max(0, min(8, int(m.group(1))))
            except Exception:
                pass

        word_map = {
            "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8,
        }
        for word, count in word_map.items():
            if re.search(rf"\b{word}\b", text):
                return count

        # Semantic phrases.
        if any(
            phrase in text
            for phrase in (
                "all_remaining",
                "all remaining",
                "until graduation",
                "to graduation",
                "through graduation",
                "full graduation",
                "full plan",
                "entire plan",
                "graduation plan",
                "rest of degree",
                "remaining semesters",
            )
        ):
            return 8
        if any(
            phrase in text
            for phrase in ("next semester", "upcoming semester", "this semester")
        ):
            return 1

    # Plan-scope default beats the planner's prior fallback when the prior
    # fallback would contradict the scope.
    scope_default = _PLAN_SCOPE_DEFAULT_HORIZON.get(plan_scope)
    if scope_default is not None:
        return max(0, min(8, scope_default))
    return max(0, min(8, fallback))


def _reconcile_scope_and_horizon(plan_scope: str, semester_horizon: int) -> int:
    """Final invariant: scope and horizon must agree.

    If ``plan_scope`` is a multi-semester or full-graduation plan but the
    horizon is 0 (e.g. the LLM omitted it), raise it to the scope default so
    ``_build_must_not_do`` doesn't generate ``Do not return more than 1
    semesters`` for a plan that the planner said is multi-semester.
    """
    if plan_scope in {"multi_semester", "full_graduation"} and semester_horizon <= 0:
        return _PLAN_SCOPE_DEFAULT_HORIZON.get(plan_scope, 1)
    if plan_scope == "single_semester" and semester_horizon != 1:
        return 1
    if plan_scope in {"direct_answer", "audit"}:
        return 0
    return semester_horizon


def _sanitize_plan_dict(
    parsed: Dict[str, Any],
    fallback: OrchestrationPlan,
    available_tools: Sequence[str],
) -> OrchestrationPlan:
    normalized_question = _normalize_question_text(
        str(parsed.get("normalized_question") or fallback.normalized_question)
    )
    corrected_question = _apply_common_corrections(
        str(parsed.get("corrected_question") or fallback.corrected_question or normalized_question)
    )
    domain = str(parsed.get("domain") or fallback.domain).strip().lower()
    if domain not in VALID_DOMAINS:
        domain = fallback.domain

    deliverable = str(parsed.get("deliverable") or fallback.deliverable).strip().lower()
    if deliverable not in VALID_DELIVERABLES:
        deliverable = fallback.deliverable

    plan_scope = str(parsed.get("plan_scope") or fallback.plan_scope).strip().lower()
    if plan_scope not in VALID_PLAN_SCOPES:
        plan_scope = fallback.plan_scope

    semester_horizon = _coerce_semester_horizon(
        parsed.get("semester_horizon"),
        plan_scope,
        fallback.semester_horizon,
    )
    semester_horizon = _reconcile_scope_and_horizon(plan_scope, semester_horizon)

    use_tools = bool(parsed.get("use_tools", fallback.use_tools))
    allowed_tools = parsed.get("allowed_tools")
    if not isinstance(allowed_tools, list):
        allowed_tools = fallback.allowed_tools
    allowed_tools = _filter_allowed_tools(allowed_tools, available_tools, use_tools)

    executor_path = str(parsed.get("executor_path") or fallback.executor_path).strip() or fallback.executor_path

    must_not_do = parsed.get("must_not_do")
    if not isinstance(must_not_do, list) or not must_not_do:
        must_not_do = fallback.must_not_do

    success_criteria = parsed.get("success_criteria")
    if not isinstance(success_criteria, list) or not success_criteria:
        success_criteria = fallback.success_criteria

    resolved_user_need = str(parsed.get("resolved_user_need") or fallback.resolved_user_need).strip()
    if not resolved_user_need:
        resolved_user_need = fallback.resolved_user_need

    retrieval_plan = _sanitize_retrieval_plan(
        parsed.get("retrieval_plan"), available_tools
    )
    synthesis_brief = str(parsed.get("synthesis_brief") or fallback.synthesis_brief or "").strip()
    majors = _sanitize_majors_list(parsed.get("majors"), fallback.majors)

    # #region debug log (H-A1/H-A2: planner output)
    _dbg("H-A1+H-A2", "orchestration_pipeline.py:_parse_planner_json", "planner parsed", {
        "raw_majors_field_type": type(parsed.get("majors")).__name__,
        "raw_majors_field": parsed.get("majors"),
        "sanitized_majors": majors,
        "deliverable": deliverable,
        "plan_scope": plan_scope,
        "executor_path": executor_path,
        "domain": domain,
        "use_tools": use_tools,
        "allowed_tools": list(allowed_tools or []),
        "corrected_question": corrected_question,
        "resolved_user_need_first_300": (resolved_user_need or "")[:300],
        "retrieval_plan_steps": [
            {"tool": s.get("tool"), "query": (s.get("query") or "")[:120]}
            for s in (retrieval_plan or [])
        ],
    })
    # #endregion

    return OrchestrationPlan(
        original_question=fallback.original_question,
        normalized_question=normalized_question or fallback.normalized_question,
        corrected_question=corrected_question or fallback.corrected_question,
        resolved_user_need=resolved_user_need,
        domain=domain,
        deliverable=deliverable,
        plan_scope=plan_scope,
        semester_horizon=semester_horizon,
        use_tools=use_tools,
        allowed_tools=allowed_tools,
        executor_path=executor_path,
        must_not_do=[str(item) for item in must_not_do],
        success_criteria=[str(item) for item in success_criteria],
        retrieval_plan=retrieval_plan,
        synthesis_brief=synthesis_brief,
        majors=majors,
    )


def _sanitize_majors_list(raw: Any, fallback: List[str]) -> List[str]:
    """Coerce planner-emitted ``majors`` into a clean ``List[str]``.

    Accepts:
      - a list of strings: deduped, lowercased-stripped-titlecased
      - a single string: split on common conjunctions ("and", "&", ",", "/")
        so the planner can be lazy and emit ``"art and accounting"``
      - empty/None: fall back to ``fallback`` (typically empty)

    The returned list preserves first-seen order, removes empty entries, and
    title-cases every entry so downstream prompts can show them consistently.
    """
    items: List[str] = []
    if isinstance(raw, list):
        items = [str(item) for item in raw]
    elif isinstance(raw, str):
        text = raw.strip()
        if text:
            parts = re.split(r"\s*(?:,|/|&|\band\b|\bplus\b)\s*", text, flags=re.IGNORECASE)
            items = [p for p in parts if p]
    elif raw is not None:
        items = [str(raw)]

    cleaned: List[str] = []
    seen: set = set()
    for item in items:
        norm = " ".join(item.strip().split())
        if not norm:
            continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(norm.title())

    if not cleaned and fallback:
        return list(fallback)
    return cleaned


def _sanitize_retrieval_plan(
    raw: Any, available_tools: Sequence[str]
) -> List[Dict[str, str]]:
    """Validate and normalize a planner-emitted ``retrieval_plan``.

    Each step must be a dict-like with at least ``tool`` and ``query``. Tools
    not in the available inventory are dropped silently (the deterministic
    override layer will fill canonical steps back in if needed). Order is
    preserved; duplicates against the same tool are kept (the planner may
    legitimately want two different queries against the same retriever).
    """
    if not isinstance(raw, list):
        return []
    available_set = {str(t) for t in available_tools}
    sanitized: List[Dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "").strip()
        if not tool or (available_set and tool not in available_set):
            continue
        query = str(item.get("query") or "").strip()
        if not query:
            continue
        reason = str(item.get("reason") or "").strip()
        sanitized.append({"tool": tool, "query": query, "reason": reason})
    return sanitized


def _apply_deterministic_overrides(
    plan: OrchestrationPlan,
    conversation_history: Optional[List[dict]],
    available_tools: Sequence[str],
) -> OrchestrationPlan:
    """Repair the orchestrator plan without discarding the LLM's domain decision.

    Design contract (executed in phases):

      1. Protected follow-up paths: profile-recall and plan-slice are binary.
         If their preconditions fire, they take precedence and return immediately.

      2. Semantic domain repair: keyword signals are evidence, not authority.
         They can *upgrade* a generic classification to a grounded one, but
         they are NOT allowed to downgrade a grounded domain the LLM already
         picked correctly. This is the core fix for the class of bug where
         the LLM correctly inferred ``domain="major"`` but the override layer
         silently dropped it to ``general_academic_chat``.

      3. Executor-path repair: once the domain is final, align executor_path
         with it via the canonical mapping. If the LLM picked a path that
         cannot service the chosen domain, it is replaced.

      4. Tool guardrail for grounded domains: for any domain in
         {course, major, policy, transcript} we force ``use_tools=True`` and
         install the canonical toolset for the path. This closes the
         "LLM forgot to set use_tools=true" failure mode for *any* unexpected
         grounded-knowledge question — no keyword list required.

      5. Normalize must_not_do / success_criteria / final allowed_tools list.
    """
    corrected = _apply_common_corrections(plan.corrected_question or plan.normalized_question)
    requested_horizon = _extract_requested_horizon(corrected)
    recent_plan = _find_recent_assistant_plan(conversation_history)
    is_profile_recall = _is_profile_recall_question(corrected)
    has_course_code = bool(COURSE_CODE_RE.search(corrected))

    # ------------------------------------------------------------------ Phase 1
    # Protected follow-up paths. These are hard routing rules that beat the LLM.
    if is_profile_recall:
        plan.domain = "general_academic_chat"
        plan.deliverable = "direct_answer"
        plan.plan_scope = "direct_answer"
        plan.semester_horizon = 0
        plan.use_tools = False
        plan.allowed_tools = []
        plan.executor_path = "conversation_profile_recall"
        plan.resolved_user_need = "Recall the student's year and/or major from earlier conversation context."
        plan.corrected_question = corrected
        plan.normalized_question = _normalize_question_text(plan.normalized_question or plan.original_question)
        plan.must_not_do = _build_must_not_do(plan.plan_scope, plan.semester_horizon, plan.use_tools)
        plan.success_criteria = _build_success_criteria(plan.plan_scope, plan.semester_horizon, corrected)
        return plan

    if recent_plan and requested_horizon > 0:
        plan.domain = "transcript"
        plan.plan_scope = "single_semester" if requested_horizon == 1 else "multi_semester"
        plan.deliverable = "single_semester_plan" if requested_horizon == 1 else "multi_semester_plan"
        plan.semester_horizon = requested_horizon
        plan.use_tools = False
        plan.allowed_tools = []
        plan.executor_path = "conversation_plan_slice"
        plan.resolved_user_need = (
            f"Narrow the previously generated semester plan to the next {requested_horizon} semester(s)."
        )
        plan.corrected_question = corrected
        plan.normalized_question = _normalize_question_text(plan.normalized_question or plan.original_question)
        plan.must_not_do = _build_must_not_do(plan.plan_scope, plan.semester_horizon, plan.use_tools)
        plan.success_criteria = _build_success_criteria(plan.plan_scope, plan.semester_horizon, corrected)
        return plan

    # ------------------------------------------------------------------ Phase 2
    # Semantic domain repair. Only allow keyword signals to change the domain
    # when they carry strong positive evidence and they are *upgrading* a
    # weaker classification (not overriding a legitimate grounded domain).
    if has_course_code and plan.domain not in {"policy", "transcript"}:
        # An explicit course code is a hard signal, unless the LLM already
        # classified this as a policy/transcript question (e.g. "is CMPSC 221
        # a degree requirement" stays with whatever the LLM picked).
        plan.domain = "course"
        plan.deliverable = "course_lookup"
        plan.plan_scope = "direct_answer"
        plan.semester_horizon = 0
    elif _is_policy_question(corrected) and plan.domain in {"general_academic_chat", ""}:
        plan.domain = "policy"
        plan.deliverable = "policy_lookup"
        plan.plan_scope = "direct_answer"
        plan.semester_horizon = 0
    elif (
        (_looks_like_plan_request(corrected) or requested_horizon > 0)
        and not _llm_plan_is_coherent(plan)
    ):
        # Plan-request rewrite. Only fires when the LLM did NOT produce a
        # coherent grounded plan. A coherent ``major + major_lookup + 5-step
        # retrieval_plan`` (Trace 1) is left intact even if the question
        # contains the words "course plan", because the LLM has clearly
        # already routed it correctly. Keyword fallback only takes over when
        # the LLM's plan is empty/incoherent.
        plan.domain = "transcript"
        if requested_horizon == 1:
            plan.plan_scope = "single_semester"
            plan.deliverable = "single_semester_plan"
            plan.semester_horizon = 1
        elif requested_horizon > 1:
            plan.plan_scope = "multi_semester"
            plan.deliverable = "multi_semester_plan"
            plan.semester_horizon = requested_horizon
        else:
            plan.plan_scope = "full_graduation"
            plan.deliverable = "full_graduation_plan"
            plan.semester_horizon = max(plan.semester_horizon, 8)
    elif _is_major_question(corrected) and plan.domain in {"general_academic_chat", ""}:
        plan.domain = "major"
        plan.deliverable = "major_lookup"
        plan.plan_scope = "direct_answer"
        plan.semester_horizon = 0

    # ------------------------------------------------------------------ Phase 3
    # Align executor_path with the (now final) domain. This eliminates the
    # failure mode where domain=="major" but executor_path=="general_academic_chat".
    if not _path_is_compatible_with_domain(plan.executor_path, plan.domain):
        plan.executor_path = _canonical_path_for_domain(plan.domain, plan.deliverable)

    # Lookup-style deliverables are inherently direct answers, never multi-semester.
    # Normalize here so a bad plan_scope from the LLM (e.g. "single_semester" attached
    # to a policy_lookup) doesn't leak into the must_not_do / validation layers.
    if plan.deliverable in {"direct_answer", "course_lookup", "major_lookup", "policy_lookup"}:
        plan.plan_scope = "direct_answer"
        plan.semester_horizon = 0
    elif plan.deliverable == "audit":
        plan.plan_scope = "audit"
        plan.semester_horizon = 0

    # ------------------------------------------------------------------ Phase 4
    # Tool guardrail. Grounded domains MUST use retrievers — we never allow the
    # LLM's use_tools=false to win for a grounded domain.
    if _requires_grounded_answer(plan.domain, plan.deliverable):
        plan.use_tools = True
        canonical_tools = _canonical_tools_for_path(plan.executor_path, available_tools)
        existing = _filter_allowed_tools(plan.allowed_tools or [], available_tools, use_tools=True)
        # Preserve any LLM-chosen tools that belong to the canonical set, then
        # union in any missing canonical tools so we always have a safety net.
        keep = [t for t in existing if t in canonical_tools]
        for t in canonical_tools:
            if t not in keep:
                keep.append(t)
        plan.allowed_tools = keep
    elif plan.executor_path in _CONVERSATION_ONLY_PATHS:
        plan.use_tools = False
        plan.allowed_tools = []
    elif plan.executor_path == "general_academic_chat":
        # Generic conversational fall-through: reset scope to direct_answer so the
        # finalizer doesn't try to enforce a single_semester block count.
        plan.plan_scope = "direct_answer"
        plan.semester_horizon = 0
        if _looks_like_generic_research_question(corrected):
            plan.use_tools = True
            plan.allowed_tools = _filter_allowed_tools(
                ["conversation_history", "web_search"], available_tools, use_tools=True
            )
            plan.executor_path = "research_lookup"
    elif _looks_like_generic_research_question(corrected):
        plan.use_tools = True
        plan.allowed_tools = _filter_allowed_tools(
            ["conversation_history", "web_search"], available_tools, use_tools=True
        )
        plan.executor_path = "research_lookup"

    # ------------------------------------------------------------------ Phase 5
    if not plan.use_tools:
        plan.allowed_tools = []
    plan.corrected_question = corrected
    plan.normalized_question = _normalize_question_text(plan.normalized_question or plan.original_question)
    # Final scope/horizon reconciliation. Any earlier phase may have changed
    # ``plan_scope`` (e.g. a deliverable normalization) without updating
    # ``semester_horizon``. We re-check the invariant here so the must_not_do
    # / success_criteria builders below never see a contradictory pair.
    plan.semester_horizon = _reconcile_scope_and_horizon(plan.plan_scope, plan.semester_horizon)
    plan.must_not_do = _build_must_not_do(plan.plan_scope, plan.semester_horizon, plan.use_tools)
    plan.success_criteria = _build_success_criteria(plan.plan_scope, plan.semester_horizon, corrected)

    # ------------------------------------------------------------------ Phase 6
    # Retrieval plan repair. The planner LLM is allowed to be wrong/lazy: if it
    # didn't produce a retrieval_plan for a grounded path, install the canonical
    # one. For course-recommendation deliverables we additionally guarantee a
    # prerequisite_graph step is present — without prereq edges in the bundle,
    # the synthesizer cannot avoid scheduling violations.
    if plan.use_tools and plan.executor_path not in _CONVERSATION_ONLY_PATHS:
        # Drop any planner steps that target tools that didn't survive the
        # allowed_tools filter (e.g. degraded retrievers).
        if plan.retrieval_plan:
            allowed_set = set(plan.allowed_tools or [])
            plan.retrieval_plan = [
                step for step in plan.retrieval_plan
                if str(step.get("tool", "")).strip() in allowed_set
            ]
        if not plan.retrieval_plan:
            plan.retrieval_plan = _canonical_retrieval_plan(
                plan.executor_path, corrected, plan.allowed_tools,
                majors=plan.majors,
            )
        if _is_course_recommendation_deliverable(plan.deliverable, plan.executor_path):
            available_set = set(plan.allowed_tools or [])
            if "prerequisite_graph" in available_set:
                plan.retrieval_plan = _ensure_retrieval_plan_has_tool(
                    plan.retrieval_plan,
                    "prerequisite_graph",
                    {
                        "tool": "prerequisite_graph",
                        "query": corrected or "Penn State Abington course prerequisites",
                        "reason": (
                            "Course-recommendation deliverable: enrollment prereqs are required "
                            "to avoid scheduling violations like CMPSC 132 before CMPSC 131."
                        ),
                    },
                )
            # Multi-major floor: the planner is allowed to emit compound
            # queries, but for >1 majors we still guarantee per-major catalog
            # retrievals so the vector store can't collapse both majors onto
            # the same nearest-neighbor document.
            if plan.executor_path == "major_lookup" and len(plan.majors or []) > 1:
                plan.retrieval_plan = _ensure_per_major_catalog_steps(
                    plan.retrieval_plan, plan.majors, plan.allowed_tools,
                )
    elif plan.executor_path in _CONVERSATION_ONLY_PATHS:
        plan.retrieval_plan = []
    return plan


def build_orchestration_plan(
    user_question: str,
    conversation_history: Optional[List[dict]],
    has_transcript: bool,
    available_tools: Sequence[str],
    llm: Any = None,
) -> Tuple[OrchestrationPlan, Dict[str, Any]]:
    fallback = _fallback_plan(user_question, conversation_history, has_transcript, available_tools)
    raw_output = ""
    parsed: Optional[Dict[str, Any]] = None

    recent_messages = []
    for message in (conversation_history or [])[-6:]:
        role = str(message.get("role", "user"))
        content = str(message.get("content", "") or "")
        recent_messages.append(f"{role}: {content[:350]}")

    available_tools_list = ", ".join(dict.fromkeys(str(tool) for tool in available_tools))
    time_context = _current_datetime_context()
    system_prompt = (
        "You are the orchestrator for an academic advising assistant. "
        "Return ONLY valid JSON with these exact keys: "
        "normalized_question, corrected_question, resolved_user_need, domain, deliverable, "
        "plan_scope, semester_horizon, use_tools, allowed_tools, executor_path, must_not_do, "
        "success_criteria, retrieval_plan, synthesis_brief, majors.\n\n"
        "MAJORS EXTRACTION (you MUST do this):\n"
        "Whenever the question references one or more academic majors, populate the `majors` "
        "field with a clean list of every major name you can identify. Examples:\n"
        "  - \"computer science\"                       -> [\"Computer Science\"]\n"
        "  - \"art and accounting double major\"        -> [\"Art\", \"Accounting\"]\n"
        "  - \"psych / bio dual\"                       -> [\"Psychology\", \"Biology\"]\n"
        "  - \"undecided\" / \"any major\" / no major    -> []\n"
        "When `majors` has more than one entry, the executor will fan retrieval out per major and "
        "the synthesizer will be told to cover requirements from EACH major. Your retrieval_plan "
        "for a multi-major question SHOULD also include per-major retrievals (one majors_retriever "
        "step per major name) so each catalog actually gets hit.\n\n"
        "TOOL SELECTION RULES (you MUST follow these):\n"
        "1. Any question about specific Penn State courses, majors/degree requirements, academic "
        "policies/deadlines, or the student's transcript MUST set use_tools=true and include the "
        "appropriate retrievers in allowed_tools. These are grounded-knowledge domains: "
        "course, major, policy, transcript. Do NOT answer them from memory.\n"
        "2. For grounded domains, set executor_path to match the domain:\n"
        "   - major     -> major_lookup        (tools: majors_retriever, course_retriever, prerequisite_graph, web_search)\n"
        "   - course    -> course_lookup       (tools: course_retriever, prerequisite_graph, web_search)\n"
        "   - policy    -> policy_lookup       (tools: policy_retriever, web_search)\n"
        "   - transcript-> deterministic_plan  (tools: transcript_state, deterministic_planner, conversation_history)\n"
        "             or transcript_audit       when the deliverable is an audit.\n"
        "3. use_tools=false is ONLY appropriate for:\n"
        "   - profile recall (\"what's my major\", \"what year am I\") -> conversation_profile_recall\n"
        "   - narrowing a previously given plan                      -> conversation_plan_slice\n"
        "   - pure conversational threading where no Penn State-specific fact is claimed.\n"
        "4. For unexpected factual questions that need fresh/sourced information (not stored in our "
        "retrievers), use executor_path=research_lookup with web_search.\n"
        "5. NEVER return specific course codes, course titles, deadlines, or policy text in a downstream "
        "response unless you enabled a retriever for it. If uncertain, include web_search in allowed_tools "
        "as a safety net.\n\n"
        "RETRIEVAL PLAN (you MUST think in terms of 'what do I need to retrieve and why'):\n"
        "Data retrieval is itself a tool. The executor will not pick tools on its own — it executes the "
        "exact retrieval_plan you produce. Each step is an object with three keys:\n"
        "  - tool   : name from allowed_tools (e.g. \"majors_retriever\", \"course_retriever\", \"prerequisite_graph\", \"policy_retriever\", \"web_search\").\n"
        "  - query  : the natural-language query you want that tool to run.\n"
        "  - reason : one short sentence explaining why this retrieval is needed before synthesis.\n"
        "Order matters: list earlier the retrievals whose results inform later ones.\n"
        "MANDATORY: when the deliverable involves recommending or scheduling courses for the student "
        "(single_semester_plan, multi_semester_plan, full_graduation_plan, or any major_lookup that asks "
        "for specific courses), retrieval_plan MUST include a prerequisite_graph step covering every "
        "candidate course. Without prereq edges in the bundle, the executor cannot avoid scheduling "
        "violations like placing CMPSC 132 in the same semester as its prereq CMPSC 131.\n\n"
        "MANDATORY: when the deliverable involves recommending courses, your course_retriever queries "
        "MUST cover not only the prescribed/required courses but also concrete examples for every gen-ed "
        "or elective bucket the major contains (humanities/GH, social science/GS, natural science/GN, "
        "arts/GA, quantification/GQ, health and wellness/GHW, writing-speaking/GWS, US/IL cultures, "
        "foreign language, option/track electives, free electives). Issue separate course_retriever steps "
        "(one per bucket if needed) so the executor's bundle contains specific course codes for every "
        "category the synthesizer might need to fill — otherwise the synthesizer is forced to write bare "
        "'Elective' / 'Gen Ed' lines that the validator will reject.\n\n"
        "synthesis_brief is one short paragraph telling the executor's synthesizer what to do with the "
        "retrieved bundle (e.g. \"Build a 15-credit Semester 1 block for Computer Science using only "
        "courses present in majors_retriever and course_retriever; never schedule a course unless its "
        "enrollment prerequisites from prerequisite_graph are absent or already satisfied in earlier "
        "semesters of this response.\").\n\n"
        "Worked example (course recommendation, single semester):\n"
        "{\n"
        "  \"domain\": \"major\",\n"
        "  \"deliverable\": \"single_semester_plan\",\n"
        "  \"executor_path\": \"major_lookup\",\n"
        "  \"allowed_tools\": [\"majors_retriever\", \"course_retriever\", \"prerequisite_graph\"],\n"
        "  \"retrieval_plan\": [\n"
        "    {\"tool\": \"majors_retriever\", \"query\": \"Computer Science B.S. Abington prescribed and option courses\", \"reason\": \"Get the official major requirements and the prescribed sequence.\"},\n"
        "    {\"tool\": \"course_retriever\", \"query\": \"CMPSC 131 CMPSC 132 MATH 140 ENGL 15 first semester courses\", \"reason\": \"Get titles, credit values, and descriptions for candidate first-semester courses.\"},\n"
        "    {\"tool\": \"prerequisite_graph\", \"query\": \"CMPSC 131,CMPSC 132,MATH 140,MATH 141,PHYS 211,ENGL 15,CAS 100A\", \"reason\": \"Get enrollment and concurrent prereqs so the synthesis avoids scheduling violations.\"}\n"
        "  ],\n"
        "  \"synthesis_brief\": \"Build a 15-credit Semester 1 schedule for Computer Science. Only schedule a course whose enrollment prerequisites in prerequisite_graph are absent (this is semester 1, so no prereqs can be satisfied yet). Use exact course codes and titles from the retrieved bundle.\"\n"
        "}\n\n"
        "If the deliverable is direct_answer with no retrieval needed (e.g. profile recall, plan slice), "
        "return retrieval_plan as an empty list and synthesis_brief as a one-line directive.\n\n"
        "Enum values:\n"
        "- plan_scope: single_semester, multi_semester, full_graduation, audit, direct_answer.\n"
        "- domain: transcript, course, major, policy, general_academic_chat.\n"
        "- deliverable: direct_answer, audit, single_semester_plan, multi_semester_plan, "
        "full_graduation_plan, course_lookup, major_lookup, policy_lookup."
    )
    user_prompt = (
        f"{time_context}\n"
        f"User question: {user_question}\n"
        f"Has transcript uploaded: {has_transcript}\n"
        f"Available tools: {available_tools_list}\n"
        f"Recent conversation:\n" + ("\n".join(recent_messages) if recent_messages else "[none]")
    )

    raw_output = _safe_llm_content(
        llm,
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
    )
    parsed = _extract_json_object(raw_output)

    if parsed:
        plan = _sanitize_plan_dict(parsed, fallback, available_tools)
    else:
        plan = fallback

    plan = _apply_deterministic_overrides(plan, conversation_history, available_tools)

    metadata = {
        "raw_output": raw_output,
        "fallback_used": parsed is None,
        "recent_plan_available": _find_recent_assistant_plan(conversation_history) is not None,
        "available_tools": list(dict.fromkeys(str(tool) for tool in available_tools)),
    }
    return plan, metadata


def _parse_prereq_graph_text(text: str) -> Dict[str, Dict[str, List[str]]]:
    """Parse the multi-course output of ``get_course_prerequisites`` into a map.

    Expected input shape (per course, separated by blank lines):

        Prerequisites for CMPSC 132:
        - Enforced prerequisite at enrollment: CMPSC 121, CMPSC 131
        - Enforced concurrent at enrollment (can be taken together): MATH 140

    Output:
        {
          "CMPSC 132": {
            "enforced": ["CMPSC 121", "CMPSC 131"],
            "concurrent": ["MATH 140"],
          },
          ...
        }

    The validator uses this to check, for each course in semester N, that
    every "enforced" prereq either appears in some semester < N of the
    response OR appears in this course's "concurrent" list.
    """
    result: Dict[str, Dict[str, List[str]]] = {}
    if not text:
        return result
    current_course: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"Prerequisites for\s+([A-Z]{2,}\s+\d{3}[A-Z]?)\s*:?\s*$", line, re.IGNORECASE)
        if m:
            current_course = m.group(1).upper().strip()
            result.setdefault(current_course, {"enforced": [], "concurrent": []})
            continue
        if current_course is None:
            continue
        m_enf = re.match(
            r"-\s*Enforced prerequisite at enrollment\s*:\s*(.*)$", line, re.IGNORECASE
        )
        if m_enf:
            payload = m_enf.group(1).strip()
            if payload.lower() in {"none", ""}:
                result[current_course]["enforced"] = []
            else:
                codes = re.findall(r"\b[A-Z]{2,}\s+\d{3}[A-Z]?\b", payload, re.IGNORECASE)
                result[current_course]["enforced"] = [c.upper() for c in codes]
            continue
        m_con = re.match(
            r"-\s*Enforced concurrent at enrollment[^:]*:\s*(.*)$", line, re.IGNORECASE
        )
        if m_con:
            payload = m_con.group(1).strip()
            if payload.lower() in {"none", ""}:
                result[current_course]["concurrent"] = []
            else:
                codes = re.findall(r"\b[A-Z]{2,}\s+\d{3}[A-Z]?\b", payload, re.IGNORECASE)
                result[current_course]["concurrent"] = [c.upper() for c in codes]
    return result


# ---------------------------------------------------------------------------
# Universal specificity rule for course-recommendation responses.
#
# Rationale: a recommendation line that says only "Elective", "Gen Ed", or
# "Option Course" is not actionable — the student cannot register for it. The
# rule below is purely structural: inside any semester block, every line must
# be either a specific course code, OR a clearly-categorized placeholder
# (gen-ed tag or category word), OR an explicit advisor/catalog deferral.
# We do NOT hardcode majors, deliverables, or expected courses; we only check
# the *shape* of each recommendation slot. This catches bare-"Elective" bugs
# for any future major or gen-ed system without keyword routing logic.
# ---------------------------------------------------------------------------

# Words/phrases that indicate the line is describing a course slot (as
# opposed to commentary, headers, or credit totals). These trigger the
# specificity check.
_RECOMMENDATION_TRIGGER_RE = re.compile(
    r"\b(elective|gen\.?\s*ed|general\s+education|option\s+course|major\s+course|free\s+choice|requirement|category\s+slot|breadth\s+requirement)\b",
    re.IGNORECASE,
)

# Recognized academic-taxonomy descriptors. The presence of ANY of these on a
# slot line satisfies the specificity rule (the student knows what bucket to
# pick from). The list is the actual taxonomy used by the catalog, not a
# routing keyword list — it is deliberately broad so future buckets can be
# added without changing the rule itself.
_TAXONOMY_TAG_RE = re.compile(
    r"\b(GHW|GH|GS|GN|GA|GQ|GWS|US|IL|GFL|GHA)\b"
)
_TAXONOMY_WORD_RE = re.compile(
    r"\b(humanities|social\s+science|social\s+and\s+behavioral|natural\s+science|natural\s+sciences|"
    r"arts?|fine\s+arts|performing\s+arts|quantification|quantitative|"
    r"health\s+and\s+wellness|wellness|writing|speaking|writing\s*/\s*speaking|"
    r"cultural\s+(?:diversity|studies)|us\s+cultures?|international\s+cultures?|"
    r"foreign\s+language|world\s+language|"
    r"capstone|honors|seminar)\b",
    re.IGNORECASE,
)
# Explicit advisor/catalog deferral: an honest "I don't have eligible courses
# in the bundle, please check with X" is acceptable because it tells the
# student exactly what to do next.
_DEFERRAL_RE = re.compile(
    r"\b(advisor|registrar|catalog|gen[- ]ed\s+catalog|consult|please\s+confirm|please\s+check)\b",
    re.IGNORECASE,
)
# Lines we should skip — block headers and credit totals are not slots.
_SKIPPABLE_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?(?:semester\s+\+?\d+|total\s+credits?)\b",
    re.IGNORECASE,
)


def _is_recommendation_line(line: str) -> bool:
    """A line is a 'recommendation slot' if the model is using it to fill a
    course-load position — i.e. it has a course code, OR it uses one of the
    trigger words ('Elective', 'Gen Ed', 'Option Course', etc.). Pure prose
    lines (e.g. 'Talk to your advisor about minors.') are not slots and are
    not subject to the specificity rule."""
    if not line:
        return False
    if _SKIPPABLE_LINE_RE.match(line.strip()):
        return False
    if COURSE_CODE_RE.search(line):
        return True
    return bool(_RECOMMENDATION_TRIGGER_RE.search(line))


def _line_satisfies_specificity(line: str) -> bool:
    if not line:
        return True
    # A specific course code is the strongest form of specificity.
    if COURSE_CODE_RE.search(line):
        return True
    # Categorized placeholder: must carry at least one taxonomy tag/word.
    if _TAXONOMY_TAG_RE.search(line) or _TAXONOMY_WORD_RE.search(line):
        return True
    # Explicit deferral to an authoritative source.
    if _DEFERRAL_RE.search(line):
        return True
    return False


def _validate_recommendation_specificity(response_text: str) -> Optional[str]:
    """Universal under-specified-recommendation check.

    Walks every line inside every semester block. If a line is a
    recommendation slot (course code OR an "elective/gen ed/option" trigger)
    AND fails ``_line_satisfies_specificity``, it is reported as a violation.

    Returns the first violating line (with surrounding context), or None if
    the response is fully specific. Independent of major, deliverable, and
    gen-ed system.
    """
    if not response_text:
        return None
    blocks = _extract_semester_blocks(response_text)
    if not blocks:
        return None
    for block in blocks:
        sem_num = _semester_number_from_block(block)
        sem_label = f"Semester {sem_num}" if sem_num is not None else "Semester block"
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if _SKIPPABLE_LINE_RE.match(line):
                continue
            if not _is_recommendation_line(line):
                continue
            if _line_satisfies_specificity(line):
                continue
            return (
                f"{sem_label}: line '{line}' is a bare placeholder. "
                "Every recommendation must include a specific course code, "
                "a recognized category tag (e.g. GH, GA, GS, GN, GQ, GHW, GWS, "
                "humanities, social science, arts), or an explicit "
                "advisor/catalog deferral."
            )
    return None


def _semester_number_from_block(block: str) -> Optional[int]:
    m = re.search(r"Semester\s+\+?(\d+)", block, re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


def _course_codes_in_block(block: str) -> List[str]:
    return [c.upper() for c in COURSE_CODE_RE.findall(block)]


# A line carries an inline prereq annotation if it contains a parenthetical
# starting with "prerequisite" / "prerequisites" / "prereq" / "prereqs".
# The contents of the parenthetical can be a list, "None", "not retrieved",
# etc. — the rule is structural ("did the synthesizer surface prereq context
# at all?"), not semantic, so it stays universal across majors.
_INLINE_PREREQ_RE = re.compile(
    r"\(\s*prereq(?:uisite)?s?\s*:",
    re.IGNORECASE,
)


def _validate_inline_prereqs(response_text: str) -> Optional[str]:
    """Universal "every recommended course must show its prereqs inline" check.

    For every semester block, every line that contains a course code must
    also contain a ``(prerequisites: ...)`` annotation on the same line. The
    annotation can carry a list of codes, ``None``, or an honest "not
    retrieved" — what we are enforcing is that the synthesizer never
    recommends a course without surfacing the prereq context the student
    needs to act on it.

    Lines without a course code (gen-ed placeholders, deferral lines, prose,
    block headers, credit totals) are exempt because there is no specific
    course whose prereqs would even apply.

    Returns the first violating line (with surrounding context), or ``None``
    if the response is fully annotated. Independent of major, deliverable,
    and gen-ed system.
    """
    if not response_text:
        return None
    blocks = _extract_semester_blocks(response_text)
    if not blocks:
        return None
    for block in blocks:
        sem_num = _semester_number_from_block(block)
        sem_label = f"Semester {sem_num}" if sem_num is not None else "Semester block"
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or _SKIPPABLE_LINE_RE.match(line):
                continue
            code_match = COURSE_CODE_RE.search(line)
            if not code_match:
                continue
            # If the line is a categorized placeholder that lists example
            # codes (e.g. "GH Humanities elective — examples: HIST 020, ENGL 015"),
            # the course codes are *suggestions for the student to pick from*,
            # not the course being recommended for that slot. The slot itself
            # is a category, so per-course inline-prereqs do not apply.
            # Detect this structurally: a taxonomy tag/word IS present on the
            # same line. The specificity rule still governs such lines.
            if _TAXONOMY_TAG_RE.search(line) or _TAXONOMY_WORD_RE.search(line):
                continue
            if _INLINE_PREREQ_RE.search(line):
                continue
            return (
                f"{sem_label}: line '{line}' recommends {code_match.group(0).upper()} "
                "without an inline (prerequisites: ...) annotation. Every recommended "
                "course must state its prerequisites on the same line so the student "
                "can see what is required to enroll."
            )
    return None


def _validate_multi_major_coverage(
    response_text: str,
    majors: Sequence[str],
) -> Optional[str]:
    """Whether the response actually covers every major the planner identified.

    For a question like "art and accounting double major", ``plan.majors``
    will be ``["Art", "Accounting"]``. The shipped response MUST mention
    each of those names somewhere (case-insensitive). If any major is
    missing, we surface a ``missing_major_coverage`` retry reason so the
    synthesizer re-runs with the omission flagged.

    The check is purely structural: it does not try to decide whether the
    *coverage* is good (that's the synthesizer's job), only whether each
    major was named at all. A response that omits one of the requested
    majors is by definition incomplete for the deliverable.
    """
    text = (response_text or "").lower()
    if not text:
        # #region debug log (H-A3: validator early-exit)
        _dbg("H-A3", "orchestration_pipeline.py:_validate_multi_major_coverage", "early-exit empty text", {
            "majors": list(majors or []),
        })
        # #endregion
        return None
    missing = [m for m in (majors or []) if m and m.strip() and m.strip().lower() not in text]
    # #region debug log (H-A3: substring coverage check)
    _dbg("H-A3", "orchestration_pipeline.py:_validate_multi_major_coverage", "checked", {
        "plan_majors": list(majors or []),
        "missing": missing,
        "response_len": len(text),
    })
    # #endregion
    if not missing:
        return None
    requested = ", ".join(m for m in majors if m)
    missing_str = ", ".join(missing)
    return (
        f"The student asked about {len(list(majors))} majors ({requested}), but the "
        f"response does not mention {missing_str}. A double/multi-major plan must "
        "cover required courses from EACH major and label which courses come from "
        "which. Re-do the answer including the missing major(s)."
    )


def _validate_prerequisite_satisfaction(
    response_text: str,
    prereq_graph_text: str,
) -> Optional[str]:
    """Return a human-readable violation description, or None if the schedule is legal.

    A schedule is legal when, for every course C placed in Semester N of the
    response, every code in C's "Enforced prerequisite at enrollment" list
    either appears in some Semester < N of the response OR is in C's
    "Enforced concurrent at enrollment" list.
    """
    if not response_text or not prereq_graph_text:
        return None
    blocks = _extract_semester_blocks(response_text)
    if len(blocks) < 1:
        return None

    prereq_map = _parse_prereq_graph_text(prereq_graph_text)
    if not prereq_map:
        return None

    courses_per_semester: Dict[int, List[str]] = {}
    for block in blocks:
        n = _semester_number_from_block(block)
        if n is None:
            continue
        courses_per_semester.setdefault(n, []).extend(_course_codes_in_block(block))

    if not courses_per_semester:
        return None

    sorted_semesters = sorted(courses_per_semester.keys())
    earlier_codes_by_sem: Dict[int, set] = {}
    accumulated: set = set()
    for n in sorted_semesters:
        earlier_codes_by_sem[n] = set(accumulated)
        for code in courses_per_semester[n]:
            accumulated.add(code)

    for n in sorted_semesters:
        same_sem_codes = set(courses_per_semester[n])
        for course in courses_per_semester[n]:
            entry = prereq_map.get(course)
            if not entry:
                # No prereq info retrieved for this course; can't validate.
                continue
            enforced = [p.upper() for p in entry.get("enforced", [])]
            concurrent_allowed = {p.upper() for p in entry.get("concurrent", [])}
            for prereq in enforced:
                if prereq in earlier_codes_by_sem[n]:
                    continue
                if prereq in concurrent_allowed and prereq in same_sem_codes:
                    continue
                return (
                    f"{course} scheduled in Semester {n} but enrollment prerequisite "
                    f"{prereq} is not in an earlier semester"
                    + (
                        " (and is not listed as concurrent)"
                        if prereq not in concurrent_allowed
                        else ""
                    )
                )
    return None


def validate_final_response(
    plan: OrchestrationPlan,
    response_text: str,
    executor_result: ExecutorResult,
    validation_helpers: Optional[Dict[str, Callable[..., Any]]] = None,
) -> Dict[str, Any]:
    issues: List[str] = []
    response_text = (response_text or "").strip()
    semester_blocks = _extract_semester_blocks(response_text)

    if plan.plan_scope == "single_semester" and len(semester_blocks) != 1:
        issues.append("scope_mismatch_single_semester")
    elif plan.plan_scope == "multi_semester" and len(semester_blocks) != max(1, plan.semester_horizon):
        issues.append("scope_mismatch_multi_semester")
    elif plan.plan_scope == "full_graduation" and len(semester_blocks) < 2:
        issues.append("scope_mismatch_full_plan")

    used_tools = list(executor_result.tools_used)
    if not plan.use_tools and used_tools:
        issues.append("unexpected_tool_usage")
    if plan.allowed_tools:
        disallowed = [tool for tool in used_tools if tool not in plan.allowed_tools]
        if disallowed:
            issues.append("disallowed_tool_usage")

    # Semantic hallucination guard: if the response claims Penn State-specific
    # specificity (course codes) but the executor reported zero tools used, the
    # model almost certainly invented those codes from pre-training. Trigger a
    # retry so the executor can re-run with grounded retrieval.
    if (
        not used_tools
        and _response_claims_specificity(response_text)
        and plan.executor_path not in _CONVERSATION_ONLY_PATHS
    ):
        issues.append("ungrounded_specificity")

    # Deterministic prerequisite-satisfaction check. For any course-recommendation
    # deliverable, if the executor retrieved prereq edges and the response
    # schedules a course before its enrollment prereq, we surface a
    # ``prerequisite_violation`` so the executor re-plans with the offending
    # pair fed back through the retry path. This is what stops bugs like
    # "CMPSC 131 and CMPSC 132 in the same semester" from ever shipping.
    prereq_violation_detail: Optional[str] = None
    specificity_violation_detail: Optional[str] = None
    inline_prereq_violation_detail: Optional[str] = None
    multi_major_coverage_detail: Optional[str] = None

    # Universal multi-major coverage check. When the planner identified more
    # than one major in the question (e.g. ``["Art", "Accounting"]``), the
    # final response MUST mention each major by name. Otherwise we shipped a
    # single-major answer to a multi-major question (Trace 1).
    if plan.majors and len(plan.majors) > 1:
        multi_major_coverage_detail = _validate_multi_major_coverage(
            response_text, plan.majors
        )
        if multi_major_coverage_detail:
            issues.append("missing_major_coverage")

    if (
        _is_course_recommendation_deliverable(plan.deliverable, plan.executor_path)
        and semester_blocks
    ):
        prereq_text = ""
        if executor_result.metadata:
            prereq_text = str(executor_result.metadata.get("prereq_graph_text") or "")
        prereq_violation_detail = _validate_prerequisite_satisfaction(response_text, prereq_text)
        if prereq_violation_detail:
            issues.append("prerequisite_violation")

        # Universal specificity check. A response that says "Elective (3 credits)"
        # is structurally unactionable for the student regardless of major.
        # We surface this as ``under_specified_recommendation`` so the executor's
        # retry path forces the synthesizer to either name a real course code,
        # add a category tag (GH, GA, GWS, humanities, social science, etc.),
        # or explicitly defer to the advisor/catalog.
        specificity_violation_detail = _validate_recommendation_specificity(response_text)
        if specificity_violation_detail:
            issues.append("under_specified_recommendation")

        # Universal inline-prereq check. The student needs to see what each
        # recommended course requires to enroll. If a course code appears in
        # a semester block without a "(prerequisites: ...)" annotation on the
        # same line, surface ``missing_inline_prereqs`` so the synthesizer
        # re-runs and pulls that information out of the prerequisite_graph
        # bundle (or states it explicitly).
        inline_prereq_violation_detail = _validate_inline_prereqs(response_text)
        if inline_prereq_violation_detail:
            issues.append("missing_inline_prereqs")

    helpers = validation_helpers or {}
    needs_specific_date = helpers.get("needs_specific_date")
    has_concrete_date = helpers.get("has_concrete_date")
    is_time_sensitive_fact = helpers.get("is_time_sensitive_fact")
    extract_source_urls = helpers.get("extract_source_urls")
    indicates_no_answer = helpers.get("indicates_no_answer")

    question_for_validation = plan.corrected_question or plan.normalized_question
    search_resolution = executor_result.metadata.get("search_resolution", {}) if executor_result.metadata else {}

    if callable(needs_specific_date) and callable(has_concrete_date):
        try:
            if bool(needs_specific_date(question_for_validation)) and not bool(has_concrete_date(response_text)):
                issues.append("missing_concrete_date")
        except Exception:
            pass

    if callable(is_time_sensitive_fact) and callable(extract_source_urls):
        try:
            if bool(is_time_sensitive_fact(question_for_validation)) and not list(extract_source_urls(response_text)):
                issues.append("missing_source")
        except Exception:
            pass

    if search_resolution:
        source_urls = list(search_resolution.get("source_urls") or [])
        if source_urls and callable(extract_source_urls) and not list(extract_source_urls(response_text)):
            issues.append("missing_source")
        if callable(indicates_no_answer):
            try:
                if bool(indicates_no_answer(response_text)) and not bool(search_resolution.get("exhausted", False)):
                    issues.append("incomplete_search")
            except Exception:
                pass
        elif source_urls and "couldn" in response_text.lower() and not bool(search_resolution.get("exhausted", False)):
            issues.append("incomplete_search")

    retry_reason = issues[0] if issues else None
    result: Dict[str, Any] = {
        "passed": not issues,
        "issues": issues,
        "retry_reason": retry_reason,
        "semester_block_count": len(semester_blocks),
        "used_tools": used_tools,
    }
    if prereq_violation_detail:
        result["prereq_violation_detail"] = prereq_violation_detail
    if specificity_violation_detail:
        result["specificity_violation_detail"] = specificity_violation_detail
    if inline_prereq_violation_detail:
        result["inline_prereq_violation_detail"] = inline_prereq_violation_detail
    if multi_major_coverage_detail:
        result["multi_major_coverage_detail"] = multi_major_coverage_detail
    return result


def _append_follow_up_invitation(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return "What would you like to ask next?"
    lowered = stripped.lower()
    if "what would you like to ask next" in lowered or "what else would you like to ask" in lowered:
        return stripped
    if stripped[-1] not in ".!?":
        stripped += "."
    return stripped + " What would you like to ask next?"


def run_finalizer_stage(
    plan: OrchestrationPlan,
    executor_result: ExecutorResult,
    conversation_history: Optional[List[dict]],
    llm: Any,
    validation_helpers: Optional[Dict[str, Callable[..., Any]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    draft = (executor_result.response or "").strip()
    recent_messages = []
    for message in (conversation_history or [])[-4:]:
        role = str(message.get("role", "user"))
        content = str(message.get("content", "") or "")
        recent_messages.append(f"{role}: {content[:250]}")

    system_prompt = (
        ADVISOR_PERSONA
        + "\n\nYou are the final response layer. Rewrite the draft so it follows the persona contract "
        "above WITHOUT changing the answer scope or inventing new facts.\n\n"
        "Required response shape:\n"
        "1. A warm, encouraging answer to what the student asked, drawn entirely from the draft and the "
        "supporting context.\n"
        "2. A 'Things to consider next:' section with 3-5 short bullets covering related details the "
        "student would benefit from (prereqs, gen-ed tags, registration windows, related courses, advising "
        "office). Skip this section ONLY if there is genuinely nothing grounded to add (rare).\n"
        "3. A closing line: if a critical piece of info would meaningfully change the answer, end with ONE "
        "specific clarifying question; otherwise end with a brief forward-looking nudge. Skip the question "
        "if the same one was clearly asked in the recent conversation.\n\n"
        "Hard rules (do NOT violate):\n"
        "- Quote any 'Semester N: ... Total Credits: N' block in the draft VERBATIM. Do not wrap it in "
        "prose, do not split it, do not change the labels. Downstream parsing depends on this exact shape.\n"
        "- Do not introduce course codes, deadlines, or policy text that aren't in the draft or recent "
        "context.\n"
        "- Do not change the answer's scope (e.g., do not add semesters not in the draft).\n"
        "- INLINE PREREQUISITE RULE (semester blocks only): every Penn State course code that appears "
        "inside a 'Semester N:' block MUST be immediately followed by '(prerequisites: <list>)' or "
        "'(prerequisites: None)'. Pull the list from the draft. "
        "Outside of semester blocks (prose, 'Things to consider next', closing questions, asides), "
        "prerequisite annotations are optional: include them only when the prereq list is already known "
        "from the draft. If you do NOT know the prereqs for a course being mentioned in prose, simply "
        "use the bare course code — do NOT write '(prerequisites: not retrieved)' or any similar "
        "placeholder; these confuse students and pollute the reading flow.\n"
        "Examples:\n"
        "    (semester block) 'CMPSC 132 (prerequisites: CMPSC 131)'\n"
        "    (prose, known)   'Take CMPSC 132 (prerequisites: CMPSC 131) after finishing CMPSC 131.'\n"
        "    (prose, unknown) 'CMPSC 312 appears often as a prerequisite, so confirm it is completed.'\n\n"
        "Return ONLY valid JSON with keys: final_response, finalizer_notes."
    )
    user_prompt = (
        f"{_current_datetime_context()}\n"
        f"Original question: {plan.original_question}\n"
        f"Corrected question: {plan.corrected_question}\n"
        f"Resolved need: {plan.resolved_user_need}\n"
        f"Plan scope: {plan.plan_scope}\n"
        f"Success criteria: {plan.success_criteria}\n"
        f"Recent conversation:\n" + ("\n".join(recent_messages) if recent_messages else "[none]") + "\n\n"
        f"Draft response:\n{draft}"
    )

    raw_output = _safe_llm_content(
        llm,
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
    )
    parsed = _extract_json_object(raw_output)
    final_response = ""
    if parsed:
        final_response = str(parsed.get("final_response", "") or "").strip()
    if not final_response:
        final_response = _append_follow_up_invitation(draft)
    else:
        final_response = _append_follow_up_invitation(final_response)

    # Re-verify any '(prerequisites: ...)' annotation the FINALIZER added
    # (e.g. in 'Things to consider next' bullets or follow-up prose).
    # The executor already ran this pass on its draft, but the finalizer
    # introduces new course mentions that never went through it.
    rewrite_inline_prereqs = (validation_helpers or {}).get("rewrite_inline_prereqs")
    if callable(rewrite_inline_prereqs) and executor_result.metadata:
        prereq_graph_text = str(executor_result.metadata.get("prereq_graph_text") or "")
        if prereq_graph_text:
            try:
                final_response = rewrite_inline_prereqs(final_response, prereq_graph_text)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"⚠️ finalizer-stage prereq rewrite failed: {exc}")

    validation = validate_final_response(
        plan=plan,
        response_text=final_response,
        executor_result=executor_result,
        validation_helpers=validation_helpers,
    )
    metadata = {
        "raw_output": raw_output,
        "validation": validation,
    }
    return final_response, metadata


def run_three_stage_pipeline(
    user_question: str,
    conversation_history: Optional[List[dict]],
    has_transcript: bool,
    available_tools: Sequence[str],
    orchestrator_llm: Any,
    finalizer_llm: Any,
    execute_plan: Callable[[OrchestrationPlan, Optional[str]], ExecutorResult],
    trace: Optional[Dict[str, Any]] = None,
    trace_event: Optional[Callable[..., Any]] = None,
    trace_quality_check: Optional[Callable[..., Any]] = None,
    validation_helpers: Optional[Dict[str, Callable[..., Any]]] = None,
) -> Tuple[str, OrchestrationPlan]:
    plan, orchestrator_meta = build_orchestration_plan(
        user_question=user_question,
        conversation_history=conversation_history,
        has_transcript=has_transcript,
        available_tools=available_tools,
        llm=orchestrator_llm,
    )

    if trace is not None:
        trace["orchestrator"] = {
            **plan.to_dict(),
            "metadata": orchestrator_meta,
        }
        trace["plan"] = plan.to_dict()
    if trace_event is not None:
        trace_event(trace, "orchestrator", "Built orchestration plan", plan.to_dict())

    attempts: List[Dict[str, Any]] = []
    executor_result = execute_plan(plan, None)
    attempts.append(
        {
            "retry_reason": None,
            "result": executor_result.to_dict(),
        }
    )

    final_response, finalizer_meta = run_finalizer_stage(
        plan=plan,
        executor_result=executor_result,
        conversation_history=conversation_history,
        llm=finalizer_llm,
        validation_helpers=validation_helpers,
    )

    retry_reason = finalizer_meta.get("validation", {}).get("retry_reason")
    if retry_reason:
        executor_result = execute_plan(plan, retry_reason)
        attempts.append(
            {
                "retry_reason": retry_reason,
                "result": executor_result.to_dict(),
            }
        )
        final_response, finalizer_meta = run_finalizer_stage(
            plan=plan,
            executor_result=executor_result,
            conversation_history=conversation_history,
            llm=finalizer_llm,
            validation_helpers=validation_helpers,
        )

    if trace is not None:
        trace["executor"] = {
            "executor_path": plan.executor_path,
            "attempts": attempts,
        }
        trace["finalizer"] = finalizer_meta

    if trace_quality_check is not None:
        validation = finalizer_meta.get("validation", {})
        trace_quality_check(
            trace,
            "final_response_matches_plan",
            bool(validation.get("passed", False)),
            validation,
        )

    return final_response, plan
