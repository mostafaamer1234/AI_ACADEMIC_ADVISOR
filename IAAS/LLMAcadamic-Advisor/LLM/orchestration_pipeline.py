from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
    return any(
        token in lowered
        for token in [
            "major requirement",
            "major requirements",
            "what courses do i need for",
            "what classes do i need for",
            "prescribed courses",
            "option courses",
        ]
    )


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

    try:
        semester_horizon = int(parsed.get("semester_horizon", fallback.semester_horizon))
    except Exception:
        semester_horizon = fallback.semester_horizon
    semester_horizon = max(0, min(8, semester_horizon))

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
    )


def _apply_deterministic_overrides(
    plan: OrchestrationPlan,
    conversation_history: Optional[List[dict]],
    available_tools: Sequence[str],
) -> OrchestrationPlan:
    corrected_question = _apply_common_corrections(plan.corrected_question or plan.normalized_question)
    requested_horizon = _extract_requested_horizon(corrected_question)
    recent_plan = _find_recent_assistant_plan(conversation_history)
    is_profile_recall = _is_profile_recall_question(corrected_question)

    if is_profile_recall:
        plan.domain = "general_academic_chat"
        plan.deliverable = "direct_answer"
        plan.plan_scope = "direct_answer"
        plan.semester_horizon = 0
        plan.use_tools = False
        plan.allowed_tools = []
        plan.executor_path = "conversation_profile_recall"
        plan.resolved_user_need = "Recall the student's year and/or major from earlier conversation context."

    elif recent_plan and requested_horizon > 0:
        plan.plan_scope = "single_semester" if requested_horizon == 1 else "multi_semester"
        plan.deliverable = "single_semester_plan" if requested_horizon == 1 else "multi_semester_plan"
        plan.domain = "transcript"
        plan.semester_horizon = requested_horizon
        plan.use_tools = False
        plan.allowed_tools = []
        plan.executor_path = "conversation_plan_slice"
        plan.resolved_user_need = f"Narrow the previously generated semester plan to the next {requested_horizon} semester(s)."

    elif requested_horizon > 0 and plan.domain == "transcript":
        plan.plan_scope = "single_semester" if requested_horizon == 1 else "multi_semester"
        plan.deliverable = "single_semester_plan" if requested_horizon == 1 else "multi_semester_plan"
        plan.semester_horizon = requested_horizon
        plan.use_tools = True
        plan.allowed_tools = _filter_allowed_tools(
            ["transcript_state", "deterministic_planner", "conversation_history"],
            available_tools,
            use_tools=True,
        )
        plan.executor_path = "deterministic_plan"

    elif _looks_like_plan_request(corrected_question) and requested_horizon == 0:
        plan.domain = "transcript"
        plan.plan_scope = "full_graduation"
        plan.deliverable = "full_graduation_plan"
        plan.semester_horizon = max(plan.semester_horizon, 8)
        plan.use_tools = True
        plan.allowed_tools = _filter_allowed_tools(
            ["transcript_state", "deterministic_planner", "conversation_history"],
            available_tools,
            use_tools=True,
        )
        plan.executor_path = "deterministic_plan"

    if COURSE_CODE_RE.search(corrected_question):
        plan.domain = "course"
        plan.deliverable = "course_lookup"
        plan.plan_scope = "direct_answer"
        plan.semester_horizon = 0
        plan.use_tools = True
        plan.allowed_tools = _filter_allowed_tools(
            ["prerequisite_graph", "course_retriever", "web_search"],
            available_tools,
            use_tools=True,
        )
        plan.executor_path = "course_lookup"

    elif _is_policy_question(corrected_question) or plan.domain == "policy" or plan.deliverable == "policy_lookup":
        plan.domain = "policy"
        plan.deliverable = "policy_lookup"
        plan.plan_scope = "direct_answer"
        plan.semester_horizon = 0
        plan.use_tools = True
        plan.allowed_tools = _filter_allowed_tools(
            ["policy_retriever", "web_search"],
            available_tools,
            use_tools=True,
        )
        plan.executor_path = "policy_lookup"

    elif _is_major_question(corrected_question) and plan.plan_scope == "direct_answer":
        plan.domain = "major"
        plan.deliverable = "major_lookup"
        plan.use_tools = True
        plan.allowed_tools = _filter_allowed_tools(
            ["majors_retriever", "web_search"],
            available_tools,
            use_tools=True,
        )
        plan.executor_path = "major_lookup"

    elif _looks_like_generic_research_question(corrected_question):
        plan.domain = "general_academic_chat"
        plan.deliverable = "direct_answer"
        plan.plan_scope = "direct_answer"
        plan.semester_horizon = 0
        plan.use_tools = True
        plan.allowed_tools = _filter_allowed_tools(
            ["conversation_history", "web_search"],
            available_tools,
            use_tools=True,
        )
        plan.executor_path = "research_lookup"

    valid_conversation_slice = bool(recent_plan and requested_horizon > 0)
    if plan.executor_path == "conversation_plan_slice" and not valid_conversation_slice:
        if _looks_like_plan_request(corrected_question) or requested_horizon > 0:
            plan.domain = "transcript"
            plan.plan_scope = "single_semester" if requested_horizon == 1 else "multi_semester" if requested_horizon > 1 else "full_graduation"
            plan.deliverable = "single_semester_plan" if requested_horizon == 1 else "multi_semester_plan" if requested_horizon > 1 else "full_graduation_plan"
            plan.semester_horizon = requested_horizon if requested_horizon > 0 else max(plan.semester_horizon, 8)
            plan.use_tools = True
            plan.allowed_tools = _filter_allowed_tools(
                ["transcript_state", "deterministic_planner", "conversation_history"],
                available_tools,
                use_tools=True,
            )
            plan.executor_path = "deterministic_plan"
            plan.resolved_user_need = "Generate a semester plan that matches the requested planning horizon."
        else:
            plan.domain = "general_academic_chat"
            plan.deliverable = "direct_answer"
            plan.plan_scope = "direct_answer"
            plan.semester_horizon = 0
            plan.use_tools = False
            plan.allowed_tools = []
            plan.executor_path = "conversation_profile_recall" if is_profile_recall else "general_academic_chat"

    if not plan.use_tools:
        plan.allowed_tools = []

    plan.corrected_question = corrected_question
    plan.normalized_question = _normalize_question_text(plan.normalized_question or plan.original_question)
    plan.must_not_do = _build_must_not_do(plan.plan_scope, plan.semester_horizon, plan.use_tools)
    plan.success_criteria = _build_success_criteria(plan.plan_scope, plan.semester_horizon, corrected_question)
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
        "plan_scope, semester_horizon, use_tools, allowed_tools, executor_path, must_not_do, success_criteria. "
        "Use conversation history to resolve follow-ups. "
        "Prefer use_tools=false when the question can be answered from conversation history and reasoning alone. "
        "Do not force a special executor path when a direct conversational answer is enough. "
        "Unexpected academic follow-ups should stay in general_academic_chat unless a tool is clearly required. "
        "Unexpected factual questions that require fresh or sourced information should use web_search and executor_path=research_lookup. "
        "If the user narrows a previously given semester plan, set use_tools=false and executor_path=conversation_plan_slice. "
        "plan_scope must be one of: single_semester, multi_semester, full_graduation, audit, direct_answer. "
        "domain must be one of: transcript, course, major, policy, general_academic_chat. "
        "deliverable must be one of: direct_answer, audit, single_semester_plan, multi_semester_plan, "
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
    return {
        "passed": not issues,
        "issues": issues,
        "retry_reason": retry_reason,
        "semester_block_count": len(semester_blocks),
        "used_tools": used_tools,
    }


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
        "You are the final response layer for an academic advising assistant. "
        "Rewrite the draft so it is friendly, professional, and appropriately detailed without changing the answer scope. "
        "Keep semester blocks intact when present. End with a short invitation for the next question. "
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
