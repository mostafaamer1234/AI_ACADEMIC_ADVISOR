import sys
import unittest
from pathlib import Path
from unittest.mock import patch


LLM_DIR = Path(__file__).resolve().parent / "LLMAcadamic-Advisor" / "LLM"
if str(LLM_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_DIR))

from orchestration_pipeline import ExecutorResult, build_orchestration_plan, run_three_stage_pipeline  # noqa: E402

MAIN_COPY_IMPORT_ERROR = None
try:
    import main_copy  # noqa: E402
except Exception as exc:  # pragma: no cover - environment-dependent
    main_copy = None
    MAIN_COPY_IMPORT_ERROR = exc


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(self, responses=None):
        self.responses = list(responses or [""])
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.responses:
            content = self.responses.pop(0)
        else:
            content = ""
        return FakeResponse(content)


FULL_PLAN_TEXT = """Semester 3:
BA 243 (prerequisites: None)
CAS 352 (prerequisites: None)
Total Credits: 15
---
Semester 4:
MGMT 301 (prerequisites: None)
STAT 200 (prerequisites: None)
Total Credits: 15
---
Semester 5:
FIN 301 (prerequisites: None)
SCM 301 (prerequisites: None)
Total Credits: 15"""

MARKDOWN_PLAN_TEXT = """Here's a suggested graduation plan for a junior Computer Science major.

**Semester 5:**
- CMPSC 313
- MATH 425
- **Total Credits:** 6

---

**Semester 6:**
- CMPSC 330
- CMPSC 360
- **Total Credits:** 15"""


class OrchestrationPipelineTests(unittest.TestCase):
    def test_build_orchestration_plan_narrows_prior_plan_without_tools(self):
        history = [
            {"role": "user", "content": "give me a graduation plan for a sophmore in business managment"},
            {"role": "assistant", "content": FULL_PLAN_TEXT},
        ]
        plan, metadata = build_orchestration_plan(
            user_question="how about just a plan for next semester only",
            conversation_history=history,
            has_transcript=False,
            available_tools=["conversation_history", "deterministic_planner", "transcript_state"],
            llm=FakeLLM([""]),
        )

        self.assertEqual(plan.plan_scope, "single_semester")
        self.assertEqual(plan.deliverable, "single_semester_plan")
        self.assertFalse(plan.use_tools)
        self.assertEqual(plan.executor_path, "conversation_plan_slice")
        self.assertIn("sophomore", build_orchestration_plan(
            user_question="give me a graduation plan for a sophmore in business managment",
            conversation_history=[],
            has_transcript=False,
            available_tools=["deterministic_planner"],
            llm=FakeLLM([""]),
        )[0].corrected_question.lower())
        self.assertTrue(metadata["recent_plan_available"])

    def test_build_orchestration_plan_corrects_typos(self):
        plan, _ = build_orchestration_plan(
            user_question="give me a graduation plan for a sophmore in business managment",
            conversation_history=[],
            has_transcript=False,
            available_tools=["deterministic_planner"],
            llm=FakeLLM([""]),
        )

        self.assertIn("sophomore", plan.corrected_question.lower())
        self.assertIn("management", plan.corrected_question.lower())
        self.assertEqual(plan.deliverable, "full_graduation_plan")

    def test_build_orchestration_plan_overrides_bad_slice_for_profile_recall(self):
        history = [
            {"role": "user", "content": "give me a graduation plan for a junior computer science major"},
            {"role": "assistant", "content": MARKDOWN_PLAN_TEXT},
            {"role": "user", "content": "just for next semester"},
            {"role": "assistant", "content": "As a junior Computer Science major, here is a suggested course plan for your next semester."},
        ]
        bad_orchestrator_json = """{
          "normalized_question": "What year and major am I again?",
          "corrected_question": "What year and major am I again?",
          "resolved_user_need": "User needs to confirm their academic year and major.",
          "domain": "general_academic_chat",
          "deliverable": "direct_answer",
          "plan_scope": "single_semester",
          "semester_horizon": "next semester",
          "use_tools": false,
          "allowed_tools": [],
          "executor_path": "conversation_plan_slice",
          "must_not_do": [],
          "success_criteria": "User confirms their academic year and major."
        }"""

        plan, metadata = build_orchestration_plan(
            user_question="what year and major am i again?",
            conversation_history=history,
            has_transcript=False,
            available_tools=["conversation_history", "deterministic_planner"],
            llm=FakeLLM([bad_orchestrator_json]),
        )

        self.assertEqual(plan.plan_scope, "direct_answer")
        self.assertEqual(plan.executor_path, "conversation_profile_recall")
        self.assertEqual(plan.semester_horizon, 0)
        self.assertFalse(plan.use_tools)
        self.assertTrue(metadata["recent_plan_available"])

    def test_build_orchestration_plan_keeps_unexpected_followup_in_reasoning_mode(self):
        history = [
            {"role": "user", "content": "give me a graduation plan for a junior computer science major"},
            {"role": "assistant", "content": MARKDOWN_PLAN_TEXT},
        ]
        bad_orchestrator_json = """{
          "normalized_question": "Why did you recommend that schedule?",
          "corrected_question": "Why did you recommend that schedule?",
          "resolved_user_need": "Explain the plan.",
          "domain": "general_academic_chat",
          "deliverable": "direct_answer",
          "plan_scope": "single_semester",
          "semester_horizon": "next semester",
          "use_tools": false,
          "allowed_tools": [],
          "executor_path": "conversation_plan_slice",
          "must_not_do": [],
          "success_criteria": "Explain the recommendation."
        }"""

        plan, _ = build_orchestration_plan(
            user_question="why did you recommend that schedule?",
            conversation_history=history,
            has_transcript=False,
            available_tools=["conversation_history", "deterministic_planner"],
            llm=FakeLLM([bad_orchestrator_json]),
        )

        self.assertEqual(plan.plan_scope, "direct_answer")
        self.assertEqual(plan.executor_path, "general_academic_chat")
        self.assertFalse(plan.use_tools)

    def test_build_orchestration_plan_forces_policy_followup_to_direct_answer(self):
        history = [
            {"role": "user", "content": "when is the late drop and early drop deadlines for this current semester"},
            {"role": "assistant", "content": "Tell me your campus and semester and I can narrow it down."},
            {"role": "user", "content": "I am at penn state abington"},
            {"role": "assistant", "content": "I can help find the exact dates if you tell me the semester."},
        ]
        bad_orchestrator_json = """{
          "normalized_question": "spring 2026",
          "corrected_question": "What are the exact Penn State Abington academic calendar dates for Spring 2026?",
          "resolved_user_need": "The user is asking for the Spring 2026 semester dates.",
          "domain": "policy",
          "deliverable": "policy_lookup",
          "plan_scope": "single_semester",
          "semester_horizon": "spring_2026",
          "use_tools": true,
          "allowed_tools": ["policy_retriever", "web_search"],
          "executor_path": "policy_lookup",
          "must_not_do": [],
          "success_criteria": "Return the exact dates."
        }"""

        plan, _ = build_orchestration_plan(
            user_question="spring 2026",
            conversation_history=history,
            has_transcript=False,
            available_tools=["conversation_history", "policy_retriever", "web_search"],
            llm=FakeLLM([bad_orchestrator_json]),
        )

        self.assertEqual(plan.domain, "policy")
        self.assertEqual(plan.plan_scope, "direct_answer")
        self.assertEqual(plan.executor_path, "policy_lookup")
        self.assertEqual(plan.semester_horizon, 0)

    def test_build_orchestration_plan_routes_unexpected_fact_to_research_lookup(self):
        plan, _ = build_orchestration_plan(
            user_question="when was Python 3.12 released",
            conversation_history=[],
            has_transcript=False,
            available_tools=["conversation_history", "web_search"],
            llm=FakeLLM([""]),
        )

        self.assertEqual(plan.plan_scope, "direct_answer")
        self.assertTrue(plan.use_tools)
        self.assertEqual(plan.executor_path, "research_lookup")
        self.assertIn("web_search", plan.allowed_tools)

    def test_generate_conversational_response_slices_prior_plan_followup(self):
        if main_copy is None:
            self.skipTest(f"main_copy unavailable in this environment: {MAIN_COPY_IMPORT_ERROR}")
        user_id = "pipeline-followup-test"
        main_copy.execution_trace_store.clear()
        history = [
            {"role": "user", "content": "give me a graduation plan for a sophmore in business managment"},
            {"role": "assistant", "content": FULL_PLAN_TEXT},
        ]

        with patch.object(main_copy, "llm_classifier", FakeLLM([""])), \
             patch.object(main_copy, "llm1", FakeLLM([""])), \
             patch.object(main_copy, "llm2", FakeLLM([""])), \
             patch.object(main_copy, "has_transcript", False), \
             patch.object(main_copy, "stored_pdf_docs", None), \
             patch.object(main_copy, "latest_student_data", None):
            response = main_copy.generate_conversational_response(
                "how about just a plan for next semester only",
                history,
                user_id=user_id,
                user_has_transcript=False,
            )

        self.assertIn("Semester 3:", response)
        self.assertNotIn("Semester 4:", response)

        trace = main_copy.get_execution_trace_for_user(user_id)
        self.assertEqual(trace["orchestrator"]["plan_scope"], "single_semester")
        self.assertEqual(trace["executor"]["attempts"][0]["result"]["tools_used"], [])

    def test_generate_conversational_response_recalls_major_and_year_from_history(self):
        if main_copy is None:
            self.skipTest(f"main_copy unavailable in this environment: {MAIN_COPY_IMPORT_ERROR}")
        user_id = "pipeline-profile-recall-test"
        main_copy.execution_trace_store.clear()
        history = [
            {"role": "user", "content": "give me a graduation plan for a junior computer science major"},
            {"role": "assistant", "content": MARKDOWN_PLAN_TEXT},
            {"role": "user", "content": "what year and major am i again?"},
        ]
        bad_orchestrator_json = """{
          "normalized_question": "What year and major am I again?",
          "corrected_question": "What year and major am I again?",
          "resolved_user_need": "User needs to confirm their academic year and major.",
          "domain": "general_academic_chat",
          "deliverable": "direct_answer",
          "plan_scope": "single_semester",
          "semester_horizon": "next semester",
          "use_tools": false,
          "allowed_tools": [],
          "executor_path": "conversation_plan_slice",
          "must_not_do": [],
          "success_criteria": "User confirms their academic year and major."
        }"""

        with patch.object(main_copy, "llm_classifier", FakeLLM([bad_orchestrator_json])), \
             patch.object(main_copy, "llm1", FakeLLM([""])), \
             patch.object(main_copy, "llm2", FakeLLM([""])), \
             patch.object(main_copy, "has_transcript", False), \
             patch.object(main_copy, "stored_pdf_docs", None), \
             patch.object(main_copy, "latest_student_data", None):
            response = main_copy.generate_conversational_response(
                "what year and major am i again?",
                history[:-1],
                user_id=user_id,
                user_has_transcript=False,
                debug_mode=True,
            )

        self.assertIn("junior", response.lower())
        self.assertIn("computer science", response.lower())

        trace = main_copy.get_execution_trace_for_user(user_id)
        self.assertEqual(trace["orchestrator"]["executor_path"], "conversation_profile_recall")
        self.assertEqual(trace["orchestrator"]["plan_scope"], "direct_answer")

    def test_run_three_stage_pipeline_retries_on_scope_mismatch(self):
        attempts = []
        trace = {"enabled": True, "events": [], "quality_checks": []}

        def execute_plan(plan, retry_reason=None):
            attempts.append(retry_reason)
            if len(attempts) == 1:
                return ExecutorResult(response=FULL_PLAN_TEXT, tools_used=[], metadata={})
            return ExecutorResult(
                response="Semester 3:\nBA 243 (prerequisites: None)\nTotal Credits: 15",
                tools_used=[],
                metadata={},
            )

        final_response, plan = run_three_stage_pipeline(
            user_question="just next semester only",
            conversation_history=[{"role": "assistant", "content": FULL_PLAN_TEXT}],
            has_transcript=False,
            available_tools=["conversation_history"],
            orchestrator_llm=FakeLLM([""]),
            finalizer_llm=FakeLLM(["", ""]),
            execute_plan=execute_plan,
            trace=trace,
            validation_helpers={},
        )

        self.assertEqual(plan.plan_scope, "single_semester")
        self.assertEqual(attempts, [None, "scope_mismatch_single_semester"])
        self.assertIn("Semester 3:", final_response)
        self.assertNotIn("Semester 4:", final_response)


if __name__ == "__main__":
    unittest.main()
