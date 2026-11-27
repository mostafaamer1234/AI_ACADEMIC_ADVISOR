from __future__ import annotations

import pytest

from evals.plan_checker import Catalog, check_plan, parse_prereq_expression


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return Catalog.load()


def test_parses_and_of_ors_into_cnf():
    assert parse_prereq_expression("(CMPSC\xa0122 or CMPSC\xa0132) and (CMPSC\xa0360 or MATH\xa0311W)") == [
        {"CMPSC 122", "CMPSC 132"},
        {"CMPSC 360", "MATH 311W"},
    ]


def test_ignores_clauses_without_course_codes():
    assert parse_prereq_expression("fifth-semester standing and CMPSC 121") == [{"CMPSC 121"}]


def test_ambiguous_or_non_course_prerequisites_are_not_enforced():
    assert parse_prereq_expression("Math 22 and Math 26 or Math 40 or satisfactory performance on the placement examination") == []
    assert parse_prereq_expression("CMPSC 121 or permission of the instructor") == []


def test_catalog_loads_real_prerequisites(catalog):
    assert "CMPSC 465" in catalog.courses
    assert {"CMPSC 122", "CMPSC 132"} in catalog.prereqs["CMPSC 465"]


def test_valid_sequence_passes(catalog):
    plan = """
    Semester 1: CMPSC 131, MATH 140, ENGL 15
    Total Credits: 13
    Semester 2: CMPSC 132, MATH 141, CMPSC 360
    Total Credits: 13
    Semester 3: CMPSC 221, CMPSC 465
    Total Credits: 13
    """
    report = check_plan(plan, catalog)
    assert report.passed, report


def test_prerequisite_violation_is_reported(catalog):
    plan = "Semester 1: CMPSC 465, MATH 140\nTotal Credits: 13\nSemester 2: CMPSC 132\nTotal Credits: 13"
    report = check_plan(plan, catalog)
    assert not report.passed
    assert any(v.startswith("CMPSC 465 in semester 1") for v in report.prereq_violations)


def test_honors_variant_satisfies_base_prerequisite(catalog):
    plan = "Semester 1: MATH 140H\nTotal Credits: 13\nSemester 2: MATH 141\nTotal Credits: 13"
    assert not check_plan(plan, catalog).prereq_violations


def test_hallucinated_course_and_duplicates_are_flagged(catalog):
    plan = "**Semester 1:** CMPSC 999, MATH 140\nTotal Credits: 25\nSemester 2: MATH 140\nTotal Credits: 13"
    report = check_plan(plan, catalog)
    assert "CMPSC 999" in report.unknown_courses
    assert "MATH 140" in report.duplicate_courses
    assert report.credit_issues == ["semester 1: 25 credits"]
