"""Independent, deterministic checker for generated graduation plans.

The advisor already validates its own output inside the pipeline. This module is a
separate referee built directly from the course catalog, so evaluation does not
reuse (and cannot inherit bugs from) the code under test.

For a plan written as "Semester N:" blocks it reports:
  * unknown_courses      course codes that do not exist in the catalog (hallucinations)
  * prereq_violations    a course scheduled before at least one of its prerequisite groups is satisfied
  * duplicate_courses    the same course scheduled more than once
  * credit_issues        a stated semester total outside the allowed range
"""

from __future__ import annotations

import ast
import csv
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_CATALOG = Path(__file__).resolve().parent.parent / "IAAS" / "LLMAcadamic-Advisor" / "LLM" / "processed_psu_courses.csv"

CODE_RE = re.compile(r"\b([A-Z]{2,8}(?:/[A-Z]{2,8})?)\s?(\d{1,3}[A-Z]?)\b")
SEMESTER_RE = re.compile(r"^\W*semester\s+\+?(\d+)\b", re.IGNORECASE)
_NON_COURSE_ALTERNATIVE = re.compile(r"placement|examination|permission|consent|standing|equivalent|approval|instructor|concurrent", re.IGNORECASE)
CREDITS_RE = re.compile(r"total\s+credits?\W*\s*([\d.]+)", re.IGNORECASE)


def normalize(code: str) -> str:
    code = code.replace("\xa0", " ").upper().strip()
    m = CODE_RE.search(code)
    return f"{m.group(1)} {m.group(2)}" if m else code


def base_code(code: str) -> str:
    """MATH 140A -> MATH 140 (honors/variant suffixes satisfy the base requirement)."""
    return re.sub(r"(\d)[A-Z]$", r"\1", code)


def _split_top_level(expr: str, word: str) -> list[str]:
    parts, depth, buf, i = [], 0, [], 0
    token = f" {word} "
    low = expr.lower()
    while i < len(expr):
        ch = expr[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and low.startswith(token, i):
            parts.append("".join(buf))
            buf, i = [], i + len(token)
            continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def parse_prereq_expression(text: str) -> list[set[str]]:
    """Parse '(A or B) and (C or D)' into CNF: [{A, B}, {C, D}].

    Clauses that mention no course codes (e.g. 'fifth-semester standing') are dropped,
    since they cannot be checked from a course list.
    """
    text = text.replace("\xa0", " ").strip().rstrip(".")
    low = f" {text.lower()} "
    if " and " in low and " or " in low and "(" not in text:
        return []  # mixed and/or prose without grouping is ambiguous; don't guess precedence
    clauses: list[set[str]] = []
    for part in _split_top_level(text, "and"):
        part = part.strip()
        while part.startswith("(") and part.endswith(")"):
            part = part[1:-1].strip()
        if _NON_COURSE_ALTERNATIVE.search(part):
            continue  # e.g. "... or placement examination": satisfiable without any course
        codes = {normalize(f"{m.group(1)} {m.group(2)}") for m in CODE_RE.finditer(part.upper())}
        if codes:
            clauses.append(codes)
    return clauses


@dataclass
class Catalog:
    courses: set[str]
    prereqs: dict[str, list[set[str]]]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CATALOG) -> Catalog:
        courses: set[str] = set()
        prereqs: dict[str, list[set[str]]] = {}
        with open(path, newline="", encoding="utf-8", errors="ignore") as f:
            for row in csv.DictReader(f):
                code = normalize(row["course_number"])
                courses.add(code)
                raw = (row.get("prerequisite_data") or "").strip()
                try:
                    exprs = ast.literal_eval(raw) if raw.startswith("[") else [raw]
                except (ValueError, SyntaxError):
                    exprs = [raw]
                clauses = [c for e in exprs if e for c in parse_prereq_expression(str(e))]
                if clauses:
                    prereqs[code] = clauses
        return cls(courses, prereqs)


@dataclass
class Semester:
    number: int
    courses: list[str]
    stated_credits: float | None


@dataclass
class PlanReport:
    semesters: int
    courses: int
    unknown_courses: list[str] = field(default_factory=list)
    prereq_violations: list[str] = field(default_factory=list)
    duplicate_courses: list[str] = field(default_factory=list)
    credit_issues: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.semesters > 0 and not (self.unknown_courses or self.prereq_violations or self.duplicate_courses or self.credit_issues)

    def to_dict(self) -> dict:
        return asdict(self) | {"passed": self.passed}


def parse_plan(text: str) -> list[Semester]:
    semesters: list[Semester] = []
    current: Semester | None = None
    for line in text.splitlines():
        m = SEMESTER_RE.match(line.replace("*", ""))
        if m:
            current = Semester(int(m.group(1)), [], None)
            semesters.append(current)
            line = line[m.end() :]
        if current is None:
            continue
        credits = CREDITS_RE.search(line)
        if credits:
            current.stated_credits = float(credits.group(1))
            line = line[: credits.start()]
        current.courses.extend(normalize(f"{c.group(1)} {c.group(2)}") for c in CODE_RE.finditer(line) if len(c.group(2)) == 3 or c.group(2)[:-1].isdigit())
    return semesters


def check_plan(text: str, catalog: Catalog, min_credits: float = 12, max_credits: float = 19) -> PlanReport:
    semesters = parse_plan(text)
    all_courses = [c for s in semesters for c in s.courses]
    report = PlanReport(semesters=len(semesters), courses=len(all_courses))
    known_bases = {base_code(c) for c in catalog.courses}

    seen: set[str] = set()
    for code in all_courses:
        if code not in catalog.courses and base_code(code) not in known_bases and code not in report.unknown_courses:
            report.unknown_courses.append(code)
        if code in seen and code not in report.duplicate_courses:
            report.duplicate_courses.append(code)
        seen.add(code)

    taken: set[str] = set()
    for sem in sorted(semesters, key=lambda s: s.number):
        taken_bases = {base_code(c) for c in taken}
        for code in sem.courses:
            for clause in catalog.prereqs.get(code, []):
                if not any(p in taken or base_code(p) in taken_bases for p in clause):
                    report.prereq_violations.append(f"{code} in semester {sem.number} needs one of {sorted(clause)} earlier")
                    break
        taken.update(sem.courses)
        if sem.stated_credits is not None and not (min_credits <= sem.stated_credits <= max_credits):
            report.credit_issues.append(f"semester {sem.number}: {sem.stated_credits:g} credits")
    return report
