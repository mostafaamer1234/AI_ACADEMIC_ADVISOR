"""Run the graduation-plan eval and print a scorecard.

Live mode sends each request in evals/requests.jsonl to a running advisor API and
scores every plan with the independent catalog checker:

    python IAAS/LLMAcadamic-Advisor/start_flask.py          # needs OPENAI_API_KEY
    python -m evals.run_eval --api http://localhost:5002

Offline mode scores plans you already saved (one JSON object per line with
"id" and "plan"):

    python -m evals.run_eval --plans saved_plans.jsonl

Results are written to evals/results/<timestamp>.json; the markdown summary is
printed so it can be pasted into the README.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests

from evals.plan_checker import Catalog, check_plan

HERE = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ask_advisor(api: str, question: str, timeout: float) -> tuple[str, float]:
    start = time.perf_counter()
    resp = requests.post(
        f"{api.rstrip('/')}/llm/generate-response",
        json={"user_question": question, "conversation_history": [], "user_id": f"eval-{uuid.uuid4()}"},
        timeout=timeout,
        verify=False,  # local/self-signed dev servers
    )
    resp.raise_for_status()
    return resp.json().get("graduation_plan", ""), time.perf_counter() - start


def summarize(rows: list[dict]) -> str:
    n = len(rows)
    scored = [r for r in rows if r["report"]["semesters"] > 0]
    passed = sum(r["report"]["passed"] for r in rows)
    courses = sum(r["report"]["courses"] for r in scored) or 1
    unknown = sum(len(r["report"]["unknown_courses"]) for r in scored)
    violations = sum(len(r["report"]["prereq_violations"]) for r in scored)
    dupes = sum(bool(r["report"]["duplicate_courses"]) for r in scored)
    credit = sum(bool(r["report"]["credit_issues"]) for r in scored)
    lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    lines = [
        f"| Metric | Value (n={n} plans) |",
        "|---|---|",
        f"| Plans passing every check | {passed}/{n} ({passed / n:.0%}) |",
        f"| Plans with a parseable semester schedule | {len(scored)}/{n} |",
        f"| Courses not in the catalog (hallucination rate) | {unknown}/{courses} ({unknown / courses:.1%}) |",
        f"| Prerequisite-order violations | {violations} across {sum(bool(r['report']['prereq_violations']) for r in scored)} plans |",
        f"| Plans with duplicate courses | {dupes} |",
        f"| Plans with a semester outside 12-19 credits | {credit} |",
    ]
    if lat:
        lines.append(f"| Median / p90 latency | {statistics.median(lat):.1f}s / {sorted(lat)[int(0.9 * (len(lat) - 1))]:.1f}s |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--api", help="base URL of a running advisor API")
    src.add_argument("--plans", type=Path, help="JSONL of saved plans to score")
    ap.add_argument("--requests", type=Path, default=HERE / "requests.jsonl")
    ap.add_argument("--timeout", type=float, default=300)
    args = ap.parse_args()

    catalog = Catalog.load()
    rows: list[dict] = []
    if args.api:
        for item in load_jsonl(args.requests):
            try:
                plan, latency = ask_advisor(args.api, item["question"], args.timeout)
            except requests.RequestException as exc:
                plan, latency = f"ERROR: {exc}", None
            rows.append({"id": item["id"], "question": item["question"], "plan": plan, "latency_s": latency,
                         "report": check_plan(plan, catalog).to_dict()})
            print(f"{item['id']}: {'PASS' if rows[-1]['report']['passed'] else 'FAIL'}", flush=True)
    else:
        for item in load_jsonl(args.plans):
            rows.append({"id": item["id"], "plan": item["plan"], "latency_s": item.get("latency_s"),
                         "report": check_plan(item["plan"], catalog).to_dict()})

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}\n")
    print(summarize(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
