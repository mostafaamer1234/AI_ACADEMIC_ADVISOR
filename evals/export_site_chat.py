"""Export the data the browser version of Leo (site/index.html) answers from.

Writes four files to site/data/:
  courses.json       every catalog course: code, title, credits, gen-ed tags, prerequisite
                     text and parsed prerequisite clauses (same parser as the referee)
  descriptions.json  readable course descriptions, loaded after the page is up (the catalog
                     CSV stores them without spaces, so they are re-segmented into words here)
  majors.json        the Abington majors table, with empty columns dropped
  policy.json        the advising and registrar FAQ split into short passages with page numbers

    pip install wordninja pypdf
    python -m evals.export_site_chat
"""

from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path

import pypdf
import wordninja

from evals.plan_checker import CODE_RE, normalize, parse_prereq_expression

ROOT = Path(__file__).resolve().parent.parent
LLM = ROOT / "IAAS" / "LLMAcadamic-Advisor" / "LLM"
OUT = ROOT / "site" / "data"
DICT = Path("/usr/share/dict/words")

WORDS = {w.strip().lower() for w in DICT.read_text().splitlines()} if DICT.exists() else set()
with open(LLM / "processed_psu_courses.csv", newline="", encoding="utf-8", errors="ignore") as _f:
    WORDS |= {r["course_number"].replace("\xa0", " ").split()[0].lower() for r in csv.DictReader(_f)}
WORDS |= {"programming", "cybersecurity", "software", "online", "internship", "database", "databases", "multimedia"}


def segment(squashed: str) -> str:
    """'courseintroductionbasicconcepts...' -> 'course introduction basic concepts ...'.

    wordninja splits rare words into fragments ('sum mari z ation'), so adjacent tokens are
    re-joined whenever the joined string is a dictionary word and a fragment is involved.
    """
    toks = wordninja.split(squashed)
    out, i = [], 0
    while i < len(toks):
        best = 1
        for j in range(min(len(toks), i + 6), i + 1, -1):
            joined = "".join(toks[i:j])
            if joined in WORDS and any(len(t) <= 3 or t not in WORDS for t in toks[i:j]):
                best = j - i
                break
        out.append("".join(toks[i : i + best]))
        i += best
    return " ".join(out)


def credits_of(name: str) -> tuple[str, str]:
    title, _, rest = name.partition("\n")
    m = re.search(r"([\d.]+(?:-[\d.]+)?)\s+Credits?", rest)
    return title.strip(), (m.group(1) if m else "")


def courses() -> tuple[list[list], dict[str, str]]:
    rows, descs = [], {}
    with open(LLM / "processed_psu_courses.csv", newline="", encoding="utf-8", errors="ignore") as f:
        for r in csv.DictReader(f):
            code = normalize(r["course_number"])
            title, cr = credits_of(r["course_name"])
            raw = (r.get("prerequisite_data") or "").strip()
            try:
                exprs = ast.literal_eval(raw) if raw.startswith("[") else ([raw] if raw else [])
            except (ValueError, SyntaxError):
                exprs = [raw]
            text = "; ".join(str(e).replace("\xa0", " ").strip() for e in exprs if e)
            clauses = [sorted(c) for e in exprs if e for c in parse_prereq_expression(str(e))]
            try:
                tags = [t.strip("()") for t in ast.literal_eval(r.get("learning_objectives") or "[]")]
            except (ValueError, SyntaxError):
                tags = []
            rows.append([code, title, cr, tags, text, clauses])
            if r["description"]:
                descs[code] = segment(r["description"])
    return rows, descs


def majors() -> list[dict]:
    out = []
    with open(LLM / "all_abington_majors_combined(5).csv", newline="", encoding="utf-8", errors="ignore") as f:
        for r in csv.DictReader(f):
            out.append({k: v.strip() for k, v in r.items() if k and v and v.strip()})
    return out


FOOTER = re.compile(r"1600 Woodland Road.*?© \d{4}\s*")
NOISE = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2}, .*(AM|PM).*|https?://\S+\s*\d+/\d+|Home».*|\d+/\d+)$")


def policy() -> list[dict]:
    reader = pypdf.PdfReader(str(LLM / "PennState_Combined_Advising_and_Registrar_VERBATIM.pdf"))
    chunks = []
    for page_no, page in enumerate(reader.pages, 1):
        paras, buf = [], []
        for line in (page.extract_text() or "").splitlines():
            s = line.strip()
            if not s or NOISE.match(s):
                continue
            if s.endswith("?") and len(s) < 140:
                if buf:
                    paras.append(" ".join(buf))
                    buf = []
                continue  # the FAQ prints its question list apart from the answers
            buf.append(s)
            if line.endswith(" ") and s.endswith((".", ":")) or (s.endswith(".") and len(s) < 60):
                paras.append(" ".join(buf))
                buf = []
        if buf:
            paras.append(" ".join(buf))
        merged, cur = [], ""
        for p in paras:
            if len(cur) + len(p) < 650:
                cur = f"{cur} {p}".strip()
            else:
                if cur:
                    merged.append(cur)
                cur = p
        if cur:
            merged.append(cur)
        for m in merged:
            m = FOOTER.sub("", re.sub(r"\s+", " ", m)).strip()
            if len(m) > 100 and "." in m:
                chunks.append({"p": page_no, "t": m})
    return chunks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (c, d), m, p = courses(), majors(), policy()
    for name, data in (("courses", c), ("descriptions", d), ("majors", m), ("policy", p)):
        (OUT / f"{name}.json").write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    print(len(c), "courses,", len(m), "majors,", len(p), "policy passages")


if __name__ == "__main__":
    main()
