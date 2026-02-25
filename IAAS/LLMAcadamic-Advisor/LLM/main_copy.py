import os
import sys
import re
import json
import shutil
import hashlib
import pickle
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from urllib.parse import urlparse, urljoin
from html import unescape

# Add current directory to Python path BEFORE importing local modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

import pandas as pd

# Make sure local modules can be imported


from graphExperiment import course_graph, get_course_prerequisites, get_major_course_codes


# =========================
# Globals
# =========================

pdf_retriever = None
course_retriever = None  # For individual course queries (processed_psu_courses.csv)
majors_retriever = None  # For major requirements queries (all_abington_majors_combined(5).csv)
policy_retriever = None  # For advising/registrar policy PDF queries
policy_vectorstore = None
llm1 = None
llm2 = None
llm_classifier = None  # Faster model for intent classification (gpt-4o-mini)
has_transcript = False
stored_pdf_docs = None  # Store original PDF docs for deterministic extraction
latest_student_data = None  # Cache most recent extracted student data
execution_trace_store = {}  # Per-session lightweight execution traces

# Absolute paths for Chroma persistence
LLM_DIR = os.path.abspath(os.path.dirname(__file__))
CHROMA_DATA_DIR = os.path.join(LLM_DIR, "chroma_data")
COLLECTION_METADATA_FILE = os.path.join(CHROMA_DATA_DIR, "collection_metadata.json")
POLICY_PDF_PATH = os.path.join(LLM_DIR, "PennState_Combined_Advising_and_Registrar_VERBATIM.pdf")

# =========================
# Helper Functions for Document Processing
# =========================

def get_document_hash(file_path: str) -> str:
    """Generate SHA256 hash of PDF file."""
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_transcript_collection_name(user_id: str, doc_hash: str) -> str:
    """Generate collection name: transcript_userid_hash8"""
    return f"transcript_{user_id}_{doc_hash[:8]}"

def chunk_with_metadata(pdf_docs: List[Document]) -> List[Document]:
    """Chunk PDF with page and section metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=130
    )
    chunks = text_splitter.split_documents(pdf_docs)
    
    # Add metadata
    for i, chunk in enumerate(chunks):
        page = chunk.metadata.get("page", 0)
        content = chunk.page_content
        
        # Detect section
        section = "general"
        if "Not Satisfied" in content or "Unmet" in content:
            section = "unmet_requirements"
        elif "In Progress" in content:
            section = "in_progress"
        elif "Course History" in content or re.search(r'\b[A-Z]{2,}\s+\d{3}\b', content):
            section = "course_history"
        elif "Major" in content and "Option" in content:
            section = "major_info"
        
        chunk.metadata.update({
            "page": page,
            "section": section,
            "chunk_id": f"page_{page}_chunk_{i}"
        })
    
    return chunks


def is_policy_question(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    policy_terms = [
        "advisor", "adviser", "advising", "advising center",
        "drop", "add", "late drop", "withdraw", "audit",
        "leave of absence", "declare major", "declare minor",
        "double major", "concurrent major", "change my major",
        "what-if report", "lionpath", "academic requirements",
        "entry to major", "early progress report", "internship",
        "aleks", "world campus", "registration", "bursar",
        "tuition penalty", "credits", "f grade", "gpa"
    ]
    return any(term in t for term in policy_terms)


def is_academic_or_policy_question(text: str) -> bool:
    """
    Check if the question is about academics or Penn State policy.
    Used to: (1) allow web search fallback, (2) reject non-academic questions with scope message.
    """
    if not text or len(text.strip()) < 5:
        return False
    t = text.lower()
    academic_policy_terms = [
        # Policy / institutional
        "advisor", "adviser", "advising", "drop", "add", "withdraw", "audit",
        "leave of absence", "declare major", "declare minor", "registration",
        "bursar", "tuition", "gpa", "grade", "credit", "lionpath",
        "penn state", "psu", "university policy", "academic policy",
        "academic requirement", "graduation", "degree", "enrollment",
        "course", "class", "major", "minor", "prerequisite",
        "semester", "transcript", "degree audit", "what-if",
        "internship", "co-op", "study abroad", "transfer credit",
    ]
    return any(term in t for term in academic_policy_terms)


def _response_indicates_no_answer_found(response: str) -> bool:
    """Check if the LLM response indicates it couldn't find the answer in available data."""
    if not response or len(response.strip()) < 20:
        return True
    lower = response.lower()
    indicators = [
        "couldn't find", "could not find", "isn't present", "not in the policy",
        "not found in", "unable to find", "i don't have", "i do not have",
        "couldn't find that", "no relevant", "please rephrase", "not available",
        "couldn't verify", "could not verify",
    ]
    return any(ind in lower for ind in indicators)


def _is_time_sensitive_identity_question(text: str) -> bool:
    """Detect questions that require up-to-date role holder verification."""
    if not text:
        return False
    t = text.lower()
    role_terms = [
        "chancellor", "program chair", "chair", "dean", "director",
        "president", "provost", "head of", "department head",
    ]
    asking_identity = any(
        p in t for p in [
            "who is", "who's", "current", "currently", "now", "at present",
            "who serves as", "who holds",
        ]
    )
    return asking_identity and any(r in t for r in role_terms)


def _is_time_sensitive_factual_question(text: str) -> bool:
    """Generic detector for questions requiring fresh, source-backed facts."""
    if not text:
        return False
    t = text.lower().strip()
    starters = re.search(r"^(who|what|when)\b", t) is not None
    freshness_terms = any(k in t for k in ["current", "currently", "latest", "now", "this semester", "this year", "upcoming"])
    institution_terms = any(k in t for k in [
        "penn state", "psu", "abington", "chancellor", "chair", "dean", "director",
        "deadline", "calendar", "registration", "policy"
    ])
    return (starters and institution_terms) or (freshness_terms and institution_terms)


def _build_execution_profile(
    contextual_question: str,
    query_type: str,
    reasoning_plan: Dict[str, object]
) -> Dict[str, object]:
    """Universal execution profile used by all query paths."""
    tools = reasoning_plan.get("tools", {}) if reasoning_plan else {}
    web_enabled = bool(tools.get("web_search")) or is_academic_or_policy_question(contextual_question)
    mode = "none"
    require_web_precheck = False
    must_include_source = False
    prefer_current_or_future = False

    if _question_needs_specific_date(contextual_question):
        mode = "date"
        require_web_precheck = True
        prefer_current_or_future = True
    elif _is_time_sensitive_factual_question(contextual_question) and query_type in {"chat", "policy", "major", "course"}:
        mode = "fresh_fact"
        require_web_precheck = True
        must_include_source = True

    return {
        "verification_mode": mode,
        "require_web_precheck": require_web_precheck and web_enabled,
        "must_include_source": must_include_source,
        "prefer_current_or_future": prefer_current_or_future,
        "web_enabled": web_enabled,
        "identity_like": _is_time_sensitive_identity_question(contextual_question),
        "source_domain_hints": _extract_domain_hints(contextual_question),
    }


def _recent_user_messages(conversation_history: Optional[List[dict]], limit: int = 10) -> List[str]:
    if not conversation_history:
        return []
    msgs = [str(m.get("content", "")).strip() for m in conversation_history if m.get("role") == "user"]
    return [m for m in msgs[-limit:] if m]


def _is_vague_followup(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    explicit_followup_terms = [
        "current semester", "this semester", "for this semester", "for the current semester",
        "that one", "for that", "for this", "what about this", "after that", "same one",
    ]
    if any(term in t for term in explicit_followup_terms):
        return True

    # Term/session-only answers are typical follow-ups to deadline questions.
    if len(t.split()) <= 5 and (
        re.search(r"\b(spring|summer|fall|winter)\b", t)
        or re.search(r"\b20\d{2}\b", t)
        or "regular session" in t
        or "7-week" in t
        or "7 week" in t
        or "session" in t
    ):
        return True

    # Treat full WH questions as new topics unless explicit follow-up markers exist.
    if re.search(r"^(who|what|when|where|why|how)\b", t) and len(t.split()) >= 4:
        return False

    return False


def _contextualize_policy_question(user_question: str, conversation_history: Optional[List[dict]] = None) -> str:
    """Attach prior policy/deadline context for vague follow-up messages."""
    q = (user_question or "").strip()
    if not q:
        return q
    if not conversation_history or not _is_vague_followup(q):
        return q

    prior_user = _recent_user_messages(conversation_history, limit=8)
    for msg in reversed(prior_user):
        m = msg.lower()
        if any(k in m for k in ["deadline", "late drop", "withdraw", "drop deadline", "drop period"]):
            return f"{msg}. Clarification: {q}"
    return q


def _infer_term_year_from_context(
    user_question: str,
    conversation_history: Optional[List[dict]] = None
) -> Tuple[Optional[str], Optional[int]]:
    """Infer target term/year, preferring current or upcoming semester for vague requests."""
    q = _contextualize_policy_question(user_question, conversation_history).lower()
    now = datetime.now()
    current_year = now.year

    if now.month <= 5:
        current_term = "spring"
    elif now.month <= 8:
        current_term = "summer"
    else:
        current_term = "fall"

    year_match = re.search(r"\b(20\d{2})\b", q)
    target_year = int(year_match.group(1)) if year_match else None

    term = None
    for term_token in ["spring", "summer", "fall", "winter"]:
        if term_token in q:
            term = term_token
            break

    if "next semester" in q or "upcoming semester" in q:
        if current_term == "spring":
            term, target_year = "summer", current_year
        elif current_term == "summer":
            term, target_year = "fall", current_year
        else:
            term, target_year = "spring", current_year + 1
    elif "current semester" in q or "this semester" in q:
        term = term or current_term
        target_year = target_year or current_year

    if target_year is None:
        target_year = current_year

    return term, target_year


def _question_needs_specific_date(text: str) -> bool:
    """Detect date/deadline style questions that usually need a concrete date."""
    if not text:
        return False
    t = text.lower()
    date_terms = [
        "deadline", "due date", "what date", "exact date", "by when",
        "when is", "last day", "cutoff", "closing date", "ends on",
    ]
    academic_event_terms = [
        "late drop", "drop", "withdraw", "registration", "add period",
        "semester start", "semester end", "tuition penalty",
    ]
    return any(dt in t for dt in date_terms) and any(ev in t for ev in academic_event_terms)


def _has_concrete_date_in_text(text: str) -> bool:
    """Heuristic check for concrete calendar dates in a response."""
    if not text:
        return False
    t = text.lower()
    month_pattern = r"\b(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\b"
    numeric_date_pattern = r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b"
    long_date_pattern = r"\b\d{1,2}(st|nd|rd|th)?\b"
    has_month = re.search(month_pattern, t) is not None
    has_numeric = re.search(numeric_date_pattern, t) is not None
    # A month + day signal like "March 22" (allow optional ordinal suffix)
    has_month_and_day = has_month and re.search(long_date_pattern, t) is not None
    return has_numeric or has_month_and_day


def _extract_dates_from_text(text: str) -> List[str]:
    """Extract concrete date strings from free text."""
    if not text:
        return []
    month_date = re.findall(
        r"\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)\s+\d{1,2}(?:,\s*\d{2,4})?\b",
        text,
        flags=re.IGNORECASE,
    )
    numeric_date = re.findall(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", text)
    dates: List[str] = []
    for d in month_date + numeric_date:
        cleaned = " ".join(d.split()).strip()
        if cleaned not in dates:
            dates.append(cleaned)
    return dates


def _parse_date_candidate(date_text: str, fallback_year: Optional[int] = None) -> Optional[datetime]:
    """Parse supported date formats into datetime for ranking/validation."""
    if not date_text:
        return None
    s = date_text.strip()
    s = re.sub(r'(\d)(st|nd|rd|th)\b', r'\1', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+', ' ', s)

    formats = [
        "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
        "%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%m-%d-%y",
        "%B %d", "%b %d", "%m/%d", "%m-%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            if "%Y" not in fmt and "%y" not in fmt:
                dt = dt.replace(year=fallback_year or datetime.now().year)
            return dt
        except Exception:
            continue
    return None


def _build_deadline_search_queries(
    user_question: str,
    conversation_history: Optional[List[dict]] = None
) -> List[str]:
    """Build targeted PSU policy/deadline web queries."""
    context_q = _contextualize_policy_question(user_question, conversation_history)
    term, year = _infer_term_year_from_context(user_question, conversation_history)
    year_txt = str(year) if year else ""
    term_txt = term if term else "current"

    queries = [
        f"Penn State {term_txt} {year_txt} late drop deadline",
        f"site:psu.edu {term_txt} {year_txt} late drop deadline",
        f"site:pennstate.edu academic calendar {term_txt} {year_txt} late drop",
    ]
    if "abington" in context_q.lower():
        queries.insert(1, f"Penn State Abington {term_txt} {year_txt} late drop deadline")
    return queries


def _build_identity_search_queries(user_question: str) -> List[str]:
    q = (user_question or "").lower()
    keyword_q = _keywordize_query(user_question)
    scope_abington = "abington" in q
    base = []
    if scope_abington:
        base.extend([
            f"site:abington.psu.edu {keyword_q}",
            f"Penn State Abington {keyword_q}",
        ])
    base.extend([
        f"site:psu.edu {keyword_q}",
        f"Penn State {keyword_q}",
    ])
    if "abington" in q:
        base.insert(0, f"Penn State Abington {keyword_q}")
    if "chancellor" in q:
        if scope_abington:
            base.insert(0, "site:abington.psu.edu chancellor")
        else:
            base.insert(0, "site:psu.edu leadership")
            base.insert(1, "Penn State leadership")
    if "program chair" in q or "chair" in q:
        base.insert(0, "site:abington.psu.edu computer science program chair")
        base.insert(1, "site:abington.psu.edu computer science faculty")
    if "provost" in q:
        base.insert(0, "site:psu.edu provost")
        base.insert(1, "Penn State provost")
    # Deduplicate while preserving order
    seen = set()
    out = []
    for b in base:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out[:6]


def _extract_domain_hints(question: str) -> List[str]:
    q = (question or "").lower()
    hints: List[str] = []
    if any(k in q for k in ["penn state", "psu"]):
        hints.extend(["psu.edu", "pennstate.edu"])
    # Default to PSU domains for in-scope academic advising questions in this app.
    if not hints and is_academic_or_policy_question(question):
        hints.extend(["psu.edu", "pennstate.edu"])
    return hints


def _domain_from_url(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _is_domain_allowed(url: str, domain_hints: List[str]) -> bool:
    if not domain_hints:
        return True
    d = _domain_from_url(url)
    return any(d.endswith(h) for h in domain_hints)


def _keywordize_query(query: str) -> str:
    q = (query or "").lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    stop = {
        "who", "what", "when", "where", "why", "how", "is", "are", "the", "a", "an",
        "at", "of", "for", "to", "current", "currently", "please", "me", "tell",
        "then", "uh", "um", "about",
    }
    toks = [t for t in q.split() if t and t not in stop]
    return " ".join(toks[:10]).strip() or q.strip()


def _normalize_question_for_search(question: str) -> str:
    q = (question or "").strip().lower()
    # Fix common typo/extra-token patterns that hurt retrieval quality.
    q = re.sub(r"\bwho\s+is\s+then\b", "who is the", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _expand_academic_query_terms(question: str) -> str:
    q = _normalize_question_for_search(question)
    q = re.sub(r"\bcs\b", "computer science", q, flags=re.IGNORECASE)
    q = re.sub(r"\bclasses\b", "courses", q, flags=re.IGNORECASE)
    if is_academic_or_policy_question(q) and "penn state" not in q and "psu" not in q:
        q = f"penn state {q}"
    return " ".join(q.split()).strip()


def _relax_search_query(query: str) -> str:
    """Relax strict operators/phrasing to improve hit rate."""
    q = (query or "").strip()
    q = re.sub(r"\bsite:[^\s]+\s*", "", q, flags=re.IGNORECASE)
    q = q.replace('"', '').replace("  ", " ").strip()
    return q


def _build_general_search_queries(user_question: str) -> List[str]:
    q = (user_question or "").strip()
    if not q:
        return []
    lower_q = q.lower()
    keyword_q = _keywordize_query(q)
    variants = [
        q,
        keyword_q,
        f"{q} official source",
        f"{q} latest update",
    ]
    if "penn state" in lower_q or "psu" in lower_q or "abington" in lower_q:
        variants.extend([
            f"Penn State {q}",
            f"site:psu.edu {q}",
            f"site:pennstate.edu {q}",
        ])

    out: List[str] = []
    seen = set()
    for v in variants:
        vv = " ".join(v.split()).strip()
        if vv and vv not in seen:
            seen.add(vv)
            out.append(vv)
    return out[:8]


def _run_multi_query_web_search(
    query_variants: List[str],
    trace: Optional[Dict[str, Any]] = None,
    max_per_query: int = 5,
    domain_hints: Optional[List[str]] = None,
    require_domain_match: bool = False
) -> List[dict]:
    """Universal retrieval: strict queries first, then relaxed fallback."""
    from duckduckgo_search import DDGS

    if domain_hints is None:
        domain_hints = []
    gathered: List[dict] = []
    seen = set()

    def _collect(query: str) -> int:
        hits = 0
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_per_query))
        _trace_event(trace, "execute", "Executed web query", {"query": query, "results": len(results)})
        for r in results:
            href = str(r.get("href", "") or "")
            key = href or f"{r.get('title','')}-{r.get('body','')[:120]}"
            if key in seen:
                continue
            seen.add(key)
            gathered.append(r)
            hits += 1
        return hits

    # Pass 1: strict/original variants
    for q in query_variants[:8]:
        try:
            _collect(q)
        except Exception as e:
            _trace_event(trace, "execute", "Web query failed", {"query": q, "error": str(e)})

    # Pass 2: relaxed variants if we got nothing
    if len(gathered) == 0:
        relaxed = []
        for q in query_variants[:8]:
            rq = _relax_search_query(q)
            if rq and rq not in relaxed:
                relaxed.append(rq)
        for q in relaxed[:8]:
            try:
                _collect(q)
            except Exception as e:
                _trace_event(trace, "execute", "Relaxed web query failed", {"query": q, "error": str(e)})

    # Domain filtering (authoritative-source preference/requirement)
    if domain_hints:
        filtered = [r for r in gathered if _is_domain_allowed(str(r.get("href", "") or ""), domain_hints)]
        _trace_event(
            trace,
            "execute",
            "Applied domain filtering",
            {
                "domain_hints": domain_hints,
                "before": len(gathered),
                "after": len(filtered),
                "required": require_domain_match
            }
        )
        if filtered:
            return filtered
        if require_domain_match:
            return []

    return gathered


def _strip_html_to_text(html: str) -> str:
    if not html:
        return ""
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_candidate_links(base_url: str, html: str) -> List[str]:
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html or "", flags=re.IGNORECASE)
    base_domain = _domain_from_url(base_url)
    candidates: List[str] = []
    for h in hrefs:
        if not h or h.startswith("#") or h.startswith("mailto:") or h.startswith("tel:"):
            continue
        full = urljoin(base_url, h)
        if not full.startswith("http"):
            continue
        if _domain_from_url(full) != base_domain:
            continue
        # Skip static assets and noisy non-content URLs.
        low = full.lower()
        if any(low.endswith(ext) for ext in [".css", ".js", ".ico", ".png", ".jpg", ".jpeg", ".svg", ".woff", ".woff2", ".ttf", ".pdf"]):
            continue
        if any(tok in low for tok in ["/sites/", "/themes/", "/files/css/", "/files/js/"]):
            continue
        candidates.append(full)
    # Prefer leadership/administration-relevant paths
    priority_tokens = ["leadership", "administration", "about", "chancellor", "provost", "faculty", "staff", "directory"]
    candidates = sorted(
        list(dict.fromkeys(candidates)),
        key=lambda u: sum(1 for t in priority_tokens if t in u.lower()),
        reverse=True,
    )
    return candidates[:20]


def _extract_role_keywords(question: str) -> List[str]:
    q = (question or "").lower()
    kws: List[str] = []
    for k in ["chancellor", "provost", "president", "program chair", "chair", "dean", "director"]:
        if k in q:
            kws.append(k)
    return kws or ["leadership"]


def _is_plausible_person_name(name: str) -> bool:
    if not name:
        return False
    parts = [p for p in re.split(r"\s+", name.strip()) if p]
    if len(parts) < 2:
        return False
    bad_tokens = {
        "development", "office", "staff", "administration", "admissions",
        "directory", "results", "campus", "leadership", "university",
        "department", "college", "program", "penn", "state", "abington"
    }
    first = parts[0].lower().strip(".,")
    second = parts[1].lower().strip(".,")
    if first in bad_tokens or second in bad_tokens:
        return False
    # Must look like person-case words (allow apostrophes/hyphens)
    token_ok = all(re.match(r"^[A-Z][a-zA-Z'\-]+\.?$", p) for p in parts[:2])
    return bool(token_ok)


def _extract_role_holder_candidates(text: str, role_keywords: List[str]) -> List[str]:
    """Extract candidate role-holder names from role-aware text patterns."""
    if not text:
        return []
    candidates: List[str] = []
    role_alt = "|".join(re.escape(k) for k in role_keywords if k)

    patterns = [
        # Chancellor Dr. Jane Doe
        rf"\b(?:{role_alt})\b[^A-Za-z]{{0,20}}(Dr\.\s+[A-Z][a-zA-Z'\-]+\s+[A-Z][a-zA-Z'\-]+)",
        # Chancellor Jane Doe
        rf"\b(?:{role_alt})\b[^A-Za-z]{{0,20}}([A-Z][a-zA-Z'\-]+\s+[A-Z][a-zA-Z'\-]+)",
        # Jane Doe, Chancellor
        rf"\b([A-Z][a-zA-Z'\-]+\s+[A-Z][a-zA-Z'\-]+)\b[^A-Za-z]{{0,20}}\b(?:{role_alt})\b",
    ]

    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            cand = m.group(1).strip()
            # normalize "dr." casing
            cand = re.sub(r"^dr\.\s+", "Dr. ", cand, flags=re.IGNORECASE)
            cl = cand.lower()
            if any(rk in cl for rk in ["president", "chancellor", "provost", "dean", "director", "chair"]):
                continue
            if _is_plausible_person_name(cand) and cand not in candidates:
                candidates.append(cand)
    return candidates


def _fetch_official_role_holder(
    question: str,
    domain_hints: List[str],
    trace: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Deterministic fallback: scrape known official leadership pages and extract likely role holder.
    """
    q = _normalize_question_for_search(question)
    role_keywords = _extract_role_keywords(q)
    scope_abington = "abington" in q
    candidate_urls = []
    if scope_abington:
        candidate_urls.extend([
            "https://www.abington.psu.edu/",
            "https://www.abington.psu.edu/about",
            "https://www.abington.psu.edu/about/administration",
            "https://www.abington.psu.edu/faculty-and-staff",
        ])
    else:
        candidate_urls.extend([
            "https://www.psu.edu/",
            "https://www.psu.edu/about/leadership/",
            "https://www.psu.edu/about/administration/",
        ])
    if "provost" in q:
        candidate_urls.insert(0, "https://www.psu.edu/about/administration/")
    if "chancellor" in q:
        if scope_abington:
            candidate_urls.insert(0, "https://www.abington.psu.edu/about/leadership")
        else:
            candidate_urls.insert(0, "https://www.psu.edu/about/leadership/")

    for url in candidate_urls:
        if domain_hints and not _is_domain_allowed(url, domain_hints):
            continue
        try:
            _trace_tool(trace, "requests_get", "official page fallback", metadata={"url": url})
            resp = requests.get(url, timeout=7, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                _trace_event(trace, "execute", "Official page fetch non-200", {"url": url, "status": resp.status_code})
                continue
            pages_to_scan = [(url, resp.text)]
            for link in _extract_candidate_links(url, resp.text)[:8]:
                try:
                    _trace_tool(trace, "requests_get", "official link follow", metadata={"url": link})
                    lr = requests.get(link, timeout=7, headers={"User-Agent": "Mozilla/5.0"})
                    if lr.status_code == 200:
                        pages_to_scan.append((link, lr.text))
                except Exception:
                    continue

            for page_url, html in pages_to_scan:
                text = _strip_html_to_text(html)
                low = text.lower()
                if scope_abington and "abington" not in low:
                    continue
                if not any(k in low for k in role_keywords):
                    continue
                # Role-aware extraction first (higher precision), then fallback.
                candidates = _extract_role_holder_candidates(text, role_keywords)
                if candidates:
                    return f"The current office holder appears to be {candidates[0]}.\nSource: {page_url}"

                # Fallback: search around role windows for plausible person names.
                for k in role_keywords:
                    for m in re.finditer(re.escape(k), low):
                        start = max(0, m.start() - 180)
                        end = min(len(text), m.end() + 220)
                        window = text[start:end]
                        win_candidates = _extract_role_holder_candidates(window, [k])
                        if win_candidates:
                            return f"The current office holder appears to be {win_candidates[0]}.\nSource: {page_url}"
        except Exception as e:
            _trace_event(trace, "execute", "Official page fetch failed", {"url": url, "error": str(e)})
            continue
    return None


def _extract_person_name_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    # Prefer "Dr. First Last" style names
    m = re.search(r"\b(Dr\.\s+[A-Z][a-zA-Z'-]+\s+[A-Z][a-zA-Z'-]+)\b", text)
    if m:
        return m.group(1).strip()
    # Fallback: First Last (two capitalized words) with optional middle initial
    m2 = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z]\.)?\s+[A-Z][a-z]+)\b", text)
    if m2:
        candidate = m2.group(1).strip()
        bad_starts = {
            "as", "the", "this", "that", "for", "from", "with", "in", "on",
            "staff", "office", "penn", "state", "chancellor", "provost",
            "dean", "director", "president"
        }
        first = candidate.split()[0].lower() if candidate.split() else ""
        if candidate.lower() not in {"penn state", "state university"} and first not in bad_starts:
            return candidate
    return None


def _rank_identity_result(result: dict, role_keywords: List[str]) -> int:
    title = str(result.get("title", "") or "")
    body = str(result.get("body", "") or "")
    href = str(result.get("href", "") or "").lower()
    text = f"{title}\n{body}".lower()
    score = 0
    if "abington.psu.edu" in href:
        score += 6
    elif "psu.edu" in href or "pennstate.edu" in href:
        score += 4
    if any(k in text for k in role_keywords):
        score += 4
    if "directory" in href or "faculty" in href or "leadership" in href or "administration" in href:
        score += 2
    if "linkedin" in href or "facebook" in href:
        score -= 3
    return score


def _rank_deadline_date_candidates(
    results: List[dict],
    target_year: Optional[int] = None,
    prefer_current_or_future: bool = True,
    strict_target_year: bool = False,
    strict_current_or_future: bool = False
) -> List[Tuple[datetime, int]]:
    """Extract and score date candidates from search results."""
    scored: List[Tuple[datetime, int]] = []
    now = datetime.now()

    for r in results:
        title = str(r.get("title", "") or "")
        body = str(r.get("body", "") or "")
        href = str(r.get("href", "") or "")
        snippet = f"{title}\n{body}"
        snippet_lower = snippet.lower()
        has_deadline_terms = any(k in snippet_lower for k in ["late drop", "drop deadline", "withdraw", "deadline", "drop"])
        is_calendar_like = any(k in href.lower() for k in ["calendar", "academic", "registrar", "deadline", "lionpath"])

        for d in _extract_dates_from_text(snippet):
            dt = _parse_date_candidate(d, fallback_year=target_year or now.year)
            if not dt:
                continue

            score = 0
            if "psu.edu" in href or "pennstate.edu" in href:
                score += 3
            if has_deadline_terms:
                score += 2
            if "late drop" in snippet_lower:
                score += 3
            if is_calendar_like:
                score += 2
            if target_year is not None:
                if strict_target_year and dt.year != target_year:
                    continue
                score += 3 if dt.year == target_year else -3
            if prefer_current_or_future:
                if strict_current_or_future and dt.date() < now.date():
                    continue
                score += 2 if dt.date() >= now.date() else -4

            # Penalize clearly stale years unless user explicitly asked for them
            if target_year is None and dt.year < now.year:
                score -= 3

            scored.append((dt, score))

    scored.sort(key=lambda x: (x[1], -abs((x[0].date() - datetime.now().date()).days)), reverse=True)
    return scored


def _web_search_and_synthesize(
    user_question: str,
    date_only: bool = False,
    conversation_history: Optional[List[dict]] = None,
    prefer_current_or_future: bool = False,
    trace: Optional[Dict[str, Any]] = None,
    force_fresh_identity: bool = False
) -> Optional[str]:
    """
    Run web search for Penn State / academic questions when our data doesn't have the answer.
    Returns synthesized response or None if search fails.
    """
    global llm2
    try:
        normalized_question = _normalize_question_for_search(user_question)
        domain_hints = _extract_domain_hints(normalized_question)
        _trace_tool(
            trace,
            "duckduckgo_search",
            "web retrieval",
            metadata={"date_only": date_only, "prefer_current_or_future": prefer_current_or_future}
        )
        if date_only:
            term, target_year = _infer_term_year_from_context(normalized_question, conversation_history)
            queries = _build_deadline_search_queries(normalized_question, conversation_history)
            ql = (normalized_question or "").lower()
            strict_target_year = target_year is not None and any(
                k in ql for k in ["current semester", "this semester", "next semester", "upcoming semester", "spring", "summer", "fall", "winter"]
            )
            strict_current_or_future = any(
                k in ql for k in ["current semester", "this semester", "next semester", "upcoming semester"]
            )
            _trace_event(
                trace,
                "execute",
                "Built targeted date queries",
                {
                    "term": term,
                    "target_year": target_year,
                    "query_count": len(queries),
                    "strict_target_year": strict_target_year,
                    "strict_current_or_future": strict_current_or_future,
                }
            )
            all_results = _run_multi_query_web_search(
                queries[:4],
                trace=trace,
                max_per_query=4,
                domain_hints=domain_hints,
                require_domain_match=bool(domain_hints)
            )

            if not all_results:
                _trace_quality_check(trace, "web_results_non_empty", False, {"mode": "date_only"})
                return "DATE_NOT_FOUND"

            ranked = _rank_deadline_date_candidates(
                all_results,
                target_year=target_year,
                prefer_current_or_future=prefer_current_or_future,
                strict_target_year=strict_target_year,
                strict_current_or_future=strict_current_or_future
            )
            _trace_quality_check(
                trace,
                "date_candidates_found",
                len(ranked) > 0,
                {"candidates": len(ranked), "target_year": target_year}
            )
            if ranked:
                best = ranked[0][0]
                return best.strftime("%B %d, %Y")

            # Fallback: let LLM pick a concrete late-drop date from snippets when heuristic ranking is inconclusive.
            search_context = "\n\n".join(
                f"[{i+1}] {r.get('title', '')}\n{r.get('body', '')}\nSource: {r.get('href', '')}"
                for i, r in enumerate(all_results[:8])
            )
            extracted = llm2.invoke([
                SystemMessage(content="""Extract the exact late-drop deadline date from the provided search snippets.
Rules:
- Return ONLY one date in format: Month DD, YYYY
- The date must match the asked term/year if provided
- Prefer Penn State official sources
- If uncertain, return exactly: DATE_NOT_FOUND"""),
                HumanMessage(content=f"Question: {user_question}\n\nSearch snippets:\n{search_context}")
            ]).content.strip()
            if extracted and extracted.lower() != "date_not_found":
                parsed = _parse_date_candidate(extracted, fallback_year=target_year or datetime.now().year)
                if parsed:
                    return parsed.strftime("%B %d, %Y")
            return "DATE_NOT_FOUND"

        if force_fresh_identity:
            role_keywords = ["chancellor", "program chair", "chair", "computer science", "leadership", "provost"]
            queries = _build_identity_search_queries(normalized_question)
            gathered: List[dict] = []
            gathered = _run_multi_query_web_search(
                queries,
                trace=trace,
                max_per_query=5,
                domain_hints=domain_hints,
                require_domain_match=bool(domain_hints)
            )

            if not gathered:
                # Deterministic fallback to official leadership pages.
                direct = _fetch_official_role_holder(normalized_question, domain_hints, trace=trace)
                if direct:
                    return direct
                _trace_quality_check(trace, "web_results_non_empty", False, {"mode": "identity"})
                return "I couldn't verify the current office holder.\nSource: unavailable"

            ranked_results = sorted(
                gathered,
                key=lambda r: _rank_identity_result(r, role_keywords),
                reverse=True
            )
            top = ranked_results[:8]
            _trace_quality_check(trace, "identity_results_ranked", len(top) > 0, {"candidates": len(gathered)})

            # Deterministic pass: extract name from top result snippets first.
            for r in top[:4]:
                text = f"{r.get('title', '')}. {r.get('body', '')}"
                names = _extract_role_holder_candidates(text, role_keywords)
                if names:
                    name = names[0]
                    src = str(r.get("href", "") or "unavailable")
                    return f"The current office holder appears to be {name}.\nSource: {src}"

            search_context = "\n\n".join(
                f"[{i+1}] {r.get('title', '')}\n{r.get('body', '')}\nSource: {r.get('href', '')}"
                for i, r in enumerate(top)
            )
            response = llm2.invoke([
                SystemMessage(content="""You are verifying a potentially time-sensitive staff role answer.
Use ONLY the web results. Prefer official Penn State pages.
Return:
1) The best current answer in one sentence.
2) A 'Source:' line with the strongest URL from the results.
If not verifiable, return:
I couldn't verify the current office holder.
Source: <best available URL or unavailable>"""),
                HumanMessage(content=f"Student question: {normalized_question}\n\nWeb search results:\n{search_context}")
            ]).content
            if "source:" not in response.lower():
                best_url = str(top[0].get("href", "") or "unavailable")
                response = response.strip() + f"\nSource: {best_url}"
            return response

        query_variants = _build_general_search_queries(normalized_question)
        results = _run_multi_query_web_search(
            query_variants,
            trace=trace,
            max_per_query=5,
            domain_hints=domain_hints,
            require_domain_match=False
        )
        if not results:
            _trace_quality_check(trace, "web_results_non_empty", False, {"mode": "generic"})
            return None

        search_context = "\n\n".join(
            f"[{i+1}] {r.get('title', '')}\n{r.get('body', '')}\nSource: {r.get('href', '')}"
            for i, r in enumerate(results[:5])
        )

        response = llm2.invoke([
            SystemMessage(content="""You are an academic advisor. The student asked a question that wasn't in our local policy documents.
Use the web search results below to answer their question. Cite that this is from a web search.
Be concise and accurate. If the search results don't contain a clear answer, say so.
Focus on Penn State / academic policy information."""),
            HumanMessage(content=f"Student question: {normalized_question}\n\nWeb search results:\n{search_context}\n\nProvide a helpful answer:")
        ]).content
        return response
    except Exception as e:
        print(f"⚠️ Web search failed: {e}")
        _trace_tool(trace, "duckduckgo_search", "web retrieval", status="failed", metadata={"error": str(e)})
        return None


def _observe_and_repair_response(
    user_question: str,
    response: str,
    reasoning_plan: Dict[str, object],
    query_type: str,
    conversation_history: Optional[List[dict]] = None,
    trace: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Universal observe/repair stage for all query types:
      - Validate whether response satisfies plan intent.
      - Attempt one repair pass using available tools.
    """
    candidate = (response or "").strip()
    if not candidate:
        candidate = "I couldn't produce a clear answer. Please rephrase your question."

    issues: List[str] = []
    contextual_question = str(reasoning_plan.get("contextual_question") or user_question)
    tools = reasoning_plan.get("tools", {}) or {}
    wants_web = bool(tools.get("web_search"))
    execution_profile = reasoning_plan.get("execution_profile", {}) or {}
    verification_mode = str(execution_profile.get("verification_mode", "none"))
    requires_source = bool(execution_profile.get("must_include_source", False))
    identity_like = bool(execution_profile.get("identity_like", False))
    source_domain_hints = execution_profile.get("source_domain_hints", []) or []

    if _response_indicates_no_answer_found(candidate) and wants_web and is_academic_or_policy_question(contextual_question):
        issues.append("no_answer")

    if verification_mode == "date":
        dates = _extract_dates_from_text(candidate)
        if not dates:
            issues.append("missing_date")
        else:
            term, target_year = _infer_term_year_from_context(contextual_question, conversation_history)
            parsed = _parse_date_candidate(dates[0], fallback_year=target_year or datetime.now().year)
            cql = contextual_question.lower()
            current_or_future_requested = any(t in cql for t in ["current semester", "this semester", "next semester", "upcoming"])
            if parsed and current_or_future_requested and parsed.date() < (datetime.now().date() - timedelta(days=120)):
                issues.append("stale_date")

    if reasoning_plan.get("intent") == "recommendation":
        horizon = int(reasoning_plan.get("semester_horizon", 1) or 1)
        if horizon >= 2 and "semester +2" not in candidate.lower():
            issues.append("missing_horizon")

    if requires_source:
        web_used = False
        if trace and trace.get("tools_used"):
            web_used = any(str(t.get("tool", "")).lower() == "duckduckgo_search" for t in trace.get("tools_used", []))
        if not web_used:
            issues.append("unverified_fresh_fact")
        if "source:" not in candidate.lower() and "unavailable" not in candidate.lower():
            issues.append("missing_source")
        elif "unavailable" not in candidate.lower() and not _response_has_allowed_source(candidate, source_domain_hints):
            issues.append("untrusted_source")
        m_name = re.search(r"appears to be\s+([^\n\.]+)", candidate, flags=re.IGNORECASE)
        if m_name:
            extracted_name = m_name.group(1).strip()
            if not _is_plausible_person_name(extracted_name):
                issues.append("low_confidence_name")

    _trace_quality_check(
        trace,
        "response_satisfies_plan",
        len(issues) == 0,
        {
            "issues": issues,
            "intent": reasoning_plan.get("intent"),
            "query_type": query_type,
        }
    )

    if not issues:
        return candidate

    if trace and trace.get("enabled"):
        trace["repairs"].append({
            "ts": datetime.now().isoformat(),
            "issues": issues,
            "attempt": "repair_pass_1",
        })

    # Repair path 1: web search backfill for missing/no-answer policy/course/major/chat contexts.
    if wants_web and is_academic_or_policy_question(contextual_question):
        if verification_mode == "date" and any(i in issues for i in ["missing_date", "stale_date", "no_answer"]):
            web = _web_search_and_synthesize(
                contextual_question,
                date_only=True,
                conversation_history=conversation_history,
                prefer_current_or_future=True,
                trace=trace
            )
            if web and web != "DATE_NOT_FOUND":
                _trace_event(trace, "observe", "Repair succeeded with date web search", {"issues": issues})
                return web
        if "no_answer" in issues:
            web = _web_search_and_synthesize(contextual_question, conversation_history=conversation_history, trace=trace)
            if web:
                _trace_event(trace, "observe", "Repair succeeded with generic web search", {"issues": issues})
                return web
        if any(i in issues for i in ["unverified_fresh_fact", "missing_source", "untrusted_source", "low_confidence_name"]):
            web = _web_search_and_synthesize(
                contextual_question,
                conversation_history=conversation_history,
                trace=trace,
                force_fresh_identity=identity_like
            )
            if web:
                _trace_event(trace, "observe", "Repair succeeded with fresh-fact verification web search", {"issues": issues})
                return web

    # Repair path 2: recommendation horizon mismatch fallback.
    if "missing_horizon" in issues and reasoning_plan.get("intent") == "recommendation":
        _trace_event(trace, "observe", "Applied recommendation horizon fallback", {"issues": issues})
        return candidate + "\n\nI can also provide Semester +2 explicitly if you want me to regenerate with a 2-semester horizon."

    return candidate


def _term_or_year_was_provided(text: str, conversation_history: Optional[List[dict]] = None) -> bool:
    term, year = _infer_term_year_from_context(text, conversation_history)
    q = (text or "").lower()
    has_explicit_term = any(t in q for t in ["spring", "summer", "fall", "winter", "current semester", "this semester", "next semester", "upcoming semester"])
    has_explicit_year = re.search(r"\b20\d{2}\b", q) is not None
    return has_explicit_term or has_explicit_year or (term is not None and year is not None)


def _format_deadline_answer(date_text: str, question_text: str, conversation_history: Optional[List[dict]] = None) -> str:
    """Return a short, contextual deadline answer with term/year."""
    term, year = _infer_term_year_from_context(question_text, conversation_history)
    term_label = f"{(term or 'requested term').title()} {year}".strip()
    parsed = _parse_date_candidate(date_text, fallback_year=year or datetime.now().year)
    normalized = parsed.strftime("%B %d, %Y") if parsed else date_text
    return f"Late-drop deadline ({term_label}): {normalized}."


def _extract_source_urls_from_response(response: str) -> List[str]:
    if not response:
        return []
    return re.findall(r"https?://[^\s)]+", response)


def _response_has_allowed_source(response: str, domain_hints: List[str]) -> bool:
    if not domain_hints:
        return True
    urls = _extract_source_urls_from_response(response)
    if not urls:
        return False
    return any(_is_domain_allowed(u, domain_hints) for u in urls)

def track_collection_creation(collection_name: str, user_id: str):
    """Track when collection was created for TTL cleanup."""
    metadata = {}
    if os.path.exists(COLLECTION_METADATA_FILE):
        try:
            with open(COLLECTION_METADATA_FILE, 'r') as f:
                metadata = json.load(f)
        except:
            metadata = {}
    
    metadata[collection_name] = {
        "user_id": user_id,
        "created_at": datetime.now().isoformat()
    }
    
    with open(COLLECTION_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def cleanup_old_collections(ttl_days=7):
    """Remove transcript collections older than TTL."""
    if not os.path.exists(COLLECTION_METADATA_FILE):
        return
    
    try:
        with open(COLLECTION_METADATA_FILE, 'r') as f:
            metadata = json.load(f)
    except:
        return
    
    cutoff = datetime.now() - timedelta(days=ttl_days)
    to_remove = []
    
    for collection_name, info in metadata.items():
        try:
            created_at = datetime.fromisoformat(info['created_at'])
            if created_at < cutoff:
                to_remove.append(collection_name)
                # Delete collection directory
                collection_path = os.path.join(CHROMA_DATA_DIR, "transcripts", collection_name)
                if os.path.exists(collection_path):
                    shutil.rmtree(collection_path)
        except:
            continue
    
    # Update metadata
    for name in to_remove:
        del metadata[name]
    
    with open(COLLECTION_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if to_remove:
        print(f"🧹 Cleaned up {len(to_remove)} old collections")


def _is_debug_enabled(flag: Any) -> bool:
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, str):
        return flag.strip().lower() in {"1", "true", "yes", "on"}
    return bool(flag)


def _new_execution_trace(user_id: str, user_question: str, enabled: bool) -> Dict[str, Any]:
    return {
        "enabled": enabled,
        "user_id": user_id,
        "user_question": user_question,
        "started_at": datetime.now().isoformat(),
        "plan": {},
        "events": [],
        "tools_used": [],
        "quality_checks": [],
        "repairs": [],
        "result_preview": "",
        "finished_at": None,
    }


def _trace_event(trace: Optional[Dict[str, Any]], stage: str, message: str, data: Optional[Dict[str, Any]] = None):
    if not trace or not trace.get("enabled"):
        return
    trace["events"].append({
        "ts": datetime.now().isoformat(),
        "stage": stage,
        "message": message,
        "data": data or {},
    })


def _trace_tool(
    trace: Optional[Dict[str, Any]],
    tool_name: str,
    purpose: str,
    status: str = "used",
    metadata: Optional[Dict[str, Any]] = None,
):
    if not trace or not trace.get("enabled"):
        return
    trace["tools_used"].append({
        "ts": datetime.now().isoformat(),
        "tool": tool_name,
        "purpose": purpose,
        "status": status,
        "metadata": metadata or {},
    })


def _trace_quality_check(
    trace: Optional[Dict[str, Any]],
    check_name: str,
    passed: bool,
    details: Optional[Dict[str, Any]] = None,
):
    if not trace or not trace.get("enabled"):
        return
    trace["quality_checks"].append({
        "ts": datetime.now().isoformat(),
        "check": check_name,
        "passed": passed,
        "details": details or {},
    })


def get_execution_trace_for_user(user_id: str) -> Optional[Dict[str, Any]]:
    return execution_trace_store.get(user_id)


class CachedEmbeddings:
    """Wrapper around OpenAIEmbeddings that caches results."""
    
    def __init__(self, embedding_fn, cache_dir=None):
        self.embedding_fn = embedding_fn
        if cache_dir is None:
            cache_dir = os.path.join(CHROMA_DATA_DIR, "embedding_cache")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, text: str) -> str:
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{text_hash}.pkl")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with caching."""
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        for i, text in enumerate(texts):
            cache_path = self._get_cache_path(text)
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        embeddings.append((i, pickle.load(f)))
                except:
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # Embed uncached texts
        if texts_to_embed:
            new_embeddings = self.embedding_fn.embed_documents(texts_to_embed)
            for idx, text, emb in zip(indices_to_embed, texts_to_embed, new_embeddings):
                cache_path = self._get_cache_path(text)
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(emb, f)
                except:
                    pass
                embeddings.append((idx, emb))
        
        # Return in original order
        return [emb for _, emb in sorted(embeddings)]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query with caching."""
        cache_path = self._get_cache_path(text)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        
        embedding = self.embedding_fn.embed_query(text)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except:
            pass
        return embedding

# =========================
# Environment / Setup
# =========================

def setup_environment(whatIfUpload=None, user_id='default'):
    """
    Initialize environment, LLMs, vectorstores for PDF + courses.
    If whatIfUpload is None or has no 'transcript', run in no-transcript mode.
    """
    global pdf_retriever, course_retriever, majors_retriever, llm1, llm2, llm_classifier, has_transcript, stored_pdf_docs, latest_student_data

    has_transcript = False  # reset flag each call
    
    # Ensure Chroma data directory exists
    os.makedirs(CHROMA_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(CHROMA_DATA_DIR, "courses"), exist_ok=True)
    os.makedirs(os.path.join(CHROMA_DATA_DIR, "transcripts"), exist_ok=True)

    #  Environment Variables
    LANGCHAIN_PROJECT = "pr-notable-garbage-99"
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = 'REDACTED'
    os.environ['OPENAI_API_KEY'] = "REDACTED"

    if not os.getenv('LANGCHAIN_API_KEY'):
        print("⚠️ LANGCHAIN_API_KEY is not set in the environment.")
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️ OPENAI_API_KEY is not set in the environment.")

    # ---- Optional transcript / What-If report ----
    if whatIfUpload is not None and 'transcript' in whatIfUpload.files:
        transcript_file = whatIfUpload.files['transcript']
        filename = secure_filename(transcript_file.filename)
        temp_pdf_path = f"/tmp/{filename}"
        transcript_file.save(temp_pdf_path)

        # Generate document hash
        doc_hash = get_document_hash(temp_pdf_path)
        collection_name = get_transcript_collection_name(user_id, doc_hash)
        collection_path = os.path.join(CHROMA_DATA_DIR, "transcripts", collection_name)
        
        # Check if collection exists (cached)
        if os.path.exists(collection_path):
            try:
                cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
                vectorstore_pdf = Chroma(
                    collection_name=collection_name,
                    persist_directory=os.path.join(CHROMA_DATA_DIR, "transcripts"),
                    embedding_function=cached_embeddings
                )
                pdf_retriever = vectorstore_pdf.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 10, "fetch_k": 40, "lambda_mult": 0.5}
                )
                # For cached transcripts, reload PDF for deterministic extraction
                loader = PyPDFLoader(temp_pdf_path)
                stored_pdf_docs = redact_student_info(loader.load())
                try:
                    latest_student_data = extract_student_data_deterministic(stored_pdf_docs)
                except Exception as e:
                    print(f"⚠️ Failed to cache student data from cached transcript: {e}")
                has_transcript = True
                print(f"✅ Using cached transcript: {collection_name}")
            except Exception as e:
                print(f"⚠️ Error loading cached transcript: {e}, recreating...")
                # Fall through to create new
                vectorstore_pdf = None
        else:
            vectorstore_pdf = None
        
        # Create new collection if not cached
        if vectorstore_pdf is None:
            loader = PyPDFLoader(temp_pdf_path)
            pdf_docs = redact_student_info(loader.load())
            stored_pdf_docs = pdf_docs  # Store for deterministic extraction
            try:
                latest_student_data = extract_student_data_deterministic(pdf_docs)
            except Exception as e:
                print(f"⚠️ Failed to cache student data on upload: {e}")
            pdf_splits = chunk_with_metadata(pdf_docs)

            # Use cached embeddings
            cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
            vectorstore_pdf = Chroma.from_documents(
                pdf_splits,
                collection_name=collection_name,
                persist_directory=os.path.join(CHROMA_DATA_DIR, "transcripts"),
                embedding=cached_embeddings
            )

            # Track collection creation for TTL cleanup
            track_collection_creation(collection_name, user_id)

            pdf_retriever = vectorstore_pdf.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 10, "fetch_k": 40, "lambda_mult": 0.5}
            )
            print(f"✅ Transcript loaded and vectorstore built: {collection_name}")
        has_transcript = True
    else:
        pdf_retriever = None
        stored_pdf_docs = None
        has_transcript = False
        print("ℹ️ No transcript provided. Running in no-transcript mode.")

    # ---- Load course-level data (processed_psu_courses.csv) for individual course queries ----
    courses_csv_path = os.path.join(os.path.dirname(__file__), "processed_psu_courses.csv")
    if not os.path.exists(courses_csv_path):
        alt_path = os.path.join(os.path.dirname(__file__), "..", "LLM", "processed_psu_courses.csv")
        if os.path.exists(alt_path):
            courses_csv_path = alt_path
        else:
            print("⚠️ processed_psu_courses.csv not found")
            courses_csv_path = None
    
    if courses_csv_path and os.path.exists(courses_csv_path):
        df_courses = pd.read_csv(courses_csv_path)
        course_texts = []
        course_metadata = []

        for _, row in df_courses.iterrows():
            course_text = f"Course: {row.get('course_number', '')}. "
            course_text += f"Title: {row.get('course_name', '')}. "
            course_text += f"Description: {row.get('description', '')}. "
            prereq_str = str(row.get('prerequisite_course_numbers', ''))
            if prereq_str and prereq_str != 'nan':
                course_text += f"Prerequisites: {prereq_str}. "
            course_texts.append(course_text)
            
            course_code = str(row.get('course_number', '')).strip()
            course_metadata.append({
                "course_code": course_code,
                "department": course_code.split()[0] if ' ' in course_code else ''
            })
        
        chroma_courses_dir = os.path.join(CHROMA_DATA_DIR, "courses")
        try:
            if os.path.exists(chroma_courses_dir):
                try:
                    cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
                    vectorstore_courses = Chroma(
                        persist_directory=chroma_courses_dir,
                        embedding_function=cached_embeddings
                    )
                    _ = vectorstore_courses.as_retriever(search_kwargs={"k": 1})
                    print("✅ Using existing course-level vectorstore.")
                except Exception as e:
                    print(f"⚠️ Course vectorstore corrupted, recreating... Error: {e}")
                    if os.path.exists(chroma_courses_dir):
                        shutil.rmtree(chroma_courses_dir)
                    cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
                    vectorstore_courses = Chroma.from_texts(
                        texts=course_texts,
                        embedding=cached_embeddings,
                        metadatas=course_metadata,
                        persist_directory=chroma_courses_dir
                    )
                    vectorstore_courses.persist()
                    print("✅ Created new course-level vectorstore.")
            else:
                cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
                vectorstore_courses = Chroma.from_texts(
                    texts=course_texts,
                    embedding=cached_embeddings,
                    metadatas=course_metadata,
                    persist_directory=chroma_courses_dir
                )
                vectorstore_courses.persist()
                print("✅ Created new course-level vectorstore.")
        except Exception as e:
            print(f"⚠️ Error with course vectorstore: {e}")
            if os.path.exists(chroma_courses_dir):
                shutil.rmtree(chroma_courses_dir)
            cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
            vectorstore_courses = Chroma.from_texts(
                texts=course_texts,
                embedding=cached_embeddings,
                metadatas=course_metadata,
                persist_directory=chroma_courses_dir
            )
            vectorstore_courses.persist()

        course_retriever = vectorstore_courses.as_retriever(search_kwargs={"k": 5})
        print("✅ Course-level vectorstore initialized.")
    else:
        course_retriever = None
        print("⚠️ Course-level retriever not available (processed_psu_courses.csv missing)")

    # ---- Load majors CSV & build Chroma vectorstore for major requirements queries ----
    majors_csv_path = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")
    if not os.path.exists(majors_csv_path):
        alt_path = os.path.join(os.path.dirname(__file__), "..", "LLM", "all_abington_majors_combined(5).csv")
        if os.path.exists(alt_path):
            majors_csv_path = alt_path
        else:
            print("⚠️ all_abington_majors_combined(5).csv not found")
            majors_csv_path = None
    
    if majors_csv_path and os.path.exists(majors_csv_path):
        df_majors = pd.read_csv(majors_csv_path)
        majors_texts = []
        majors_metadata = []
        
        for _, row in df_majors.iterrows():
            text_data = ""
            metadata = {}
            for col in df_majors.columns:
                value = str(row[col]) if not pd.isna(row[col]) else ""
                text_data += f"{col}: {value}. "
                metadata[col] = str(value) if value else ""
            majors_texts.append(text_data.strip())
            majors_metadata.append(metadata)
        
        chroma_majors_dir = os.path.join(CHROMA_DATA_DIR, "majors")
        try:
            if os.path.exists(chroma_majors_dir):
                try:
                    cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
                    vectorstore_majors = Chroma(
                        persist_directory=chroma_majors_dir,
                        embedding_function=cached_embeddings
                    )
                    _ = vectorstore_majors.as_retriever(search_kwargs={"k": 1})
                    print("✅ Using existing majors vectorstore.")
                except Exception as e:
                    print(f"⚠️ Majors vectorstore corrupted, recreating... Error: {e}")
                    if os.path.exists(chroma_majors_dir):
                        shutil.rmtree(chroma_majors_dir)
                    cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
                    vectorstore_majors = Chroma.from_texts(
                        texts=majors_texts,
                        embedding=cached_embeddings,
                        metadatas=majors_metadata,
                        persist_directory=chroma_majors_dir
                    )
                    vectorstore_majors.persist()
                    print("✅ Created new majors vectorstore.")
            else:
                cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
                vectorstore_majors = Chroma.from_texts(
                    texts=majors_texts,
                    embedding=cached_embeddings,
                    metadatas=majors_metadata,
                    persist_directory=chroma_majors_dir
                )
                vectorstore_majors.persist()
                print("✅ Created new majors vectorstore.")
        except Exception as e:
            print(f"⚠️ Error with majors vectorstore: {e}")
            if os.path.exists(chroma_majors_dir):
                shutil.rmtree(chroma_majors_dir)
            cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
            vectorstore_majors = Chroma.from_texts(
                texts=majors_texts,
                embedding=cached_embeddings,
                metadatas=majors_metadata,
                persist_directory=chroma_majors_dir
            )
            vectorstore_majors.persist()
        
        majors_retriever = vectorstore_majors.as_retriever(search_kwargs={"k": 3})
        print("✅ Majors vectorstore initialized.")
    else:
        majors_retriever = None
        print("⚠️ Majors retriever not available (all_abington_majors_combined(5).csv missing)")

    # ---- Load advising/registrar policy PDF into vectorstore ----
    global policy_retriever, policy_vectorstore
    policy_pdf_path = os.path.join(LLM_DIR, "PennState_Combined_Advising_and_Registrar_VERBATIM.pdf")
    if os.path.exists(policy_pdf_path):
        chroma_policy_dir = os.path.join(CHROMA_DATA_DIR, "policies")
        policy_collection = "advising_registrar_policies_v1"
        try:
            if os.path.exists(chroma_policy_dir):
                try:
                    cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
                    policy_vectorstore = Chroma(
                        collection_name=policy_collection,
                        persist_directory=chroma_policy_dir,
                        embedding_function=cached_embeddings
                    )
                    _ = policy_vectorstore.as_retriever(search_kwargs={"k": 1})
                    print("✅ Using existing policy vectorstore.")
                except Exception as e:
                    print(f"⚠️ Policy vectorstore corrupted, recreating... Error: {e}")
                    if os.path.exists(chroma_policy_dir):
                        shutil.rmtree(chroma_policy_dir)
                    cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
                    loader = PyPDFLoader(policy_pdf_path)
                    policy_docs = loader.load()
                    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
                    policy_chunks = splitter.split_documents(policy_docs)
                    for chunk in policy_chunks:
                        chunk.metadata["source"] = "advising_registrar_policy"
                    policy_vectorstore = Chroma.from_documents(
                        policy_chunks,
                        collection_name=policy_collection,
                        persist_directory=chroma_policy_dir,
                        embedding=cached_embeddings
                    )
                    policy_vectorstore.persist()
                    print("✅ Created new policy vectorstore.")
            else:
                cached_embeddings = CachedEmbeddings(OpenAIEmbeddings())
                loader = PyPDFLoader(policy_pdf_path)
                policy_docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
                policy_chunks = splitter.split_documents(policy_docs)
                for chunk in policy_chunks:
                    chunk.metadata["source"] = "advising_registrar_policy"
                policy_vectorstore = Chroma.from_documents(
                    policy_chunks,
                    collection_name=policy_collection,
                    persist_directory=chroma_policy_dir,
                    embedding=cached_embeddings
                )
                policy_vectorstore.persist()
                print("✅ Created new policy vectorstore.")
        except Exception as e:
            print(f"⚠️ Error with policy vectorstore: {e}")
            policy_vectorstore = None
        policy_retriever = policy_vectorstore.as_retriever(search_kwargs={"k": 4}) if policy_vectorstore else None
        if policy_retriever:
            print("✅ Policy retriever initialized.")
    else:
        policy_retriever = None
        print("⚠️ Policy PDF not found for RAG.")

    # ---- LLM Instances ----
    global llm1, llm2, llm_classifier
    llm1 = ChatOpenAI(model_name="gpt-4o", temperature=0)
    llm2 = ChatOpenAI(model_name="gpt-4o", temperature=0)
    llm_classifier = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)  # Faster for intent classification
    print(" LLMs initialized.")


# =========================
# Helper Functions
# =========================

def redact_student_info(docs):
    """
    Roughly redact student name and ID from the PDF text.
    """
    redacted_docs = []

    for doc in docs:
        content = doc.page_content

        # Mask patterns like "Name: John Doe 123456789"
        content = re.sub(
            r"\b[A-Z][a-z]+\s[A-Z][a-z]+(?=:\s?\d{6,})",
            "[REDACTED NAME]",
            content
        )

        # Mask 9-digit ID
        content = re.sub(r"\b\d{9}\b", "[REDACTED ID]", content)

        # Optionally mask first occurrence of a "First Last" pattern
        content = re.sub(
            r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b",
            "[REDACTED NAME]",
            content,
            count=1
        )

        doc.page_content = content
        redacted_docs.append(doc)

    return redacted_docs


def extract_completed_and_ip_courses(text):
    pattern = re.findall(r"\b[A-Z]{2,}\s+[0-9]{3}\b", text)
    courses = {re.sub(r"\s+", " ", course.strip()) for course in pattern}
    return courses


def extract_courses_from_term_lines(full_text: str) -> Tuple[List[str], List[str]]:
    """
    Extract completed and in-progress courses from term-based lines like:
    "FA 2024 CMPSC 131 ... 3.00 A" or "... 3.00 IP"
    """
    term_pattern = re.compile(r"\b(FA|SP|SU)\s+\d{4}\s+([A-Z]{2,})\s+(\d{1,3}[A-Z]?)\b")
    grade_pattern = re.compile(r"\b(A\+|A-|A|B\+|B-|B|C\+|C-|C|D\+|D-|D|F|IP|W|WF|LD)\b")

    completed = []
    in_progress = []
    completed_seen = set()
    in_progress_seen = set()

    for match in term_pattern.finditer(full_text):
        course_code = f"{match.group(2)} {match.group(3)}"
        context = full_text[match.end():match.end() + 80]
        grade_match = grade_pattern.search(context)
        grade = grade_match.group(1) if grade_match else ""

        if grade == "IP":
            if course_code not in in_progress_seen:
                in_progress.append(course_code)
                in_progress_seen.add(course_code)
            continue
        if grade in {"W", "WF", "LD", "F"}:
            continue
        if course_code not in completed_seen:
            completed.append(course_code)
            completed_seen.add(course_code)

    return completed, in_progress


def extract_total_credits_from_text(full_text: str) -> Optional[Tuple[float, float, float]]:
    """
    Extract total required/used/needed credits from summary lines.
    Prefer the line with the largest required credits (e.g., 120.00).
    """
    pattern = re.compile(
        r"Units:\s*([\d.]+)\s*required,\s*([\d.]+)\s*used,\s*([\d.]+)\s*needed",
        re.IGNORECASE
    )
    matches = []
    for match in pattern.finditer(full_text):
        try:
            required = float(match.group(1))
            used = float(match.group(2))
            needed = float(match.group(3))
            matches.append((required, used, needed))
        except Exception:
            continue

    if not matches:
        return None

    required, used, needed = max(matches, key=lambda t: (t[0], t[2]))
    return required, used, needed


def extract_student_data_deterministic(pdf_docs: List[Document]) -> dict:
    """
    Extract student data from full transcript text using deterministic parsing.
    More accurate and faster than RAG for structured data.
    """
    full_text = "\n".join(doc.page_content for doc in pdf_docs)
    
    # Extract major - try multiple patterns
    major = None
    
    # Pattern 1: "Major/Option: Computer Science, B.S. (Abington) Req Term: ..."
    # Stop at "Req Term" or end of line
    major_match = re.search(r'Major/Option[:\s]+([^\n]+?)(?:\s+Req\s+Term|$)', full_text, re.IGNORECASE)
    if major_match:
        major = major_match.group(1).strip()
        # Clean up - remove "Req Term" and anything after it
        major = re.sub(r'\s+Req\s+Term.*$', '', major, flags=re.IGNORECASE).strip()
        # Remove trailing punctuation/whitespace
        major = major.rstrip('.,;:')
    
    # Pattern 2: Look for major name with degree type pattern: "Name, B.S. (Location)"
    if not major or len(major) < 5 or 'Req Term' in major:
        # Look for pattern like "Computer Science, B.S. (Abington)"
        degree_pattern = r'([A-Z][a-zA-Z\s&]+(?:,\s*[A-Z][a-zA-Z\s&]+)*)\s*,\s*B\.?[AS]\.?\s*\([^)]+\)'
        match = re.search(degree_pattern, full_text)
        if match:
            major = match.group(1).strip()
            # Try to get the full string including degree
            full_match = re.search(rf'{re.escape(major)}[^\n]*B\.?[AS]\.?[^\n]*\([^)]+\)', full_text, re.IGNORECASE)
            if full_match:
                major = full_match.group(0).strip()
    
    # Pattern 3: If still not found, look for common major keywords
    if not major or len(major) < 5:
        # Look for major names near "Major/Option" label
        major_section = re.search(r'Major/Option[:\s]+([^\n]{10,100})', full_text, re.IGNORECASE)
        if major_section:
            section_text = major_section.group(1)
            # Try to extract just the major name part (before "Req Term" or other labels)
            clean_major = re.sub(r'\s+Req\s+Term.*$', '', section_text, flags=re.IGNORECASE).strip()
            if clean_major and len(clean_major) > 5:
                major = clean_major.split(',')[0].strip()  # Take first part before comma if exists
    
    # Extract completed and in-progress courses (term-based lines)
    completed_courses, in_progress = extract_courses_from_term_lines(full_text)

    # Extract total credits from summary lines
    totals = extract_total_credits_from_text(full_text)
    if totals:
        credits_required, credits_completed, credits_needed = totals
    else:
        credits_match = re.search(
            r'(\d+\.\d+)\s+units?\s+required.*?(\d+\.\d+)\s+used',
            full_text,
            re.IGNORECASE
        )
        credits_completed = float(credits_match.group(2)) if credits_match else 0.0
        credits_required = float(credits_match.group(1)) if credits_match else 120.0
        credits_needed = max(0.0, credits_required - credits_completed)
    
    # Extract unmet requirements (deterministic parsing)
    unmet_requirements = extract_not_satisfied_sections_deterministic(full_text)
    
    # Normalize major name if extracted
    if major:
        major = normalize_major_name(major)
    
    return {
        "major": major,
        "completed_courses": completed_courses,
        "in_progress_courses": in_progress,
        "credits_completed": credits_completed,
        "credits_required": credits_required,
        "credits_needed": credits_needed,
        "unmet_requirements": unmet_requirements
    }


def _clean_major_phrase(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    stopwords = {
        "major", "degree", "program", "student", "studying", "study", "in", "of",
        "ba", "bs", "b", "a", "s", "abington", "penn", "state", "university",
        "freshman", "sophomore", "junior", "senior", "first", "second", "third", "fourth", "year"
    }
    tokens = [t for t in text.split() if t not in stopwords and len(t) > 2]
    return " ".join(tokens)


def normalize_major_name(extracted_major: str) -> str:
    """
    Normalize extracted major name to match CSV format using fuzzy matching.
    """
    if not extracted_major or len(extracted_major.strip()) < 3:
        return None
    
    user_major = extracted_major.strip()
    user_clean = _clean_major_phrase(user_major)
    user_tokens = {w.lower() for w in re.split(r"[^A-Za-z]+", user_clean) if len(w) > 2}
    if not user_tokens and not user_clean:
        return None

    # Load majors CSV to get available majors
    majors_csv_path = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")
    if not os.path.exists(majors_csv_path):
        alt_path = os.path.join(os.path.dirname(__file__), "..", "LLM", "all_abington_majors_combined(5).csv")
        if os.path.exists(alt_path):
            majors_csv_path = alt_path
        else:
            return extracted_major  # Return as-is if CSV not found
    
    try:
        df_majors = pd.read_csv(majors_csv_path)
        available_majors = df_majors['Major Name'].tolist()
        cleaned_majors = {major: _clean_major_phrase(major) for major in available_majors}
        
        # First, try exact match (case-insensitive)
        for major in available_majors:
            if user_major.lower() == major.lower().strip():
                return major

        # Cleaned exact match (removes degree/abington/etc.)
        for major, cleaned in cleaned_majors.items():
            if user_clean and user_clean == cleaned:
                return major

        # Token subset match: require at least 2 overlapping tokens
        best_token_match = None
        best_token_count = 0
        for major in available_majors:
            major_tokens = {w.lower() for w in re.split(r"[^A-Za-z]+", cleaned_majors.get(major, "")) if len(w) > 2}
            overlap = user_tokens.intersection(major_tokens) if user_tokens else set()
            if len(overlap) >= 2 and len(overlap) > best_token_count:
                best_token_match = major
                best_token_count = len(overlap)
        if best_token_match:
            return best_token_match

        # Single-token match for common majors (e.g., "business")
        if len(user_tokens) == 1:
            token = next(iter(user_tokens))
            best_single = None
            for major, cleaned in cleaned_majors.items():
                if token in cleaned:
                    # Prefer the shortest cleaned major containing the token
                    if not best_single or len(cleaned) < len(cleaned_majors.get(best_single, "")):
                        best_single = major
            if best_single:
                return best_single

        # Substring match (either direction)
        for major in available_majors:
            cleaned = cleaned_majors.get(major, "")
            if user_clean and (user_clean in cleaned or cleaned in user_clean):
                return major

        # Try fuzzy matching with higher threshold to avoid wrong majors
        try:
            from fuzzywuzzy import process
            best_match = process.extractOne(user_clean or user_major, available_majors)
            if best_match and best_match[1] >= 70:  # slightly more forgiving
                return best_match[0]
        except ImportError:
            try:
                import difflib
                candidates = difflib.get_close_matches(user_clean or user_major, available_majors, n=1, cutoff=0.6)
                if candidates:
                    return candidates[0]
            except Exception:
                pass

        # If nothing matches, return None to force the caller to ask for clarification
        return None
    except Exception as e:
        print(f"⚠️ Error normalizing major name: {e}")
        return None


def extract_not_satisfied_sections_deterministic(full_text: str) -> List[dict]:
    """Extract unmet requirements using deterministic parsing (faster than RAG)."""
    unmet = []
    
    # Find all "Not Satisfied" sections
    not_satisfied_pattern = r'Not Satisfied.*?Units:\s*([\d.]+)\s*required.*?([\d.]+)\s*used.*?([\d.]+)\s*needed'
    for match in re.finditer(not_satisfied_pattern, full_text, re.DOTALL | re.IGNORECASE):
        try:
            unmet.append({
                "category": "Requirement",
                "required": float(match.group(1)),
                "used": float(match.group(2)),
                "needed": float(match.group(3))
            })
        except:
            continue
    
    return unmet


def extract_not_satisfied_sections(docs):
    """
    Parse What-If report text to find 'Not Satisfied' / unmet requirement blocks.
    Expects docs to be a list of Documents (e.g., from pdf_retriever.invoke()).
    """
    all_text = "\n".join(doc.page_content for doc in docs)
    print(f"🔍 Analyzing text sample: {all_text[:500]}...")

    # Pattern 1: Standard "Not Satisfied" sections
    pattern1 = re.compile(
        r"(.*?)Not Satisfied.*?Units:\s*([\d.]+)\s*required,\s*([\d.]+)\s*used(?:,\s*([\d.]+)\s*needed)?",
        re.DOTALL | re.IGNORECASE
    )
    results1 = pattern1.findall(all_text)
    print(f"🔍 Pattern 1 matches: {len(results1)}")

    # Pattern 2: Course tables with "needed" credits
    pattern2 = re.compile(
        r"(Units:\s*[\d.]+\s*required,\s*[\d.]+\s*used,\s*([\d.]+)\s*needed)",
        re.IGNORECASE
    )
    results2 = pattern2.findall(all_text)
    print(f"🔍 Pattern 2 matches: {len(results2)}")

    # Pattern 3: More flexible pattern for course tables
    pattern3 = re.compile(
        r"(Units:\s*([\d.]+)\s*required,\s*([\d.]+)\s*used,\s*([\d.]+)\s*needed)",
        re.IGNORECASE
    )
    results3 = pattern3.findall(all_text)
    print(f"🔍 Pattern 3 matches: {len(results3)}")

    extracted = []

    # Process Pattern 1: Not Satisfied sections
    for match in results1:
        category, required, used, needed = match
        extracted.append({
            "category": category.strip(),
            "required": float(required),
            "used": float(used),
            "needed": float(needed) if needed else round(float(required) - float(used), 2),
            "type": "not_satisfied"
        })

    # Process Pattern 2 & 3 only if near "Not Satisfied" context
    for m in pattern2.finditer(all_text):
        full_match = m.group(1)
        needed = m.group(2)
        start_idx = m.start()
        context_start = max(0, start_idx - 500)
        context_text = all_text[context_start:start_idx]
        if re.search(r"Not\s+Satisfied|Complete\s+the\s+following|Unmet|Needed", context_text, re.IGNORECASE):
            if float(needed) > 0:
                extracted.append({
                    "category": "Course Requirements Table",
                    "required": 0,
                    "used": 0,
                    "needed": float(needed),
                    "type": "course_table",
                    "context": full_match.strip()
                })

    for m in pattern3.finditer(all_text):
        full_match = m.group(1)
        required = m.group(2)
        used = m.group(3)
        needed = m.group(4)
        start_idx = m.start()
        context_start = max(0, start_idx - 500)
        context_text = all_text[context_start:start_idx]
        if re.search(r"Not\s+Satisfied|Complete\s+the\s+following|Unmet|Needed", context_text, re.IGNORECASE):
            if float(needed) > 0:
                extracted.append({
                    "category": "Course Requirements Table (Detailed)",
                    "required": float(required),
                    "used": float(used),
                    "needed": float(needed),
                    "type": "course_table_detailed",
                    "context": full_match.strip()
                })

    return extracted


# Valid intent labels from LLM classifier (maps to route types)
INTENT_TO_ROUTE = {
    "transcript_review": "transcript",
    "graduation_plan": "transcript",
    "policy": "policy",
    "course_info": "course",
    "major_requirements": "major",
    "general_chat": "chat",
}


def _classify_intent_llm(
    user_question: str,
    has_transcript: bool,
    conversation_history: Optional[List[dict]] = None,
) -> str:
    """
    Use LLM to classify user intent when keyword routing is ambiguous.
    Returns route type: 'course', 'major', 'transcript', 'policy', or 'chat'
    """
    global llm_classifier
    if llm_classifier is None:
        try:
            from langchain_openai import ChatOpenAI
            llm_classifier = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        except Exception:
            return "chat"

    recent_context = ""
    if conversation_history and len(conversation_history) > 0:
        recent = conversation_history[-2:]
        recent_context = "\nRecent messages: " + " | ".join(
            f"{m.get('role', '')}: {str(m.get('content', ''))[:80]}..." for m in recent
        )

    prompt = f"""Classify the student's intent. Reply with ONLY one word from this exact list:
transcript_review, graduation_plan, policy, course_info, major_requirements, general_chat

Definitions:
- transcript_review: Student wants you to REVIEW/ANALYZE their academic record (e.g., "audit my requirements", "review my transcript", "check my progress", "go through my record", "where do I stand")
- graduation_plan: Student wants a semester-by-semester graduation plan
- policy: Student asks about institutional processes/rules (e.g., "how do I audit a course?", "drop deadline", "withdraw policy")
- course_info: Student asks about a specific course (prerequisites, description)
- major_requirements: Student asks what courses a major requires (general catalog, not their own record)
- general_chat: Other questions

Student has transcript uploaded: {has_transcript}
{recent_context}

Question: "{user_question}"

Intent:"""

    try:
        result = llm_classifier.invoke([HumanMessage(content=prompt)]).content.strip().lower()
        # Extract first valid intent if LLM added extra text
        for intent in INTENT_TO_ROUTE:
            if intent in result:
                return INTENT_TO_ROUTE[intent]
        return "chat"
    except Exception as e:
        print(f"⚠️ Intent classification failed: {e}")
        return "chat"


def route_query(
    user_question: str,
    has_transcript: bool = False,
    conversation_history: Optional[List[dict]] = None,
) -> str:
    """
    Hybrid routing: keywords first for clear cases, LLM classification when ambiguous.
    Returns: 'course', 'major', 'transcript', 'policy', or 'chat'
    """
    question_lower = user_question.lower()

    # 1. Clear: Course code pattern (e.g., "CMPSC 131", "MATH 140")
    if re.search(r'\b[A-Z]{2,}\s+\d{3}[A-Z]?\b', user_question, re.IGNORECASE):
        return 'course'

    # 2. Clear: Strong transcript/planning keywords
    strong_transcript_terms = [
        'graduation plan', 'generate plan', 'create plan', 'semester plan',
        'my transcript', 'my courses', 'what do i need', 'generate plan',
        'my academic', 'my requirements', 'my progress', 'my record',
        'next semester', 'upcoming semester', 'what should i take next',
        'recommend courses', 'recommend me courses',
        'audit my document', 'audit my transcript', 'audit my academic requirements',
    ]
    if any(term in question_lower for term in strong_transcript_terms):
        return 'transcript'

    # 3. Clear: Year + major (planning intent)
    year_terms = ['freshman', 'sophomore', 'junior', 'senior', 'first-year', 'second-year', 'third-year', 'fourth-year']
    major_terms = ['computer science', 'biology', 'business', 'psychology', 'engineering', 'math', 'chemistry', 'physics', 'major', 'studying']
    if any(y in question_lower for y in year_terms) and any(m in question_lower for m in major_terms):
        return 'transcript'

    # 4. Clear: Major requirements (catalog lookup, not personal)
    if any(term in question_lower for term in ['major requirement', 'major courses', 'prescribed', 'option', 'what courses do i need for']):
        return 'major'

    # 5. Ambiguous: use LLM to classify (handles policy vs transcript_review, etc.)
    # e.g. "audit my requirements" -> transcript_review; "how do I audit a course?" -> policy
    return _classify_intent_llm(user_question, has_transcript, conversation_history)


def _extract_requested_semesters(user_question: str, conversation_history: Optional[List[dict]] = None) -> int:
    """Infer how many upcoming semesters the user is asking for."""
    q = (user_question or "").lower()
    # Numeric form: "next 2 semesters"
    m = re.search(r"\b(?:next\s+)?(\d+)\s+semesters?\b", q)
    if m:
        try:
            return max(1, min(8, int(m.group(1))))
        except Exception:
            pass

    # Word form: "next two semesters"
    words_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8,
    }
    for w, n in words_to_num.items():
        if re.search(rf"\b(?:next\s+)?{w}\s+semesters?\b", q):
            return n

    # Continuation phrasing ("for the semester after that too")
    continuation_terms = [
        "after that", "semester after that", "following semester",
        "and after that", "too", "also",
    ]
    if any(t in q for t in continuation_terms):
        # If prior context was recommendation-like, interpret as 2 upcoming semesters.
        if conversation_history:
            recent_context = " ".join(
                str(m.get("content", "")).lower()
                for m in conversation_history[-8:]
            )
            if any(k in recent_context for k in ["recommend", "next semester", "courses for next", "semester +1"]):
                return 2

    # Default
    return 1


def _build_reasoning_plan(
    user_question: str,
    conversation_history: Optional[List[dict]],
    has_transcript: bool,
) -> Dict[str, object]:
    """
    Lightweight reasoning layer:
      1) What does user want?
      2) What data is needed?
      3) How should we deliver?
    """
    q = (user_question or "").lower()

    recommendation_terms = [
        "next semester", "upcoming semester", "what should i take next",
        "recommend courses", "recommend me courses", "what should i take",
    ]
    followup_reco_terms = [
        "after that", "semester after that", "following semester",
        "for the semester after", "and after that", "too",
    ]
    audit_terms = ["audit", "review", "check", "analyze", "analyse", "evaluate", "where do i stand"]
    full_plan_terms = ["full graduation plan", "graduation plan", "semester-by-semester", "full plan"]

    recent_context_text = ""
    if conversation_history:
        recent_context_text = " ".join(str(m.get("content", "")).lower() for m in conversation_history[-8:])
    is_followup_recommendation = any(t in q for t in followup_reco_terms) and any(
        k in recent_context_text for k in ["recommend", "next semester", "courses for next", "semester +1"]
    )

    tools = {
        "local_retrievers": True,
        "web_search": False,
    }

    if any(t in q for t in recommendation_terms) or is_followup_recommendation:
        return {
            "intent": "recommendation",
            "needed_data": ["transcript_plan_state"],
            "delivery": "semester_recommendations",
            "semester_horizon": _extract_requested_semesters(user_question, conversation_history),
            "tools": tools,
            "web_search_mode": "never",
        }

    if any(t in q for t in audit_terms) and ("transcript" in q or "document" in q or "requirements" in q or has_transcript):
        return {
            "intent": "audit",
            "needed_data": ["transcript_summary", "unmet_requirements"],
            "delivery": "detailed_but_concise_audit",
            "semester_horizon": 0,
            "tools": tools,
            "web_search_mode": "never",
        }

    if any(t in q for t in full_plan_terms):
        return {
            "intent": "full_plan",
            "needed_data": ["transcript_or_major_year"],
            "delivery": "full_semester_plan",
            "semester_horizon": 8,
            "tools": tools,
            "web_search_mode": "never",
        }

    # Fallback to existing router/classifier
    contextual_q = _contextualize_policy_question(user_question, conversation_history)
    routed = route_query(contextual_q, has_transcript=has_transcript, conversation_history=conversation_history)
    if is_academic_or_policy_question(contextual_q):
        tools["web_search"] = True
    web_search_mode = "if_missing"
    if _is_time_sensitive_factual_question(contextual_q):
        web_search_mode = "verify_current_fact"
    recent_policy_deadline = any(k in recent_context_text for k in ["late drop", "deadline", "withdraw", "drop period"])
    if _question_needs_specific_date(contextual_q) or (_is_vague_followup(user_question) and recent_policy_deadline):
        routed = "policy"
        web_search_mode = "verify_date"
    return {
        "intent": routed,
        "needed_data": [],
        "delivery": "default",
        "semester_horizon": 0,
        "tools": tools,
        "web_search_mode": web_search_mode,
        "contextual_question": contextual_q,
    }


def _course_level_num(code: str) -> int:
    m = re.search(r'\b(\d{3})\b', code or '')
    try:
        return int(m.group(1)) if m else 999
    except Exception:
        return 999

def _normalize_year_token(year: str) -> Optional[str]:
    """Normalize year tokens to freshman/sophomore/junior/senior."""
    if not year:
        return None
    year_lower = year.strip().lower()
    year_mapping = {
        'first-year': 'freshman', '1st year': 'freshman', 'first year': 'freshman',
        'second-year': 'sophomore', '2nd year': 'sophomore', 'second year': 'sophomore',
        'third-year': 'junior', '3rd year': 'junior', 'third year': 'junior',
        'fourth-year': 'senior', '4th year': 'senior', 'fourth year': 'senior'
    }
    if year_lower in year_mapping:
        return year_mapping[year_lower]
    if year_lower in ['freshman', 'sophomore', 'junior', 'senior']:
        return year_lower
    return None

def _fuzzy_year_from_text(text: str) -> Optional[str]:
    """Handle common typos like 'freshamn' by fuzzy matching."""
    import difflib
    tokens = re.split(r'[^A-Za-z]+', text or "")
    choices = ['freshman', 'sophomore', 'junior', 'senior', 'firstyear', 'secondyear', 'thirdyear', 'fourthyear']
    for token in tokens:
        if len(token) < 5:
            continue
        token_lower = token.lower()
        match = difflib.get_close_matches(token_lower, choices, n=1, cutoff=0.78)
        if match:
            normalized = match[0]
            normalized = normalized.replace('firstyear', 'freshman').replace('secondyear', 'sophomore') \
                                   .replace('thirdyear', 'junior').replace('fourthyear', 'senior')
            return _normalize_year_token(normalized)
    return None

def _format_course_for_display(course: str) -> str:
    try:
        from graduationFlowchart import format_planned_course_label
        return format_planned_course_label(course)
    except Exception:
        if course.startswith("GENED__"):
            parts = course.split("__")
            key = parts[1] if len(parts) > 1 else "GENED"
            return f"Gen Ed ({key})"
        if course.startswith("OPT__"):
            return "Option Elective"
        if course.startswith("FREE__") or course.startswith("ELECTIVE__FREE__"):
            return "Free Elective"
        return course

def _format_plan_structure(plan_structure: dict, start_semester_number: int = 1) -> str:
    semesters = plan_structure.get("semesters", []) if plan_structure else []
    prereq_map = plan_structure.get("prerequisites", {}) if plan_structure else {}
    if not semesters:
        return "I couldn't generate a semester plan. Please try again or upload your transcript."
    lines = []
    for idx, sem in enumerate(sorted(semesters, key=lambda s: s.get("semester", 0))):
        display_semester = start_semester_number + idx
        lines.append(f"Semester {display_semester}:")
        courses = sem.get("courses", [])
        for course in courses:
            prereqs = prereq_map.get(course, []) or []
            prereq_text = ", ".join(_format_course_for_display(p) for p in prereqs) if prereqs else "None"
            display_course = _format_course_for_display(course)
            lines.append(f"{display_course} (prerequisites: {prereq_text})")
        lines.append(f"Total Credits: {sem.get('credits', len(courses) * 3)}")
        lines.append("---")
    return "\n".join(lines).rstrip("-\n")


def get_student_data_from_transcript() -> Optional[dict]:
    """Return deterministic student data if transcript is loaded."""
    global stored_pdf_docs, latest_student_data
    if stored_pdf_docs:
        try:
            latest_student_data = extract_student_data_deterministic(stored_pdf_docs)
            return latest_student_data
        except Exception as e:
            print(f"⚠️ Failed to extract student data deterministically: {e}")
    return latest_student_data

def process_psu_courses(user_question):
    """Process course-level queries using course_retriever."""
    global course_retriever
    if course_retriever is None:
        return "Course information not available."
    relevant_courses = course_retriever.invoke(user_question)
    if not relevant_courses:
        return "No relevant courses found."
    context_summary = "\n\n".join(doc.page_content for doc in relevant_courses)
    return context_summary


def extract_major_from_summary(summary_text):
    match = re.search(r"Major/Option\s*:\s*(.+)", summary_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return None


def extract_completed_courses_from_record(student_record):
    """
    Extract all completed and in-progress course codes from the LLM's
    structured student record summary.
    """
    patterns = [
        r'\b([A-Z]{2,}\s+\d{3}[A-Z]?)\b',  # e.g. "CMPSC 472" or "MATH 140A"
        r'\b([A-Z]{2,}\s+\d{3})\b',
        r'([A-Z]{2,}\s+\d{3}[A-Z]?)',
    ]

    courses = []
    for pattern in patterns:
        matches = re.findall(pattern, student_record)
        courses.extend(matches)

    seen = set()
    unique_courses = []
    for course in courses:
        normalized = course.replace(' ', ' ').strip()
        if normalized not in seen:
            seen.add(normalized)
            unique_courses.append(normalized)

    print(f" Extracted {len(unique_courses)} courses: {unique_courses[:10]}...")
    return unique_courses


def validate_prerequisites_in_plan(course_list, major_graph, completed_courses=None):
    """
    Check that for each course in course_list, all prerequisite courses
    are either completed or appear earlier in the list.
    """
    if completed_courses is None:
        completed_courses = []

    invalid_courses = []
    all_available_courses = set(completed_courses + course_list)

    for course in course_list:
        prereqs = [
            u for u, v, d in major_graph.in_edges(course, data=True)
            if d.get("type") == "prereq"
        ]

        if prereqs:
            missing_prereqs = [pr for pr in prereqs if pr not in all_available_courses]

            if missing_prereqs:
                invalid_courses.append({
                    'course': course,
                    'missing_prereqs': missing_prereqs
                })

    return invalid_courses


def enhance_graduation_plan_with_prerequisites(graduation_plan, major_graph):
    """
    Scan the generated plan and, for each course, append a clean
    "prerequisites: ..." clause based on the course graph.
    """
    enhanced_plan = graduation_plan

    course_pattern = r'\b([A-Z]{2,}\s+\d{3})\b'
    courses_in_plan = re.findall(course_pattern, graduation_plan)

    for course in courses_in_plan:
        prereqs = [
            u for u, v, d in major_graph.in_edges(course, data=True)
            if d.get("type") == "prereq"
        ]

        if prereqs:
            course_section_pattern = rf'\*\*{re.escape(course)}[^*]*\*\*'
            course_match = re.search(course_section_pattern, enhanced_plan)

            if course_match:
                course_text = course_match.group()

                # Remove any existing prereq text
                course_text = re.sub(
                    r',\s*prerequisites:\s*None',
                    '',
                    course_text
                )
                course_text = re.sub(
                    r',\s*prerequisites:\s*[^,]*(?:,\s*prerequisites:\s*[^,]*)*',
                    '',
                    course_text
                )

                prereq_text = f", prerequisites: {', '.join(prereqs)}"
                enhanced_plan = enhanced_plan.replace(
                    course_match.group(),
                    course_text + prereq_text
                )

    return enhanced_plan


# =========================
# MAIN: Transcript-based plan
# =========================

def aggregate_and_generate_response(user_question=None):
    """
    Main function to generate graduation plan using deterministic extraction and planning.
    Requires that a transcript was uploaded and stored_pdf_docs is available.
    """
    global stored_pdf_docs, has_transcript
    
    if not has_transcript or stored_pdf_docs is None:
        return "No transcript found. Please upload a What-If Report, or tell me your major and year level (e.g., 'I'm a Computer Science freshman') to generate a graduation plan."

    print("\n🔹 Using Deterministic Extraction and Planning...")

    # Step 1: Deterministic extraction (no RAG)
    student_data = extract_student_data_deterministic(stored_pdf_docs)
    print(f"✅ Extracted major: {student_data.get('major')}")
    completed_courses = student_data.get("completed_courses", [])
    in_progress_courses = student_data.get("in_progress_courses", [])
    completed_and_ip = []
    for course in completed_courses + in_progress_courses:
        if course and course not in completed_and_ip:
            completed_and_ip.append(course)

    print(f"✅ Extracted {len(completed_courses)} completed courses")
    if in_progress_courses:
        print(f"✅ Extracted {len(in_progress_courses)} in-progress courses")
    print(f"✅ Credits completed: {student_data.get('credits_completed')}")
    print(f"✅ Credits needed: {student_data.get('credits_needed')}")

    # Step 2: Deterministic planning
    from graduationFlowchart import generate_semester_plan
    
    courses_csv = os.path.join(os.path.dirname(__file__), "processed_psu_courses.csv")
    majors_csv = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")

    # Ensure CSV paths exist
    if not os.path.exists(courses_csv):
        alt_courses = os.path.join(os.path.dirname(__file__), "..", "LLM", "processed_psu_courses.csv")
        if os.path.exists(alt_courses):
            courses_csv = alt_courses
    
    if not os.path.exists(majors_csv):
        alt_majors = os.path.join(os.path.dirname(__file__), "..", "LLM", "all_abington_majors_combined(5).csv")
        if os.path.exists(alt_majors):
            majors_csv = alt_majors
    
    # Normalize major name before planning
    extracted_major = student_data.get("major")
    if extracted_major:
        normalized_major = normalize_major_name(extracted_major)
        if normalized_major != extracted_major:
            print(f"🔄 Normalized major: '{extracted_major}' → '{normalized_major}'")
        student_data["major"] = normalized_major
    else:
        # Try to extract from PDF text using LLM as fallback
        print("⚠️ Could not extract major deterministically, trying LLM extraction...")
        try:
            transcript_context = "\n".join(doc.page_content[:500] for doc in stored_pdf_docs[:3])
            major_extraction = llm1.invoke([
                SystemMessage(content="Extract the student's major name from the transcript. Return ONLY the full major name as it appears in university records (e.g., 'Computer Science, B.S. (Abington)')."),
                HumanMessage(content=f"Extract major from: {transcript_context[:1000]}")
            ]).content.strip()
            if major_extraction:
                student_data["major"] = normalize_major_name(major_extraction)
        except:
            pass
    
    if not student_data.get("major"):
        return "I couldn't determine your major from the transcript. Please specify your major (e.g., 'Computer Science, B.S. (Abington)') or contact support."
    
    try:
        plan_structure = generate_semester_plan(
            major_name=student_data.get("major"),
            completed_courses=completed_and_ip,
            unmet_requirements=student_data.get("unmet_requirements", []),
            courses_csv_path=courses_csv,
            majors_csv_path=majors_csv,
            credits_needed=student_data.get("credits_needed")
        )
        print(f"✅ Generated plan with {len(plan_structure['semesters'])} semesters")
    except ValueError as e:
        error_msg = str(e)
        if "not found in majors CSV" in error_msg:
            # Try fuzzy matching to suggest correct major
            print(f"⚠️ Major '{student_data.get('major')}' not found, trying fuzzy match...")
            try:
                from fuzzywuzzy import process
                df_majors = pd.read_csv(majors_csv)
                available_majors = df_majors['Major Name'].tolist()
                best_match = process.extractOne(student_data.get("major"), available_majors)
                if best_match and best_match[1] >= 50:
                    suggested_major = best_match[0]
                    return f"I couldn't find the exact major '{student_data.get('major')}' in the system. Did you mean '{suggested_major}'? Please verify your major name or contact support."
            except:
                pass
            return f"I couldn't find the major '{student_data.get('major')}' in the system. Please verify your major name is correct, or contact support."
        else:
            print(f"⚠️ Error in deterministic planning: {e}")
            import traceback
            traceback.print_exc()
            return f"I encountered an error generating your plan: {str(e)}. Please try again or contact support."
    except Exception as e:
        print(f"⚠️ Error in deterministic planning: {e}")
        import traceback
        traceback.print_exc()
        return f"I encountered an error generating your plan: {str(e)}. Please try again or contact support."

    # Step 3: deterministic formatting for consistency across all plans
    # (ensures GenEd category labels are shown, not raw placeholders).
    print("\n🔹 Formatting plan deterministically...")
    return _format_plan_structure(plan_structure, start_semester_number=1)
    course_list = re.findall(r'\b([A-Z]{2,}\s+\d{3})\b', enhanced_plan)
    validation_warnings = validate_prerequisites_in_plan(
        course_list,
        major_specific_graph,
        completed_courses
    )

    if validation_warnings:
        print("⚠️ Validation warnings (not shown to user):")
        for warning in validation_warnings:
            print(f"   - {warning['course']}: needs {', '.join(warning['missing_prereqs'])}")

    return enhanced_plan


def simple_extract_major_and_year(user_text):
    """Simple pattern-based extraction as fallback."""
    import re
    major = None
    year = None
    
    # Common major patterns
    major_patterns = [
        r'(?:studying|major in|major is|studies)\s+([^,\.\n]+)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:major|degree)',
        r'(computer science|biology|business|engineering|psychology|mathematics|chemistry|physics|english|history|economics|accounting|finance|marketing|management)',
    ]
    
    # Year patterns
    year_patterns = [
        r'\b(freshman|sophomore|junior|senior|first-year|second-year|third-year|fourth-year|1st year|2nd year|3rd year|4th year)\b',
    ]
    
    text_lower = user_text.lower()
    
    # Extract year
    for pattern in year_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            year_str = match.group(1).lower()
            year_mapping = {
                'first-year': 'freshman', '1st year': 'freshman', 'first year': 'freshman',
                'second-year': 'sophomore', '2nd year': 'sophomore', 'second year': 'sophomore',
                'third-year': 'junior', '3rd year': 'junior', 'third year': 'junior',
                'fourth-year': 'senior', '4th year': 'senior', 'fourth year': 'senior'
            }
            year = year_mapping.get(year_str, year_str)
            if year in ['freshman', 'sophomore', 'junior', 'senior']:
                break
    
    # Extract major - try to find in CSV
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(__file__), "..", "LLM", "all_abington_majors_combined(5).csv")
        majors_df = pd.read_csv(csv_path)
        
        # Common major keywords
        major_keywords = {
            'computer science': ['computer science', 'cs', 'comp sci'],
            'biology': ['biology', 'bio'],
            'business': ['business', 'business administration'],
            'psychology': ['psychology', 'psych'],
            'mathematics': ['mathematics', 'math'],
        }
        
        for keyword, variations in major_keywords.items():
            if any(var in text_lower for var in variations):
                # Find matching major in CSV
                matching = majors_df[majors_df['Major Name'].str.contains(keyword, case=False, na=False)]
                if not matching.empty:
                    major = matching.iloc[0]['Major Name']
                    break
        
        # If not found, try direct search
        if not major:
            for major_name in majors_df['Major Name'].tolist():
                major_lower = major_name.lower()
                # Check if any significant word from major name is in user text
                major_words = [w for w in major_lower.split() if len(w) > 3]
                if any(word in text_lower for word in major_words):
                    major = major_name
                    break
    except Exception as e:
        print(f"⚠️ Error in simple extraction: {e}")
    
    return major, year

def extract_major_and_year_from_text(user_text, conversation_history):
    """Extract major and year level from user text using LLM."""
    try:
        # Get list of available majors from CSV
        csv_path = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")
        if not os.path.exists(csv_path):
            # Try alternative path
            csv_path = os.path.join(os.path.dirname(__file__), "..", "LLM", "all_abington_majors_combined(5).csv")
        majors_df = pd.read_csv(csv_path)
        available_majors = majors_df['Major Name'].tolist()[:50]  # Limit to first 50 for prompt size
        
        # Build context from conversation history
        history_context = ""
        if conversation_history and len(conversation_history) > 0:
            recent_history = conversation_history[-4:]  # Last 4 messages
            history_context = "\n\nPrevious conversation:\n"
            for msg in recent_history:
                role = "Student" if msg.get('role') == 'user' else "AI"
                history_context += f"{role}: {msg.get('content', '')}\n"
        
        # Use LLM to extract major and year
        # Show some common majors first
        common_majors = [m for m in available_majors if any(x in m.lower() for x in ['computer', 'biology', 'business', 'psychology', 'mathematics'])][:10]
        other_majors = [m for m in available_majors if m not in common_majors][:10]
        
        extraction_prompt = f"""Extract the student's major and year level from their message.

Common majors: {', '.join(common_majors)}
Other majors available: {', '.join(other_majors)}

Year levels: freshman, sophomore, junior, senior (or first-year, second-year, etc.)

Student message: "{user_text}"
{history_context}

CRITICAL INSTRUCTIONS:
1. Look carefully for year level words: "freshman", "sophomore", "junior", "senior", "first-year", "second-year", etc.
2. Even if the student says "I am a freshman" or "I am a computer science major freshman", extract BOTH the major AND the year.
3. The year might appear anywhere in the message or conversation history.

IMPORTANT: Respond ONLY in this exact format (one line, no other text):
MAJOR: [exact major name from list, or closest match]
YEAR: [freshman OR sophomore OR junior OR senior]

Examples:
- "I'm a computer science freshman" → MAJOR: Computer Science, B.S. (Abington) YEAR: freshman
- "I am a computer science. major freshman, give me a graduation plan" → MAJOR: Computer Science, B.S. (Abington) YEAR: freshman
- "I am a freshman studying computer science" → MAJOR: Computer Science, B.S. (Abington) YEAR: freshman
- "biology sophomore" → MAJOR: Biology, B.S. (Abington) YEAR: sophomore
- "business junior" → MAJOR: Business, B.S. (Abington) YEAR: junior

If you cannot determine, use:
MAJOR: UNKNOWN
YEAR: UNKNOWN"""
        
        extraction_result = llm1.invoke([
            SystemMessage(content="You are a precise information extractor. Extract major and year level from student messages."),
            HumanMessage(content=extraction_prompt)
        ]).content
        
        # Parse the result - be more flexible with parsing
        major = None
        year = None
        
        # Try multiple parsing strategies
        extraction_result_lower = extraction_result.lower()
        
        # Strategy 1: Look for explicit MAJOR: and YEAR: labels
        for line in extraction_result.split('\n'):
            line_upper = line.upper()
            if 'MAJOR:' in line_upper:
                major = line.split(':', 1)[1].strip()
                if major == 'UNKNOWN':
                    major = None
            elif 'YEAR:' in line_upper:
                year = line.split(':', 1)[1].strip().lower()
                if year == 'UNKNOWN':
                    year = None
                # Normalize year terms
                year_mapping = {
                    'first-year': 'freshman', '1st year': 'freshman', 'first year': 'freshman',
                    'second-year': 'sophomore', '2nd year': 'sophomore', 'second year': 'sophomore',
                    'third-year': 'junior', '3rd year': 'junior', 'third year': 'junior',
                    'fourth-year': 'senior', '4th year': 'senior', 'fourth year': 'senior'
                }
                year = year_mapping.get(year, year)
        
        # Strategy 2: If year not found, try direct pattern matching in the LLM response
        if not year:
            year_patterns = [
                r'\b(freshman|sophomore|junior|senior)\b',
                r'\b(first-year|second-year|third-year|fourth-year)\b',
                r'\b(1st year|2nd year|3rd year|4th year)\b'
            ]
            for pattern in year_patterns:
                match = re.search(pattern, extraction_result_lower, re.IGNORECASE)
                if match:
                    year_str = match.group(1).lower()
                    year_mapping = {
                        'first-year': 'freshman', '1st year': 'freshman', 'first year': 'freshman',
                        'second-year': 'sophomore', '2nd year': 'sophomore', 'second year': 'sophomore',
                        'third-year': 'junior', '3rd year': 'junior', 'third year': 'junior',
                        'fourth-year': 'senior', '4th year': 'senior', 'fourth year': 'senior'
                    }
                    year = year_mapping.get(year_str, year_str)
                    if year in ['freshman', 'sophomore', 'junior', 'senior']:
                        break
        
        # Strategy 3: If still no year, check the original user text and conversation history
        if not year:
            # Check current user message
            year_patterns = [
                r'\b(freshman|sophomore|junior|senior)\b',
                r'\b(first-year|second-year|third-year|fourth-year)\b',
                r'\b(1st year|2nd year|3rd year|4th year)\b'
            ]
            for pattern in year_patterns:
                match = re.search(pattern, user_text.lower(), re.IGNORECASE)
                if match:
                    year_str = match.group(1).lower()
                    year_mapping = {
                        'first-year': 'freshman', '1st year': 'freshman', 'first year': 'freshman',
                        'second-year': 'sophomore', '2nd year': 'sophomore', 'second year': 'sophomore',
                        'third-year': 'junior', '3rd year': 'junior', 'third year': 'junior',
                        'fourth-year': 'senior', '4th year': 'senior', 'fourth year': 'senior'
                    }
                    year = year_mapping.get(year_str, year_str)
                    if year in ['freshman', 'sophomore', 'junior', 'senior']:
                        break
            
            # Check conversation history if still not found
            if not year and conversation_history:
                for msg in reversed(conversation_history[-4:]):
                    if msg.get('role') == 'user':
                        msg_text = msg.get('content', '').lower()
                        for pattern in year_patterns:
                            match = re.search(pattern, msg_text, re.IGNORECASE)
                            if match:
                                year_str = match.group(1).lower()
                                year_mapping = {
                                    'first-year': 'freshman', '1st year': 'freshman', 'first year': 'freshman',
                                    'second-year': 'sophomore', '2nd year': 'sophomore', 'second year': 'sophomore',
                                    'third-year': 'junior', '3rd year': 'junior', 'third year': 'junior',
                                    'fourth-year': 'senior', '4th year': 'senior', 'fourth year': 'senior'
                                }
                                year = year_mapping.get(year_str, year_str)
                                if year in ['freshman', 'sophomore', 'junior', 'senior']:
                                    break
                        if year:
                            break

        # Strategy 4: Fuzzy year matching for typos (e.g., "freshamn")
        if not year:
            year = _fuzzy_year_from_text(user_text)
        
        # Try to match major name with available majors using fuzzy matching if needed
        if major and major != 'UNKNOWN':
            # Check if major is in the list or find close match
            df = majors_df
            matching_major = df[df['Major Name'].str.contains(major, case=False, na=False)]
            if not matching_major.empty:
                major = matching_major.iloc[0]['Major Name']
            else:
                # Try fuzzy matching on first few words
                try:
                    from fuzzywuzzy import process
                    major_names = majors_df['Major Name'].tolist()
                    best_match = process.extractOne(major, major_names)
                    if best_match and best_match[1] >= 70:  # 70% similarity threshold
                        major = best_match[0]
                    else:
                        major = None
                except ImportError:
                    # Fallback: try partial string matching
                    for major_name in majors_df['Major Name'].tolist():
                        if major.lower() in major_name.lower() or major_name.lower() in major.lower():
                            major = major_name
                            break
                    else:
                        major = None
        
        # Final fallback: simple deterministic extraction from user text
        if not major or not year:
            simple_major, simple_year = simple_extract_major_and_year(user_text)
            major = major or simple_major
            year = year or simple_year

        # Fuzzy year matching for misspellings in user text
        if not year:
            year = _fuzzy_year_from_text(user_text)

        # Normalize major to a CSV-backed canonical name
        if major:
            normalized_major = normalize_major_name(major)
            if normalized_major:
                major = normalized_major
        
        print(f"🔍 Extracted - Major: {major}, Year: {year}")
        print(f"🔍 LLM raw response: {extraction_result[:200]}...")  # Debug: show first 200 chars of LLM response
        return major, year
        
    except Exception as e:
        print(f"⚠️ Error extracting major/year: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_conversational_response(
    user_question,
    conversation_history,
    user_id='default',
    user_has_transcript=None,
    debug_mode: bool = False
):
    """Generate a conversational response using conversation history for context.
    user_has_transcript: If provided, overrides global has_transcript for plan routing.
    Use doc_hash != 'no_transcript' from API to avoid using another user's transcript.
    """
    print("\n🔹 Generating Conversational Response...")

    global llm1, llm2, llm_classifier, has_transcript, stored_pdf_docs, latest_student_data
    global course_retriever, majors_retriever, pdf_retriever, policy_retriever
    global execution_trace_store

    # Keep lightweight tracing always on so debugging info is consistently available.
    trace_enabled = True
    trace = _new_execution_trace(user_id=user_id, user_question=user_question, enabled=trace_enabled)
    _trace_event(trace, "start", "Request received", {"has_history": bool(conversation_history), "history_len": len(conversation_history or [])})

    # Use per-user transcript status when provided (avoids cross-user contamination)
    if user_has_transcript is not None:
        has_transcript_for_plan = user_has_transcript and (stored_pdf_docs is not None)
    else:
        has_transcript_for_plan = stored_pdf_docs is not None and has_transcript

    if llm1 is None or llm2 is None:
        print("⚠️ LLMs not initialized. Initializing now...")
        try:
            setup_environment(None)
        except Exception as e:
            print(f"⚠️ Error initializing LLMs: {e}")
            from langchain_openai import ChatOpenAI
            if not os.getenv('OPENAI_API_KEY'):
                os.environ['OPENAI_API_KEY'] = "REDACTED"
            if not os.getenv('LANGCHAIN_API_KEY'):
                os.environ['LANGCHAIN_API_KEY'] = 'REDACTED'
            llm1 = ChatOpenAI(model_name="gpt-4o", temperature=0)
            llm2 = ChatOpenAI(model_name="gpt-4o", temperature=0)
            llm_classifier = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    reasoning_plan = _build_reasoning_plan(user_question, conversation_history, has_transcript_for_plan)
    contextual_question = str(reasoning_plan.get("contextual_question") or user_question)
    query_type = route_query(contextual_question, has_transcript=has_transcript_for_plan, conversation_history=conversation_history)
    if reasoning_plan.get("intent") in {"policy", "course", "major", "transcript"}:
        query_type = reasoning_plan.get("intent")
    student_data = get_student_data_from_transcript()
    execution_profile = _build_execution_profile(contextual_question, query_type, reasoning_plan)
    reasoning_plan["execution_profile"] = execution_profile
    trace["plan"] = {
        "intent": reasoning_plan.get("intent"),
        "query_type": query_type,
        "delivery": reasoning_plan.get("delivery"),
        "needed_data": reasoning_plan.get("needed_data"),
        "tools": reasoning_plan.get("tools"),
        "web_search_mode": reasoning_plan.get("web_search_mode"),
        "execution_profile": execution_profile,
        "contextual_question": contextual_question,
    }
    _trace_event(trace, "plan", "Built reasoning plan", trace["plan"])

    def finalize_response(text: str) -> str:
        final_text = _observe_and_repair_response(
            user_question=user_question,
            response=text,
            reasoning_plan=reasoning_plan,
            query_type=query_type,
            conversation_history=conversation_history,
            trace=trace
        )
        if trace.get("enabled"):
            trace["result_preview"] = final_text[:400]
            trace["finished_at"] = datetime.now().isoformat()
            execution_trace_store[user_id] = trace
        return final_text

    # Handle recommendation intent early via reasoning layer (before route-specific handlers).
    if reasoning_plan.get("intent") == "recommendation":
        if not student_data and has_transcript_for_plan and stored_pdf_docs:
            try:
                student_data = extract_student_data_deterministic(stored_pdf_docs)
                latest_student_data = student_data
            except Exception as e:
                print(f"⚠️ Failed to extract student data for recommendations: {e}")
                student_data = latest_student_data

        if student_data:
            completed = set(student_data.get("completed_courses", []))
            in_progress = set(student_data.get("in_progress_courses", []))
            all_done = completed.union(in_progress)
            major_raw = student_data.get("major", "Unknown") or "Unknown"
            major = normalize_major_name(major_raw) or major_raw

            try:
                from graduationFlowchart import generate_semester_plan
                courses_csv = os.path.join(os.path.dirname(__file__), "processed_psu_courses.csv")
                majors_csv = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")
                completed_and_ip = list(all_done)
                plan = generate_semester_plan(
                    major_name=major or "Unknown",
                    completed_courses=completed_and_ip,
                    unmet_requirements=student_data.get("unmet_requirements", []),
                    courses_csv_path=courses_csv,
                    majors_csv_path=majors_csv,
                    credits_needed=student_data.get("credits_needed"),
                    max_semesters=8,
                    max_credits_per_sem=15,
                    min_credits_per_sem=15
                )
                if plan.get("semesters"):
                    horizon = int(reasoning_plan.get("semester_horizon", 1) or 1)
                    ordered = sorted(plan["semesters"], key=lambda s: s.get("semester", 0))
                    selected = ordered[:max(1, min(8, horizon))]
                    lines = [f"Here are recommended courses for your next {len(selected)} semester(s) (based on your transcript and major {major}):"]
                    for idx, sem in enumerate(selected, start=1):
                        lines.append(f"\nSemester +{idx}:")
                        for c in sem.get("courses", [])[:8]:
                            lines.append(f"- {_format_course_for_display(c)}")
                        lines.append(f"Total Credits: {sem.get('credits', 0)}")
                    lines.append("\nIf you want, I can also generate your full semester-by-semester graduation plan.")
                    return finalize_response("\n".join(lines))
            except Exception as e:
                print(f"⚠️ Could not build next-semester recommendation: {e}")

        return finalize_response("I need your transcript to recommend next-semester courses. Please upload it first, then ask again.")

    # Universal pre-execution verification stage (applies to all query types by profile).
    if execution_profile.get("require_web_precheck"):
        mode = execution_profile.get("verification_mode")
        if mode == "date" and query_type in {"policy", "chat"}:
            web_response = _web_search_and_synthesize(
                contextual_question,
                date_only=True,
                conversation_history=conversation_history,
                prefer_current_or_future=bool(execution_profile.get("prefer_current_or_future")),
                trace=trace
            )
            if web_response and web_response != "DATE_NOT_FOUND":
                return finalize_response(_format_deadline_answer(web_response, contextual_question, conversation_history))
        elif mode == "fresh_fact":
            web_response = _web_search_and_synthesize(
                contextual_question,
                conversation_history=conversation_history,
                trace=trace,
                force_fresh_identity=bool(execution_profile.get("identity_like"))
            )
            if web_response:
                return finalize_response(web_response)

    # Quick course prerequisite path
    if query_type == 'course':
        expanded_course_question = _expand_academic_query_terms(user_question)
        course_match = re.search(r'\b([A-Z]{2,})\s+(\d{3}[A-Z]?)\b', user_question, re.IGNORECASE)
        if course_match:
            course_code = f"{course_match.group(1).upper()} {course_match.group(2)}"
            try:
                try:
                    from graphExperiment import course_graph as cg
                    if cg:
                        prereq_text = get_course_prerequisites(course_code, cg)
                        return finalize_response(prereq_text)
                except ImportError:
                    pass
            except Exception as e:
                print(f"⚠️ Could not fetch prerequisites: {e}")
        course_info = process_psu_courses(expanded_course_question)
        if course_info and "No relevant courses found" not in course_info:
            return finalize_response(course_info)
        # Course not in our data - try web search for Penn State course info
        if is_academic_or_policy_question(user_question):
            web_response = _web_search_and_synthesize(expanded_course_question, trace=trace)
            if web_response:
                return finalize_response(web_response)
        return finalize_response("I couldn't find that course. Please check the code (e.g., 'CMPSC 131').")

    # Policy questions directly answered from the policy PDF
    if query_type == 'policy':
        response = None
        if policy_retriever:
            try:
                policy_docs = policy_retriever.invoke(contextual_question)
                if policy_docs:
                    policy_context = "\n\n".join(doc.page_content[:900] for doc in policy_docs[:4])
                    response = llm2.invoke([
                        SystemMessage(content="Answer the student's question using ONLY the provided policy text. If the answer isn't present, say you couldn't find it in the policy."),
                        HumanMessage(content=f"Policy context:\n{policy_context}\n\nQuestion: {contextual_question}")
                    ]).content
                    # Universal execution profile: enforce date verification when required.
                    if execution_profile.get("verification_mode") == "date":
                        # If policy text doesn't provide a concrete date, verify via web immediately.
                        if not _has_concrete_date_in_text(response):
                            web_response = _web_search_and_synthesize(
                                contextual_question,
                                date_only=True,
                                conversation_history=conversation_history,
                                prefer_current_or_future=True,
                                trace=trace
                            )
                            if web_response and web_response != "DATE_NOT_FOUND":
                                return finalize_response(_format_deadline_answer(web_response, contextual_question, conversation_history))
                            if _term_or_year_was_provided(contextual_question, conversation_history):
                                term, year = _infer_term_year_from_context(contextual_question, conversation_history)
                                target_label = f"{(term or 'the requested term').title()} {year}".strip()
                                return finalize_response(
                                    f"I couldn't verify an exact late-drop date for {target_label} from the available sources. "
                                    "Please specify campus/session type (e.g., Abington regular session or 7-week) so I can narrow it down."
                                )
                            return finalize_response("I need the specific term/session (e.g., Spring 2026, Fall 2026, or Summer) to return the exact late-drop date.")
                        # Local policy answer already includes a concrete date; return only the date(s).
                        local_dates = _extract_dates_from_text(response)
                        if local_dates:
                            return finalize_response(_format_deadline_answer(local_dates[0], contextual_question, conversation_history))
                        return finalize_response("I need the specific term/session (e.g., Spring 2026, Fall 2026, or Summer) to return the exact late-drop date.")
                    if not _response_indicates_no_answer_found(response):
                        return finalize_response(response)
                    # Answer not in our data - try web search
                    print("🔍 Policy answer not found in local data, trying web search...")
                    web_response = _web_search_and_synthesize(contextual_question, conversation_history=conversation_history, trace=trace)
                    if web_response:
                        return finalize_response(web_response)
            except Exception as e:
                print(f"⚠️ Could not retrieve policy information: {e}")
        # No policy docs or web search failed - try web search as fallback
        if response is None or _response_indicates_no_answer_found(response):
            web_response = _web_search_and_synthesize(contextual_question, conversation_history=conversation_history, trace=trace)
            if web_response:
                return finalize_response(web_response)
        return finalize_response("I couldn't find that policy information. Please rephrase your question or check the Penn State website directly.")

    # Transcript-based planning or transcript Q&A
    if query_type == 'transcript':
        plan_keywords = ['graduation plan', 'generate plan', 'create plan', 'semester plan', 'course plan', 'plan', 'schedule', 'next semester']
        wants_plan = any(k in user_question.lower() for k in plan_keywords)

        if wants_plan:
            # FIRST: If user explicitly states major and year in their question, use that
            # (avoids wrong major from another user's transcript / global state)
            major_from_text, year_from_text = None, None
            try:
                major_from_text, year_from_text = extract_major_and_year_from_text(user_question, conversation_history)
                if not major_from_text or not year_from_text:
                    major_from_text, year_from_text = simple_extract_major_and_year(user_question)
                if not year_from_text:
                    year_from_text = _fuzzy_year_from_text(user_question)
            except Exception as e:
                print(f"⚠️ Error extracting major/year from text: {e}")

            if major_from_text and year_from_text:
                print(f"📋 Using explicit major/year from question: {major_from_text}, {year_from_text}")
                return finalize_response(generate_plan_from_major_and_year(major_from_text, year_from_text))

            # SECOND: Use transcript only if THIS user has one
            if has_transcript_for_plan:
                print("📋 Detected plan request with transcript, using deterministic pipeline...")
                return finalize_response(aggregate_and_generate_response(user_question))

            # No explicit major/year and no transcript
            return finalize_response("I need your major and year (e.g., 'Computer Science freshman' or 'Business, B.S. freshman') or a transcript upload to build a plan.")

        if student_data:
            credits_needed = student_data.get("credits_needed", "unknown")
            credits_completed = student_data.get("credits_completed", "unknown")
            unmet = student_data.get("unmet_requirements", [])
            # If deterministic unmet data is too generic, enrich from full transcript text parser.
            if unmet and all(
                isinstance(x, dict) and x.get("category", "").strip().lower() in {"", "requirement"}
                for x in unmet
            ) and stored_pdf_docs:
                try:
                    parsed_unmet = extract_not_satisfied_sections(stored_pdf_docs)
                    if parsed_unmet:
                        unmet = parsed_unmet
                except Exception as e:
                    print(f"⚠️ Could not enrich unmet requirement details: {e}")

            def _format_unmet_item(item) -> str:
                if isinstance(item, dict):
                    category = str(item.get("category", "Requirement")).strip() or "Requirement"
                    needed = item.get("needed", 0)
                    used = item.get("used", None)
                    required = item.get("required", None)
                    if required is not None and used is not None:
                        return f"- {category}: {needed} credits needed ({used}/{required} completed)"
                    return f"- {category}: {needed} credits needed"
                if isinstance(item, (list, tuple)):
                    return f"- Requirement: {', '.join(str(x) for x in item)}"
                return f"- Requirement: {str(item)}"

            unmet_summary = "\n".join(
                _format_unmet_item(item) for item in unmet[:8]
            ) if unmet else "No unmet requirements found."

            # Audit-focused wording when user asks to "audit/review/check" transcript.
            audit_terms = ["audit", "review", "check", "analyze", "analyse", "evaluate", "where do i stand"]
            is_audit_request = any(t in user_question.lower() for t in audit_terms)

            if is_audit_request or reasoning_plan.get("intent") == "audit":
                return finalize_response(
                    f"Academic Audit Summary\n"
                    f"- Major: {student_data.get('major', 'Unknown')}\n"
                    f"- Credits completed: {credits_completed}\n"
                    f"- Credits remaining: {credits_needed}\n"
                    f"- In-progress courses: {', '.join(student_data.get('in_progress_courses', [])[:8]) if student_data.get('in_progress_courses') else 'None'}\n"
                    f"- Recently completed courses: {', '.join(student_data.get('completed_courses', [])[-8:]) if student_data.get('completed_courses') else 'Not detected'}\n"
                    f"- Key unmet requirements:\n{unmet_summary}\n\n"
                    f"Next step: ask \"recommend me courses for next 2 semesters\" for concrete near-term scheduling, "
                    f"or \"give me a full graduation plan\" for the full 8-semester path."
                )

            return finalize_response(
                f"From your transcript:\n"
                f"- Major: {student_data.get('major', 'Unknown')}\n"
                f"- Credits completed: {credits_completed}\n"
                f"- Credits remaining: {credits_needed}\n"
                f"- Top unmet requirements:\n{unmet_summary}\n"
                f"If you want a detailed plan, ask me to generate a graduation plan."
            )
        return finalize_response("I couldn't read a transcript. Please upload one so I can answer transcript-specific questions.")

    # Major requirements lookups
    if query_type == 'major':
        major_info = process_psu_courses(user_question)
        if major_info and "No relevant courses found" not in major_info:
            return finalize_response(major_info)
        # Major not in our data - try web search for Penn State major requirements
        if is_academic_or_policy_question(user_question):
            web_response = _web_search_and_synthesize(user_question, trace=trace)
            if web_response:
                return finalize_response(web_response)
        return finalize_response("I couldn't find that major's requirements. Please provide the exact major name.")

    # Otherwise, use conversational approach with context or short recommendations
    # Scope restriction: non-academic questions get scope message
    if not is_academic_or_policy_question(user_question):
        return finalize_response("Sorry, this question is beyond the scope of this conversation. I can only help with academic advising, course planning, Penn State policies, and graduation requirements.")

    try:
        if reasoning_plan.get("intent") != "recommendation" and any(
            k in user_question.lower() for k in ['next semester', 'upcoming semester', 'what should i take next', 'recommend courses', 'recommend me courses']
        ):
            if not student_data:
                if stored_pdf_docs:
                    try:
                        student_data = extract_student_data_deterministic(stored_pdf_docs)
                        latest_student_data = student_data
                    except Exception as e:
                        print(f"⚠️ Failed to extract student data: {e}")
                        student_data = latest_student_data
                else:
                    student_data = latest_student_data

            if student_data:
                completed = set(student_data.get("completed_courses", []))
                in_progress = set(student_data.get("in_progress_courses", []))
                all_done = completed.union(in_progress)
                major_raw = student_data.get("major", "Unknown") or "Unknown"
                major = normalize_major_name(major_raw) or major_raw

                remaining_courses = []
                try:
                    major_course_codes = get_major_course_codes(major) or []
                    if not major_course_codes:
                        majors_csv = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")
                        if os.path.exists(majors_csv):
                            dfm = pd.read_csv(majors_csv)
                            candidates = dfm[dfm['Major Name'].str.contains(major, case=False, na=False)]
                            if not candidates.empty:
                                try:
                                    major_course_codes = get_major_course_codes(candidates.iloc[0]['Major Name']) or []
                                except Exception:
                                    major_course_codes = []
                    for c in major_course_codes:
                        if c not in all_done:
                            remaining_courses.append(c)
                    try:
                        from graduationFlowchart import GraduationFlowchart
                        majors_csv = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")
                        if os.path.exists(majors_csv):
                            flowchart = GraduationFlowchart(
                                os.path.join(os.path.dirname(__file__), "processed_psu_courses.csv"),
                                majors_csv
                            )
                            row = flowchart._get_major_row(major)
                            if row is not None:
                                from graduationFlowchart import extract_all_courses, resolve_common_or_groups
                                prescribed = resolve_common_or_groups(extract_all_courses(str(row.get("Prescribed Courses", "") or "")))
                                for c in prescribed:
                                    if c not in all_done:
                                        remaining_courses.append(c)
                    except Exception as e2:
                        print(f"⚠️ Fallback also failed: {e2}")
                except Exception as e:
                    print(f"⚠️ Could not load major course codes: {e}")

                remaining_courses = sorted(set(remaining_courses), key=_course_level_num)
                suggested = remaining_courses[:5] if remaining_courses else []
                if not suggested:
                    try:
                        from graduationFlowchart import generate_semester_plan
                        courses_csv = os.path.join(os.path.dirname(__file__), "processed_psu_courses.csv")
                        majors_csv = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")
                        completed_and_ip = list(all_done)
                        plan = generate_semester_plan(
                            major_name=major or "Unknown",
                            completed_courses=completed_and_ip,
                            unmet_requirements=student_data.get("unmet_requirements", []),
                            courses_csv_path=courses_csv,
                            majors_csv_path=majors_csv,
                            credits_needed=student_data.get("credits_needed")
                        )
                        if plan.get("semesters"):
                            next_sem = sorted(plan["semesters"], key=lambda s: s.get("semester", 0))[0]
                            suggested = next_sem.get("courses", [])[:5]
                    except Exception as e:
                        print(f"⚠️ Could not build next-semester suggestion from plan: {e}")

                if suggested:
                    return finalize_response(
                        f"Here are recommended courses for your next semester (based on your transcript and major {major}):\n"
                        + "\n".join([f"- {_format_course_for_display(c)}" for c in suggested])
                        + "\nIf you'd like a full graduation plan, ask me to generate one."
                    )
                return finalize_response("I couldn't confidently pick next-semester courses from your record. Please ask me to generate a full graduation plan.")

            if has_transcript and stored_pdf_docs:
                return finalize_response("I'm processing your transcript. Please try asking again in a moment, or ask me to generate a full graduation plan.")
            return finalize_response("I need your transcript to recommend next-semester courses. Please upload it or tell me your major/year.")

        student_record = ""
        major = ""
        completed_courses = []
        try:
            if student_data:
                major = student_data.get("major", "") or ""
                completed_courses = student_data.get("completed_courses", [])
                student_record = json.dumps(student_data, indent=2)
        except Exception as e:
            print(f"⚠️ Could not retrieve student record: {e}")
        
        conversation_context = ""
        if conversation_history and len(conversation_history) > 1:
            recent_history = conversation_history[-6:]
            conversation_context = "\n\nPrevious conversation:\n"
            for msg in recent_history:
                role = "Student" if msg.get('role') == 'user' else "AI Advisor"
                conversation_context += f"{role}: {msg.get('content', '')}\n"
        
        system_content = """You are a friendly and helpful AI Academic Advisor. You help students with:
- Course planning and scheduling
- Prerequisite questions
- Major requirements
- Graduation planning
- General academic advice

Be conversational, helpful, and empathetic. Use the conversation history to maintain context and provide personalized advice. If the student asks about their graduation plan, you can generate one. If they ask general questions, answer them naturally using the available course and student information."""
        
        if student_record:
            system_content += f"\n\nStudent Information:\n{student_record}\n"
            if major:
                system_content += f"Major: {major}\n"
            if completed_courses:
                system_content += f"Completed Courses: {', '.join(completed_courses[:10])}\n"
        
        course_context = ""
        policy_context = ""
        if policy_retriever and is_policy_question(user_question):
            try:
                policy_docs = policy_retriever.invoke(user_question)
                if policy_docs:
                    policy_context = "\n\nRelevant Advising/Registrar Policy:\n" + "\n\n".join(
                        doc.page_content[:600] for doc in policy_docs[:3]
                    )
            except Exception as e:
                print(f"⚠️ Could not retrieve policy information: {e}")

        try:
            if query_type == 'course':
                if course_retriever:
                    relevant_docs = course_retriever.invoke(user_question)
                    if relevant_docs:
                            course_context = "\n\nRelevant Course Information:\n" + "\n\n".join(
                            doc.page_content[:500] for doc in relevant_docs[:3]
                        )
            elif query_type == 'major':
                if majors_retriever:
                    relevant_docs = majors_retriever.invoke(user_question)
                    if relevant_docs:
                        course_context = "\n\nRelevant Major Requirements:\n" + "\n\n".join(
                            doc.page_content[:500] for doc in relevant_docs[:3]
                            )
            elif query_type == 'transcript':
                if pdf_retriever:
                    relevant_docs = pdf_retriever.invoke(user_question)
                    if relevant_docs:
                        course_context = "\n\nRelevant Transcript Information:\n" + "\n\n".join(
                            doc.page_content[:500] for doc in relevant_docs[:3]
                        )
            else:
                if course_retriever:
                    course_docs = course_retriever.invoke(user_question)
                    if course_docs:
                        course_context += "\n\nRelevant Course Information:\n" + "\n\n".join(
                            doc.page_content[:500] for doc in course_docs[:2]
                        )
                if majors_retriever:
                    major_docs = majors_retriever.invoke(user_question)
                    if major_docs:
                        course_context += "\n\nRelevant Major Requirements:\n" + "\n\n".join(
                            doc.page_content[:500] for doc in major_docs[:2]
                        )
            if policy_context:
                course_context += policy_context
        except Exception as e:
            print(f"⚠️ Could not retrieve course information: {e}")
        
        full_prompt = f"{conversation_context}\n\nCurrent Question: {user_question}"
        if course_context:
            full_prompt += course_context

        # If we have little/no context from our data, add web search for academic/policy questions
        if (not course_context or len(course_context.strip()) < 300) and is_academic_or_policy_question(user_question):
            try:
                from duckduckgo_search import DDGS
                search_query = f"Penn State {user_question}" if "penn state" not in user_question.lower() and "psu" not in user_question.lower() else user_question
                _trace_tool(trace, "duckduckgo_search", "augment llm prompt context")
                with DDGS() as ddgs:
                    results = list(ddgs.text(search_query, max_results=4))
                _trace_event(trace, "execute", "Web context query executed", {"query": search_query, "results": len(results)})
                if results:
                    web_context = "\n\n".join(
                        f"[{i+1}] {r.get('title', '')}\n{r.get('body', '')}"
                        for i, r in enumerate(results[:4])
                    )
                    full_prompt += f"\n\nWeb search results (use if our data doesn't have the answer):\n{web_context}"
            except Exception as e:
                print(f"⚠️ Web search for context failed: {e}")
                _trace_tool(trace, "duckduckgo_search", "augment llm prompt context", status="failed", metadata={"error": str(e)})

        response = llm2.invoke([
            SystemMessage(content=system_content),
            HumanMessage(content=full_prompt)
        ]).content

        # If response indicates we couldn't find the answer, try dedicated web search
        if _response_indicates_no_answer_found(response):
            web_response = _web_search_and_synthesize(user_question, trace=trace)
            if web_response:
                return finalize_response(web_response)

        return finalize_response(response)
        
    except Exception as e:
        import traceback
        print(f"Error in conversational response: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return finalize_response("I apologize, but I encountered an error processing your question. Please try rephrasing it or ask me to generate a graduation plan if you haven't already uploaded your transcript.")


# =========================
# NEW: Major + Year (no transcript)
# =========================

def generate_plan_from_major_and_year(major, year_level, target_grad_in_years=4):
    """
    Generate a graduation plan when the student does NOT upload a transcript.
    Uses major, year_level, course graph, and CSV-based major requirements.
    """
    global llm1, llm2, course_graph
    
    # Ensure LLMs are initialized
    if llm1 is None or llm2 is None:
        print("⚠️ LLMs not initialized. Initializing now...")
        try:
            setup_environment(None)  # Initialize without transcript
        except Exception as e:
            print(f"⚠️ Error in setup_environment: {e}")
            # Manual initialization as fallback
            from langchain_openai import ChatOpenAI
            if not os.getenv('OPENAI_API_KEY'):
                os.environ['OPENAI_API_KEY'] = "REDACTED"
            if not os.getenv('LANGCHAIN_API_KEY'):
                os.environ['LANGCHAIN_API_KEY'] = 'REDACTED'
            llm1 = ChatOpenAI(model_name="gpt-4o", temperature=0)
            llm2 = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    # Ensure course_graph is available
    if 'course_graph' not in globals() or course_graph is None:
        print("⚠️ Course graph not available. Importing...")
        try:
            from graphExperiment import course_graph as cg
            course_graph = cg
        except Exception as e:
            print(f"⚠️ Could not import course_graph: {e}")
            course_graph = None
    
    # Normalize year input (handles typos and variants)
    year_level = _normalize_year_token(year_level) or year_level
    if year_level not in ['freshman', 'sophomore', 'junior', 'senior']:
        year_level = _fuzzy_year_from_text(year_level or "") or "freshman"

    # Calculate semesters completed and remaining
    typical_total_credits = 120
    year_lower = year_level.lower()

    if year_lower == "freshman":
        assumed_completed = 0
        semesters_completed = 0
    elif year_lower == "sophomore":
        assumed_completed = 30
        semesters_completed = 2
    elif year_lower == "junior":
        assumed_completed = 60
        semesters_completed = 4
    elif year_lower == "senior":
        assumed_completed = 90
        semesters_completed = 6
    else:
        assumed_completed = 0
        semesters_completed = 0

    total_credits_needed = max(0, typical_total_credits - assumed_completed)
    semesters_remaining = 8 - semesters_completed
    credits_per_semester = total_credits_needed / semesters_remaining if semesters_remaining > 0 else 15
    # Keep plans spread across expected timeline (freshman->8, sophomore->6, etc.)
    # Prefer ~15-credit semesters when possible, allow up to 18 only when needed.
    planning_min_credits = 15
    planning_max_credits = 15 if total_credits_needed <= semesters_remaining * 15 else 18
    
    print(f"📊 Student status: {year_level}, {assumed_completed} credits completed, {total_credits_needed} credits needed")
    print(f"📊 Semesters remaining: {semesters_remaining} ({credits_per_semester:.1f} credits per semester)")

    # Normalize major and load CSVs
    major_norm = normalize_major_name(major)
    if not major_norm:
        return f"I couldn't match the major '{major}' to a known program. Please provide the exact major name as listed in the catalog (e.g., 'Business, B.A. (Abington)')."
    courses_csv = os.path.join(os.path.dirname(__file__), "processed_psu_courses.csv")
    majors_csv = os.path.join(os.path.dirname(__file__), "all_abington_majors_combined(5).csv")
    if not os.path.exists(courses_csv):
        alt_courses = os.path.join(os.path.dirname(__file__), "..", "LLM", "processed_psu_courses.csv")
        if os.path.exists(alt_courses):
            courses_csv = alt_courses
    if not os.path.exists(majors_csv):
        alt_majors = os.path.join(os.path.dirname(__file__), "..", "LLM", "all_abington_majors_combined(5).csv")
        if os.path.exists(alt_majors):
            majors_csv = alt_majors

    # Deterministic schedule using graduationFlowchart
    from graduationFlowchart import generate_semester_plan
    try:
        # First build a full plan to infer completed courses for upperclassmen
        full_plan = generate_semester_plan(
            major_name=major_norm,
            completed_courses=[],
            unmet_requirements=[],
            courses_csv_path=courses_csv,
            majors_csv_path=majors_csv,
            credits_needed=typical_total_credits,
            max_semesters=8,
            max_credits_per_sem=planning_max_credits,
            min_credits_per_sem=planning_min_credits
        )
        completed_courses = []
        if semesters_completed > 0:
            for sem in full_plan.get("semesters", []):
                if sem.get("semester", 0) <= semesters_completed:
                    completed_courses.extend(sem.get("courses", []))

        plan_structure = generate_semester_plan(
            major_name=major_norm,
            completed_courses=completed_courses,
            unmet_requirements=[],
            courses_csv_path=courses_csv,
            majors_csv_path=majors_csv,
            credits_needed=total_credits_needed,
            max_semesters=8,
            max_credits_per_sem=planning_max_credits,
            min_credits_per_sem=planning_min_credits
        )
        print(f"✅ Generated deterministic plan with {len(plan_structure.get('semesters', []))} semesters")
    except Exception as e:
        print(f"⚠️ Error generating deterministic plan (no transcript): {e}")
        # Try fuzzy match on major name as a fallback
        try:
            dfm = pd.read_csv(majors_csv)
            try:
                from fuzzywuzzy import process
                best_match = process.extractOne(major_norm, dfm['Major Name'].tolist())
                if best_match and best_match[1] >= 60:
                    print(f"🔄 Retrying with fuzzy matched major: {best_match[0]}")
                    plan_structure = generate_semester_plan(
                        major_name=best_match[0],
                        completed_courses=[],
                        unmet_requirements=[],
                        courses_csv_path=courses_csv,
                        majors_csv_path=majors_csv,
                        credits_needed=total_credits_needed,
                        max_semesters=8,
                        max_credits_per_sem=planning_max_credits,
                        min_credits_per_sem=planning_min_credits
                    )
                else:
                    return "I couldn't generate a plan without your transcript right now. Please upload your transcript or try again."
            except Exception:
                # Fallback to stdlib fuzzy matching if fuzzywuzzy is unavailable
                import difflib
                candidates = difflib.get_close_matches(
                    major_norm,
                    dfm['Major Name'].tolist(),
                    n=1,
                    cutoff=0.6
                )
                if candidates:
                    print(f"🔄 Retrying with difflib matched major: {candidates[0]}")
                    plan_structure = generate_semester_plan(
                        major_name=candidates[0],
                        completed_courses=[],
                        unmet_requirements=[],
                        courses_csv_path=courses_csv,
                        majors_csv_path=majors_csv,
                        credits_needed=total_credits_needed,
                        max_semesters=8,
                        max_credits_per_sem=planning_max_credits,
                        min_credits_per_sem=planning_min_credits
                    )
                else:
                    return "I couldn't generate a plan without your transcript right now. Please upload your transcript or try again."
        except Exception as e2:
            print(f"⚠️ Fuzzy fallback failed: {e2}")
            return "I couldn't generate a plan without your transcript right now. Please upload your transcript or try again."

    # Deterministic formatting (avoid LLM hallucinations)
    start_semester_number = semesters_completed + 1 if semesters_completed > 0 else 1
    return _format_plan_structure(plan_structure, start_semester_number=start_semester_number)


# =========================
# Flask App / Endpoint
# =========================

app = Flask(__name__)

@app.route("/generate_plan", methods=["POST"])
def generate_plan_route():
    """
    Unified endpoint:
    - If 'transcript' file is present → transcript-based plan.
    - Else → expects JSON or form with 'major' and 'year' to generate plan.
    """
    # If there's a transcript file, use transcript-based mode
    if 'transcript' in request.files:
        setup_environment(request)
        plan = aggregate_and_generate_response(
            user_question="Generate my graduation plan from my transcript."
        )
        return jsonify({"plan": plan})

    # Otherwise, expect major + year
    data = request.get_json(silent=True) or request.form
    major = data.get("major")
    year_level = data.get("year")

    if not major or not year_level:
        return jsonify({
            "error": "Please provide 'major' and 'year' (e.g., 'Computer Science, BS (CMPAB_BS)' and 'junior')."
        }), 400

    setup_environment(None)
    plan = generate_plan_from_major_and_year(
        major=major,
        year_level=year_level
    )
    return jsonify({"plan": plan})


if __name__ == "__main__":
    # For local testing
    # Make sure OPENAI_API_KEY and LANGCHAIN_API_KEY are set in your environment first.
    app.run(host="0.0.0.0", port=5000, debug=True)
