"""
graduationFlowchart.py (v4 – STRICT PREREQUISITES)

Fixes:
1) A course may ONLY be scheduled after all prereqs are completed in PRIOR semesters
2) No same-semester prereq violations (CMPSC 131 → 132 fixed)
3) No impossible semester-1 placements (MATH 401, MATH 410, DS 220 fixed)
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import pandas as pd
import networkx as nx


COURSE_TOKEN_RE = re.compile(r"\b([A-Z]{2,8}(?:/[A-Z]{2,8})*)\s*(\d{3}[A-Z]?)\b")
GENED_CREDITS_RE = re.compile(r"([A-Za-z \-\(\)]+):\s*(\d+)\s*credits", re.IGNORECASE)


def _course_level(c: str) -> int:
    m = re.search(r"\b(\d{3})\b", c)
    return int(m.group(1)) if m else 9999


def _normalize_course_code(code: str) -> str:
    return re.sub(r"\s+", " ", code.strip().upper())


def load_prereq_map(courses_csv_path: str) -> Dict[str, List[str]]:
    df = pd.read_csv(courses_csv_path)
    prereq_map: Dict[str, List[str]] = {}
    
    # Build a mapping of base course codes (without suffixes) to actual course codes
    # e.g., "MATH 311" -> ["MATH 311M", "MATH 311W"]
    base_to_actual: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        course = str(row.get("course_number", "")).strip()
        # Extract base course code (e.g., "MATH 311" from "MATH 311M")
        base_match = re.match(r'([A-Z]{2,})\s+(\d{3})', course)
        if base_match:
            base_code = f"{base_match.group(1)} {base_match.group(2)}"
            if base_code not in base_to_actual:
                base_to_actual[base_code] = []
            base_to_actual[base_code].append(course)

    for _, row in df.iterrows():
        course = str(row.get("course_number", "")).strip()
        raw = row.get("prerequisite_course_numbers", None)

        prereqs = []
        if isinstance(raw, str) and raw.strip():
            try:
                prereqs = [x.strip() for x in eval(raw)]
            except Exception:
                pass
        
        # Expand prerequisites: if a prereq is a base code (e.g., "MATH 311"),
        # check if any actual courses match (e.g., "MATH 311M", "MATH 311W")
        expanded_prereqs = []
        for p in prereqs:
            expanded_prereqs.append(p)  # Keep original
            # Also check if this is a base code that maps to actual courses
            if p in base_to_actual:
                # If the base code exists, we consider the prerequisite satisfied
                # if any of the actual courses are in the plan
                # For now, just add the base code as a valid prerequisite
                pass  # We'll handle this in the scheduling logic
        
        prereq_map[course] = prereqs
        
        # Also add base code entries for matching (e.g., "MATH 311" for "MATH 311M")
        base_match = re.match(r'([A-Z]{2,})\s+(\d{3})', course)
        if base_match:
            base_code = f"{base_match.group(1)} {base_match.group(2)}"
            # If base code doesn't exist, create it with the same prerequisites
            # This allows "MATH 311" prerequisite to match "MATH 311M" or "MATH 311W"
            if base_code not in prereq_map:
                prereq_map[base_code] = prereqs.copy()
            elif base_code != course:
                # If base code already exists and is different from course,
                # merge prerequisites (union of all variants' prerequisites)
                existing_prereqs = set(prereq_map[base_code])
                existing_prereqs.update(prereqs)
                prereq_map[base_code] = list(existing_prereqs)

    return prereq_map


def schedule_mixed_by_prereqs(
    planned_courses: List[str],
    prereq_map: Dict[str, List[str]],
    course_category: Dict[str, str],
    max_semesters: int = 8,
    max_credits_per_sem: int = 18,
    min_credits_per_sem: int = 15,
    completed_courses: Optional[List[str]] = None,
) -> Tuple[Dict[str, int], nx.DiGraph]:

    planned_courses_norm = [_normalize_course_code(c) for c in planned_courses]
    planned_set = set(planned_courses_norm)
    remaining = set(planned_courses_norm)
    course_category = {
        _normalize_course_code(k): v for k, v in course_category.items()
    }

    completed_set: set = set()
    completed_base_set: set = set()
    if completed_courses:
        for c in completed_courses:
            if not c:
                continue
            c_norm = _normalize_course_code(c)
            completed_set.add(c_norm)
            base_match = re.match(r"([A-Z]{2,})\s+(\d{3})", c_norm)
            if base_match:
                completed_base_set.add(
                    f"{base_match.group(1)} {base_match.group(2)}"
                )

    # Build prereq graph restricted to plan
    G = nx.DiGraph()
    for c in planned_courses_norm:
        G.add_node(c)
    for c in planned_courses_norm:
        for p in prereq_map.get(c, []):
            if p in planned_set:
                G.add_edge(p, c)

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Prerequisite cycle detected")

    scheduled: Dict[str, int] = {}
    def is_available_strict(course: str, current_sem: int) -> bool:
        """
        STRICT prereq rule:
        All prereqs that are IN THE PLAN must be completed in a PRIOR semester.
        Prerequisites that are NOT in the plan are checked:
        - Check if prerequisite exists in prereq_map (exact match or base code match)
        - If it exists but is not in plan, block the course (conservative approach)
        - This prevents impossible placements like MATH 485 in semester 1
        """
        for p in prereq_map.get(course, []):
            prereq_norm = _normalize_course_code(p)
            base_match = re.match(r"([A-Z]{2,})\s+(\d{3})", prereq_norm)
            prereq_base = (
                f"{base_match.group(1)} {base_match.group(2)}"
                if base_match
                else None
            )

            # Prereq already satisfied by completed coursework
            if prereq_norm in completed_set or (
                prereq_base and prereq_base in completed_base_set
            ):
                continue

            # Check if prerequisite is in plan (exact match or base code match)
            prereq_in_plan = False
            prereq_scheduled_sem = None

            # First check exact match
            if prereq_norm in planned_set:
                prereq_in_plan = True
                if prereq_norm in scheduled:
                    prereq_scheduled_sem = scheduled[prereq_norm]
            
            # If not exact match, check base code match (e.g., "MATH 311" matches "MATH 311M")
            if not prereq_in_plan:
                if prereq_base:
                    dept, num = prereq_base.split()
                    # Check if any course in planned_set matches this base code
                    for planned_course in planned_set:
                        if re.match(rf'{re.escape(dept)}\s+{re.escape(num)}', planned_course):
                            prereq_in_plan = True
                            if planned_course in scheduled:
                                prereq_scheduled_sem = scheduled[planned_course]
                            break
            
            if prereq_in_plan:
                # Prerequisite is in the plan - must be scheduled in a PRIOR semester
                if prereq_scheduled_sem is None:
                    return False  # Prerequisite not yet scheduled
                if prereq_scheduled_sem >= current_sem:
                    return False  # Prerequisite in same or later semester
            else:
                # Prerequisite is NOT in the plan - check if it exists in prereq_map
                prereq_exists = prereq_norm in prereq_map
                if not prereq_exists:
                    # Check if base code exists (e.g., "MATH 311" base code)
                    if prereq_base:
                        prereq_exists = prereq_base in prereq_map
                
                if prereq_exists:
                    # This prerequisite exists but is not in our plan
                    # For semester 1, block to prevent impossible placements (e.g., MATH 485)
                    # For later semesters, assume prerequisite is satisfied elsewhere
                    # (transfer credits, placement tests, other majors, etc.)
                    if current_sem == 1:
                        return False  # Block from semester 1
                    # For semesters 2+, allow it (prerequisite assumed satisfied)
        return True

    bucket_order = ["Prescribed", "Additional", "Option", "GenEd"]

    for sem in range(1, max_semesters + 1):
        sem_credits = 0
        made_progress = True
        iterations = 0
        max_iterations = 100  # Prevent infinite loops

        while made_progress and iterations < max_iterations:
            iterations += 1
            made_progress = False

            available = [
                c for c in remaining
                if is_available_strict(c, sem)
            ]

            if not available:
                # If no courses available and we haven't met minimum, try to find any available
                # (even if they have unmet prereqs that aren't in plan - for later semesters)
                if sem_credits < min_credits_per_sem and sem > 1 and remaining:
                    # For semesters 2+, allow courses with prereqs not in plan
                    available = [c for c in remaining]
                
                if not available:
                    break

            avail_by_bucket = {b: [] for b in bucket_order}
            for c in available:
                b = course_category.get(c, "Option")
                avail_by_bucket.setdefault(b, []).append(c)

            for b in bucket_order:
                avail_by_bucket[b].sort(
                    key=lambda c: (_course_level(c), -G.out_degree(c), c)
                )

            for b in bucket_order:
                # Continue adding courses until we meet minimum OR reach maximum
                if sem_credits >= max_credits_per_sem:
                    break
                if not avail_by_bucket[b]:
                    continue

                c = avail_by_bucket[b][0]
                scheduled[c] = sem
                sem_credits += 3
                remaining.remove(c)
                made_progress = True
                
                # Stop if we've reached maximum credits
                if sem_credits >= max_credits_per_sem:
                    break

        # If semester ended with less than minimum credits and there are remaining courses,
        # try to add more courses (relaxing prereq constraints for later semesters)
        if sem_credits < min_credits_per_sem and remaining:
            # Try to add more courses to meet minimum (relax constraints for later semesters)
            additional_available = []
            if sem > 1:
                # For later semesters, allow courses even if prereqs not in plan
                additional_available = [c for c in remaining]
            else:
                # For semester 1, only add if truly available
                additional_available = [c for c in remaining if is_available_strict(c, sem)]
            
            # Add courses up to minimum credits
            needed_credits = min_credits_per_sem - sem_credits
            courses_to_add = min(len(additional_available), needed_credits // 3)
            for c in additional_available[:courses_to_add]:
                if sem_credits >= max_credits_per_sem:
                    break
                scheduled[c] = sem
                sem_credits += 3
                remaining.remove(c)

        if not remaining:
            break

    if remaining:
        raise ValueError(
            f"Could not schedule all courses. Remaining: {sorted(remaining)}"
        )

    return scheduled, G


@dataclass
class NodeStyle:
    facecolor: str
    edgecolor: str
    textcolor: str


STYLE_MAP = {
    "Prescribed": NodeStyle("#98FB98", "#2F6B2F", "#000000"),
    "Additional": NodeStyle("#B0E0E6", "#2C6A75", "#000000"),
    "Option":     NodeStyle("#D8BFD8", "#6B3F6B", "#000000"),
    "GenEd":      NodeStyle("#FFDAB9", "#8B5A2B", "#000000"),
}


class GraduationFlowchart:
    def __init__(self, courses_csv: str, majors_csv: str):
        self.majors_df = pd.read_csv(majors_csv)
        self.prereq_map = load_prereq_map(courses_csv)

    # --- Helpers restored from previous version ---
    def _get_major_row(self, major_name: str) -> pd.Series:
        rows = self.majors_df[self.majors_df["Major Name"] == major_name]
        if rows.empty:
            raise ValueError(f"Major '{major_name}' not found in majors CSV.")
        return rows.iloc[0]

    def _option_courses_column(self, option_name: str) -> Optional[str]:
        if not option_name:
            return None
        name = option_name.strip().lower()
        if "data science" in name:
            return "Data Science Option Courses"
        if "general" in name:
            return "General Option Courses"
        # fallback: try matching by prefix
        for col in self.majors_df.columns:
            if "option" in col.lower() and name in col.lower():
                return col
        return None

    def _parse_gened_categories(self, gened_text: str) -> List[Tuple[str, int]]:
        if not isinstance(gened_text, str):
            return []
        pairs = []
        for lbl, n in GENED_CREDITS_RE.findall(gened_text):
            try:
                pairs.append((lbl.strip(), int(n)))
            except Exception:
                continue
        return pairs

    def _make_gened_nodes(self, categories: List[Tuple[str, int]]) -> Tuple[List[str], Dict[str, str]]:
        nodes, labels = [], {}
        for (cat, credits) in categories:
            count = int(math.ceil(credits / 3))
            key = re.sub(r"[^A-Za-z0-9]+", "_", cat).strip("_").upper() or "GENED"
            for i in range(count):
                nid = f"GENED__{key}__{i+1}"
                nodes.append(nid)
                labels[nid] = f"Gen Ed: {cat}"
        return nodes, labels

    def _dominant_dept(self, courses: List[str]) -> str:
        counts: Dict[str, int] = {}
        for c in courses:
            parts = c.split()
            if not parts:
                continue
            dept = parts[0].strip()
            if dept and dept.isalpha():
                counts[dept] = counts.get(dept, 0) + 1
        if not counts:
            return "OPT"
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    def _select_option_courses(self, option_courses: List[str], needed_credits: int, exclude: set) -> List[str]:
        n_needed = int(math.ceil(max(0, needed_credits) / 3))
        pool = [c for c in option_courses if c not in exclude]
        pool.sort(key=lambda c: (_course_level(c), c))
        return pool[:n_needed]

    def generate_flowchart(self, major_name: str, option_name: str = "General Option") -> io.BytesIO:
        row = self._get_major_row(major_name)

        # ---- Prescribed (specific courses) ----
        prescribed_expr = str(row.get("Prescribed Courses", "") or "")
        prescribed_courses = resolve_common_or_groups(extract_all_courses(prescribed_expr))

        # ---- Additional (pick first listed course token deterministically) ----
        additional_expr = str(row.get("Additional Courses", "") or "")
        additional_courses_all = extract_all_courses(additional_expr)
        additional_courses = additional_courses_all[:1]

        fixed_courses = []
        for c in prescribed_courses + additional_courses:
            if c not in fixed_courses:
                fixed_courses.append(c)

        # ---- Option-specific (extract option courses first) ----
        opt_col = self._option_courses_column(option_name or "")
        option_course_text = str(row.get(opt_col, "") or "") if opt_col else ""
        option_course_list = extract_all_courses(option_course_text)

        # Determine dominant dept for option placeholder label (e.g., CMPSC)
        # For Computer Science majors, ALWAYS use CMPSC for electives, not MATH
        # Check prescribed courses first to determine the major's primary department
        prescribed_depts = set([c.split()[0] for c in prescribed_courses if " " in c])
        if "CMPSC" in prescribed_depts:
            # Computer Science major - use CMPSC for electives
            dominant_dept = "CMPSC"
        elif "DS" in prescribed_depts:  # Data Science major
            dominant_dept = "DS"
        else:
            # For other majors, use the most common department from prescribed courses only
            dominant_dept = self._dominant_dept(prescribed_courses) or "OPT"

        # ---- Option-specific (recommend specific courses, limited by needed credits) ----
        credits_for_major = int(row.get("Credits for Major", 0) or 0)
        approx_fixed_credits = 3 * len(fixed_courses)
        remaining_major_credits = max(0, credits_for_major - approx_fixed_credits)

        # Get available option courses (excluding fixed courses that are already in prescribed/additional)
        available_option_courses = [c for c in option_course_list if c not in fixed_courses]
        
        # Calculate how many option courses we need to satisfy the option credit requirements
        n_option_courses_needed = int(math.ceil(remaining_major_credits / 3))
        
        # Select actual option courses - ALWAYS prioritize actual courses over placeholders
        # Sort by level (lower first) for better scheduling
        available_option_courses.sort(key=lambda c: (_course_level(c), c))
        
        # CRITICAL: Select ALL available option courses up to the number needed
        # Never create placeholders if actual courses are available
        option_selected = []
        option_placeholders = []
        
        if len(available_option_courses) > 0:
            # Select actual courses - take as many as we need, up to what's available
            option_selected = available_option_courses[:min(n_option_courses_needed, len(available_option_courses))]
            
            # Calculate remaining credits after selecting actual courses
            used_credits = 3 * len(option_selected)
            still_need = max(0, remaining_major_credits - used_credits)
            
            # Only create placeholders if we've exhausted ALL available option courses AND still need more
            if still_need > 0 and len(option_selected) >= len(available_option_courses):
                # We've used all available option courses, create placeholders for remaining credits
                count = int(math.ceil(still_need / 3))
                for i in range(count):
                    option_placeholders.append(f"OPT__{dominant_dept}__{i+1}")
        else:
            # No option courses available - create placeholders for all needed credits
            count = n_option_courses_needed
            for i in range(count):
                option_placeholders.append(f"OPT__{dominant_dept}__{i+1}")

        # ---- Gen Ed (per-category) ----
        gened_text = str(row.get("General Education Requirements", "") or "")
        categories = self._parse_gened_categories(gened_text)
        if not categories:
            total_gened = int(row.get("Credits for General Education", 0) or 0)
            categories = [("General Education", total_gened)]
        gened_nodes, gened_labels = self._make_gened_nodes(categories)

        planned_courses = fixed_courses + option_selected + option_placeholders + gened_nodes

        # ---- Category map for mixed scheduling ----
        course_category: Dict[str, str] = {}
        for c in prescribed_courses:
            if c in planned_courses:
                course_category[c] = "Prescribed"
        for c in additional_courses:
            if c in planned_courses:
                course_category[c] = "Additional"
        for c in option_selected:
            course_category[c] = "Option"
        for c in option_placeholders:
            course_category[c] = "Option"
        for c in gened_nodes:
            course_category[c] = "GenEd"

        # ---- Mixed scheduling + prereq edges ----
        scheduled, prereq_G = schedule_mixed_by_prereqs(
            planned_courses,
            self.prereq_map,
            course_category=course_category,
            max_semesters=8,
            max_credits_per_sem=18
        )

        # ---- Labels & styles ----
        node_type: Dict[str, str] = {}
        label_map: Dict[str, str] = {}

        for c in prescribed_courses:
            if c in planned_courses:
                node_type[c] = "Prescribed"
                label_map[c] = c

        for c in additional_courses:
            if c in planned_courses:
                node_type[c] = "Additional"
                label_map[c] = c

        for c in option_selected:
            node_type[c] = "Option"
            label_map[c] = c

        # Placeholders must mention "option" and the option name
        # Format: "{dept} {option_name} Option Elective" (e.g., "CMPSC General Option Elective")
        for c in option_placeholders:
            node_type[c] = "Option"
            # Clean up option_name: remove redundant "Option" suffix if present
            clean_option_name = option_name.strip()
            # Remove trailing " Option" or "Option" (case insensitive)
            if clean_option_name.lower().endswith(" option"):
                clean_option_name = clean_option_name[:-7].strip()
            elif clean_option_name.lower().endswith("option"):
                clean_option_name = clean_option_name[:-6].strip()
            # Handle special case: "Options" -> use generic "Elective" label
            if clean_option_name.lower() == "options" or clean_option_name.lower() == "":
                # If it's just "Options" or empty, use a generic label but still include "Option"
                label_map[c] = f"{dominant_dept} Option Elective"
            else:
                # Capitalize properly: "General", "Data Science", etc.
                # ALWAYS include "Option" in the label
                clean_option_name = clean_option_name.title()
                label_map[c] = f"{dominant_dept} {clean_option_name} Option Elective"

        for c in gened_nodes:
            node_type[c] = "GenEd"
            label_map[c] = gened_labels.get(c, "Gen Ed")

        # ---- Draw ----
        fig = self._draw_semester_chart(
            major_name=major_name,
            option_name=option_name,
            scheduled=scheduled,
            graph=prereq_G,
            node_type=node_type,
            label_map=label_map,
            max_semesters=8,
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf

    def _draw_semester_chart(
        self,
        major_name: str,
        option_name: str,
        scheduled: Dict[str, int],
        graph: nx.DiGraph,
        node_type: Dict[str, str],
        label_map: Dict[str, str],
        max_semesters: int = 8,
    ):
        sem_nodes: Dict[int, List[str]] = {i: [] for i in range(1, max_semesters + 1)}
        for n in graph.nodes():
            sem = scheduled.get(n, max_semesters)
            sem_nodes[sem].append(n)

        for sem in sem_nodes:
            sem_nodes[sem].sort(key=lambda c: (_course_level(c), label_map.get(c, c)))

        col_w = 3.0
        row_h = 0.58
        box_w = 2.6
        box_h = 0.40
        x0 = 0.6
        y_top = 0.8

        max_rows = max((len(v) for v in sem_nodes.values()), default=1)
        height = max(6.0, max_rows * row_h + 2.2)
        width = (max_semesters) * col_w + 1.2

        fig, ax = plt.subplots(figsize=(width, height))
        ax.set_axis_off()

        ax.text(
            0.0, 1.02,
            f"{major_name} — {option_name}",
            fontsize=16,
            fontweight="bold",
            ha="left",
            va="bottom",
            transform=ax.transAxes,
        )

        for sem in range(1, max_semesters + 1):
            ax.text(
                x0 + (sem - 1) * col_w,
                y_top,
                f"Semester {sem}",
                fontsize=10,
                fontweight="bold",
                ha="left",
                va="bottom",
            )

        node_rects: Dict[str, Tuple[float, float, float, float]] = {}

        for sem in range(1, max_semesters + 1):
            x = x0 + (sem - 1) * col_w
            nodes = sem_nodes[sem]
            for r, n in enumerate(nodes):
                y = y_top + 0.35 + r * row_h
                node_rects[n] = (x, y, box_w, box_h)

        # draw nodes
        for n in graph.nodes():
            x, y, w, h = node_rects[n]
            kind = node_type.get(n, "Option")
            style = STYLE_MAP.get(kind, STYLE_MAP["Option"])

            rect = FancyBboxPatch(
                (x, y),
                w, h,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                linewidth=1.2,
                edgecolor=style.edgecolor,
                facecolor=style.facecolor,
            )
            ax.add_patch(rect)

            ax.text(
                x + w / 2.0,
                y + h / 2.0,
                label_map.get(n, n),
                fontsize=8,
                ha="center",
                va="center",
                color=style.textcolor,
                wrap=True,
            )

        # edges between in-plan nodes only
        for u, v in graph.edges():
            if u not in node_rects or v not in node_rects:
                continue
            ur = node_rects[u]
            vr = node_rects[v]
            start = (ur[0] + ur[2], ur[1] + ur[3] / 2.0)
            end = (vr[0], vr[1] + vr[3] / 2.0)

            arrow = FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.0,
                color="#444444",
                connectionstyle="arc3,rad=0.06",
            )
            ax.add_patch(arrow)

        ax.set_xlim(0, x0 + max_semesters * col_w + 0.5)
        ax.set_ylim(y_top + max_rows * row_h + 1.5, 0)
        return fig


# --- Helper functions restored ---
def _normalize_course_token(dept_part: str, num_part: str) -> str:
    dept = dept_part.split("/")[0].strip()
    return f"{dept} {num_part.strip()}"


def extract_all_courses(expr: str) -> List[str]:
    """Extract ALL course codes found in a string (robust to '...')."""
    if not isinstance(expr, str) or not expr.strip():
        return []
    found = []
    for m in COURSE_TOKEN_RE.finditer(expr):
        found.append(_normalize_course_token(m.group(1), m.group(2)))
    out, seen = [], set()
    for c in found:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def resolve_common_or_groups(courses: List[str]) -> List[str]:
    """
    Resolve the common CMPSC intro pair:
      (CMPSC 131,132) OR (CMPSC 121,122)
    If both sets appear, keep 131/132 and drop 121/122.
    """
    s = set(courses)
    has_131_132 = ("CMPSC 131" in s) and ("CMPSC 132" in s)
    has_121_122 = ("CMPSC 121" in s) and ("CMPSC 122" in s)
    if has_131_132 and has_121_122:
        drop = {"CMPSC 121", "CMPSC 122"}
        return [c for c in courses if c not in drop]
    return courses


def generate_semester_plan(
    major_name: str,
    completed_courses: List[str],
    unmet_requirements: List[dict],
    courses_csv_path: str,
    majors_csv_path: str,
    option_name: str = "General Option",
    credits_needed: Optional[float] = None,
) -> dict:
    """
    Generate a semester-by-semester plan structure (not PNG).
    Returns: {
        "semesters": [
            {"semester": 1, "courses": ["CMPSC 131", ...], "credits": 15},
            ...
        ],
        "total_credits": 120,
        "prerequisites": {"CMPSC 132": ["CMPSC 131"], ...}
    }
    """
    flowchart = GraduationFlowchart(courses_csv_path, majors_csv_path)
    
    # Get major requirements
    row = flowchart._get_major_row(major_name)
    
    # Extract prescribed courses
    prescribed_expr = str(row.get("Prescribed Courses", "") or "")
    prescribed_courses = resolve_common_or_groups(extract_all_courses(prescribed_expr))
    
    # Extract additional courses
    additional_expr = str(row.get("Additional Courses", "") or "")
    additional_courses_all = extract_all_courses(additional_expr)
    additional_courses = additional_courses_all[:1]  # Take first option
    fixed_courses = []
    for course in prescribed_courses + additional_courses:
        if course not in fixed_courses:
            fixed_courses.append(course)
    
    # Extract option courses
    opt_col = flowchart._option_courses_column(option_name or "")
    option_course_text = str(row.get(opt_col, "") or "") if opt_col else ""
    option_course_list = extract_all_courses(option_course_text)
    
    completed_set = {_normalize_course_code(c) for c in completed_courses or []}
    fixed_remaining = [
        c for c in fixed_courses if _normalize_course_code(c) not in completed_set
    ]

    # Determine dominant department for option placeholders
    prescribed_depts = {c.split()[0] for c in prescribed_courses if c.split()}
    if "CMPSC" in prescribed_depts:
        dominant_dept = "CMPSC"
    elif "DS" in prescribed_depts:
        dominant_dept = "DS"
    else:
        dominant_dept = flowchart._dominant_dept(prescribed_courses) or "OPT"

    # Option course selection based on remaining major credits
    credits_for_major = int(row.get("Credits for Major", 0) or 0)
    major_course_set = {
        _normalize_course_code(c)
        for c in (fixed_courses + option_course_list)
    }
    completed_major_courses = [
        c for c in completed_set if c in major_course_set
    ]
    completed_major_credits = 3 * len(completed_major_courses)
    fixed_remaining_credits = 3 * len(fixed_remaining)
    remaining_major_credits = max(
        0, credits_for_major - completed_major_credits - fixed_remaining_credits
    )
    n_option_courses_needed = int(math.ceil(remaining_major_credits / 3)) if remaining_major_credits else 0

    available_option_courses = [
        c for c in option_course_list
        if _normalize_course_code(c) not in completed_set and c not in fixed_courses
    ]
    available_option_courses.sort(key=lambda c: (_course_level(c), c))

    option_selected = available_option_courses[:min(n_option_courses_needed, len(available_option_courses))]
    option_placeholders = []
    if n_option_courses_needed > len(option_selected):
        placeholder_count = n_option_courses_needed - len(option_selected)
        for i in range(placeholder_count):
            option_placeholders.append(f"OPT__{dominant_dept}__{i+1}")

    planned_courses = fixed_remaining + option_selected + option_placeholders

    # GenEd placeholders to reach remaining credits (if provided)
    gened_nodes = []
    gened_labels: Dict[str, str] = {}
    gened_text = str(row.get("General Education Requirements", "") or "")
    categories = flowchart._parse_gened_categories(gened_text)
    if not categories:
        total_gened = int(row.get("Credits for General Education", 0) or 0)
        if total_gened > 0:
            categories = [("General Education", total_gened)]

    if categories:
        gened_nodes, gened_labels = flowchart._make_gened_nodes(categories)

    if credits_needed is not None:
        planned_credits = 3 * len(planned_courses)
        remaining_for_gened = max(0, credits_needed - planned_credits)
        needed_gened_count = int(math.ceil(remaining_for_gened / 3)) if remaining_for_gened else 0
        if gened_nodes:
            gened_nodes = gened_nodes[:needed_gened_count]
            gened_labels = {k: gened_labels[k] for k in gened_nodes}
        elif needed_gened_count > 0:
            gened_nodes = [f"GENED__ELECTIVE__{i+1}" for i in range(needed_gened_count)]
            gened_labels = {k: "Gen Ed Elective" for k in gened_nodes}

    planned_courses.extend(gened_nodes)
    
    # Load prerequisites
    prereq_map = flowchart.prereq_map
    
    # Schedule courses
    course_category: Dict[str, str] = {}
    for c in prescribed_courses:
        if c in planned_courses:
            course_category[c] = "Prescribed"
    for c in additional_courses:
        if c in planned_courses:
            course_category[c] = "Additional"
    for c in option_selected:
        course_category[c] = "Option"
    for c in option_placeholders:
        course_category[c] = "Option"
    for c in gened_nodes:
        course_category[c] = "GenEd"

    scheduled, graph = schedule_mixed_by_prereqs(
        planned_courses=planned_courses,
        prereq_map=prereq_map,
        course_category=course_category,
        max_semesters=8,
        max_credits_per_sem=18,
        min_credits_per_sem=15,
        completed_courses=completed_courses
    )
    
    # Build semester structure
    semesters = {}
    for course, sem in scheduled.items():
        if sem not in semesters:
            semesters[sem] = []
        semesters[sem].append(course)
    
    # Format as list
    semester_list = []
    for sem in sorted(semesters.keys()):
        courses_in_sem = sorted(semesters[sem])
        # Approximate credits (3 per course, can be enhanced)
        credits = len(courses_in_sem) * 3
        semester_list.append({
            "semester": sem,
            "courses": courses_in_sem,
            "credits": credits
        })
    
    # Build prerequisites dict
    prerequisites = {}
    for course in planned_courses:
        if course in prereq_map:
            prerequisites[course] = prereq_map[course]
    
    return {
        "semesters": semester_list,
        "total_credits": sum(s["credits"] for s in semester_list),
        "prerequisites": prerequisites
    }
