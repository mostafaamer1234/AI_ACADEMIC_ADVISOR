from __future__ import annotations

import io
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from fuzzywuzzy import fuzz

matplotlib.use("Agg")

from graduationFlowchart import GraduationFlowchart


class MajorGraph:
    """Wrapper that uses GraduationFlowchart to produce major option flowcharts."""

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.major_names = self.df["Major Name"].tolist()

    def _find_matching_major(self, query: str) -> str | None:
        if not query:
            return None
        query = query.lower().strip()
        best_match = None
        best_score = 0
        for name in self.major_names:
            score = fuzz.partial_ratio(query, name.lower())
            if score > best_score:
                best_score = score
                best_match = name
        if best_score < 60:
            return None
        return best_match

    def _get_major_options(self, major_name: str) -> list[str]:
        options = []
        for col in self.df.columns:
            if "Option" in col:
                # Extract option name by removing " Option Summary" and " Option Courses"
                option_name = col.replace(" Option Summary", "").replace(" Option Courses", "")
                # Handle special case: column is just "Options" -> skip it or use a generic name
                if option_name == "Options":
                    # This is likely a catch-all, use "General" as default or skip
                    option_name = "General"
                # Also remove trailing " Option" if present (e.g., "General Option" -> "General")
                elif option_name.endswith(" Option"):
                    option_name = option_name[:-7].strip()
                val = self.df.loc[self.df["Major Name"] == major_name, col].iloc[0]
                if not pd.isna(val):
                    options.append(option_name)
        return options

    def _create_option_flowcharts(self, flowchart_gen, major_name: str, options: list[str]) -> io.BytesIO:
        """Create flowchart for each option using the flowchart generator."""
        import tempfile
        import matplotlib.image as mpimg

        num_options = len(options)
        fig, axes = plt.subplots(num_options, 1, figsize=(20, 12 * num_options))
        if num_options == 1:
            axes = [axes]

        for i, option in enumerate(options):
            buf = flowchart_gen.generate_flowchart(major_name, option)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(buf.read())
                tmp_path = tmp.name

            try:
                img = mpimg.imread(tmp_path)
                axes[i].imshow(img)
                axes[i].axis("off")
                axes[i].set_title(f"{major_name} - {option} Option", fontsize=14, fontweight="bold", pad=10)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        plt.tight_layout()
        out = io.BytesIO()
        plt.savefig(out, format="png", dpi=150, bbox_inches="tight")
        plt.close()
        out.seek(0)
        return out

    def draw_major_graph(self, major_name: str) -> io.BytesIO:
        # Match major
        actual_major_name = self._find_matching_major(major_name)
        if not actual_major_name:
            raise ValueError(f"Could not find a matching major for '{major_name}'. Please try a different name.")

        # Options
        options = self._get_major_options(actual_major_name)

        # Initialize flowchart generator
        csv_dir = os.path.dirname(__file__)
        courses_csv = os.path.join(csv_dir, "processed_psu_courses.csv")
        majors_csv = os.path.join(csv_dir, "all_abington_majors_combined(5).csv")
        if not os.path.exists(courses_csv):
            courses_csv = os.path.join(csv_dir, "..", "LLM", "processed_psu_courses.csv")
        if not os.path.exists(majors_csv):
            majors_csv = os.path.join(csv_dir, "..", "LLM", "all_abington_majors_combined(5).csv")

        flowchart_gen = GraduationFlowchart(courses_csv, majors_csv)

        if not options:
            return flowchart_gen.generate_flowchart(actual_major_name)
        else:
            return self._create_option_flowcharts(flowchart_gen, actual_major_name, options)

