"""Export the referee's parsed catalog and code for the static site.

Writes site/catalog.json (course codes and parsed prerequisite clauses) and copies
evals/plan_checker.py to site/, so the page runs the same checker in the browser.

    python -m evals.export_site_catalog
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from evals.plan_checker import Catalog

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    cat = Catalog.load()
    payload = {"courses": sorted(cat.courses), "prereqs": {k: [sorted(c) for c in v] for k, v in cat.prereqs.items()}}
    (ROOT / "site").mkdir(exist_ok=True)
    (ROOT / "site" / "catalog.json").write_text(json.dumps(payload, separators=(",", ":")))
    shutil.copy(ROOT / "evals" / "plan_checker.py", ROOT / "site" / "plan_checker.py")
    print(len(cat.courses), "courses,", len(cat.prereqs), "with prerequisites")


if __name__ == "__main__":
    main()
