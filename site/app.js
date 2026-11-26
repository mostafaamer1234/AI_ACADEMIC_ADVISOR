// Runs evals/plan_checker.py (copied by python -m evals.export_site_catalog) in Pyodide.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const EXAMPLES = {
  "Valid plan": `Semester 1: CMPSC 131, MATH 140, ENGL 15
Total Credits: 13
Semester 2: CMPSC 132, MATH 141, CMPSC 360
Total Credits: 13
Semester 3: CMPSC 221, CMPSC 465, STAT 318
Total Credits: 13`,
  "Out of order": `Semester 1: CMPSC 131, MATH 140, CMPSC 465
Total Credits: 13
Semester 2: CMPSC 132, MATH 141, CMPSC 360
Total Credits: 13`,
  "Made-up course": `Semester 1: CMPSC 131, MATH 140, CMPSC 999, ENGL 15
Total Credits: 13
Semester 2: CMPSC 132, MATH 140, MATH 141
Total Credits: 22`,
};
let py = null;

function render(r) {
  const row = (label, items) => `<li><b>${label}<em class="${items.length ? "bad" : "ok"}">${items.length || "None"}</em></b>${items.map((i) => `<span>${esc(i)}</span>`).join("")}</li>`;
  $("report").innerHTML = `
    <p class="verdict ${r.passed ? "pass" : "fail"}">${r.passed ? "Plan passes." : "Plan fails."}</p>
    <p class="q-small">${r.semesters} semester${r.semesters === 1 ? "" : "s"}, ${r.courses} course${r.courses === 1 ? "" : "s"} checked.</p>
    <ul class="checks">
      ${row("Courses not in the catalog", r.unknown_courses)}
      ${row("Prerequisites out of order", r.prereq_violations)}
      ${row("Duplicate courses", r.duplicate_courses)}
      ${row("Credit loads outside 12–19", r.credit_issues)}
    </ul>`;
}

function check() {
  if (!py) return;
  py.globals.set("PLAN", $("plan").value);
  render(JSON.parse(py.runPython("json.dumps(check_plan(PLAN, CAT).to_dict())")));
}

function load(name) {
  document.querySelectorAll("#examples button").forEach((b) => b.setAttribute("aria-pressed", String(b.textContent === name)));
  $("plan").value = EXAMPLES[name];
  check();
}

$("examples").innerHTML = Object.keys(EXAMPLES).map((n) => `<button aria-pressed="false">${n}</button>`).join("");
document.querySelectorAll("#examples button").forEach((b) => b.addEventListener("click", () => load(b.textContent)));
$("check").addEventListener("click", check);
let t = null;
$("plan").addEventListener("input", () => { clearTimeout(t); t = setTimeout(check, 400); });
$("plan").value = EXAMPLES["Out of order"];

(async () => {
  try {
    const [pyodide, src, cat] = await Promise.all([loadPyodide(), fetch("plan_checker.py").then((r) => r.text()), fetch("catalog.json").then((r) => r.text())]);
    pyodide.FS.writeFile("/home/pyodide/plan_checker.py", src);
    pyodide.globals.set("CATALOG_JSON", cat);
    pyodide.runPython(`
import json
from plan_checker import Catalog, check_plan
_d = json.loads(CATALOG_JSON)
CAT = Catalog(set(_d["courses"]), {k: [set(c) for c in v] for k, v in _d["prereqs"].items()})
`);
    py = pyodide;
    $("check").disabled = false;
    $("check").textContent = "Check plan";
    load("Out of order");
  } catch (err) {
    $("report").innerHTML = `<p class="q-small">Could not start the checker: ${esc(err.message)}</p>`;
  }
})();
