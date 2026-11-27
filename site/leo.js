// Leo, browser edition: answers from the Penn State catalog, the Abington majors table and the
// advising FAQ with deterministic tools (course lookup, prerequisite chains, a prerequisite-aware
// semester planner, the plan referee from evals/plan_checker.py, and BM25 search over the FAQ).
// No model and no server: every answer is computed from data/*.json in the browser.
// Data: python -m evals.export_site_chat
(function (root) {
  "use strict";

  const GENED = { GA: "Arts", GH: "Humanities", GN: "Natural Sciences", GQ: "Quantification", GS: "Social and Behavioral Sciences",
    GWS: "Writing and Speaking", GHW: "Health and Wellness", US: "US Cultures", IL: "International Cultures" };
  const DEPT_ALIAS = { CS: "CMPSC", COMPSCI: "CMPSC", CMPSCI: "CMPSC", ACCT: "ACCTG", STATS: "STAT" };
  const STOP = new Set(("a an and are as at be but by can could do does for from get have how i if in into is it its me my " +
    "of on or our should so that the their them then there these they this to was we what when where which who will with would " +
    "you your am im i'm ive i've work works happen happens does mean course courses class classes about any some find show list tell give want need know please " +
    "penn state psu abington").split(" "));

  let D = null;

  // ---------- helpers ----------
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const base = (code) => code.replace(/(\d)[A-Z]$/, "$1");
  const num = (code) => parseInt(code.split(" ")[1], 10) || 0;
  const dept = (code) => code.split(" ")[0];
  const uniq = (arr) => [...new Set(arr)];
  const plural = (n, w, p) => `${n} ${n === 1 ? w : p || w + "s"}`;
  const listText = (xs) => xs.length <= 1 ? xs.join("") : xs.slice(0, -1).join(", ") + " and " + xs[xs.length - 1];

  function creditsOf(c) {
    if (!c || !c.cr) return 3;
    const parts = c.cr.split("-").map(Number);
    if (parts.length > 1) return Math.min(3, parts[1]);
    return parts[0] || 3;
  }

  function stem(w) {
    if (w.length > 5 && w.endsWith("ing")) return w.slice(0, -3);
    if (w.length > 4 && w.endsWith("ies")) return w.slice(0, -3) + "y";
    if (w.length > 4 && w.endsWith("ed")) return w.slice(0, -2);
    if (w.length > 3 && w.endsWith("s") && !w.endsWith("ss")) return w.slice(0, -1);
    return w;
  }
  const tokens = (s) => (s.toLowerCase().match(/[a-z0-9]+/g) || []).filter((w) => !STOP.has(w) && w.length > 1).map(stem);

  class BM25 {
    constructor(docs) {
      this.docs = docs.map((d) => { const tf = new Map(); d.forEach((w) => tf.set(w, (tf.get(w) || 0) + 1)); return { tf, len: d.length }; });
      this.df = new Map();
      this.docs.forEach((d) => d.tf.forEach((_, w) => this.df.set(w, (this.df.get(w) || 0) + 1)));
      this.avg = this.docs.reduce((s, d) => s + d.len, 0) / Math.max(1, this.docs.length);
    }
    idf(w) { const n = this.docs.length, df = this.df.get(w) || 0; return Math.log(1 + (n - df + 0.5) / (df + 0.5)); }
    search(q, k = 10) {
      const terms = uniq(q);
      const out = [];
      this.docs.forEach((d, i) => {
        let s = 0;
        for (const w of terms) {
          const tf = d.tf.get(w);
          if (tf) s += this.idf(w) * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * d.len / this.avg));
        }
        if (s > 0) out.push([i, s]);
      });
      return out.sort((a, b) => b[1] - a[1]).slice(0, k);
    }
  }

  // ---------- data ----------
  function load(courses, majors, policy) {
    const byCode = new Map(), byBase = new Map(), depts = new Set(), unlocks = new Map();
    for (const [code, title, cr, tags, pre, clauses] of courses) {
      const c = { code, title, cr, tags, pre, clauses };
      byCode.set(code, c);
      depts.add(dept(code));
      const b = base(code);
      if (!byBase.has(b)) byBase.set(b, []);
      byBase.get(b).push(code);
    }
    for (const c of byCode.values())
      for (const clause of c.clauses) for (const p of clause) {
        const b = base(p);
        if (!unlocks.has(b)) unlocks.set(b, new Set());
        unlocks.get(b).add(c.code);
      }
    D = { byCode, byBase, depts, unlocks, desc: null, majors: [], policy, policyIdx: new BM25(policy.map((p) => tokens(p.t))), topicIdx: null };
    D.majors = majors.map(parseMajor);
    D.majorCourses = new Set(D.majors.flatMap((m) => m.items.flatMap((it) => it.alts.flat())));
    return D;
  }

  function setDescriptions(desc) { D.desc = desc; D.topicIdx = null; }

  function find(code) {
    if (D.byCode.has(code)) return D.byCode.get(code);
    const alts = D.byBase.get(base(code));
    return alts ? D.byCode.get(alts[0]) : null;
  }
  const known = (code) => D.byCode.has(code) || D.byBase.has(base(code));

  // Course codes in free text: "CMPSC 131, 132 and 360", "cmpsc132", "STAT/MATH 318", "CS 465".
  function extractCodes(text, { strict = false } = {}) {
    const re = /\b([A-Za-z]{2,8}(?:\/[A-Za-z]{2,8})?)\s?(\d{1,3}[A-Za-z]?)\b((?:\s*(?:,|&|\/|\band\b|\bor\b)\s*\d{1,3}[A-Za-z]?\b)*)/g;
    const out = [];
    let m;
    while ((m = re.exec(text))) {
      const raw = m[1];
      let dp = raw.toUpperCase().split("/").find((d) => D.depts.has(d) || DEPT_ALIAS[d]);
      if (!dp) continue;
      if (!D.depts.has(dp)) dp = DEPT_ALIAS[dp];
      const nums = [m[2], ...(m[3].match(/\d{1,3}[A-Za-z]?/g) || [])];
      for (const n of nums) {
        if (n.length < 2) continue;
        const code = `${dp} ${n.toUpperCase()}`;
        const upperTyped = raw === raw.toUpperCase();
        if (known(code) || (upperTyped && !strict && DEPT_ALIAS[raw.toUpperCase()] === undefined)) out.push(code);
      }
    }
    return uniq(out);
  }

  function makeTaken(codes) {
    const set = new Set(codes), bases = new Set([...set].map(base));
    return { set, has: (p) => set.has(p) || bases.has(base(p)) };
  }
  const clauseMet = (clause, taken) => clause.some((p) => taken.has(p));

  // ---------- majors ----------
  const MAJOR_ALIASES = [
    ["Computer Science", ["computer science", "comp ?sci", "\\bcmpsc major"], ["\\bcs\\b"]],
    ["Cybersecurity Analytics and Operations", ["cyber ?security", "\\bcyber\\b", "\\bcyaop\\b"]],
    ["Data Sciences", ["data sciences?", "\\bds major"]],
    ["Information Technology", ["information technology", "info tech"], ["\\bIT\\b"]],
    ["Integrative Arts", ["integrative arts?"]],
    ["Integrative Science", ["integrative sciences?"]],
    ["American Studies", ["american studies"]],
    ["Health Humanities", ["health humanities"]],
    ["Race and Ethnic Studies", ["race and ethnic", "ethnic studies"]],
    ["Corporate Communication", ["corporate comm", "\\bcommunications? major"]],
    ["Criminal Justice", ["criminal justice", "\\bcrim(inology)?\\b"], ["\\bcj\\b"]],
    ["Project and Supply Chain Management", ["supply chain", "project management", "\\bpscm\\b"]],
    ["Recreation, Park, and Tourism Management", ["recreation", "tourism", "\\brptm\\b"]],
    ["Rehabilitation and Human Services", ["rehabilitation", "human services", "\\brehab\\b"]],
    ["Psychological and Social Sciences", ["psycholog", "\\bpsych\\b", "social sciences"]],
    ["Multidisciplinary Studies", ["multidisciplinary"]],
    ["Accounting", ["accounting"]],
    ["Finance", ["finance major", "\\bfinance\\b"]],
    ["Biology", ["biology"], ["\\bbio\\b"]],
    ["Business", ["business major", "\\bbusiness\\b"]],
    ["English", ["english major"], ["\\benglish\\b"]],
    ["History", ["history major"], ["\\bhistory\\b"]],
    ["Art", ["\\bart major", "studio art"], ["\\bart\\b"]],
  ];
  const MAJOR_CONTEXT = /\b(major|majoring|degree|plan|semester|require|requirement|studying|study|program|graduat)/i;

  function splitTop(s, sep) {
    const parts = []; let depth = 0, buf = "";
    for (let i = 0; i < s.length; i++) {
      const ch = s[i];
      if (ch === "(") depth++;
      if (ch === ")") depth--;
      if (depth === 0 && s.slice(i, i + sep.length).toLowerCase() === sep) { parts.push(buf); buf = ""; i += sep.length - 1; continue; }
      buf += ch;
    }
    parts.push(buf);
    return parts.map((p) => p.trim()).filter(Boolean);
  }
  function stripParens(s) {
    s = s.trim();
    while (s.startsWith("(") && s.endsWith(")")) {
      let depth = 0, wraps = true;
      for (let i = 0; i < s.length - 1; i++) { if (s[i] === "(") depth++; if (s[i] === ")") depth--; if (depth === 0) { wraps = false; break; } }
      if (!wraps) break;
      s = s.slice(1, -1).trim();
    }
    return s;
  }
  function parseReqs(str) {
    const items = [], notes = [];
    // "One of: A, B, C, One from: D, E" -> "(A or B or C), (D or E)"
    str = (str || "").replace(/One (?:of|from):\s*(.*?)(?=,\s*One (?:of|from):|$)/gi, (_, inner) => `(${inner.split(/,\s*/).join(" or ")})`);
    for (const piece of splitTop(str, ",")) {
      const p = stripParens(piece);
      if (/credits? from|\blike\b|etc\b|choose|select|elective/i.test(p)) { notes.push(p.replace(/\.+$/, "")); continue; }
      const alts = splitTop(p, " or ").map((a) => extractCodes(stripParens(a).replace(/&/g, ","), { strict: true })).filter((a) => a.length);
      if (alts.length) items.push({ text: p, alts });
      else if (/[a-z]{4}/i.test(p)) notes.push(p);
    }
    return { items, notes };
  }

  function parseMajor(r) {
    const name = r["Major Name"];
    const short = name.replace(/,\s*B\.[AS]\.\s*\(Abington\)$/, "");
    const degree = /B\.A\./.test(name) ? "BA" : "BS";
    const pres = parseReqs(r["Prescribed Courses"]), add = parseReqs(r["Additional Courses"]);
    const extras = Object.entries(r).filter(([k, v]) => /Option Summary|Supporting|Special Requirements/.test(k) && v).map(([k, v]) => [k.replace(/ Summary$/, ""), v]);
    const optionCourses = Object.entries(r).filter(([k]) => /Option Courses$/.test(k)).map(([k, v]) => [k.replace(/ Courses$/, ""), v]);
    const genEd = (r["General Education Requirements"] || "").split(/,\s*(?![^()]*\))/).map((s) => s.trim()).filter(Boolean);
    return { name, short, degree, label: `${short} (${degree === "BA" ? "B.A." : "B.S."})`, code: r["Major Code"], total: r["Total Credits Required"], majorCr: r["Credits for Major"], genCr: r["Credits for General Education"],
      prescribed: pres.items, additional: add.items, items: [...pres.items, ...add.items], notes: [...pres.notes, ...add.notes], options: r["Options"] || "", optionCourses, extras, genEd };
  }

  function detectMajor(text) {
    const low = text.toLowerCase();
    const contextual = MAJOR_CONTEXT.test(text) || text.trim().split(/\s+/).length <= 4;
    for (const [short, pats, ambiguous = []] of MAJOR_ALIASES) {
      const hit = pats.some((p) => new RegExp(p, "i").test(low)) ||
        (contextual && ambiguous.some((p) => (p === "\\bIT\\b" ? new RegExp(p).test(text) : new RegExp(p, "i").test(low))));
      if (!hit) continue;
      const opts = D.majors.filter((m) => m.short === short);
      if (opts.length <= 1) return opts[0] || null;
      const wantBA = /\bB\.?A\.?\b|\bbachelor of arts\b/i.test(text);
      return opts.find((m) => m.degree === (wantBA ? "BA" : "BS")) || opts[0];
    }
    return null;
  }

  // ---------- planning ----------
  function pickOption(clause, ctx) {
    const known = clause.filter((p) => find(p));
    if (!known.length) return null;
    return known.find((p) => ctx.need && ctx.need.has(p)) || known.find((p) => ctx.majorSet && ctx.majorSet.has(p)) ||
      [...known].sort((a, b) => {
        const sa = ctx.near && dept(a) === dept(ctx.near) ? 0 : 1, sb = ctx.near && dept(b) === dept(ctx.near) ? 0 : 1;
        if (sa !== sb) return sa - sb;
        const na = ctx.near ? Math.abs(num(ctx.near) - num(a)) : num(a), nb = ctx.near ? Math.abs(num(ctx.near) - num(b)) : num(b);
        return na - nb;
      })[0];
  }

  // Finishing CMPSC 132 implies its prerequisite chain (CMPSC 131) is done too.
  function impliedBy(codes) {
    const taken = makeTaken(codes), implied = [];
    const visit = (code, depth) => {
      const c = find(code);
      if (!c || depth > 4) return;
      for (const clause of c.clauses) {
        if (clauseMet(clause, taken)) continue;
        const p = pickOption(clause, { near: code });
        if (!p) continue;
        implied.push(p); taken.set.add(p);
        visit(p, depth + 1);
      }
    };
    codes.forEach((c) => visit(c, 0));
    return uniq(implied).filter((c) => !codes.includes(c));
  }

  function termNames(n, now = new Date()) {
    const m = now.getMonth();
    let season = m >= 8 ? "Spring" : "Fall", year = m >= 8 ? now.getFullYear() + 1 : now.getFullYear();
    const out = [];
    for (let i = 0; i < n; i++) {
      out.push(`${season} ${year}`);
      if (season === "Spring") season = "Fall"; else { season = "Spring"; year++; }
    }
    return out;
  }

  function buildPlan(major, completed, nSem, opts = {}) {
    const maxCr = opts.maxCredits || 16;
    const taken = makeTaken(completed);
    const majorSet = new Set(major.items.flatMap((it) => it.alts.flat()));
    const need = new Set();
    const chosen = [];
    for (const it of major.items) {
      if (it.alts.some((alt) => alt.every((c) => taken.has(c)))) continue;
      const alt = [...it.alts].sort((a, b) => a.filter((c) => !taken.has(c)).length - b.filter((c) => !taken.has(c)).length)[0];
      alt.filter((c) => !taken.has(c) && find(c)).forEach((c) => need.add(find(c).code));
      if (it.alts.length > 1) chosen.push({ text: it.text, alt });
    }
    // add missing prerequisites of needed courses
    const added = [];
    const queue = [...need];
    while (queue.length) {
      const code = queue.shift(), c = find(code);
      for (const clause of (c ? c.clauses : [])) {
        if (clause.some((p) => taken.has(p) || need.has(p) || [...need].some((n) => base(n) === base(p)))) continue;
        const p = pickOption(clause, { need, majorSet, near: code });
        if (!p) continue;
        const pc = find(p).code;
        need.add(pc); added.push(pc); queue.push(pc);
      }
    }
    // longest chain of needed courses that depend on each course
    const memo = new Map();
    const depth = (code, seen = new Set()) => {
      if (memo.has(code)) return memo.get(code);
      if (seen.has(code)) return 0;
      seen.add(code);
      let d = 0;
      for (const other of need) {
        const oc = find(other);
        if (oc && oc.clauses.some((cl) => cl.some((p) => base(p) === base(code)))) d = Math.max(d, 1 + depth(other, seen));
      }
      memo.set(code, d);
      return d;
    };
    const genEdCycle = ["GWS", "GH", "GS", "GN", "GA", "GHW", "GQ"];
    let gi = 0;
    const names = termNames(Math.max(nSem, 8), opts.now);
    const semesters = [];
    const done = makeTaken(completed);
    const remaining = new Set(need);
    const limit = nSem || 8;
    for (let s = 0; s < limit; s++) {
      if (!remaining.size && nSem === 0) break;
      const avail = [...remaining].filter((code) => find(code).clauses.every((cl) => clauseMet(cl, done)))
        .sort((a, b) => depth(b) - depth(a) || num(a) - num(b) || a.localeCompare(b));
      const sem = { term: names[s], courses: [], credits: 0 };
      for (const code of avail) {
        const cr = creditsOf(find(code));
        if (sem.credits + cr > maxCr || sem.courses.filter((c) => !c.placeholder).length >= 5) continue;
        sem.courses.push({ code, title: find(code).title, cr });
        sem.credits += cr;
      }
      while (sem.credits < 15) {
        const g = genEdCycle[gi++ % genEdCycle.length];
        sem.courses.push({ code: null, placeholder: true, title: `Gen ed: ${GENED[g]} (${g})`, cr: 3 });
        sem.credits += 3;
      }
      sem.courses.filter((c) => c.code).forEach((c) => { remaining.delete(c.code); done.set.add(c.code); });
      semesters.push(sem);
    }
    const text = semesters.map((s, i) => `Semester ${i + 1}: ${s.courses.filter((c) => c.code).map((c) => c.code).join(", ")}\nTotal Credits: ${s.credits}`).join("\n");
    const report = checkPlan(text, { prior: completed });
    return { semesters, remaining: [...remaining], added, chosen, text, report };
  }

  // ---------- referee (JS port of evals/plan_checker.py) ----------
  const SEMESTER_RE = /^\W*semester\s+\+?(\d+)\b/i;
  const CREDITS_RE = /total\s+credits?\W*\s*([\d.]+)/i;
  const PLAN_CODE_RE = /\b([A-Z]{2,8}(?:\/[A-Z]{2,8})?)\s?(\d{1,3}[A-Z]?)\b/g;
  function normalize(d, n) { return `${d} ${n}`; }
  function parsePlan(text) {
    const sems = []; let cur = null;
    for (let line of text.split(/\r?\n/)) {
      line = line.replace(/\*/g, "");
      const m = line.match(SEMESTER_RE);
      if (m) { cur = { number: +m[1], courses: [], stated: null }; sems.push(cur); line = line.slice(m[0].length); }
      if (!cur) continue;
      const cm = line.match(CREDITS_RE);
      if (cm) { cur.stated = parseFloat(cm[1]); line = line.slice(0, cm.index); }
      for (const c of line.toUpperCase().matchAll(PLAN_CODE_RE)) {
        if (c[2].length === 3 || /^\d+$/.test(c[2].slice(0, -1)) && c[2].length > 1) cur.courses.push(normalize(c[1], c[2]));
      }
    }
    return sems;
  }
  function checkPlan(text, { prior = [], minCr = 12, maxCr = 19 } = {}) {
    const sems = parsePlan(text);
    const all = sems.flatMap((s) => s.courses);
    const r = { semesters: sems.length, courses: all.length, unknown_courses: [], prereq_violations: [], duplicate_courses: [], credit_issues: [] };
    const seen = new Set();
    for (const code of all) {
      if (!known(code) && !r.unknown_courses.includes(code)) r.unknown_courses.push(code);
      if (seen.has(code) && !r.duplicate_courses.includes(code)) r.duplicate_courses.push(code);
      seen.add(code);
    }
    let taken = makeTaken(prior);
    for (const s of [...sems].sort((a, b) => a.number - b.number)) {
      for (const code of s.courses) {
        const c = D.byCode.get(code);
        for (const clause of (c ? c.clauses : [])) {
          if (!clauseMet(clause, taken)) { r.prereq_violations.push(`${code} in semester ${s.number} needs one of ${[...clause].sort().join(", ")} earlier`); break; }
        }
      }
      taken = makeTaken([...taken.set, ...s.courses]);
      if (s.stated !== null && !(s.stated >= minCr && s.stated <= maxCr)) r.credit_issues.push(`semester ${s.number}: ${s.stated} credits`);
    }
    r.passed = r.semesters > 0 && !(r.unknown_courses.length || r.prereq_violations.length || r.duplicate_courses.length || r.credit_issues.length);
    return r;
  }

  // ---------- rendering ----------
  const chip = (code) => `<button class="lc" data-ask="Tell me about ${esc(code)}">${esc(code)}</button>`;
  const courseRow = (code, extra = "") => {
    const c = find(code);
    return `<li class="row"><button class="row-btn" data-ask="Tell me about ${esc(c ? c.code : code)}"><span class="code">${esc(c ? c.code : code)}</span><span class="title">${esc(c ? c.title : "Not in the catalog")}</span><span class="cr">${c && c.cr ? esc(c.cr) + " cr" : ""}</span></button>${extra}</li>`;
  };
  const rows = (codes, max = 12) => `<ul class="rows">${codes.slice(0, max).map((c) => courseRow(c)).join("")}</ul>${codes.length > max ? `<p class="muted">and ${codes.length - max} more.</p>` : ""}`;
  const p = (s) => `<p>${s}</p>`;
  const bulletin = (code) => `https://bulletins.psu.edu/university-course-descriptions/undergraduate/${dept(code).toLowerCase().replace(/\//g, "")}/`;

  function courseCard(c) {
    const tags = c.tags.filter((t) => GENED[t]).map((t) => `<span class="tag">${esc(GENED[t])} (${t})</span>`).join("");
    const desc = D.desc ? D.desc[c.code] : null;
    const opens = D.unlocks.get(base(c.code));
    return `<article class="card">
      <header><span class="code big">${esc(c.code)}</span>${c.cr ? `<span class="pill">${esc(c.cr)} credits</span>` : ""}</header>
      <h3>${esc(c.title)}</h3>
      ${tags ? `<div class="tags">${tags}</div>` : ""}
      <p class="desc" data-desc-for="${esc(c.code)}">${desc ? esc(cap(desc)) + "." : D.desc ? "" : '<span class="muted">Loading description…</span>'}</p>
      <dl>
        <dt>Prerequisites</dt><dd>${c.pre ? linkCodes(esc(c.pre)) : "None listed"}</dd>
        ${opens ? `<dt>Required for</dt><dd>${plural(opens.size, "course")}${opens.size ? ", including " + [...opens].sort((a, b) => num(a) - num(b)).slice(0, 4).map(chip).join(" ") : ""}</dd>` : ""}
      </dl>
      <a class="ext" href="${bulletin(c.code)}" target="_blank" rel="noopener">Penn State bulletin</a>
    </article>`;
  }
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
  function linkCodes(html) {
    return html.replace(/\b([A-Z]{2,8}) (\d{1,3}[A-Z]?)\b/g, (m) => (known(m) ? chip(m) : m));
  }

  function planBlock(major, plan) {
    const cols = plan.semesters.map((s) => `<section class="sem"><h4>${esc(s.term)}<span>${s.credits} cr</span></h4><ul>${s.courses.map((c) => c.code
      ? `<li><button class="row-btn" data-ask="Tell me about ${esc(c.code)}"><span class="code">${esc(c.code)}</span><span class="title">${esc(c.title)}</span><span class="cr">${c.cr}</span></button></li>`
      : `<li class="ph"><span class="title">${esc(c.title)}</span><span class="cr">${c.cr}</span></li>`).join("")}</ul></section>`).join("");
    return `<div class="plan">${cols}</div>${refereeBlock(plan.report)}`;
  }
  function refereeBlock(r) {
    const n = (a) => a.length;
    const detail = [...r.unknown_courses.map((c) => `${c} is not in the catalog`), ...r.prereq_violations, ...r.duplicate_courses.map((c) => `${c} appears twice`), ...r.credit_issues.map((c) => `${c} (outside 12–19)`)];
    return `<div class="referee ${r.passed ? "ok" : "bad"}"><b>${r.passed ? "Referee: plan passes" : "Referee: plan fails"}</b>
      <span>${n(r.unknown_courses)} unknown · ${n(r.prereq_violations)} prerequisite ${n(r.prereq_violations) === 1 ? "violation" : "violations"} · ${n(r.duplicate_courses)} duplicate · ${n(r.credit_issues)} credit ${n(r.credit_issues) === 1 ? "issue" : "issues"}</span>
      ${detail.length ? `<ul>${detail.map((d) => `<li>${linkCodes(esc(d))}</li>`).join("")}</ul>` : ""}</div>`;
  }

  // ---------- intents ----------
  const WORDNUM = { one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, a: 1 };
  function semestersAsked(low) {
    if (/\b(graduat|full plan|whole|entire|all (my )?(remaining )?semesters|rest of|four[- ]year|4[- ]year|every semester|until i (finish|graduate))/.test(low)) return 0;
    const m = low.match(/\b(\d|one|two|three|four|five|six|seven|eight)\s+(more\s+)?(semesters?|terms?)\b/);
    if (m) return Math.min(8, WORDNUM[m[1]] || +m[1]);
    if (/\bnext (semester|term)\b|this (spring|fall)|upcoming semester|\bnext spring\b|\bnext fall\b/.test(low)) return 1;
    if (/\b(next year|this year|year)\b/.test(low)) return 2;
    return 2;
  }

  const DONE_VERB = /\b(took|taken|finished|completed|done with|passed|did|have done|already (have|did)|got credit)\b/i;
  const NEED_WORDS = /\b(before|need|needs|require|required|prereq|prerequisite|can i|could i|should i)\b/i;
  function completedIn(text) {
    const parts = text.split(/[.;!?\n]|\bbut\b|\bwhat\b|\bcan i\b|\bplan\b|\bshould\b/i);
    return uniq(parts.filter((s) => DONE_VERB.test(s) && !NEED_WORDS.test(s)).flatMap((s) => extractCodes(s)).filter(known).map((c) => find(c).code));
  }

  function newState() { return { major: null, completed: [], implied: [], pending: null }; }

  function addCompleted(state, codes) {
    const fresh = codes.filter((c) => !state.completed.includes(c));
    state.completed.push(...fresh);
    state.implied = impliedBy(state.completed).filter((c) => !state.completed.includes(c));
    return fresh;
  }
  const allDone = (state) => uniq([...state.completed, ...state.implied]);

  function respond(text, state) {
    state = state || newState();
    const raw = text.trim();
    const low = raw.toLowerCase();
    const out = { html: "", chips: [], state };
    if (!raw) return out;

    if (/^(clear|reset|start over|new chat|forget (it|everything|that))\b/.test(low)) {
      Object.assign(state, newState());
      out.html = p("Cleared. I've forgotten your major and the courses you told me about.");
      out.chips = STARTERS.slice(0, 3);
      return out;
    }

    const major = detectMajor(raw);
    const majorChanged = major && (!state.major || state.major !== major.name);
    if (major) state.major = major.name;
    const done = completedIn(raw);
    const fresh = done.length ? addCompleted(state, done) : [];
    const codes = extractCodes(raw).filter((c) => !done.includes(c) || /\b(about|is|what|tell)\b/i.test(low) && !/\b(plan|next|after)\b/.test(low));
    const M = state.major ? D.majors.find((m) => m.name === state.major) : null;

    // a pasted plan goes to the referee
    if (raw.split(/\n/).filter((l) => SEMESTER_RE.test(l.replace(/\*/g, ""))).length >= 1 && extractCodes(raw).length >= 2) return refereeAnswer(raw, out);

    if (/^(hi|hello|hey|yo|good (morning|afternoon|evening)|thanks|thank you|ty)\b[\s!.]*$/i.test(raw)) {
      out.html = /thank/.test(low) ? p("Anytime. What else can I help you plan?") : intro();
      out.chips = STARTERS.slice(0, 4);
      return out;
    }
    if (/\b(what can you do|who are you|what are you|help me$|^help|how do(es)? (this|you) work|what do you know)\b/.test(low)) {
      out.html = intro(true);
      out.chips = STARTERS;
      return out;
    }

    // finish an earlier question that was waiting on a major
    if (state.pending && major) {
      const pend = state.pending; state.pending = null;
      if (pend.intent === "plan") return planAnswer(M, state, pend.n, out, fresh);
      if (pend.intent === "eligible") return eligibleAnswer(state, M, out);
    }

    if (/\b(which|what|list( of)?|all( the)?)\b.*\bmajors\b|\bmajors (are )?(offered|available)\b/.test(low) && !major) return majorsList(out);

    const wantsPlan = /\b(plan|schedule|roadmap|map out|lay out|sequence|semester by semester)\b/.test(low) && !/\bplan b\b/.test(low);
    const wantsNext = /\b(what|which)\b.*\b(should|can|could)\s+i\s+take\b|\btake next\b|\bnext (semester|term)\b.*\btake\b|\beligible\b|\bready (to|for)\b.*\btake\b|\bwhat now\b/.test(low);
    const canTake = /\b(can|could|may|am i allowed to|am i able to|am i ready to)\s+i?\s*take\b|\bam i (eligible|ready) for\b/.test(low);
    const askPrereq = /\bpre-?req|\bprerequisite|\bneed(ed)? (to take |to have )?before\b|\brequired (for|before)\b|\bwhat do i need (for|to take)\b|\brequirements? (for|of) [a-z]{2,8} ?\d/.test(low);
    const askUnlocks = /\b(unlock|lead(s)? to|open(s)? up|what (comes|can i take|courses? need|needs)|courses? that (need|require)|require(s)? [a-z]{2,8} ?\d)\b|\bafter (taking |finishing )?[a-z]{2,8} ?\d/.test(low);
    const askMajor = M && (majorChanged || major) && /\b(require|requirement|need to graduate|degree|curriculum|courses? (do i need|for|in)|major|credits|prescribed|what does)\b/.test(low);
    const askGenEd = /\bgen(eral)? ?ed(ucation)?\b|\b(GA|GH|GN|GQ|GS|GWS|GHW)\b|\b(humanities|social (and behavioral )?sciences?|natural sciences?|quantification|health and wellness|us cultures|international cultures)\b/i.test(raw);

    if (wantsPlan || (wantsNext && M && !codes.length)) {
      const n = wantsPlan ? semestersAsked(low) : 1;
      if (!M) { state.pending = { intent: "plan", n }; return askForMajor(out, "plan"); }
      return planAnswer(M, state, n, out, fresh);
    }
    if (canTake && codes.length) return canTakeAnswer(codes[0], state, out);
    if (askPrereq && codes.length) return prereqAnswer(codes[0], state, out);
    if (askUnlocks && (codes.length || done.length)) return unlocksAnswer(codes[0] || done[done.length - 1], state, out);
    if (wantsNext) {
      if (!state.completed.length && !M) {
        state.pending = { intent: "plan", n: 1 };
        out.html = p("Happy to. Which major are you in? If you've already finished some courses, tell me those too, for example: <i>Computer Science, and I've taken CMPSC 131 and MATH 140.</i>");
        out.chips = ["Computer Science", "Data Sciences", "Information Technology", "Business", "Psychological and Social Sciences", "Biology"];
        return out;
      }
      if (!state.completed.length) { out.html = p("Tell me which courses you've finished and I'll work out what you can take next. For example: <i>I've taken CMPSC 131, MATH 140 and ENGL 15.</i>"); out.chips = ["I've taken CMPSC 131 and MATH 140", "I'm a Computer Science major"]; return out; }
      return eligibleAnswer(state, M, out);
    }
    if (askMajor || (major && !codes.length && raw.split(/\s+/).length <= 6)) return majorAnswer(major || M, state, out);
    if (askGenEd && !codes.length) return genEdAnswer(raw, out);
    if (codes.length) return courseAnswer(codes, out, state);
    if (fresh.length) return ackCompleted(state, fresh, out, majorChanged);

    return searchAnswer(raw, out, state);
  }

  const STARTERS = [
    "I'm a Computer Science major and just finished CMPSC 132. Plan my next two semesters.",
    "What do I need before CMPSC 465?",
    "What does the Data Sciences major require?",
    "How does late drop work?",
    "Find courses about machine learning",
    "Which majors are offered?",
  ];

  function intro(full) {
    return p("Hi, I'm Leo, an academic advisor for Penn State Abington. Ask me about any of the 9,248 courses in the catalog, the 26 Abington majors, prerequisites, semester plans or advising policies.") +
      (full ? `<ul class="bullets">
        <li><b>Plan semesters.</b> Tell me your major and what you've finished, and I'll build a plan that respects prerequisites. A separate referee then checks it.</li>
        <li><b>Check prerequisites.</b> What a course needs, whether you can take it yet, and what it unlocks.</li>
        <li><b>Explore majors and gen eds.</b> Required courses, choices and credit totals, plus gen-ed courses by area.</li>
        <li><b>Answer policy questions.</b> Drop and add, late drop, declaring majors and minors, transfer credit, grade forgiveness and more, quoted from the advising FAQ.</li>
        <li><b>Check a plan.</b> Paste lines like <i>Semester 1: CMPSC 131, MATH 140</i> and I'll check it.</li>
      </ul>` + p('<span class="muted">Everything runs in your browser from the catalog data. I don\'t see your transcript, so I only know the courses you tell me about.</span>') : "");
  }

  function askForMajor(out, why) {
    out.html = p(why === "plan" ? "Happy to. Which major should I plan for?" : "Which major are you in?");
    out.chips = ["Computer Science", "Data Sciences", "Information Technology", "Business", "Psychological and Social Sciences", "Biology"];
    return out;
  }

  function contextLine(state) {
    const parts = [];
    if (state.completed.length) parts.push(`I counted ${listText(state.completed.map((c) => `<b>${esc(c)}</b>`))} as done`);
    if (state.implied.length) parts.push(`plus ${listText(state.implied.map(esc))} from ${state.implied.length === 1 ? "its" : "their"} prerequisites`);
    return parts.length ? parts.join(", ") + "." : "You haven't told me any finished courses, so I started from the beginning of the major.";
  }

  function planAnswer(M, state, n, out, fresh) {
    const plan = buildPlan(M, allDone(state), n);
    const sems = plan.semesters.length;
    const when = n === 0 ? "every remaining required course" : sems === 1 ? "your next semester" : `your next ${sems} semesters`;
    let html = p(`Here's a plan for ${when} in <b>${esc(M.label)}</b>. ${contextLine(state)}`);
    html += planBlock(M, plan);
    const notes = [];
    if (plan.added.length) notes.push(`I added ${listText(plan.added.map(chip))} because later courses in the major need ${plan.added.length === 1 ? "it" : "them"}.`);
    if (plan.chosen.length) notes.push(`Where the major offers a choice, I picked ${listText(plan.chosen.map((c) => c.alt.join(" + ")))}.`);
    if (plan.remaining.length && n !== 0) notes.push(`After this, ${plural(plan.remaining.length, "required course")} remain${plan.remaining.length === 1 ? "s" : ""}, including ${plan.remaining.slice(0, 5).map(chip).join(" ")}.`);
    if (plan.remaining.length && n === 0) notes.push(`I couldn't place ${listText(plan.remaining.map(chip))} in 8 semesters because of ${plan.remaining.length === 1 ? "its" : "their"} prerequisites.`);
    if (!plan.remaining.length && n !== 0 && sems) notes.push("That covers every required course I could find for the major.");
    if (n === 0 && !plan.remaining.length) {
      const doneCr = allDone(state).reduce((s, c) => s + creditsOf(find(c)), 0);
      const planned = plan.semesters.reduce((s, x) => s + x.credits, 0);
      const left = Math.max(0, (+M.total || 120) - doneCr - planned);
      notes.push(`That schedules every required course for the major in ${plural(sems, "semester")}${left ? `. You'd still need about ${left} more credits of gen eds and electives to reach ${esc(M.total)}` : ""}.`);
    }
    notes.push('<span class="muted">Gen-ed slots are placeholders, since I don\'t know which ones you\'ve already met. Class-standing rules aren\'t checked. Confirm with your adviser before you register.</span>');
    html += notes.map(p).join("");
    out.html = html;
    out.chips = [n === 0 ? "Just the next semester" : "Plan all the way to graduation", `What does the ${M.short} major require?`, "What can I take next?"];
    return out;
  }

  function eligibleAnswer(state, M, out) {
    const done = makeTaken(allDone(state));
    let pool, scope;
    if (M) {
      const plan = buildPlan(M, allDone(state), 0);
      pool = [...plan.remaining, ...plan.semesters.flatMap((s) => s.courses.filter((c) => c.code).map((c) => c.code))];
      pool = uniq(pool).filter((c) => find(c).clauses.every((cl) => clauseMet(cl, done)));
      scope = `required for ${esc(M.short)}`;
    } else {
      const unlocked = new Set();
      allDone(state).forEach((c) => (D.unlocks.get(base(c)) || []).forEach((u) => unlocked.add(u)));
      pool = [...unlocked].filter((c) => !done.has(c) && find(c).clauses.every((cl) => clauseMet(cl, done)));
      scope = "unlocked by what you've finished";
    }
    pool.sort((a, b) => num(a) - num(b) || a.localeCompare(b));
    out.html = p(`${contextLine(state)} You can take ${plural(pool.length, "course")} ${scope} right now:`) + rows(pool, 14) +
      (M ? "" : p('<span class="muted">Tell me your major and I\'ll narrow this down to the courses it requires.</span>'));
    out.chips = M ? ["Plan my next two semesters", `What does the ${M.short} major require?`] : ["I'm a Computer Science major", "Plan my next two semesters"];
    return out;
  }

  function canTakeAnswer(code, state, out) {
    const c = find(code);
    if (!c) return notFound(code, out);
    const done = makeTaken(allDone(state));
    if (!c.clauses.length) { out.html = p(`Yes. <b>${esc(c.code)}</b> (${esc(c.title)}) has no course prerequisites${c.pre ? `, though the catalog says: <i>${linkCodes(esc(c.pre))}</i>` : ""}.`); out.chips = [`Tell me about ${c.code}`]; return out; }
    if (!allDone(state).length) {
      out.html = p(`<b>${esc(c.code)}</b> needs ${linkCodes(esc(c.pre))}. Tell me which courses you've finished and I'll check whether you're ready.`);
      out.chips = [`I've taken ${c.clauses.map((cl) => cl[0]).join(" and ")}`, `What do I need before ${c.code}?`];
      return out;
    }
    const missing = c.clauses.filter((cl) => !clauseMet(cl, done));
    out.html = missing.length
      ? p(`Not yet. <b>${esc(c.code)}</b> still needs ${listText(missing.map((cl) => cl.map(chip).join(" or ")))}.`) + p(`<span class="muted">${contextLine(state)}</span>`)
      : p(`Yes. You've met every course prerequisite for <b>${esc(c.code)}</b> (${esc(c.title)}).`) + p(`<span class="muted">${contextLine(state)}</span>`);
    out.chips = [`What do I need before ${c.code}?`, `What does ${c.code} unlock?`];
    return out;
  }

  function chainTo(code) {
    const order = [], seen = new Set();
    const visit = (cd, depth) => {
      const c = find(cd);
      if (!c || seen.has(c.code) || depth > 8) return;
      seen.add(c.code);
      for (const cl of c.clauses) { const p = pickOption(cl, { near: c.code }); if (p) visit(p, depth + 1); }
      order.push(c.code);
    };
    visit(code, 0);
    return order;
  }

  function prereqAnswer(code, state, out) {
    const c = find(code);
    if (!c) return notFound(code, out);
    if (!c.clauses.length) {
      out.html = p(`<b>${esc(c.code)}</b> (${esc(c.title)}) has no course prerequisites.${c.pre ? ` The catalog notes: <i>${linkCodes(esc(c.pre))}</i>.` : ""}`);
      out.chips = [`Tell me about ${c.code}`, `What does ${c.code} unlock?`];
      return out;
    }
    const groups = c.clauses.map((cl) => `<li>${cl.length > 1 ? "One of " : ""}${cl.map(chip).join(cl.length > 2 ? ", " : " or ")}</li>`).join("");
    const chain = chainTo(c.code);
    let html = p(`<b>${esc(c.code)}</b> (${esc(c.title)}) requires:`) + `<ul class="bullets">${groups}</ul>`;
    if (c.pre && /standing|permission|concurrent|placement/i.test(c.pre)) html += p(`<span class="muted">The catalog also says: ${linkCodes(esc(c.pre))}.</span>`);
    if (chain.length > 2) html += p("The shortest path from scratch:") + `<div class="chain">${chain.map(chip).join('<span class="arrow" aria-hidden="true">→</span>')}</div>`;
    if (allDone(state).length) {
      const done = makeTaken(allDone(state));
      const miss = c.clauses.filter((cl) => !clauseMet(cl, done));
      html += p(miss.length ? `From what you've told me, you still need ${listText(miss.map((cl) => cl.map(chip).join(" or ")))}.` : "From what you've told me, you've met all of these.");
    }
    out.html = html;
    out.chips = [`Can I take ${c.code}?`, `What does ${c.code} unlock?`, `Tell me about ${c.code}`];
    return out;
  }

  function unlocksAnswer(code, state, out) {
    const c = find(code);
    if (!c) return notFound(code, out);
    const list = [...(D.unlocks.get(base(c.code)) || [])].sort((a, b) => num(a) - num(b) || a.localeCompare(b));
    if (!list.length) { out.html = p(`No course in the catalog lists <b>${esc(c.code)}</b> as a prerequisite.`); out.chips = [`Tell me about ${c.code}`]; return out; }
    const done = makeTaken([...allDone(state), c.code]);
    const ready = list.filter((u) => find(u).clauses.every((cl) => clauseMet(cl, done)));
    out.html = p(`${plural(list.length, "course")} list${list.length === 1 ? "s" : ""} <b>${esc(c.code)}</b> as a prerequisite.` +
      (ready.length && ready.length < list.length ? ` With ${esc(c.code)}${allDone(state).length ? " and what you've told me" : ""}, you'd meet every prerequisite for ${ready.length} of them:` : "")) +
      rows(ready.length && ready.length < list.length ? ready : list, 12) +
      (ready.length && ready.length < list.length ? p(`<span class="muted">The other ${list.length - ready.length} need more than ${esc(c.code)}.</span>`) : "");
    out.chips = [`What do I need before ${list[list.length - 1]}?`, "Plan my next two semesters"];
    return out;
  }

  function majorAnswer(M, state, out) {
    if (!M) return askForMajor(out);
    const shown = new Set();
    const items = M.items.filter((it) => { const k = JSON.stringify(it.alts); if (shown.has(k)) return false; shown.add(k); return true; }).map((it) => it.alts.length > 1
      ? `<li class="choice">${it.alts.map((a) => a.map(chip).join(" + ")).join(' <span class="or">or</span> ')}</li>`
      : it.alts[0].map((c) => `<li>${chip(c)}</li>`).join("")).join("");
    let html = p(`<b>${esc(M.label)}</b> takes ${esc(M.total)} credits: ${esc(M.majorCr || "?")} for the major and ${esc(M.genCr || "?")} in general education.`);
    if (M.items.length) html += `<h4 class="sub">Required courses</h4><ul class="chips">${items}</ul>`;
    if (M.notes.length) html += `<h4 class="sub">Also</h4><ul class="bullets">${M.notes.map((n) => `<li>${linkCodes(esc(n))}</li>`).join("")}</ul>`;
    if (M.options) html += `<h4 class="sub">Options</h4>${p(linkCodes(esc(M.options)))}`;
    for (const [k, v] of M.optionCourses) html += `<details><summary>${esc(k)} courses</summary>${p(linkCodes(esc(v)))}</details>`;
    for (const [k, v] of M.extras) html += `<details><summary>${esc(k)}</summary>${p(linkCodes(esc(v)))}</details>`;
    if (M.genEd.length) html += `<details><summary>General education</summary><ul class="bullets">${M.genEd.map((g) => `<li>${esc(g)}</li>`).join("")}</ul></details>`;
    out.html = html;
    out.chips = [`Plan my first two semesters as ${/^[AEIOU]/.test(M.short) ? "an" : "a"} ${M.short} major`, "Plan all the way to graduation", "Which majors are offered?"];
    return out;
  }

  function majorsList(out) {
    out.html = p(`Penn State Abington offers ${D.majors.length} majors in my data:`) +
      `<ul class="rows">${D.majors.map((m) => `<li class="row"><button class="row-btn" data-ask="What does the ${esc(m.short)}${D.majors.filter((x) => x.short === m.short).length > 1 ? (m.degree === "BA" ? " B.A." : " B.S.") : ""} major require?"><span class="title">${esc(m.name.replace(" (Abington)", ""))}</span><span class="cr">${esc(m.total)} cr</span></button></li>`).join("")}</ul>`;
    out.chips = ["What does the Computer Science major require?", "What does the Business major require?"];
    return out;
  }

  function genEdAnswer(raw, out) {
    const low = raw.toLowerCase();
    const byWord = [["GWS", /\bgws\b|writing|speaking/], ["GHW", /\bghw\b|health and wellness|wellness/], ["GQ", /\bgq\b|quantification|math gen/],
      ["GA", /\bga\b|\barts?\b/], ["GH", /\bgh\b|humanities/], ["GN", /\bgn\b|natural sciences?/], ["GS", /\bgs\b|social/], ["US", /\bus cultures?\b/], ["IL", /international cultures?|\bil\b/]];
    const hit = byWord.find(([, re]) => re.test(low));
    if (!hit) {
      out.html = p("Penn State gen eds come in these areas. Which one do you want courses for?") + `<ul class="bullets">${Object.entries(GENED).map(([k, v]) => `<li><b>${k}</b> ${esc(v)}</li>`).join("")}</ul>`;
      out.chips = ["GH courses with no prerequisites", "GN courses with no prerequisites", "GS courses with no prerequisites"];
      return out;
    }
    const tag = hit[0];
    const topic = tokens(raw).filter((w) => !/^(gen|ed|general|education|gh|gn|gs|ga|gq|gws|ghw|humaniti|social|scienc|natural|art|prerequisit|no|without|easy|intro|introductory|behavioral|quantification|wellness|health|writ|speak|il|us|culture|international|requirement|satisfy|count|good|best|recommend)$/.test(w));
    let list = [...D.byCode.values()].filter((c) => c.tags.includes(tag));
    const noPre = /\bno prereq|without prereq|easy|intro|no requirements/.test(low);
    if (noPre) list = list.filter((c) => !c.clauses.length && !c.pre);
    list = list.filter((c) => num(c.code) < 400 && !/special topics|independent studies|foreign studies|research projects/i.test(c.title));
    if (topic.length) {
      const scored = list.map((c) => [c, topic.filter((w) => tokens(c.title + " " + (D.desc && D.desc[c.code] || "")).includes(w)).length]).filter(([, s]) => s > 0);
      if (scored.length) list = scored.sort((a, b) => b[1] - a[1]).map(([c]) => c);
    }
    list.sort((a, b) => (topic.length ? 0 : num(a.code) - num(b.code)));
    out.html = p(`${plural(list.length, "course")} in the catalog count${list.length === 1 ? "s" : ""} toward <b>${esc(GENED[tag])} (${tag})</b>${noPre ? " with no prerequisites" : ""}. Some to start with:`) +
      rows(list.map((c) => c.code), 12);
    out.chips = [`${tag} courses with no prerequisites`, "What other gen-ed areas are there?"];
    return out;
  }

  function courseAnswer(codes, out, state) {
    const found = codes.map(find).filter(Boolean);
    if (!found.length) return notFound(codes[0], out);
    out.html = found.slice(0, 3).map(courseCard).join("");
    const c = found[0];
    out.chips = [`What do I need before ${c.code}?`, `What does ${c.code} unlock?`, `Can I take ${c.code}?`];
    return out;
  }

  function notFound(code, out) {
    const dp = dept(code);
    const same = [...D.byCode.keys()].filter((k) => dept(k) === dp).sort((a, b) => Math.abs(num(a) - num(code)) - Math.abs(num(b) - num(code))).slice(0, 4);
    out.html = p(`I can't find <b>${esc(code)}</b> in the Penn State catalog.${same.length ? ` The closest ${esc(dp)} courses are ${listText(same.map(chip))}.` : ""}`);
    out.chips = same.slice(0, 2).map((c) => `Tell me about ${c}`);
    return out;
  }

  function ackCompleted(state, fresh, out, majorChanged) {
    const M = state.major && D.majors.find((m) => m.name === state.major);
    out.html = p(`Got it${majorChanged && M ? `: <b>${esc(M.label)}</b>` : ""}. ${contextLine(state)}`) + (state.major ? "" : p("Which major are you in? I'll use it to plan."));
    out.chips = state.major ? ["What can I take next?", "Plan my next two semesters"] : ["What can I take next?", "I'm a Computer Science major"];
    return out;
  }

  const SYN = { drop: ["drop", "dropp"], withdraw: ["withdraw", "withdrawal"], advisor: ["adviser"], adviser: ["adviser"], gpa: ["gpa", "grade", "point", "average", "cgpa"],
    retake: ["repeat", "forgiveness"], repeat: ["repeat", "forgiveness"], double: ["concurrent", "double"], switch: ["change"], change: ["change"],
    leave: ["leave", "absence"], credit: ["credit"], transfer: ["transfer"], minor: ["minor"], audit: ["audit"], late: ["late"], add: ["add"] };

  function topicIndex() {
    if (!D.topicIdx) {
      D.topicList = [...D.byCode.values()];
      D.topicIdx = new BM25(D.topicList.map((c) => { const t = tokens(c.title); return [...t, ...t, ...t, ...tokens((D.desc && D.desc[c.code]) || "")]; }));
    }
    return D.topicIdx;
  }

  // the best window of up to three consecutive sentences
  function bestSentences(text, q, n = 3) {
    const sents = text.split(/(?<=[.!?])\s+(?=[A-Z0-9"(])/);
    const hits = sents.map((s) => uniq(tokens(s)).filter((w) => q.includes(w)).length);
    let best = 0, at = 0;
    for (let i = 0; i < sents.length; i++) {
      const sc = hits.slice(i, i + n).reduce((a, b) => a + b, 0) + (hits[i] ? 0.5 : 0);
      if (sc > best) { best = sc; at = i; }
    }
    return sents.slice(at, at + n).join(" ").trim();
  }

  function searchAnswer(raw, out, state) {
    const q = tokens(raw);
    const qx = uniq(q.flatMap((w) => [w, ...(SYN[w] || [])]).map(stem));
    const pol = D.policyIdx.search(qx, 3);
    const topic = /\b(courses?|classes?)\b|\b(teach|learn|study|studying)\b/i.test(raw);
    const polStrong = pol.length && pol[0][1] >= 2.2 * Math.min(3, q.length) / 2;
    if (polStrong && !(topic && !/\b(how|when|can i|what happens|policy|deadline|rule)\b/i.test(raw))) {
      const picks = [pol[0]];
      if (pol[1] && pol[1][1] >= 0.7 * pol[0][1] && D.policy[pol[1][0]].p !== D.policy[pol[0][0]].p) picks.push(pol[1]);
      out.html = p("From the Penn State Abington advising and registrar FAQ:") + picks.map(([i]) => {
        const best = D.policy[i], quote = bestSentences(best.t, qx);
        return `<blockquote>${linkCodes(esc(quote))}</blockquote><p class="source">Advising FAQ, page ${best.p}</p>` +
          (quote.length < best.t.length - 40 ? `<details><summary>Full passage</summary>${p(esc(best.t))}</details>` : "");
      }).join("");
      out.chips = ["How does late drop work?", "How do I declare a minor?", "How does grade forgiveness work?"].filter((c) => c.toLowerCase() !== raw.toLowerCase()).slice(0, 2);
      return out;
    }
    if (q.length) {
      const hits = topicIndex().search(q, 10).filter(([, s]) => s > 1.5);
      if (hits.length) {
        const codes = hits.map(([i]) => D.topicList[i].code);
        out.html = p(`Courses that match <b>${esc(q.join(" "))}</b>${D.desc ? "" : " (titles only while descriptions load)"}:`) + rows(codes, 10);
        out.chips = [`Tell me about ${codes[0]}`, `What do I need before ${codes[0]}?`];
        return out;
      }
    }
    if (pol.length && pol[0][1] > 1) {
      const best = D.policy[pol[0][0]];
      out.html = p("I'm not certain this answers it, but here's the closest passage in the advising FAQ:") + `<blockquote>${esc(bestSentences(best.t, qx))}</blockquote><p class="source">Advising FAQ, page ${best.p}</p>`;
      out.chips = STARTERS.slice(1, 3);
      return out;
    }
    out.html = p("I couldn't match that to a course, a major or an advising policy. I'm good at questions like these:");
    out.chips = STARTERS.slice(0, 4);
    return out;
  }

  function refereeAnswer(raw, out) {
    const r = checkPlan(raw);
    out.html = p(`I checked ${plural(r.semesters, "semester")} and ${plural(r.courses, "course")} against the catalog.`) + refereeBlock(r);
    out.chips = ["Plan my next two semesters", "What do I need before CMPSC 465?"];
    return out;
  }

  const api = { load, setDescriptions, respond, newState, checkPlan, buildPlan, extractCodes, detectMajor, impliedBy, termNames, STARTERS, courseCard, find,
    get data() { return D; } };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.Leo = api;
})(typeof window !== "undefined" ? window : globalThis);
