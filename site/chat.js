// Chat UI for Leo. The answers come from leo.js; this file only handles the conversation.
(function () {
  const $ = (id) => document.getElementById(id);
  const thread = $("thread"), input = $("input"), send = $("send"), form = $("composer");
  const KEY = "leo-chat-v1";
  const MARK = '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M11 8v16h11" /></svg>';
  const STARTERS = [
    ["Plan my next semesters", "I'm a Computer Science major and just finished CMPSC 132. Plan my next two semesters."],
    ["Check a prerequisite", "What do I need before CMPSC 465?"],
    ["Explore a major", "What does the Data Sciences major require?"],
    ["Ask about a policy", "How does late drop work?"],
  ];
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  let state = Leo.newState(), history = [], busy = false;

  const getJSON = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.json(); });
  const ready = Promise.all([getJSON("data/courses.json"), getJSON("data/majors.json"), getJSON("data/policy.json")])
    .then(([c, m, p]) => Leo.load(c, m, p));
  ready.then(() => {
    const later = window.requestIdleCallback || ((f) => setTimeout(f, 300));
    later(() => getJSON("data/descriptions.json").then((d) => { Leo.setDescriptions(d); fillDescriptions(); }).catch(() => {}));
  });

  function fillDescriptions() {
    const d = Leo.data && Leo.data.desc;
    if (!d) return;
    document.querySelectorAll("[data-desc-for]").forEach((el) => {
      if (el.dataset.filled) return;
      const t = d[el.dataset.descFor];
      el.textContent = t ? t.charAt(0).toUpperCase() + t.slice(1) + "." : "";
      el.dataset.filled = "1";
    });
  }

  function save() {
    try { sessionStorage.setItem(KEY, JSON.stringify({ history, state })); } catch (_) {}
  }
  function restore() {
    try {
      const s = JSON.parse(sessionStorage.getItem(KEY) || "null");
      if (s && Array.isArray(s.history) && s.history.length) { history = s.history; state = Object.assign(Leo.newState(), s.state); }
    } catch (_) {}
  }

  function setEmpty() {
    const empty = history.length === 0;
    document.body.classList.toggle("empty", empty);
    $("new-chat").hidden = empty;
  }

  function userNode(text) {
    const el = document.createElement("div");
    el.className = "msg user";
    el.innerHTML = `<div class="bubble">${esc(text)}</div>`;
    return el;
  }
  function botNode(html, chips) {
    const el = document.createElement("div");
    el.className = "msg bot";
    el.innerHTML = `<div class="avatar">${MARK}</div><div class="body">${html}${chips && chips.length
      ? `<div class="follow">${chips.map((c) => `<button type="button" data-ask="${esc(c)}">${esc(c)}</button>`).join("")}</div>` : ""}</div>`;
    return el;
  }
  function renderAll() {
    thread.innerHTML = "";
    for (const m of history) thread.appendChild(m.role === "user" ? userNode(m.text) : botNode(m.html, m.chips));
    setEmpty();
    renderContext();
    fillDescriptions();
  }

  function renderContext() {
    const box = $("context");
    const parts = [];
    if (state.major) parts.push(`<b>${esc(state.major.replace(/,\s*B\.([AS])\.\s*\(Abington\)$/, " (B.$1.)"))}</b>`);
    if (state.completed.length) parts.push(`${state.completed.length} finished ${state.completed.length === 1 ? "course" : "courses"}${state.implied.length ? ` + ${state.implied.length} assumed` : ""}`);
    if (!parts.length || !history.length) { box.hidden = true; return; }
    box.innerHTML = `<span>Leo remembers: ${parts.join(" · ")}</span><button type="button" id="forget">Forget</button>`;
    box.hidden = false;
  }

  function scrollDown() {
    requestAnimationFrame(() => window.scrollTo({ top: document.documentElement.scrollHeight, behavior: "smooth" }));
  }

  async function ask(text) {
    text = text.trim();
    if (!text || busy) return;
    busy = true;
    input.value = ""; grow(); send.disabled = true;
    history.push({ role: "user", text });
    if (history.length === 1) setEmpty();
    thread.appendChild(userNode(text));
    const wait = botNode('<span class="dots" aria-label="Leo is thinking"><i></i><i></i><i></i></span>');
    thread.appendChild(wait);
    scrollDown();
    const t0 = performance.now();
    let reply;
    try {
      await ready;
      reply = Leo.respond(text, state);
      state = reply.state;
    } catch (err) {
      console.error(err);
      reply = { html: "<p>Something went wrong on my side while answering that. Try asking it another way.</p>", chips: Leo.STARTERS.slice(0, 2) };
    }
    const pause = Math.max(0, 380 - (performance.now() - t0));
    await new Promise((r) => setTimeout(r, pause));
    const node = botNode(reply.html, reply.chips);
    wait.replaceWith(node);
    history.push({ role: "bot", html: reply.html, chips: reply.chips });
    save();
    renderContext();
    fillDescriptions();
    busy = false;
    send.disabled = !input.value.trim();
    // keep the start of the answer in view for long replies
    requestAnimationFrame(() => {
      const top = node.getBoundingClientRect().top + window.scrollY - 72;
      const bottom = document.documentElement.scrollHeight - window.innerHeight;
      window.scrollTo({ top: Math.min(top, bottom), behavior: "smooth" });
    });
    if (matchMedia("(hover: hover)").matches) input.focus();
  }

  function reset() {
    history = []; state = Leo.newState();
    try { sessionStorage.removeItem(KEY); } catch (_) {}
    renderAll();
    input.focus();
  }

  function grow() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
  }

  $("starters").innerHTML = STARTERS.map(([t, q]) => `<button type="button" data-ask="${esc(q)}"><b>${esc(t)}</b><span>${esc(q)}</span></button>`).join("");
  input.addEventListener("input", () => { grow(); send.disabled = busy || !input.value.trim(); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); ask(input.value); }
  });
  form.addEventListener("submit", (e) => { e.preventDefault(); ask(input.value); });
  document.addEventListener("click", (e) => {
    const b = e.target.closest("[data-ask]");
    if (b) { e.preventDefault(); ask(b.dataset.ask); return; }
    if (e.target.closest("#forget")) { state = Leo.newState(); save(); renderContext(); }
  });
  $("new-chat").addEventListener("click", reset);

  restore();
  renderAll();
  const q = new URLSearchParams(location.search).get("q");
  if (q) { if (!history.length) ask(q); window.history.replaceState(null, "", location.pathname); }
  if (history.length) scrollDown();
  if (matchMedia("(hover: hover)").matches) input.focus();
})();
