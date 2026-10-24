"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const log = $("log");
  const hero = $("hero");
  const form = $("composer");
  const input = $("input");
  const sendBtn = $("send");
  const stopBtn = $("stop");
  const toasts = $("toasts");
  const announce = $("announce");
  const keyDialog = $("key-dialog");
  const keyInput = $("key-input");

  let threadId = null;
  let controller = null;
  let busy = false;
  let counter = 0;
  let apiKey = readKey();

  // --- small helpers ---------------------------------------------------------------------------

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function readKey() {
    try {
      return sessionStorage.getItem("apiKey") || "";
    } catch {
      return "";
    }
  }

  function saveKey(value) {
    apiKey = value;
    try {
      sessionStorage.setItem("apiKey", value);
    } catch {
      /* storage unavailable: the key lives in memory for this page only */
    }
  }

  function headers(extra) {
    const h = Object.assign({}, extra);
    if (apiKey) h["X-API-Key"] = apiKey;
    return h;
  }

  function nearBottom() {
    return log.scrollHeight - log.scrollTop - log.clientHeight < 80;
  }

  function stick(wasNear) {
    if (wasNear) log.scrollTop = log.scrollHeight;
  }

  function toast(text) {
    const node = el("div", "toast", text);
    toasts.append(node);
    while (toasts.children.length > 4) toasts.firstChild.remove();
    setTimeout(() => node.remove(), 6000);
  }

  async function httpError(res) {
    let detail = "";
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d) => d.msg).join("; ");
      else if (body.error) detail = body.error;
    } catch {
      /* not JSON */
    }
    const err = new Error(detail || "Request failed (" + res.status + ")");
    err.status = res.status;
    return err;
  }

  function friendly(err) {
    if (err.status === 401) return "This server needs an API key.";
    if (err.status === 409) return "Another question is still running in this chat. Wait for it or press Stop.";
    if (err.status === 503) return "The model or database is not reachable. " + err.message;
    if (err.name === "TypeError") return "Cannot reach the server. Is it still running?";
    return err.message;
  }

  // --- server status ---------------------------------------------------------------------------

  async function refreshStatus() {
    const dot = $("status-dot");
    const text = $("status-text");
    try {
      const res = await fetch("/ready");
      const b = await res.json();
      if (res.ok) {
        dot.className = "dot ok";
        text.textContent = "Ready · " + b.model + " · " + (b.documents ?? 0) + " chunks";
      } else {
        const why = [];
        if (!b.ollama) why.push("Ollama unreachable");
        else if (!b.model_available) why.push("model " + b.model + " not pulled");
        if (!b.qdrant) why.push("Qdrant unreachable");
        dot.className = "dot bad";
        text.textContent = "Not ready: " + (why.join(", ") || "unknown");
      }
    } catch {
      dot.className = "dot bad";
      text.textContent = "Server unreachable";
    }
  }

  // --- rendering -------------------------------------------------------------------------------

  const INLINE = /(`[^`\n]+`|\*\*[^*\n]+\*\*|\[\d+\])/g;

  function inline(parent, text, ctx) {
    let last = 0;
    for (const m of text.matchAll(INLINE)) {
      if (m.index > last) parent.append(text.slice(last, m.index));
      const tok = m[0];
      if (tok[0] === "`") {
        parent.append(el("code", null, tok.slice(1, -1)));
      } else if (tok.startsWith("**")) {
        parent.append(el("strong", null, tok.slice(2, -2)));
      } else {
        const n = Number(tok.slice(1, -1));
        if (ctx.known.has(n)) {
          const a = el("a", "cite", String(n));
          a.href = "#" + ctx.prefix + n;
          a.dataset.target = ctx.prefix + n;
          a.setAttribute("aria-label", "Source " + n);
          parent.append(a);
        } else {
          parent.append(tok);
        }
      }
      last = m.index + tok.length;
    }
    if (last < text.length) parent.append(text.slice(last));
  }

  function blocks(container, chunk, ctx) {
    let para = [];
    let list = null;
    const flushPara = () => {
      if (!para.length) return;
      const p = el("p");
      para.forEach((line, i) => {
        if (i) p.append(el("br"));
        inline(p, line, ctx);
      });
      container.append(p);
      para = [];
    };
    for (const line of chunk.split("\n")) {
      if (!line.trim()) {
        flushPara();
        list = null;
        continue;
      }
      const ul = /^\s*[-*]\s+(.*)$/.exec(line);
      const ol = /^\s*\d+[.)]\s+(.*)$/.exec(line);
      if (ul || ol) {
        flushPara();
        const tag = ul ? "ul" : "ol";
        if (!list || list.tag !== tag) {
          list = { tag, node: el(tag) };
          container.append(list.node);
        }
        const li = el("li");
        inline(li, (ul || ol)[1], ctx);
        list.node.append(li);
      } else {
        list = null;
        para.push(line);
      }
    }
    flushPara();
  }

  // Builds DOM nodes only (no HTML strings), so model or document text cannot inject markup.
  function renderMarkdown(container, text, ctx) {
    container.replaceChildren();
    const fence = /```[^\n]*\n([\s\S]*?)```/g;
    let last = 0;
    let m;
    while ((m = fence.exec(text)) !== null) {
      blocks(container, text.slice(last, m.index), ctx);
      const pre = el("pre");
      pre.append(el("code", null, m[1].replace(/\n$/, "")));
      container.append(pre);
      last = fence.lastIndex;
    }
    blocks(container, text.slice(last), ctx);
  }

  function renderSources(body, citations, prefix) {
    if (!citations.length) return;
    const used = citations.filter((c) => c.cited).length;
    const details = el("details", "sources");
    const noun = citations.length === 1 ? "source" : "sources";
    details.append(el("summary", null, citations.length + " " + noun + (used ? " · " + used + " cited" : "")));
    for (const c of citations) {
      const box = el("div", "src" + (c.cited ? "" : " unused"));
      box.id = prefix + c.n;
      const head = el("div", "src-head", "[" + c.n + "] " + c.source);
      if (c.location) head.append(" ", el("span", "src-loc", "· " + c.location));
      box.append(head, el("p", "src-snippet", c.snippet));
      details.append(box);
    }
    body.append(details);
  }

  const WORKER = {
    rag: { doing: "Looking in your documents", did: "Searched your documents", chip: "documents" },
    web: { doing: "Looking on the web", did: "Searched the web", chip: "web" },
    python: { doing: "Writing and running code", did: "Ran code in the sandbox", chip: "code" },
  };

  function renderMeta(body, response) {
    const meta = el("div", "meta");
    const used = [...new Set(response.route.filter((n) => WORKER[n]))];
    used.forEach((n) => meta.append(el("span", "chip", WORKER[n].chip)));
    if (!used.length) meta.append(el("span", "chip", "answered directly"));
    const m = response.metrics;
    meta.append(el("span", "chip", m.total_latency_s.toFixed(1) + " s"));
    if (m.tokens_per_second) meta.append(el("span", "chip", m.tokens_per_second.toFixed(1) + " tok/s"));
    body.append(meta);
  }

  // --- messages --------------------------------------------------------------------------------

  function addUser(text) {
    const row = el("div", "msg user");
    row.append(el("div", "bubble", text));
    log.append(row);
    stick(true);
  }

  function addAssistant() {
    counter += 1;
    const row = el("div", "msg assistant");
    const avatar = el("div", "avatar");
    avatar.setAttribute("aria-hidden", "true");
    avatar.append(document.querySelector(".mark svg").cloneNode(true));
    const body = el("div", "body");
    const progress = el("div", "progress");
    const answer = el("div", "answer");
    answer.hidden = true;
    body.append(progress, answer);
    row.append(avatar, body);
    log.append(row);
    return { row, body, progress, answer, prefix: "src-" + counter + "-", live: null, text: null };
  }

  function setLive(msg, label) {
    if (!msg.live) {
      msg.live = el("div", "live");
      msg.progress.append(msg.live);
    }
    msg.live.textContent = label;
    msg.progress.append(msg.live); // keep the live line last
  }

  function stepLine(msg, label) {
    const line = el("div", null, label);
    msg.progress.insertBefore(line, msg.live);
  }

  function onStep(msg, ev) {
    const d = ev.detail || {};
    if (ev.node === "supervisor") {
      if (d.next && d.next !== "answer" && WORKER[d.next]) setLive(msg, WORKER[d.next].doing);
      else setLive(msg, "Thinking");
      return;
    }
    if (WORKER[ev.node]) {
      const summary = d.summary ? " — " + String(d.summary).slice(0, 140) : "";
      stepLine(msg, WORKER[ev.node].did + summary);
      setLive(msg, "Thinking");
    }
  }

  function showError(msg, text) {
    msg.answer.hidden = false;
    msg.answer.classList.add("error");
    msg.answer.classList.remove("streaming");
    const partial = msg.text ? msg.text.data : "";
    msg.answer.replaceChildren(
      ...(partial ? [document.createTextNode(partial + "\n\n")] : []),
      el("strong", null, "Something went wrong. "),
      document.createTextNode(text)
    );
    msg.live?.remove();
  }

  function onFinal(msg, response) {
    const wasNear = nearBottom();
    msg.live?.remove();
    msg.answer.hidden = false;
    msg.answer.classList.remove("streaming");
    const ctx = { prefix: msg.prefix, known: new Set(response.citations.map((c) => c.n)) };
    renderMarkdown(msg.answer, response.answer || "(no answer)", ctx);
    renderSources(msg.body, response.citations, msg.prefix);
    renderMeta(msg.body, response);
    announce.textContent = "Answer received.";
    stick(wasNear);
  }

  // --- streaming -------------------------------------------------------------------------------

  function parseBlock(block, cb) {
    let name = "message";
    const data = [];
    for (const line of block.split("\n")) {
      if (line.startsWith(":")) continue; // keep-alive comment
      if (line.startsWith("event:")) name = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
    }
    if (!data.length) return;
    let parsed;
    try {
      parsed = JSON.parse(data.join("\n"));
    } catch {
      return;
    }
    cb(name, parsed);
  }

  async function readEvents(stream, cb) {
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, i);
        buf = buf.slice(i + 2);
        parseBlock(block, cb);
      }
    }
    if (buf.trim()) parseBlock(buf, cb);
  }

  function setBusy(value) {
    busy = value;
    sendBtn.hidden = value;
    stopBtn.hidden = !value;
  }

  async function ask(question) {
    question = question.trim();
    if (busy || !question) return;
    hero.hidden = true;
    setBusy(true);
    addUser(question);
    const msg = addAssistant();
    setLive(msg, "Thinking");
    controller = new AbortController();
    let finished = false;
    try {
      const res = await fetch("/v1/ask/stream", {
        method: "POST",
        headers: headers({ "Content-Type": "application/json" }),
        body: JSON.stringify(threadId ? { question, thread_id: threadId } : { question }),
        signal: controller.signal,
      });
      if (!res.ok) throw await httpError(res);
      await readEvents(res.body, (name, data) => {
        if (name === "start") {
          threadId = data.thread_id;
        } else if (name === "step") {
          onStep(msg, data);
        } else if (name === "token") {
          const wasNear = nearBottom();
          if (!msg.text) {
            msg.text = document.createTextNode("");
            msg.answer.append(msg.text);
            msg.answer.hidden = false;
            msg.answer.classList.add("streaming");
            setLive(msg, "Writing");
          }
          msg.text.data += data.text;
          stick(wasNear);
        } else if (name === "final") {
          finished = true;
          onFinal(msg, data.response);
        } else if (name === "error") {
          finished = true;
          showError(msg, data.message || data.error_type);
        }
      });
      if (!finished) showError(msg, "The connection closed before the answer finished.");
    } catch (err) {
      if (err.name === "AbortError") {
        msg.live?.remove();
        if (msg.text) {
          msg.answer.classList.remove("streaming");
          const meta = el("div", "meta");
          meta.append(el("span", "chip", "Stopped"));
          msg.body.append(meta);
        } else {
          msg.row.remove();
        }
      } else {
        showError(msg, friendly(err));
        if (err.status === 401) openKeyDialog();
      }
    } finally {
      controller = null;
      setBusy(false);
      refreshStatus();
      input.focus();
    }
  }

  // --- controls --------------------------------------------------------------------------------

  function openKeyDialog() {
    if (typeof keyDialog.showModal === "function" && !keyDialog.open) keyDialog.showModal();
  }

  keyDialog.addEventListener("close", () => {
    if (keyDialog.returnValue === "save" && keyInput.value.trim()) {
      saveKey(keyInput.value.trim());
      toast("Key saved for this tab. Send your question again.");
    }
    keyInput.value = "";
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = input.value;
    if (!text.trim()) return;
    input.value = "";
    autosize();
    ask(text);
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  function autosize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
  }
  input.addEventListener("input", autosize);

  stopBtn.addEventListener("click", () => controller?.abort());

  $("cards").addEventListener("click", (e) => {
    const card = e.target.closest(".card");
    if (card) ask(card.dataset.q);
  });

  log.addEventListener("click", (e) => {
    const link = e.target.closest("a.cite");
    if (!link) return;
    e.preventDefault();
    const target = document.getElementById(link.dataset.target);
    if (!target) return;
    const details = target.closest("details");
    if (details) details.open = true;
    log.querySelectorAll(".src.target").forEach((n) => n.classList.remove("target"));
    target.classList.add("target");
    target.scrollIntoView({ block: "nearest" });
  });

  $("new-chat").addEventListener("click", () => {
    controller?.abort();
    if (threadId) {
      fetch("/v1/sessions/" + encodeURIComponent(threadId), { method: "DELETE", headers: headers() }).catch(() => {});
    }
    threadId = null;
    log.querySelectorAll(".msg").forEach((n) => n.remove());
    hero.hidden = false;
    input.focus();
  });

  const fileInput = $("file-input");
  $("attach").addEventListener("click", () => fileInput.click());
  const FILE_STATUS = { ingested: "added", skipped_unchanged: "already up to date", failed: "failed" };
  fileInput.addEventListener("change", async () => {
    const files = [...fileInput.files];
    fileInput.value = "";
    if (!files.length) return;
    const data = new FormData();
    files.forEach((f) => data.append("files", f, f.name));
    toast("Uploading " + files.length + (files.length === 1 ? " file" : " files") + "…");
    try {
      const res = await fetch("/v1/documents", { method: "POST", headers: headers(), body: data });
      if (!res.ok) throw await httpError(res);
      const body = await res.json();
      for (const f of body.files) {
        const chunks = f.chunks ? " (" + f.chunks + " chunks)" : "";
        toast(f.source + ": " + (FILE_STATUS[f.status] || f.status) + chunks + (f.error ? " — " + f.error : ""));
      }
      refreshStatus();
    } catch (err) {
      toast("Upload failed: " + friendly(err));
      if (err.status === 401) openKeyDialog();
    }
  });

  refreshStatus();
  setInterval(refreshStatus, 30000);
  input.focus();
})();
