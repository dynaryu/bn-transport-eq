/* Ask renders the engine's events; model calls remain in the server's chat layer. */
const conversation = [];
let latestClarification = null;

function tableCard(headers, rows, caption) {
  return `<div class="table-wrap"><table class="data"><caption>${esc(caption)}</caption>` +
    `<thead><tr>${headers.map(h => `<th scope="col">${esc(h)}</th>`).join("")}</tr></thead><tbody>` +
    rows.map(row => `<tr>${row.map((value, i) => i === 0
      ? `<th scope="row">${esc(value)}</th>` : `<td>${esc(value)}</td>`).join("")}</tr>`).join("") +
    `</tbody></table></div>`;
}

function calculationDetails(value, label = "Full calculation") {
  return `<details><summary>${esc(label)}</summary><pre>${esc(JSON.stringify(value, null, 2))}</pre></details>`;
}

function intervalText(interval) {
  return interval?.length === 2 ? `[${interval.map(fmtP).join(", ")}]` : "—";
}

function queryResults(result) {
  const rows = (result.events || []).map(e => [
    e.op === "joint" ? e.node : `${e.node} ${e.op} ${e.value}`, fmtP(e.p), intervalText(e.ci95),
  ]);
  for (const target of result.targets || []) {
    if (target.summary.P) {
      for (const [value, p] of Object.entries(target.summary.P)) rows.push([`${target.node} = ${value}`, fmtP(p), "—"]);
    } else {
      const labels = {P_inf: "probability of non-finite value", mean_finite: "mean of finite values", p5: "5th percentile", p50: "median", p95: "95th percentile"};
      for (const [key, value] of Object.entries(target.summary)) {
        rows.push([`${target.node}: ${labels[key] || key}`, Number(value).toPrecision(4), "—"]);
      }
    }
  }
  const headers = ["Event or quantity", "Estimate", "95% interval"];
  if (!rows.length) return `<p class="meta">No reported events or summaries.</p>`;
  return tableCard(headers, rows.slice(0, 8), "Computed results") + (rows.length > 8
    ? `<details><summary>Show ${rows.length - 8} more results</summary>${tableCard(headers, rows.slice(8), "Additional results")}</details>` : "");
}

function analysisCard(result) {
  const titles = {compare: "Scenario comparison", importance: "Bridge availability comparison", paths: "Candidate paths", voi: "Inspection value of information"};
  const num = value => Number(value).toFixed(2);
  let headers, rows, extra = "";
  if (result.kind === "compare") {
    headers = ["Variant", "Baseline P(fails)", "Variant P(fails)", "Reduction (pp)"];
    rows = result.rows.map(r => [r.name, fmtP(r.baseline_p), fmtP(r.variant_p), num(r.reduction_pp)]);
  } else if (result.kind === "importance") {
    headers = ["Bridge", "P(fails), its edge forced closed", "P(fails), its edge forced open", "Change (pp)"];
    rows = result.rows.map(r => [r.bridge, fmtP(r.forced_closed_p), fmtP(r.forced_open_p), num(100 * r.importance)]);
  } else if (result.kind === "paths") {
    headers = ["Route", "Path", "P(survives)", "95% interval", "Time (h)"];
    rows = result.rows.map(r => [r.name, r.nodes.join(" → "), fmtP(r.p), intervalText(r.ci95), num(r.travel_time_h)]);
    extra = `<p>Selected among evaluated routes: <b>${esc(result.recommendation)}</b>.</p>`;
  } else if (result.kind === "voi") {
    headers = ["Inspection", "Expected value (EVSI)", "Cost", "Net value"];
    rows = result.rows.map(r => [r.node, num(r.evsi), num(r.cost), num(r.net_value)]);
    extra = `<p>Values in <b>${esc(result.loss_units)}</b>. Current best action: ${esc(result.current_action)}; expected loss: ${esc(num(result.current_expected_loss))}.</p>` +
      `<p>${result.recommendation === "do not inspect" ? "No inspection has positive net value." : `Selected inspection: <b>${esc(result.recommendation)}</b>.`}</p>` +
      result.rows.map(r => `<details><summary>Actions after inspecting ${esc(r.node)}</summary>` +
        tableCard(["Reported state", "P(outcome)", "Best action", `Expected loss (${result.loss_units})`],
          r.outcomes.map(o => [o.reported_state, fmtP(o.p), o.action, num(o.expected_loss)]), "Decisions after each possible outcome") + `</details>`).join("");
  } else {
    return `<section class="pq analysis-card"><p class="pq-head">Analysis result</p>${calculationDetails(result)}</section>`;
  }
  return `<section class="pq analysis-card"><p class="pq-head">${esc(titles[result.kind])}<span class="pq-state">computed</span></p>` +
    tableCard(headers, rows, titles[result.kind]) + extra + `<p class="meta">${esc(result.note || "")}</p>` + calculationDetails(result) + `</section>`;
}

function proposalCard(proposal, {activate = false} = {}) {
  const spec = proposal.spec || {}, errors = proposal.validation || [];
  const source = Object.entries(spec.source || {}).map(([key, value]) => [key, typeof value === "string" ? value : JSON.stringify(value)]);
  return `<section class="pq proposal-card" data-proposal-id="${esc(proposal.id)}" tabindex="-1">` +
    `<p class="pq-head">${esc(spec.title || "Proposed model input")}<span class="pq-state">${esc(proposal.status)}</span></p>` +
    `<p>${esc(spec.reason || "")}</p>` + tableCard(["Property", "Value"], [
      ["Type", ({scenario: "Saved scenario", parameter_set: "Parameter sets"})[spec.kind] || "Unspecified"],
      ["Validation", errors.length ? `${errors.length} issue(s) to resolve` : "Structural checks passed; source and assumptions need human review"],
      ...(proposal.reviewer ? [["Reviewer", proposal.reviewer]] : []),
      ...(proposal.file ? [["Active file", proposal.file]] : []),
    ], "Proposal summary") + (source.length ? tableCard(["Source field", "Details"], source, "Source and assumptions") : "") +
    (errors.length ? `<ul class="validation reason" role="alert">${errors.map(e => `<li>${esc(e)}</li>`).join("")}</ul>` : "") +
    calculationDetails(proposal, "Review proposed data and validation") +
    (activate && proposal.status === "draft" && !errors.length
      ? `<p><input aria-label="Reviewer name for ${esc(spec.title || proposal.id)}" placeholder="Reviewer name" data-reviewer="${esc(proposal.id)}"> ` +
        `<button data-activate="${esc(proposal.id)}" data-sha="${esc(proposal.sha256)}">Activate reviewed proposal</button></p>`
      : !activate ? `<button data-open-proposal="${esc(proposal.id)}">Review in Proposals</button>` : "") +
    `<p class="proposal-id">Draft ${esc(proposal.id)}</p><p class="review-error" role="alert"></p></section>`;
}

function clarificationCard(value) {
  return `<section class="pq clarification-card"><p class="pq-head">Choose what to calculate</p><p>${esc(value.question)}</p>` +
    (value.choices?.length ? `<div class="clarification-options">${value.choices.map((choice, i) =>
      `<button class="choice" data-choice="${i}" disabled><b>${esc(choice.label)}</b><span>${esc(choice.definition)}</span></button>`).join("")}</div>`
      : `<p class="meta">Reply below with: ${value.missing.map(esc).join(", ")}.</p>`) + `</section>`;
}

function setAskBusy(busy) {
  state.asking = busy;
  $("#chat-send").disabled = busy;
  for (const el of document.querySelectorAll("#chat-examples button, #example, #scenario-preset, #controls input, #controls select, #run, button[data-activate]")) el.disabled = busy;
  for (const el of document.querySelectorAll(".clarification-options button")) el.disabled = busy || !latestClarification?.contains(el);
}

function bubble(role, html) {
  const el = document.createElement("div");
  el.className = `bubble ${role}`;
  el.innerHTML = html;
  $("#chat-log").append(el);
  return el;
}

function scrollChat() {
  const log = $("#chat-log");
  log.scrollTop = log.scrollHeight;
}

async function consumeAskStream(response, onEvent) {
  if (!response.ok) throw new Error((await response.json()).error || response.statusText);
  if (!response.body) throw new Error("The response did not contain a conversation stream.");
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value ?? new Uint8Array(), {stream: !done});
      const lines = buffer.split("\n");
      buffer = done ? "" : lines.pop();
      for (const line of lines) if (line.trim()) onEvent(JSON.parse(line));
      if (done) break;
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

async function ask(question) {
  question = question.trim();
  if (!question || state.asking || !state.ex) return;
  latestClarification = null;
  setAskBusy(true);
  const turn = {question, assembly: state.ex.assembly.id, model: $("#chat-status").dataset.model || "",
    scenario: structuredClone(currentScenario()), started: new Date().toISOString(), queries: [], analyses: [], proposals: [], answer: null, error: null};
  conversation.push(turn);
  bubble("user", esc(question));
  const reply = bubble("assistant", `<div class="steps"></div><div class="answer"></div><span class="stage" role="status"></span>`);
  const steps = reply.querySelector(".steps"), started = Date.now();
  let current = null, stageText = "Translating the question…", review = null;
  const showStage = () => {
    const el = reply.querySelector(".stage");
    if (el) el.textContent = `${stageText} ${Math.round((Date.now() - started) / 1000)} s`;
  };
  const stage = text => { stageText = text; showStage(); scrollChat(); };
  const ticker = setInterval(showStage, 1000);
  stage(stageText);
  function queryCard(event) {
    const item = {query: event.query || null, statement: event.statement || "", readback: event.readback || "", state: "not run", result: null, reason: null};
    turn.queries.push(item);
    const card = document.createElement("section");
    card.className = "pq running";
    card.innerHTML = `<p class="pq-head">Probability query ${turn.queries.length}<span class="pq-state">preparing…</span></p>` +
      `<p>${esc(item.readback || item.statement)}</p>` + `<details><summary>Probability expression</summary><code>${esc(item.statement)}</code></details>`;
    steps.append(card);
    current = {item, card};
    return current;
  }
  function finishReview() {
    if (review) review.panel.querySelectorAll("button").forEach(b => { b.disabled = true; });
    review = null;
  }
  reply.addEventListener("click", async event => {
    const button = event.target.closest("button[data-review], button[data-choice]");
    if (!button || button.disabled) return;
    if (button.dataset.choice != null) {
      const choice = turn.clarification.choices[Number(button.dataset.choice)];
      return ask(`${choice.label}: ${choice.definition}`);
    }
    if (!review || review.submitting) return;
    const pending = review, approved = button.dataset.approve === "true";
    pending.submitting = true;
    pending.panel.querySelectorAll("button").forEach(b => { b.disabled = true; });
    try {
      await getJSON("/api/review", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({token: pending.token, approved})});
      pending.item.review = approved ? "approved" : "cancelled";
      pending.panel.querySelector(".review-message").textContent = approved ? "Approved. Computing…" : "Cancelled.";
      stage(approved ? "Computing the approved query…" : "Finishing the cancelled query…");
    } catch (err) {
      pending.panel.querySelector(".review-error").textContent = err.message;
      if (review === pending) pending.panel.querySelectorAll("button").forEach(b => { b.disabled = false; });
    } finally { pending.submitting = false; }
  });
  const onEvent = event => {
    if (event.type === "query") {
      finishReview();
      queryCard(event).card.querySelector(".pq-state").textContent = "calculating…";
      stage("Calculating through the BN…");
    } else if (event.type === "review") {
      if (!current) queryCard({statement: event.readback});
      current.card.className = "pq reviewing";
      current.card.querySelector(".pq-state").textContent = "awaiting your review";
      const panel = document.createElement("div");
      panel.innerHTML = `<div class="review-actions"><button data-review="${esc(event.token)}" data-approve="true">Compute this query</button>` +
        `<button data-review="${esc(event.token)}" data-approve="false">Cancel query</button></div><p class="review-message" role="status"></p><p class="review-error" role="alert"></p>`;
      current.card.append(panel);
      current.item.review = "pending";
      review = {panel, item: current.item, token: event.token, submitting: false};
      stage("Waiting for your query review…");
    } else if (event.type === "computed") {
      finishReview();
      const result = event.result;
      if (!current || current.item.state !== "not run") queryCard({query: result.query, statement: event.statement || result.statement});
      current.card.className = "pq";
      current.card.querySelector(".pq-state").textContent = `computed · ${result.samples.toLocaleString()} draws` +
        (result.inference?.method === "rejection" ? ` · ${result.inference.accepted.toLocaleString()} matched evidence` : "") + ` · ${result.elapsed_s} s`;
      current.card.insertAdjacentHTML("beforeend", queryResults(result) + `<p class="from">${esc(result.samples_from || "")}` +
        (result.record ? ` · record ${esc(result.record.id)}` : "") + `</p>` +
        (result.warnings?.length ? `<details><summary>Calculation notes</summary><ul>${result.warnings.map(w => `<li>${esc(w)}</li>`).join("")}</ul></details>` : ""));
      Object.assign(current.item, {state: "computed", result});
      state.result = result;
      renderMap(); renderResults();
      stage("Writing the answer from these results…");
    } else if (event.type === "refused") {
      finishReview();
      if (!current || current.item.state !== "not run") queryCard(event);
      current.card.className = "pq refused";
      current.card.querySelector(".pq-state").textContent = "not computed";
      current.card.insertAdjacentHTML("beforeend", `<p class="reason">${esc(event.error)}</p>`);
      Object.assign(current.item, {state: "not computed", reason: event.error});
      stage("Explaining why the query was not computed…");
    } else if (event.type === "analysis_plan") {
      (turn.plans ??= []).push(event.readback);
      steps.insertAdjacentHTML("beforeend", `<p class="meta">${esc(event.readback)}</p>`);
    } else if (event.type === "analysis") {
      turn.analyses.push(event.result);
      steps.insertAdjacentHTML("beforeend", analysisCard(event.result));
    } else if (event.type === "proposal") {
      turn.proposals.push(event.proposal);
      steps.insertAdjacentHTML("beforeend", proposalCard(event.proposal));
      renderProposals();
      stage("Explaining the proposed input…");
    } else if (event.type === "answer") {
      turn.answer = event.answer; turn.record = event.record;
      reply.querySelector(".stage")?.remove();
      if (event.clarification) {
        turn.clarification = event.clarification;
        steps.insertAdjacentHTML("beforeend", clarificationCard(event.clarification));
        latestClarification = steps.lastElementChild;
      } else reply.querySelector(".answer").textContent = event.answer;
      state.chat.push({role: "user", content: question}, {role: "assistant", content: event.answer});
    } else if (event.type === "error") throw new Error(event.error);
    scrollChat();
  };
  try {
    const response = await fetch("/api/ask", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({assembly: turn.assembly, question, history: state.chat, defaults: turn.scenario, review: $("#review-queries").checked})});
    await consumeAskStream(response, onEvent);
    if (turn.answer == null) throw new Error("The connection closed before an answer arrived.");
  } catch (err) {
    turn.error = err.message;
    reply.querySelector(".stage")?.remove();
    for (const item of turn.queries.filter(q => q.state === "not run")) {
      item.state = "interrupted"; item.reason = err.message;
      if (item.review === "pending") item.review = "expired or interrupted";
    }
    for (const card of steps.querySelectorAll(".pq.running, .pq.reviewing")) {
      card.className = "pq refused"; card.querySelector(".pq-state").textContent = "interrupted";
    }
    reply.insertAdjacentHTML("beforeend", `<p class="reason" role="alert">${esc(err.message)}</p>`);
  } finally {
    clearInterval(ticker);
    finishReview();
    setAskBusy(false);
    $("#chat-download").disabled = false;
    scrollChat();
  }
}

function conversationMarkdown() {
  const lines = ["# quakenet conversation", "", `- Saved: ${new Date().toISOString()}`, ""];
  conversation.forEach((turn, i) => {
    lines.push(`## ${i + 1}. ${turn.question}`, "", `- Example: ${turn.assembly}`, `- Model: ${turn.model}`, `- Started: ${turn.started}`, "",
      "Scenario settings:", "```json", JSON.stringify(turn.scenario, null, 2), "```", "");
    for (const plan of turn.plans || []) lines.push(plan, "");
    turn.queries.forEach((query, j) => {
      lines.push(`**Probability query ${j + 1}** (${query.state})`, "", query.readback, "", query.statement, "");
      if (query.query) lines.push("```json", JSON.stringify(query.query, null, 2), "```", "");
      if (query.review) lines.push(`- Review: ${query.review}`);
      if (query.result) lines.push("Computed result:", "```json", JSON.stringify(query.result, null, 2), "```", "");
      if (query.reason) lines.push(`- Not computed: ${query.reason}`, "");
    });
    for (const analysis of turn.analyses) lines.push("**Analysis**", "```json", JSON.stringify(analysis, null, 2), "```", "");
    for (const proposal of turn.proposals) lines.push("**Proposal**", "```json", JSON.stringify(proposal, null, 2), "```", "");
    if (turn.clarification) lines.push("**Clarification**", "```json", JSON.stringify(turn.clarification, null, 2), "```", "");
    lines.push(turn.answer != null ? `**Answer:** ${turn.answer}` : `**No answer:** ${turn.error || "interrupted"}`, "");
    if (turn.error && turn.answer != null) lines.push(`**Connection error:** ${turn.error}`, "");
    if (turn.record) lines.push(`- Ask record: ${turn.record}`, "");
  });
  return lines.join("\n");
}

function downloadConversation() {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([conversationMarkdown()], {type: "text/markdown;charset=utf-8"}));
  link.download = `quakenet-conversation-${new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "")}.md`;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

function mountAsk() {
  $("#chat-form").addEventListener("submit", event => {
    event.preventDefault();
    if (state.asking) return;
    const question = $("#chat-q").value.trim();
    if (question) { $("#chat-q").value = ""; ask(question); }
  });
  $("#chat-q").addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); $("#chat-form").requestSubmit(); }
  });
  $("#chat-download").addEventListener("click", downloadConversation);
  document.addEventListener("click", event => {
    const button = event.target.closest("button[data-open-proposal]");
    if (!button) return;
    document.querySelector('nav button[data-pane="proposals"]').click();
    const card = [...$("#proposal-list").querySelectorAll("[data-proposal-id]")].find(el => el.dataset.proposalId === button.dataset.openProposal);
    card?.scrollIntoView({block: "nearest"}); card?.focus();
  });
}
