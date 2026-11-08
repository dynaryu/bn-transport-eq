// Run with: node src/app/tests/test_example_switching.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const page = fs.readFileSync(path.join(__dirname, "../quakenet.html"), "utf8");
const section = (from, to) => page.slice(page.indexOf(from), page.indexOf(to));
const code = section("async function load(id)", "// ---- controls:") +
  section("async function loadSession(", "function evidenceText(") +
  section("function showSession(s)", 'for (const form of document.querySelectorAll("#map-side .say"))');

class Element {
  constructor(tag = "div", root = false) {
    Object.assign(this, {tag, root, children: [], parent: null, className: "", attrs: {}, value: "", style: {}});
    this.classList = {add() {}, remove() {}};
  }
  get isConnected() { return this.root || !!this.parent?.isConnected; }
  append(child) { child.parent = this; this.children.push(child); }
  clear() { this.children.forEach(c => { c.parent = null; }); this.children = []; }
  set textContent(text) { this.clear(); this.text = text; }
  get textContent() { return (this.text || "") + this.children.map(c => c.textContent).join(""); }
  set innerHTML(html) { this.clear(); this.text = html; }
  get scrollHeight() { return this.children.length * 50; }
  setAttribute(key, value) { this.attrs[key] = value; }
  removeAttribute(key) { delete this.attrs[key]; }
  closest() { return this.form; }
  querySelector() { return this.button; }
  querySelectorAll() {
    return this.children.flatMap(c => [...(c.tag === "button" ? [c] : []), ...c.querySelectorAll()]);
  }
}

const clone = value => JSON.parse(JSON.stringify(value));
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return {promise, resolve}; };
const tick = async () => { for (let i = 0; i < 8; i++) await Promise.resolve(); };
const toy = "toy-ca/demo", ema = "ema-highway/gmm";

function session(id, completed = true) {
  const name = id === toy ? "Inspection priority for towns' access" : "Inspection priority for people's access";
  const query = {id: "q1", name, target: {kind: "cut_off"}, decision: "inspect"};
  const timeline = [{kind: "start", at: "12:00:00", values: {}, elapsed_s: 0.5}];
  if (completed) timeline.push({kind: "query", at: "12:00:01", query, values: {q1: {}}});
  return {n: id === toy ? 10000 : 20000, ess: 1, unit: id === toy ? "towns" : "people",
    queries: completed ? [query] : [], evidence: {}, variables: {}, timeline,
    scenario: {rows: {[id]: {mw: id === toy ? 6 : 7}}}, result: {marker: id, elapsed_s: 0.5}};
}

function harness(sessions = {[toy]: session(toy), [ema]: session(ema)}, storage = new Map()) {
  const ids = ["q-talk", "e-talk", "q-demo", "e-demo", "q-words", "e-words", "agent-status", "samples",
    "run", "new-session", "node-pop", "msg-pop", "road"];
  const nodes = Object.fromEntries(ids.map(id => ["#" + id, new Element("div", true)]));
  nodes.main = new Element("main", true);
  const controls = [];
  for (const id of ["q", "e"]) {
    const form = new Element("form", true);
    form.button = new Element("button", true);
    nodes[`#${id}-words`].form = form;
    controls.push(form.button, nodes[`#${id}-words`]);
  }
  const painted = {}, calls = [];
  const ctx = {console, state: {epoch: 0, loading: false, ex: null, session: null, preset: {}},
    $: selector => nodes[selector], esc: s => String(s ?? ""), share: () => "100%", cliquePick: {},
    store: {get: key => storage.get(key) || null, set: (key, value) => storage.set(key, value)},
    history: {replaceState() {}}, setStatus: message => { painted.status = message; }, agentStatus() {},
    currentScenario: () => ctx.state.preset, applyScenarioControls: () => { painted.scenario = ctx.state.preset; },
    document: {createElement: tag => new Element(tag), querySelectorAll: () =>
      [...controls, ...nodes["#q-demo"].children, ...nodes["#e-demo"].children]},
    setTimeout: cb => queueMicrotask(cb),
    getJSON: async url => {
      const parsed = new URL(url, "http://localhost"), id = parsed.searchParams.get("assembly");
      return parsed.pathname === "/api/example" ? {assembly: {id}} : {session: clone(sessions[id] || null)};
    },
    post: async (url, body) => { calls.push({url, body: clone(body)}); throw Error("Unexpected calculation request"); },
    renderRoadmap: () => { nodes["#road"].textContent = ctx.state.session?.result.marker || ""; },
  };
  for (const name of ["Controls", "Map", "Results", "BN", "Cliques", "Schema", "Assumptions"])
    ctx[`render${name}`] = () => { painted[name] = ctx.state.ex.assembly.id; };
  vm.createContext(ctx);
  vm.runInContext(code, ctx);
  return {ctx, nodes, painted, calls, storage, eval: source => vm.runInContext(source, ctx)};
}

async function check() {
  new vm.Script(page.match(/<script>([\s\S]*?)<\/script>/)[1]);
  const h = harness();
  await h.ctx.load(toy);
  assert.match(h.nodes["#q-talk"].textContent, /number of towns/);
  assert.doesNotMatch(h.nodes["#q-talk"].textContent, /pop_k/);
  h.nodes["#q-words"].value = "a toy draft";
  const toyChat = h.nodes["#q-talk"].textContent;
  await h.ctx.load(ema);
  assert.match(h.nodes["#q-talk"].textContent, /number of people/);
  assert.match(h.nodes["#q-talk"].textContent, /pop_k/);
  assert.doesNotMatch(h.nodes["#q-talk"].textContent, /number of towns/);
  assert.equal(h.nodes["#samples"].value, 20000);
  assert.equal(h.nodes["#road"].textContent, ema);
  assert.equal(h.painted.scenario.rows[ema].mw, 7);
  await h.ctx.load(toy);
  assert.equal(h.nodes["#q-talk"].textContent, toyChat);
  assert.equal(h.nodes["#q-words"].value, "a toy draft");
  const reopened = harness(undefined, h.storage);
  await reopened.ctx.load(toy);
  assert.equal(reopened.nodes["#q-talk"].textContent, toyChat);

  // An unfinished clarification and draft belong to their original network.
  const pending = harness({[toy]: session(toy, false), [ema]: session(ema)});
  await pending.ctx.load(toy);
  await pending.ctx.draft("query", "Inspection priority?");
  await pending.ctx.load(ema);
  await pending.ctx.load(toy);
  assert.equal(pending.eval('asked.query === DEMO["toy-ca/demo"][0]'), true);
  await pending.ctx.draft("query", "repair");
  assert.equal(pending.eval("pending.query.name"), "Inspection priority for towns' access");
  await pending.ctx.load(ema);
  assert.equal(pending.eval("pending.query"), null);
  await pending.ctx.load(toy);
  assert.equal(pending.eval("pending.query.name"), "Inspection priority for towns' access");

  // A slow example response and a slow session response must not overwrite a newer selection.
  for (const delayedPath of ["/api/example", "/api/session"]) {
    const race = harness(), gate = deferred(), original = race.ctx.getJSON;
    race.ctx.getJSON = async url => {
      const parsed = new URL(url, "http://localhost");
      if (parsed.pathname === delayedPath && parsed.searchParams.get("assembly") === toy) await gate.promise;
      return original(url);
    };
    const slow = race.ctx.load(toy);
    await tick();
    await race.ctx.load(ema);
    gate.resolve();
    await slow;
    assert.equal(race.ctx.state.ex.assembly.id, ema);
    assert.equal(race.nodes["#road"].textContent, ema);
    assert.equal(race.painted.Map, ema);
    assert.equal(race.nodes.main.inert, false);
  }

  // Calculations already submitted for the old network may finish, but cannot repaint the new network.
  const late = harness(), gate = deferred();
  await late.ctx.load(toy);
  late.ctx.post = async (url, body) => { assert.equal(body.assembly, toy); return gate.promise; };
  const calculation = late.ctx.sessionCall("/api/session/query", {query: {}}, "Computing");
  await late.ctx.load(ema);
  gate.resolve(session(toy));
  assert.equal(await calculation, false);
  assert.equal(late.nodes["#road"].textContent, ema);

  // Switching while the demo is typing cancels the remaining autoplay steps.
  const playing = harness({[toy]: session(toy, false), [ema]: session(ema)}), timers = [];
  await playing.ctx.load(toy);
  playing.ctx.setTimeout = cb => timers.push(cb);
  const playback = playing.nodes["#q-demo"].children[0].onclick();
  await playing.ctx.load(ema);
  while (timers.length) { timers.shift()(); await tick(); }
  await playback;
  assert.equal(playing.calls.length, 0);
  assert.equal(playing.eval("demoPlaying"), false);
  assert.match(playing.nodes["#q-talk"].textContent, /number of people/);

  const empty = harness({[toy]: session(toy)});
  await empty.ctx.load(toy);
  await empty.ctx.load(ema);
  assert.equal(empty.ctx.state.session, null);
  assert.equal(empty.nodes["#q-talk"].textContent, "Press Run first.");
  assert.equal(empty.nodes["#road"].textContent, "");

  const custom = "hebei-transport/shakemap", reply = deferred();
  const model = harness({[custom]: session(custom, false), [toy]: session(toy)});
  await model.ctx.load(custom);
  model.ctx.post = () => reply.promise;
  const drafting = model.ctx.draft("query", "a question for Hebei");
  await tick();
  await model.ctx.load(toy);
  reply.resolve({note: "A late reply for Hebei"});
  await drafting;
  assert.doesNotMatch(model.nodes["#q-talk"].textContent, /Hebei/);

  // The ordinary five-step demo still submits exactly its original decision and evidence.
  for (const id of [toy, ema]) {
    const data = {[id]: session(id, false)}, full = harness(data);
    full.ctx.post = async (url, body) => {
      full.calls.push({url, body: clone(body)});
      const s = data[id];
      if (url.endsWith("/query")) {
        const query = {...body.query, id: "q1"};
        s.queries.push(query);
        s.timeline.push({kind: "query", at: "12:00:01", query, values: {q1: {}}});
      } else {
        s.evidence[body.variable] = {value: body.value};
        s.timeline.push({kind: "evidence", at: "12:00:02", evidence: body, update: "reweighted", values: {q1: {}}});
      }
      return clone(s);
    };
    await full.ctx.load(id);
    for (let i = 0; i < 5; i++) {
      const button = full.nodes["#q-demo"].children[0] || full.nodes["#e-demo"].children[0];
      assert.ok(button);
      await button.onclick();
    }
    const expected = clone(full.eval(`DEMO[${JSON.stringify(id)}]`));
    assert.equal(full.calls.length, 5);
    full.calls.forEach(({url, body}, i) => {
      assert.equal(body.assembly, id);
      assert.equal(url, i === 0 ? "/api/session/query" : "/api/session/evidence");
      if (i === 0) assert.deepEqual(body.query, expected[i].draft);
      else {
        assert.equal(body.variable, expected[i].draft.variable);
        assert.equal(body.value, expected[i].draft.value);
      }
    });
  }
  return "Example switching passed: separate chats, reload, drafts, scenarios, samples, roadmaps, response races, and typing cancellation.";
}

module.exports = check;
if (require.main === module) check().then(console.log).catch(error => { console.error(error); process.exitCode = 1; });
