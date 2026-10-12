/* Forward BN pilot - stakeholder interface.
 *
 * Two graphs carry the explanation, and they answer different questions:
 *   #bn-svg       what the model IS - the assembled DAG in plate notation, including
 *                 the extension points that are not wired up yet.
 *   #network-svg  what the model SAYS - the physical network, coloured by result.
 *
 * Colour encodes magnitude only, as two single-hue sequential ramps (orange for
 * risk, teal for reliability). Both ramps and the diagram's identity pair were
 * checked against the OKLab/CVD gates rather than chosen by eye; the passing steps
 * are the RISK_RAMP / SERVICE_RAMP anchors below. Risk is also redundantly encoded
 * as stroke width, so the map survives colour-vision deficiency and greyscale print.
 */

const $ = (id) => document.getElementById(id);

let catalog = [];
let current = null;
let lastResult = null;
let mapColourMode = 'risk';
let view = { k: 1, x: 0, y: 0 };      // map zoom/pan
let projected = null;                  // cached node screen coordinates

const pct = (v, dp = 1) => v == null ? '—' : `${(100 * v).toFixed(dp)}%`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const nodeLabel = (id) => {
  const n = current?.network?.nodes?.[id];
  if (!n) return id;
  return n.city ? `${id} · ${n.city}` : n.label ? `${id} · ${n.label}` : id;
};

/* ── validated sequential ramps ──────────────────────────────────────────────
 * Anchors cleared the ordinal gate on the panel surface (#fffefa): monotone
 * lightness, step gaps >= 0.06 OKLCH L, single hue, light end >= 2:1 contrast. */
const RISK_RAMP = ['#f09468', '#e56b38', '#b94d21', '#8f3914'];
const SERVICE_RAMP = ['#7fc2bd', '#3da39c', '#0a7d77', '#0b5551'];
const NEUTRAL_EDGE = '#c2cbc8';

/* Fixed, policy-meaningful bands rather than a per-run rescaling. A colour has to
 * mean the same thing in every scenario, otherwise two runs cannot be compared -
 * which is most of what this tool is for. The observed range goes in the legend so
 * a mild scenario is not mistaken for a broken map. */
const RISK_BANDS = [0.05, 0.15, 0.35];
const SERVICE_BANDS = [0.5, 0.75, 0.9];    // ascending; reliability is read downward

const bandIndex = (v, edges) => edges.reduce((acc, e) => acc + (v >= e ? 1 : 0), 0);
const riskColour = (v) => RISK_RAMP[bandIndex(v ?? 0, RISK_BANDS)];
// Reliability: high is good, so the darkest step marks the LOWEST reliability.
const serviceColour = (v) => SERVICE_RAMP[3 - bandIndex(v ?? 0, SERVICE_BANDS)];

/* ── tooltip ─────────────────────────────────────────────────────────────── */
const tip = {
  show(html, evt) {
    const el = $('tooltip');
    el.innerHTML = html;
    el.hidden = false;
    const pad = 14, w = el.offsetWidth, h = el.offsetHeight;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + w > innerWidth - 8) x = evt.clientX - w - pad;
    if (y + h > innerHeight - 8) y = evt.clientY - h - pad;
    el.style.transform = `translate(${x}px, ${y}px)`;
  },
  hide() { $('tooltip').hidden = true; },
};

/* ── boot ────────────────────────────────────────────────────────────────── */
async function boot() {
  let response;
  try {
    response = await fetch('./api/catalog', { cache: 'no-store' });
    if (!response.ok) throw new Error(`catalog returned ${response.status}`);
  } catch {
    // Opened as a plain file, or the service is not running. Say so plainly and
    // stop, rather than leaving dead controls on screen.
    $('offline').hidden = false;
    $('status').className = 'status error';
    $('status').textContent = 'Not connected to the pilot service.';
    document.querySelectorAll('select, input, button').forEach((el) => { el.disabled = true; });
    return;
  }
  catalog = (await response.json()).packages;
  $('case-select').innerHTML = catalog.map((c) => `<option value="${esc(c.case_id)}">${esc(c.title)}</option>`).join('');
  $('case-select').addEventListener('change', loadCase);
  $('hazard-model').addEventListener('change', renderScenarioFields);
  $('samples').addEventListener('input', () => { $('samples-value').value = Number($('samples').value).toLocaleString(); });
  $('run').addEventListener('click', runQuery);
  $('show-future').addEventListener('change', drawBnGraph);
  document.querySelectorAll('[data-bnview]').forEach((btn) => btn.addEventListener('click', () => {
    document.querySelectorAll('[data-bnview]').forEach((b) => b.classList.toggle('is-on', b === btn));
    bnView = btn.dataset.bnview;
    $('show-future').closest('.switch').style.visibility = bnView === 'plate' ? '' : 'hidden';
    drawBnGraph();
  }));
  $('reset-view').addEventListener('click', () => { view = { k: 1, x: 0, y: 0 }; drawNetwork(lastResult); });
  // Scoped to [data-colour]: the BN view switch uses the same .seg-btn styling,
  // and a bare .seg-btn selector would drive both groups from either click.
  document.querySelectorAll('[data-colour]').forEach((btn) => btn.addEventListener('click', () => {
    document.querySelectorAll('[data-colour]').forEach((b) => b.classList.toggle('is-on', b === btn));
    mapColourMode = btn.dataset.colour;
    drawNetwork(lastResult);
  }));
  addEventListener('resize', debounce(() => { drawBnGraph(); drawNetwork(lastResult); drawRiskChart(lastResult); }, 160));
  await loadCase();
}

const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

async function loadCase() {
  const id = $('case-select').value;
  current = await (await fetch(`./api/package/${id}`)).json();
  lastResult = null;
  bnInstance = null;          // structure is per package
  view = { k: 1, x: 0, y: 0 };
  const n = current.network;

  $('case-meta').textContent =
    `${Object.keys(n.nodes).length} nodes · ${Object.keys(n.edges).length} ${n.directed ? 'directed arcs' : 'undirected links'}. ${n.description}`;
  $('hazard-model').innerHTML = n.hazard_models
    .map((m) => `<option value="${esc(m)}">${HAZARD_LABELS[m] || esc(m)}</option>`).join('');
  $('hazard-model').value = n.default_hazard.model;

  const origins = n.default_origins;
  $('origin-select').innerHTML = `<option value="__all__">All configured origins (${origins.length})</option>`
    + origins.map((o) => `<option value="${esc(o)}">${esc(nodeLabel(o))}</option>`).join('');
  $('destination-list').innerHTML = '<label>Destinations</label>'
    + n.destinations.map((d) => `<span class="check"><input type="checkbox" value="${esc(d)}" checked> ${esc(nodeLabel(d))}</span>`).join('');

  $('network-title').textContent = n.title;
  $('assumptions').innerHTML = Object.values(n.assumptions).map((v) => `<li>${esc(v)}</li>`).join('');
  $('extensions').innerHTML = current.bn_spec.extension_points
    .map((x) => `<li><strong>${esc(x.name)}:</strong> ${esc(x.future_arc)}</li>`).join('');

  renderScenarioFields();
  drawBnGraph();
  drawNetwork();
  drawRiskChart(null);
  resetMetrics();
  if (!$('chat-card').hidden) { chat.history = []; chat.transcript = []; $('chat-download').disabled = true; renderExamples(); }
  $('status').className = 'status';
  $('status').textContent = 'Package loaded. Adjust the earthquake and run forward inference.';
}

const HAZARD_LABELS = {
  uniform_sa1: 'Uniform SA(1 s)',
  ema_reference_scenario: 'EMA reference scenario',
  shakemap: 'USGS ShakeMap (recorded event)',
};

function renderScenarioFields() {
  const model = $('hazard-model').value;
  const d = current.network.default_hazard;
  if (model === 'shakemap') {
    $('scenario-fields').innerHTML = `<label for="shakemap-event">Event with a ShakeMap</label>
      <select id="shakemap-event"><option value="">Searching USGS…</option></select>
      <label for="shakemap-id">or USGS event id</label>
      <input id="shakemap-id" type="text" spellcheck="false" placeholder="e.g. iscgem849480">`;
    loadShakemapEvents();
    return;
  }
  $('scenario-fields').innerHTML = model === 'uniform_sa1'
    ? `<label for="sa1">SA(1 s), g</label><input id="sa1" type="number" min="0" max="5" step="0.05" value="${d.sa1_g ?? 0.30}">`
    : `<label for="magnitude">Magnitude, Mw</label><input id="magnitude" type="number" min="4" max="9.5" step="0.1" value="${d.magnitude ?? 8}">
       <div class="field-row">
         <div><label for="epicentre-x">Epicentre X</label><input id="epicentre-x" type="number" step="1" value="${d.epicentre_x ?? 1}"></div>
         <div><label for="epicentre-y">Epicentre Y</label><input id="epicentre-y" type="number" step="1" value="${d.epicentre_y ?? 10}"></div>
       </div>`;
}

/* Events are searched around the package's own node coordinates, so the list
 * only offers ShakeMaps that cover the network. A typed id overrides it. */
async function loadShakemapEvents() {
  const select = $('shakemap-event');
  try {
    const response = await fetch(`./api/shakemap-events/${current.network.case_id}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'search failed');
    select.innerHTML = data.events.length
      ? data.events.map((e) => `<option value="${esc(e.event_id)}">${esc((e.time || '').slice(0, 10))} · M${e.magnitude ?? '?'} · ${esc(e.place || e.event_id)}</option>`).join('')
      : '<option value="">No ShakeMaps found near this network</option>';
  } catch (err) {
    select.innerHTML = `<option value="">${esc(`Could not search (${err.message}); type an id below`)}</option>`;
  }
}

/* ══ 1. THE BAYESIAN NETWORK ═══════════════════════════════════════════════
 * Plate notation, built from bn_spec.json. The individual spec lists hundreds of
 * nodes (SA1::e0001 ...); drawing them one by one would communicate nothing, so
 * each repeated family becomes one box carrying its count. */
function bnLayout() {
  const spec = current.bn_spec;
  const nEdges = Object.keys(current.network.edges).length;
  const byType = (t) => spec.factor_templates.find((f) => f.type === t);
  const count = (t) => (byType(t)?.nodes?.length) ?? 0;
  const serviceCount = count('deterministic_network_connectivity') || Object.keys(current.network.nodes).length;

  return {
    input: { id: 'eq', label: 'Earthquake', sub: 'magnitude, epicentre', role: 'input' },
    plateA: {
      label: `per bridge × ${nEdges}`,
      nodes: [
        { id: 'sa1', label: 'SA(1 s)', sub: 'ground motion, g', role: 'model' },
        { id: 'dmg', label: 'Damage', sub: 'Extensive threshold', role: 'model' },
        { id: 'use', label: 'Usable', sub: 'open / closed', role: 'model' },
      ],
    },
    plateB: {
      label: `per node × ${serviceCount}`,
      nodes: [{ id: 'svc', label: 'Service', sub: 'reaches a destination', role: 'model' }],
    },
    output: { id: 'pop', label: 'Population', sub: 'the decision metric', role: 'model' },
    future: [
      { id: 'fragpar', label: 'Fragility parameters', sub: 'median, dispersion' },
      { id: 'interv', label: 'Intervention', sub: 'retrofit, replace' },
      { id: 'deter', label: 'Deterioration', sub: 'age, condition' },
      { id: 'resid', label: 'Spatial residual', sub: 'correlated motion' },
    ],
  };
}

function drawBnGraph() {
  const svg = $('bn-svg');
  if (!current) return;
  if (bnView === 'instance') { drawBnInstance(); return; }
  if (bnView === 'layers') { drawBnLayers(); return; }
  $('bn-svg').onmousemove = null;        // the layer view installs its own hover
  $('bn-svg').onmouseleave = null;
  $('bn-note').hidden = true;
  setBnCaption('plate');
  const showFuture = $('show-future').checked;
  const L = bnLayout();

  // Geometry is laid out left to right and measured, not guessed, so the plates
  // cannot swallow the input box and the labels cannot run past their borders.
  const boxW = 158, boxH = 56, gap = 44, padX = 10, platePad = 16;
  const inputX = padX;
  const plateAx = inputX + boxW + gap;
  const aBoxX = plateAx + platePad;
  const plateAw = 3 * boxW + 2 * gap + 2 * platePad;
  const plateBx = plateAx + plateAw + gap;
  const bBoxX = plateBx + platePad;
  const plateBw = boxW + 2 * platePad;
  const outputX = plateBx + plateBw + gap;
  const W = outputX + boxW + padX;

  const futureRowY = 34;
  const rowY = showFuture ? 196 : 40;
  const residY = rowY + boxH + 58;
  const blockedY = (showFuture ? residY + boxH : rowY + boxH) + 46;
  const H = blockedY + 26;

  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  svg.style.height = `${Math.round(H * 0.70)}px`;

  const chainBoxes = [
    { ...L.input, x: inputX, y: rowY },
    ...L.plateA.nodes.map((n, i) => ({ ...n, x: aBoxX + i * (boxW + gap), y: rowY })),
    { ...L.plateB.nodes[0], x: bBoxX, y: rowY },
    { ...L.output, x: outputX, y: rowY },
  ];
  const dmg = chainBoxes[2], sa1 = chainBoxes[1];

  let g = '';

  // -- plates ---------------------------------------------------------------
  const plate = (x, w, label) => `
    <g class="plate">
      <rect x="${x}" y="${rowY - 26}" width="${w}" height="${boxH + 56}" rx="10"></rect>
      <text class="plate-label" x="${x + w - 11}" y="${rowY + boxH + 21}" text-anchor="end">${esc(label)}</text>
    </g>`;
  g += plate(plateAx, plateAw, L.plateA.label);
  g += plate(plateBx, plateBw, L.plateB.label);

  // -- the solid chain ------------------------------------------------------
  for (let i = 0; i < chainBoxes.length - 1; i++) {
    const a = chainBoxes[i], b = chainBoxes[i + 1];
    g += `<line class="bn-arrow" x1="${a.x + boxW}" y1="${a.y + boxH / 2}" x2="${b.x - 8}" y2="${b.y + boxH / 2}" marker-end="url(#bn-head)"></line>`;
  }

  // -- planned extensions ---------------------------------------------------
  if (showFuture) {
    const at = {
      fragpar: { x: dmg.x, y: futureRowY },
      interv: { x: dmg.x - boxW - gap, y: futureRowY },
      deter: { x: dmg.x + boxW + gap, y: futureRowY },
      resid: { x: sa1.x, y: residY },
    };
    for (const f of L.future) g += futureBox(at[f.id].x, at[f.id].y, boxW, boxH, f);

    const fMid = futureRowY + boxH / 2;
    const fa = (x1, y1, x2, y2) => `<line class="bn-arrow future" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" marker-end="url(#bn-head-future)"></line>`;
    g += fa(at.interv.x + boxW, fMid, at.fragpar.x - 8, fMid);
    g += fa(at.deter.x, fMid, at.fragpar.x + boxW + 8, fMid);
    g += fa(at.fragpar.x + boxW / 2, futureRowY + boxH, dmg.x + boxW / 2, rowY - 8);
    g += fa(at.resid.x + boxW / 2, residY, sa1.x + boxW / 2, rowY + boxH + 8);
  }

  // -- the boundary: backward inference is not offered -----------------------
  g += `<g class="blocked">
      <line x1="${outputX + boxW - 20}" y1="${blockedY}" x2="${inputX + 20}" y2="${blockedY}" marker-end="url(#bn-head-blocked)"></line>
      <text x="${W / 2}" y="${blockedY - 11}" text-anchor="middle">backward inference — not in scope for this pilot</text>
    </g>`;

  // -- boxes last, so they sit above the plates and the arrows ---------------
  for (const b of chainBoxes) {
    g += `<g class="bn-node ${b.role}">
        <rect x="${b.x}" y="${b.y}" width="${boxW}" height="${boxH}" rx="7"></rect>
        <text class="bn-title" x="${b.x + 12}" y="${b.y + 23}">${esc(b.label)}</text>
        <text class="bn-sub" x="${b.x + 12}" y="${b.y + 41}">${esc(b.sub)}</text>
      </g>`;
  }

  svg.innerHTML = `<defs>
      <marker id="bn-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5d6f74"></path></marker>
      <marker id="bn-head-future" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#9aa7a4"></path></marker>
      <marker id="bn-head-blocked" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#b2452a"></path></marker>
    </defs>${g}`;
}

/* ── the same network, drawn as actual nodes ───────────────────────────────
 * The plate view shows the repetition as repetition; this shows what one plate
 * stands for. The structure comes from the live tbnpy model, so a node that
 * gains a parent gains an arrow here without anyone redrawing anything. */
let bnView = 'plate';
let bnInstance = null;

const BN_CAPTIONS = {
  plate: 'Plate notation: one box stands for every bridge or node in the package. Arrows are the direction evidence travels.',
  instance: 'The actual tbnpy nodes, read off the assembled model. A few bridges are shown; the rest repeat the same chain.',
  layers: 'Every variable at the place it describes: ground motion above each bridge, usability at the bridge itself, service at the junctions below.',
};

function setBnCaption(view) {
  const card = document.querySelector('.graph-card');
  card.querySelector('.sub').textContent = BN_CAPTIONS[view];
  // The planned-extension and out-of-scope keys belong to the plate drawing only.
  card.querySelectorAll('.graph-legend span').forEach((el, i) => {
    el.style.display = (view !== 'plate' && i >= 2) ? 'none' : '';
  });
}

const BN_KIND_CLASS = { hazard: 'model', asset: 'model', system: 'model', impact: 'model' };

async function drawBnInstance() {
  const svg = $('bn-svg');
  const note = $('bn-note');
  setBnCaption('instance');
  if (!bnInstance) {
    try {
      const response = await fetch(`./api/bn-graph/${current.network.case_id}?edges=3`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'could not read the model structure');
      bnInstance = data;
    } catch (error) {
      svg.innerHTML = '';
      note.hidden = false;
      note.textContent = error.message;
      return;
    }
  }

  const g = bnInstance;
  const cols = [[], [], [], []];
  const colOf = { hazard: 0, asset: 1, system: 2, impact: 3 };
  for (const n of g.nodes) cols[colOf[n.kind] ?? 3].push(n);

  const boxW = 168, boxH = 46, colGap = 84, rowGap = 16, padX = 12, padY = 26;
  const colX = (i) => padX + i * (boxW + colGap);
  const place = {};
  const heights = cols.map((c) => c.length * boxH + (c.length - 1) * rowGap);
  const H = Math.max(...heights) + 2 * padY;
  cols.forEach((col, i) => {
    const top = (H - heights[i]) / 2;
    col.forEach((n, j) => { place[n.id] = { x: colX(i), y: top + j * (boxH + rowGap) }; });
  });
  const W = colX(3) + boxW + padX;

  let out = `<defs><marker id="bn-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#5d6f74"></path></marker></defs>`;

  // Arrows use the PARENT's own box where it is drawn, so the picture never
  // implies a dependency the model does not have.
  for (const n of g.nodes) {
    for (const parentVar of n.parents) {
      const owner = parentVar.startsWith('S::') ? 'S' : parentVar;
      const from = place[owner], to = place[n.id];
      if (!from || !to) continue;
      out += `<line class="bn-arrow" x1="${from.x + boxW}" y1="${from.y + boxH / 2}" x2="${to.x - 8}" y2="${to.y + boxH / 2}" marker-end="url(#bn-head)"></line>`;
    }
    if (n.hidden_parents > 0) {
      const to = place[n.id];
      out += `<text class="bn-hidden" x="${to.x + boxW / 2}" y="${to.y - 7}" text-anchor="middle">+${n.hidden_parents} more parents</text>`;
    }
  }

  for (const n of g.nodes) {
    const p = place[n.id];
    const label = n.id === 'S' ? `S :: ${g.shown.origins.join(', ')}` : n.id;
    const sub = n.id === 'S'
      ? `${n.n_childs} origins, one node`
      : n.root ? 'root · scenario sets it'
      : n.deterministic ? 'deterministic' : 'conditional';
    out += `<g class="bn-node ${BN_KIND_CLASS[n.kind] || 'model'}">
        <rect x="${p.x}" y="${p.y}" width="${boxW}" height="${boxH}" rx="7"></rect>
        <title>${esc(n.description)}</title>
        <text class="bn-title mono" x="${p.x + 11}" y="${p.y + 20}">${esc(label)}</text>
        <text class="bn-sub" x="${p.x + 11}" y="${p.y + 36}">${esc(sub)}</text>
      </g>`;
  }

  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  svg.style.height = `${Math.round(H * 0.82)}px`;
  svg.innerHTML = out;

  note.hidden = false;
  note.innerHTML = `<strong>${g.totals.variables} variables, ${g.totals.nodes} nodes</strong> `
    + `for ${g.totals.bridges} bridges and ${g.totals.origins} origins. ${esc(g.note)} `
    + `<span class="dim">"+n" marks parents that exist in the model but are not drawn.</span>`;
}

/* ── the same network again, stacked over its own geography ────────────────
 * Each layer is one family of variables drawn at the place it belongs to: the
 * X layer sits at the bridge midpoints, SA1 directly above its own bridge, and
 * the service variables at the junctions they describe. Vertical lines are the
 * model's arrows, so "ground motion here causes damage here" is a line you can
 * follow rather than a sentence you have to take on trust.
 *
 * The hourglass in the middle is the honest shape of the system node: 129
 * bridges converge into ONE deterministic function, which then determines 72
 * service variables. */
function bnGeometry(net) {
  const nodes = net.nodes;
  const xs = Object.values(nodes).map((n) => Number(n.x));
  const ys = Object.values(nodes).map((n) => Number(n.y));
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const spanX = (maxX - minX) || 1, spanY = (maxY - minY) || 1;
  // Normalised, so every layer uses one coordinate system and the stack lines up.
  const uv = (x, y) => [(Number(x) - minX) / spanX, (Number(y) - minY) / spanY];
  return {
    uv,
    midpoint: (edge) => {
      const a = nodes[edge.from], b = nodes[edge.to];
      return uv((Number(a.x) + Number(b.x)) / 2, (Number(a.y) + Number(b.y)) / 2);
    },
  };
}

const BN_LAYERS = { PW: 660, PD: 118, SHEAR: 168, OX: 118 };

function planePoint(u, v, layerY) {
  const { PW, PD, SHEAR, OX } = BN_LAYERS;
  return [OX + u * PW + v * SHEAR, layerY + v * PD];
}

function planeOutline(layerY) {
  const corners = [[0, 0], [1, 0], [1, 1], [0, 1]]
    .map(([u, v]) => planePoint(u, v, layerY).map((n) => n.toFixed(1)).join(','));
  return `<polygon class="bn-plane" points="${corners.join(' ')}"></polygon>`;
}

function drawBnLayers() {
  const svg = $('bn-svg');
  const note = $('bn-note');
  setBnCaption('layers');

  const net = current.network;
  const geom = bnGeometry(net);
  const result = lastResult;
  const risk = Object.fromEntries((result?.components || []).map((c) => [c.edge_id, c]));
  const service = Object.fromEntries((result?.origin_service || []).map((o) => [o.origin, o]));

  const edges = Object.entries(net.edges);
  const origins = net.default_origins;
  // Mirrors model.build: no weights, no PopFraction node. Drawing one anyway
  // would show a variable the assembled model does not contain.
  const hasPopulation = origins.some((o) => Number(net.nodes[o]?.population) > 0);

  const Y = { sa1: 26, x: 212, hub: 408, s: 470, pop: 672 };
  const W = BN_LAYERS.OX + BN_LAYERS.PW + BN_LAYERS.SHEAR + 40;
  const H = hasPopulation ? Y.pop + 70 : Y.s + BN_LAYERS.PD + 50;
  const hub = [BN_LAYERS.OX + BN_LAYERS.PW / 2 + BN_LAYERS.SHEAR / 2, Y.hub];
  const popBox = { w: 150, h: 40 };
  const pop = [hub[0] - popBox.w / 2, Y.pop];

  // Intensity drives radius, not a third colour ramp: two sequential ramps are
  // already in play and a third would stop any of them reading as magnitude.
  const intensities = edges.map(([id]) => risk[id]?.sa1_g ?? 0);
  const maxIntensity = Math.max(0.0001, ...intensities);

  let g = '';
  for (const y of [Y.sa1, Y.x, Y.s]) g += planeOutline(y);

  // --- vertical arrows, SA1 -> X, one per bridge --------------------------
  for (const [id, edge] of edges) {
    const [u, v] = geom.midpoint(edge);
    const [x1, y1] = planePoint(u, v, Y.sa1);
    const [x2, y2] = planePoint(u, v, Y.x);
    g += `<line class="bn-drop" x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}"></line>`;
  }

  // --- the hourglass: every X into one node, out to every service variable --
  for (const [id, edge] of edges) {
    const [u, v] = geom.midpoint(edge);
    const [x1, y1] = planePoint(u, v, Y.x);
    g += `<line class="bn-bundle" x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${hub[0]}" y2="${hub[1]}"></line>`;
  }
  for (const origin of origins) {
    const n = net.nodes[origin];
    if (!n) continue;
    const [u, v] = geom.uv(n.x, n.y);
    const [x2, y2] = planePoint(u, v, Y.s);
    g += `<line class="bn-bundle out" x1="${hub[0]}" y1="${hub[1]}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}"></line>`;
  }
  if (hasPopulation) {
    for (const origin of origins.slice(0, 30)) {
      const n = net.nodes[origin];
      if (!n) continue;
      const [u, v] = geom.uv(n.x, n.y);
      const [x1, y1] = planePoint(u, v, Y.s);
      g += `<line class="bn-bundle out" x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${pop[0] + popBox.w / 2}" y2="${Y.pop}"></line>`;
    }
  }

  // --- the marks -----------------------------------------------------------
  for (const [id, edge] of edges) {
    const [u, v] = geom.midpoint(edge);
    const c = risk[id];
    const [sx, sy] = planePoint(u, v, Y.sa1);
    const r = c ? 2.2 + 3.4 * Math.sqrt((c.sa1_g || 0) / maxIntensity) : 2.6;
    g += `<circle class="bn-dot sa1" data-bnedge="${esc(id)}" cx="${sx.toFixed(1)}" cy="${sy.toFixed(1)}" r="${r.toFixed(1)}"></circle>`;

    const [xx, xy] = planePoint(u, v, Y.x);
    const fill = c ? riskColour(c.p_unusable) : '#ffffff';
    g += `<circle class="bn-dot x" data-bnedge="${esc(id)}" cx="${xx.toFixed(1)}" cy="${xy.toFixed(1)}" r="4" fill="${fill}"></circle>`;
  }
  for (const origin of origins) {
    const n = net.nodes[origin];
    if (!n) continue;
    const [u, v] = geom.uv(n.x, n.y);
    const [sx, sy] = planePoint(u, v, Y.s);
    const s = service[origin];
    const fill = s ? serviceColour(s.probability) : '#ffffff';
    g += `<circle class="bn-dot s" data-bnorigin="${esc(origin)}" cx="${sx.toFixed(1)}" cy="${sy.toFixed(1)}" r="4" fill="${fill}"></circle>`;
  }
  for (const id of net.destinations) {
    const n = net.nodes[id];
    if (!n) continue;
    const [u, v] = geom.uv(n.x, n.y);
    const [sx, sy] = planePoint(u, v, Y.s);
    g += `<path class="bn-dest" d="M ${sx} ${sy - 7} L ${sx + 7} ${sy} L ${sx} ${sy + 7} L ${sx - 7} ${sy} Z"></path>`;
  }

  g += `<g class="bn-hub"><circle cx="${hub[0]}" cy="${hub[1]}" r="17"></circle>
      <text x="${hub[0]}" y="${hub[1] + 4}" text-anchor="middle">S</text></g>`;
  if (hasPopulation) {
    g += `<g class="bn-node model"><rect x="${pop[0]}" y="${pop[1]}" width="${popBox.w}" height="${popBox.h}" rx="7"></rect>
        <text class="bn-title mono" x="${pop[0] + 11}" y="${pop[1] + 17}">PopFraction</text>
        <text class="bn-sub" x="${pop[0] + 11}" y="${pop[1] + 31}">the decision metric</text></g>`;
  }

  const label = (y, title, sub) =>
    `<text class="bn-layer" x="10" y="${y + 18}">${esc(title)}</text>`
    + `<text class="bn-layer sub" x="10" y="${y + 33}">${esc(sub)}</text>`;
  g += label(Y.sa1, `SA1 × ${edges.length}`, 'ground motion');
  g += label(Y.x, `X × ${edges.length}`, 'bridge usable?');
  g += label(Y.s, `S × ${origins.length}`, 'origin served?');

  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  svg.style.height = `${Math.round(H * 0.74)}px`;
  svg.innerHTML = g;
  attachLayerHover(risk, service);

  note.hidden = false;
  note.innerHTML = result
    ? `Coloured by the last run. <strong>Each column is one bridge:</strong> its ground motion, its usability, and the junctions downstream. `
      + `<span class="dim">SA1 dot size is intensity; X colour is failure probability; S colour is reliability.</span>`
    : `<strong>Each column is one bridge:</strong> ground motion above, usability at the bridge's own position, service below. `
      + `<span class="dim">Run a scenario to colour the layers.</span>`;
}

function attachLayerHover(risk, service) {
  const svg = $('bn-svg');
  svg.onmousemove = (evt) => {
    const t = evt.target.closest('[data-bnedge],[data-bnorigin]');
    svg.querySelectorAll('.hot').forEach((e) => e.classList.remove('hot'));
    if (!t) { tip.hide(); return; }
    if (t.dataset.bnedge) {
      const id = t.dataset.bnedge, c = risk[id];
      // Both layers of the same bridge light up: one column, one component.
      svg.querySelectorAll(`[data-bnedge="${CSS.escape(id)}"]`).forEach((e) => e.classList.add('hot'));
      tip.show(`<b>${esc(id)}</b><span>SA1::${esc(id)} → X::${esc(id)}</span>`
        + (c ? `<span class="hi">P(unusable) ${pct(c.p_unusable, 2)}</span><span>SA(1 s) ${c.sa1_g.toFixed(3)} g</span>`
             : '<span class="dim">Run a scenario</span>'), evt);
    } else {
      const id = t.dataset.bnorigin, s = service[id];
      t.classList.add('hot');
      tip.show(`<b>S::${esc(id)}</b><span>${esc(nodeLabel(id))}</span>`
        + (s ? `<span class="hi">P(reaches destination) ${pct(s.probability, 2)}</span>`
             : '<span class="dim">Run a scenario</span>'), evt);
    }
  };
  svg.onmouseleave = () => { tip.hide(); svg.querySelectorAll('.hot').forEach((e) => e.classList.remove('hot')); };
}

function futureBox(x, y, w, h, f) {
  return `<g class="bn-node future">
      <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7"></rect>
      <text class="bn-title" x="${x + 12}" y="${y + 23}">${esc(f.label)}</text>
      <text class="bn-sub" x="${x + 12}" y="${y + 40}">${esc(f.sub)}</text>
    </g>`;
}

/* ══ 2. THE PHYSICAL NETWORK ═══════════════════════════════════════════════ */
function project() {
  const nodes = current.network.nodes;
  const svg = $('network-svg');
  const W = svg.clientWidth || 900, H = svg.clientHeight || 430;
  const xs = Object.values(nodes).map((n) => Number(n.x));
  const ys = Object.values(nodes).map((n) => Number(n.y));
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const pad = 46;   // room for the destination labels, which extend right and up

  // One scale for both axes. The previous version stretched each axis to fill the
  // box, which silently distorted the geography - EMA coordinates are kilometres.
  const k = Math.min((W - 2 * pad) / (maxX - minX || 1), (H - 2 * pad) / (maxY - minY || 1));
  const ox = (W - (maxX - minX) * k) / 2;
  const oy = (H - (maxY - minY) * k) / 2;
  return {
    W, H,
    at: (n) => [ox + (Number(n.x) - minX) * k, H - oy - (Number(n.y) - minY) * k],
  };
}

function drawNetwork(result = null) {
  const svg = $('network-svg');
  if (!current) return;
  const net = current.network;
  const p = project();
  projected = p;
  svg.setAttribute('viewBox', `0 0 ${p.W} ${p.H}`);

  const risk = Object.fromEntries((result?.components || []).map((c) => [c.edge_id, c.p_unusable]));
  const service = Object.fromEntries((result?.origin_service || []).map((o) => [o.origin, o.probability]));
  const byNode = mapColourMode === 'service' && result;

  let g = '';
  for (const [id, e] of Object.entries(net.edges)) {
    const [x1, y1] = p.at(net.nodes[e.from]);
    const [x2, y2] = p.at(net.nodes[e.to]);
    const r = risk[id];
    const colour = result && !byNode ? riskColour(r) : NEUTRAL_EDGE;
    // Width repeats the risk signal, so the map still reads in greyscale or with
    // colour-vision deficiency. It steps with the band, not continuously, so the
    // line weight and the colour never disagree.
    const width = result && !byNode ? [1.4, 2.4, 3.6, 5.0][bandIndex(r ?? 0, RISK_BANDS)] : 1.6;
    g += `<line class="edge" data-edge="${esc(id)}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"
            stroke="${colour}" stroke-width="${width.toFixed(2)}"
            ${net.directed ? 'marker-end="url(#net-head)"' : ''}></line>`;
  }

  for (const [id, n] of Object.entries(net.nodes)) {
    const [x, y] = p.at(n);
    const dest = net.destinations.includes(id);
    const sv = service[id];
    const fill = byNode && sv != null ? serviceColour(sv) : '#ffffff';
    if (dest) {
      // Destinations are a different SHAPE as well as a different colour, so they
      // stay identifiable in greyscale and under colour-vision deficiency.
      const s = 8;
      g += `<path class="node destination" data-node="${esc(id)}" d="M ${x} ${y - s} L ${x + s} ${y} L ${x} ${y + s} L ${x - s} ${y} Z"></path>`;
    } else {
      g += `<circle class="node" data-node="${esc(id)}" cx="${x}" cy="${y}" r="4.2" fill="${fill}"></circle>`;
    }
  }
  for (const id of net.destinations) {
    const n = net.nodes[id];
    if (!n) continue;
    const [x, y] = p.at(n);
    g += `<text class="node-label strong" x="${x + 12}" y="${y - 10}">${esc(nodeLabel(id))}</text>`;
  }

  svg.innerHTML = `<defs>
      <marker id="net-head" viewBox="0 0 10 10" refX="16" refY="5" markerWidth="4.5" markerHeight="4.5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#8fa3a0"></path></marker>
    </defs><g id="map-zoom" transform="translate(${view.x},${view.y}) scale(${view.k})">${g}</g>`;

  attachMapInteraction(result);
  drawMapLegend(result);
}

function attachMapInteraction(result) {
  const svg = $('network-svg');
  const net = current.network;
  const risk = Object.fromEntries((result?.components || []).map((c) => [c.edge_id, c]));
  const service = Object.fromEntries((result?.origin_service || []).map((o) => [o.origin, o]));

  svg.onmousemove = (evt) => {
    const t = evt.target.closest('[data-edge],[data-node]');
    if (!t) { tip.hide(); svg.querySelectorAll('.hot').forEach((e) => e.classList.remove('hot')); return; }
    svg.querySelectorAll('.hot').forEach((e) => e.classList.remove('hot'));
    t.classList.add('hot');
    if (t.dataset.edge) {
      const id = t.dataset.edge, e = net.edges[id], c = risk[id];
      tip.show(`<b>${esc(id)}</b><span>${esc(e.from)} → ${esc(e.to)}</span>
        <span>Bridge class ${esc(e.bridge_class || '—')}</span>
        ${c ? `<span class="hi">P(unusable) ${pct(c.p_unusable, 2)}</span><span>SA(1 s) ${c.sa1_g.toFixed(3)} g</span>` : '<span class="dim">Run a scenario for risk</span>'}`, evt);
    } else {
      const id = t.dataset.node, s = service[id];
      const n = net.nodes[id];
      tip.show(`<b>${esc(nodeLabel(id))}</b>
        ${net.destinations.includes(id) ? '<span class="hi">Destination</span>' : ''}
        ${n.population != null ? `<span>Population ${Math.round(n.population).toLocaleString()}</span>` : ''}
        ${s ? `<span class="hi">P(reaches destination) ${pct(s.probability, 2)}</span>` : '<span class="dim">Run a scenario for reliability</span>'}`, evt);
    }
  };
  svg.onmouseleave = () => { tip.hide(); svg.querySelectorAll('.hot').forEach((e) => e.classList.remove('hot')); };

  svg.onwheel = (evt) => {
    evt.preventDefault();
    const rect = svg.getBoundingClientRect();
    const p = projected;
    const mx = (evt.clientX - rect.left) * (p.W / rect.width);
    const my = (evt.clientY - rect.top) * (p.H / rect.height);
    const factor = evt.deltaY < 0 ? 1.15 : 1 / 1.15;
    const k = Math.max(0.6, Math.min(9, view.k * factor));
    const ratio = k / view.k;
    view = { k, x: mx - ratio * (mx - view.x), y: my - ratio * (my - view.y) };
    const gEl = $('map-zoom');
    if (gEl) gEl.setAttribute('transform', `translate(${view.x},${view.y}) scale(${view.k})`);
  };

  let drag = null;
  svg.onmousedown = (evt) => { drag = { x: evt.clientX, y: evt.clientY, ox: view.x, oy: view.y }; svg.classList.add('grabbing'); };
  addEventListener('mouseup', () => { drag = null; svg.classList.remove('grabbing'); });
  addEventListener('mousemove', (evt) => {
    if (!drag) return;
    const rect = svg.getBoundingClientRect();
    const sx = projected.W / rect.width, sy = projected.H / rect.height;
    view.x = drag.ox + (evt.clientX - drag.x) * sx;
    view.y = drag.oy + (evt.clientY - drag.y) * sy;
    const gEl = $('map-zoom');
    if (gEl) gEl.setAttribute('transform', `translate(${view.x},${view.y}) scale(${view.k})`);
  });
}

function drawMapLegend(result) {
  const el = $('map-legend');
  if (!result) {
    el.innerHTML = '<span class="legend-title">Run a scenario to colour the map</span>'
      + '<span class="key"><i class="dia"></i>Destination</span>';
    return;
  }
  const dest = '<span class="key"><i class="dia"></i>Destination</span>';
  const keys = (ramp, labels) => ramp.map((c, i) => `<span class="key"><i style="background:${c}"></i>${labels[i]}</span>`).join('');
  if (mapColourMode === 'service') {
    const vals = result.origin_service.map((o) => o.probability);
    el.innerHTML = `<span class="legend-title">P(node reaches a destination) · observed ${pct(Math.min(...vals))}–${pct(Math.max(...vals))}</span>`
      + keys(SERVICE_RAMP, ['≥ 90%', '75–90%', '50–75%', '&lt; 50%']) + dest;
    return;
  }
  const vals = result.components.map((c) => c.p_unusable);
  el.innerHTML = `<span class="legend-title">P(bridge unusable) · observed ${pct(Math.min(...vals))}–${pct(Math.max(...vals))}</span>`
    + keys(RISK_RAMP, ['&lt; 5%', '5–15%', '15–35%', '≥ 35%']) + dest;
}

/* ══ 3. COMPONENT RISK DISTRIBUTION ════════════════════════════════════════ */
/* Round tick values: steps of 1, 2 or 5 x 10^k, covering 0..max with about
 * `target` intervals, so the axis reads in clean numbers (0 / 5 / 10 ...). */
function niceTicks(max, target = 4, integer = false) {
  if (!(max > 0)) return [0];
  const raw = max / target;
  const mag = 10 ** Math.floor(Math.log10(raw));
  let step = [1, 2, 5, 10].map((f) => f * mag).find((s) => s >= raw);
  if (integer) step = Math.max(1, Math.ceil(step));
  const ticks = [];
  for (let v = 0; v <= max + step * 1e-9; v += step) ticks.push(+v.toPrecision(12));
  if (ticks.at(-1) < max) ticks.push(+(ticks.at(-1) + step).toPrecision(12));
  return ticks;
}

function drawRiskChart(result) {
  const svg = $('risk-chart');
  if (!result) {
    svg.innerHTML = '';
    svg.setAttribute('viewBox', '0 0 420 180');
    $('chart-sub').textContent = 'How the per-bridge risk is spread across the network.';
    return;
  }
  const values = result.components.map((c) => c.p_unusable);
  // Left margin holds the count ticks and the rotated axis title.
  const W = 420, H = 180, m = { t: 12, r: 12, b: 36, l: 46 };
  const nBins = 12;
  const peak = Math.max(0.0001, Math.max(...values));
  // The x-axis ends on a round tick, and the bins span exactly that range.
  const xTicks = niceTicks(peak, 4);
  const hi = xTicks.at(-1);
  const edges = Array.from({ length: nBins + 1 }, (_, i) => (i / nBins) * hi);
  const bins = new Array(nBins).fill(0);
  for (const v of values) bins[Math.min(nBins - 1, Math.floor((v / hi) * nBins))]++;
  const yTicks = niceTicks(Math.max(...bins), 4, true);
  const maxCount = yTicks.at(-1);
  const plotW = W - m.l - m.r;
  const bw = plotW / nBins;
  const x = (p) => m.l + (p / hi) * plotW;
  const y = (c) => H - m.b - (c / maxCount) * (H - m.t - m.b);
  // Enough decimals that neighbouring ticks never print the same label.
  const xStep = xTicks[1] - xTicks[0];
  const dp = Math.max(0, Math.ceil(-Math.log10(xStep * 100) + 1e-9));

  let g = '';
  for (const c of yTicks) {
    g += `<line class="grid" x1="${m.l}" y1="${y(c)}" x2="${W - m.r}" y2="${y(c)}"></line>
          <text class="tick" x="${m.l - 6}" y="${y(c) + 3.5}" text-anchor="end">${c}</text>`;
  }
  bins.forEach((c, i) => {
    if (!c) return;
    const mid = (edges[i] + edges[i + 1]) / 2;
    const h = H - m.b - y(c);
    // 2px surface gap between adjacent bars; 4px rounded top anchored to baseline.
    g += `<rect class="bar-mark" x="${m.l + i * bw + 1}" y="${y(c)}" width="${bw - 2}" height="${h}" rx="3"
            fill="${riskColour(mid)}" data-bin="${i}"
            ><title>${c} bridge${c === 1 ? '' : 's'} at ${pct(edges[i], 1)}–${pct(edges[i + 1], 1)}</title></rect>`;
  });
  g += `<line class="axis" x1="${m.l}" y1="${H - m.b}" x2="${W - m.r}" y2="${H - m.b}"></line>`;
  xTicks.forEach((p, i) => {
    const anchor = i === 0 ? 'start' : i === xTicks.length - 1 ? 'end' : 'middle';
    g += `<line class="axis" x1="${x(p)}" y1="${H - m.b}" x2="${x(p)}" y2="${H - m.b + 3}"></line>
          <text class="tick" x="${x(p)}" y="${H - m.b + 13}" text-anchor="${anchor}">${pct(p, dp)}</text>`;
  });
  g += `<text class="axis-title" x="${m.l + plotW / 2}" y="${H - 6}" text-anchor="middle">P(bridge unusable)</text>`;
  g += `<text class="axis-title" transform="translate(12 ${m.t + (H - m.t - m.b) / 2}) rotate(-90)"
          text-anchor="middle">Number of bridges</text>`;

  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.innerHTML = g;

  const over = values.filter((v) => v >= 0.35).length;
  $('chart-sub').textContent = `${values.length} bridges. Highest ${pct(peak, 1)}; ${over} at or above 35%.`;
}

/* ══ query ════════════════════════════════════════════════════════════════ */
function queryPayload() {
  const model = $('hazard-model').value;
  const destinations = [...$('destination-list').querySelectorAll('input:checked')].map((x) => x.value);
  const origin = $('origin-select').value;
  const hazard = model === 'shakemap'
    ? { model, event_id: $('shakemap-id').value.trim() || $('shakemap-event').value }
    : model === 'uniform_sa1'
    ? { model, sa1_g: Number($('sa1').value) }
    : { model, magnitude: Number($('magnitude').value), epicentre_x: Number($('epicentre-x').value), epicentre_y: Number($('epicentre-y').value) };
  return {
    package_id: current.network.case_id,
    query_type: 'population_accessibility',
    origins: origin === '__all__' ? current.network.default_origins : [origin],
    destinations,
    samples: Number($('samples').value),
    seed: 20261004,
    hazard,
  };
}

async function runQuery() {
  const button = $('run');
  const payload = queryPayload();
  if (!payload.destinations.length) {
    $('status').className = 'status error';
    $('status').textContent = 'Select at least one destination.';
    return;
  }
  button.disabled = true;
  button.querySelector('span').textContent = 'Simulating…';
  $('status').className = 'status';
  $('status').textContent = 'Sampling bridge states and propagating them through the network…';
  try {
    const response = await fetch('./api/query', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Query failed');
    lastResult = result;
    showResult(result);
  } catch (error) {
    $('status').className = 'status error';
    $('status').textContent = error.message;
  } finally {
    button.disabled = false;
    button.querySelector('span').textContent = 'Run forward inference';
  }
}

function showResult(result) {
  const weakest = result.origin_service[0];
  const riskiest = result.components[0];
  $('min-reliability').textContent = pct(weakest.probability);
  $('min-origin').textContent = nodeLabel(weakest.origin);
  $('population').textContent = result.population_accessibility ? pct(result.population_accessibility.expected_fraction) : 'n/a';
  $('population-note').textContent = result.population_accessibility
    ? `of ${Math.round(result.population_accessibility.population_in_scope).toLocaleString()} in scope`
    : 'Expected share';
  $('max-failure').textContent = pct(riskiest.p_unusable);
  $('max-edge').textContent = `${riskiest.edge_id} · ${riskiest.from} → ${riskiest.to}`;
  $('simulation-tag').textContent = `${result.samples.toLocaleString()} samples`;

  $('origin-table').innerHTML = result.origin_service.map((row) => `
    <tr data-origin="${esc(row.origin)}">
      <td>${esc(nodeLabel(row.origin))}</td>
      <td class="num"><span class="bar"><i style="width:${(100 * row.probability).toFixed(1)}%;background:${serviceColour(row.probability)}"></i></span>${pct(row.probability)}</td>
      <td class="num dim">${pct(row.ci95[0])}–${pct(row.ci95[1])}</td>
    </tr>`).join('');

  // Hovering a row highlights that origin on the map - the table and the map are
  // two views of one result, so they should respond together.
  $('origin-table').querySelectorAll('tr').forEach((tr) => {
    tr.addEventListener('mouseenter', () => {
      const el = $('network-svg').querySelector(`[data-node="${CSS.escape(tr.dataset.origin)}"]`);
      if (el) el.classList.add('hot');
    });
    tr.addEventListener('mouseleave', () => {
      $('network-svg').querySelectorAll('.hot').forEach((e) => e.classList.remove('hot'));
    });
  });

  const warn = $('warnings');
  if (result.warnings?.length) {
    warn.hidden = false;
    warn.innerHTML = '<p class="eyebrow">CAVEATS FROM THIS RUN</p><ul>'
      + result.warnings.map((w) => `<li>${esc(w)}</li>`).join('') + '</ul>';
  } else { warn.hidden = true; }

  $('status').className = 'status';
  $('status').textContent = `Forward inference complete for ${result.destinations.join(' or ')} · ${result.samples.toLocaleString()} samples · seed ${result.seed}.`;
  drawNetwork(result);
  drawRiskChart(result);
  if (bnView === 'layers') drawBnLayers();   // the layers carry the result too
}

function resetMetrics() {
  ['min-reliability', 'population', 'max-failure'].forEach((id) => { $(id).textContent = '—'; });
  $('min-origin').textContent = 'Run a scenario';
  $('population-note').textContent = 'Expected share';
  $('max-edge').textContent = 'Conditional on scenario';
  $('origin-table').innerHTML = '';
  $('warnings').hidden = true;
}

/* ══ 4. ASK IN WORDS ═══════════════════════════════════════════════════════
 * The panel is additive: with no model backend available the endpoint reports
 * unavailable and the card stays hidden, leaving the pilot exactly as it was.
 *
 * The model never supplies a number here. It picks the scenario, the engine runs,
 * and the server returns both the answer and the full result - which is why the
 * reply can redraw the map: the page is showing the same run the model described.
 *
 * The reply is a stream, so each step shows as it happens: the probabilistic
 * query first, then the values the engine computed for it, then the answer. */
// `history` is what the agent is sent back (text only); `transcript` is the full
// record of each turn -- queries and engine values included -- for the download.
const chat = { history: [], transcript: [], busy: false };

const EXAMPLES = {
  'ema-highway': [
    'What happens to airport access in a magnitude 7 west of the network?',
    'Compare magnitude 6 and magnitude 8 at the default epicentre.',
    'Which bridges are most likely to close in the default scenario?',
  ],
  'hebei-transport': [
    'How reliable is access to Handan at 0.7 g?',
    'What if the shaking were only 0.3 g instead?',
    'Which arcs are riskiest in the default scenario?',
  ],
};

async function initChat() {
  let status;
  try {
    status = await (await fetch('./api/chat-status')).json();
  } catch { return; }
  if (!status.available) return;      // not configured: leave the pilot untouched
  $('chat-card').hidden = false;
  $('chat-model').textContent = status.model;
  $('chat-form').addEventListener('submit', (evt) => { evt.preventDefault(); askQuestion(); });
  $('chat-download').addEventListener('click', downloadConversation);
  renderExamples();
}

/* The agent has no way to write files, so saving a conversation happens here,
 * in the browser, from what the page already holds: each question, every query
 * as stated with the engine's values, and the answer. */
function conversationText() {
  const n = current.network;
  const lines = [
    `# Forward BN pilot - conversation`,
    '',
    `- Package: ${n.title} (${n.case_id})`,
    `- Model: ${$('chat-model').textContent}`,
    `- Saved: ${new Date().toISOString()}`,
    '',
  ];
  chat.transcript.forEach((turn, i) => {
    lines.push(`## ${i + 1}. ${turn.question}`, '');
    turn.queries.forEach((q, j) => {
      lines.push(`**Probabilistic query ${j + 1}** (${q.state})`, '');
      q.targets.forEach((t, k) => {
        const v = q.values?.[k];
        lines.push(`- \`${t.expr}\`${v ? ` = ${pct(v.value)} (${v.note})` : ''} - ${[t.over && `for ${t.over}`, t.meaning, t.how].filter(Boolean).join(' · ')}`);
      });
      if (q.reason) lines.push(`- Not computed: ${q.reason}`);
      lines.push('');
    });
    lines.push(turn.answer ? `**Answer:** ${turn.answer}` : `**No answer:** ${turn.error || 'interrupted'}`, '');
  });
  return lines.join('\n');
}

function downloadConversation() {
  if (!chat.transcript.length) return;
  const stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, '');
  const blob = new Blob([conversationText()], { type: 'text/markdown;charset=utf-8' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `chat-${current.network.case_id}-${stamp}.md`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

function renderExamples() {
  const list = EXAMPLES[current?.network?.case_id] || [];
  $('chat-examples').innerHTML = list
    .map((q) => `<button type="button" class="chip">${esc(q)}</button>`).join('');
  $('chat-examples').querySelectorAll('.chip').forEach((b) => b.addEventListener('click', () => {
    $('chat-input').value = b.textContent;
    askQuestion();
  }));
}

function chatBubble(role, html, cls = '') {
  const el = document.createElement('div');
  el.className = `bubble ${role} ${cls}`.trim();
  el.innerHTML = html;
  $('chat-log').append(el);
  $('chat-log').scrollTop = $('chat-log').scrollHeight;
  return el;
}

/* One probabilistic query, as the server stated it before running the engine.
 * The value slots are left empty on purpose: they are filled by the `computed`
 * event, so what the reader sees first is the question put to the model. */
function queryBlock(n, statement) {
  const el = document.createElement('div');
  el.className = 'pq is-running';
  el.innerHTML = `<p class="pq-head">Probabilistic query ${n}<span class="pq-state">calculating…</span></p>`
    + '<ul>' + statement.targets.map((t) => `<li>
        <code>${esc(t.expr)}</code><span class="pq-value"></span>
        <span class="pq-meaning">${esc([t.over && `for ${t.over}`, t.meaning, t.how].filter(Boolean).join(' · '))}</span>
      </li>`).join('') + '</ul>';
  return el;
}

/* The reply arrives as one JSON object per line, in the order things happen. */
async function readEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const lines = buffer.split('\n');
    buffer = done ? '' : lines.pop();
    for (const line of lines) if (line.trim()) onEvent(JSON.parse(line));
    if (done) return;
  }
}

async function askQuestion() {
  if (chat.busy) return;
  const input = $('chat-input');
  const question = input.value.trim();
  if (!question) return;

  chat.busy = true;
  input.value = '';
  $('chat-send').disabled = true;
  chatBubble('user', esc(question));
  const pending = chatBubble('assistant',
    '<div class="steps"></div><div class="answer" hidden></div><span class="thinking"></span>');
  const steps = pending.querySelector('.steps');
  // Elapsed seconds tick beside each stage, so a slow model reads as working
  // rather than stuck: a hosted model can take a while, a local one longer.
  const started = Date.now();
  let stageText = '';
  const showStage = () => {
    const el = pending.querySelector('.thinking');
    if (el) el.textContent = `${stageText} ${Math.round((Date.now() - started) / 1000)} s`;
  };
  const ticker = setInterval(showStage, 1000);
  const stage = (text) => {
    stageText = text;
    showStage();
    $('chat-log').scrollTop = $('chat-log').scrollHeight;
  };
  stage('Translating the question into a probabilistic query…');

  let block = null;       // the query currently being calculated
  let answered = false;
  const turn = { question, queries: [], answer: null, error: null };
  chat.transcript.push(turn);

  // The bubble fills top to bottom in the order the work is done: the query,
  // then its values, then the prose. The query comes first so the translation
  // from question to parameters is read before the answer, not after it.
  const onEvent = (evt) => {
    if (evt.type === 'query') {
      turn.queries.push({ targets: evt.statement.targets, state: 'not run', values: null, reason: null });
      block = queryBlock(steps.children.length + 1, evt.statement);
      steps.append(block);
      stage(evt.reused
        ? 'Reusing the result of the identical query run earlier…'
        : `Calculating · ${evt.query.samples.toLocaleString()} forward samples through the network…`);
    } else if (evt.type === 'computed') {
      block.className = 'pq';
      // Same query, same seed: the stored result is what a rerun would compute.
      block.querySelector('.pq-state').textContent = evt.reused ? 'reused from an earlier run' : 'computed';
      Object.assign(turn.queries.at(-1), { state: evt.reused ? 'reused from an earlier run' : 'computed', values: evt.values });
      block.querySelectorAll('.pq-value').forEach((slot, i) => {
        const v = evt.values[i];
        if (v) slot.innerHTML = `= ${pct(v.value)} <small>${esc(v.note)}</small>`;
      });
      stage('Writing the answer from these values…');
    } else if (evt.type === 'not_computed') {
      block.className = 'pq is-refused';
      block.querySelector('.pq-state').textContent = 'not computed';
      Object.assign(turn.queries.at(-1), { state: 'not computed', reason: evt.reason });
      block.insertAdjacentHTML('beforeend', `<p class="pq-reason">${esc(evt.reason)}</p>`);
      stage('Writing the answer…');
    } else if (evt.type === 'answer') {
      answered = true;
      turn.answer = evt.answer;
      pending.querySelector('.thinking').remove();
      const answer = pending.querySelector('.answer');
      answer.innerHTML = esc(evt.answer).replace(/\n/g, '<br>');
      answer.hidden = false;
      $('chat-log').scrollTop = $('chat-log').scrollHeight;

      chat.history.push({ role: 'user', content: question });
      chat.history.push({ role: 'assistant', content: evt.answer });

      // Redraw the page from the same run the answer describes.
      if (evt.result) {
        lastResult = evt.result;
        showResult(evt.result);
        $('status').textContent = 'Showing the scenario from the answer above. '
          + $('status').textContent;
      }
    } else if (evt.type === 'error') {
      throw new Error(evt.error);
    }
  };

  try {
    const response = await fetch('./api/ask', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      // The scenario set in the page goes along, so "given the scenario" means this one.
      body: JSON.stringify({ package_id: current.network.case_id, question, history: chat.history,
                             scenario: queryPayload().hazard }),
    });
    // A failure before the stream starts is an ordinary JSON error with a status.
    if (!response.ok) throw new Error((await response.json()).error || 'The request failed.');
    await readEvents(response, onEvent);
    if (!answered) throw new Error('The connection closed before an answer arrived.');
  } catch (error) {
    turn.error = error.message;
    if (steps.children.length) {
      // Keep the queries that were stated and run; only the reply is missing.
      pending.querySelector('.thinking')?.remove();
      steps.querySelectorAll('.is-running').forEach((el) => {
        el.className = 'pq is-refused';
        el.querySelector('.pq-state').textContent = 'interrupted';
      });
      pending.insertAdjacentHTML('beforeend', `<p class="pq-reason">${esc(error.message)}</p>`);
    } else {
      pending.className = 'bubble assistant error';
      pending.textContent = error.message;
    }
  } finally {
    clearInterval(ticker);
    $('chat-download').disabled = false;
    chat.busy = false;
    $('chat-send').disabled = false;
    input.focus();
  }
}

boot()
  .then(initChat)
  .catch((error) => {
    $('status').className = 'status error';
    $('status').textContent = error.message;
  });

