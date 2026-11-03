/* Shared navigation for the schema canvas and the application's BN drawing. */
function diagramNavigation(viewport, { get, set, fit, ignoreDrag = () => false, onClick = () => {}, help = "" }) {
  if (!document.getElementById("diagram-navigation-style")) {
    const style = document.createElement("style");
    style.id = "diagram-navigation-style";
    style.textContent = `
      .diagram-viewport { cursor: grab; touch-action: none; container-type: inline-size; }
      .diagram-viewport.panning { cursor: grabbing; }
      .diagram-viewport:focus-visible { outline: 2px solid var(--hi, var(--accent)); outline-offset: -3px; }
      .diagram-navigation { position: absolute; right: 12px; bottom: 12px; z-index: 4; max-width: calc(100% - 24px);
        background: var(--card, var(--raised)); color: var(--text, var(--ink)); border: 1px solid var(--card-line, var(--border));
        border-radius: 9px; padding: 5px; box-shadow: 0 2px 12px #0005; cursor: default; }
      .diagram-navigation .nav-buttons { display: flex; gap: 3px; align-items: center; }
      .diagram-navigation button, .diagram-navigation summary { min-width: 38px; min-height: 38px; padding: 6px 9px;
        background: var(--chip, var(--surface)); color: inherit; border: 1px solid transparent; border-radius: 5px;
        font: inherit; cursor: pointer; text-align: center; }
      .diagram-navigation button:focus-visible, .diagram-navigation summary:focus-visible { outline: 2px solid var(--hi, var(--accent)); }
      .diagram-navigation button:hover, .diagram-navigation summary:hover { border-color: var(--faint, var(--muted)); }
      .diagram-navigation button:disabled { opacity: .4; cursor: default; }
      .diagram-navigation [data-nav=actual] { min-width: 62px; font-variant-numeric: tabular-nums; }
      .diagram-navigation summary { display: grid; place-items: center; list-style: none; }
      .diagram-navigation summary::-webkit-details-marker { display: none; }
      .diagram-navigation .nav-help { position: absolute; right: 0; bottom: 56px; width: 290px; max-width: calc(100vw - 36px);
        padding: 12px; border-radius: 8px; background: inherit; border: inherit; box-shadow: inherit; font: 12px/1.7 system-ui; }
      .diagram-navigation .nav-hint { padding: 2px 6px 5px; color: var(--muted, var(--ink-2)); font: 11px/1.4 system-ui; }
      .compact-navigation .diagram-navigation .nav-hint { display: none; }
      @container (max-width: 480px) {
        .diagram-navigation { right: 8px; bottom: 8px; padding: 3px; }
        .diagram-navigation .nav-hint { display: none; }
      }
    `;
    document.head.append(style);
  }
  viewport.classList.add("diagram-viewport");
  viewport.tabIndex = 0;
  const controls = document.createElement("div");
  controls.className = "diagram-navigation";
  controls.setAttribute("role", "group");
  controls.setAttribute("aria-label", "Diagram navigation");
  controls.innerHTML = `<div class="nav-hint">Drag to pan · Pinch or Ctrl/⌘ + scroll to zoom</div><div class="nav-buttons">
    <button type="button" data-nav="out" aria-label="Zoom out" title="Zoom out (−)">−</button>
    <button type="button" data-nav="actual" aria-label="Reset zoom to 100 percent" title="Reset zoom to 100% (0)">100%</button>
    <button type="button" data-nav="in" aria-label="Zoom in" title="Zoom in (+)">+</button>
    <button type="button" data-nav="fit" title="Fit the whole diagram (F or Home)">Fit</button>
    <details><summary aria-label="Navigation help" title="Navigation help">?</summary><div class="nav-help">
      Drag the diagram to move around.<br>Scroll to move; Shift + scroll to move sideways.<br>
      Pinch or Ctrl/⌘ + scroll to zoom at the pointer.<br>Double-click to zoom in; Shift + double-click to zoom out.<br>
      Keyboard: + / − to zoom, arrows to move, F to fit, 0 for 100%.${help ? `<br>${help}` : ""}
    </div></details></div>`;
  viewport.append(controls);
  const MIN = 0.02, MAX = 4;
  let fitted = true;
  const update = () => {
    const { k } = get();
    controls.querySelector('[data-nav="actual"]').textContent = `${Math.round(k * 100)}%`;
    controls.querySelector('[data-nav="out"]').disabled = k <= MIN;
    controls.querySelector('[data-nav="in"]').disabled = k >= MAX;
  };
  const change = value => { fitted = false; set(value); update(); };
  const fitAll = () => { fit(); fitted = true; update(); };
  const zoom = (k, x = viewport.clientWidth / 2, y = viewport.clientHeight / 2) => {
    const old = get();
    k = Math.min(MAX, Math.max(MIN, k));
    change({ k, tx: x - (x - old.tx) * k / old.k, ty: y - (y - old.ty) * k / old.k });
  };
  const pan = (dx, dy) => { const v = get(); change({ ...v, tx: v.tx + dx, ty: v.ty + dy }); };
  const local = e => { const r = viewport.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
  controls.addEventListener("click", e => {
    const action = e.target.closest("[data-nav]")?.dataset.nav;
    if (!action) return;
    if (action === "fit") fitAll();
    else zoom(action === "actual" ? 1 : get().k * (action === "in" ? 1.25 : 0.8));
  });
  viewport.addEventListener("wheel", e => {
    if (e.target.closest(".diagram-navigation")) return;
    e.preventDefault();
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? viewport.clientHeight : 1;
    if (e.ctrlKey || e.metaKey) {
      const p = local(e);
      zoom(get().k * Math.exp(-Math.max(-200, Math.min(200, e.deltaY * unit)) * 0.01), p.x, p.y);
    } else if (e.shiftKey && !e.deltaX) pan(-e.deltaY * unit, 0);
    else pan(-e.deltaX * unit, -e.deltaY * unit);
  }, { passive: false });
  viewport.addEventListener("dblclick", e => {
    if (e.target.closest(".diagram-navigation, #legend, #banner")) return;
    e.preventDefault();
    const p = local(e);
    zoom(get().k * (e.shiftKey ? 0.5 : 2), p.x, p.y);
  });
  const pointers = new Map();
  let gesture = null;
  const seed = () => {
    if (!pointers.size) { gesture = null; viewport.classList.remove("panning"); return; }
    const ps = [...pointers.values()];
    const center = ps.length > 1 ? { x: (ps[0].x + ps[1].x) / 2, y: (ps[0].y + ps[1].y) / 2 } : ps[0];
    gesture = { ...get(), center, distance: ps.length > 1 ? Math.hypot(ps[1].x - ps[0].x, ps[1].y - ps[0].y) : 0,
      moved: ps.length > 1, target: ps[0].target, button: ps[0].button };
  };
  viewport.addEventListener("pointerdown", e => {
    if (![0, 1].includes(e.button) || e.target.closest(".diagram-navigation, #legend, #banner") || ignoreDrag(e)) return;
    viewport.focus({ preventScroll: true });
    e.preventDefault();
    pointers.set(e.pointerId, { ...local(e), target: e.target, button: e.button });
    viewport.setPointerCapture(e.pointerId);
    seed();
  });
  viewport.addEventListener("pointermove", e => {
    if (!pointers.has(e.pointerId)) return;
    pointers.set(e.pointerId, { ...pointers.get(e.pointerId), ...local(e) });
    const ps = [...pointers.values()];
    const center = ps.length > 1 ? { x: (ps[0].x + ps[1].x) / 2, y: (ps[0].y + ps[1].y) / 2 } : ps[0];
    const dx = center.x - gesture.center.x, dy = center.y - gesture.center.y;
    if (ps.length > 1 || Math.abs(dx) + Math.abs(dy) > 3) gesture.moved = true;
    if (!gesture.moved) return;
    viewport.classList.add("panning");
    const k = ps.length > 1 && gesture.distance
      ? Math.min(MAX, Math.max(MIN, gesture.k * Math.hypot(ps[1].x - ps[0].x, ps[1].y - ps[0].y) / gesture.distance)) : gesture.k;
    change({ k, tx: center.x - (gesture.center.x - gesture.tx) * k / gesture.k,
      ty: center.y - (gesture.center.y - gesture.ty) * k / gesture.k });
  });
  const end = e => {
    if (!pointers.has(e.pointerId)) return;
    if (e.type === "pointerup" && pointers.size === 1 && !gesture.moved && gesture.button === 0) onClick(gesture.target);
    pointers.delete(e.pointerId);
    if (viewport.hasPointerCapture(e.pointerId)) viewport.releasePointerCapture(e.pointerId);
    seed();
    if (gesture) gesture.moved = true; // Lifting one finger after a pinch is never a click.
  };
  for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) viewport.addEventListener(event, end);
  viewport.addEventListener("keydown", e => {
    if (e.target !== viewport || e.ctrlKey || e.metaKey || e.altKey) return;
    if (["+", "="].includes(e.key)) zoom(get().k * 1.25);
    else if (e.key === "-") zoom(get().k * 0.8);
    else if (e.key === "0") zoom(1);
    else if (["f", "F", "Home"].includes(e.key)) fitAll();
    else if (e.key.startsWith("Arrow")) {
      const d = e.shiftKey ? 160 : 60;
      pan(e.key === "ArrowLeft" ? d : e.key === "ArrowRight" ? -d : 0,
        e.key === "ArrowUp" ? d : e.key === "ArrowDown" ? -d : 0);
    } else return;
    e.preventDefault();
  });
  update();
  let previousSize = null;
  new ResizeObserver(() => {
    const width = viewport.clientWidth, height = viewport.clientHeight;
    if (!width || !height) return;
    const compact = width < 480 || height < 320;
    const controlsChanged = viewport.classList.contains("compact-navigation") !== compact;
    viewport.classList.toggle("compact-navigation", compact);
    if (controlsChanged || previousSize && (width !== previousSize.width || height !== previousSize.height)) {
      if (fitted) fitAll();
      else if (previousSize) {
        const value = get();
        set({ ...value, tx: value.tx + (width - previousSize.width) / 2, ty: value.ty + (height - previousSize.height) / 2 });
        update();
      }
    }
    previousSize = { width, height };
  }).observe(viewport);
  return { update, zoom, fit: fitAll, markFitted: () => { fitted = true; } };
}
