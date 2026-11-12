// The instrument kit: one visual language for every model's live instrument.
// Dark "scope" screen in both themes, the model's accent for its signal, mono for every number.
// Plain DOM + canvas, no framework.

export const css = (name: string, el: Element = document.documentElement) =>
  getComputedStyle(el).getPropertyValue(name).trim();

export function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Record<string, string> = {}, ...kids: (Node | string)[]): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k);
  return e;
}

export const fmt = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '—');

/** Seeded RNG (mulberry32) and Gaussian draws, so "add noise" is repeatable per seed. */
export function rng(seed: number) {
  let a = seed >>> 0;
  const u = () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const n = () => { let x = 0, y = 0; while (x === 0) x = u(); y = u(); return Math.sqrt(-2 * Math.log(x)) * Math.cos(2 * Math.PI * y); };
  return { u, n };
}

// ---- shell ------------------------------------------------------------------------------------
export interface Shell {
  root: HTMLElement;
  picker: HTMLElement;
  screen: HTMLElement;
  controls: HTMLElement;
  readout: HTMLElement;
  setStatus: (html: string) => void;
  setBusy: (busy: boolean) => void;
  stamp: (ms: number, note?: string) => void;
}

export function shell(root: HTMLElement, opts: { title: string; subtitle: string; params?: string }): Shell {
  root.innerHTML = '';
  root.classList.add('inst', 'ready');
  const live = h('span', { class: 'inst-live' }, h('i'), 'live');
  const head = h('div', { class: 'inst-head' }, live, h('span', { class: 'inst-title' }, opts.title), h('span', { class: 'inst-sub' }, opts.subtitle));
  const picker = h('div', { class: 'inst-picker', role: 'group', 'aria-label': 'Real exam example' });
  const screen = h('div', { class: 'inst-screen' });
  const controls = h('div', { class: 'inst-controls' });
  const readout = h('div', { class: 'inst-readout', 'aria-live': 'polite' });
  const status = h('div', { class: 'inst-status' });
  const body = h('div', { class: 'inst-body' }, h('div', { class: 'inst-left' }, screen, controls), readout);
  root.append(head, picker, body, status);
  let timer = 0;
  return {
    root, picker, screen, controls, readout,
    setStatus: (s) => { status.innerHTML = s; },
    setBusy: (b) => {
      clearTimeout(timer);
      if (b) timer = window.setTimeout(() => root.classList.add('busy'), 60);
      else root.classList.remove('busy');
    },
    stamp: (ms, note) => {
      status.innerHTML = `<span class="ok">●</span> network ran in <b>${ms < 10 ? ms.toFixed(1) : Math.round(ms)} ms</b> in your browser`
        + (opts.params ? ` · ${opts.params}` : '') + (note ? ` · ${note}` : '');
    },
  };
}

// ---- controls ---------------------------------------------------------------------------------
export function picker<T>(el: HTMLElement, items: { label: string; sub?: string; value: T }[], onPick: (v: T, i: number) => void, start = 0) {
  el.innerHTML = '';
  const btns = items.map((it, i) => {
    const b = h('button', { type: 'button', class: 'inst-pick' }, h('span', {}, it.label), ...(it.sub ? [h('small', {}, it.sub)] : []));
    b.addEventListener('click', () => { set(i); onPick(it.value, i); });
    el.append(b);
    return b;
  });
  const set = (i: number) => btns.forEach((b, j) => b.setAttribute('aria-pressed', String(i === j)));
  set(start);
  return { set };
}

export interface SliderOpts {
  label: string; min: number; max: number; step: number; value: number;
  fmt?: (v: number) => string; hint?: string;
  onInput: (v: number) => void;
}
export function slider(el: HTMLElement, o: SliderOpts) {
  const id = `s${Math.random().toString(36).slice(2, 8)}`;
  const val = h('output', { for: id, class: 'inst-val' }, (o.fmt ?? String)(o.value));
  const input = h('input', { id, type: 'range', min: String(o.min), max: String(o.max), step: String(o.step), value: String(o.value) }) as HTMLInputElement;
  const row = h('label', { class: 'inst-slider', for: id }, h('span', { class: 'inst-sl-top' }, h('span', {}, o.label), val), input, ...(o.hint ? [h('small', {}, o.hint)] : []));
  const paint = () => input.style.setProperty('--p', `${((+input.value - o.min) / (o.max - o.min)) * 100}%`);
  paint();
  input.addEventListener('input', () => { val.textContent = (o.fmt ?? String)(+input.value); paint(); o.onInput(+input.value); });
  el.append(row);
  return { input, set: (v: number) => { input.value = String(v); val.textContent = (o.fmt ?? String)(v); paint(); } };
}

export function button(el: HTMLElement, label: string, onClick: () => void) {
  const b = h('button', { type: 'button', class: 'inst-btn' }, label);
  b.addEventListener('click', onClick);
  el.append(b);
  return b;
}

/** Run `fn` after input settles; at most one run in flight. */
export function scheduler(fn: () => Promise<void> | void, delay = 40) {
  let t = 0, running = false, again = false;
  const go = async () => {
    if (running) { again = true; return; }
    running = true;
    try { await fn(); } finally { running = false; if (again) { again = false; go(); } }
  };
  return () => { clearTimeout(t); t = window.setTimeout(go, delay); };
}

// ---- readouts ---------------------------------------------------------------------------------
export interface Gauge {
  label: string;
  unit?: string;
  value: number;
  lo?: number; // 68% range
  hi?: number;
  truth?: number | null;
  truthLabel?: string;
  range: [number, number];
  digits?: number;
  log?: boolean;
}

/** A horizontal gauge: the reading with its 68% range, and the truth as a tick. */
export function gauges(el: HTMLElement, gs: Gauge[]) {
  el.querySelectorAll('.inst-gauge').forEach((n) => n.remove());
  for (const g of gs) {
    const [a, b] = g.range;
    const pos = (v: number) => {
      const t = g.log ? (Math.log10(v) - Math.log10(a)) / (Math.log10(b) - Math.log10(a)) : (v - a) / (b - a);
      return `${Math.max(0, Math.min(1, t)) * 100}%`;
    };
    const d = g.digits ?? 2;
    const within = g.truth != null && g.lo != null && g.hi != null && g.truth >= g.lo && g.truth <= g.hi;
    const row = h('div', { class: 'inst-gauge' });
    row.innerHTML = `
      <div class="ig-top"><span class="ig-l">${g.label}</span>
        <span class="ig-v"><b>${fmt(g.value, d)}</b>${g.unit ? `<em>${g.unit}</em>` : ''}${g.lo != null ? `<small>${fmt(g.lo, d)}–${fmt(g.hi!, d)}</small>` : ''}</span></div>
      <div class="ig-track">
        ${g.lo != null ? `<span class="ig-band" style="left:${pos(g.lo)};right:calc(100% - ${pos(g.hi!)})"></span>` : ''}
        <span class="ig-dot" style="left:${pos(g.value)}"></span>
        ${g.truth != null ? `<span class="ig-truth" style="left:${pos(g.truth)}" title="${g.truthLabel ?? 'truth'}"></span>` : ''}
      </div>
      ${g.truth != null ? `<div class="ig-foot"><span>${g.truthLabel ?? 'truth'} <b>${fmt(g.truth, d)}</b></span><span class="${within ? 'in' : 'out'}">${g.lo != null ? (within ? 'inside the 68% bar' : 'outside the bar') : ''}</span></div>` : ''}`;
    el.append(row);
  }
}

/** Class probabilities as bars, with the true class marked. */
export function probs(el: HTMLElement, title: string, items: { label: string; p: number; truth?: boolean }[]) {
  el.querySelectorAll('.inst-probs').forEach((n) => n.remove());
  const box = h('div', { class: 'inst-probs' }, h('div', { class: 'ig-l' }, title));
  const top = Math.max(...items.map((i) => i.p));
  for (const it of items) {
    const r = h('div', { class: `ip-row${it.p === top ? ' top' : ''}${it.truth ? ' truth' : ''}` });
    r.innerHTML = `<span class="ip-l">${it.label}${it.truth ? ' <i>truth</i>' : ''}</span><span class="ip-bar"><span style="width:${(it.p * 100).toFixed(1)}%"></span></span><span class="ip-v">${(it.p * 100).toFixed(0)}%</span>`;
    box.append(r);
  }
  el.append(box);
}

/** Compass for directions: the reading's arc (± half-width) and the truth as a needle. */
export function compass(el: HTMLElement, o: { label: string; deg: number; half: number; truth?: number | null }) {
  el.querySelectorAll('.inst-compass').forEach((n) => n.remove());
  const R = 54, cx = 64, cy = 64;
  const pt = (d: number, r: number) => [cx + r * Math.sin((d * Math.PI) / 180), cy - r * Math.cos((d * Math.PI) / 180)];
  const half = Math.min(o.half, 179);
  const [x0, y0] = pt(o.deg - half, R), [x1, y1] = pt(o.deg + half, R);
  const large = half * 2 > 180 ? 1 : 0;
  const [tx, ty] = pt(o.deg, R - 2);
  const truth = o.truth != null ? pt(o.truth, R + 6) : null;
  const ticks = Array.from({ length: 36 }, (_, i) => { const [a, b] = pt(i * 10, R + (i % 9 === 0 ? 6 : 3)); const [c, d] = pt(i * 10, R); return `<line x1="${a}" y1="${b}" x2="${c}" y2="${d}"/>`; }).join('');
  const box = h('div', { class: 'inst-compass' });
  box.innerHTML = `<svg viewBox="0 0 128 128" aria-label="${o.label}: ${Math.round(o.deg)} degrees">
    <circle cx="${cx}" cy="${cy}" r="${R}" class="ic-ring"/><g class="ic-ticks">${ticks}</g>
    <path class="ic-arc" d="M${cx} ${cy}L${x0} ${y0}A${R} ${R} 0 ${large} 1 ${x1} ${y1}Z"/>
    <line class="ic-needle" x1="${cx}" y1="${cy}" x2="${tx}" y2="${ty}"/>
    ${truth ? `<line class="ic-truth" x1="${cx}" y1="${cy}" x2="${truth[0]}" y2="${truth[1]}"/>` : ''}
    <text x="${cx}" y="10" class="ic-n">N</text></svg>
    <div class="ic-txt"><span class="ig-l">${o.label}</span><b>${Math.round(o.deg)}°</b><small>± ${Math.round(half)}°</small>${o.truth != null ? `<span class="ic-t">truth ${Math.round(o.truth)}°</span>` : ''}</div>`;
  el.append(box);
}

// ---- plotting ---------------------------------------------------------------------------------
export interface Canvas2D { cv: HTMLCanvasElement; ctx: CanvasRenderingContext2D; w: number; h: number; }

/** A crisp, responsive canvas inside `el`; `draw` is called on every resize. */
export function canvas(el: HTMLElement, aspect: number, draw: (c: Canvas2D) => void): { redraw: () => void } {
  const cv = h('canvas', { class: 'inst-cv' });
  el.append(cv);
  const ctx = cv.getContext('2d')!;
  let w = 0, hh = 0;
  const fit = () => {
    const r = el.getBoundingClientRect();
    w = Math.max(200, r.width);
    hh = Math.round(w / aspect);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(hh * dpr);
    cv.style.height = `${hh}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw({ cv, ctx, w, h: hh });
  };
  new ResizeObserver(fit).observe(el);
  fit();
  return { redraw: () => draw({ cv, ctx, w, h: hh }) };
}

export const SCOPE = { bg: '#0d0f0c', grid: 'rgba(235,230,216,0.07)', text: 'rgba(235,230,216,0.55)', faint: 'rgba(235,230,216,0.28)', ink: '#ebe6d8' };

export function gridLines(c: Canvas2D, x0: number, y0: number, w: number, h: number, nx = 10, ny = 4) {
  const { ctx } = c;
  ctx.strokeStyle = SCOPE.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= nx; i++) { const x = Math.round(x0 + (w * i) / nx) + 0.5; ctx.moveTo(x, y0); ctx.lineTo(x, y0 + h); }
  for (let j = 0; j <= ny; j++) { const y = Math.round(y0 + (h * j) / ny) + 0.5; ctx.moveTo(x0, y); ctx.lineTo(x0 + w, y); }
  ctx.stroke();
}

/** Min/max decimated trace (fast and faithful for long records). */
export function trace(c: Canvas2D, data: ArrayLike<number>, x0: number, y0: number, w: number, h: number, color: string, scale: number, alpha = 1, i0 = 0, i1 = data.length) {
  const { ctx } = c;
  const n = i1 - i0;
  const cols = Math.max(1, Math.floor(w));
  const mid = y0 + h / 2;
  ctx.strokeStyle = color;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = 1.1;
  ctx.beginPath();
  for (let px = 0; px < cols; px++) {
    const a = i0 + Math.floor((px / cols) * n), b = Math.max(a + 1, i0 + Math.floor(((px + 1) / cols) * n));
    let lo = Infinity, hi = -Infinity;
    for (let i = a; i < b && i < i1; i++) { const v = data[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
    const yl = mid - Math.max(-1, Math.min(1, hi * scale)) * (h / 2);
    const yh = mid - Math.max(-1, Math.min(1, lo * scale)) * (h / 2);
    ctx.moveTo(x0 + px + 0.5, yl);
    ctx.lineTo(x0 + px + 0.5, Math.max(yh, yl + 0.8));
  }
  ctx.stroke();
  ctx.globalAlpha = 1;
}

export function label(c: Canvas2D, text: string, x: number, y: number, color = SCOPE.text, align: CanvasTextAlign = 'left', size = 10) {
  const { ctx } = c;
  ctx.fillStyle = color;
  ctx.font = `${size}px "Martian Mono Variable", ui-monospace, monospace`;
  ctx.textAlign = align;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x, y);
}

/** Heatmap of a [rows, cols] array into a rectangle, with a perceptual dark->accent ramp. */
export function heatmap(c: Canvas2D, data: ArrayLike<number>, rows: number, cols: number, x0: number, y0: number, w: number, hgt: number, accent: string, lo?: number, hi?: number) {
  let mn = lo ?? Infinity, mx = hi ?? -Infinity;
  if (lo == null || hi == null) for (let i = 0; i < data.length; i++) { if (data[i] < mn) mn = data[i]; if (data[i] > mx) mx = data[i]; }
  const img = c.ctx.createImageData(cols, rows);
  const [ar, ag, ab] = hex(accent);
  for (let r = 0; r < rows; r++) for (let q = 0; q < cols; q++) {
    const t = Math.max(0, Math.min(1, (data[r * cols + q] - mn) / (mx - mn || 1)));
    const k = ((rows - 1 - r) * cols + q) * 4;
    const g = t ** 0.8;
    img.data[k] = 13 + (ar - 13) * g + (255 - ar) * Math.max(0, g - 0.85) * 4;
    img.data[k + 1] = 15 + (ag - 15) * g + (255 - ag) * Math.max(0, g - 0.85) * 4;
    img.data[k + 2] = 12 + (ab - 12) * g + (255 - ab) * Math.max(0, g - 0.85) * 4;
    img.data[k + 3] = 255;
  }
  const off = document.createElement('canvas');
  off.width = cols; off.height = rows;
  off.getContext('2d')!.putImageData(img, 0, 0);
  c.ctx.imageSmoothingEnabled = true;
  c.ctx.drawImage(off, x0, y0, w, hgt);
}

export function hex(c: string): [number, number, number] {
  const m = c.replace('#', '');
  return [parseInt(m.slice(0, 2), 16), parseInt(m.slice(2, 4), 16), parseInt(m.slice(4, 6), 16)];
}
