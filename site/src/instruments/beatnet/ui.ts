// BeatNet live instrument: four real MIT-BIH exam minutes, every beat read by the real network in the browser,
// the cardiologists' labels underneath. Add muscle noise or baseline wander to the raw signal and the whole front end
// (band-pass, windows, local QRS scale, RR features) and the network rerun. NOT A MEDICAL DEVICE.
import type { InstrumentContext } from '../types.ts';
import { fetchPack } from '../../lib/nn/pack.ts';
import { ecgOf, perturb, filt, features, forwardFast, readout, shownBeats, FS0, type BMeta, type BExample, type BeatFeatures, type BeatReading } from './net.ts';
import { shell, picker, slider, button, scheduler, gauges, probs, canvas, label, h, fmt, SCOPE, type Canvas2D } from '../kit.ts';

const CLS = ['N', 'S', 'V', 'F'];
const CNAME: Record<string, string> = { N: 'normal', S: 'supraventricular', V: 'ventricular', F: 'fusion', Q: 'unclassifiable' };
const COL: Record<string, string> = { N: 'rgba(235,230,216,0.6)', S: '#f2c14e', V: '#4fc3f7', F: '#b98cff', Q: 'rgba(235,230,216,0.3)' };
const SHORT: Record<string, string> = { N: 'normal', S: 'supraventr.', V: 'ventricular', F: 'fusion' };
const MISS = '#ff8a7a';
const FS = 180;

const STYLE = `
.bn-sel{display:flex;flex-direction:column;gap:10px}
.bn-sel-h{display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap}
.bn-sel-h b{font-family:var(--mono);font-variation-settings:'wdth' 87.5;font-weight:400;font-size:13px;color:var(--scr-ink)}
.bn-verdict{font-family:var(--mono);font-variation-settings:'wdth' 87.5;font-size:10.5px;padding:2px 6px;border-radius:3px;border:1px solid var(--scr-rule);color:var(--scr-2ink)}
.bn-verdict.ok{color:#7fd8a6;border-color:rgb(127 216 166 / .35)}
.bn-verdict.no{color:${MISS};border-color:rgb(255 138 122 / .4)}
.bn-win{border:1px solid var(--scr-rule);border-radius:3px;background:rgb(0 0 0 / .25)}
.bn-kv{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;margin:0;font-size:11.5px;color:var(--scr-3ink)}
.bn-kv dd{margin:0;text-align:right;font-family:var(--mono);font-variation-settings:'wdth' 87.5;color:var(--scr-2ink)}
.bn-sum{display:flex;flex-direction:column;gap:6px;border-top:1px solid var(--scr-rule);padding-top:10px}
.bn-tab{width:100%;border-collapse:collapse;font-family:var(--mono);font-variation-settings:'wdth' 87.5;font-size:11px;color:var(--scr-2ink)}
.bn-tab th{font-weight:400;font-size:9.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--scr-3ink);text-align:right;padding:2px 4px}
.bn-tab th:first-child,.bn-tab td:first-child{text-align:left}
.bn-tab td{text-align:right;padding:3px 4px;border-top:1px solid var(--scr-rule);white-space:nowrap}
.bn-tab td small{font-size:9.5px;color:var(--scr-3ink)}
.bn-tab td i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px;vertical-align:0}
.bn-note{font-size:11.5px;line-height:1.45;color:var(--scr-3ink)}
.bn-note b{font-weight:400;color:var(--scr-ink)}
.bn-acts{display:flex;gap:8px;flex-wrap:wrap;align-items:end}
`;

export async function mount(ctx: InstrumentContext) {
  const [W, X] = await Promise.all([fetchPack<BMeta>(`${ctx.base}weights`), fetchPack<BExample[]>(`${ctx.base}examples`)]);
  const ex = X.meta;
  const ui = shell(ctx.root, {
    title: 'BeatNet',
    subtitle: 'every heartbeat of a real ECG minute, classified live · research and teaching only — not a medical device',
    params: '171 k parameters · matches Python to 1e-5',
  });
  ctx.root.append(h('style', {}, STYLE));
  const narrow = ctx.root.getBoundingClientRect().width < 640;

  // ---- per-record data ----------------------------------------------------------------------
  const recs = ex.map((e, i) => {
    const pos = X.t[`beats${i}`].data, keep = shownBeats(pos, e.show);
    const ecg = ecgOf(X, i);
    let m = 0;
    for (let k = 0; k < ecg.length; k++) m += ecg[k];
    const rr = keep.slice(1).map((k) => (pos[k] - pos[k - 1]) / FS0).sort((a, b) => a - b);
    return { e, pos, keep, ecg, offset: m / ecg.length, hr: 60 / rr[rr.length >> 1] };
  });

  const st = { i: 3, noise: 0, wander: 0, seed: 1, span: narrow ? 6 : 10, t0: 0, sel: 0 };
  let cur: { x: Float64Array; y: Float64Array; F: BeatFeatures[]; R: (BeatReading | null)[] } | null = null;
  let gen = 0;

  picker(ui.picker, recs.map((r) => ({
    label: `Record ${r.e.record}`,
    sub: `${r.e.why} · ${r.keep.length} beats in 60 s`,
    value: 0,
  })), (_, i) => { st.i = i; pickStart(); syncView(); go(); }, st.i);

  const R = () => recs[st.i];
  const tOf = (k: number) => (R().pos[k] - R().e.show[0]) / FS0; // seconds into the exam minute
  const refOf = (j: number) => R().e.ref[R().keep[j]];
  const callOf = (j: number) => (cur?.R[j] ? CLS[cur.R[j]!.cls] : null);

  // ---- screen -------------------------------------------------------------------------------
  let geo = { x0: 0, w: 1, ovTop: 0, ovH: 0, mainTop: 0, mainH: 0 };
  const plot = canvas(ui.screen, ctx.compact ? 1.7 : narrow ? 1.2 : 1.95, (c) => draw(c));
  const cv = ui.screen.querySelector('canvas') as HTMLCanvasElement;
  cv.tabIndex = 0;
  cv.setAttribute('aria-label', 'ECG strip; click a beat to inspect it, click the overview to move; arrow keys step through beats');
  cv.style.cursor = 'pointer';

  // ---- controls -----------------------------------------------------------------------------
  slider(ui.controls, {
    label: 'Muscle noise', min: 0, max: 0.4, step: 0.01, value: 0,
    fmt: (v) => (v === 0 ? 'none' : `σ ${v.toFixed(2)} mV`), hint: 'white noise like muscle tremor, added to the raw signal',
    onInput: (v) => { st.noise = v; go(); },
  });
  slider(ui.controls, {
    label: 'Baseline wander', min: 0, max: 3, step: 0.1, value: 0,
    fmt: (v) => (v === 0 ? 'none' : `${v.toFixed(1)} mV`), hint: 'slow breathing-like drift (0.07 + 0.21 Hz)',
    onInput: (v) => { st.wander = v; go(); },
  });
  let spanS: ReturnType<typeof slider> | null = null, posS: ReturnType<typeof slider> | null = null;
  if (!ctx.compact) {
    spanS = slider(ui.controls, {
      label: 'Zoom', min: 4, max: 60, step: 1, value: st.span, fmt: (v) => `${v} s shown`, hint: 'of the exam minute',
      onInput: (v) => { const c = st.t0 + st.span / 2; st.span = v; st.t0 = c - v / 2; syncView(); plot.redraw(); },
    });
    posS = slider(ui.controls, {
      label: 'Scroll', min: 0, max: 56, step: 0.5, value: 0, fmt: (v) => `${v.toFixed(0)}–${Math.min(60, v + st.span).toFixed(0)} s`, hint: 'or click the overview strip',
      onInput: (v) => { st.t0 = v; syncView(); plot.redraw(); },
    });
  }
  const acts = h('div', { class: 'bn-acts' });
  ui.controls.append(acts);
  button(acts, 'Next disagreement →', () => jumpMismatch());
  if (!ctx.compact) button(acts, 'New noise draw', () => { st.seed++; go(); });

  // ---- readout skeleton ---------------------------------------------------------------------
  const sel = h('div', { class: 'bn-sel' });
  const selHead = h('div', { class: 'bn-sel-h' });
  const selProbs = h('div', {});
  const selGauge = h('div', {});
  const winBox = h('div', { class: 'bn-win' });
  const kv = h('dl', { class: 'bn-kv' });
  sel.append(selHead, selProbs, selGauge);
  if (!ctx.compact) sel.append(h('div', { class: 'ig-l' }, 'What the network sees · 2.5 s, scaled by the local QRS size'), winBox);
  sel.append(kv);
  const sum = h('div', { class: 'bn-sum' });
  ui.readout.append(sel, sum);
  const win = canvas(winBox, 3.2, (c) => drawWindow(c));

  // ---- helpers ------------------------------------------------------------------------------

  function syncView() {
    st.span = Math.min(60, st.span);
    st.t0 = Math.max(0, Math.min(60 - st.span, st.t0));
    if (posS) { posS.input.max = String(Math.max(0.5, 60 - st.span)); posS.set(Math.round(st.t0 * 2) / 2); }
    if (spanS) spanS.set(st.span);
  }
  function center(j: number, margin: number) {
    const t = tOf(R().keep[j]), m = Math.max(0.4, margin * st.span);
    if (t < st.t0 + m || t > st.t0 + st.span - m) { st.t0 = t - st.span / 2; syncView(); }
  }
  function pickStart() {
    const r = R();
    // start on the first beat of the record's rarest reference class (the interesting one), else the middle
    let best = Math.floor(r.keep.length / 2);
    for (const c of ['F', 'S', 'V']) { const j = r.keep.findIndex((k) => r.e.ref[k] === c); if (j >= 0 && r.e.why.startsWith(c === 'F' ? 'fusion' : c)) { best = j; break; } }
    st.sel = best;
    st.t0 = tOf(r.keep[best]) - st.span * 0.35;
    syncView();
  }
  function select(j: number, margin = 0.2) {
    st.sel = Math.max(0, Math.min(R().keep.length - 1, j));
    center(st.sel, margin);
    plot.redraw();
    showBeat();
  }
  function jumpMismatch() {
    if (!cur) return;
    const n = R().keep.length;
    for (let d = 1; d <= n; d++) { const j = (st.sel + d) % n; const c = callOf(j); if (c && c !== refOf(j)) { select(j); return; } }
  }

  // ---- run: front end + network, visible beats first, then the rest in chunks ----------------
  async function run() {
    const my = ++gen;
    ui.setBusy(true);
    await new Promise((r) => requestAnimationFrame(r));
    const r = R();
    let t = performance.now();
    const x = perturb(r.ecg, { noise: st.noise, wander: st.wander, seed: st.seed * 7919 + r.e.record });
    const y = filt(x, W.meta);
    const F = features(y, r.pos, r.keep, W.meta);
    let ms = performance.now() - t;
    const prev = cur && cur.R.length === r.keep.length && cur.F.length === F.length && cur.x.length === x.length ? cur.R : null;
    cur = { x, y, F, R: prev && prevRec === st.i ? prev.slice() : r.keep.map(() => null) };
    prevRec = st.i;
    const vis: number[] = [], rest: number[] = [];
    r.keep.forEach((k, j) => { const tt = tOf(k); (tt >= st.t0 - 1 && tt <= st.t0 + st.span + 1 || j === st.sel ? vis : rest).push(j); });
    const order = [vis, ...chunk(rest, 18)];
    const fresh = new Set<number>();
    for (const part of order) {
      t = performance.now();
      for (const j of part) { cur.R[j] = readout(W.meta, forwardFast(W, F[j].w, F[j].rr)); fresh.add(j); }
      ms += performance.now() - t;
      if (part === vis) { plot.redraw(); showBeat(); }
      await new Promise((res) => setTimeout(res, 0));
      if (my !== gen) return; // input changed: a newer run takes over
    }
    plot.redraw();
    showBeat();
    summary();
    win.redraw();
    ui.stamp(ms, `${r.keep.length} beats · front end + network`);
    ui.setBusy(false);
  }
  let prevRec = -1;
  const sched = scheduler(run, 40);
  const go = () => { gen++; sched(); };
  const chunk = <T,>(a: T[], n: number) => Array.from({ length: Math.ceil(a.length / n) }, (_, i) => a.slice(i * n, i * n + n));

  // ---- drawing ------------------------------------------------------------------------------
  function draw(c: Canvas2D) {
    const { ctx: g, w, h: H } = c;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, w, H);
    const r = R();
    const x0 = 34, x1 = w - 10, iw = x1 - x0;
    const ovTop = 8, ovH = Math.max(30, Math.min(52, H * 0.13));
    const refY = ovTop + ovH + 26, netY = H - 34;
    const mainTop = refY + 12, mainH = netY - 12 - mainTop;
    geo = { x0, w: iw, ovTop, ovH, mainTop, mainH };

    // overview: the whole minute, beats as ticks (reference above, BeatNet below), the view as a box
    const ovX = (t: number) => x0 + (t / 60) * iw;
    if (cur) traceRange(c, cur.y, r.e.show[0] / 2, r.e.show[1] / 2, x0, ovTop + 7, iw, ovH - 14, 'rgba(235,230,216,0.35)', 0.6);
    r.keep.forEach((k, j) => {
      const xx = Math.round(ovX(tOf(k))) + 0.5, ref = r.e.ref[k], call = callOf(j);
      g.fillStyle = COL[ref]; g.fillRect(xx - 0.5, ovTop, 1.5, ref === 'N' ? 4 : 7);
      if (call) { g.fillStyle = call !== ref ? MISS : COL[call]; g.fillRect(xx - 0.5, ovTop + ovH - (call === 'N' && call === ref ? 4 : 7), 1.5, call === 'N' && call === ref ? 4 : 7); }
    });
    g.strokeStyle = ctx.accent; g.lineWidth = 1;
    g.strokeRect(Math.round(ovX(st.t0)) + 0.5, ovTop - 2.5, Math.max(3, (st.span / 60) * iw), ovH + 5);
    label(c, 'ref', x0 - 6, ovTop + 3, SCOPE.faint, 'right', 8.5);
    label(c, 'net', x0 - 6, ovTop + ovH - 3, SCOPE.faint, 'right', 8.5);

    // main strip: ECG paper grid (0.2 s, 0.5 mV)
    const ts = (t: number) => x0 + ((t - st.t0) / st.span) * iw;
    const mvPx = mainH / 5; // 5 mV across the strip
    const mid = mainTop + mainH * 0.55;
    g.save();
    g.beginPath(); g.rect(x0, mainTop, iw, mainH); g.clip();
    g.lineWidth = 1;
    const minor = st.span <= 12 ? 0.2 : st.span <= 30 ? 1 : 2;
    for (let tt = Math.ceil(st.t0 / minor) * minor; tt <= st.t0 + st.span; tt += minor) {
      const major = Math.abs(tt - Math.round(tt)) < 1e-6 && (minor < 1 || Math.round(tt) % 5 === 0);
      g.strokeStyle = major ? 'rgba(235,230,216,0.1)' : SCOPE.grid;
      g.beginPath(); g.moveTo(Math.round(ts(tt)) + 0.5, mainTop); g.lineTo(Math.round(ts(tt)) + 0.5, mainTop + mainH); g.stroke();
    }
    for (let v = -5; v <= 5; v += 0.5) {
      const yy = Math.round(mid - v * mvPx) + 0.5;
      if (yy < mainTop || yy > mainTop + mainH) continue;
      g.strokeStyle = v === 0 ? 'rgba(235,230,216,0.12)' : SCOPE.grid;
      g.beginPath(); g.moveTo(x0, yy); g.lineTo(x1, yy); g.stroke();
    }
    // selected beat and mismatch columns
    r.keep.forEach((k, j) => {
      const tt = tOf(k);
      if (tt < st.t0 - 1 || tt > st.t0 + st.span + 1) return;
      const xx = ts(tt), call = callOf(j), half = Math.min(15, Math.max(6, (0.28 / st.span) * iw));
      if (j === st.sel) { g.fillStyle = 'rgba(235,230,216,0.08)'; g.fillRect(xx - half, mainTop, 2 * half, mainH); }
      if (call && call !== r.e.ref[k]) { g.fillStyle = 'rgba(255,138,122,0.07)'; g.fillRect(xx - half, mainTop, 2 * half, mainH); }
    });
    // the raw input (faint) and what the band-pass hands the network (bright)
    if (cur) {
      const a = r.e.show[0] + st.t0 * FS0, b = r.e.show[0] + (st.t0 + st.span) * FS0;
      line(c, cur.x, FS0, a, b, ts, mid, mvPx, r.offset, 'rgba(235,230,216,0.28)', 1);
      line(c, cur.y, FS, a / 2, b / 2, ts, mid, mvPx, 0, ctx.accent, 1.4);
    }
    g.restore();
    g.strokeStyle = SCOPE.grid; g.strokeRect(x0 + 0.5, mainTop + 0.5, iw - 1, mainH - 1);

    // letters: reference above, BeatNet below
    const fs = st.span <= 15 ? 12 : st.span <= 30 ? 10 : 8;
    label(c, 'ref', x0 - 6, refY, SCOPE.text, 'right', 9);
    label(c, 'net', x0 - 6, netY, SCOPE.text, 'right', 9);
    r.keep.forEach((k, j) => {
      const tt = tOf(k);
      if (tt < st.t0 || tt > st.t0 + st.span) return;
      const xx = ts(tt), ref = r.e.ref[k], call = callOf(j);
      const showN = st.span <= 30;
      if (ref !== 'N' || showN) label(c, ref, xx, refY, COL[ref], 'center', fs);
      if (call) {
        const miss = call !== ref;
        if (call !== 'N' || showN || miss) label(c, call, xx, netY, miss ? MISS : COL[call], 'center', fs);
        if (miss) {
          g.strokeStyle = MISS; g.lineWidth = 1; g.setLineDash([2, 2]);
          const hw = Math.max(5, fs * 0.6);
          g.strokeRect(Math.round(xx - hw) + 0.5, refY - fs * 0.75, Math.round(2 * hw), netY - refY + fs * 1.5);
          g.setLineDash([]);
        }
      } else label(c, '·', xx, netY, SCOPE.faint, 'center', fs);
    });
    // time axis
    const step = st.span <= 8 ? 1 : st.span <= 20 ? 2 : st.span <= 40 ? 5 : 10;
    for (let tt = Math.ceil(st.t0 / step) * step; tt <= st.t0 + st.span + 1e-6; tt += step) label(c, `${tt}s`, ts(tt), H - 10, SCOPE.faint, 'center', 9);
    label(c, '1 mV', x0 + 4, mainTop + 9, SCOPE.faint, 'left', 8.5);
    g.strokeStyle = SCOPE.faint; g.beginPath(); g.moveTo(x0 + 0.5, mainTop + 16); g.lineTo(x0 + 0.5, mainTop + 16 + mvPx); g.stroke();
    label(c, narrow ? 'faint: input · bright: band-passed, what BeatNet reads' : 'faint: the signal as recorded + your noise · bright: after the 0.5–40 Hz band-pass, what BeatNet reads', x1, ovTop + ovH + 12, SCOPE.faint, 'right', 8.5);
  }

  /** Min/max-decimated polyline of samples [a, b) of data (at fs Hz, seconds measured from the exam minute). */
  function line(c: Canvas2D, d: ArrayLike<number>, fs: number, a: number, b: number, ts: (t: number) => number, mid: number, mvPx: number, off: number, color: string, lw: number) {
    const g = c.ctx, r = R(), s0 = r.e.show[0] / FS0;
    a = Math.max(0, Math.floor(a)); b = Math.min(d.length, Math.ceil(b));
    const px0 = ts((a / fs) - s0), px1 = ts((b / fs) - s0), cols = Math.max(1, Math.round(px1 - px0));
    g.strokeStyle = color; g.lineWidth = lw; g.lineJoin = 'round';
    g.beginPath();
    if (b - a <= cols * 2) {
      for (let i = a; i < b; i++) { const X = ts(i / fs - s0), Y = mid - (d[i] - off) * mvPx; i === a ? g.moveTo(X, Y) : g.lineTo(X, Y); }
    } else {
      for (let p = 0; p < cols; p++) {
        const i0 = a + Math.floor((p / cols) * (b - a)), i1 = Math.max(i0 + 1, a + Math.floor(((p + 1) / cols) * (b - a)));
        let lo = Infinity, hi = -Infinity;
        for (let i = i0; i < i1; i++) { const v = d[i] - off; if (v < lo) lo = v; if (v > hi) hi = v; }
        const X = px0 + p + 0.5;
        if (p === 0) g.moveTo(X, mid - hi * mvPx); else g.lineTo(X, mid - hi * mvPx);
        g.lineTo(X, mid - lo * mvPx);
      }
    }
    g.stroke();
  }
  function traceRange(c: Canvas2D, d: ArrayLike<number>, a: number, b: number, x0: number, y0: number, w: number, hh: number, color: string, scale: number) {
    const g = c.ctx, cols = Math.floor(w), n = b - a, m = y0 + hh / 2;
    g.strokeStyle = color; g.lineWidth = 1; g.beginPath();
    for (let p = 0; p < cols; p++) {
      const i0 = a + Math.floor((p / cols) * n), i1 = Math.max(i0 + 1, a + Math.floor(((p + 1) / cols) * n));
      let lo = Infinity, hi = -Infinity;
      for (let i = i0; i < i1; i++) { const v = d[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
      const cl = (v: number) => Math.max(-1, Math.min(1, v * scale));
      g.moveTo(x0 + p + 0.5, m - cl(hi) * hh / 2); g.lineTo(x0 + p + 0.5, m - cl(lo) * hh / 2 + 0.8);
    }
    g.stroke();
  }

  function drawWindow(c: Canvas2D) {
    const { ctx: g, w, h: H } = c;
    g.fillStyle = SCOPE.bg; g.fillRect(0, 0, w, H);
    const f = cur?.F[st.sel];
    const q = 0.08 / 2.5; // ±80 ms, where the QRS size is measured
    g.fillStyle = 'rgba(235,230,216,0.06)'; g.fillRect(w * (0.5 - q), 0, w * 2 * q, H);
    g.strokeStyle = SCOPE.grid; g.beginPath(); g.moveTo(w / 2 + 0.5, 0); g.lineTo(w / 2 + 0.5, H); g.moveTo(0, H * 0.58 + 0.5); g.lineTo(w, H * 0.58 + 0.5); g.stroke();
    if (!f) return;
    const sy = H / 2.4, my = H * 0.58, n = f.w.length;
    g.strokeStyle = ctx.accent; g.lineWidth = 1.3; g.beginPath();
    for (let i = 0; i < n; i++) { const X = (i / (n - 1)) * w, Y = Math.max(1, Math.min(H - 1, my - f.w[i] * sy)); i ? g.lineTo(X, Y) : g.moveTo(X, Y); }
    g.stroke();
    label(c, '−1.25 s', 4, H - 8, SCOPE.faint, 'left', 8.5);
    label(c, '+1.25 s', w - 4, H - 8, SCOPE.faint, 'right', 8.5);
  }

  // ---- readouts -----------------------------------------------------------------------------
  function showBeat() {
    const r = R(), j = st.sel, k = r.keep[j], ref = r.e.ref[k], o = cur?.R[j], f = cur?.F[j];
    const call = o ? CLS[o.cls] : null;
    selHead.innerHTML = `<span class="ig-l">Beat ${j + 1} of ${r.keep.length} · ${fmt(tOf(k), 2)} s</span>`
      + (call ? `<span class="bn-verdict ${call === ref ? 'ok' : 'no'}">${call === ref ? 'agrees with the cardiologists' : `cardiologists: ${ref}`}</span>` : '');
    if (!o || !f) { selProbs.innerHTML = ''; return; }
    probs(selProbs, 'BeatNet’s class probabilities', CLS.map((c, i) => ({ label: `${c} · ${CNAME[c]}`, p: o.p[i], truth: c === ref })));
    gauges(selGauge, [{ label: 'QRS width', unit: 'ms', value: o.qrs, lo: o.qrs - o.qrsSig, hi: o.qrs + o.qrsSig, range: [40, 300], digits: 0 }]);
    const preRR = k > 0 ? (r.pos[k] - r.pos[k - 1]) / FS0 : NaN, postRR = k + 1 < r.pos.length ? (r.pos[k + 1] - r.pos[k]) / FS0 : NaN;
    kv.innerHTML = `<dt>RR before / after</dt><dd>${fmt(preRR * 1000, 0)} / ${fmt(postRR * 1000, 0)} ms</dd>`
      + `<dt>Prematurity</dt><dd>${fmt(Math.exp(f.rr[0]) * 100, 0)}% of the local mean RR</dd>`
      + `<dt>Local QRS size</dt><dd>${fmt(f.scale, 2)} mV peak-to-peak</dd>`
      + (Number.isFinite(o.af) ? `<dt>Atrial fibrillation?</dt><dd>${fmt(o.af * 100, 0)}% <span title="the excerpts carry beat labels only, no rhythm labels">· no rhythm truth</span></dd>` : '');
    win.redraw();
  }

  function summary() {
    if (!cur) return;
    const r = R();
    const cnt = CLS.map(() => ({ ref: 0, call: 0, hit: 0 }));
    let agree = 0, changed = 0, done = 0;
    r.keep.forEach((k, j) => {
      const ref = r.e.ref[k], o = cur!.R[j];
      if (!o) return;
      done++;
      const call = CLS[o.cls], ri = CLS.indexOf(ref);
      if (ri >= 0) cnt[ri].ref++;
      cnt[o.cls].call++;
      if (call === ref) { agree++; cnt[o.cls].hit++; }
      if (call !== r.e.python.label[k]) changed++;
    });
    const rows = CLS.map((c, i) => ({ c, ...cnt[i] })).filter((x) => x.ref || x.call)
      .map((x) => `<tr><td><i style="background:${COL[x.c]}"></i>${x.c} ${SHORT[x.c]}</td><td>${x.ref}</td><td>${x.call}</td><td>${x.ref ? `${x.hit} <small>${Math.round((100 * x.hit) / x.ref)}%</small>` : '—'}</td></tr>`).join('');
    const clean = st.noise === 0 && st.wander === 0;
    sum.innerHTML = `<div class="ig-l">This minute · record ${r.e.record}, ${fmt(r.e.start_s / 60, 1)} min in</div>
      <table class="bn-tab"><thead><tr><th>class</th><th>ref</th><th>BeatNet</th><th>found</th></tr></thead><tbody>${rows}</tbody></table>
      <div class="bn-note"><b>${agree} of ${done}</b> beats agree with the cardiologists · heart rate ${Math.round(r.hr)} bpm.
      ${clean ? `Every call is identical to the Python release on the clean recording${changed ? ` except ${changed}` : ''}.` : `The added noise changed <b>${changed}</b> of ${done} calls from the clean reading.`}</div>`;
  }

  // ---- interaction --------------------------------------------------------------------------
  cv.addEventListener('click', (ev) => {
    const b = cv.getBoundingClientRect(), px = ev.clientX - b.left, py = ev.clientY - b.top;
    if (py < geo.ovTop + geo.ovH + 6) { st.t0 = ((px - geo.x0) / geo.w) * 60 - st.span / 2; syncView(); plot.redraw(); return; }
    const t = st.t0 + ((px - geo.x0) / geo.w) * st.span;
    let best = -1, bd = Infinity;
    R().keep.forEach((k, j) => { const d = Math.abs(tOf(k) - t); if (d < bd) { bd = d; best = j; } });
    if (best >= 0) select(best, 0);
  });
  cv.addEventListener('keydown', (ev) => {
    if (ev.key === 'ArrowRight') { select(st.sel + 1); ev.preventDefault(); }
    if (ev.key === 'ArrowLeft') { select(st.sel - 1); ev.preventDefault(); }
  });

  pickStart();
  await run();
}
