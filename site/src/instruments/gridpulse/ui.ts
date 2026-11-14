// GridPulse live instrument: ten real GB trips from the exam (NESO 1 s frequency, 2020-2025), never used in training.
// The browser runs the network on the window the controls make: how many seconds after the trip it may see (N), whether
// it is told the transmission demand, and how much extra real grid noise is added. The truths stay on screen: NESO's
// outturn inertia, the loss label where GridData has one, and the measured nadir and frequency at 60 s.
import type { InstrumentContext } from '../types.ts';
import { fetchPack } from '../../lib/nn/pack.ts';
import { read, perturb, PRE, POST, LEN, type GMeta, type GExample, type Reading } from './net.ts';
import { shell, picker, slider, button, scheduler, gauges, canvas, gridLines, label, h, SCOPE, type Canvas2D } from '../kit.ts';

const STYLE = `
.gp-kv { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 3px 10px; font-size: 12px; color: var(--scr-2ink); }
.gp-kv b { font-family: var(--mono); font-variation-settings: 'wdth' 87.5; font-weight: 400; font-size: 11.5px; color: var(--scr-ink); text-align: right; white-space: nowrap; }
.gp-note { font-size: 12px; line-height: 1.5; color: var(--scr-3ink); margin: 0; }
.gp-note b { font-weight: 500; color: var(--scr-2ink); }
.gp-btns { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 8px; }
.gp-btns .inst-btn[aria-pressed='true'] { border-color: var(--c); color: var(--c); }
`;

const ROCOF_K = 7.0005; // s: the exam's dev-fitted inertia per MW of demand (7.0 s x demand baseline, 15.1 % MdAPE)
const NMIN = 2, NMAX = 60;
const mhz = (v: number) => `${v > 0 ? '+' : ''}${Math.round(v * 1000)}`;
const big = (v: number) => Math.round(v).toLocaleString('en-GB');

export async function mount(ctx: InstrumentContext) {
  const [W, X] = await Promise.all([fetchPack<GMeta>(`${ctx.base}weights`), fetchPack<GExample[]>(`${ctx.base}examples`)]);
  const ex = X.meta;
  if (!document.getElementById('gp-inst-style')) document.head.append(h('style', { id: 'gp-inst-style' }, STYLE));
  const ui = shell(ctx.root, {
    title: 'GridPulse',
    subtitle: 'MW lost, inertia and the nadir from 1 s grid frequency — a real GB trip, never used in training',
    params: `${(W.meta.n_params / 1e6).toFixed(2)} M parameters · matches Python to 1e-5`,
  });
  const st = { i: 0, N: 10, demand: true, noise: 0 };
  let view: { r: Reading | null; w: Float64Array | null; sweep: { N: number; r: Reading }[] } = { r: null, w: null, sweep: [] };

  picker(ui.picker, ex.map((e, i) => ({
    label: e.t,
    sub: `NESO ${Math.round(e.E_true_GVAs)} GVA s · ${e.loss_label_MW != null ? `label ${big(e.loss_label_MW)} MW` : 'no loss label'}`,
    value: i,
  })), (i) => { st.i = i; syncDemand(); go(); }, 0);

  const narrow = () => ui.screen.getBoundingClientRect().width < 520;
  const plot = canvas(ui.screen, ctx.compact ? 1.9 : narrow() ? 0.95 : 1.5, (c) => draw(c));

  const sN = slider(ui.controls, {
    label: 'Seconds seen after the trip', min: NMIN, max: NMAX, step: 1, value: st.N,
    fmt: (v) => `N = ${v} s`,
    hint: 'the network sees the window up to N s after the trip; the rest is zeroed, as in training',
    onInput: (v) => { st.N = v; go(); },
  });
  const sZ = slider(ui.controls, {
    label: 'Extra grid noise', min: 0, max: 3, step: 0.25, value: 0,
    fmt: (v) => (v === 0 ? 'none' : `×${v.toFixed(2)} quiet NESO slice`),
    hint: 'adds a real quiet 1 s NESO record (Jan 2022, no event) to the whole window',
    onInput: (v) => { st.noise = v; go(); },
  });
  const btns = h('div', { class: 'gp-btns' });
  ui.controls.append(btns);
  const dBtn = button(btns, 'Demand: known', () => { st.demand = !st.demand; syncDemand(); go(); });
  dBtn.title = 'transmission demand of the half hour (NESO generation mix); without it the network reads the trace alone';
  if (!ctx.compact) button(btns, 'As recorded', () => { Object.assign(st, { N: 10, demand: true, noise: 0 }); sN.set(10); sZ.set(0); syncDemand(); go(); });
  function syncDemand() {
    const e = ex[st.i];
    dBtn.textContent = st.demand ? `Demand: ${big(e.demand_MW)} MW` : 'Demand: unknown';
    dBtn.setAttribute('aria-pressed', String(!st.demand));
  }
  syncDemand();

  const gaugeBox = h('div', {}), kvBox = h('div', { class: 'gp-kv' }), note = h('p', { class: 'gp-note' });
  ui.readout.append(gaugeBox, kvBox, note);

  const input = (N: number) => {
    const e = ex[st.i];
    return perturb(X.t[`w${st.i}`].data, e.demand_MW, X.t[`n${st.i}`].data, { N, demand: st.demand, noise: st.noise });
  };

  async function run() {
    ui.setBusy(true);
    await new Promise((r) => requestAnimationFrame(r));
    const t0 = performance.now();
    const p = input(st.N);
    const r = read(W, p.w, p.N, p.d);
    const ms = performance.now() - t0;
    let note = ex[st.i].id;
    const sweep: typeof view.sweep = [];
    if (!ctx.compact) {
      const t1 = performance.now();
      for (let N = NMIN; N <= NMAX; N += 2) { const q = input(N); sweep.push({ N, r: read(W, q.w, q.N, q.d) }); }
      note += ` · readings vs N: ${sweep.length} more runs in ${Math.round(performance.now() - t1)} ms`;
    }
    view = { r, w: p.w, sweep };
    plot.redraw();
    render(r, p.w);
    ui.stamp(ms, note);
    ui.setBusy(false);
  }
  const go = scheduler(run, 30);

  function lowestSoFar(w: ArrayLike<number>, N: number) { let m = Infinity; for (let i = PRE; i <= PRE + N; i++) m = Math.min(m, w[i]); return m; }

  function render(r: Reading, w: Float64Array) {
    const e = ex[st.i];
    gaugeBox.innerHTML = '';
    gauges(gaugeBox, [
      { label: 'MW lost', unit: 'MW', value: r.loss[0], lo: r.loss[1], hi: r.loss[2], truth: e.loss_label_MW, truthLabel: `loss label (${e.tier ?? '—'})`, range: [0, 2600], digits: 0 },
      { label: 'Inertia', unit: 'GVA s', value: r.E[0], lo: r.E[1], hi: r.E[2], truth: e.E_true_GVAs, truthLabel: 'NESO outturn', range: [50, 450], digits: 0 },
      { label: 'Nadir forecast (0-60 s)', unit: 'mHz', value: r.nadir[0] * 1000, lo: (r.nadir[0] - r.nadir[1]) * 1000, hi: (r.nadir[0] + r.nadir[1]) * 1000, truth: e.nadir_Hz * 1000, truthLabel: 'measured', range: [-700, 0], digits: 0 },
      { label: 'Frequency at 60 s', unit: 'mHz', value: r.settle[0] * 1000, lo: (r.settle[0] - r.settle[1]) * 1000, hi: (r.settle[0] + r.settle[1]) * 1000, truth: e.settle_Hz * 1000, truthLabel: 'measured (55-65 s)', range: [-600, 100], digits: 0 },
    ]);
    const ls = lowestSoFar(w, st.N);
    kvBox.innerHTML = [
      ['Lowest value so far (baseline)', `${mhz(ls)} mHz`],
      ['7.0 s × demand (baseline)', `${big((ROCOF_K * e.demand_MW) / 1000)} GVA s`],
      ['Response, MW/Hz (no real truth)', `${big(r.beta[0])} (${big(r.beta[1])}–${big(r.beta[2])})`],
      ['Headroom / FFR (weak, no truth)', `${big(r.head)} / ${big(Math.max(r.ffr, 0))} MW`],
    ].map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join('');
    const changed: string[] = [];
    if (!st.demand) changed.push('demand withheld');
    if (st.noise) changed.push(`extra noise ×${st.noise.toFixed(2)}`);
    note.innerHTML = (changed.length ? `<b>Changed: ${changed.join(', ')}.</b> ` : '')
      + `Real event ${e.id}, ${e.source}. Bands are 1 sigma after the dev widening (MW lost ×1.11, frequency at 60 s ×1.28). `
      + (e.loss_label_MW == null ? 'GridData has no loss label for this event, so MW lost has no truth. ' : '')
      + 'Withhold the demand to see what the trace alone gives: in the exam, inertia from the trace alone was 20.3 % off (median) against 16.4 % with demand, no better than a constant guess.';
  }

  function draw(c: Canvas2D) {
    const { ctx: g, w, h: H } = c;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, w, H);
    const r = view.r, wv = view.w;
    if (!r || !wv) return;
    const e = ex[st.i], raw = X.t[`w${st.i}`].data;
    const sweep = view.sweep.length > 0;
    const padL = 44, padR = 12, top = 24, gap = 30, bot = 20;
    const iw = w - padL - padR, ih = H - top - bot - (sweep ? gap : 0);
    const h1 = sweep ? Math.round(ih * 0.58) : ih, h2 = ih - h1, y2 = top + h1 + gap;
    const small = w < 520;
    // --- the window: frequency deviation (mHz) from the pre-event mean, -20..+60 s
    let lo = Math.min(e.nadir_Hz, r.nadir[0] - r.nadir[1]), hi = 0.03;
    for (let i = 0; i < LEN; i++) { lo = Math.min(lo, wv[i]); hi = Math.max(hi, wv[i]); }
    const span = hi - lo; lo -= span * 0.08; hi += span * 0.06;
    const xt = (t: number) => padL + ((t + PRE) / (PRE + POST)) * iw;
    const yf = (v: number) => top + ((hi - v) / (hi - lo)) * h1;
    gridLines(c, padL, top, iw, h1, 8, 4);
    for (let j = 0; j <= 4; j++) { const v = hi - ((hi - lo) * j) / 4; label(c, mhz(v), padL - 4, yf(v), SCOPE.faint, 'right', 9); }
    for (let t = -20; t <= 60; t += 10) label(c, `${t}s`, xt(t), top + h1 + 9, SCOPE.faint, 'center', 9);
    // zero line
    g.strokeStyle = SCOPE.faint; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(padL, yf(0)); g.lineTo(padL + iw, yf(0)); g.stroke(); g.setLineDash([]);
    // after N: what the network does not see (the recorded trace, faint)
    const xn = xt(st.N);
    g.fillStyle = 'rgba(235,230,216,0.04)'; g.fillRect(xn, top, padL + iw - xn, h1);
    g.strokeStyle = SCOPE.text; g.lineWidth = 1.1; g.globalAlpha = 0.45; g.beginPath();
    for (let i = 0; i < LEN; i++) { const x = xt(i - PRE), y = yf(raw[i]); i ? g.lineTo(x, y) : g.moveTo(x, y); }
    g.stroke(); g.globalAlpha = 1;
    // forecast bands: nadir (from N to 60 s) and frequency at 60 s (55-60 s)
    const band = (x0: number, x1: number, m: number, s: number) => {
      g.globalAlpha = 0.2; g.fillStyle = ctx.accent; g.fillRect(x0, yf(m + s), x1 - x0, yf(m - s) - yf(m + s)); g.globalAlpha = 1;
      g.strokeStyle = ctx.accent; g.lineWidth = 1.4; g.beginPath(); g.moveTo(x0, yf(m)); g.lineTo(x1, yf(m)); g.stroke();
    };
    band(xn, xt(60), r.nadir[0], r.nadir[1]);
    band(xt(55), xt(60), r.settle[0], r.settle[1]);
    // what the network sees: the input window up to N
    g.strokeStyle = ctx.accent; g.lineWidth = 1.7; g.beginPath();
    for (let i = 0; i <= PRE + st.N; i++) { const x = xt(i - PRE), y = yf(wv[i]); i ? g.lineTo(x, y) : g.moveTo(x, y); }
    g.stroke();
    g.strokeStyle = ctx.accent; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(xn, top); g.lineTo(xn, top + h1); g.stroke(); g.setLineDash([]);
    label(c, `N = ${st.N} s`, xn + (xn > padL + iw * 0.8 ? -4 : 4), top + 8, ctx.accent, xn > padL + iw * 0.8 ? 'right' : 'left', 9.5);
    // truths: measured nadir (at its time) and frequency at 60 s
    let im = PRE; for (let i = PRE; i <= PRE + 60; i++) if (raw[i] < raw[im]) im = i;
    g.strokeStyle = SCOPE.ink; g.lineWidth = 1.5;
    g.beginPath(); g.arc(xt(im - PRE), yf(e.nadir_Hz), 4.5, 0, 2 * Math.PI); g.stroke();
    label(c, small ? 'nadir' : 'measured nadir', xt(im - PRE) + 8, yf(e.nadir_Hz) + 10, SCOPE.ink, 'left', 9);
    g.beginPath(); g.moveTo(xt(55), yf(e.settle_Hz)); g.lineTo(xt(60), yf(e.settle_Hz)); g.stroke();
    label(c, small ? 'mHz vs pre-trip mean' : 'frequency deviation from the 60 s pre-trip mean, mHz · bands: forecasts', padL, top - 12, SCOPE.text, 'left', 9.5);
    if (!sweep) return;
    // --- readings vs N (every 2 s): MW lost and inertia with their 1-sigma bands, the truth dashed
    const S = view.sweep;
    const halves: [string, (q: Reading) => [number, number, number], number | null, number][] = [
      ['MW lost', (q) => q.loss, e.loss_label_MW, padL],
      ['inertia', (q) => q.E, e.E_true_GVAs, padL + iw / 2 + 8],
    ];
    const hw = iw / 2 - 8;
    for (const [name, f, truth, x0] of halves) {
      let a = Infinity, b = -Infinity;
      for (const p of S) { const [, l, u] = f(p.r); a = Math.min(a, l); b = Math.max(b, u); }
      if (truth != null) { a = Math.min(a, truth); b = Math.max(b, truth); }
      const pad = (b - a) * 0.08; a = Math.max(0, a - pad); b += pad;
      const xs = (N: number) => x0 + ((N - NMIN) / (NMAX - NMIN)) * hw;
      const ys = (v: number) => y2 + h2 - ((v - a) / (b - a || 1)) * h2;
      gridLines(c, x0, y2, hw, h2, 6, 2);
      g.beginPath();
      S.forEach((p, k) => (k ? g.lineTo(xs(p.N), ys(f(p.r)[2])) : g.moveTo(xs(p.N), ys(f(p.r)[2]))));
      for (let k = S.length - 1; k >= 0; k--) g.lineTo(xs(S[k].N), ys(f(S[k].r)[1]));
      g.closePath(); g.globalAlpha = 0.18; g.fillStyle = ctx.accent; g.fill(); g.globalAlpha = 1;
      g.strokeStyle = ctx.accent; g.lineWidth = 1.5; g.beginPath();
      S.forEach((p, k) => (k ? g.lineTo(xs(p.N), ys(f(p.r)[0])) : g.moveTo(xs(p.N), ys(f(p.r)[0]))));
      g.stroke();
      if (truth != null) {
        g.strokeStyle = SCOPE.ink; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(x0, ys(truth)); g.lineTo(x0 + hw, ys(truth)); g.stroke(); g.setLineDash([]);
      }
      g.strokeStyle = SCOPE.faint; g.beginPath(); g.moveTo(xs(st.N), y2); g.lineTo(xs(st.N), y2 + h2); g.stroke();
      label(c, small ? name : `${name} vs N${truth != null ? ' (dashed: truth)' : ''}`, x0 + 2, y2 - 8, SCOPE.text, 'left', 9);
      label(c, `${Math.round(a)}`, x0 + 2, y2 + h2 - 6, SCOPE.faint, 'left', 8.5);
      label(c, `${Math.round(b)}`, x0 + 2, y2 + 6, SCOPE.faint, 'left', 8.5);
      label(c, `${NMIN}s`, x0, y2 + h2 + 10, SCOPE.faint, 'left', 9);
      label(c, `${NMAX}s`, x0 + hw, y2 + h2 + 10, SCOPE.faint, 'right', 9);
    }
  }

  await run();
}
