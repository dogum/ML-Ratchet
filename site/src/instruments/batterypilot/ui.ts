// BatteryPilot live instrument: three real exam node-years (ERCOT 2023 and 2024 hubs, CAISO SP15 2025), never used in
// training. For the chosen month the browser rebuilds the 135 causal features (and, for v1.3, the day-ahead backbone
// curve) every hour, runs the network for the value curve, and runs a 1 MW battery against the real RT price with
// the offer rule; GridData's daily-spread rule runs alongside, and both settle with rainflow wear. Controls change
// the battery (duration, round-trip efficiency, wear cost), which changes the network's input (the wear cost it is
// told) and the physics it is graded with.
// Perfect hindsight is an LP solved in Python for every slider setting and is labelled as precomputed.
import type { InstrumentContext } from '../types.ts';
import { fetchPack } from '../../lib/nn/pack.ts';
import { battery, dailySpread, makeNet, pathOf, runWindow, settle, type BPMeta, type Path, type Run, type Settled, type XMeta } from './net.ts';
import { shell, picker, slider, button, scheduler, gauges, canvas, gridLines, label, h, SCOPE, type Canvas2D } from '../kit.ts';

const STYLE = `
.bpi-seg { display: flex; flex-direction: column; gap: 6px; }
.bpi-seg > span { font-size: 12.5px; color: var(--scr-2ink); }
.bpi-seg > div { display: flex; gap: 4px; }
.bpi-seg button { flex: 1 1 auto; min-width: 0; padding: 5px 4px; border-radius: 3px; border: 1px solid var(--scr-rule); background: transparent; color: var(--scr-2ink); font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.04em; cursor: pointer; }
.bpi-seg button[aria-pressed='true'] { border-color: var(--c); color: var(--scr); background: var(--c); }
.bpi-kv { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 3px 10px; font-size: 12px; color: var(--scr-2ink); }
.bpi-kv b { font-family: var(--mono); font-variation-settings: 'wdth' 87.5; font-weight: 400; font-size: 11.5px; color: var(--scr-ink); text-align: right; white-space: nowrap; }
.bpi-kv .h { grid-column: 1 / -1; margin-top: 6px; font-size: 11px; letter-spacing: 0.04em; text-transform: uppercase; color: var(--scr-3ink); }
.bpi-note { font-size: 12px; line-height: 1.5; color: var(--scr-3ink); margin: 0; }
.bpi-note b { font-weight: 500; color: var(--scr-2ink); }
.bpi-vc { display: flex; flex-direction: column; gap: 4px; }
.bpi-vc .ig-l { display: block; }
.bpi-vc .inst-cv { border: 1px solid var(--scr-rule); border-radius: 3px; }
`;

const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const ETA0 = 0.85, COST0 = 130;
const CHARGE = '#5ec8d8';

const dateOf = (day: number) => new Date(Date.UTC(2000, 0, 1) + day * 864e5);
const ymd = (day: number) => { const d = dateOf(day); return `${d.getUTCDate()} ${MON[d.getUTCMonth()]}`; };
const money = (v: number, d = 2) => `${v < 0 ? '−' : ''}$${Math.abs(v).toFixed(d)}`;
/** price axis: asinh(p / 50), so $20 and $5,000 both read */
const sq = (p: number) => Math.asinh(p / 50);

interface View {
  a: number; b: number; bp: Run; sp: { ch: Float64Array; dis: Float64Array; soc: Float64Array };
  sBP: Settled; sSP: Settled; pf: number; cumBP: Float64Array; cumSP: Float64Array; ms: number;
}

export async function mount(ctx: InstrumentContext) {
  const [W, X] = await Promise.all([fetchPack<BPMeta>(`${ctx.base}weights`), fetchPack<XMeta>(`${ctx.base}examples`)]);
  const M = X.meta, ex = M.examples, K = W.meta.knots.length;
  if (!document.getElementById('bpi-inst-style')) document.head.append(h('style', { id: 'bpi-inst-style' }, STYLE));
  const ui = shell(ctx.root, {
    title: 'BatteryPilot',
    subtitle: 'a 1 MW grid battery on a real exam year of prices, never used in training',
    params: W.meta.hybrid
      ? `day-ahead backbone + MLP ${W.t.x_mu.data.length}→128→128→5 residual, ${W.meta.n_params.toLocaleString('en-US')} weights (float16) · matches Python to 1e-5`
      : `MLP 135→128→128→5, ${W.meta.n_params.toLocaleString('en-US')} weights (float16) · matches Python to 1e-5`,
  });
  const net = makeNet(W);
  const paths: Record<number, Path> = {};
  const path = (i: number) => (paths[i] ??= pathOf(X, i));
  const peakMonth = (i: number) => { const ms = ex[i].month_start; let m = 0; while (m < 11 && ms[m + 1] <= ex[i].peak.i) m++; return m; };
  const st = { i: 0, d: 2, m: peakMonth(0), eta: ETA0, cost: COST0, hover: -1 };
  let view: View | null = null;

  picker(ui.picker, ex.map((e, i) => ({
    label: `${e.market} ${e.label} ${e.year}`,
    sub: `peak RT $${Math.round(e.peak.rt).toLocaleString('en-US')} · ${e.n.toLocaleString('en-US')} h`,
    value: i,
  })), (i) => { st.i = i; st.m = peakMonth(i); sMon.set(st.m + 1); st.hover = -1; go(); }, 0);

  const narrow = () => ui.screen.getBoundingClientRect().width < 520;
  const plot = canvas(ui.screen, ctx.compact ? 1.5 : narrow() ? 1.0 : 1.45, (c) => draw(c));
  const cv = ui.screen.querySelector('canvas') as HTMLCanvasElement;

  // ---- controls
  const seg = h('div', { class: 'bpi-seg', role: 'group', 'aria-label': 'Battery duration' }, h('span', {}, 'Battery: 1 MW for'));
  const segRow = h('div', {});
  const dBtns = M.durations.map((d) => {
    const b = h('button', { type: 'button', 'aria-pressed': String(d === st.d) }, `${d} h`);
    b.title = `${d} MWh usable`;
    b.addEventListener('click', () => { st.d = d; dBtns.forEach((x, k) => x.setAttribute('aria-pressed', String(M.durations[k] === d))); go(); });
    segRow.append(b);
    return b;
  });
  seg.append(segRow);
  ui.controls.append(seg);
  const sMon = slider(ui.controls, {
    label: 'Month', min: 1, max: 12, step: 1, value: st.m + 1,
    fmt: (v) => `${MON[v - 1]} ${ex[st.i].year}`,
    hint: 'the battery starts the month empty; features use the hours before it',
    onInput: (v) => { st.m = v - 1; st.hover = -1; go(); },
  });
  let sEta: ReturnType<typeof slider> | null = null, sCost: ReturnType<typeof slider> | null = null;
  if (!ctx.compact) {
    sEta = slider(ui.controls, {
      label: 'Round-trip efficiency', min: 75, max: 95, step: 5, value: ETA0 * 100,
      fmt: (v) => `${v} %${v === ETA0 * 100 ? ' (as built)' : ''}`,
      hint: 'losses in the offer rule and the settlement; also changes the wear cost per MWh the network is told',
      onInput: (v) => { st.eta = v / 100; go(); },
    });
    sCost = slider(ui.controls, {
      label: 'Cell replacement cost', min: 0, max: 260, step: 65, value: COST0,
      fmt: (v) => `$${v}/kWh${v === COST0 ? ' (as built)' : v === 0 ? ' (no wear)' : ''}`,
      hint: 'sets the wear cost per MWh sold (a network input) and the rainflow wear charged',
      onInput: (v) => { st.cost = v; go(); },
    });
    const row = h('div', {});
    ui.controls.append(row);
    button(row, 'As built', () => { st.eta = ETA0; st.cost = COST0; sEta!.set(ETA0 * 100); sCost!.set(COST0); go(); });
  }

  // ---- readout
  const gaugeBox = h('div', {});
  const vcBox = h('div', { class: 'bpi-vc' }, h('span', { class: 'ig-l' }, 'Value curve ($/MWh stored vs charge), hour under the cursor'));
  const vcText = h('div', { class: 'bpi-kv' });
  const kv = h('div', { class: 'bpi-kv' });
  const note = h('p', { class: 'bpi-note' });
  ui.readout.append(gaugeBox, vcBox, kv, note);
  const vc = canvas(vcBox, ctx.compact ? 2.4 : 1.9, (c) => drawCurve(c));
  vcBox.append(vcText);

  const gridIdx = (arr: number[], v: number) => { let k = 0; for (let j = 1; j < arr.length; j++) if (Math.abs(arr[j] - v) < Math.abs(arr[k] - v)) k = j; return k; };

  async function run() {
    ui.setBusy(true);
    await new Promise((r) => requestAnimationFrame(r));
    const p = path(st.i), e = ex[st.i], ms = e.month_start, a = ms[st.m], b = ms[st.m + 1];
    const bat = battery(st.d, st.eta, st.cost);
    const t0 = performance.now();
    const bp = runWindow(W, p, bat, a, b, net);
    const msRun = performance.now() - t0;
    const sp = dailySpread(p, bat, a, b);
    const sBP = settle(bat, p, a, bp.ch, bp.dis, bp.soc), sSP = settle(bat, p, a, sp.ch, sp.dis, sp.soc);
    const pfT = X.t[`pf${st.i}`];
    const di = M.durations.indexOf(st.d), ei = gridIdx(M.etas, st.eta), ci = gridIdx(M.costs, st.cost);
    const pf = pfT.data[((di * M.etas.length + ei) * M.costs.length + ci) * 12 + st.m];
    // cumulative gross minus wear spread over the hours (wear shown in total; the curve is gross revenue so far)
    const n = b - a, cumBP = new Float64Array(n + 1), cumSP = new Float64Array(n + 1);
    for (let k = 0; k < n; k++) { cumBP[k + 1] = cumBP[k] + p.rt[a + k] * (bp.dis[k] - bp.ch[k]); cumSP[k + 1] = cumSP[k] + p.rt[a + k] * (sp.dis[k] - sp.ch[k]); }
    if (st.hover < a || st.hover >= b) { let im = a; for (let t = a; t < b; t++) if (p.rt[t] > p.rt[im]) im = t; st.hover = im; }
    view = { a, b, bp, sp, sBP, sSP, pf, cumBP, cumSP, ms: msRun };
    plot.redraw(); vc.redraw();
    render();
    ui.stamp(msRun, `${n} hours: ${n} network runs + the battery, ${MON[st.m]} ${e.year}`);
    ui.setBusy(false);
  }
  const go = scheduler(run, 30);

  function render() {
    if (!view) return;
    const e = ex[st.i], v = view, kW = 1000;
    const top = Math.max(v.pf, v.sBP.net, v.sSP.net, 1) / kW * 1.08, lo = Math.min(0, v.sBP.net, v.sSP.net) / kW;
    gaugeBox.innerHTML = '';
    const tl = 'perfect hindsight (LP, Python)';
    gauges(gaugeBox, [
      { label: 'BatteryPilot, net of wear', unit: '$/kW', value: v.sBP.net / kW, truth: v.pf / kW, truthLabel: tl, range: [lo, top], digits: 2 },
      { label: 'Daily-spread rule (GridData)', unit: '$/kW', value: v.sSP.net / kW, truth: v.pf / kW, truthLabel: tl, range: [lo, top], digits: 2 },
    ]);
    const share = (x: number) => (v.pf > 0 ? `${Math.round((100 * x) / v.pf)} %` : '—');
    const an = e.annual[String(st.d)];
    const asBuilt = st.eta === ETA0 && st.cost === COST0;
    kv.innerHTML = [
      ['h', `${MON[st.m]} ${e.year}, this battery`],
      ['Share of hindsight: BatteryPilot', share(v.sBP.net)],
      ['Share of hindsight: spread rule', share(v.sSP.net)],
      ['Gross / wear, BatteryPilot', `${money(v.sBP.gross / kW)} / ${money(v.sBP.wear / kW)}`],
      ['Gross / wear, spread rule', `${money(v.sSP.gross / kW)} / ${money(v.sSP.wear / kW)}`],
      ['Cycles: BatteryPilot / spread', `${v.sBP.cycles.toFixed(1)} / ${v.sSP.cycles.toFixed(1)}`],
      ['h', `whole year, ${st.d} h, 85 %, $130/kWh (Python)`],
      ['Perfect hindsight', `${money(an.pf, 1)}/kW-yr`],
      ['BatteryPilot', `${money(an.bp, 1)} (${Math.round((100 * an.bp) / an.pf)} %)`],
      ['Day-ahead optimiser', `${money(an.rolling, 1)} (${Math.round((100 * an.rolling) / an.pf)} %)`],
      ['Daily-spread rule', `${money(an.spread, 1)} (${Math.round((100 * an.spread) / an.pf)} %)`],
    ].map(([k, x]) => (k === 'h' ? `<span class="h">${x}</span>` : `<span>${k}</span><b>${x}</b>`)).join('');
    note.innerHTML = asBuilt
      ? `<b>Real ${e.market} ${e.node} prices, ${e.year}</b> (exam year). Each hour the network sees only what was known before it: past RT prices, published day-ahead prices, the demand forecast and the temperature. The battery is settled at the real RT price. Hindsight knows the whole month; no causal policy can reach it.`
      : `<b>Changed battery: ${Math.round(st.eta * 100)} % round trip, $${st.cost}/kWh.</b> The network is told the new wear cost per MWh; losses and wear change in the offer rule and in the settlement. Hindsight is re-solved in Python for this setting.`;
    updateCurveText();
  }

  // ---- the hovered hour's value curve
  function updateCurveText() {
    if (!view || st.hover < 0) return;
    const p = path(st.i), k = st.hover - view.a, bat = battery(st.d, st.eta, st.cost);
    const price = p.rt[st.hover], cl = view.bp.cl[k];
    const s0 = view.bp.soc[k], s1 = view.bp.soc[k + 1];
    const act = view.bp.dis[k] > 1e-6 ? `sold ${view.bp.dis[k].toFixed(2)} MWh` : view.bp.ch[k] > 1e-6 ? `bought ${view.bp.ch[k].toFixed(2)} MWh` : 'held';
    vcText.innerHTML = [
      ['Hour (local)', `${ymd(p.day[st.hover])}, ${String(p.hour[st.hover]).padStart(2, '0')}:00`],
      ['Real-time price (truth)', `${money(price, 2)}/MWh`],
      ['Sell if value below', `${money(bat.etad * (price - cl), 1)} = η(p − wear ${money(cl, 1)})`],
      ['Buy if value above', `${money(price / bat.etac, 1)} = p / η`],
      ['Charge before → after', `${Math.round(s0 * 100)} % → ${Math.round(s1 * 100)} %, ${act}`],
    ].map(([a, b]) => `<span>${a}</span><b>${b}</b>`).join('');
  }

  function drawCurve(c: Canvas2D) {
    const { ctx: g, w, h: H } = c;
    g.fillStyle = SCOPE.bg; g.fillRect(0, 0, w, H);
    if (!view || st.hover < 0) return;
    const p = path(st.i), k = st.hover - view.a, bat = battery(st.d, st.eta, st.cost);
    const V = Array.from(view.bp.V.subarray(k * K, k * K + K)), knots = W.meta.knots;
    const price = p.rt[st.hover], cl = view.bp.cl[k];
    const yd = bat.etad * (price - cl), yc = price / bat.etac;
    const vals = [...V, yd, yc].filter(Number.isFinite);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = (hi - lo || 10) * 0.12; lo -= pad; hi += pad;
    const L = 46, R = 8, T = 10, B = 18, iw = w - L - R, ih = H - T - B;
    const xs = (s: number) => L + s * iw, ys = (v: number) => T + ih - ((v - lo) / (hi - lo)) * ih;
    gridLines(c, L, T, iw, ih, 4, 3);
    for (let j = 0; j <= 3; j++) { const v = lo + ((hi - lo) * j) / 3; label(c, money(v, 0), L - 4, ys(v), SCOPE.faint, 'right', 9); }
    for (let j = 0; j <= 4; j++) label(c, `${j * 25}%`, xs(j / 4), T + ih + 10, SCOPE.faint, j === 0 ? 'left' : j === 4 ? 'right' : 'center', 9);
    const hl = (y: number, col: string, txt: string, below: boolean) => {
      if (y < lo || y > hi) return;
      g.strokeStyle = col; g.setLineDash([4, 3]); g.lineWidth = 1; g.beginPath(); g.moveTo(L, ys(y)); g.lineTo(L + iw, ys(y)); g.stroke(); g.setLineDash([]);
      label(c, txt, L + iw - 2, ys(y) + (below ? 8 : -7), col, 'right', 9);
    };
    hl(yc, CHARGE, 'p/η: charge where the curve is above', yc < yd);
    hl(yd, SCOPE.ink, 'η(p − wear): discharge where the curve is below', !(yc < yd));
    g.strokeStyle = ctx.accent; g.lineWidth = 2; g.beginPath();
    V.forEach((v, j) => (j ? g.lineTo(xs(knots[j]), ys(v)) : g.moveTo(xs(knots[j]), ys(v))));
    g.stroke();
    g.fillStyle = ctx.accent;
    V.forEach((v, j) => { g.beginPath(); g.arc(xs(knots[j]), ys(v), 2.5, 0, 2 * Math.PI); g.fill(); });
    const s0 = view.bp.soc[k], s1 = view.bp.soc[k + 1];
    g.strokeStyle = SCOPE.faint; g.lineWidth = 1; g.beginPath(); g.moveTo(xs(s0), T); g.lineTo(xs(s0), T + ih); g.stroke();
    g.strokeStyle = SCOPE.ink; g.lineWidth = 1.4; g.beginPath(); g.moveTo(xs(s1), T); g.lineTo(xs(s1), T + ih); g.stroke();
    label(c, 'after', xs(s1) + (s1 > 0.8 ? -3 : 3), T + 6, SCOPE.ink, s1 > 0.8 ? 'right' : 'left', 9);
  }

  // ---- the month: prices, the battery's charge, money
  let layout = { L: 44, iw: 1, n: 1 };
  function draw(c: Canvas2D) {
    const { ctx: g, w, h: H } = c;
    g.fillStyle = SCOPE.bg; g.fillRect(0, 0, w, H);
    if (!view) return;
    const p = path(st.i), v = view, n = v.b - v.a;
    const L = 44, R = 10, T = 18, gap = 22, B = 16;
    const iw = w - L - R, ih = H - T - 2 * gap - B;
    const h1 = Math.round(ih * 0.42), h2 = Math.round(ih * 0.26), h3 = ih - h1 - h2;
    const y1 = T, y2 = y1 + h1 + gap, y3 = y2 + h2 + gap;
    layout = { L, iw, n };
    const xs = (k: number) => L + (k / n) * iw;
    const small = w < 520;
    // panel 1: prices (asinh axis) with the battery's actions as bars
    let pmin = 0, pmax = 50;
    for (let t = v.a; t < v.b; t++) { pmin = Math.min(pmin, p.rt[t], p.da[t]); pmax = Math.max(pmax, p.rt[t], p.da[t]); }
    const qa = sq(pmin) - 0.1, qb = sq(pmax) + 0.15;
    const yp = (pr: number) => y1 + h1 - ((sq(pr) - qa) / (qb - qa)) * h1;
    gridLines(c, L, y1, iw, h1, 8, 0);
    for (const tk of [-100, 0, 30, 100, 300, 1000, 3000]) if (tk >= pmin && tk <= pmax * 1.1) {
      g.strokeStyle = SCOPE.grid; g.beginPath(); g.moveTo(L, Math.round(yp(tk)) + 0.5); g.lineTo(L + iw, Math.round(yp(tk)) + 0.5); g.stroke();
      label(c, `$${tk.toLocaleString('en-US')}`, L - 4, yp(tk), SCOPE.faint, 'right', 9);
    }
    const bw = Math.max(1, iw / n);
    for (let k = 0; k < n; k++) {
      const x = xs(k);
      if (v.bp.dis[k] > 1e-6) { g.fillStyle = ctx.accent; g.globalAlpha = 0.25 + 0.5 * v.bp.dis[k]; g.fillRect(x, y1 + h1 - 6, bw, 6); }
      else if (v.bp.ch[k] > 1e-6) { g.fillStyle = CHARGE; g.globalAlpha = 0.25 + 0.5 * v.bp.ch[k]; g.fillRect(x, y1 + h1 - 6, bw, 6); }
    }
    g.globalAlpha = 1;
    const line = (arr: ArrayLike<number>, col: string, lw: number, dash: number[] = []) => {
      g.strokeStyle = col; g.lineWidth = lw; g.setLineDash(dash); g.beginPath();
      for (let k = 0; k < n; k++) { const y = yp(arr[v.a + k]); k ? g.lineTo(xs(k + 0.5), y) : g.moveTo(xs(k + 0.5), y); }
      g.stroke(); g.setLineDash([]);
    };
    line(p.da, SCOPE.faint, 1, [3, 3]);
    line(p.rt, SCOPE.ink, 1.1);
    label(c, small ? 'RT price (truth), DA dashed' : 'real-time price (truth) · day-ahead dashed · bars: sold / bought', L + 4, y1 - 8, SCOPE.text, 'left', 9.5);
    // panel 2: state of charge
    gridLines(c, L, y2, iw, h2, 8, 2);
    const ys2 = (s: number) => y2 + h2 - s * h2;
    label(c, '0%', L - 4, ys2(0), SCOPE.faint, 'right', 9); label(c, '100%', L - 4, ys2(1), SCOPE.faint, 'right', 9);
    const socLine = (s: ArrayLike<number>, col: string, lw: number, alpha = 1) => {
      g.strokeStyle = col; g.lineWidth = lw; g.globalAlpha = alpha; g.beginPath();
      for (let k = 0; k <= n; k++) { const y = ys2(Math.min(Math.max(s[k], 0), 1)); k ? g.lineTo(xs(k), y) : g.moveTo(xs(k), y); }
      g.stroke(); g.globalAlpha = 1;
    };
    socLine(v.sp.soc, SCOPE.faint, 1);
    socLine(v.bp.soc, ctx.accent, 1.4);
    label(c, 'state of charge: BatteryPilot · spread rule (grey)', L + 4, y2 - 8, SCOPE.text, 'left', 9.5);
    // panel 3: gross revenue so far, and the month's net results
    let cmin = 0, cmax = Math.max(1, v.pf);
    for (let k = 0; k <= n; k++) { cmin = Math.min(cmin, v.cumBP[k], v.cumSP[k]); cmax = Math.max(cmax, v.cumBP[k], v.cumSP[k]); }
    cmax *= 1.06;
    const ys3 = (x: number) => y3 + h3 - ((x - cmin) / (cmax - cmin || 1)) * h3;
    gridLines(c, L, y3, iw, h3, 8, 2);
    for (let j = 0; j <= 2; j++) { const x = cmin + ((cmax - cmin) * j) / 2; label(c, `${money(x / 1000, Math.abs(x) > 20000 ? 0 : 1)}k`, L - 4, ys3(x), SCOPE.faint, 'right', 9); }
    const cum = (arr: Float64Array, col: string, lw: number) => {
      g.strokeStyle = col; g.lineWidth = lw; g.beginPath();
      for (let k = 0; k <= n; k++) { const y = ys3(arr[k]); k ? g.lineTo(xs(k), y) : g.moveTo(xs(k), y); }
      g.stroke();
    };
    cum(v.cumSP, SCOPE.faint, 1);
    cum(v.cumBP, ctx.accent, 1.5);
    g.strokeStyle = SCOPE.ink; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(L, ys3(v.pf)); g.lineTo(L + iw, ys3(v.pf)); g.stroke(); g.setLineDash([]);
    label(c, `hindsight, net (Python) $${(v.pf / 1000).toFixed(1)}k`, L + iw - 2, ys3(v.pf) - 7, SCOPE.ink, 'right', 9);
    label(c, small ? 'gross $ so far' : 'gross revenue so far, $ for 1 MW: BatteryPilot · spread rule (grey)', L + 4, y3 - 8, SCOPE.text, 'left', 9.5);
    // day ticks
    const days: number[] = [];
    for (let k = 0; k < n; k++) if (k === 0 || p.day[v.a + k] !== p.day[v.a + k - 1]) days.push(k);
    const every = days.length > 20 ? (small ? 10 : 5) : 2;
    days.forEach((k, j) => { if (j % every === 0) label(c, ymd(p.day[v.a + k]), xs(k), H - 7, SCOPE.faint, j === 0 ? 'left' : 'center', 9); });
    // cursor
    if (st.hover >= v.a && st.hover < v.b) {
      const x = xs(st.hover - v.a + 0.5);
      g.strokeStyle = 'rgba(235,230,216,0.35)'; g.lineWidth = 1; g.beginPath(); g.moveTo(x, T); g.lineTo(x, y3 + h3); g.stroke();
    }
  }

  const hoverAt = (clientX: number) => {
    if (!view) return;
    const r = cv.getBoundingClientRect();
    const k = Math.floor(((clientX - r.left - layout.L) / layout.iw) * layout.n);
    if (k < 0 || k >= layout.n) return;
    const t = view.a + k;
    if (t === st.hover) return;
    st.hover = t;
    plot.redraw(); vc.redraw(); updateCurveText();
  };
  cv.addEventListener('pointermove', (ev) => hoverAt(ev.clientX));
  cv.addEventListener('pointerdown', (ev) => hoverAt(ev.clientX));

  await run();
}
