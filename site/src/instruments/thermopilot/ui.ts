// ThermoPilot live instrument: twins of three RECS 2020 exam respondents (never used in training), each in five real
// 2017 climates (BDG2 site weather), a winter and a summer week, under five real tariffs. Every hour the browser
// builds the 298 features, runs the network, applies the comfort shield and steps the house (BuildingSim's hourly
// 5R1C maps of the twin with its equipment tables), closed-loop, against a fixed thermostat and the occupant's own
// setback schedule. The "truth" next to the reading is the full BuildingSim run of the same week (Python, from the lab
// page). Controls change the input: weather (snaps between the real climates), tariff, the comfort weight the network
// is told, the random forecast error and a forecast bias.
import type { InstrumentContext } from '../types.ts';
import { fetchPack } from '../../lib/nn/pack.ts';
import { ctxOf, fixedCtrl, houseOf, loadNet, N_RUN, runPilot, setbackCtrl, simulate, type Ctx, type House, type Sim, type WMeta, type XMeta } from './net.ts';
import { shell, picker, slider, button, scheduler, gauges, canvas, gridLines, label, h, SCOPE, type Canvas2D } from '../kit.ts';

const STYLE = `
.tpi-seg { display: flex; flex-direction: column; gap: 6px; }
.tpi-seg > span { font-size: 12.5px; color: var(--scr-2ink); }
.tpi-seg > div { display: flex; flex-wrap: wrap; gap: 4px; }
.tpi-seg button { flex: 1 1 auto; min-width: 0; padding: 5px 6px; border-radius: 3px; border: 1px solid var(--scr-rule); background: transparent; color: var(--scr-2ink); font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.03em; cursor: pointer; }
.tpi-seg button[aria-pressed='true'] { border-color: var(--c); color: var(--scr); background: var(--c); }
.tpi-kv { display: grid; grid-template-columns: minmax(0, 1fr) auto auto; gap: 3px 10px; font-size: 12px; color: var(--scr-2ink); }
.tpi-kv b { font-family: var(--mono); font-variation-settings: 'wdth' 87.5; font-weight: 400; font-size: 11.5px; color: var(--scr-ink); text-align: right; white-space: nowrap; }
.tpi-kv .h { grid-column: 1 / -1; margin-top: 6px; font-size: 11px; letter-spacing: 0.04em; text-transform: uppercase; color: var(--scr-3ink); }
.tpi-note { font-size: 12px; line-height: 1.5; color: var(--scr-3ink); margin: 0; }
.tpi-note b { font-weight: 500; color: var(--scr-2ink); }
`;

const FIXED = 'rgba(235,230,216,0.55)', SETBACK = '#e0a93a', BAND = 'rgba(127,176,143,0.16)', PRICE = 'rgba(205,187,92,0.28)';
const LAMS = [0.25, 0.5, 1, 2, 4];
const SHORT_T: Record<string, string> = {
  pge_e_tou_c: 'PG&E TOU-C', aps_r3: 'APS R-3 demand', xcel_mn_a04_tod: 'Xcel MN TOD', austin_res_tou: 'Austin TOU', rtp_ercot: 'ERCOT RTP',
};
// each home starts in its own state's climate, with a tariff and week that suit it
const START: Record<number, { c: number; w: number; t: number }> = { 0: { c: 4, w: 1, t: 1 }, 1: { c: 2, w: 0, t: 0 }, 2: { c: 1, w: 0, t: 2 } };
const money = (v: number) => `$${v.toFixed(2)}`;
const pct = (v: number) => `${v >= 0 ? '' : '−'}${Math.abs(v).toFixed(1)} %`;

interface View { tp: Sim; fixed: Sim; setback: Sim; ctx: Ctx; house: House; ri: number; ms: number }

export async function mount(ctx: InstrumentContext) {
  const [Wp, X] = await Promise.all([fetchPack<WMeta>(`${ctx.base}weights`), fetchPack<XMeta>(`${ctx.base}examples`)]);
  const M = X.meta, nC = M.climates.length, nW = M.weeks.length, nT = M.tariffs.length;
  if (!document.getElementById('tpi-inst-style')) document.head.append(h('style', { id: 'tpi-inst-style' }, STYLE));
  const net = loadNet(Wp);
  const ui = shell(ctx.root, {
    title: 'ThermoPilot',
    subtitle: 'a learned thermostat on twins of real exam homes, real 2017 weather and real tariffs',
    params: `MLP 298→384→384→36, ${net.nParams.toLocaleString('en-US')} weights (float16) · matches Python to 1e-14`,
  });
  const st = { home: 0, c: START[0].c, w: START[0].w, t: START[0].t, lam: 2, fe: 1, bias: 0 };
  let view: View | null = null;

  picker(ui.picker, M.homes.map((hm, i) => ({
    label: hm.label,
    sub: `RECS ${hm.id.replace('RECS_', '#')} · ${hm.area} m² · ${hm.state}`,
    value: i,
  })), (i) => { st.home = i; Object.assign(st, { c: START[i].c, w: START[i].w, t: START[i].t }); segC.set(st.c); segW.set(st.w); segT.set(st.t); go(); }, 0);

  const narrow = () => ui.screen.getBoundingClientRect().width < 520;
  const plot = canvas(ui.screen, ctx.compact ? 1.6 : narrow() ? 1.05 : 1.55, (c) => draw(c));

  // ---- controls
  function seg(title: string, items: string[], start: number, onPick: (i: number) => void) {
    const box = h('div', { class: 'tpi-seg', role: 'group', 'aria-label': title }, h('span', {}, title));
    const rowEl = h('div', {});
    const btns = items.map((txt, i) => {
      const b = h('button', { type: 'button', 'aria-pressed': String(i === start) }, txt);
      b.addEventListener('click', () => { set(i); onPick(i); });
      rowEl.append(b);
      return b;
    });
    const set = (i: number) => btns.forEach((b, j) => b.setAttribute('aria-pressed', String(i === j)));
    box.append(rowEl);
    ui.controls.append(box);
    return { set };
  }
  const segC = seg('Weather: real 2017 hourly, five climates', M.climates.map((c) => c[1].replace(/ \(.*\)/, '')), st.c, (i) => { st.c = i; go(); });
  const segW = seg('Week', ['winter, 9–15 Jan', 'summer, 10–16 Jul'], st.w, (i) => { st.w = i; go(); });
  const segT = seg('Tariff (real rates)', M.tariffs.map((t) => SHORT_T[t[0]] ?? t[1]), st.t, (i) => { st.t = i; go(); });
  let sLam: ReturnType<typeof slider> | null = null, sFe: ReturnType<typeof slider> | null = null, sBias: ReturnType<typeof slider> | null = null;
  if (!ctx.compact) {
    sLam = slider(ui.controls, {
      label: 'Comfort weight the network is told', min: 0, max: 4, step: 1, value: st.lam,
      fmt: (v) => `λ × ${LAMS[v]}${v === 2 ? ' (as trained: $3 per K h per 150 m²)' : ''}`,
      hint: 'trained at one λ; the network only sees log(λ / (mean price × size)), so this is also how it reads a price change',
      onInput: (v) => { st.lam = v; go(); },
    });
    sFe = slider(ui.controls, {
      label: 'Forecast error', min: 0, max: 3, step: 0.5, value: st.fe,
      fmt: (v) => (v ? `× ${v}${v === 1 ? ' (about 0.6 K at 1 h, 1.6 K at 24 h)' : ''}` : 'none (perfect forecast)'),
      hint: 'random error on the 24 h outdoor-temperature forecast; the house sees the real weather',
      onInput: (v) => { st.fe = v; go(); },
    });
    sBias = slider(ui.controls, {
      label: 'Forecast bias', min: -4, max: 4, step: 1, value: st.bias,
      fmt: (v) => (v ? `${v > 0 ? '+' : '−'}${Math.abs(v)} K (forecast ${v > 0 ? 'warmer' : 'colder'} than reality)` : 'none'),
      hint: 'a forecast that is wrong the same way every hour; the house keeps the real weather',
      onInput: (v) => { st.bias = v; go(); },
    });
    const rowEl = h('div', {});
    ui.controls.append(rowEl);
    button(rowEl, 'As built', () => { st.lam = 2; st.fe = 1; st.bias = 0; sLam!.set(2); sFe!.set(1); sBias!.set(0); go(); });
  }

  // ---- readout
  const gaugeBox = h('div', {});
  const kv = h('div', { class: 'tpi-kv' });
  const note = h('p', { class: 'tpi-note' });
  ui.readout.append(gaugeBox, kv, note);

  async function run() {
    ui.setBusy(true);
    await new Promise((r) => requestAnimationFrame(r));
    const li = (st.home * nC + st.c) * nW + st.w, ri = li * nT + st.t;
    const house = houseOf(X, li), c = ctxOf(X, house, ri, M.homes[st.home]);
    const t0 = performance.now();
    const tp = runPilot(net, house, c, X.t.z.data, { lamMul: LAMS[st.lam], fe: st.fe, bias: st.bias });
    const ms = performance.now() - t0;
    view = { tp, fixed: simulate(house, c, N_RUN, fixedCtrl(c)), setback: simulate(house, c, N_RUN, setbackCtrl(c)), ctx: c, house, ri, ms };
    plot.redraw();
    render();
    ui.stamp(ms, `${N_RUN} hours: ${N_RUN} network runs + the house, closed loop`);
    ui.setBusy(false);
  }
  const go = scheduler(run, 30);

  function render() {
    if (!view) return;
    const v = view, full = M.runs[v.ri], lam = v.house.lam;
    const sav = (b: number, ref: number) => (100 * (ref - b)) / ref;
    const liveSave = sav(v.tp.bill, v.fixed.bill), fullSave = sav(full.thermopilot.bill, full.fixed.bill);
    const lo = Math.min(-2, liveSave, fullSave, sav(full.mpc_c.bill, full.fixed.bill)) - 1;
    const hi = Math.max(4, liveSave, fullSave, sav(full.oracle.bill, full.fixed.bill)) + 1;
    gaugeBox.innerHTML = '';
    gauges(gaugeBox, [
      { label: 'ThermoPilot: bill saving vs fixed (browser house)', unit: '%', value: liveSave, truth: fullSave, truthLabel: 'full BuildingSim run', range: [lo, hi], digits: 1 },
      { label: 'ThermoPilot: K h outside the comfort band', unit: 'K h', value: v.tp.DH, truth: full.thermopilot.DH, truthLabel: 'full BuildingSim run', range: [0, Math.max(2, v.tp.DH, full.thermopilot.DH, v.setback.DH) * 1.15], digits: 1 },
    ]);
    const rows: [string, string, string][] = [
      ['fixed thermostat', money(v.fixed.bill), `${money(full.fixed.bill)}`],
      ["occupant's setback", money(v.setback.bill), `${money(full.setback.bill)}`],
      ['ThermoPilot', money(v.tp.bill), `${money(full.thermopilot.bill)}`],
      ['MPC with forecast', '—', `${money(full.mpc_c.bill)}`],
      ['perfect foresight', '—', `${money(full.oracle.bill)}`],
    ];
    kv.innerHTML = `<span class="h">bill for the week (energy + demand + gas)</span><span></span><b>browser</b><b>BuildingSim</b>`
      + rows.map(([a, b, c]) => `<span>${a}</span><b>${b}</b><b>${c}</b>`).join('')
      + `<span class="h">comfort, K h outside the band</span>`
      + `<span>fixed / setback / ThermoPilot</span><b>${v.fixed.DH.toFixed(1)} / ${v.setback.DH.toFixed(1)} / ${v.tp.DH.toFixed(1)}</b><b>${full.fixed.DH.toFixed(1)} / ${full.setback.DH.toFixed(1)} / ${full.thermopilot.DH.toFixed(1)}</b>`
      + `<span>J = bill + λ × K h, ThermoPilot (λ $${(lam * LAMS[st.lam]).toFixed(2)})</span><b>${money(v.tp.bill + lam * LAMS[st.lam] * v.tp.DH)}</b><b>${money(full.thermopilot.J)}</b>`;
    const asBuilt = st.lam === 2 && st.fe === 1 && st.bias === 0;
    const hm = M.homes[st.home];
    note.innerHTML = `<b>${hm.label}: twin of RECS 2020 exam respondent ${hm.id.replace('RECS_', '#')} (${hm.state}), ${M.climates[st.c][1]} weather, ${M.weeks[st.w]} week.</b> `
      + `The browser house is the controller's own linear model of the twin, so its bills differ from the full simulator's (ThermoPilot's browser bill is a median 5 % below the full run over the 150 lab weeks; ${pct((100 * (v.tp.bill - full.thermopilot.bill)) / full.thermopilot.bill)} here). `
      + (asBuilt
        ? 'The grade is simulated: there are no real thermostat logs, and on 37 real WPuQ heat-pump homes the simulator could not settle ThermoPilot against setback (the order flipped in 18).'
        : `Changed input: λ × ${LAMS[st.lam]}, forecast error × ${st.fe}, bias ${st.bias > 0 ? '+' : ''}${st.bias} K. The BuildingSim column stays at the as-built settings.`);
  }

  // ---- the week: indoor temperature with the band, then price and HVAC power
  function draw(c: Canvas2D) {
    const { ctx: g, w, h: Hh } = c;
    g.fillStyle = SCOPE.bg; g.fillRect(0, 0, w, Hh);
    if (!view) return;
    const v = view, n = N_RUN, small = w < 520;
    const L = 40, R = 10, T = 18, gap = 24, B = 16;
    const iw = w - L - R, ih = Hh - T - gap - B;
    const h1 = Math.round(ih * 0.62), h2 = ih - h1;
    const y1 = T, y2 = y1 + h1 + gap;
    const xs = (k: number) => L + (k / n) * iw;
    // panel 1: indoor temperature
    let tmin = Infinity, tmax = -Infinity;
    const full = X.t.R_Ttp.data.subarray(v.ri * n, (v.ri + 1) * n);
    for (let k = 0; k < n; k++) for (const x of [v.tp.T[k], v.fixed.T[k], v.setback.T[k], v.ctx.lo[k], Math.min(v.ctx.hi[k], 32), full[k]]) { tmin = Math.min(tmin, x); tmax = Math.max(tmax, x); }
    tmin = Math.floor(tmin - 0.5); tmax = Math.ceil(tmax + 0.5);
    const yT = (t: number) => y1 + h1 - ((t - tmin) / (tmax - tmin)) * h1;
    gridLines(c, L, y1, iw, h1, 7, 4);
    for (let j = 0; j <= 4; j++) { const t = tmin + ((tmax - tmin) * j) / 4; label(c, `${t.toFixed(0)}°`, L - 4, yT(t), SCOPE.faint, 'right', 9); }
    g.fillStyle = BAND; g.beginPath();
    for (let k = 0; k < n; k++) { const y = yT(Math.min(v.ctx.hi[k], tmax)); k ? g.lineTo(xs(k), y) : g.moveTo(xs(k), y); g.lineTo(xs(k + 1), y); }
    for (let k = n - 1; k >= 0; k--) { const y = yT(Math.max(v.ctx.lo[k], tmin)); g.lineTo(xs(k + 1), y); g.lineTo(xs(k), y); }
    g.closePath(); g.fill();
    const line = (arr: ArrayLike<number>, yf: (x: number) => number, col: string, lw: number, dash: number[] = []) => {
      g.strokeStyle = col; g.lineWidth = lw; g.setLineDash(dash); g.beginPath();
      for (let k = 0; k < n; k++) { const y = yf(arr[k]); k ? g.lineTo(xs(k + 0.5), y) : g.moveTo(xs(k + 0.5), y); }
      g.stroke(); g.setLineDash([]);
    };
    line(full, yT, SCOPE.ink, 1, [3, 3]);
    line(v.fixed.T, yT, FIXED, 1);
    line(v.setback.T, yT, SETBACK, 1.1);
    line(v.tp.T, yT, ctx.accent, 2);
    label(c, small ? 'indoor °C · band green' : 'indoor °C: ThermoPilot · setback (amber) · fixed (grey) · full BuildingSim ThermoPilot (dashed) · comfort band (green)', L + 4, y1 - 8, SCOPE.text, 'left', 9.5);
    // panel 2: price bars and HVAC power
    let pmax = 0.01, kmax = 0.5;
    for (let k = 0; k < n; k++) { pmax = Math.max(pmax, v.ctx.price[k]); kmax = Math.max(kmax, v.tp.P[k], v.fixed.P[k]); }
    const yP = (p: number) => y2 + h2 - (p / kmax) * h2;
    gridLines(c, L, y2, iw, h2, 7, 2);
    g.fillStyle = PRICE;
    for (let k = 0; k < n; k++) { const ph = (v.ctx.price[k] / pmax) * h2 * 0.9; g.fillRect(xs(k), y2 + h2 - ph, Math.max(1, iw / n), ph); }
    for (let k = 0; k < n; k++) if (v.ctx.drate[k] > 0) { g.fillStyle = 'rgba(240,116,93,0.5)'; g.fillRect(xs(k), y2 + h2 - 3, Math.max(1, iw / n), 3); }
    line(v.fixed.P, yP, FIXED, 1);
    line(v.tp.P, yP, ctx.accent, 1.5);
    label(c, `${kmax.toFixed(1)} kW`, L - 4, y2 + 4, SCOPE.faint, 'right', 9);
    label(c, small ? 'HVAC kW · price bars' : `HVAC power (electric + gas), kW: ThermoPilot · fixed (grey) · bars: price (max $${pmax.toFixed(2)}/kWh), red: demand-charge hours`, L + 4, y2 - 8, SCOPE.text, 'left', 9.5);
    const dw = M.week_days[st.w];
    label(c, dw[0], L, Hh - 7, SCOPE.faint, 'left', 9);
    label(c, dw[1], L + iw, Hh - 7, SCOPE.faint, 'right', 9);
  }

  await run();
}
