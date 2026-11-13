// EnergyNet live instrument: five real Ann Arbor trips from the exam (VED, one per powertrain), never used in training.
// The browser recomputes the route features, the road-load physics and the five-network ensemble from the 1 Hz route
// on every change. Controls change the trip (air temperature, wind, grade) or the vehicle record (mass, start charge,
// warm or cold start); the measured energy of the trip as driven stays on screen as the truth.
import type { InstrumentContext } from '../types.ts';
import { fetchPack } from '../../lib/nn/pack.ts';
import { predict, perturb, type EMeta, type EExample, type Perturb, type Prediction, type Vehicle } from './net.ts';
import { tripOf } from './parity.ts';
import { shell, picker, slider, button, scheduler, gauges, canvas, gridLines, label, h, SCOPE, type Canvas2D } from '../kit.ts';

const STYLE = `
.en-block { display: flex; flex-direction: column; gap: 6px; }
.en-row { display: grid; grid-template-columns: 124px minmax(0, 1fr) 56px; gap: 8px; align-items: center; }
.en-row .n { font-size: 12px; color: var(--scr-2ink); }
.en-row .v { font-family: var(--mono); font-variation-settings: 'wdth' 87.5; font-size: 11.5px; text-align: right; color: var(--scr-ink); white-space: nowrap; }
.en-trk { position: relative; height: 10px; }
.en-trk span { position: absolute; left: 0; top: 1px; height: 8px; border-radius: 2px; background: var(--k); transition: width 0.3s var(--ease, ease); }
.en-kv { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 3px 10px; font-size: 12px; color: var(--scr-2ink); }
.en-kv b { font-family: var(--mono); font-variation-settings: 'wdth' 87.5; font-weight: 400; font-size: 11.5px; color: var(--scr-ink); text-align: right; white-space: nowrap; }
.en-note { font-size: 12px; line-height: 1.5; color: var(--scr-3ink); margin: 0; }
.en-note b { font-weight: 500; color: var(--scr-2ink); }
.en-seg { display: flex; flex-direction: column; gap: 6px; }
.en-seg > span { font-size: 12.5px; color: var(--scr-2ink); }
.en-seg > div { display: flex; gap: 4px; }
.en-seg button { flex: 1 1 auto; min-width: 0; padding: 5px 4px; border-radius: 3px; border: 1px solid var(--scr-rule); background: transparent; color: var(--scr-2ink); font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.04em; cursor: pointer; }
.en-seg button[aria-pressed='true'] { border-color: var(--c); color: var(--scr); background: var(--c); }
.en-btns { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 8px; }
.en-btns .inst-btn[aria-pressed='true'] { border-color: var(--c); color: var(--c); }
@media (max-width: 760px) { .en-row { grid-template-columns: 116px minmax(0, 1fr) 52px; } }
`;

const NAME: Record<string, string> = { ICE: 'Petrol car', HEV: 'Hybrid', P2: 'Parallel hybrid', PHEV: 'Plug-in hybrid', EV: 'Electric car' };
const ORDER = ['ICE', 'HEV', 'P2', 'PHEV', 'EV'];
const COMP: [string, string][] = [['Air drag', '#5ec8d8'], ['Rolling', '#c9c3b2'], ['Friction brakes', '#e07a5f'], ['Accessories, HVAC', '#b79be0'], ['Powertrain losses', '']];
const WIND: [string, number | null][] = [['Station', null], ['Head', 0], ['Cross', 90], ['Tail', 180]];
const T_LO = -25, T_HI = 45; // temperature sweep, C

/** Energy (MJ) in the unit people use for this powertrain. */
function unitOf(v: Vehicle) {
  if (v.architecture === 'bev') return { u: 'kWh', f: 1 / 3.6, d: 2 };
  if (v.architecture.startsWith('phev')) return { u: 'MJ', f: 1, d: 2 };
  const diesel = v.fuel === 'diesel';
  return { u: diesel ? 'L diesel' : 'L petrol', f: 1 / 0.042 / (diesel ? 832 : 745), d: 2 };
}
const plugIn = (v: Vehicle) => v.architecture === 'bev' || v.architecture.startsWith('phev');

export async function mount(ctx: InstrumentContext) {
  const [W, X] = await Promise.all([fetchPack<EMeta>(`${ctx.base}weights`), fetchPack<EExample[]>(`${ctx.base}examples`)]);
  const ex = X.meta;
  if (!document.getElementById('en-inst-style')) document.head.append(h('style', { id: 'en-inst-style' }, STYLE));
  const ui = shell(ctx.root, {
    title: 'EnergyNet',
    subtitle: 'trip energy before the drive — a real Ann Arbor trip, never used in training',
    params: `5 networks, ${(W.meta.n_params / 1e6).toFixed(2)} M parameters · matches Python to 1e-6`,
  });
  const order = ex.map((_, i) => i).sort((a, b) => ORDER.indexOf(ex[a].stratum) - ORDER.indexOf(ex[b].stratum));
  const st = { i: order[0], dT: 0, mass: 1, grade: 0, wind: null as number | null, ws: 5, soc: null as number | null, cold: false };
  let view: { r: Prediction | null; sweep: { T: number; v: number; lo: number; hi: number }[] } = { r: null, sweep: [] };

  picker(ui.picker, order.map((i) => {
    const e = ex[i], u = unitOf(e.vehicle);
    return { label: NAME[e.stratum] ?? e.stratum, sub: `${e.km.toFixed(1)} km · ${Math.round(e.T_mean)} °C · ${(e.measured_MJ * u.f).toFixed(u.d)} ${u.u}`, value: i };
  }), (i) => { st.i = i; st.dT = 0; st.soc = null; sT.set(0); sM.set(st.mass); syncSoc(); syncWind(); go(); }, 0);

  const narrow = () => ui.screen.getBoundingClientRect().width < 520;
  const plot = canvas(ui.screen, ctx.compact ? 1.9 : narrow() ? 1.15 : 1.55, (c) => draw(c));

  const e0 = () => ex[st.i];
  const sT = slider(ui.controls, {
    label: 'Air temperature', min: -25, max: 15, step: 1, value: 0,
    fmt: (v) => `${(e0().T_mean + v).toFixed(1)} °C${v === 0 ? '' : ` (${v > 0 ? '+' : ''}${v})`}`,
    hint: 'shifts every second of the station record; air density follows',
    onInput: (v) => { st.dT = v; go(); },
  });
  const sM = slider(ui.controls, {
    label: 'Vehicle mass', min: 0.7, max: 1.6, step: 0.05, value: 1,
    fmt: (v) => `${Math.round((e0().vehicle.mass_kg ?? 0) * v)} kg${v === 1 ? '' : ` (×${v.toFixed(2)})`}`,
    hint: 'load or passengers; the network is told the new mass',
    onInput: (v) => { st.mass = v; go(); },
  });
  const sG = slider(ui.controls, {
    label: 'Extra grade', min: -3, max: 3, step: 0.25, value: 0,
    fmt: (v) => (v === 0 ? 'none' : `${v > 0 ? '+' : ''}${v.toFixed(2)} %`),
    hint: 'added to every second of the lidar grade',
    onInput: (v) => { st.grade = v / 100; go(); },
  });
  let sW: ReturnType<typeof slider> | null = null, sS: ReturnType<typeof slider> | null = null, warmBtn: HTMLButtonElement | null = null;
  if (!ctx.compact) {
    const seg = h('div', { class: 'en-seg', role: 'group', 'aria-label': 'Wind direction' }, h('span', {}, 'Wind'));
    const row = h('div', {});
    const wbtns = WIND.map(([n, d]) => {
      const b = h('button', { type: 'button', 'aria-pressed': String(d === st.wind) }, n);
      b.title = d == null ? 'the airport station record' : `${n.toLowerCase()}wind at the speed below, relative to the car’s heading every second`;
      b.addEventListener('click', () => { st.wind = d; wbtns.forEach((x, k) => x.setAttribute('aria-pressed', String(WIND[k][1] === d))); syncWind(); go(); });
      row.append(b);
      return b;
    });
    seg.append(row);
    ui.controls.append(seg);
    sW = slider(ui.controls, {
      label: 'Wind speed', min: 0, max: 15, step: 0.5, value: st.ws,
      fmt: (v) => (st.wind == null ? `station ${stationWind().toFixed(1)} m/s` : `${v.toFixed(1)} m/s`),
      hint: 'replaces the station wind when a direction is picked',
      onInput: (v) => { st.ws = v; if (st.wind != null) go(); },
    });
    sS = slider(ui.controls, {
      label: 'Start charge', min: 0, max: 1, step: 0.05, value: 0,
      fmt: (v) => `${Math.round(v * 100)} %${st.soc == null ? ' as driven' : ''}`,
      hint: 'battery charge when the trip starts (plug-ins only)',
      onInput: (v) => { st.soc = v; sS!.set(v); go(); },
    });
    const btns = h('div', { class: 'en-btns' });
    ui.controls.append(btns);
    warmBtn = button(btns, 'Start: warm', () => {
      st.cold = !st.cold;
      warmBtn!.textContent = `Start: ${st.cold ? 'cold' : 'warm'}`;
      warmBtn!.setAttribute('aria-pressed', String(st.cold));
      go();
    });
    warmBtn.title = 'cold: engine, cabin and battery start at air temperature';
    warmBtn.setAttribute('aria-pressed', 'false');
    button(btns, 'As driven', reset);
  }
  function stationWind() { const w = tripOf(X, st.i).wind_speed; let s = 0; for (let k = 0; k < w.length; k++) s += w[k]; return s / w.length; }
  function syncWind() { if (!sW) return; sW.input.disabled = st.wind == null; sW.set(st.ws); }
  function syncSoc() {
    if (!sS) return;
    const v = e0().vehicle;
    (sS.input.closest('.inst-slider') as HTMLElement).style.display = plugIn(v) ? '' : 'none';
    sS.set(st.soc ?? v.soc_start ?? 0);
  }
  function reset() {
    Object.assign(st, { dT: 0, mass: 1, grade: 0, wind: null, soc: null, cold: false });
    sT.set(0); sM.set(1); sG.set(0);
    ui.controls.querySelectorAll('.en-seg button').forEach((b, k) => b.setAttribute('aria-pressed', String(k === 0)));
    if (warmBtn) { warmBtn.textContent = 'Start: warm'; warmBtn.setAttribute('aria-pressed', 'false'); }
    syncWind(); syncSoc(); go();
  }
  syncWind(); syncSoc();

  const gaugeBox = h('div', {}), compBox = h('div', { class: 'en-block' }), kvBox = h('div', { class: 'en-kv' }), note = h('p', { class: 'en-note' });
  ui.readout.append(gaugeBox, compBox, kvBox, note);

  function settings(dT = st.dT): Perturb {
    return { dT, mass: st.mass, grade: st.grade, wind: st.wind == null ? null : [st.ws, st.wind], soc: st.soc, thermal: st.cold ? 'cold' : null };
  }
  function runOnce(dT: number) {
    const e = e0();
    const { trip, veh } = perturb(tripOf(X, st.i), e.vehicle, settings(dT));
    return predict(W, trip, veh);
  }

  async function run() {
    ui.setBusy(true);
    await new Promise((r) => requestAnimationFrame(r));
    const e = e0(), u = unitOf(e.vehicle);
    const t0 = performance.now();
    const r = runOnce(st.dT);
    const ms = performance.now() - t0;
    const t1 = performance.now();
    const sweep: typeof view.sweep = [];
    for (let T = T_LO; T <= T_HI; T += 5) {
      const q = runOnce(T - e.T_mean);
      sweep.push({ T, v: q.energy_MJ * u.f, lo: q.energy_MJ * Math.exp(-q.sigma_log) * u.f, hi: q.energy_MJ * Math.exp(q.sigma_log) * u.f });
    }
    const msSweep = performance.now() - t1;
    view = { r, sweep };
    plot.redraw();
    render(r, e);
    ui.stamp(ms, `${e.id} · temperature curve: ${sweep.length} more runs in ${Math.round(msSweep)} ms`);
    ui.setBusy(false);
  }
  const go = scheduler(run, 30);

  function render(r: Prediction, e: EExample) {
    const u = unitOf(e.vehicle), d = u.d;
    const truth = e.measured_MJ * u.f;
    const top = Math.max(2 * truth, 1.3 * e.python.energy_MJ * Math.exp(e.python.sigma_log) * u.f); // fixed per trip
    const lo = r.energy_MJ * Math.exp(-r.sigma_log) * u.f, hi = r.energy_MJ * Math.exp(r.sigma_log) * u.f;
    gaugeBox.innerHTML = '';
    gauges(gaugeBox, [
      { label: 'Trip energy, EnergyNet', unit: u.u, value: r.energy_MJ * u.f, lo, hi, truth, truthLabel: 'measured', range: [0, top], digits: d },
      { label: 'Road-load physics alone', unit: u.u, value: r.physics_estimate_MJ * u.f, truth, truthLabel: 'measured', range: [0, top], digits: d },
    ]);
    const c = r.components_MJ, cmax = Math.max(...c);
    compBox.innerHTML = `<span class="ig-l">Where it goes (component heads, MJ)</span>` + COMP.map(([n, col], k) =>
      `<div class="en-row"><span class="n">${n}</span><span class="en-trk" style="--k:${col || ctx.accent}"><span style="width:${(100 * c[k] / cmax).toFixed(1)}%"></span></span><span class="v">${c[k].toFixed(2)}</span></div>`).join('');
    const per100 = (r.energy_MJ * u.f) / r.distance_km * 100;
    const b = e.baselines_MJ;
    kvBox.innerHTML = [
      ['Per 100 km', `${per100.toFixed(per100 < 10 ? 2 : 1)} ${u.u.split(' ')[0]}`],
      ['Energy', `${r.energy_MJ.toFixed(2)} MJ (measured ${e.measured_MJ.toFixed(2)})`],
      ['Window sticker × distance', `${(b.label * u.f).toFixed(d)} ${u.u.split(' ')[0]}`],
      ['MOVES-like baseline', `${(b.moves_like * u.f).toFixed(d)} ${u.u.split(' ')[0]}`],
    ].map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join('');
    const changed: string[] = [];
    if (st.dT) changed.push(`air ${st.dT > 0 ? '+' : ''}${st.dT} °C`);
    if (st.mass !== 1) changed.push(`mass ×${st.mass.toFixed(2)}`);
    if (st.grade) changed.push(`grade ${st.grade > 0 ? '+' : ''}${(st.grade * 100).toFixed(2)} %`);
    if (st.wind != null) changed.push(`${WIND.find((w) => w[1] === st.wind)![0].toLowerCase()}wind ${st.ws} m/s`);
    if (st.soc != null) changed.push(`start charge ${Math.round(st.soc * 100)} %`);
    if (st.cold) changed.push('cold start');
    const v = e.vehicle;
    const car = `${v.class}, ${v.architecture.replace(/_/g, ' ')}${v.displacement_L ? `, ${v.displacement_L} L engine` : ''}${v.battery_kWh ? `, ${v.battery_kWh} kWh battery` : ''}`;
    note.innerHTML = changed.length
      ? `<b>Changed: ${changed.join(', ')}.</b> The measured value is for the trip as driven; this is now a different trip, so read the change, not the error.`
      : `<b>As driven:</b> ${e.km.toFixed(1)} km in ${Math.round(e.n / 60)} min, ${e.T_mean.toFixed(1)} °C, ${Math.round(e.climb_m)} m of climb. ${car}. The network knew the class, powertrain, engine size, mass rounded to 500 lb and the window-sticker label${v.obd_maf ? '; measured fuel is the car’s OBD estimate' : ''}. Bars and band: 68% range after the dev widening factor.`;
  }

  function draw(c: Canvas2D) {
    const { ctx: g, w, h: H } = c;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, w, H);
    const r = view.r;
    if (!r) return;
    const e = e0(), u = unitOf(e.vehicle);
    const padL = 40, padR = 12, top = 16, gap = 30, bot = 20;
    const iw = w - padL - padR, ih = H - top - gap - bot;
    const h1 = Math.round(ih * 0.46), h2 = ih - h1, y2 = top + h1 + gap;
    // --- route: speed and the elevation implied by the grade (with any extra grade)
    const t = tripOf(X, st.i), N = t.v.length;
    let vmax = 1;
    for (let k = 0; k < N; k++) vmax = Math.max(vmax, t.v[k]);
    const z = new Float64Array(N);
    let acc = 0, zmin = 0, zmax = 0;
    for (let k = 0; k < N; k++) { acc += t.v[k] * (t.grade[k] + st.grade); z[k] = acc; zmin = Math.min(zmin, acc); zmax = Math.max(zmax, acc); }
    gridLines(c, padL, top, iw, h1, 6, 2);
    const xs = (k: number) => padL + (k / Math.max(N - 1, 1)) * iw;
    g.beginPath();
    for (let k = 0; k < N; k++) { const y = top + h1 - ((z[k] - zmin) / (zmax - zmin || 1)) * h1 * 0.5; k ? g.lineTo(xs(k), y) : g.moveTo(xs(k), y); }
    g.lineTo(xs(N - 1), top + h1); g.lineTo(xs(0), top + h1); g.closePath();
    g.fillStyle = 'rgba(235,230,216,0.10)'; g.fill();
    g.strokeStyle = ctx.accent; g.lineWidth = 1.3; g.beginPath();
    for (let k = 0; k < N; k++) { const y = top + h1 - (t.v[k] / (vmax * 1.08)) * h1; k ? g.lineTo(xs(k), y) : g.moveTo(xs(k), y); }
    g.stroke();
    const small = w < 520;
    label(c, `speed, top ${Math.round(vmax * 3.6)} km/h`, padL + 4, top + 6, ctx.accent, 'left', 9.5);
    label(c, small ? `elevation, ${Math.round(zmax - zmin)} m` : `shaded: elevation, ${Math.round(zmax - zmin)} m range`, padL + iw - 2, top + 6, SCOPE.text, 'right', 9.5);
    const mins = N / 60, stepM = mins > 12 ? 5 : 2;
    for (let m = 0; m <= mins; m += stepM) label(c, `${m}m`, xs(m * 60), top + h1 + 9, SCOPE.faint, m === 0 ? 'left' : 'center', 9);
    // --- the same trip at other air temperatures
    const sw = view.sweep;
    const truth = e.measured_MJ * u.f;
    let lo = Math.min(truth, ...sw.map((p) => p.lo)), hi = Math.max(truth, ...sw.map((p) => p.hi));
    const span = hi - lo || 1; lo = Math.max(0, lo - span * 0.08); hi += span * 0.08;
    const tx = (T: number) => padL + ((T - T_LO) / (T_HI - T_LO)) * iw;
    const ty = (v: number) => y2 + h2 - ((v - lo) / (hi - lo)) * h2;
    gridLines(c, padL, y2, iw, h2, 7, 3);
    label(c, small ? `${u.u} vs air temperature` : `energy (${u.u}) vs air temperature, other settings as set`, padL + 4, y2 - 8, SCOPE.text, 'left', 9.5);
    for (let j = 0; j <= 3; j++) { const v = lo + ((hi - lo) * j) / 3; label(c, v.toFixed(v < 10 ? 2 : 1), padL - 4, ty(v), SCOPE.faint, 'right', 9); }
    for (let T = -20; T <= 40; T += 10) label(c, `${T}°`, tx(T), y2 + h2 + 10, SCOPE.faint, 'center', 9);
    g.beginPath();
    sw.forEach((p, k) => (k ? g.lineTo(tx(p.T), ty(p.hi)) : g.moveTo(tx(p.T), ty(p.hi))));
    for (let k = sw.length - 1; k >= 0; k--) g.lineTo(tx(sw[k].T), ty(sw[k].lo));
    g.closePath(); g.globalAlpha = 0.18; g.fillStyle = ctx.accent; g.fill(); g.globalAlpha = 1;
    g.strokeStyle = ctx.accent; g.lineWidth = 1.6; g.beginPath();
    sw.forEach((p, k) => (k ? g.lineTo(tx(p.T), ty(p.v)) : g.moveTo(tx(p.T), ty(p.v))));
    g.stroke();
    // truth: measured, at the temperature it was driven
    const Tr = e.T_mean;
    g.strokeStyle = SCOPE.ink; g.lineWidth = 1.5; g.beginPath(); g.arc(tx(Tr), ty(truth), 4.5, 0, 2 * Math.PI); g.stroke();
    label(c, 'measured', tx(Tr) + (tx(Tr) > padL + iw * 0.75 ? -8 : 8), ty(truth) + (ty(truth) < y2 + 12 ? 9 : -9), SCOPE.ink, tx(Tr) > padL + iw * 0.75 ? 'right' : 'left', 9);
    // the current reading
    const Tc = e.T_mean + st.dT, vc = r.energy_MJ * u.f;
    if (Tc >= T_LO && Tc <= T_HI) {
      g.fillStyle = ctx.accent; g.beginPath(); g.arc(tx(Tc), ty(vc), 4, 0, 2 * Math.PI); g.fill();
      g.strokeStyle = SCOPE.bg; g.lineWidth = 1; g.stroke();
    }
  }

  await run();
}
