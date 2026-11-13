// HybridPilot live instrument. Three real VED exam routes, each driven by RoadSim's simulated twin of the car that
// drove it. The trips (every strategy against the DP optimum) are precomputed in Python, because RoadSim is Python;
// the network and the decision rule run here, live: pick a moment of the trip, change the battery charge, the wheel
// power the driver asks for, the coolant temperature or the navigation preview, and watch s and the chosen engine
// power respond.
import type { InstrumentContext } from '../types.ts';
import { fetchPack } from '../../lib/nn/pack.ts';
import { loadNet, decide, cands, features, forward, sOf, equivalence, tripS, type HMeta, type WMeta, type Decision } from './net.ts';
import { shell, picker, slider, button, scheduler, gauges, probs, canvas, gridLines, label, fmt, h, SCOPE, hex, type Canvas2D } from '../kit.ts';

const STYLE = `
.hp-box { display: flex; flex-direction: column; gap: 6px; }
.hp-pick { font-family: var(--mono); font-variation-settings: 'wdth' 87.5; display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.hp-pick b { font-weight: 400; font-size: 20px; color: var(--scr-ink); letter-spacing: -0.03em; }
.hp-pick em { font-style: normal; font-size: 11px; color: var(--scr-2ink); }
.hp-sm { font-family: var(--mono); font-variation-settings: 'wdth' 87.5; font-size: 10px; line-height: 1.5; color: var(--scr-3ink); }
.hp-sm b { font-weight: 400; color: var(--scr-2ink); }
.hp-sm .w { color: #ffb36b; }
.hp-tbl { width: 100%; border-collapse: collapse; font-family: var(--mono); font-variation-settings: 'wdth' 87.5; font-size: 10.5px; }
.hp-tbl th { font-weight: 400; text-align: right; color: var(--scr-3ink); padding: 3px 4px; border-bottom: 1px solid var(--scr-rule); white-space: nowrap; }
.hp-tbl td { text-align: right; padding: 3px 4px; color: var(--scr-2ink); white-space: nowrap; }
.hp-tbl th:first-child, .hp-tbl td:first-child { text-align: left; white-space: normal; }
.hp-tbl tr.hp td { color: var(--scr-ink); }
.hp-tbl tr.hp td:first-child { color: var(--c); }
.hp-tbl tr.sub td { font-size: 9.5px; color: var(--scr-3ink); }
.hp-note { font-size: 11.5px; line-height: 1.45; color: var(--scr-3ink); margin: 0; }
.hp-note b { font-weight: 500; color: var(--scr-2ink); }
.hp-cv { cursor: crosshair; }
.hp-gap { height: 10px; }
`;

const NAME: Record<string, string> = { rule: 'Rule-based', ecms: 'ECMS', aecms: 'A-ECMS', mpc: 'MPC', hybridpilot: 'HybridPilot', dp: 'DP optimum' };
const ROWS = ['dp', 'hybridpilot', 'aecms', 'mpc', 'ecms', 'rule'];
const RULE: Record<string, string> = { hev_powersplit: 'Prius-style rule', phev_series: 'deplete, then hold', hev_series: 'thermostat', hev_parallel_p2: 'electric assist', phev_powersplit: 'deplete, then hold' };
const ARCH: Record<string, string> = { hev_powersplit: 'power-split hybrid', phev_series: 'series plug-in hybrid' };
const PREV = [0, 5, 10, 20, 30, 60];
const AMBER = '#d9a441';

function lighten(c: string, f: number) {
  const [r, g, b] = hex(c);
  const m = (v: number) => Math.round(v + (255 - v) * f);
  return `rgb(${m(r)},${m(g)},${m(b)})`;
}
const mmss = (k: number) => `${Math.floor(k / 60)}:${String(k % 60).padStart(2, '0')}`;
const degC = (t: number) => `${Math.round(t) || 0} °C`;
const kW = (w: number, d = 1) => `${(w / 1000).toFixed(d)} kW`;

export async function mount(ctx: InstrumentContext) {
  const [WP, X] = await Promise.all([fetchPack<WMeta>(`${ctx.base}weights`), fetchPack<HMeta>(`${ctx.base}examples`)]);
  const net = loadNet(WP);
  const M = X.meta, trips = M.trips;
  if (!document.getElementById('hp-inst-style')) document.head.append(h('style', { id: 'hp-inst-style' }, STYLE));
  const sig = lighten(ctx.accent, 0.35);

  const ui = shell(ctx.root, {
    title: 'HybridPilot',
    subtitle: 'simulated twins of three real hybrids, on the real exam routes they drove — never used in training',
    params: `${WP.meta.params.toLocaleString('en-US')} parameters · matches Python to 4e-15`,
  });

  const st = { ti: 0, mi: 0, soc: 0.6, d: 0, t: 0, L: 0, temp: 0, mass: 1, cold: false };
  const tripCache: Record<number, ReturnType<typeof tripS>> = {};
  let dec: Decision | null = null;
  let sweep: { soc: number; sNet: number; s: number }[] = [];
  const narrow = () => ui.screen.getBoundingClientRect().width < 520;

  const trip = () => trips[st.ti];
  const mom = () => trip().moments[st.mi];
  const variant = () => (st.cold ? 'cold' : st.temp > 0 ? `T${M.temps[st.temp - 1]}` : st.mass !== 1 ? `m${M.masses.find((m) => m === st.mass)}` : 'base');
  const runIdx = (tag: string, s: string) => trip().runs.findIndex((r) => r.v === tag && r.s === s);
  const row = (name: string, r: number) => { const t = X.t[`t${st.ti}_${name}`]; const N = t.shape[1]; return t.data.subarray(r * N, (r + 1) * N); };

  picker(ui.picker, trips.map((t, i) => ({ label: `${t.short} · ${t.vehicle}`, sub: `${t.dist_km.toFixed(1)} km · ${Math.round(t.N / 60)} min · ${degC(t.T_route)}`, value: i })),
    (i) => { st.ti = i; st.temp = 0; st.mass = 1; st.cold = false; syncTrip(); setMoment(Math.floor(trip().moments.length / 2)); }, st.ti);

  // ---- screens: the trip (precomputed runs + the network's s, live) and the decision at one second (live)
  const tripBox = h('div', {}), decBox = h('div', {});
  ui.screen.append(tripBox, decBox);
  const plotTrip = canvas(tripBox, ctx.compact ? 2.1 : narrow() ? 1.05 : 2.0, (c) => drawTrip(c));
  const plotDec = canvas(decBox, ctx.compact ? 2.4 : narrow() ? 1.0 : 2.5, (c) => drawDec(c));
  const tripCv = tripBox.querySelector('canvas')!;
  tripCv.classList.add('hp-cv');
  tripCv.addEventListener('click', (ev) => {
    const r = tripCv.getBoundingClientRect();
    const { padL, iw } = tripGeom(r.width);
    const k = ((ev.clientX - r.left - padL) / iw) * (trip().N - 1);
    let best = 0;
    trip().moments.forEach((m, i) => { if (Math.abs(m.k - k) < Math.abs(trip().moments[best].k - k)) best = i; });
    setMoment(best);
  });

  // ---- controls
  const sMom = slider(ui.controls, { label: 'Moment', min: 0, max: trip().moments.length - 1, step: 1, value: 0,
    fmt: (v) => { const m = trip().moments[v]; return m ? `${mmss(m.k)} · ${m.v_kmh.toFixed(0)} km/h` : ''; },
    hint: 'or click the trip; seconds HybridPilot drove in the recorded conditions', onInput: (v) => setMoment(v) });
  const socLo = () => trip().soc_lo - 0.03, socSpan = () => trip().soc_hi - socLo();
  const sSoc = slider(ui.controls, { label: 'Battery charge', min: 0, max: 1, step: 0.0025, value: 0.5,
    fmt: (v) => `${(100 * (socLo() + v * socSpan())).toFixed(1)} %`,
    hint: 'window ' + '', onInput: (v) => { st.soc = socLo() + v * socSpan(); go(); } });
  const socHint = sSoc.input.parentElement!.querySelector('small')!;
  const sDem = slider(ui.controls, { label: 'Wheel power', min: 0, max: 6, step: 1, value: 0,
    fmt: (v) => { const m = mom(); return m && m.D[v] != null ? `${kW(m.D[v])}${v === m.d_rec ? ', as driven' : ''}` : ''; },
    hint: 'same speed, harder or softer on the pedal', onInput: (v) => { st.d = v; go(); } });
  const sCool = slider(ui.controls, { label: 'Coolant', min: 0, max: 6, step: 1, value: 0,
    fmt: (v) => { const m = mom(); return m && m.T[v] != null ? `${m.T[v].toFixed(0)} °C${v === m.t_rec ? ', as driven' : ''}` : ''; },
    hint: 'below 35 °C the catalyst must light off; below 55 °C the cabin wants engine heat', onInput: (v) => { st.t = v; go(); } });
  let sPrev: ReturnType<typeof slider> | null = null;
  if (!ctx.compact) {
    sPrev = slider(ui.controls, { label: 'Preview', min: 0, max: PREV.length - 1, step: 1, value: 0,
      fmt: (v) => (PREV[v] === 0 ? 'none (as shipped)' : `next ${PREV[v]} s`),
      hint: 'navigation: speed and grade ahead, in 5 bins; the released model drives without', onInput: (v) => { st.L = PREV[v]; go(); } });
  }
  let sTemp: ReturnType<typeof slider> | null = null, sMass: ReturnType<typeof slider> | null = null, bCold: HTMLButtonElement | null = null;
  if (!ctx.compact) {
    sTemp = slider(ui.controls, { label: 'Outside air', min: 0, max: M.temps.length, step: 1, value: 0,
      fmt: (v) => (v === 0 ? `${degC(trip().T_route)}, recorded` : `${M.temps[v - 1]} °C`),
      hint: 'whole trip, every strategy, re-run in RoadSim (precomputed)', onInput: (v) => { st.temp = v; if (v) { st.mass = 1; st.cold = false; } syncTrip(); go(); } });
    const MS = [M.masses[0], 1, ...M.masses.slice(1)];
    sMass = slider(ui.controls, { label: 'Vehicle mass', min: 0, max: MS.length - 1, step: 1, value: 1,
      fmt: (v) => (MS[v] === 1 ? `${Math.round(trip().mass_kg)} kg, as built` : `×${MS[v]} (${Math.round(trip().mass_kg * MS[v])} kg)`),
      hint: 'payload added or removed; trip re-run (precomputed)', onInput: (v) => { st.mass = MS[v]; if (MS[v] !== 1) { st.temp = 0; st.cold = false; } syncTrip(); go(); } });
    bCold = button(ui.controls, 'Trip: cold start', () => { st.cold = !st.cold; if (st.cold) { st.temp = 0; st.mass = 1; } syncTrip(); go(); });
  }

  function syncTrip() {
    sMom.input.max = String(trip().moments.length - 1);
    sTemp?.set(st.temp);
    if (sMass) sMass.set([M.masses[0], 1, ...M.masses.slice(1)].indexOf(st.mass));
    if (bCold) bCold.textContent = st.cold ? 'Trip: warm start' : 'Trip: cold start';
    socHint.textContent = `window ${(100 * trip().soc_lo).toFixed(0)}–${(100 * trip().soc_hi).toFixed(0)} %, target ${(100 * trip().ref).toFixed(0)} %`;
  }
  function setMoment(i: number) {
    st.mi = i;
    const m = mom();
    st.soc = m.soc; st.d = m.d_rec; st.t = m.t_rec; st.L = 0;
    sMom.set(i);
    sSoc.set((m.soc - socLo()) / socSpan());
    sDem.input.max = String(m.D.length - 1); sDem.set(st.d);
    sCool.input.max = String(m.T.length - 1); sCool.set(st.t);
    sPrev?.set(0);
    go();
  }

  // ---- the live part
  async function run() {
    ui.setBusy(true);
    await new Promise((r) => requestAnimationFrame(r));
    const T = trip(), m = mom();
    const t0 = performance.now();
    let n = 0;
    if (!tripCache[st.ti]) { tripCache[st.ti] = tripS(net, M, T, X, st.ti); n += tripCache[st.ti].k.length; }
    const set = { soc: st.soc, d: st.d, t: st.t, L: st.L };
    dec = decide(net, M, T, m, cands(X, st.ti, st.mi, st.d, st.t), set);
    n++;
    sweep = [];
    for (let i = 0; i <= 40; i++) {
      const soc = socLo() + (socSpan() * i) / 40;
      const sN = sOf(forward(net, features(M, T, m, { ...set, soc }).x));
      sweep.push({ soc, sNet: sN, s: equivalence(M.policy, T, sN, soc) });
    }
    n += 41;
    const ms = performance.now() - t0;
    plotTrip.redraw(); plotDec.redraw();
    readout();
    ui.stamp(ms, `${n} network runs · ${T.id}: real route, simulated car`);
    ui.setBusy(false);
  }
  const go = scheduler(run, 25);

  // ---- readout
  const decEl = h('div', { class: 'hp-box' }), tblEl = h('div', { class: 'hp-box' }), noteEl = h('p', { class: 'hp-note' });
  function readout() {
    if (!dec) return;
    const T = trip(), m = mom(), d = dec;
    gauges(ui.readout, [{ label: 'Equivalence factor s (fuel J per battery J)', value: d.s, range: [0, 6], digits: 2 }]);
    const rDP = runIdx('base', 'dp');
    const dpOn = rDP >= 0 ? row('on', rDP)[m.k] > 0.5 : null;
    probs(ui.readout, 'Network: engine on? (not used by the shipped rule)', [
      { label: 'on', p: d.pon, truth: dpOn === true }, { label: 'off', p: 1 - d.pon, truth: dpOn === false }]);
    const atRec = st.d === m.d_rec && st.t === m.t_rec && Math.abs(st.soc - m.soc) < 1e-9 && st.L === 0;
    const P = d.j >= 0 ? T.ctrl[d.j] : -1;
    const what = d.j < 0 ? 'no feasible control' : d.on[d.j] ? `engine ${kW(P, 0)}` : 'engine off';
    const why: string[] = [];
    if (d.forced) why.push(d.cat ? 'catalyst warm-up forces the engine on' : d.need_heat ? 'cabin heat forces the engine on' : 'charge below the window forces the engine on');
    else if (m.eng_on && m.t_on < M.policy.min_on_s) why.push(`engine started ${m.t_on.toFixed(0)} s ago: kept on (min ${M.policy.min_on_s} s)`);
    else if (!m.eng_on && m.t_off < M.policy.min_off_s) why.push(`engine stopped ${m.t_off.toFixed(0)} s ago: kept off (min ${M.policy.min_off_s} s)`);
    const nOk = d.ok.filter(Boolean).length;
    decEl.innerHTML = `<div class="ig-l">HybridPilot's choice this second</div>
      <div class="hp-pick"><b>${what}</b>${d.j >= 0 && d.on[d.j] ? '<em>to the generator / wheels</em>' : '<em>electric only</em>'}</div>
      <div class="hp-sm">s = network <b>${fmt(d.sNet, 2)}</b> ${d.s >= d.sNet ? '+' : '−'} SOC feedback <b>${fmt(Math.abs(d.s - d.sNet), 2)}</b> · ${nOk} of ${d.ok.length} candidates possible</div>
      ${why.length ? `<div class="hp-sm w">${why.join(' · ')}</div>` : ''}
      <div class="hp-sm">${atRec ? `RoadSim recorded: <b>${m.j_rec < 0 ? '—' : T.ctrl[m.j_rec] > 0 ? kW(T.ctrl[m.j_rec], 0) : 'engine off'}</b>${m.j_rec === d.j ? ' (same)' : ''}` : 'changed from the recorded second'}${dpOn != null ? ` · DP had the engine <b>${dpOn ? 'on' : 'off'}</b>` : ''}</div>`;
    const V = variant();
    const rows = ROWS.map((s) => ({ s, r: T.runs[runIdx(V, s)] })).filter((x) => x.r);
    const extra = V === 'base' ? M.previews.map((p) => ({ s: `hybridpilot_p${p}`, r: T.runs[runIdx('base', `hybridpilot_p${p}`)], p })).filter((x) => x.r) : [];
    const ekwh = (e: number) => (Math.abs(e) < 5e-4 ? '0.000' : `${e < 0 ? '−' : ''}${Math.abs(e).toFixed(3)}`);
    const pct = (r: { pct_dp: number; extrap: boolean }) => `${(100 * r.pct_dp).toFixed(2)}${r.extrap ? '*' : ''}`;
    const vlabel = V === 'base' ? 'as recorded' : V === 'cold' ? 'cold start' : V.startsWith('T') ? `${V.slice(1)} °C outside` : `mass ×${V.slice(1)}`;
    tblEl.innerHTML = `<div class="ig-l">Whole trip, ${vlabel} · % of DP at matched charge</div>
      <table class="hp-tbl"><thead><tr><th>Strategy</th><th>fuel g</th><th>battery kWh</th><th>% of DP</th></tr></thead><tbody>
      ${rows.map(({ s, r }) => `<tr class="${s === 'hybridpilot' ? 'hp' : ''}"><td>${s === 'rule' ? `Rule (${RULE[T.arch] ?? 'rule'})` : NAME[s]}</td><td>${r.fuel_g.toFixed(0)}</td><td>${ekwh(r.E_kWh)}</td><td>${pct(r)}</td></tr>`).join('')}
      ${extra.map(({ r, p }) => `<tr class="sub"><td>HybridPilot + ${p} s preview</td><td>${r.fuel_g.toFixed(0)}</td><td>${ekwh(r.E_kWh)}</td><td>${pct(r)}</td></tr>`).join('')}
      </tbody></table>`;
    const base = T.runs[runIdx('base', 'hybridpilot')];
    const ratio = base ? base.fuel_g / T.logged_fuel_g : NaN;
    noteEl.innerHTML = `<b>Simulated car, real route.</b> ${T.likely ? `A RoadSim twin of a ${ARCH[T.arch] ?? T.arch} (likely ${T.likely.replace(/\s*\([^)]*\)/g, '')}), ` : ''}`
      + `driving the speed, grade and weather VED recorded. `
      + (T.logged_fuel_g >= 1 ? `The real car logged <b>${T.logged_fuel_g.toFixed(0)} g</b> of fuel; the twin under HybridPilot burns <b>${base ? base.fuel_g.toFixed(0) : '—'} g</b> (${(100 * ratio).toFixed(0)}%, starting warm). `
        : `The real car logged no fuel on this trip; the twin under HybridPilot burns <b>${base ? base.fuel_g.toFixed(0) : '—'} g</b>. `)
      + `Across 252 exam trips RoadSim reads about 20% low, far more than the 1–3% between strategies, so this ranking holds only inside the simulator. `
      + `Battery kWh: chemical energy taken from the battery (− = charged). * = DP extrapolated; above 100% means DP was not tight.`;
    ui.readout.append(decEl, tblEl, noteEl); // (re-appended: gauges() and probs() add theirs at the end)
  }

  // ---- drawing: the trip
  function tripGeom(w: number) {
    const padL = w < 520 ? 50 : 74, padR = 8;
    return { padL, padR, iw: w - padL - padR };
  }
  function drawTrip(c: Canvas2D) {
    const { ctx: g, w, h: H } = c;
    g.fillStyle = SCOPE.bg; g.fillRect(0, 0, w, H);
    const T = trip(), N = T.N, V = variant();
    const { padL, iw } = tripGeom(w);
    const top = 8, bot = 16, gap = 8;
    const avail = H - top - bot - 3 * gap;
    const hSp = avail * 0.2, hBar = avail * 0.3, hSoc = avail * 0.3, hS = avail * 0.2;
    const X = (k: number) => padL + (k / (N - 1)) * iw;
    const fs = w < 520 ? 8.5 : 9.5;
    // speed
    let y0 = top;
    const speed = Array.from(XT('v'), (q) => q * 3.6);
    const vmax = Math.max(40, Math.ceil(Math.max(...speed) / 20) * 20);
    gridLines(c, padL, y0, iw, hSp, 8, 2);
    g.beginPath(); g.moveTo(X(0), y0 + hSp);
    speed.forEach((q, k) => g.lineTo(X(k), y0 + hSp - (q / vmax) * hSp));
    g.lineTo(X(N - 1), y0 + hSp); g.closePath();
    g.fillStyle = `rgba(${hex(ctx.accent).join(',')},0.22)`; g.fill();
    g.strokeStyle = SCOPE.ink; g.globalAlpha = 0.7; g.lineWidth = 1;
    g.beginPath(); speed.forEach((q, k) => (k ? g.lineTo(X(k), y0 + hSp - (q / vmax) * hSp) : g.moveTo(X(k), y0 + hSp - (q / vmax) * hSp))); g.stroke();
    g.globalAlpha = 1;
    label(c, 'speed', 4, y0 + 6, SCOPE.text, 'left', fs);
    label(c, `${vmax} km/h`, 4, y0 + 17, SCOPE.faint, 'left', fs - 1);
    // moment ticks
    T.moments.forEach((m, i) => {
      const x = X(m.k);
      g.fillStyle = i === st.mi ? sig : SCOPE.faint;
      g.beginPath(); g.moveTo(x, y0 + hSp - 5); g.lineTo(x - 3.5, y0 + hSp + 1); g.lineTo(x + 3.5, y0 + hSp + 1); g.closePath(); g.fill();
    });
    y0 += hSp + gap;
    // engine on/off barcodes
    const rowH = hBar / ROWS.length;
    const order = ['rule', 'ecms', 'aecms', 'mpc', 'hybridpilot', 'dp'];
    order.forEach((s, i) => {
      const r = runIdx(V, s);
      const yy = y0 + i * rowH;
      const col = s === 'hybridpilot' ? sig : s === 'dp' ? SCOPE.ink : s === 'aecms' ? AMBER : SCOPE.text;
      label(c, s === 'dp' ? 'DP' : s === 'hybridpilot' ? (w < 520 ? 'HP' : 'HybridPilot') : w < 520 && s === 'rule' ? 'Rule' : NAME[s], 4, yy + rowH / 2, s === 'hybridpilot' ? sig : SCOPE.text, 'left', fs);
      g.fillStyle = 'rgba(235,230,216,0.04)'; g.fillRect(padL, yy + 1, iw, rowH - 2);
      if (r < 0) return;
      const on = row('on', r);
      g.fillStyle = col;
      for (let k = 0; k < N; k++) {
        if (on[k] < 0.5) continue;
        let e = k; while (e + 1 < N && on[e + 1] > 0.5) e++;
        g.fillRect(X(k), yy + 2, Math.max(X(e + 1) - X(k), 0.8), rowH - 4);
        k = e;
      }
    });
    label(c, 'engine on', padL + iw - 2, y0 - 4, SCOPE.faint, 'right', fs - 1);
    y0 += hBar + gap;
    // state of charge
    let lo = T.soc_lo, hi = T.soc_hi;
    const socRows = order.map((s) => runIdx(V, s));
    socRows.forEach((r) => { if (r < 0) return; const q = row('soc', r); for (let k = 0; k < N; k++) { lo = Math.min(lo, q[k]); hi = Math.max(hi, q[k]); } });
    if (!T.phev) { lo = Math.max(lo - 0.02, 0); hi = Math.min(hi + 0.02, 1); }
    const Ys = (q: number) => y0 + hSoc - ((q - lo) / (hi - lo)) * hSoc;
    gridLines(c, padL, y0, iw, hSoc, 8, 2);
    g.setLineDash([3, 3]); g.strokeStyle = SCOPE.faint;
    [T.soc_lo, T.soc_hi].forEach((b) => { if (b > lo && b < hi) { g.beginPath(); g.moveTo(padL, Ys(b)); g.lineTo(padL + iw, Ys(b)); g.stroke(); } });
    g.setLineDash([]);
    order.forEach((s, i) => {
      const r = socRows[i]; if (r < 0) return;
      const q = row('soc', r);
      g.strokeStyle = s === 'hybridpilot' ? sig : s === 'dp' ? SCOPE.ink : s === 'aecms' ? AMBER : SCOPE.faint;
      g.lineWidth = s === 'hybridpilot' ? 2 : 1.2;
      g.setLineDash(s === 'dp' ? [5, 3] : []);
      g.beginPath(); for (let k = 0; k < N; k++) (k ? g.lineTo(X(k), Ys(q[k])) : g.moveTo(X(k), Ys(q[k]))); g.stroke();
    });
    g.setLineDash([]); g.lineWidth = 1;
    label(c, 'charge', 4, y0 + 6, SCOPE.text, 'left', fs);
    label(c, `${(100 * hi).toFixed(0)}%`, 4, y0 + 17, SCOPE.faint, 'left', fs - 1);
    label(c, `${(100 * lo).toFixed(0)}%`, 4, y0 + hSoc - 4, SCOPE.faint, 'left', fs - 1);
    y0 += hSoc + gap;
    // s over the trip, live
    const S = tripCache[st.ti];
    gridLines(c, padL, y0, iw, hS, 8, 2);
    label(c, 's, live', 4, y0 + 6, sig, 'left', fs);
    if (S) {
      let smax = 4; for (let i = 0; i < S.s.length; i++) smax = Math.max(smax, S.s[i]);
      smax = Math.ceil(smax);
      label(c, `0–${smax}`, 4, y0 + 17, SCOPE.faint, 'left', fs - 1);
      const Y = (s: number) => y0 + hS - (s / smax) * hS;
      g.fillStyle = sig;
      for (let i = 0; i < S.s.length; i++) g.fillRect(X(S.k[i]) - 0.6, Y(S.s[i]) - 0.6, 1.4, 1.4);
    }
    // the selected moment
    const xm = X(mom().k);
    g.strokeStyle = sig; g.globalAlpha = 0.8; g.beginPath(); g.moveTo(xm + 0.5, top); g.lineTo(xm + 0.5, y0 + hS); g.stroke(); g.globalAlpha = 1;
    // time axis
    const step = N > 700 ? 2 : 1;
    for (let mnt = 0; mnt * 60 < N; mnt += step) label(c, `${mnt}m`, X(mnt * 60), H - 7, SCOPE.faint, mnt === 0 ? 'left' : 'center', fs - 0.5);
  }
  function XT(name: string) { return X.t[`t${st.ti}_${name}`].data; }

  // ---- drawing: the decision
  function drawDec(c: Canvas2D) {
    const { ctx: g, w, h: H } = c;
    g.fillStyle = SCOPE.bg; g.fillRect(0, 0, w, H);
    g.strokeStyle = SCOPE.grid; g.beginPath(); g.moveTo(0, 0.5); g.lineTo(w, 0.5); g.stroke();
    if (!dec) return;
    const T = trip(), m = mom(), d = dec;
    const stack = w < 520;
    const fs = stack ? 8.5 : 9.5;
    // panel 1: cost of each candidate
    const p1 = stack ? { x: 40, y: 22, w: w - 50, h: H * 0.5 - 50 } : { x: 46, y: 22, w: w * 0.6 - 60, h: H - 54 };
    const xs = T.ctrl.map((q) => q / 1000);
    const okI = d.J.map((_, i) => i).filter((i) => d.ok[i]);
    const okMax = Math.max(0, ...okI.map((i) => xs[i]));
    const xmax = Math.min(Math.max(...xs), Math.max(okMax * 1.35, xs[Math.min(4, xs.length - 1)]));
    const hidden = xs.filter((q) => q > xmax).length;
    const c0 = cands(X, st.ti, st.mi, st.d, st.t);
    let ymin = Infinity, ymax = -Infinity;
    okI.forEach((i) => { ymin = Math.min(ymin, d.J[i], c0.PF[i]); ymax = Math.max(ymax, d.J[i], c0.PF[i]); });
    if (!okI.length) { ymin = 0; ymax = 1; }
    ymin /= 1000; ymax /= 1000;
    const pad = (ymax - ymin) * 0.12 + 0.5;
    ymin -= pad; ymax += pad;
    const PX = (v: number) => p1.x + (v / xmax) * p1.w, PY = (v: number) => p1.y + p1.h - ((v - ymin) / (ymax - ymin)) * p1.h;
    gridLines(c, p1.x, p1.y, p1.w, p1.h, 6, 4);
    label(c, stack ? 'cost = fuel + s × battery (kW)' : 'cost = fuel + s × battery power (kW), per engine power', p1.x, 10, SCOPE.text, 'left', fs);
    for (let j = 0; j <= 4; j++) { const v = ymin + ((ymax - ymin) * j) / 4; label(c, v.toFixed(0), p1.x - 5, PY(v), SCOPE.faint, 'right', fs - 1); }
    for (let j = 0; j <= 4; j++) { const v = (xmax * j) / 4; label(c, j === 0 ? 'off' : v.toFixed(0), PX(v), p1.y + p1.h + 9, SCOPE.faint, 'center', fs - 1); }
    label(c, hidden ? `engine kW · ${hidden} higher settings not possible` : 'engine kW', p1.x + p1.w, p1.y + p1.h + 20, SCOPE.faint, 'right', fs - 1);
    // fuel alone (dashed) and the cost, over the possible engine-on candidates
    const onOk = okI.filter((i) => d.on[i]);
    g.setLineDash([3, 3]); g.strokeStyle = SCOPE.faint; g.beginPath();
    onOk.forEach((i, n) => (n ? g.lineTo(PX(xs[i]), PY(c0.PF[i] / 1000)) : g.moveTo(PX(xs[i]), PY(c0.PF[i] / 1000)))); g.stroke(); g.setLineDash([]);
    g.strokeStyle = sig; g.lineWidth = 1.6; g.beginPath();
    onOk.forEach((i, n) => (n ? g.lineTo(PX(xs[i]), PY(d.J[i] / 1000)) : g.moveTo(PX(xs[i]), PY(d.J[i] / 1000)))); g.stroke(); g.lineWidth = 1;
    d.J.forEach((J, i) => {
      if (xs[i] > xmax) return;
      if (!d.ok[i]) { label(c, '×', PX(xs[i]), p1.y + p1.h - 5, SCOPE.faint, 'center', fs); return; }
      g.fillStyle = i === d.j ? sig : SCOPE.ink; g.globalAlpha = i === d.j ? 1 : 0.55;
      g.beginPath(); g.arc(PX(xs[i]), PY(J / 1000), i === d.j ? 5 : 2.3, 0, 7); g.fill(); g.globalAlpha = 1;
    });
    const atRec = st.d === m.d_rec && st.t === m.t_rec && Math.abs(st.soc - m.soc) < 1e-9 && st.L === 0;
    if (atRec && m.j_rec >= 0 && d.ok[m.j_rec]) { g.strokeStyle = SCOPE.ink; g.beginPath(); g.arc(PX(xs[m.j_rec]), PY(d.J[m.j_rec] / 1000), 8.5, 0, 7); g.stroke(); }
    if (d.j >= 0) label(c, d.on[d.j] ? `picked ${xs[d.j].toFixed(0)} kW` : 'picked: off', Math.min(PX(xs[d.j]) + 9, p1.x + p1.w - 60), PY(d.J[d.j] / 1000) - 16, sig, 'left', fs);
    label(c, '- - fuel alone   × not possible', p1.x + p1.w, p1.y + 8, SCOPE.faint, 'right', fs - 1);
    // panel 2: s across the charge window
    const p2 = stack ? { x: 40, y: H * 0.5 + 22, w: w - 50, h: H * 0.5 - 50 } : { x: w * 0.6 + 26, y: 22, w: w * 0.4 - 36, h: H - 54 };
    const smax = Math.max(4, Math.ceil(Math.max(...sweep.map((p) => Math.max(p.s, p.sNet)))));
    const lo = sweep[0].soc, hi = sweep[sweep.length - 1].soc;
    const QX = (s: number) => p2.x + ((s - lo) / (hi - lo)) * p2.w, QY = (s: number) => p2.y + p2.h - (s / smax) * p2.h;
    gridLines(c, p2.x, p2.y, p2.w, p2.h, 4, 4);
    label(c, 's across the charge window', p2.x, p2.y - 12, SCOPE.text, 'left', fs);
    for (let j = 0; j <= 4; j++) label(c, ((smax * j) / 4).toFixed(smax % 4 ? 1 : 0), p2.x - 5, QY((smax * j) / 4), SCOPE.faint, 'right', fs - 1);
    [T.soc_lo, T.ref, T.soc_hi].forEach((q, j) => j !== 1 && Math.abs(QX(q) - QX(T.ref)) < 48 ? null : label(c, `${(100 * q).toFixed(0)}%${j === 1 ? ' target' : ''}`, QX(q), p2.y + p2.h + 9, SCOPE.faint, 'center', fs - 1));
    g.setLineDash([2, 3]); g.strokeStyle = SCOPE.faint; g.beginPath(); g.moveTo(QX(T.ref) + 0.5, p2.y); g.lineTo(QX(T.ref) + 0.5, p2.y + p2.h); g.stroke();
    g.setLineDash([4, 3]); g.strokeStyle = SCOPE.text; g.beginPath(); sweep.forEach((p, i) => (i ? g.lineTo(QX(p.soc), QY(p.sNet)) : g.moveTo(QX(p.soc), QY(p.sNet)))); g.stroke();
    g.setLineDash([]); g.strokeStyle = sig; g.lineWidth = 1.8; g.beginPath(); sweep.forEach((p, i) => (i ? g.lineTo(QX(p.soc), QY(Math.min(p.s, smax))) : g.moveTo(QX(p.soc), QY(Math.min(p.s, smax))))); g.stroke(); g.lineWidth = 1;
    g.fillStyle = sig; g.beginPath(); g.arc(QX(st.soc), QY(Math.min(d.s, smax)), 4.5, 0, 7); g.fill();
    label(c, '- - network alone', p2.x + p2.w, p2.y + 8, SCOPE.faint, 'right', fs - 1);
    label(c, '— with SOC feedback', p2.x + p2.w, p2.y + 20, sig, 'right', fs - 1);
  }

  syncTrip();
  setMoment(Math.floor(trip().moments.length / 2));
}
