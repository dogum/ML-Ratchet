// BuildingDetective live instrument: real exam units from the release's examples/, never used in training or for any
// choice: three BDG2 buildings and one HUE home (hourly 2017 meters and weather), four RECS homes and four CBECS
// buildings (annual survey totals). The browser recomputes every feature from the meters (a port of features.py) and
// runs the hourly, monthly or annual network on each change. The controls change the input: which resolution the
// Detective is given (the same meters as hourly data, monthly bills or one annual total), how many months of meter
// data, electricity only, a warmer or colder year (the same meters against shifted outdoor temperature) and meter noise.
// Truths come from BDG2 metadata (use, floor area, year built, which meters exist) and the surveys' answers.
import type { InstrumentContext } from '../types.ts';
import { fetchPack } from '../../lib/nn/pack.ts';
import { inputs, read, diag, CH, type Channel, type Mode, type Perturb, type Reading, type Unit, type WMeta } from './net.ts';
import { hourlyUnit } from './parity.ts';
import { shell, picker, slider, button, scheduler, gauges, probs, canvas, gridLines, label, h, SCOPE, type Canvas2D } from '../kit.ts';

const STYLE = `
.bd-seg { display: flex; flex-direction: column; gap: 6px; }
.bd-seg > span { font-size: 12.5px; color: var(--scr-2ink); }
.bd-seg > div { display: flex; gap: 4px; }
.bd-seg button { flex: 1 1 auto; min-width: 0; padding: 5px 4px; border-radius: 3px; border: 1px solid var(--scr-rule); background: transparent; color: var(--scr-2ink); font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.04em; cursor: pointer; }
.bd-seg button[aria-pressed='true'] { border-color: var(--c); color: var(--scr); background: var(--c); }
.bd-btns { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 8px; }
.bd-btns .inst-btn[aria-pressed='true'] { border-color: var(--c); color: var(--c); }
.bd-kv { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 3px 10px; font-size: 12px; color: var(--scr-2ink); }
.bd-kv b { font-family: var(--mono); font-variation-settings: 'wdth' 87.5; font-weight: 400; font-size: 11.5px; color: var(--scr-ink); text-align: right; white-space: nowrap; }
.bd-note { font-size: 12px; line-height: 1.5; color: var(--scr-3ink); margin: 0; }
.bd-note b { font-weight: 500; color: var(--scr-2ink); }
.bd-hide { display: none !important; }
`;

const MODES: [Mode, string][] = [['hourly', 'Hourly meters'], ['monthly', 'Monthly bills'], ['annual', 'Annual total']];
const CH_NAME: Record<Channel, string> = { elec: 'electricity', fuel: 'gas / fuel', dheat: 'district heat', dcool: 'chilled water' };
const CH_COL: Record<Channel, string> = { elec: '', fuel: '#e0b04a', dheat: '#f07a4c', dcool: '#5fb2db' };
const USE_NAME: Record<string, string> = { residential_single: 'home (single)', residential_multi: 'home (multi)', healthcare: 'health', warehouse: 'warehouse' };
const HEAT_NAME: Record<string, string> = { heat_pump: 'heat pump', resistance: 'resistance', fuel: 'fuel', district: 'district', none: 'none' };
const COOL_NAME: Record<string, string> = { none: 'none', dx: 'DX', chiller: 'chiller', district: 'district' };
const big = (v: number) => Math.round(v).toLocaleString('en-GB');
const sig = (v: number) => (v >= 100 ? big(v) : v.toPrecision(2));

export async function mount(ctx: InstrumentContext) {
  const [W, X] = await Promise.all([fetchPack<WMeta>(`${ctx.base}weights`), fetchPack<Unit[]>(`${ctx.base}examples`)]);
  const ex = X.meta;
  const hu = ex.map((_, i) => hourlyUnit(X, i));
  if (!document.getElementById('bd-inst-style')) document.head.append(h('style', { id: 'bd-inst-style' }, STYLE));
  const np = W.meta.n_params;
  const ui = shell(ctx.root, {
    title: 'BuildingDetective',
    subtitle: 'size, age, envelope, systems and use from a building\'s meters and the weather — real exam buildings, never used in training',
    params: `${big(np.hourly)} parameters (hourly net) · matches Python to 1e-6`,
  });
  const st = { i: 0, mode: 'hourly' as Mode, months: 12, elec: false, dT: 0, noise: 0 };
  let view: { r: Reading | null; series?: { T: Float64Array; meters: Partial<Record<Channel, Float64Array>> }; deg: number } = { r: null, deg: 0 };

  const short = (u: Unit) => u.id.replace(/^(RECS|CBECS) (RECS|CBECS)_/, '$1 ');
  picker(ui.picker, ex.map((u, i) => ({
    label: u.kind === 'bdg2' ? u.id.split('_').slice(0, 2).join(' · ') : u.kind === 'hue' ? `${u.id.replace('_', ' ')} · home` : short(u),
    sub: u.kind === 'bdg2' ? `BDG2 2017 · ${(u.channels ?? []).length} meters` : u.kind === 'hue' ? 'HUE, BC · electricity' : `${u.kind === 'recs' ? 'home' : 'commercial'} · annual survey`,
    value: i,
  })), (i) => { st.i = i; sync(); go(); }, 0);

  const narrow = () => ui.screen.getBoundingClientRect().width < 520;
  const plot = canvas(ui.screen, ctx.compact ? 1.9 : narrow() ? 1.1 : 1.6, (c) => draw(c));

  const seg = h('div', { class: 'bd-seg', role: 'group', 'aria-label': 'Input resolution' }, h('span', {}, 'The Detective is given'));
  const row = h('div', {});
  const mbtns = MODES.map(([m, n]) => {
    const b = h('button', { type: 'button', 'aria-pressed': String(m === st.mode) }, n);
    b.addEventListener('click', () => { st.mode = m; sync(); go(); });
    row.append(b);
    return b;
  });
  seg.append(row);
  ui.controls.append(seg);
  const sM = slider(ui.controls, {
    label: 'Months of meter data', min: 3, max: 12, step: 3, value: 12,
    fmt: (v) => (v === 12 ? 'all 12 months' : `Jan–${['', '', '', 'Mar', '', '', 'Jun', '', '', 'Sep'][v]} only`),
    hint: 'the meters after that are blanked; the weather stays (the exam had 3- and 6-month copies)',
    onInput: (v) => { st.months = v; go(); },
  });
  const sT = slider(ui.controls, {
    label: 'Weather shift', min: -4, max: 4, step: 1, value: 0,
    fmt: (v) => (v === 0 ? 'as measured' : `${v > 0 ? '+' : '−'}${Math.abs(v)} K outdoors`),
    hint: 'the same meters against a warmer or colder year: the balance points should move with it',
    onInput: (v) => { st.dT = v; go(); },
  });
  const sN = slider(ui.controls, {
    label: 'Meter noise', min: 0, max: 0.4, step: 0.05, value: 0,
    fmt: (v) => (v === 0 ? 'none' : `±${Math.round(v * 100)} % per hour`),
    hint: 'multiplies each hourly reading by exp(noise × z), a fixed random series per meter',
    onInput: (v) => { st.noise = v; go(); },
  });
  const btns = h('div', { class: 'bd-btns' });
  ui.controls.append(btns);
  const eBtn = button(btns, 'Meters: all', () => { st.elec = !st.elec; sync(); go(); });
  eBtn.title = 'electricity only: the exam read chilled-water and steam meters from electricity alone (AUC 0.75 and 0.70)';
  if (!ctx.compact) button(btns, 'As recorded', () => { Object.assign(st, { months: 12, elec: false, dT: 0, noise: 0 }); sM.set(12); sT.set(0); sN.set(0); sync(); go(); });

  function sync() {
    const hourly = !!hu[st.i];
    const mode: Mode = hourly ? st.mode : 'annual';
    mbtns.forEach((b, k) => { b.setAttribute('aria-pressed', String(MODES[k][0] === mode)); (b as HTMLButtonElement).disabled = !hourly && MODES[k][0] !== 'annual'; });
    for (const s of [sM, sT, sN]) s.input.closest('label')!.classList.toggle('bd-hide', !hourly || (ctx.compact && s !== sT));
    eBtn.textContent = st.elec ? 'Meters: electricity only' : 'Meters: all';
    eBtn.setAttribute('aria-pressed', String(st.elec));
  }
  sync();

  const gaugeBox = h('div', {}), probBox = h('div', {}), kvBox = h('div', { class: 'bd-kv' }), note = h('p', { class: 'bd-note' });
  ui.readout.append(gaugeBox, probBox, kvBox, note);

  async function run() {
    ui.setBusy(true);
    await new Promise((r) => requestAnimationFrame(r));
    const t0 = performance.now();
    const p: Perturb = hu[st.i] ? { mode: st.mode, months: st.months, elec: st.elec, dT: st.dT, noise: st.noise } : { elec: st.elec };
    const { mode, X: x, series } = inputs(hu[st.i], ex[st.i], p);
    const deg = diag.degenerate;
    const r = read(W, mode, x);
    const ms = performance.now() - t0;
    view = { r, series, deg };
    plot.redraw();
    render(r, mode, deg);
    ui.stamp(ms, `${ex[st.i].id} · ${mode} network · features computed here too`);
    ui.setBusy(false);
  }
  const go = scheduler(run, 40);

  function render(r: Reading, mode: Mode, deg: number) {
    const u = ex[st.i], t = u.truth, v = r.v;
    const yt = t.year_built ?? (t.year_range ? (t.year_range[0] + t.year_range[1]) / 2 : null);
    const gs = [
      { label: 'Floor area', unit: 'm²', value: v.floor_area_m2[0], lo: v.floor_area_m2[1], hi: v.floor_area_m2[2], truth: t.floor_area_m2 ?? null, truthLabel: u.kind === 'bdg2' ? 'BDG2 metadata' : 'survey', range: [30, 100000] as [number, number], digits: 0, log: true },
      { label: 'Year built', value: v.year_built[0], lo: v.year_built[1], hi: v.year_built[2], truth: yt, truthLabel: t.year_range ? `survey ${t.year_range[0]}–${t.year_range[1]}` : 'BDG2 metadata', range: [1880, 2030] as [number, number], digits: 0 },
      { label: 'Envelope UA (no real truth)', unit: 'W/K', value: v.UA_W_K[0], lo: v.UA_W_K[1], hi: v.UA_W_K[2], range: [30, 100000] as [number, number], digits: 0, log: true },
      { label: 'Time constant (no real truth)', unit: 'h', value: v.tau_h[0], lo: v.tau_h[1], hi: v.tau_h[2], range: [3, 500] as [number, number], digits: 0, log: true },
      { label: 'Heating balance point', unit: '°C', value: v.Tb_heat_C[0], lo: v.Tb_heat_C[1], hi: v.Tb_heat_C[2], range: [-15, 30] as [number, number], digits: 1 },
    ];
    gaugeBox.innerHTML = '';
    gauges(gaugeBox, ctx.compact ? [gs[0], gs[2], gs[4]] : gs);
    probBox.innerHTML = '';
    const useTruth = t.use;
    const top = (p: number[], names: string[], n: number) => p.map((q, k) => ({ q, k })).sort((a, b) => b.q - a.q).slice(0, n).map(({ q, k }) => ({ q, name: names[k] }));
    const uses = top(r.use, W.meta.uses, 4);
    if (useTruth && !uses.some((x) => x.name === useTruth)) uses.push({ q: r.use[W.meta.uses.indexOf(useTruth)], name: useTruth });
    probs(probBox, 'Primary use (top 4)', uses.map((x) => ({ label: USE_NAME[x.name] ?? x.name, p: x.q, truth: x.name === useTruth })));
    if (!ctx.compact) {
      const hb = h('div', {}), cb = h('div', {});
      probBox.append(hb, cb);
      probs(hb, 'Heating system', W.meta.heat.map((n, k) => ({ label: HEAT_NAME[n], p: r.heat[k], truth: n === t.heat })));
      probs(cb, 'Cooling system', W.meta.cool.map((n, k) => ({ label: COOL_NAME[n], p: r.cool[k] })));
    }
    const kv: [string, string][] = [
      ['Heating / cooling balance point', `${v.Tb_heat_C[0].toFixed(1)} / ${v.Tb_cool_C[0].toFixed(1)} °C`],
      ['Non-HVAC share of energy', `${Math.round(v.base_share[0] * 100)} %`],
    ];
    if (u.kind === 'bdg2' || u.kind === 'cbecs') kv.push(['Weekday hours open (no real truth)', `${v.op_hours[0].toFixed(1)} h`]);
    if (t.meters) kv.push(['Meters (BDG2)', t.meters.join(', ')]);
    if (t.codes) kv.push(['HUE heating codes', t.codes]);
    kv.push(['Python, as recorded: area / year', `${big(u.python.floor_area_m2)} m² / ${Math.round(u.python.year_built)}`]);
    kvBox.innerHTML = kv.map(([k, x]) => `<span>${k}</span><b>${x}</b>`).join('');
    const changed: string[] = [];
    if (hu[st.i] && mode !== 'hourly') changed.push(mode === 'monthly' ? 'monthly bills made from the meters' : 'one annual total per meter');
    if (st.months < 12 && hu[st.i]) changed.push(`only ${st.months} months of meter data`);
    if (st.elec) changed.push('electricity meter only');
    if (st.dT && hu[st.i]) changed.push(`outdoor temperature ${st.dT > 0 ? '+' : '−'}${Math.abs(st.dT)} K`);
    if (st.noise && hu[st.i]) changed.push(`meter noise ±${Math.round(st.noise * 100)} %`);
    note.innerHTML = (changed.length ? `<b>Changed: ${changed.join(', ')}.</b> ` : '')
      + (deg ? '<b>The change-point fit is degenerate here</b> (every day sits on one side of the best balance point, so many balance points fit equally well); numpy and this page can pick different ones, and the reading can differ from the Python release. ' : '')
      + 'Bars are 1 sigma after the dev widening (hourly floor area ×3.0). On the real exam it lost where it matters most: primary use 11 % right on BDG2 against 40 % for always guessing education, '
      + 'and year built no better than the dev median. Floor area: 48–50 % median error on BDG2 against 51.5 % for energy ÷ the true use\'s median intensity. UA and the time constant have no real truth.';
  }

  function draw(c: Canvas2D) {
    const { ctx: g, w, h: H } = c;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, w, H);
    const r = view.r, u = ex[st.i];
    if (!r) return;
    const padL = 46, padR = 14, top = 26, bot = 24;
    const iw = w - padL - padR, ih = H - top - bot;
    const s = view.series, tm = hu[st.i]?.tm;
    if (!s || !tm) { // annual survey unit: kWh per fuel
      const kwh = Object.entries(u.kwh ?? {}).filter(([k]) => !st.elec || k === 'elec') as [Channel, number][];
      const mx = Math.max(...kwh.map(([, e]) => e), 1);
      const bh = Math.min(28, ih / (kwh.length + 2));
      label(c, `${short(u)} · annual site energy (kWh) · HDD ${big(u.hdd!)} · CDD ${big(u.cdd!)} (base 18.33 °C)`, padL, top - 12, SCOPE.text, 'left', 9.5);
      kwh.forEach(([k, e], j) => {
        const y = top + 16 + j * (bh + 10);
        g.fillStyle = CH_COL[k] || ctx.accent; g.fillRect(padL, y, (e / mx) * iw * 0.8, bh);
        label(c, `${CH_NAME[k]} ${big(e)} kWh`, padL + 4, y + bh / 2, SCOPE.ink, 'left', 10);
      });
      label(c, 'Truth: ' + [u.truth.use, u.truth.floor_area_m2 ? `${big(u.truth.floor_area_m2)} m²` : '', u.truth.heat ? `heat: ${u.truth.heat}` : '', u.truth.year_range ? `built ${u.truth.year_range.join('–')}` : ''].filter(Boolean).join(' · '),
        padL, H - 12, SCOPE.faint, 'left', 9.5);
      return;
    }
    // energy signature: daily mean kW of each meter against the daily mean outdoor temperature (after the controls)
    const n = s.T.length, dsum = new Map<number, number[]>();
    const days: number[] = []; const di = new Int32Array(n);
    for (let i = 0; i < n; i++) { const d = tm.day[i]; if (!dsum.has(d)) { dsum.set(d, []); days.push(d); } di[i] = days.length - 1; }
    const nd = days.length, Td = new Float64Array(nd), cT = new Float64Array(nd);
    for (let i = 0; i < n; i++) if (Number.isFinite(s.T[i])) { Td[di[i]] += s.T[i]; cT[di[i]]++; }
    for (let d = 0; d < nd; d++) Td[d] = cT[d] ? Td[d] / cT[d] : NaN;
    const lines: { ch: Channel; e: Float64Array; mx: number }[] = [];
    for (const ch of CH) {
      const x = s.meters[ch]; if (!x) continue;
      const e = new Float64Array(nd), ce = new Float64Array(nd);
      for (let i = 0; i < n; i++) if (Number.isFinite(x[i])) { e[di[i]] += x[i]; ce[di[i]]++; }
      let mx = 0; for (let d = 0; d < nd; d++) { e[d] = ce[d] >= 12 ? e[d] / ce[d] : NaN; if (Number.isFinite(e[d])) mx = Math.max(mx, e[d]); }
      lines.push({ ch, e, mx });
    }
    const t0 = -25, t1 = 38;
    const xT = (t: number) => padL + ((t - t0) / (t1 - t0)) * iw;
    gridLines(c, padL, top, iw, ih, 9, 4);
    for (let t = -20; t <= 35; t += 10) label(c, `${t}°C`, xT(t), top + ih + 10, SCOPE.faint, 'center', 9);
    // balance points read by the network, with 1-sigma bands
    for (const [k, col] of [['Tb_heat_C', '#f07a4c'], ['Tb_cool_C', '#5fb2db']] as const) {
      const [m, lo, hi] = r.v[k];
      g.globalAlpha = 0.14; g.fillStyle = col; g.fillRect(xT(lo), top, xT(hi) - xT(lo), ih); g.globalAlpha = 1;
      g.strokeStyle = col; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(xT(m), top); g.lineTo(xT(m), top + ih); g.stroke(); g.setLineDash([]);
      label(c, `${k === 'Tb_heat_C' ? 'heat' : 'cool'} ${m.toFixed(1)}°`, xT(m) + 3, top + 8 + (k === 'Tb_cool_C' ? 12 : 0), col, 'left', 9);
    }
    // each meter scaled to its own maximum daily mean
    const yv = (q: number) => top + ih - (q * ih) * 0.95;
    for (const L of lines) {
      g.fillStyle = CH_COL[L.ch] || ctx.accent;
      g.globalAlpha = 0.75;
      for (let d = 0; d < nd; d++) if (Number.isFinite(L.e[d]) && Number.isFinite(Td[d])) g.fillRect(xT(Td[d]) - 1.2, yv(L.e[d] / L.mx) - 1.2, 2.4, 2.4);
      g.globalAlpha = 1;
    }
    const leg = lines.map((L) => `${CH_NAME[L.ch]} (max ${sig(L.mx)} kW)`).join(' · ');
    label(c, narrow() ? 'daily mean kW vs outdoor °C' : `energy signature: daily mean power (each meter / its max) vs daily mean outdoor temperature · ${leg}`, padL, top - 12, SCOPE.text, 'left', 9.5);
    lines.forEach((L, j) => { g.fillStyle = CH_COL[L.ch] || ctx.accent; g.fillRect(padL + iw - 10, top + 4 + j * 12, 7, 7); label(c, CH_NAME[L.ch], padL + iw - 14, top + 8 + j * 12, SCOPE.text, 'right', 9); });
  }

  await run();
}
