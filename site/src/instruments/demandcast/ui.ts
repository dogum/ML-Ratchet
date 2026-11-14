// DemandCast live instrument: 112 real exam days (2023-2025, 14 per grid, the release page's fixed draw), never used in
// training or choosing. The browser re-runs the building-stock physics for the 38 h from issue (10:00 the day before)
// to the end of the day, the wind and solar physics (tabulated in Python over weather multipliers) and the three
// networks on every change. The controls change tomorrow's weather; EIA-930's metered output and the grid's own
// day-ahead forecast stay on screen as the truth for the day as it happened.
import type { InstrumentContext } from '../types.ts';
import { fetchPack } from '../../lib/nn/pack.ts';
import { predict, dayOf, BAS, type DMeta, type DDay, type Reading, type Day } from './net.ts';
import { shell, picker, slider, button, scheduler, gauges, canvas, gridLines, label, h, SCOPE, type Canvas2D, type Gauge } from '../kit.ts';

const STYLE = `
.dc-sel { display: flex; flex-direction: column; gap: 6px; }
.dc-sel > span { font-size: 12.5px; color: var(--scr-2ink); }
.dc-sel select { appearance: none; -webkit-appearance: none; width: 100%; padding: 7px 28px 7px 9px; border-radius: 4px; border: 1px solid var(--scr-rule); background: transparent; color: var(--scr-ink); font-family: var(--mono); font-variation-settings: 'wdth' 87.5; font-size: 11.5px; cursor: pointer;
  background-image: linear-gradient(45deg, transparent 50%, var(--scr-3ink) 50%), linear-gradient(135deg, var(--scr-3ink) 50%, transparent 50%); background-position: calc(100% - 15px) 50%, calc(100% - 10px) 50%; background-size: 5px 5px; background-repeat: no-repeat; }
.dc-sel select:focus-visible { outline: 2px solid var(--c); outline-offset: 2px; }
.dc-sel option { background: #0d0f0c; color: #ebe6d8; }
.dc-kv { display: grid; grid-template-columns: minmax(0, 1fr) auto auto auto; gap: 3px 10px; font-size: 12px; color: var(--scr-2ink); align-items: baseline; }
.dc-kv b { font-family: var(--mono); font-variation-settings: 'wdth' 87.5; font-weight: 400; font-size: 11.5px; color: var(--scr-ink); text-align: right; white-space: nowrap; }
.dc-kv b.win { color: var(--c); }
.dc-kv i { font-style: normal; font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.04em; color: var(--scr-3ink); text-align: right; }
.dc-note { font-size: 12px; line-height: 1.5; color: var(--scr-3ink); margin: 0; }
.dc-note b { font-weight: 500; color: var(--scr-2ink); }
`;

const NAME: Record<string, string> = {
  ERCO: 'ERCOT', CISO: 'CAISO', PJM: 'PJM', MISO: 'MISO', SWPP: 'SPP', ISNE: 'ISO-NE', NYIS: 'NYISO', BPAT: 'BPA',
};
const WHERE: Record<string, string> = {
  ERCO: 'Texas', CISO: 'California', PJM: 'Mid-Atlantic', MISO: 'Midwest', SWPP: 'Plains', ISNE: 'New England', NYIS: 'New York', BPAT: 'Northwest',
};
const TRUTH = '#ebe6d8', DA = '#e0a458', PHYS = 'rgba(235,230,216,0.45)';

const finite = (v: number) => Number.isFinite(v);
function mape(p: ArrayLike<number>, y: ArrayLike<number>, also: ArrayLike<number>[] = []) {
  let s = 0, n = 0;
  for (let h = 0; h < y.length; h++) if (finite(y[h]) && y[h] > 0 && finite(p[h]) && also.every((a) => finite(a[h]))) { s += Math.abs(p[h] - y[h]) / y[h]; n++; }
  return n ? (100 * s) / n : NaN;
}
function nmae(p: ArrayLike<number>, y: ArrayLike<number>) {
  let e = 0, t = 0;
  for (let h = 0; h < y.length; h++) if (finite(y[h])) { e += Math.abs(p[h] - y[h]); t += y[h]; }
  return t > 0 ? (100 * e) / t : NaN;
}
const sum = (a: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) if (finite(a[i])) s += a[i]; return s; };
const pct = (v: number) => (finite(v) ? `${v.toFixed(v < 10 ? 2 : 1)} %` : '—');

export async function mount(ctx: InstrumentContext) {
  const [W, X] = await Promise.all([fetchPack<DMeta>(`${ctx.base}weights`), fetchPack<DDay[]>(`${ctx.base}examples`)]);
  if (!document.getElementById('dc-inst-style')) document.head.append(h('style', { id: 'dc-inst-style' }, STYLE));
  const nParams = Object.values(W.meta.n_params).reduce((a, b) => a + b, 0);
  const ui = shell(ctx.root, {
    title: 'DemandCast',
    subtitle: 'tomorrow’s hourly demand, wind and solar for a US grid — a real exam day, never used in training',
    params: `3 networks, ${(nParams / 1e6).toFixed(2)} M parameters · matches Python to 1e-6`,
  });
  const ex = X.meta;
  const byBA = (ba: string) => ex.map((_, i) => i).filter((i) => ex[i].ba === ba);
  const st = { ba: 'ERCO', i: byBA('ERCO')[0], dT: 0, hum: 1, sun: 1, wind: 1 };
  let view: { d: Day; r: Reading } | null = null;

  picker(ui.picker, BAS.map((ba) => ({ label: NAME[ba], sub: WHERE[ba], value: ba })), (ba) => {
    st.ba = ba; st.i = byBA(ba)[0]; fillDays(); go();
  }, 0);

  const narrow = () => ui.screen.getBoundingClientRect().width < 520;
  const plot = canvas(ui.screen, ctx.compact ? 1.5 : narrow() ? 0.95 : 1.25, (c) => draw(c));

  // ---- controls
  const sel = h('select', { 'aria-label': 'Exam day' });
  ui.controls.append(h('label', { class: 'dc-sel' }, h('span', {}, 'Exam day (local standard time)'), sel));
  sel.addEventListener('change', () => { st.i = +sel.value; go(); });
  function fillDays() {
    sel.innerHTML = '';
    for (const i of byBA(st.ba)) {
      const e = ex[i];
      const tag = e.tag === 'random' ? '' : ` · ${e.tag}`;
      const o = h('option', { value: String(i) }, `${e.day} · ${e.T_min.toFixed(0)}…${e.T_max.toFixed(0)} °C${tag}`);
      if (i === st.i) o.selected = true;
      sel.append(o);
    }
  }
  fillDays();
  const sT = slider(ui.controls, {
    label: 'Temperature', min: -10, max: 10, step: 0.5, value: 0,
    fmt: (v) => (v === 0 ? 'as recorded' : `${v > 0 ? '+' : ''}${v.toFixed(1)} °C`),
    hint: 'added to every hour from issue to the end of the day; humidity follows',
    onInput: (v) => { st.dT = v; go(); },
  });
  let sH: ReturnType<typeof slider> | null = null, sS: ReturnType<typeof slider> | null = null, sW: ReturnType<typeof slider> | null = null;
  if (!ctx.compact) {
    sH = slider(ui.controls, {
      label: 'Humidity', min: 0.7, max: 1.3, step: 0.05, value: 1,
      fmt: (v) => (v === 1 ? 'as recorded' : `×${v.toFixed(2)}`), hint: 'water vapour in the air (latent cooling load)',
      onInput: (v) => { st.hum = v; go(); },
    });
    sS = slider(ui.controls, {
      label: 'Sunshine', min: 0.4, max: 1.25, step: 0.05, value: 1,
      fmt: (v) => (v === 1 ? 'as recorded' : `×${v.toFixed(2)}`), hint: 'irradiance (capped at clear sky): lighting, solar gains, rooftop and utility PV',
      onInput: (v) => { st.sun = v; go(); },
    });
    sW = slider(ui.controls, {
      label: 'Wind speed', min: 0.6, max: 1.35, step: 0.05, value: 1,
      fmt: (v) => (v === 1 ? 'as recorded' : `×${v.toFixed(2)}`), hint: 'at every wind-farm weather point',
      onInput: (v) => { st.wind = v; go(); },
    });
    button(ui.controls, 'As recorded', () => {
      Object.assign(st, { dT: 0, hum: 1, sun: 1, wind: 1 });
      sT.set(0); sH!.set(1); sS!.set(1); sW!.set(1); go();
    });
  }

  const gaugeBox = h('div', {}), kv = h('div', { class: 'dc-kv' }), note = h('p', { class: 'dc-note' });
  ui.readout.append(gaugeBox, kv, note);

  async function run() {
    ui.setBusy(true);
    await new Promise((r) => requestAnimationFrame(r));
    const d = dayOf(X, st.i);
    const t0 = performance.now();
    const r = predict(W, d, { dT: st.dT, hum: st.hum, sun: st.sun, wind: st.wind });
    const ms = performance.now() - t0;
    view = { d, r };
    plot.redraw();
    render(d, r);
    ui.stamp(ms, `${NAME[d.meta.ba]} ${d.meta.day}`);
    ui.setBusy(false);
  }
  const go = scheduler(run, 30);

  function render(d: Day, r: Reading) {
    const m = d.meta, y = d.truth.y;
    let pk = 0;
    for (let k = 1; k < 24; k++) if (r.d.mw[k] > r.d.mw[pk]) pk = k;
    let ymax = -Infinity;
    for (let k = 0; k < 24; k++) if (finite(y[k])) ymax = Math.max(ymax, y[k]);
    const s68 = r.d.logSigma[pk];
    const top = Math.max(...r.d.hi95, ymax) / 1000;
    const gs: Gauge[] = [{
      label: `Peak hour demand, DemandCast (hour ${pk})`, unit: 'GW', value: r.d.mw[pk] / 1000,
      lo: (r.d.mw[pk] * Math.exp(-s68)) / 1000, hi: (r.d.mw[pk] * Math.exp(s68)) / 1000,
      truth: finite(ymax) ? ymax / 1000 : null, truthLabel: 'metered peak (EIA-930)', range: [0, top * 1.08], digits: 1,
    }];
    if (r.w) {
      const yw = sum(Array.from(d.yw, (v) => v * m.capw)) / 1000;
      gs.push({ label: 'Wind energy over the day', unit: 'GWh', value: sum(r.w.cf.map((v) => v * m.capw)) / 1000,
        truth: yw, truthLabel: 'metered', range: [0, (m.capw * 24 * 0.75) / 1000], digits: 1 });
    }
    if (r.s) {
      const ys = sum(Array.from(d.ys, (v) => v * m.caps)) / 1000;
      gs.push({ label: 'Solar energy over the day', unit: 'GWh', value: sum(r.s.cf.map((v) => v * m.caps)) / 1000,
        truth: ys, truthLabel: 'metered', range: [0, (m.caps * 24 * 0.45) / 1000], digits: 1 });
    }
    gaugeBox.innerHTML = '';
    gauges(gaugeBox, gs);
    // the day's errors against the metered truth
    const tr = d.truth, only = [tr.df];
    const dm = [mape(r.d.mw, y, only), mape(tr.df, y), mape(r.phys.total, y, only), mape(tr.pers, y, only)];
    const best = Math.min(...dm.filter(finite));
    let inb = 0, nb = 0;
    for (let k = 0; k < 24; k++) if (finite(y[k])) { nb++; if (y[k] >= r.d.lo95[k] && y[k] <= r.d.hi95[k]) inb++; }
    const cell = (v: number) => `<b class="${v === best ? 'win' : ''}">${pct(v)}</b>`;
    const rows = [`<span></span><i>DEMANDCAST</i><i>GRID DA</i><i>PHYSICS</i>`,
      `<span>Demand error (MAPE)</span>${cell(dm[0])}${cell(dm[1])}${cell(dm[2])}`];
    if (r.w) rows.push(`<span>Wind error (Σ|err| / Σ output)</span><b>${pct(nmae(r.w.cf, d.yw))}</b><b>—</b><b>${pct(nmae(r.w.prior, d.yw))}</b>`);
    if (r.s) rows.push(`<span>Solar error (Σ|err| / Σ output)</span><b>${pct(nmae(r.s.cf, d.ys))}</b><b>—</b><b>${pct(nmae(r.s.prior, d.ys))}</b>`);
    rows.push(`<span>Persistence (same hour yesterday)</span><b>${pct(dm[3])}</b><span></span><span></span>`);
    rows.push(`<span>Metered hours inside the 95 % band</span><b>${inb} of ${nb}</b><span></span><span></span>`);
    kv.innerHTML = rows.join('');
    const changed: string[] = [];
    if (st.dT) changed.push(`${st.dT > 0 ? '+' : ''}${st.dT} °C`);
    if (st.hum !== 1) changed.push(`humidity ×${st.hum.toFixed(2)}`);
    if (st.sun !== 1) changed.push(`sunshine ×${st.sun.toFixed(2)}`);
    if (st.wind !== 1) changed.push(`wind ×${st.wind.toFixed(2)}`);
    const solarNote = m.ba === 'CISO' || m.ba === 'PJM' || m.ba === 'BPAT'
      ? ' Solar here is a known failure: EIA-930’s solar timestamps moved by an hour between the dev and exam years (CAISO, PJM in 2025), and BPA’s fleet is tiny; persistence beats DemandCast on solar overall (19.1 % vs 29.9 %).'
      : m.ba === 'NYIS' ? ' NYISO does not report solar in EIA-930.' : '';
    note.innerHTML = changed.length
      ? `<b>Changed: ${changed.join(', ')}.</b> The metered values and the grid’s forecast are for the day as it happened; this is now a different day, so read the change, not the error.`
      : `<b>As recorded:</b> issued 10:00 the day before, with the weather that actually happened — an upper bound. With real forecast weather (2024 only) the pooled error rises from 2.44 % to 2.92 % (grid forecasts 3.20 %) and DemandCast beats the grid’s own forecast in 4 of 8 grids.${solarNote} Weather inputs are stored to 0.01 °C, so readings match the release to about 1e-4.`;
  }

  function draw(c: Canvas2D) {
    const { ctx: g, w, h: H } = c;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, w, H);
    if (!view) return;
    const { d, r } = view, m = d.meta;
    const small = w < 520;
    const padL = 42, padR = 10, top = 18, bot = 18, gap = 26;
    const iw = w - padL - padR, ih = H - top - bot - 2 * gap;
    const hd = Math.round(ih * (ctx.compact ? 0.6 : 0.5)), hr = Math.round((ih - hd) / 2);
    const xs = (k: number) => padL + ((k + 0.5) / 24) * iw;
    const panel = (y0: number, hh: number, title: string, series: { v: ArrayLike<number>; col: string; dash?: number[]; dots?: boolean; w?: number }[],
      band: [ArrayLike<number>, ArrayLike<number>] | null, empty?: string) => {
      gridLines(c, padL, y0, iw, hh, 8, 2);
      label(c, title, padL + 4, y0 - 8, SCOPE.text, 'left', 9.5);
      if (empty) { label(c, empty, padL + iw / 2, y0 + hh / 2, SCOPE.faint, 'center', 10); return; }
      let lo = Infinity, hi = -Infinity;
      for (const s of series) for (let k = 0; k < 24; k++) if (finite(s.v[k])) { lo = Math.min(lo, s.v[k]); hi = Math.max(hi, s.v[k]); }
      if (band) for (let k = 0; k < 24; k++) { lo = Math.min(lo, band[0][k]); hi = Math.max(hi, band[1][k]); }
      if (!finite(lo)) { lo = 0; hi = 1; }
      const span = hi - lo || Math.max(hi, 1);
      lo = Math.max(0, lo - span * 0.06); hi += span * 0.06;
      const ys = (v: number) => y0 + hh - ((v - lo) / (hi - lo)) * hh;
      for (let j = 0; j <= 2; j++) { const v = lo + ((hi - lo) * j) / 2; label(c, v.toFixed(v < 10 ? 1 : 0), padL - 4, ys(v), SCOPE.faint, 'right', 9); }
      if (band) {
        g.beginPath();
        for (let k = 0; k < 24; k++) k ? g.lineTo(xs(k), ys(band[1][k])) : g.moveTo(xs(k), ys(band[1][k]));
        for (let k = 23; k >= 0; k--) g.lineTo(xs(k), ys(band[0][k]));
        g.closePath(); g.globalAlpha = 0.2; g.fillStyle = ctx.accent; g.fill(); g.globalAlpha = 1;
      }
      for (const s of series) {
        g.strokeStyle = s.col; g.fillStyle = s.col; g.lineWidth = s.w ?? 1.4; g.setLineDash(s.dash ?? []);
        g.beginPath();
        let on = false;
        for (let k = 0; k < 24; k++) {
          if (!finite(s.v[k])) { on = false; continue; }
          on ? g.lineTo(xs(k), ys(s.v[k])) : g.moveTo(xs(k), ys(s.v[k]));
          on = true;
        }
        g.stroke(); g.setLineDash([]);
        if (s.dots) for (let k = 0; k < 24; k++) if (finite(s.v[k])) { g.beginPath(); g.arc(xs(k), ys(s.v[k]), 2.2, 0, 2 * Math.PI); g.fill(); }
      }
    };
    const GW = (a: ArrayLike<number>, k = 1) => Array.from(a, (v) => (v * k) / 1000);
    const y1 = top;
    panel(y1, hd, small ? 'demand, GW' : 'demand GW: DemandCast, 95 % band · metered · grid DA · physics',
      [{ v: GW(r.phys.total), col: PHYS, dash: [2, 3], w: 1 }, { v: GW(d.truth.df), col: DA, dash: [5, 3], w: 1.2 },
        { v: GW(r.d.mw), col: ctx.accent, w: 1.8 }, { v: GW(d.truth.y), col: TRUTH, dots: true, w: 1 }],
      [GW(r.d.lo95), GW(r.d.hi95)]);
    const y2 = y1 + hd + gap, y3 = y2 + hr + gap;
    panel(y2, hr, small ? 'wind, GW' : 'wind GW: DemandCast, 95 % band · metered · physics',
      r.w ? [{ v: GW(r.w.prior, m.capw), col: PHYS, dash: [2, 3], w: 1 }, { v: GW(r.w.cf, m.capw), col: ctx.accent, w: 1.6 }, { v: GW(d.yw, m.capw), col: TRUTH, dots: true, w: 1 }] : [],
      r.w ? [GW(r.w.lo95, m.capw), GW(r.w.hi95, m.capw)] : null, r.w ? undefined : 'no wind fleet reported');
    panel(y3, hr, small ? 'solar, GW' : 'solar GW: DemandCast, 95 % band · metered · physics',
      r.s ? [{ v: GW(r.s.prior, m.caps), col: PHYS, dash: [2, 3], w: 1 }, { v: GW(r.s.cf, m.caps), col: ctx.accent, w: 1.6 }, { v: GW(d.ys, m.caps), col: TRUTH, dots: true, w: 1 }] : [],
      r.s ? [GW(r.s.lo95, m.caps), GW(r.s.hi95, m.caps)] : null, r.s ? undefined : (m.ba === 'NYIS' ? 'NYISO does not report solar in EIA-930' : 'no solar fleet reported'));
    for (let k = 0; k <= 24; k += small ? 6 : 3) label(c, `${k}h`, padL + (k / 24) * iw, H - 8, SCOPE.faint, k === 0 ? 'left' : k === 24 ? 'right' : 'center', 9);
  }

  await run();
}
