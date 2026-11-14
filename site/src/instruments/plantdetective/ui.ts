// PlantDetective live instrument: three real Texas exam units (EPA CAMPD hourly gross output 2018-2025: a coal unit,
// a combined-cycle gas turbine and a peaking gas turbine), never used in training or for any choice. The browser
// recomputes the window features, the three-network ensemble, the pooling over windows and every reading from the
// hourly output on each change. Controls change what the Detective is given: how much of the record (30 days, 91 days,
// one calendar year, all eight years), where that stretch starts, and whether prices and the EIA-860/923 facts come
// with it. The truth is each unit's whole-record CAMPD value (output, operating time and measured heat input), the
// same truth the exam scored every window against.
import type { InstrumentContext } from '../types.ts';
import { fetchPack } from '../../lib/nn/pack.ts';
import { read, series, monthOf, type PMeta, type XMeta, type Control, type Span, type Reading, type Series } from './net.ts';
import { shell, picker, slider, button, scheduler, gauges, probs, canvas, gridLines, label, h, SCOPE, type Canvas2D } from '../kit.ts';

const STYLE = `
.pd-seg { display: flex; flex-direction: column; gap: 6px; }
.pd-seg > span { font-size: 12.5px; color: var(--scr-2ink); }
.pd-seg > div { display: flex; gap: 4px; }
.pd-seg button { flex: 1 1 auto; min-width: 0; padding: 5px 4px; border-radius: 3px; border: 1px solid var(--scr-rule); background: transparent; color: var(--scr-2ink); font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.04em; cursor: pointer; }
.pd-seg button[aria-pressed='true'] { border-color: var(--c); color: var(--scr); background: var(--c); }
.pd-btns { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 8px; }
.pd-btns .inst-btn[aria-pressed='false'] { color: var(--scr-3ink); border-color: var(--scr-rule); text-decoration: line-through; }
.pd-kv { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 3px 10px; font-size: 12px; color: var(--scr-2ink); }
.pd-kv b { font-family: var(--mono); font-variation-settings: 'wdth' 87.5; font-weight: 400; font-size: 11.5px; color: var(--scr-ink); text-align: right; white-space: nowrap; }
.pd-kv b.out { color: #e07a5f; }
.pd-note { font-size: 12px; line-height: 1.5; color: var(--scr-3ink); margin: 0; }
.pd-note b { font-weight: 500; color: var(--scr-2ink); }
`;

const TECH_NAME: Record<string, string> = { coal_steam: 'Coal steam', gas_cc: 'Gas combined cycle', gas_ct: 'Gas turbine', gas_steam: 'Gas steam' };
const ORDER = ['coal_steam', 'gas_cc', 'gas_ct', 'gas_steam'];
const SPANS: [Span, string][] = [['month', '30 days'], ['quarter', '91 days'], ['year', '1 year'], ['all', '2018–25']];
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const monthName = (m: number) => `${MON[m % 12]} ${1970 + Math.floor(m / 12)}`;
const NOTE: Record<string, string> = {
  gas_cc: 'Combined-cycle heat rate is the model’s known failure: on the exam it read 21 % high (all years), because the dev combined cycles were cogeneration or reported only the gas turbine. This unit reports the whole block.',
  gas_ct: 'Gas-turbine heat rate reads about 16 % off on the exam; the EIA-923 annual figure alone does better for turbines (6.5 %).',
  coal_steam: 'Coal is where the heat rate works: 4.4 % median error on the exam (all years), against 10.3 % for the EIA-923 annual figure.',
};

export async function mount(ctx: InstrumentContext) {
  const [W, X] = await Promise.all([fetchPack<PMeta>(`${ctx.base}weights`), fetchPack<XMeta>(`${ctx.base}examples`)]);
  const units = X.meta.units;
  if (!document.getElementById('pd-inst-style')) document.head.append(h('style', { id: 'pd-inst-style' }, STYLE));
  const ui = shell(ctx.root, {
    title: 'PlantDetective',
    subtitle: 'a power unit’s character from its hourly output — a real Texas unit, never used in training',
    params: `3 networks, ${(W.meta.n_params / 1e6).toFixed(2)} M parameters · matches Python to 1e-6`,
  });
  const S: Record<number, Series> = {};
  const ser = (i: number) => (S[i] ??= series(X, i));
  const order = units.map((_, i) => i).sort((a, b) => ORDER.indexOf(units[a].tech) - ORDER.indexOf(units[b].tech));
  const t0 = ser(order[0]).t;
  const m0 = monthOf(t0[0]), m1 = monthOf(t0[t0.length - 1]);
  const y0 = 1970 + Math.floor(m0 / 12), y1 = 1970 + Math.floor(m1 / 12);
  const st = { i: order[0], span: 'all' as Span, month: m0 + 12 * 6, year: 2024, ctx: true, price: true };
  let view: { r: Reading; start: number; hours: number } | null = null;

  picker(ui.picker, order.map((i) => {
    const u = units[i];
    return { label: TECH_NAME[u.tech] ?? u.tech, sub: `${u.name} ${u.unit} · ${u.years}`, value: i };
  }), (i) => { st.i = i; go(); }, 0);

  const narrow = () => ui.screen.getBoundingClientRect().width < 520;
  const plot = canvas(ui.screen, ctx.compact ? 1.9 : narrow() ? 1.15 : 1.55, (c) => draw(c));

  // what the Detective is given
  const seg = h('div', { class: 'pd-seg', role: 'group', 'aria-label': 'How much of the record' }, h('span', {}, 'The Detective sees'));
  const row = h('div', {});
  const sbtns = SPANS.map(([s, n]) => {
    const b = h('button', { type: 'button', 'aria-pressed': String(s === st.span) }, n);
    b.addEventListener('click', () => { st.span = s; sbtns.forEach((x, k) => x.setAttribute('aria-pressed', String(SPANS[k][0] === s))); syncStart(); go(); });
    row.append(b);
    return b;
  });
  seg.append(row);
  ui.controls.append(seg);
  const lastStart = (span: Span) => { // last month whose 30 / 91 days still fit in the record
    const L = span === 'month' ? 720 : 2184;
    for (let m = m1; m >= m0; m--) { const s = t0.findIndex((t) => monthOf(t) === m); if (s >= 0 && s + L <= t0.length) return m; }
    return m0;
  };
  const sStart = slider(ui.controls, {
    label: 'Starting', min: 0, max: 1, step: 1, value: 0,
    fmt: (v) => startLabel(v),
    hint: 'slide the stretch through the eight years',
    onInput: (v) => { if (st.span === 'year') st.year = y0 + v; else st.month = m0 + v; go(); },
  });
  function startLabel(v: number) {
    if (st.span === 'all') return `${y0}–${y1}`;
    if (st.span === 'year') return String(y0 + v);
    return monthName(m0 + v);
  }
  function syncStart() {
    const box = sStart.input.closest('.inst-slider') as HTMLElement;
    box.style.opacity = st.span === 'all' ? '0.35' : '';
    sStart.input.disabled = st.span === 'all';
    if (st.span === 'year') { sStart.input.max = String(y1 - y0); sStart.set(st.year - y0); }
    else if (st.span !== 'all') {
      const mx = lastStart(st.span) - m0;
      st.month = Math.min(st.month, m0 + mx);
      sStart.input.max = String(mx); sStart.set(st.month - m0);
    } else sStart.set(0);
    sStart.input.style.setProperty('--p', `${(+sStart.input.value / Math.max(+sStart.input.max, 1)) * 100}%`);
  }
  let priceBtn: HTMLButtonElement | null = null, ctxBtn: HTMLButtonElement | null = null;
  const btns = h('div', { class: 'pd-btns' });
  ui.controls.append(btns);
  if (!ctx.compact) {
    priceBtn = button(btns, 'Prices: on', () => { st.price = !st.price; syncBtns(); go(); });
    priceBtn.title = 'ERCOT day-ahead hub price and Henry Hub gas for the same hours (the exam fed these too)';
    ctxBtn = button(btns, 'EIA facts: on', () => { st.ctx = !st.ctx; syncBtns(); go(); });
    ctxBtn.title = 'EIA-860/923: prime mover, CHP flag, nameplate, minimum load, cold-start time, online year, annual heat rate';
  }
  button(btns, 'Reset', () => { Object.assign(st, { span: 'all', ctx: true, price: true }); sbtns.forEach((x, k) => x.setAttribute('aria-pressed', String(SPANS[k][0] === 'all'))); syncBtns(); syncStart(); go(); });
  function syncBtns() {
    if (priceBtn) { priceBtn.textContent = `Prices: ${st.price ? 'on' : 'off'}`; priceBtn.setAttribute('aria-pressed', String(st.price)); }
    if (ctxBtn) { ctxBtn.textContent = `EIA facts: ${st.ctx ? 'on' : 'off'}`; ctxBtn.setAttribute('aria-pressed', String(st.ctx)); }
  }
  syncBtns(); syncStart();

  const probBox = h('div', {}), gaugeBox = h('div', {}), kvBox = h('div', { class: 'pd-kv' }), note = h('p', { class: 'pd-note' });
  ui.readout.append(probBox, gaugeBox, kvBox, note);

  const control = (): Control => ({
    span: st.span, start: st.span === 'year' ? st.year : st.month, ctx: st.ctx, price: st.price,
  });

  async function run() {
    ui.setBusy(true);
    await new Promise((r) => requestAnimationFrame(r));
    const t1 = performance.now();
    view = read(W, ser(st.i), units[st.i], control());
    const ms = performance.now() - t1;
    plot.redraw();
    render(view.r);
    const nw = view.r.windows.length;
    ui.stamp(ms, `${units[st.i].id} · ${view.hours.toLocaleString('en-US')} hours read as ${nw} window${nw > 1 ? 's' : ''}`);
    ui.setBusy(false);
  }
  const go = scheduler(run, 30);

  function render(r: Reading) {
    const u = units[st.i], tr = u.truth, M = W.meta;
    probBox.innerHTML = '';
    probs(probBox, st.ctx ? 'Technology (with the EIA-860 prime mover)' : 'Technology, from the output alone',
      M.techs.map((k, j) => ({ label: TECH_NAME[k], p: r.tech[j], truth: k === u.tech })));
    const e = r.est;
    const pc = (k: string) => (tr[k] == null ? null : 100 * (tr[k] as number));
    const all = [
      { label: 'Minimum stable load', unit: '% of capacity', value: 100 * e.pmin_frac.value, lo: 100 * e.pmin_frac.lo68, hi: 100 * e.pmin_frac.hi68, truth: pc('pmin_frac'), range: [0, 100] as [number, number], digits: 0 },
      { label: 'Ramp up, 99th percentile', unit: '% / h', value: 100 * e.ramp_up.value, lo: 100 * e.ramp_up.lo68, hi: 100 * e.ramp_up.hi68, truth: pc('ramp_up'), range: [0, 80] as [number, number], digits: 0 },
      { label: 'Starts per year', unit: '', value: e.starts_yr.value, lo: e.starts_yr.lo68, hi: e.starts_yr.hi68, truth: tr.starts_yr, range: [0.5, 500] as [number, number], digits: 0, log: true },
      { label: 'Median run', unit: 'h', value: e.med_up.value, lo: e.med_up.lo68, hi: e.med_up.hi68, truth: tr.med_up, range: [1, 5000] as [number, number], digits: 0, log: true },
      { label: 'Heat rate, full load', unit: 'MMBtu/MWh', value: e.hr_full.value, lo: e.hr_full.lo68, hi: e.hr_full.hi68, truth: tr.hr_full, range: [5, 18] as [number, number], digits: 1 },
      { label: 'Heat rate, part load', unit: 'MMBtu/MWh', value: e.hr_min.value, lo: e.hr_min.lo68, hi: e.hr_min.hi68, truth: tr.hr_min, range: [5, 20] as [number, number], digits: 1 },
      { label: 'Capacity factor', unit: '%', value: 100 * e.cf.value, lo: 100 * e.cf.lo68, hi: 100 * e.cf.hi68, truth: pc('cf'), range: [0, 100] as [number, number], digits: 0 },
    ];
    gaugeBox.innerHTML = '';
    gauges(gaugeBox, (ctx.compact ? [all[0], all[1], all[4], all[2]] : all).map((g) => ({ ...g, truthLabel: 'CAMPD 2018–25' })));
    const [s, L] = [view!.start, view!.hours];
    const T = ser(st.i).t;
    const d0 = new Date(T[s] * 1000), d1 = new Date(T[s + L - 1] * 1000);
    const ds = (d: Date) => `${d.getUTCDate()} ${MON[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
    kvBox.innerHTML = [
      ['Operated as', `<b class="${r.flex === u.flex_true ? '' : 'out'}">${r.flex}</b>`, `truth ${u.flex_true}`],
      ['Capacity (99.5th pct. of output)', `<b>${Number.isFinite(r.capacityMW) ? `${Math.round(r.capacityMW)} MW` : 'did not run'}</b>`, ''],
      ['Hours given', `<b>${ds(d0)} – ${ds(d1)}</b>`, ''],
    ].map(([k, v, t]) => `<span>${k}${t ? ` <small>(${t})</small>` : ''}</span>${v}`).join('');
    const parts: string[] = [];
    if (!st.price) parts.push('no prices');
    if (!st.ctx) parts.push('no EIA facts');
    note.innerHTML = `<b>${u.name} unit ${u.unit}</b>, EPA CAMPD hourly gross output. Every reading is for the stretch shown${parts.length ? ` (${parts.join(', ')})` : ''}; the truth is the unit’s whole 2018–25 record, as the exam scored it. Bars: 68 % range after the dev widening factors. ${NOTE[u.tech] ?? ''}`;
  }

  function draw(c: Canvas2D) {
    const { ctx: g, w, h: H } = c;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, w, H);
    if (!view) return;
    const S0 = ser(st.i), x = S0.x, n = x.length, r = view.r, u = units[st.i];
    const padL = 40, padR = 12, top = 16, gap = 34, bot = 20;
    const iw = w - padL - padR, ih = H - top - gap - bot;
    const h1 = Math.round(ih * 0.42), h2 = ih - h1, y2 = top + h1 + gap;
    // --- the whole record: hourly output, min/max per pixel column; the stretch given is lit
    let xmax = 1;
    for (let k = 0; k < n; k++) if (x[k] > xmax) xmax = x[k];
    gridLines(c, padL, top, iw, h1, 8, 2);
    const cols = Math.max(1, Math.floor(iw));
    const a0 = view.start, a1 = view.start + view.hours;
    for (let px = 0; px < cols; px++) {
      const i0 = Math.floor((px / cols) * n), i1 = Math.max(i0 + 1, Math.floor(((px + 1) / cols) * n));
      let lo = Infinity, hi = -Infinity;
      for (let k = i0; k < i1; k++) { if (x[k] < lo) lo = x[k]; if (x[k] > hi) hi = x[k]; }
      const lit = i1 > a0 && i0 < a1;
      g.strokeStyle = lit ? ctx.accent : SCOPE.faint;
      g.lineWidth = 1;
      g.beginPath();
      const yl = top + h1 - (hi / (xmax * 1.08)) * h1, yh = top + h1 - (lo / (xmax * 1.08)) * h1;
      g.moveTo(padL + px + 0.5, yl); g.lineTo(padL + px + 0.5, Math.max(yh, yl + 0.8));
      g.stroke();
    }
    const xa = padL + (a0 / n) * iw, xb = padL + (a1 / n) * iw;
    g.fillStyle = 'rgba(235,230,216,0.06)'; g.fillRect(xa, top, Math.max(xb - xa, 1.5), h1);
    label(c, `hourly output, MW (top ${Math.round(xmax)})`, padL + 4, top + 6, SCOPE.text, 'left', 9.5);
    for (let y = y0; y <= y1; y++) {
      const k = Math.max(0, (Date.UTC(y, 0, 1) / 1000 - S0.t[0]) / 3600);
      if (narrow() && (y - y0) % 2) continue;
      label(c, String(y), padL + (k / n) * iw + 2, top + h1 + 9, SCOPE.faint, 'left', 9);
    }
    // --- load-duration curve of the stretch given, as a fraction of the capacity read from it
    const cap = Number.isFinite(r.capacityMW) ? r.capacityMW : xmax;
    const xs = Array.from(x.subarray(a0, a1)).sort((p, q) => q - p);
    const L = xs.length;
    const fy = (f: number) => y2 + h2 - (Math.min(f, 1.2) / 1.2) * h2;
    const fx = (q: number) => padL + q * iw;
    gridLines(c, padL, y2, iw, h2, 10, 3);
    label(c, narrow() ? 'output, sorted: share of hours' : 'output of the stretch given, sorted (load-duration curve), % of capacity', padL + 4, y2 - 9, SCOPE.text, 'left', 9.5);
    for (const f of [0, 0.4, 0.8, 1.2]) label(c, `${Math.round(f * 100)}`, padL - 4, fy(f), SCOPE.faint, 'right', 9);
    for (let q = 0; q <= 1.0001; q += 0.25) label(c, `${Math.round(q * 100)} %`, fx(q), y2 + h2 + 10, SCOPE.faint, q === 0 ? 'left' : q > 0.99 ? 'right' : 'center', 9);
    g.beginPath();
    const step = Math.max(1, Math.floor(L / (iw * 2)));
    for (let k = 0; k < L; k += step) { const X1 = fx(k / L), Y1 = fy(xs[k] / cap); k ? g.lineTo(X1, Y1) : g.moveTo(X1, Y1); }
    g.lineTo(fx(1), fy(xs[L - 1] / cap)); g.lineTo(fx(1), fy(0)); g.lineTo(fx(0), fy(0)); g.closePath();
    g.globalAlpha = 0.16; g.fillStyle = ctx.accent; g.fill(); g.globalAlpha = 1;
    g.strokeStyle = ctx.accent; g.lineWidth = 1.5; g.beginPath();
    for (let k = 0; k < L; k += step) { const X1 = fx(k / L), Y1 = fy(xs[k] / cap); k ? g.lineTo(X1, Y1) : g.moveTo(X1, Y1); }
    g.stroke();
    // minimum stable load: the reading (band) and the CAMPD truth
    const e = r.est.pmin_frac;
    g.fillStyle = 'rgba(235,230,216,0.08)'; g.fillRect(padL, fy(e.hi68), iw, Math.max(fy(e.lo68) - fy(e.hi68), 1));
    g.setLineDash([5, 4]); g.strokeStyle = ctx.accent; g.lineWidth = 1.3;
    g.beginPath(); g.moveTo(padL, fy(e.value)); g.lineTo(padL + iw, fy(e.value)); g.stroke(); g.setLineDash([]);
    label(c, `min stable load read ${Math.round(100 * e.value)} %`, padL + iw - 4, fy(e.value) - 8, ctx.accent, 'right', 9.5);
    const tv = u.truth.pmin_frac;
    if (tv != null) {
      g.strokeStyle = SCOPE.ink; g.lineWidth = 1.2;
      g.beginPath(); g.moveTo(padL, fy(tv)); g.lineTo(padL + iw, fy(tv)); g.stroke();
      label(c, `CAMPD ${Math.round(100 * tv)} %`, padL + iw - 4, fy(tv) + 9, SCOPE.ink, 'right', 9.5);
    }
    const share = xs.filter((v) => v > 0.5).length / Math.max(L, 1);
    label(c, `running ${Math.round(100 * share)} % of hours`, fx(Math.min(Math.max(share, 0.02), 0.98)), fy(1.12), SCOPE.text, share > 0.6 ? 'right' : 'left', 9);
  }

  await run();
}
