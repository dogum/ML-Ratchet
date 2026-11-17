// LoadCast live instrument: 16 real BDG2 exam building-meters (the release page's fixed draw, seed 20261117, stratified
// by meter type), forecast for local 2017 from metadata alone (cold) or from the meter's own 2016 history (warm). The
// browser recomputes the weather, calendar and history features and both networks on every change. The controls change
// the input: how much history the network sees, the outdoor temperature, the operating hours it is told, and a week the
// building is closed. The real 2017 meter and BuildingData's change-point baseline (fitted on 2016) stay on screen.
import type { InstrumentContext } from '../types.ts';
import { fetchPack } from '../../lib/nn/pack.ts';
import { frame, predict, cpBaseline, truth, scores, METERS, type WMeta, type XMeta, type Frame, type Forecast, type Mode, type Variant, type Sched } from './net.ts';
import { shell, picker, slider, button, scheduler, gauges, canvas, gridLines, label, h, SCOPE, type Canvas2D } from '../kit.ts';

const STYLE = `
.lc-sel { display: flex; flex-direction: column; gap: 6px; }
.lc-sel > span { font-size: 12.5px; color: var(--scr-2ink); }
.lc-sel select { appearance: none; -webkit-appearance: none; width: 100%; padding: 7px 28px 7px 9px; border-radius: 4px; border: 1px solid var(--scr-rule); background: transparent; color: var(--scr-ink); font-family: var(--mono); font-variation-settings: 'wdth' 87.5; font-size: 11.5px; cursor: pointer;
  background-image: linear-gradient(45deg, transparent 50%, var(--scr-3ink) 50%), linear-gradient(135deg, var(--scr-3ink) 50%, transparent 50%); background-position: calc(100% - 15px) 50%, calc(100% - 10px) 50%; background-size: 5px 5px; background-repeat: no-repeat; }
.lc-sel select:focus-visible { outline: 2px solid var(--c); outline-offset: 2px; }
.lc-sel option { background: #0d0f0c; color: #ebe6d8; }
.lc-grp { display: flex; flex-direction: column; gap: 6px; }
.lc-grp > span { font-size: 12.5px; color: var(--scr-2ink); }
.lc-grp .inst-picker { margin-bottom: 0; }
.lc-grp small { font-size: 11px; color: var(--scr-3ink); }
.lc-kv { display: grid; grid-template-columns: minmax(0, 1fr) auto auto; gap: 3px 10px; font-size: 12px; color: var(--scr-2ink); align-items: baseline; }
.lc-kv b { font-family: var(--mono); font-variation-settings: 'wdth' 87.5; font-weight: 400; font-size: 11.5px; color: var(--scr-ink); text-align: right; white-space: nowrap; }
.lc-kv b.win { color: var(--c); }
.lc-kv i { font-style: normal; font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.04em; color: var(--scr-3ink); text-align: right; }
.lc-note { font-size: 12px; line-height: 1.5; color: var(--scr-3ink); margin: 0; }
.lc-note b { font-weight: 500; color: var(--scr-2ink); }
`;

const MNAME: Record<string, string> = { electricity: 'electricity', chilledwater: 'chilled water', steam: 'steam', hotwater: 'hot water', gas: 'gas' };
const HIST: { label: string; sub: string; mode: Mode; v: Variant }[] = [
  { label: 'None', sub: 'cold: metadata only', mode: 'cold', v: 'full' },
  { label: '2 weeks', sub: 'last of 2016', mode: 'warm', v: 'w2' },
  { label: '8 weeks', sub: 'last of 2016', mode: 'warm', v: 'w8' },
  { label: 'All 2016', sub: 'a full year', mode: 'warm', v: 'full' },
];
const HOURS: { label: string; v: Sched }[] = [
  { label: 'Unknown', v: 'unknown' }, { label: 'Standard', v: 'standard' }, { label: 'Extended', v: 'extended' }, { label: '24/7', v: '24x7' },
];
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const TRUTH = '#ebe6d8', CP = '#e0a458';
const finite = Number.isFinite;
const kw = (v: number) => (!finite(v) ? '—' : v >= 1000 ? `${(v / 1000).toFixed(v >= 10000 ? 0 : 1)} MW` : `${v.toFixed(v >= 10 ? 0 : 1)} kW`);
const pct = (v: number, d = 1) => (finite(v) ? `${v.toFixed(d)} %` : '—');

function group(el: HTMLElement, title: string, items: { label: string; sub?: string; value: number }[], on: (i: number) => void, start = 0, hint?: string) {
  const box = h('div', { class: 'inst-picker', role: 'group', 'aria-label': title });
  el.append(h('div', { class: 'lc-grp' }, h('span', {}, title), box, ...(hint ? [h('small', {}, hint)] : [])));
  return picker(box, items, (v) => on(v), start);
}

export async function mount(ctx: InstrumentContext) {
  const [W, X] = await Promise.all([fetchPack<WMeta>(`${ctx.base}weights`), fetchPack<XMeta>(`${ctx.base}examples`)]);
  if (!document.getElementById('lc-inst-style')) document.head.append(h('style', { id: 'lc-inst-style' }, STYLE));
  const M = X.meta, U = M.units, lead = M.lead;
  const np = W.meta.n_params.cold + W.meta.n_params.warm;
  const ui = shell(ctx.root, {
    title: 'LoadCast',
    subtitle: 'a building’s hourly energy for 2017, from weather, calendar and what is known about it — a real exam meter, never used in training',
    params: `2 networks, ${Math.round(np / 1000)}k parameters · matches Python to 1e-5`,
  });
  const byM = (m: string) => U.map((_, i) => i).filter((i) => U[i].meter === m);
  const st = { m: 'electricity', u: byM('electricity')[0], hist: 3, wk: 27, dT: 0, hours: 0, closed: false };

  picker(ui.picker, METERS.map((m) => ({ label: MNAME[m], sub: `${byM(m).length} meter${byM(m).length > 1 ? 's' : ''} · exam n ${M.exam[m].n}`, value: m })), (m) => {
    st.m = m; st.u = byM(m)[0]; fill(); go(true);
  }, 0);

  const narrow = () => ui.screen.getBoundingClientRect().width < 520;
  // State the canvas draw reads: declared before canvas(), which draws at once.
  let F: Frame | null = null, fKey = '';
  let week: { a: number; f: Forecast } | null = null;
  let yr: { key: string; f: Forecast; cp: Float64Array } | null = null;
  let job = 0;
  const plot = canvas(ui.screen, ctx.compact ? 1.6 : narrow() ? 1.0 : 1.55, (c) => draw(c));

  // ---- controls
  const sel = h('select', { 'aria-label': 'Exam building-meter' });
  ui.controls.append(h('label', { class: 'lc-sel' }, h('span', {}, 'Exam building (BDG2)'), sel));
  sel.addEventListener('change', () => { st.u = +sel.value; go(true); });
  function fill() {
    sel.innerHTML = '';
    for (const i of byM(st.m)) {
      const u = U[i];
      const o = h('option', { value: String(i) }, `${u.bid.replace(/_/g, ' ')} · ${Math.round(u.area).toLocaleString('en-US')} m² · ${M.sites[u.site].tz}`);
      if (i === st.u) o.selected = true;
      sel.append(o);
    }
  }
  fill();
  group(ui.controls, 'History the network sees', HIST.map((x, i) => ({ label: x.label, sub: x.sub, value: i })), (i) => { st.hist = i; go(true); }, st.hist,
    'warm mode corrects a time-of-week-and-temperature model fitted on this history');
  const wkName = (wk: number) => {
    const i = lead + 168 * wk, s = M.units[st.u].site, c = X.t.cal.data;
    const mo = c[(s * 5 + 3) * 9480 + i], md = c[(s * 5 + 4) * 9480 + i];
    return `${MON[mo - 1]} ${md}`;
  };
  slider(ui.controls, {
    label: 'Week of 2017 shown', min: 0, max: 51, step: 1, value: st.wk,
    fmt: (v) => `from ${wkName(v)}`,
    onInput: (v) => { st.wk = v; go(st.closed); },
  });
  const sT = slider(ui.controls, {
    label: 'Outdoor temperature', min: -8, max: 8, step: 0.5, value: 0,
    fmt: (v) => (v === 0 ? 'as recorded' : `${v > 0 ? '+' : ''}${v.toFixed(1)} °C all year`),
    hint: 'added to every hour of 2017 (dewpoint capped at it); the annual climate inputs follow',
    onInput: (v) => { st.dT = v; go(true); },
  });
  let gH: ReturnType<typeof picker> | null = null, gC: ReturnType<typeof picker> | null = null;
  if (!ctx.compact) {
    gH = group(ui.controls, 'Operating hours it is told', HOURS.map((x, i) => ({ label: x.label, value: i })), (i) => { st.hours = i; go(true); }, 0,
      'a metadata input; BDG2 gives none, so the exam ran with “unknown”');
    gC = group(ui.controls, 'This week', [{ label: 'As recorded', value: 0 }, { label: 'Closed (holiday)', value: 1 }], (i) => { st.closed = i === 1; go(true); }, 0,
      'flags every day of the shown week as a holiday; the real meter stays as it was');
    button(ui.controls, 'Reset', () => {
      Object.assign(st, { dT: 0, hours: 0, closed: false });
      sT.set(0); gH!.set(0); gC!.set(0); go(true);
    });
  }

  const gaugeBox = h('div', {}), kv = h('div', { class: 'lc-kv' }), note = h('p', { class: 'lc-note' });
  ui.readout.append(gaugeBox, kv, note);

  // ---- state shared by the runs
  const cfg = () => ({ ...HIST[st.hist], sched: HOURS[st.hours].v });
  const yKey = () => `${st.u}|${st.hist}|${st.dT}|${st.hours}|${st.closed ? st.wk : -1}`;

  async function run(all: boolean) {
    ui.setBusy(true);
    await new Promise((r) => requestAnimationFrame(r));
    const u = U[st.u], c = cfg();
    const k = `${u.site}|${st.dT}|${st.closed ? st.wk : -1}`;
    const t0 = performance.now();
    if (k !== fKey) { F = frame(X, u.site, st.dT, st.closed ? st.wk : null); fKey = k; }
    const a = 168 * st.wk;
    week = { a, f: predict(W, X, F!, st.u, a, a + 168, c.mode, c.v, c.sched) };
    const ms = performance.now() - t0;
    if (all || !yr || yr.key !== yKey()) { yr = null; year(); }
    plot.redraw();
    render();
    ui.stamp(ms, `one week, ${c.mode === 'cold' ? 'cold' : 'warm'}`);
    ui.setBusy(false);
  }
  // the whole of 2017 for the scores, in slices so the page stays responsive
  function year() {
    const my = ++job, key = yKey(), c = cfg(), u = st.u, Fy = F!;
    const kW = new Float64Array(8760), sd = new Float64Array(8760);
    let t = 0, k = 1;
    const step = () => {
      if (my !== job) return;
      const b = Math.min(t + 730, 8760), r = predict(W, X, Fy, u, t, b, c.mode, c.v, c.sched);
      kW.set(r.kW, t); sd.set(r.sd, t); k = r.k; t = b;
      if (t < 8760) { setTimeout(step, 0); return; }
      yr = { key, f: { kW, sd, k }, cp: cpBaseline(X, Fy, u) };
      plot.redraw(); render();
    };
    setTimeout(step, 20);
  }
  let pendingAll = false;
  const sched = scheduler(() => { const a = pendingAll; pendingAll = false; return run(a); }, 40);
  const go = (all: boolean) => { pendingAll ||= all; sched(); };

  function render() {
    if (!week) return;
    const u = U[st.u], c = cfg(), y = truth(X, st.u), ex = M.exam[u.meter];
    const yw = y.subarray(week.a, week.a + 168);
    const sw = scores(yw, week.f.kW, week.f.sd, W.meta.z90);
    let mw = 0;
    for (let i = 0; i < 168; i++) mw += week.f.kW[i];
    mw /= 168;
    const gs = [{ label: `Mean load, week from ${wkName(st.wk)}`, unit: 'kW', value: mw, truth: finite(sw.mean) ? sw.mean : null,
      truthLabel: 'real meter', range: [0, Math.max(mw, sw.mean || 0) * 1.6 || 1] as [number, number], digits: mw < 10 ? 1 : 0 }];
    gaugeBox.innerHTML = '';
    gauges(gaugeBox, gs);
    const rows = ['<span></span><i>LOADCAST</i><i>CHANGE POINT</i>'];
    if (yr) {
      const s = scores(y, yr.f.kW, yr.f.sd, W.meta.z90), b = scores(y, yr.cp);
      const best = (x: number, o: number) => (x <= o ? 'win' : '');
      rows.push(`<span>2017 CV(RMSE), hourly</span><b class="${best(s.cv, b.cv)}">${pct(s.cv)}</b><b class="${best(b.cv, s.cv)}">${pct(b.cv)}</b>`);
      rows.push(`<span>2017 NMBE (+ = real higher)</span><b>${pct(s.nmbe)}</b><b>${pct(b.nmbe)}</b>`);
      rows.push(`<span>2017 hours inside the 90 % band</span><b>${pct(100 * s.cov, 0)}</b><b>—</b>`);
      rows.push(`<span>2017 mean load (real ${kw(s.mean)})</span><b>${kw(mean(yr.f.kW))}</b><b>${kw(mean(yr.cp))}</b>`);
    } else rows.push('<span>2017 scores</span><b>computing…</b><b></b>');
    rows.push(`<span>This week: hours inside the 90 % band</span><b>${finite(sw.cov) ? `${Math.round(sw.cov * sw.n)} of ${sw.n}` : '—'}</b><b></b>`);
    kv.innerHTML = rows.join('');
    const mkey = c.mode === 'cold' ? 'cold' : `warm_${c.v}`;
    const mexam = (ex as unknown as Record<string, number>)[mkey];
    const wnote = c.mode === 'warm' && c.v === 'w2' && (u.meter === 'chilledwater' || u.meter === 'gas')
      ? ' Two weeks of December history cannot say how this meter behaves in summer: its dev widening hit the cap (×30), so the band is very wide.' : '';
    const cnote = c.mode === 'cold'
      ? ' With no history it knows only use, floor area, the meters present and the climate: on the exam cold forecasts lose to a median-intensity guess on electricity and chilled water, and run about 2× low on campus steam and chilled water.' : '';
    const changed: string[] = [];
    if (st.dT) changed.push(`${st.dT > 0 ? '+' : ''}${st.dT} °C`);
    if (st.hours) changed.push(`told ${HOURS[st.hours].label.toLowerCase()} hours`);
    if (st.closed) changed.push('closed this week');
    note.innerHTML = (changed.length
      ? `<b>Changed: ${changed.join(', ')}.</b> The real meter is 2017 as it happened, so read the change in the forecast, not its error. `
      : '<b>As recorded</b> (2017 weather as measured). ')
      + `Exam, ${ex.n} ${MNAME[u.meter]} meters, median CV(RMSE): LoadCast ${HIST[st.hist].label.toLowerCase()} ${pct(mexam)}, TOWT ${pct(ex.towt)}, change point ${pct(ex.cp)}, median-intensity guess ${pct(ex.dumb)}.${cnote}${wnote}`
      + ' Weather is stored to 0.01 °C, so readings match the release to about 3e-4.';
  }
  const mean = (a: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; };

  function draw(c: Canvas2D) {
    const { ctx: g, w, h: H } = c;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, w, H);
    if (!week) return;
    const u = U[st.u], y = truth(X, st.u), a = week.a, f = week.f, z = W.meta.z90;
    const cpw = yr ? yr.cp.subarray(a, a + 168) : null;
    const small = w < 520, padL = 46, padR = 10, top = 20, gap = 30, bot = 18;
    const iw = w - padL - padR, hy = ctx.compact ? 0 : Math.round((H - top - bot - gap) * 0.26), hw = H - top - bot - (hy ? hy + gap : 0);
    // week panel
    let hi = 0;
    for (let i = 0; i < 168; i++) { hi = Math.max(hi, f.kW[i] + z * f.sd[i], finite(y[a + i]) ? y[a + i] : 0, cpw ? cpw[i] : 0); }
    const cap = Math.max(...Array.from(y.subarray(a, a + 168)).filter(finite), mean(f.kW)) * 3;
    hi = Math.min(hi, cap > 0 ? cap : hi) * 1.05 || 1;
    const xs = (i: number) => padL + (i / 167) * iw, ys = (v: number) => top + hw - (Math.min(v, hi) / hi) * hw;
    gridLines(c, padL, top, iw, hw, 7, 4);
    label(c, small ? `${MNAME[u.meter]}, kW` : `${MNAME[u.meter]} kW: LoadCast and 90 % band · real meter · change point (2016 fit)`, padL + 4, top - 9, SCOPE.text, 'left', 9.5);
    for (let j = 0; j <= 4; j++) label(c, kw((hi * j) / 4).replace(' kW', '').replace(' MW', 'M'), padL - 4, ys((hi * j) / 4), SCOPE.faint, 'right', 9);
    g.beginPath();
    for (let i = 0; i < 168; i++) i ? g.lineTo(xs(i), ys(f.kW[i] + z * f.sd[i])) : g.moveTo(xs(i), ys(f.kW[i] + z * f.sd[i]));
    for (let i = 167; i >= 0; i--) g.lineTo(xs(i), ys(Math.max(f.kW[i] - z * f.sd[i], 0)));
    g.closePath(); g.globalAlpha = 0.2; g.fillStyle = ctx.accent; g.fill(); g.globalAlpha = 1;
    const line = (v: ArrayLike<number>, col: string, lw: number, dash: number[] = [], x = xs, yy = ys, n = v.length) => {
      g.strokeStyle = col; g.lineWidth = lw; g.setLineDash(dash); g.beginPath();
      let on = false;
      for (let i = 0; i < n; i++) { if (!finite(v[i])) { on = false; continue; } on ? g.lineTo(x(i), yy(v[i])) : g.moveTo(x(i), yy(v[i])); on = true; }
      g.stroke(); g.setLineDash([]);
    };
    if (cpw) line(cpw, CP, 1.1, [5, 3]);
    line(y.subarray(a, a + 168), TRUTH, 1.2);
    line(f.kW, ctx.accent, 1.8);
    const s = u.site, cal = X.t.cal.data;
    for (let d = 0; d < 7; d++) {
      const i = lead + a + d * 24 + 12;
      const dw = cal[(s * 5 + 1) * 9480 + i], md = cal[(s * 5 + 4) * 9480 + i], hol = cal[(s * 5 + 2) * 9480 + i];
      label(c, small ? `${DAYS[dw][0]}${md}` : `${DAYS[dw]} ${md}${hol ? ' hol' : ''}`, xs(d * 24 + 12), top + hw + 9, st.closed || hol ? ctx.accent : SCOPE.faint, 'center', 9);
    }
    if (!hy) return;
    // year strip: daily means
    const y0 = top + hw + gap, nd = 365;
    const R = new Float64Array(nd).fill(NaN), P = new Float64Array(nd).fill(NaN), B = new Float64Array(nd).fill(NaN);
    let ym = 0;
    for (let d = 0; d < nd; d++) {
      let sr = 0, cr = 0;
      for (let k = 0; k < 24; k++) { const v = y[d * 24 + k]; if (finite(v)) { sr += v; cr++; } }
      if (cr >= 20) R[d] = sr / cr;
      if (yr) { P[d] = mean(yr.f.kW.subarray(d * 24, d * 24 + 24)); B[d] = mean(yr.cp.subarray(d * 24, d * 24 + 24)); }
      ym = Math.max(ym, finite(R[d]) ? R[d] : 0, finite(P[d]) ? P[d] : 0);
    }
    ym = ym * 1.08 || 1;
    const xd = (d: number) => padL + (d / (nd - 1)) * iw, yd = (v: number) => y0 + hy - (Math.min(v, ym) / ym) * hy;
    gridLines(c, padL, y0, iw, hy, 12, 2);
    g.fillStyle = ctx.accent; g.globalAlpha = 0.15; g.fillRect(xd(st.wk * 7), y0, Math.max(2, xd(st.wk * 7 + 7) - xd(st.wk * 7)), hy); g.globalAlpha = 1;
    label(c, yr ? 'daily mean, all of 2017 (shaded: the week above)' : 'daily mean, 2017: computing the forecast…', padL + 4, y0 - 9, SCOPE.text, 'left', 9.5);
    label(c, kw(ym).replace(' kW', '').replace(' MW', 'M'), padL - 4, y0 + 4, SCOPE.faint, 'right', 9);
    if (yr) line(B, CP, 1, [4, 3], xd, yd);
    line(R, TRUTH, 1.1, [], xd, yd);
    if (yr) line(P, ctx.accent, 1.4, [], xd, yd);
    for (let mo = 0; mo < 12; mo += small ? 3 : 1) label(c, MON[mo], xd(mo * 30.4 + 15), y0 + hy + 9, SCOPE.faint, 'center', 9);
  }

  await run(true);
}
