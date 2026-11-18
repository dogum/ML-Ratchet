// SnowSense: pick a real exam station and water year, read the pillow's SWE from the station's own gauge and
// thermometer, and change the weather the network reads. The network and two uncalibrated physics routines rerun live.
import { fetchPack, type Pack } from '../../lib/nn/pack.ts';
import type { InstrumentContext } from '../types.ts';
import { shell, picker, slider, button, gauges, canvas, scheduler, h } from '../kit.ts';
import {
  degreeDay, energyBalance, isRecorded, makeWindow, nse, observed, observedYear, precipTotal, read, snowShare,
  RECORDED, type Reading, type Scenario, type SnowWeights, type Stations, type Window,
} from './net.ts';
import { AMBER, LILAC, lighten, dayAt, decadeLayout, drawDecade, drawScope, type DecadeYear, type ScopeState } from './scope.ts';

const winterT = (w: Window) => { let s = 0; for (let t = 61; t < 61 + 182; t++) s += w.Tm[t]; return s / 182; };
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const DAY = 864e5, WARM = 61;

export async function mount(ctx: InstrumentContext) {
  const [weights, stations] = await Promise.all([
    fetchPack<SnowWeights>(ctx.base + 'weights'), fetchPack<Stations>(ctx.base + 'stations')]);
  show(ctx, weights, stations);
}

export function show(ctx: InstrumentContext, weights: Pack<SnowWeights>, stations: Pack<Stations>) {
  const list = stations.meta.stations, exam = stations.meta.exam;
  const ui = shell(ctx.root, {
    title: 'SnowSense',
    subtitle: 'snow water equivalent at a real exam pillow, from its own gauge and thermometer',
    params: 'GRU 2×64, 51k weights',
  });
  ctx.root.classList.add('ss-explorer');
  ctx.root.style.setProperty('--c', ctx.accent);
  ctx.root.append(h('style', {}, `
    .ss-key { display: flex; flex-wrap: wrap; gap: 6px 14px; padding: 7px 10px; font: 10.5px/1.5 var(--mono, monospace); color: var(--scr-2ink); }
    .ss-key button { all: unset; cursor: pointer; display: inline-flex; align-items: center; gap: 5px; opacity: .45; }
    .ss-key button[aria-pressed="true"], .ss-key span { opacity: 1; }
    .ss-key i { display: inline-block; width: 14px; border-top: 2px solid var(--k); }
    .ss-key i.dash { border-top-style: dashed; } .ss-key i.band { border-top-width: 7px; opacity: .55; }
    .ss-day { min-height: 2.9em; padding: 4px 10px 7px; font: 10.5px/1.45 var(--mono, monospace); color: var(--scr-2ink); border-top: 1px solid var(--scr-rule); }
    .ss-day b { font-weight: 400; color: var(--scr-ink); }
    .ss-head { margin: 0; color: var(--scr-ink); font: 11px/1.45 var(--mono, monospace); }
    .ss-head small { display: block; color: var(--scr-3ink); font-size: 10px; }
    .ss-tbl { width: 100%; border-collapse: collapse; font: 10.5px/1.5 var(--mono, monospace); color: var(--scr-2ink); }
    .ss-tbl th { font-weight: 400; text-align: right; color: var(--scr-3ink); font-size: 9.5px; text-transform: uppercase; letter-spacing: .05em; padding: 0 0 2px 6px; }
    .ss-tbl th:first-child, .ss-tbl td:first-child { text-align: left; padding-left: 0; }
    .ss-tbl td { padding: 1px 0 1px 6px; text-align: right; color: var(--scr-ink); white-space: nowrap; }
    .ss-tbl td:first-child { color: var(--scr-2ink); white-space: normal; }
    .ss-tbl tr.me td { color: var(--g); } .ss-tbl tr.truth td { color: var(--scr-ink); }
    .ss-tbl .in { color: color-mix(in srgb, #4cc38a 85%, white); } .ss-tbl .out { color: #ff8a7a; }
    .ss-h { font: 10px/1.4 var(--mono, monospace); text-transform: uppercase; letter-spacing: .06em; color: var(--scr-3ink); margin: 0 0 3px; }
    .ss-note { font-size: 12px; line-height: 1.5; color: var(--scr-2ink); margin: 0; }
    .ss-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: end; }
    .ss-explorer .inst-btn[aria-pressed="true"] { background: var(--c); color: #fff; border-color: var(--c); }
    .ss-explorer .inst-pick[aria-pressed='true'], .ss-explorer .inst-pick[aria-pressed='true'] small { color: #fff; }
    .ss-decade { border-top: 1px solid var(--scr-rule); cursor: pointer; }
  `));

  // ---- state
  let si = 0, wy = 2019, what: Scenario = { ...RECORDED }, hover = -1;
  const show3 = { eb: true, dd: false, band90: true };
  let w: Window, r: Reading, eb: Float64Array, dd: Float64Array, obs: Float64Array, oy = observedYear(new Float64Array(0));
  let rec = { share: 0, total: 0, tmean: 0 };
  let decade: DecadeYear[] = [];

  // ---- screen
  const box = h('div', { role: 'img', 'aria-label': 'Snow water equivalent through one water year: SnowSense with its bands, the observed pillow and two physics routines; below, the daily temperature and precipitation it read' });
  const key = h('div', { class: 'ss-key' });
  const dayLine = h('div', { class: 'ss-day', 'aria-live': 'off' });
  const decBox = h('div', { class: 'ss-decade', role: 'img', 'aria-label': 'Peak SWE for every water year 2016-2025: SnowSense and the observed pillow; click a year to open it' });
  ui.screen.append(box, key, dayLine, decBox);
  const glow = lighten(ctx.accent, 0.3);   // the accent, lifted for lines on the dark scope
  ctx.root.style.setProperty("--g", glow);
  const state = (): ScopeState => ({ w, r, obs, eb, dd, obsMelt: oy.melt, obsOk: oy.ok, modified: !isRecorded(what), show: show3, hover, accent: glow });
  let ready = false;
  const plot = canvas(box, ctx.compact ? 1.2 : 1.32, c => { if (ready) drawScope(c, state()); });
  const dec = canvas(decBox, ctx.compact ? 5.5 : 7, c => { if (decade.length) drawDecade(c, decade, wy, glow); });
  const cv = box.querySelector('canvas')!, dcv = decBox.querySelector('canvas')!;
  const dims = (el: HTMLCanvasElement) => { const b = el.getBoundingClientRect(); return { cv: el, ctx: el.getContext('2d')!, w: b.width, h: b.height }; };
  cv.addEventListener('pointermove', ev => { hover = dayAt(dims(cv), ev.clientX - cv.getBoundingClientRect().left); dayInfo(); plot.redraw(); });
  cv.addEventListener('pointerleave', () => { hover = -1; dayInfo(); plot.redraw(); });
  dcv.addEventListener('click', ev => {
    const c = dims(dcv), Lz = decadeLayout(c, decade.length);
    const i = Math.floor((ev.clientX - dcv.getBoundingClientRect().left - Lz.left) / Lz.col);
    if (i >= 0 && i < decade.length) { wy = decade[i].wy; yearS.set(wy); go(); }
  });

  const keyItems: [keyof typeof show3 | null, string, string, string][] = [
    [null, 'Observed pillow', 'rgba(235,230,216,.95)', ''], [null, 'SnowSense median, 68%', ctx.accent, 'band'],
    ['band90', '90% band', ctx.accent + '88', 'band'],
    ['eb', 'Energy balance, default (exam 0.853)', AMBER, 'dash'], ['dd', 'Degree-day, default (exam 0.691)', LILAC, 'dash'],
  ];
  for (const [flag, text, color, cls] of keyItems) {
    const sw = h('i', { class: cls, style: `--k:${color}` });
    if (!flag) { key.append(h('span', {}, sw, text)); continue; }
    const b = h('button', { type: 'button', 'aria-pressed': String(show3[flag]) }, sw, text);
    b.addEventListener('click', () => { show3[flag] = !show3[flag]; b.setAttribute('aria-pressed', String(show3[flag])); plot.redraw(); readout(); });
    key.append(b);
  }

  // ---- readout
  const head = h('p', { class: 'ss-head' });
  const gauge = h('div');
  const dates = h('table', { class: 'ss-tbl' });
  const board = h('table', { class: 'ss-tbl' });
  const weather = h('table', { class: 'ss-tbl' });
  const note = h('p', { class: 'ss-note' });
  ui.readout.append(head, gauge, dates, h('div', {}, h('p', { class: 'ss-h' }, 'This water year'), board),
    h('div', {}, h('p', { class: 'ss-h' }, 'The weather it read'), weather), note);

  // ---- controls
  const yearS = slider(ui.controls, {
    label: 'Water year (1 Oct – 30 Sep)', min: 2016, max: 2025, step: 1, value: wy, fmt: v => 'WY' + v,
    onInput: v => { wy = v; go(); },
  });
  const warmS = slider(ui.controls, {
    label: 'Warm every day by', min: -2, max: 4, step: 0.25, value: 0,
    fmt: v => (v === 0 ? 'as recorded' : (v > 0 ? '+' : '−') + Math.abs(v).toFixed(2) + ' °C'),
    hint: 'Watch rain replace snow in the bottom strip and melt-out come earlier.',
    onInput: v => { what.warm = v; go(); },
  });
  const pS = slider(ui.controls, {
    label: 'Scale the gauge precipitation ×', min: 0.6, max: 1.4, step: 0.05, value: 1,
    fmt: v => (v === 1 ? 'as recorded' : '×' + v.toFixed(2)),
    hint: 'Gauges under-catch windblown snow. SnowSense’s peaks are 9% low on the exam.',
    onInput: v => { what.pscale = v; go(); },
  });
  const zS = ctx.compact ? null : slider(ui.controls, {
    label: 'Move the station', min: -800, max: 800, step: 50, value: 0,
    fmt: v => (v === 0 ? 'as built' : (v > 0 ? '+' : '−') + Math.abs(v) + ' m'),
    hint: 'With the lapse rate on, the air cools 6.5 °C per km as you climb.',
    onInput: v => { what.dz = v; go(); },
  });
  const gapS = ctx.compact ? null : slider(ui.controls, {
    label: '30-day gauge outage from', min: 0, max: stations.meta.gap_months.length - 1, step: 1, value: 0,
    fmt: v => (v === 0 ? 'none' : '1 ' + MONTHS[stations.meta.gap_months[v]! - 1]),
    hint: 'Missing days read as no precipitation and interpolated temperature, flagged to the network.',
    onInput: v => { what.gap = v; go(); },
  });
  const row = h('div', { class: 'ss-row' });
  ui.controls.append(row);
  let lapseB: HTMLButtonElement | null = null, dropB: HTMLButtonElement | null = null;
  if (!ctx.compact) {
    lapseB = button(row, 'Lapse rate on', () => { what.lapse = !what.lapse; toggles(); go(); });
    dropB = button(row, 'Drop the biggest storm', () => { what.drop = !what.drop; toggles(); go(); });
  }
  button(row, 'Recorded weather', () => {
    what = { ...RECORDED }; warmS.set(0); pS.set(1); zS?.set(0); gapS?.set(0); toggles(); go();
  });
  function toggles() {
    if (lapseB) { lapseB.setAttribute('aria-pressed', String(what.lapse)); lapseB.textContent = what.lapse ? 'Lapse rate on' : 'Lapse rate off'; }
    if (dropB) dropB.setAttribute('aria-pressed', String(what.drop));
  }
  toggles();

  picker(ui.picker, list.map((s, i) => ({ label: `${s.name}, ${s.state}`, sub: `${s.region} · ${Math.round(s.elev_m)} m`, value: i })), i => {
    si = i; decade = []; go(true);
  });

  // ---- helpers
  const dateOf = (d: number, base = w.t0 + WARM * DAY) => {
    if (d < 0) return '—';
    const t = new Date(base + d * DAY);
    return t.getUTCDate() + ' ' + MONTHS[t.getUTCMonth()];
  };
  const mm = (v: number) => (Number.isFinite(v) ? Math.round(v).toString() : '—');

  function computeDecade() {
    decade = list[si].wys.map(y => {
      const ww = makeWindow(stations, si, y), rr = read(weights, ww), o = observed(stations, si, y), oyy = observedYear(o);
      return { wy: y, peak: rr.peak, lo: rr.peakLo68, hi: rr.peakHi68, obs: oyy.peak, ok: oyy.ok, nse: nse(o, rr.swe) };
    });
  }

  function dayInfo() {
    if (hover < 0 || !ready) {
      dayLine.innerHTML = 'Move over the chart for a day’s reading. Day D is the end of the day; the pillow’s next-morning value is the truth.';
      return;
    }
    const d = hover - WARM, t = new Date(w.t0 + hover * DAY);
    const when = t.getUTCDate() + ' ' + MONTHS[t.getUTCMonth()] + ' ' + t.getUTCFullYear();
    const wx = `P <b>${w.P[hover].toFixed(1)}</b> mm · T <b>${w.Tm[hover].toFixed(1)}</b> °C (${w.Tn[hover].toFixed(0)} to ${w.Tx[hover].toFixed(0)})` +
      (w.mP[hover] || w.mT[hover] ? ' · gap flagged' : '');
    if (d < 0) { dayLine.innerHTML = `${when} · lead-in, not scored · ${wx}`; return; }
    dayLine.innerHTML = `${when} · SnowSense <b>${mm(r.swe[d])}</b> mm [${mm(r.lo68[d])}–${mm(r.hi68[d])}]` +
      (Number.isFinite(obs[d]) ? ` · pillow <b>${mm(obs[d])}</b>${isRecorded(what) ? '' : ' (recorded weather)'}` : '') +
      (show3.eb ? ` · EB ${mm(eb[d])}` : '') + (show3.dd ? ` · DD ${mm(dd[d])}` : '') + `<br>${wx}`;
  }

  function readout() {
    const mod = !isRecorded(what), st = list[si];
    const moved = what.dz ? ` · moved to ${Math.round(st.elev_m + what.dz)} m${what.lapse ? '' : ' (temperatures not moved)'}` : '';
    head.innerHTML = `WY${wy} · ${st.name}, ${st.state} · ${Math.round(st.elev_m)} m${moved}<small>exam station, never seen in training; SnowSense reads no snow measurement</small>`;
    const truth = !mod && oy.ok ? oy.peak : null;
    gauges(gauge, [{ label: 'Peak SWE', unit: 'mm', value: r.peak, lo: r.peakLo68, hi: r.peakHi68, truth, truthLabel: 'Observed peak',
      digits: 0, range: [0, Math.max(r.peakHi68, truth ?? 0, r.peak * 1.6, 50) * 1.05] }]);

    const inside = (v: number, a: number, b: number) => v >= (a < 0 ? -1e9 : a) && v <= (b < 0 ? 1e9 : b);
    const dRows: (string | Node)[][] = [['', 'SnowSense', mod ? '' : 'Pillow']];
    const obsPeakDay = oy.ok ? oy.peakDay : -1;
    dRows.push(['Peak date', dateOf(r.peakDay), mod || !oy.ok ? '' : dateOf(obsPeakDay)]);
    const mi = r.melt >= 0 ? `${dateOf(r.melt)}` : 'none';
    const iv = `${dateOf(r.meltLo68)} – ${r.meltHi68 >= 0 ? dateOf(r.meltHi68) : 'later'}`;
    let mcell: string | Node = mod || !oy.ok ? '' : dateOf(oy.melt);
    if (!mod && oy.ok && oy.melt >= 0) {
      const ok = inside(oy.melt, r.meltLo68, r.meltHi68);
      mcell = h('span', { class: ok ? 'in' : 'out', title: ok ? 'inside the 68% interval' : 'outside the 68% interval' }, dateOf(oy.melt) + (ok ? ' ✓' : ' ✗'));
    }
    dRows.push(['Melt-out (below 5 mm)', mi, mcell]);
    dRows.push(['  68% interval', iv, '']);
    dates.replaceChildren(h('tr', {}, ...dRows[0].map(v => h('th', {}, v as string))),
      ...dRows.slice(1).map(cells => h('tr', {}, ...cells.map(v => h('td', {}, v)))));

    const melt = (s: ArrayLike<number>) => {
      let im = 0; for (let i = 1; i < s.length; i++) if (s[i] > s[im]) im = i;
      for (let i = im; i < s.length; i++) if (s[i] < 5) return i;
      return -1;
    };
    const peakOf = (s: ArrayLike<number>) => Math.max(...Array.from(s));
    const rows: [string, number, number, number, string][] = [
      ['SnowSense', r.peak, r.melt, nse(obs, r.swe), 'me'],
      ['Energy balance, default', peakOf(eb), melt(eb), nse(obs, eb), ''],
      ['Degree-day, default', peakOf(dd), melt(dd), nse(obs, dd), ''],
    ];
    const head2 = h('tr', {}, h('th', {}, ''), h('th', {}, 'peak mm'), h('th', {}, 'melt-out'), h('th', {}, mod ? '' : 'NSE'));
    const body = rows.map(([n, p, m, e, cls]) => h('tr', cls ? { class: cls } : {}, h('td', {}, n), h('td', {}, mm(p)), h('td', {}, dateOf(m)), h('td', {}, mod ? '' : Number.isFinite(e) ? e.toFixed(2) : '—')));
    if (!mod && oy.ok) body.push(h('tr', { class: 'truth' }, h('td', {}, 'Pillow (truth)'), h('td', {}, mm(oy.peak)), h('td', {}, dateOf(oy.melt)), h('td', {}, '')));
    board.replaceChildren(head2, ...body);

    const share = snowShare(w), total = precipTotal(w);
    const pct = (v: number) => Math.round(100 * v) + '%';
    weather.replaceChildren(
      h('tr', {}, h('th', {}, ''), h('th', {}, 'recorded'), h('th', {}, mod ? 'what-if' : '')),
      h('tr', {}, h('td', {}, 'Water-year precipitation'), h('td', {}, mm(rec.total) + ' mm'), h('td', {}, mod ? mm(total) + ' mm' : '')),
      h('tr', {}, h('td', {}, 'Oct–Jun share falling as snow'), h('td', {}, pct(rec.share)), h('td', {}, mod ? pct(share) : '')),
      h('tr', {}, h('td', {}, 'Mean T, Oct–Mar'), h('td', {}, rec.tmean.toFixed(1) + ' °C'), h('td', {}, mod ? winterT(w).toFixed(1) + ' °C' : '')),
    );

    const parts: string[] = [];
    if (what.warm) parts.push(`every day ${what.warm > 0 ? 'warmer' : 'colder'} by ${Math.abs(what.warm)} °C`);
    if (what.pscale !== 1) parts.push(`precipitation ×${what.pscale.toFixed(2)}`);
    if (what.dz) parts.push(`station ${what.dz > 0 ? 'up' : 'down'} ${Math.abs(what.dz)} m${what.lapse ? '' : ', only the elevation number changed'}`);
    if (what.drop) parts.push('the wettest three days removed');
    if (what.gap) parts.push('a 30-day gauge outage');
    note.textContent = mod
      ? `What-if: ${parts.join('; ')}. The network and both physics routines reran on this weather. The dotted line is the pillow under the weather that really happened, so no truth is scored.`
      : `On the 621-station exam SnowSense scored median NSE ${exam.nse.toFixed(3)}; the default energy balance ${exam.eb.toFixed(3)}; a degree-day model calibrated on each station's own snow (it reads ten years of SWE) ${exam.dd_cal.toFixed(3)}.`;
  }

  function run(newStation = false) {
    const t0 = performance.now();
    if (newStation || !decade.length) computeDecade();
    const t1 = performance.now();
    w = makeWindow(stations, si, wy, what);
    r = read(weights, w);
    eb = energyBalance(w, stations.meta.eb_default);
    dd = degreeDay(w, stations.meta.dd_default);
    obs = observed(stations, si, wy);
    oy = observedYear(obs);
    const base = makeWindow(stations, si, wy);
    rec = { share: snowShare(base), total: precipTotal(base), tmean: winterT(base) };
    const ms = performance.now() - t1;
    ready = true;
    readout(); dayInfo(); plot.redraw(); dec.redraw();
    ui.stamp(newStation ? performance.now() - t0 : ms, newStation
      ? `all ${decade.length} water years of this station, then WY${wy} with both physics routines`
      : isRecorded(what) ? `WY${wy}, 427 days, with both physics routines` : 'what-if weather, network and physics rerun');
  }
  const later = scheduler(() => { ui.setBusy(false); run(); }, 30);
  const go = (newStation = false) => {
    if (newStation) { ui.setBusy(true); setTimeout(() => { run(true); ui.setBusy(false); }, 0); return; }
    ui.setBusy(true); later();
  };
  run(true);
}
