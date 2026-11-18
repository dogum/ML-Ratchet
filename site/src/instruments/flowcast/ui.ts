// FlowCast river replay: a real exam river over two water years (WY1995-96), the observed USGS hydrograph against
// FlowCast's daily mean and its 5-95 % band. Change the WEATHER or the BASIN the network reads and it reruns live:
// rain x0.5-1.5 (ERA5-Land under-catches), air temperature +-3 C (snow or rain, melt timing), PET, a storm added or
// removed, elevation, area, forest. The climate statics are recomputed from the changed forcing, as flowcast.py
// does (or frozen, to see what they carry). Nowcast mode swaps in the gauged weights: slide the issue day and the
// network reads the gauge up to that day, against persistence.
import { fetchPack, type Pack } from '../../lib/nn/pack.ts';
import type { InstrumentContext } from '../types.ts';
import { shell, picker, slider, button, canvas, h, scheduler, fmt } from '../kit.ts';
import {
  applyOps, basin, climate, loadNet, predict, predictDay, prepare, scores, waterYield, WARM,
  type Basin, type BasinsMeta, type Clim, type Forecast, type Ops, type WeightsMeta,
} from './net.ts';
import { dayAt, drawScope, type Fan, type ScopeState } from './scope.ts';

const DAY = 86400000;
const STRATUM: Record<string, string> = { snow: 'snowy', arid: 'arid', humid: 'humid' };

export async function mount(ctx: InstrumentContext) {
  const [w, n, b] = await Promise.all([fetchPack<WeightsMeta>(ctx.base + 'weights'),
    fetchPack<WeightsMeta>(ctx.base + 'nowcast'), fetchPack<BasinsMeta>(ctx.base + 'basins')]);
  show(ctx, w, n, b);
}

export function show(ctx: InstrumentContext, wp: Pack<WeightsMeta>, np: Pack<WeightsMeta>, bp: Pack<BasinsMeta>) {
  const net = loadNet(wp), now = loadNet(np);
  const list = bp.meta.basins;
  const ui = shell(ctx.root, { title: 'FlowCast', subtitle: 'a real exam river, two water years · ungauged daily flow, or the gauged nowcast',
    params: 'LSTM, 64 units, 25k weights' });
  ctx.root.classList.add('fc-inst');
  ctx.root.style.setProperty('--c', ctx.accent);
  ctx.root.append(h('style', {}, `
    .fc-inst .fc-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 8px; }
    .fc-inst .inst-btn[aria-pressed="true"] { border-color: var(--c); color: var(--scr-ink); background: rgb(235 230 216 / .06); }
    .fc-key { display: flex; flex-wrap: wrap; gap: 4px 14px; padding: 7px 10px; font: 10.5px/1.5 var(--mono, monospace); color: var(--scr-2ink); }
    .fc-key span { display: inline-flex; align-items: center; gap: 5px; }
    .fc-key i { display: inline-block; width: 14px; border-top: 2px solid var(--k); }
    .fc-key i.band { border-top-width: 7px; opacity: .45; } .fc-key i.dash { border-top-style: dashed; }
    .fc-h { font: 10px/1.4 var(--mono, monospace); text-transform: uppercase; letter-spacing: .06em; color: var(--scr-2ink); margin: 12px 0 3px; }
    .fc-h:first-child { margin-top: 0; }
    .fc-tbl { width: 100%; border-collapse: collapse; font: 10.5px/1.55 var(--mono, monospace); color: var(--scr-2ink); }
    .fc-tbl th { font-weight: 400; text-align: right; color: var(--scr-3ink); font-size: 9.5px; }
    .fc-tbl td { text-align: right; padding: 1px 0 1px 8px; } .fc-tbl td:first-child, .fc-tbl th:first-child { text-align: left; padding-left: 0; }
    .fc-tbl tr.me td { color: var(--c); } .fc-tbl tr.truth td { color: var(--scr-ink); }
    .fc-tbl td.up { color: #8fd18f; } .fc-tbl td.down { color: #e3736b; }
    .fc-day { font: 11px/1.55 var(--mono, monospace); color: var(--scr-2ink); margin: 0; }
    .fc-day b { color: var(--scr-ink); font-weight: 500; } .fc-day em { color: var(--c); font-style: normal; }
    .fc-note { font-size: 12px; line-height: 1.5; color: var(--scr-2ink); margin: 8px 0 0; }
    .fc-sub { font: 10px/1.4 var(--mono, monospace); color: var(--scr-3ink); margin: 2px 0 6px; }
  `));

  // ---- state
  let bi = 0, b: Basin = basin(bp, 0);
  let mode: 'ungauged' | 'nowcast' = 'ungauged';
  let win: 'both' | 'wy95' | 'wy96' = 'both', log = false;
  const what = { pmul: 1, tshift: 0, petmul: 1, storm: 0, freeze: false, elev: 0, area: 1, forest: NaN };
  let drop: { day: number; rows: [number, number] } | null = null;
  let lead = 3, outageDays = 0, timer = 0;
  let cursor = 0;
  let recorded: Forecast, cur: Forecast, ref: Forecast, series: Forecast | null = null, fan: Fan | null = null;
  let lastMs = 0;

  const n = () => b.info.n_target;
  const startMs = () => Date.parse(b.info.start + 'T00:00:00Z');
  const date = (i: number) => new Date(startMs() + i * DAY).toISOString().slice(0, 10);
  const range = (): [number, number] => (win === 'wy95' ? [0, 365] : win === 'wy96' ? [365, n()] : [0, n()]);
  const ops = (): Ops => {
    const o: Ops = {};
    if (what.pmul !== 1) o.pmul = what.pmul;
    if (what.tshift !== 0) o.tshift = what.tshift;
    if (what.petmul !== 1) o.petmul = what.petmul;
    if (what.storm > 0) o.storm = [WARM + cursor, what.storm];
    if (drop) o.drop = drop.rows;
    if (what.elev !== 0) o.elev = what.elev;
    if (what.area !== 1) o.area = what.area;
    if (Number.isFinite(what.forest) && Math.abs(what.forest - b.info.attrs.forest_frac) > 1e-9) o.forest = what.forest;
    if (what.freeze) o.freeze = true;
    return o;
  };
  const modified = () => Object.keys(ops()).some(k => k !== 'freeze');
  const gaugeObs = (outage: boolean) => {
    const o = new Float64Array(b.obs);
    if (outage && outageDays > 0) for (let t = Math.max(0, cursor - outageDays + 1); t <= cursor; t++) o[t] = NaN;
    return o;
  };

  // ---- the network runs
  function runAll() {
    const t0 = performance.now();
    const o = ops();
    cur = predict(net, prepare(net, b, o), n());
    ref = cur;
    series = null;
    if (mode === 'nowcast') series = predict(now, prepare(now, b, o, { obs: b.obs, lead }), n());
    runFan(o);
    lastMs = performance.now() - t0;
  }
  function runFan(o: Ops = ops()) {
    fan = null;
    if (mode !== 'nowcast' || cursor >= n() - 1) return;
    const obs = gaugeObs(true);
    const f: Fan = { day: cursor, mean: [], lo: [], hi: [], persist: NaN };
    for (let k = 1; k <= 7 && cursor + k < n(); k++) {
      const r = predictDay(now, prepare(now, b, o, { obs, lead: k }), cursor + k);
      f.mean.push(r.mean); f.lo.push(r.q[0]); f.hi.push(r.q[4]);
    }
    for (let t = cursor; t >= 0; t--) if (Number.isFinite(obs[t])) { f.persist = obs[t]; break; }
    fan = f;
  }

  // ---- screen
  const box = h('div', { role: 'img', 'aria-label': 'Observed daily streamflow against FlowCast with its 90 % band; precipitation and air temperature below' });
  const key = h('div', { class: 'fc-key' });
  ui.screen.append(box, key);
  const state = (): ScopeState => {
    const F = applyOps(b, ops()).F, N = n();
    const Pn = new Float64Array(N), P0 = new Float64Array(N), T = new Float64Array(N), T0 = new Float64Array(N);
    for (let i = 0; i < N; i++) { const r = (WARM + i) * 8; Pn[i] = F[r]; P0[i] = b.F[r]; T[i] = F[r + 2]; T0[i] = b.F[r + 2]; }
    const [i0, i1] = range();
    const main = mode === 'nowcast' && series ? series : cur;
    return { n: N, startMs: startMs(), i0, i1, obs: b.obs, mean: main.mean, q: main.q, ref: mode === 'nowcast' ? ref.mean : null,
      fan, P: Pn, P0, T, T0, cursor, log, accent: ctx.accent, cursorLabel: (mode === 'nowcast' ? 'issued ' : '') + date(cursor),
      outage: mode === 'nowcast' && outageDays > 0 ? [Math.max(0, cursor - outageDays + 1), cursor] : null };
  };
  const plot = canvas(box, ctx.compact ? 1.35 : 1.6, c => { if (cur) drawScope(c, state()); });
  const cv = box.querySelector('canvas')!;
  cv.style.cursor = 'ew-resize';
  const pick = (ev: PointerEvent) => {
    const r = cv.getBoundingClientRect();
    const [i0, i1] = range();
    const d = Math.max(i0, Math.min(i1 - 1, dayAt({ cv, ctx: cv.getContext('2d')!, w: r.width, h: r.height }, { i0, i1 }, ev.clientX - r.left)));
    setCursor(d);
  };
  cv.addEventListener('pointerdown', ev => { stop(); cv.setPointerCapture(ev.pointerId); pick(ev); });
  cv.addEventListener('pointermove', ev => { if (cv.hasPointerCapture(ev.pointerId)) pick(ev); });

  function drawKey() {
    const items: [string, string, string][] = [['Observed (USGS)', '#ebe6d8', ''],
      [mode === 'nowcast' ? `Nowcast, ${lead} d ahead` : 'FlowCast mean', ctx.accent, ''], ['25-75 % / 5-95 %', ctx.accent, 'band']];
    if (mode === 'nowcast') items.push(['Ungauged FlowCast', 'rgba(235,230,216,.5)', 'dash'], ['Forecast issued at the cursor', '#e0a84c', ''], ['Persistence', 'rgba(235,230,216,.6)', 'dash']);
    items.push(['Rain', '#6aa7e8', 'band'], ['Snow (T < 0 °C)', 'rgba(214,205,255,.9)', 'band']);
    key.replaceChildren(...items.map(([t, c, cls]) => h('span', {}, h('i', { class: cls, style: `--k:${c}` }), t)));
  }

  // ---- readout
  const head = h('p', { class: 'fc-sub' });
  const tScore = h('table', { class: 'fc-tbl' });
  const dayBox = h('div');
  const tClim = h('table', { class: 'fc-tbl' });
  const ctxLine = h('p', { class: 'fc-note' });
  const note = h('p', { class: 'fc-note' });
  ui.readout.append(h('div', { class: 'fc-h' }, 'Against the gauge, in view'), head, tScore,
    h('div', { class: 'fc-h' }, 'On the cursor day'), dayBox,
    h('div', { class: 'fc-h' }, 'Climate statics the network reads'), tClim, ctxLine, note);

  // ---- controls
  const modeRow = h('div', { class: 'fc-row', role: 'group', 'aria-label': 'Mode' });
  const winRow = h('div', { class: 'fc-row', role: 'group', 'aria-label': 'Window' });
  ui.controls.append(modeRow, winRow);
  const bMode = { ungauged: button(modeRow, 'Ungauged', () => setMode('ungauged')), nowcast: button(modeRow, 'Nowcast (gauged)', () => setMode('nowcast')) };
  const bWin = { both: button(winRow, 'Both years', () => setWin('both')), wy95: button(winRow, 'WY1995', () => setWin('wy95')), wy96: button(winRow, 'WY1996', () => setWin('wy96')) };
  const bLog = button(winRow, 'Log scale', () => { log = !log; bLog.setAttribute('aria-pressed', String(log)); plot.redraw(); });
  bLog.setAttribute('aria-pressed', 'false');

  const nowBox = h('div');
  ui.controls.append(nowBox);
  const leadS = slider(nowBox, { label: 'Nowcast lead', min: 1, max: 7, step: 1, value: lead, fmt: v => v + (v === 1 ? ' day' : ' days'),
    hint: 'The accent line is every day forecast this many days ahead, from the gauge up to its issue day.',
    onInput: v => { lead = v; schedule(); } });
  const outS = ctx.compact ? null : slider(nowBox, { label: 'Gauge outage before the issue day', min: 0, max: 60, step: 1, value: 0,
    fmt: v => (v === 0 ? 'none' : v + ' days'), hint: 'The availability flag goes to 0: the network falls back on the weather alone.',
    onInput: v => { outageDays = v; scheduleFan(); } });
  const playRow = h('div', { class: 'fc-row' });
  nowBox.append(playRow);
  const playBtn = button(playRow, '▶ Step the issue day', () => (timer ? stop() : play()));

  const pS = slider(ui.controls, { label: 'Precipitation ×', min: 0.5, max: 1.5, step: 0.05, value: 1,
    fmt: v => (v === 1 ? 'as recorded (ERA5-Land)' : '×' + v.toFixed(2)), hint: ' ',
    onInput: v => { what.pmul = v; schedule(); } });
  const pHint = pS.input.parentElement!.querySelector('small')!;
  const tS = slider(ui.controls, { label: 'Air temperature shift', min: -3, max: 3, step: 0.25, value: 0,
    fmt: v => (v === 0 ? 'as recorded' : (v > 0 ? '+' : '−') + Math.abs(v).toFixed(2) + ' °C'),
    hint: 'Moves T_mean, T_min and T_max (PET stays as recorded). Days below 0 °C count as snow in the statics.',
    onInput: v => { what.tshift = v; schedule(); } });
  const full = !ctx.compact;
  const extra = h('div');
  if (full) ui.controls.append(extra);
  const eS = slider(extra, { label: 'Evaporative demand (PET) ×', min: 0.7, max: 1.3, step: 0.05, value: 1,
    fmt: v => (v === 1 ? 'as recorded' : '×' + v.toFixed(2)), onInput: v => { what.petmul = v; schedule(); } });
  const sS = slider(extra, { label: 'Add a two-day storm on the cursor day', min: 0, max: 150, step: 5, value: 0,
    fmt: v => (v === 0 ? 'none' : v + ' mm'), hint: '60 % on the day, 40 % the next. Click the chart to move it.',
    onInput: v => { what.storm = v; schedule(); } });
  const stormRow = h('div', { class: 'fc-row' });
  extra.append(stormRow);
  const dropBtn = button(stormRow, 'Remove the biggest storm in view', () => toggleDrop());
  const freezeBtn = button(stormRow, 'Freeze the climate statics', () => {
    what.freeze = !what.freeze; freezeBtn.setAttribute('aria-pressed', String(what.freeze)); schedule(); });
  freezeBtn.setAttribute('aria-pressed', 'false');
  extra.append(h('div', { class: 'fc-h' }, 'The basin it is told about'));
  const elS = slider(extra, { label: 'Elevation shift', min: -1000, max: 1000, step: 50, value: 0,
    fmt: v => (v === 0 ? 'as recorded' : (v > 0 ? '+' : '−') + Math.abs(v) + ' m') + ' · mean ' + Math.round(b.info.attrs.elev_mean_m + v) + ' m',
    onInput: v => { what.elev = v; schedule(); } });
  const arS = slider(extra, { label: 'Area ×', min: -1, max: 1, step: 0.05, value: 0,
    fmt: v => (v === 0 ? 'as recorded' : '×' + (10 ** v).toPrecision(2)) + ' · ' + Math.round(b.info.attrs.area_km2 * 10 ** v).toLocaleString('en-US') + ' km²',
    onInput: v => { what.area = 10 ** v; schedule(); } });
  const foS = slider(extra, { label: 'Forest fraction', min: 0, max: 1, step: 0.05, value: b.info.attrs.forest_frac,
    fmt: v => v.toFixed(2) + (Math.abs(v - b.info.attrs.forest_frac) < 0.026 ? ' (as recorded)' : ''),
    onInput: v => { what.forest = v; schedule(); } });
  const resetRow = h('div', { class: 'fc-row' });
  ui.controls.append(resetRow);
  button(resetRow, 'Restore the recorded basin', () => { resetWhatIfs(); schedule(); });

  function resetWhatIfs() {
    Object.assign(what, { pmul: 1, tshift: 0, petmul: 1, storm: 0, freeze: false, elev: 0, area: 1, forest: b.info.attrs.forest_frac });
    drop = null; dropBtn.textContent = 'Remove the biggest storm in view';
    pS.set(1); tS.set(0); eS.set(1); sS.set(0); elS.set(0); arS.set(0); foS.set(b.info.attrs.forest_frac);
    freezeBtn.setAttribute('aria-pressed', 'false');
  }
  function toggleDrop() {
    if (drop) { drop = null; dropBtn.textContent = 'Remove the biggest storm in view'; schedule(); return; }
    const [i0, i1] = range();
    let t = i0;
    for (let i = i0; i < i1; i++) if (b.F[(WARM + i) * 8] > b.F[(WARM + t) * 8]) t = i;
    drop = { day: t, rows: [WARM + t - 1, WARM + t + 1] };
    dropBtn.textContent = 'Restore the storm of ' + date(t);
    schedule();
  }

  picker(ui.picker, list.map((s, i) => ({ label: s.name.replace(/\b([A-Z])([A-Z]+)\b/g, (_, a, r) => a + r.toLowerCase()).replace(/, ([a-z]{2})$/i, (_, s2) => ', ' + s2.toUpperCase()),
    sub: (STRATUM[s.stratum] ?? s.stratum) + ' · ' + Math.round(s.attrs.area_km2).toLocaleString('en-US') + ' km² · USGS ' + s.id.split('_')[1], value: i })), i => {
    stop(); bi = i; b = basin(bp, i);
    recorded = predict(net, prepare(net, b, {}), n());
    resetWhatIfs();
    cursor = startCursor();
    schedule();
  });

  function startCursor() {
    let t = 365;
    for (let i = 365; i < n(); i++) if (b.obs[i] > b.obs[t]) t = i;
    return Math.max(365, t - 3);
  }
  function setMode(m: typeof mode) {
    mode = m; stop();
    bMode.ungauged.setAttribute('aria-pressed', String(m === 'ungauged')); bMode.nowcast.setAttribute('aria-pressed', String(m === 'nowcast'));
    nowBox.style.display = m === 'nowcast' ? '' : 'none';
    drawKey(); schedule();
  }
  function setWin(w: typeof win) {
    win = w;
    for (const [k, el] of Object.entries(bWin)) el.setAttribute('aria-pressed', String(k === w));
    const [i0, i1] = range();
    if (cursor < i0 || cursor >= i1) cursor = Math.min(i1 - 2, Math.max(i0, cursor));
    if (what.storm > 0 || mode === 'nowcast') schedule(); else update();
  }
  function setCursor(d: number) {
    if (d === cursor) return;
    cursor = d;
    if (what.storm > 0) schedule();
    else if (mode === 'nowcast') scheduleFan();
    else update();
  }
  function play() {
    playBtn.textContent = '❚❚ Pause';
    const [i0, i1] = range();
    if (cursor >= i1 - 8) cursor = i0;
    timer = window.setInterval(() => {
      if (cursor >= range()[1] - 2) { stop(); return; }
      cursor++;
      const t0 = performance.now(); runFan(); lastMs = performance.now() - t0; update();
    }, 180);
  }
  function stop() { if (timer) { clearInterval(timer); timer = 0; playBtn.textContent = '▶ Step the issue day'; } }

  const schedule = scheduler(() => { ui.setBusy(true); runAll(); ui.setBusy(false); update(); }, 30);
  const scheduleFan = scheduler(() => { const t0 = performance.now(); runFan(); lastMs = performance.now() - t0; update(); }, 15);

  // ---- readout text
  const f2 = (v: number) => fmt(v, 2), pct = (v: number) => (Number.isFinite(v) ? Math.round(100 * v) + '%' : '—');
  function climRows(c0: Clim, c1: Clim, frozen: boolean) {
    const rows: [string, number, number, number][] = [
      ['Mean P, mm/d', c0.p_mean, c1.p_mean, 2], ['Mean PET, mm/d', c0.pet_mean, c1.pet_mean, 2], ['Aridity PET/P', c0.aridity, c1.aridity, 2],
      ['Snow share of P', c0.frac_snow, c1.frac_snow, 2], ['Days P > 5× mean', c0.hp_freq, c1.hp_freq, 3], ['Days P < 1 mm', c0.lp_freq, c1.lp_freq, 2],
      ['Mean T, °C', c0.t_mean, c1.t_mean, 1]];
    tClim.replaceChildren(h('tr', {}, h('th', {}, ''), h('th', {}, 'recorded'), h('th', {}, frozen ? 'what-if (frozen)' : 'what-if')),
      ...rows.map(([name, a, v, d]) => {
        const diff = Math.abs(v - a) > 0.5 * 10 ** -d;
        return h('tr', {}, h('td', {}, name), h('td', {}, a.toFixed(d)), h('td', diff && !frozen ? { class: v > a ? 'up' : 'down' } : {}, v.toFixed(d)));
      }));
  }

  function update() {
    if (!cur) return;
    const [i0, i1] = range();
    const N = n();
    const lo = (f: Forecast) => Float64Array.from({ length: N }, (_, i) => f.q[i * 5]);
    const hi = (f: Forecast) => Float64Array.from({ length: N }, (_, i) => f.q[i * 5 + 4]);
    const mod = modified();
    const sc = scores(b.obs, cur.mean, lo(cur), hi(cur), i0, i1);
    const rows: [string, ReturnType<typeof scores>, string][] = [];
    if (mode === 'nowcast' && series) {
      const pers = Float64Array.from({ length: N }, (_, i) => (i >= lead ? b.obs[i - lead] : NaN));
      rows.push([`Nowcast +${lead} d`, scores(b.obs, series.mean, lo(series), hi(series), i0, i1), 'me']);
      rows.push([`Persistence +${lead} d`, scores(b.obs, pers, undefined, undefined, i0, i1), '']);
      rows.push(['Ungauged', sc, '']);
    } else {
      rows.push([mod ? 'What-if input' : 'FlowCast', sc, 'me']);
      if (mod) rows.push(['Recorded input', scores(b.obs, recorded.mean, lo(recorded), hi(recorded), i0, i1), '']);
    }
    tScore.replaceChildren(h('tr', {}, h('th', {}, ''), h('th', {}, 'NSE'), h('th', {}, 'KGE'), h('th', {}, 'vol.'), h('th', {}, 'in 90%')),
      ...rows.map(([name, s, cls]) => h('tr', cls ? { class: cls } : {}, h('td', {}, name), h('td', {}, f2(s.nse)), h('td', {}, f2(s.kge)),
        h('td', {}, Number.isFinite(s.ratio) ? '×' + s.ratio.toFixed(2) : '—'), h('td', {}, pct(s.cover)))));
    head.textContent = `${date(i0)} to ${date(i1 - 1)} · ${i1 - i0} days · vol. = predicted / observed volume; in 90% = days inside the 5–95 % band`;

    // cursor day
    const area = b.info.attrs.area_km2 * (what.area || 1);
    const m3 = (v: number) => (v * area / 86.4);
    const obsv = b.obs[cursor];
    const main = mode === 'nowcast' && series ? series : cur;
    const dline = h('p', { class: 'fc-day' });
    dline.innerHTML = `<b>${date(cursor)}</b> · observed <b>${fmt(obsv, 2)}</b> mm/d<br>` +
      `${mode === 'nowcast' ? `nowcast ${lead} d ahead` : 'FlowCast'} <em>${fmt(main.mean[cursor], 2)}</em> mm/d ` +
      `(90 %: ${fmt(main.q[cursor * 5], 2)}–${fmt(main.q[cursor * 5 + 4], 2)}) · ${m3(main.mean[cursor]).toFixed(1)} m³/s`;
    const kids: Node[] = [dline];
    if (mode === 'nowcast' && fan) {
      const t = h('table', { class: 'fc-tbl' });
      t.append(h('tr', {}, h('th', {}, 'ahead'), h('th', {}, 'nowcast'), h('th', {}, '5–95 %'), h('th', {}, 'persist.'), h('th', {}, 'obs.')));
      fan.mean.forEach((v, k) => {
        const d = cursor + k + 1;
        t.append(h('tr', {}, h('td', { title: date(d) }, `+${k + 1} d`), h('td', {}, fmt(v, 2)), h('td', {}, fmt(fan!.lo[k], 1) + '–' + fmt(fan!.hi[k], 1)),
          h('td', {}, fmt(fan!.persist, 2)), h('td', { style: 'color:var(--scr-ink)' }, fmt(b.obs[d], 2))));
      });
      kids.push(t);
    }
    dayBox.replaceChildren(...kids);

    // statics
    const c0 = climate(b.F, b.rows), c1 = climate(applyOps(b, ops()).F, b.rows);
    climRows(c0, c1, what.freeze);
    const m = waterYield(b.info.attrs, c0);
    pHint.textContent = `ERA5-Land under-catches mountain and Pacific Northwest rain. FlowCast's simulator assumed the true rain here is ERA5 × ${m.toFixed(2)} (p1 prior, scatter ×/÷1.16); the network still reads ERA5.`;
    const d = b.info.decade_nse;
    ctxLine.textContent = d && d.nn_dev !== undefined
      ? `Exam decade WY1990–99 at this gauge (NSE): regionalisation U-dev ${f2(d.nn_dev)}, GR4J calibrated on the gauge ${f2(d.gr4j)}, NWM v3.0 ${f2(d.nwm)}, persistence ${f2(d.persist1)}. FlowCast's own per-basin decade score was not shipped; its median over 619 exam basins is 0.34 (U-dev 0.37).`
      : '';
    note.textContent = noteText(mod);
    leadS.set(lead);
    plot.redraw();
    ui.stamp(lastMs, mode === 'nowcast'
      ? `ungauged run, ${lead}-day nowcast over ${N} days and a 1–7 day fan, each block with a 365-day warm-up`
      : `${N} days in ${Math.ceil(N / 365)} blocks, each with a 365-day warm-up`);
  }

  function noteText(mod: boolean) {
    const s = b.info.stratum;
    if (mode === 'nowcast') {
      return 'Gauged protocol: each forecast also reads the observed flow up to its issue day (hindcast: the weather of the target days is the observed one). The gauge record in this example starts on 1 Oct 1994, so the first warm-up runs with the gauge flag off. Exam medians at 1/3/7 days: 0.57/0.47/0.43 against persistence 0.48/−0.19/−0.47 and NWM 0.55.';
    }
    if (what.freeze && mod) return 'Statics frozen: the network sees the changed daily weather but still believes the recorded climate (and scales its output by the recorded mean rain). Compare with the statics unfrozen to see how much of the response comes from the climate indices.';
    if (mod) return 'What-if: the network reran on the changed input; the observed flow belongs to the recorded weather, so read the scores as a sensitivity, not a skill.';
    if (b.info.id === 'camels_12095000') return 'A known miss: FlowCast predicts 1.76× the observed volume here. The build notes put it on the p1 water-yield prior, a regional fit that is too wet for this basin on the flank of Mount Rainier; regionalisation fails here too (decade NSE −0.44). Try precipitation ×0.6.';
    if (s === 'snow') return 'Snowy basins are FlowCast\'s weakest type on the exam (median 0.40 against 0.56 for regionalisation). Shift the temperature to move the rain/snow line and the spring melt.';
    if (s === 'arid') return 'Arid basins are where FlowCast ties regionalisation (median 0.21 against 0.20). Flow here is a small residual of rain minus evaporation; try the precipitation and PET sliders.';
    return 'Click or drag on the chart to move the cursor. Every slider changes the input; the network reruns on the whole two years.';
  }

  // ---- start
  recorded = predict(net, prepare(net, b, {}), n());
  what.forest = b.info.attrs.forest_frac;
  cursor = startCursor();
  setWin('both');
  setMode('ungauged');
}
