// ReservoirPilot gauge board: pick a real exam dam, watch the learned policy, SOP, the sim-tuned rule and the
// perfect-foresight DP operate its scored decade, and change the INPUT: a drier or wetter decade, a flood pulse, a
// smaller or larger reservoir, a noisy storage gauge, more demand. Every policy re-runs live; DP follows (~0.5 s).
import { fetchPack, type Pack } from '../../lib/nn/pack.ts';
import type { InstrumentContext } from '../types.ts';
import { shell, picker, slider, button, canvas, h } from '../kit.ts';
import {
  components, damSeries, demandOf, isRecorded, makeTask, pilotNet, runDP, runPilot, runRule, runSOP, PULSE_LEN, RECORDED,
  type Components, type DamSeries, type DamsMeta, type PilotNet, type Run, type Task, type WhatIf,
} from './net.ts';
import { drawScope, dayAt, AMBER, SLATE, LILAC, type ScopeState } from './scope.ts';

export async function mount(ctx: InstrumentContext) {
  const [weights, dams] = await Promise.all([fetchPack(ctx.base + 'weights'), fetchPack<DamsMeta>(ctx.base + 'dams')]);
  show(ctx, weights, dams);
}

const f3 = (v: number) => (!Number.isFinite(v) ? '—' : v === 0 ? '0' : Math.abs(v) < 0.001 ? v.toExponential(1) : Math.abs(v) < 0.1 ? v.toFixed(4) : v.toFixed(3));
const pc = (v: number) => (Number.isFinite(v) ? `${Math.round(v * 100)}%` : '—');
const sg = (v: number) => (Number.isFinite(v) ? (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2) : '—');

export function show(ctx: InstrumentContext, weights: Pack<unknown>, dams: Pack<DamsMeta>) {
  const net: PilotNet = pilotNet(weights);
  const meta = dams.meta;
  const ui = shell(ctx.root, { title: 'ReservoirPilot', subtitle: 'a real exam dam’s scored decade, operated day by day · cost J and the storage it ignores', params: '5,314 weights' });
  ctx.root.classList.add('rp-board');
  ctx.root.style.setProperty('--c', ctx.accent);
  ctx.root.append(h('style', {}, `
    .rp-board .rp-key { display: flex; flex-wrap: wrap; gap: 6px 14px; padding: 7px 10px; font: 10.5px/1.5 var(--mono, monospace); color: var(--scr-2ink); }
    .rp-board .rp-key button { all: unset; cursor: pointer; display: inline-flex; align-items: center; gap: 5px; opacity: .45; }
    .rp-board .rp-key button[aria-pressed="true"], .rp-board .rp-key span { opacity: 1; }
    .rp-board .rp-key i { display: inline-block; width: 14px; border-top: 2px solid var(--k); }
    .rp-board .rp-key i.dash { border-top-style: dashed; }
    .rp-board .rp-when { margin: 0 0 6px; color: var(--scr-ink); font: 11px/1.5 var(--mono, monospace); }
    .rp-board .rp-tbl { width: 100%; border-collapse: collapse; font: 10.5px/1.45 var(--mono, monospace); color: var(--scr-2ink); margin: 4px 0; }
    .rp-board .rp-tbl th { font-weight: 400; text-align: right; padding: 2px 0 2px 6px; color: var(--scr-2ink); font-size: 9.5px; }
    .rp-board .rp-tbl td { padding: 1.5px 0 1.5px 6px; text-align: right; color: var(--scr-ink); white-space: nowrap; }
    .rp-board .rp-tbl td:first-child, .rp-board .rp-tbl th:first-child { text-align: left; padding-left: 0; color: var(--scr-2ink); white-space: normal; }
    .rp-board .rp-tbl td.me { color: var(--c); font-weight: 600; }
    .rp-board .rp-tbl tr.sec td { padding-top: 7px; text-transform: uppercase; letter-spacing: .05em; font-size: 9px; color: var(--scr-2ink); }
    .rp-board .rp-tbl tr.big td { font-size: 12px; }
    .rp-board .rp-h { font: 10px/1.4 var(--mono, monospace); text-transform: uppercase; letter-spacing: .06em; color: var(--scr-2ink); margin: 12px 0 3px; }
    .rp-board .rp-bars { display: grid; grid-template-columns: 5.4em 1fr 4.6em; gap: 3px 8px; align-items: center; font: 10.5px/1.4 var(--mono, monospace); color: var(--scr-2ink); }
    .rp-board .rp-bars b { display: block; height: 7px; border-radius: 2px; background: var(--k); opacity: .85; }
    .rp-board .rp-bars span:nth-child(3n) { text-align: right; color: var(--scr-ink); }
    .rp-board .rp-trade { margin: 10px 0 0; padding: 8px 10px; border-left: 2px solid var(--c); font-size: 12px; line-height: 1.5; color: var(--scr-ink); background: rgba(235,230,216,.04); }
    .rp-board .rp-note { font-size: 12px; line-height: 1.5; color: var(--scr-2ink); margin: 8px 0 0; }
    .rp-board .rp-row { display: flex; gap: 8px; flex-wrap: wrap; }
    @media (max-width: 540px) { .rp-board .inst-picker { display: grid; grid-template-columns: repeat(2, 1fr); } }
  `));

  // ---- state
  let di = 0, series: DamSeries = damSeries(dams, 0);
  let what: WhatIf = { ...RECORDED, pulseDay: wetDay(series) };
  let view: [number, number] = [0, series.info.n];
  let hover: number | null = null;
  const overlays = { sop: true, rule: true, dp: true, record: !ctx.compact };
  let task: Task, pilot: ReturnType<typeof runPilot>, rule: Run, sop: Run;
  let cP: Components, cR: Components, cS: Components, cH: Components | null = null;
  let dp: { run: Run; c: Components } | null = null, dpPending = false;
  const dpCache = new Map<string, { run: Run; c: Components }>();
  let lastMs = 0;

  // ---- screen
  const box = h('div', { role: 'img', 'aria-label': 'Storage, release and inflow of the selected dam over its scored decade, for the learned policy and its comparators' });
  const key = h('div', { class: 'rp-key' });
  ui.screen.append(box, key);
  const state = (): ScopeState => ({
    start: series.info.start, n: series.info.n, view, cap: task.cap, Qmax: task.Qmax, capRecord: series.info.cap,
    I: task.I, seen: task.seen === task.I ? null : task.seen, Iclim: clim(task), D: demandOf(task),
    pilot, rule, sop, dp: dp ? dp.run : null, record: { R: series.Robs, S: series.Sobs },
    show: overlays, means: means(), pulse: what.pulse > 0 ? [what.pulseDay - 3, what.pulseDay - 3 + PULSE_LEN] : null,
    hover, accent: ctx.accent, compact: ctx.compact,
  });
  const plot = canvas(box, ctx.compact ? 1.2 : 1.38, c => { if (task) drawScope(c, state()); });
  const cv = box.querySelector('canvas')!;
  const geom = () => { const r = cv.getBoundingClientRect(); return { cv, ctx: cv.getContext('2d')!, w: r.width, h: r.height }; };
  cv.style.cursor = 'zoom-in';
  cv.addEventListener('pointermove', ev => { hover = dayAt(geom(), state(), ev.clientX - cv.getBoundingClientRect().left); plot.redraw(); readHover(); });
  cv.addEventListener('pointerleave', () => { hover = null; plot.redraw(); readHover(); });
  cv.addEventListener('click', ev => {
    const d = dayAt(geom(), state(), ev.clientX - cv.getBoundingClientRect().left);
    setView(view[1] - view[0] < series.info.n ? 0 : Math.min(10, Math.floor(d / 365.25) + 1));
  });

  const keyItems: [keyof typeof overlays | null, string, string, string][] = [
    [null, 'ReservoirPilot', ctx.accent, ''], ['rule', 'Sim-tuned rule', AMBER, 'dash'], ['sop', 'SOP', SLATE, ''],
    ['dp', 'Perfect-foresight DP (bound)', LILAC, 'dash'], ['record', 'Record (real operators)', 'rgba(235,230,216,.85)', ''],
  ];
  for (const [flag, text, color, cls] of keyItems) {
    const sw = h('i', { class: cls, style: `--k:${color}` });
    if (!flag) { key.append(h('span', {}, sw, text)); continue; }
    const b = h('button', { type: 'button', 'aria-pressed': String(overlays[flag]) }, sw, text);
    b.addEventListener('click', () => { overlays[flag] = !overlays[flag]; b.setAttribute('aria-pressed', String(overlays[flag])); plot.redraw(); });
    key.append(b);
  }

  // ---- readout
  const when = h('p', { class: 'rp-when' });
  const hoverLine = h('p', { class: 'rp-when' });
  const tbl = h('table', { class: 'rp-tbl' });
  const jbars = h('div', { class: 'rp-bars' });
  const trade = h('p', { class: 'rp-trade' });
  const truth = h('p', { class: 'rp-note' });
  ui.readout.append(when, h('div', { class: 'rp-h' }, 'Cost J over the decade (lower is better, log scale)'), jbars, trade,
    h('div', { class: 'rp-h' }, 'Every component'), tbl, truth, hoverLine);

  // ---- controls
  const viewS = slider(ui.controls, {
    label: 'View', min: 0, max: 10, step: 1, value: 0, fmt: v => viewLabel(v),
    hint: ctx.compact ? undefined : 'Or click the chart to zoom into a year, and again to zoom out.', onInput: v => setView(v),
  });
  const inflowS = slider(ui.controls, {
    label: 'What if the decade’s inflow ×', min: 0.6, max: 1.4, step: 0.05, value: 1,
    fmt: v => (v === 1 ? 'as recorded' : '×' + v.toFixed(2)),
    hint: 'Demand and the inflow normal come from the calibration decade. In a drier scored decade the pilot keeps releasing for a normal that no longer comes: its known failure on large supply dams.',
    onInput: v => { what.inflow = v; update(); },
  });
  const pulseS = slider(ui.controls, {
    label: 'Add a flood pulse, peak', min: 0, max: 3, step: 0.25, value: 0,
    fmt: v => (v === 0 ? 'none' : v.toFixed(2) + ' × Qmax'),
    hint: ctx.compact ? undefined : 'A month-long flood hydrograph on top of the inflow, peaking 3 days in. The pilot sees it day by day; DP knew it was coming.',
    onInput: v => { what.pulse = v; if (v > 0 && view[1] - view[0] >= series.info.n) setView(yearOf(what.pulseDay), false); update(); },
  });
  const dayS = ctx.compact ? null : slider(ui.controls, {
    label: 'Flood pulse peak on', min: 3, max: series.info.n - 30, step: 1, value: what.pulseDay,
    fmt: v => dateOf(v), onInput: v => { what.pulseDay = v; if (what.pulse > 0) { if (view[1] - view[0] < series.info.n) setView(yearOf(v), false); update(); } },
  });
  const capS = ctx.compact ? null : slider(ui.controls, {
    label: 'What if the reservoir were ×', min: 0.5, max: 2, step: 0.05, value: 1,
    fmt: v => (v === 1 ? 'as built' : '×' + v.toFixed(2) + ' capacity'),
    hint: 'Same starting fill. A small reservoir cannot store a flood; a large one can carry water across dry years.',
    onInput: v => { what.capacity = v; update(); },
  });
  const gaugeS = slider(ui.controls, {
    label: 'Storage-gauge error the policies read', min: 0, max: 0.5, step: 0.05, value: 0,
    fmt: v => (v === 0 ? 'none' : 'σ ' + v.toFixed(2) + ' % of capacity'),
    hint: ctx.compact ? undefined : 'ResOpsUS inflow is back-calculated from storage changes, so a storage-gauge error e becomes (e_t − e_t−1)/K of inflow noise. The pilot and the rule read it; the water balance uses the true inflow.',
    onInput: v => { what.gauge = v; update(); },
  });
  const demS = ctx.compact ? null : slider(ui.controls, {
    label: 'What if demand were ×', min: 0.6, max: 1.6, step: 0.05, value: 1,
    fmt: v => (v === 1 ? 'as calibrated' : '×' + v.toFixed(2)),
    hint: 'Demand is the calibration half’s median release by day of year; J charges shortage relative to it.',
    onInput: v => { what.demand = v; update(); },
  });
  const row = h('div', { class: 'rp-row' });
  ui.controls.append(row);
  button(row, 'Restore the record', () => reset());
  if (!ctx.compact) button(row, 'New gauge-noise draw', () => { what.seed += 1; if (what.gauge > 0) update(); });

  picker(ui.picker, meta.dams.map((d, i) => ({ label: d.label, sub: `${d.use_name} · ${d.cap_years.toFixed(1)} yr of inflow`, value: i })), i => {
    di = i; series = damSeries(dams, i); reset();
  });

  // ---- helpers
  function wetDay(s: DamSeries) {
    let jm = 0; for (let j = 1; j < 366; j++) if (s.Icurve[j] > s.Icurve[jm]) jm = j;
    const mid = Math.floor(s.info.n / 2);
    for (let k = 0; k < 366; k++) { if (s.doy[mid + k] === jm + 1) return mid + k; }
    return mid;
  }
  function dateOf(d: number) { return new Date(Date.parse(series.info.start + 'T00:00:00Z') + d * 86400000).toISOString().slice(0, 10); }
  function yearOf(d: number) { return Math.min(10, Math.floor(d / 365.25) + 1); }
  function viewLabel(v: number) {
    if (v === 0) return `whole decade ${dateOf(0).slice(0, 4)}–${dateOf(series.info.n - 1).slice(0, 4)}`;
    const a = Math.round((v - 1) * 365.25);
    return `year ${v}: ${dateOf(a).slice(0, 7)} → ${dateOf(Math.min(series.info.n - 1, a + 365)).slice(0, 7)}`;
  }
  function setView(v: number, redraw = true) {
    viewS.set(v);
    if (v === 0) view = [0, series.info.n];
    else { const a = Math.min(series.info.n - 366, Math.round((v - 1) * 365.25)); view = [a, a + 366]; }
    cv.style.cursor = v === 0 ? 'zoom-in' : 'zoom-out';
    if (redraw) plot.redraw();
  }
  function clim(t: Task) { const o = new Float64Array(t.I.length); for (let i = 0; i < o.length; i++) o[i] = t.Icurve[t.doy[i] - 1]; return o; }
  function means() {
    const m = [{ name: 'pilot', v: cP.S_mean, color: ctx.accent }];
    if (overlays.rule) m.push({ name: 'rule', v: cR.S_mean, color: AMBER });
    if (overlays.sop) m.push({ name: 'SOP', v: cS.S_mean, color: SLATE });
    if (overlays.dp && dp) m.push({ name: 'DP', v: dp.c.S_mean, color: LILAC });
    return m;
  }
  function reset() {
    what = { ...RECORDED, pulseDay: wetDay(series) };
    inflowS.set(1); pulseS.set(0); gaugeS.set(0); capS?.set(1); demS?.set(1);
    if (dayS) { dayS.input.max = String(series.info.n - 30); dayS.set(what.pulseDay); }
    setView(0, false);
    update();
  }
  const dpKey = () => JSON.stringify([di, what.inflow, what.pulse, what.pulse > 0 ? what.pulseDay : 0, what.capacity, what.demand]);

  // ---- the run: every causal policy now, the DP bound right after
  function run() {
    const t0 = performance.now();
    task = makeTask(series, what);
    pilot = runPilot(net, task); rule = runRule(task); sop = runSOP(task);
    cP = components(pilot, task); cR = components(rule, task); cS = components(sop, task);
    cH = isRecorded(what) ? components({ R: series.Robs, S: series.Sobs }, task) : null;
    lastMs = performance.now() - t0;
    dp = dpCache.get(dpKey()) ?? null;
    dpPending = !dp;
  }
  function runDPNow() {
    if (!dpPending) return;
    const k = dpKey(), t0 = performance.now();
    const r = runDP(task); const c = components(r, task);
    dpCache.set(k, { run: r, c });
    if (dpCache.size > 24) dpCache.delete(dpCache.keys().next().value!);
    if (k === dpKey()) { dp = { run: r, c }; dpPending = false; paint(); ui.stamp(lastMs, `DP re-solved in ${Math.round(performance.now() - t0)} ms with the whole decade known in advance`); }
  }
  let dpTimer = 0;
  function update() {
    run(); paint();
    ui.stamp(lastMs, dpPending ? 'pilot, rule and SOP over 3,653 days · DP solving…' : 'pilot, rule and SOP over 3,653 days, DP from cache');
    clearTimeout(dpTimer);
    if (dpPending) dpTimer = window.setTimeout(runDPNow, ctx.compact ? 500 : 250);
  }

  function paint() {
    const i = series.info, rec = isRecorded(what);
    const ratio = mean(task.I) / mean(clim(task));
    when.innerHTML = `${i.label} · ${i.use_name} · ${Math.round(task.cap).toLocaleString('en-US')} hm³ (${(i.cap_years * what.capacity).toFixed(2)} yr of inflow) · Qmax ${Math.round(task.Qmax)} m³/s<br>`
      + `scored ${dateOf(0)} → ${dateOf(i.n - 1)} · inflow ${ratio.toFixed(2)}× the calibration-half normal`;
    // J bars on a log scale
    const items: [string, number | null, string, boolean][] = [
      ['Pilot', cP.J, ctx.accent, true], ['Tuned rule', cR.J, AMBER, false], ['SOP', cS.J, SLATE, false],
      ['Record', cH ? cH.J : null, 'rgba(235,230,216,.8)', false], ['DP bound', dp ? dp.c.J : null, LILAC, false]];
    const lo = 1e-4, hi = 2, pos = (v: number) => Math.max(1.5, Math.min(100, (100 * (Math.log10(Math.max(v, lo)) - Math.log10(lo))) / (Math.log10(hi) - Math.log10(lo))));
    jbars.replaceChildren(...items.flatMap(([n, v, col, me]) => [
      h('span', me ? { style: `color:${ctx.accent}` } : {}, n),
      h('span', {}, v == null ? '' : h('b', { style: `--k:${col};width:${pos(v)}%` })),
      h('span', {}, v == null ? (n === 'DP bound' && dpPending ? 'solving…' : '—') : f3(v))]));
    // the trade J does not see
    const lower = cP.S_mean < cS.S_mean;
    trade.innerHTML = `The pilot keeps <b>${pc(cP.S_mean)}</b> of capacity on average; SOP keeps ${pc(cS.S_mean)}, the tuned rule ${pc(cR.S_mean)}`
      + (dp ? `, DP ${pc(dp.c.S_mean)}` : '') + `. It ends the decade at ${sg(cP.dS_end)} of capacity from the start (SOP ${sg(cS.dS_end)}). `
      + (lower ? 'J charges nothing for stored water, so this lower reservoir is part of how the pilot wins.' : 'Here it holds as much as SOP.');
    // every component
    const cols: [string, Components | null][] = [['Pilot', cP], ['Rule', cR], ['SOP', cS], ['DP', dp ? dp.c : null], ['Record', cH]];
    const rows: [string, (c: Components) => string, boolean?][] = [
      ['J', c => f3(c.J), true], ['shortage part', c => f3(c.J_short)], ['flood part', c => f3(c.J_flood)],
      ['shortage ratio', c => pc(c.shortage_ratio)], ['days short', c => pc(c.shortage_days)],
      ['days/yr > Qmax', c => c.flood_days.toFixed(1)], ['worst excess ÷ Qmax', c => c.max_excess.toFixed(2)],
    ];
    const rows2: [string, (c: Components) => string][] = [
      ['mean storage', c => pc(c.S_mean)], ['end − start', c => sg(c.dS_end)], ['days near empty', c => pc(c.empty)], ['days full', c => pc(c.full)],
    ];
    const tr = ([name, fn, big]: [string, (c: Components) => string, boolean?]) => h('tr', big ? { class: 'big' } : {}, h('td', {}, name),
      ...cols.map(([n, c], k) => h('td', k === 0 ? { class: 'me' } : {}, c ? fn(c) : (n === 'DP' && dpPending ? '…' : '—'))));
    tbl.replaceChildren(h('tr', {}, h('th', {}, ''), ...cols.map(([n]) => h('th', {}, n))), ...rows.map(tr),
      h('tr', { class: 'sec' }, h('td', { colspan: '6' }, 'Storage: what J does not charge')), ...rows2.map(r => tr(r)));
    // the truth: this dam's exam run
    const e = i.exam_J;
    truth.innerHTML = rec
      ? `<b>Exam run, this dam (truth):</b> pilot ${f3(e.pilot)} ${Math.abs(cP.J - e.pilot) <= 1e-4 * Math.max(e.pilot, 1e-3) ? '(matches the live run)' : ''} · SOP ${f3(e.sop)} · WaterData rule curve ${f3(e.rule)} · historical ${f3(e.historical)} · DP ${f3(e.dp)}. `
        + `Exam medians over 244 dams: pilot 0.0154, tuned rule 0.0285, SOP 0.108, historical 0.183, DP 0.0063; mean storage 38 %, 40 %, 59 %, 44 %, 68 %.`
      : `<b>What-if input.</b> The exam numbers belong to the recorded decade (pilot ${f3(e.pilot)}, SOP ${f3(e.sop)}); the record cannot be re-run, so its column is blank. `
        + (what.gauge > 0 ? 'The pilot and the rule read the noisy inflow (faint bars); SOP ignores inflow; DP knows the true inflow. ' : '')
        + 'Every policy above re-ran on the changed input.';
    plot.redraw();
    readHover();
  }
  function readHover() {
    if (hover == null) { hoverLine.textContent = ctx.compact ? '' : 'Hover the chart for any day’s numbers.'; return; }
    const t = hover, q = task.Qmax, cap = task.cap;
    hoverLine.innerHTML = `${dateOf(t)} · inflow ${Math.round(task.I[t])} m³/s${task.seen !== task.I ? ` (read ${Math.round(task.seen[t])})` : ''} · demand ${Math.round(demandOf(task)[t])}<br>`
      + `storage pilot ${pc(pilot.S[t] / cap)}, rule ${pc(rule.S[t] / cap)}, SOP ${pc(sop.S[t] / cap)}${dp ? `, DP ${pc(dp.run.S[t] / cap)}` : ''}${isRecorded(what) ? `, record ${pc(series.Sobs[t] / series.info.cap)}` : ''}<br>`
      + `release pilot ${(pilot.R[t] / q).toFixed(2)} Qmax (h ${pilot.H[t].toFixed(2)}, e ${pilot.E[t].toFixed(2)}), SOP ${(sop.R[t] / q).toFixed(2)}`;
  }
  const mean = (v: Float64Array) => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i]; return s / v.length; };

  // first paint: the recorded decade of the first dam; DP re-solves right after
  update();
}
