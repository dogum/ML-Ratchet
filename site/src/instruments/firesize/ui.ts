// FireSize: pick a real exam ignition, see its 3.25 km fuel patch and the predicted size distribution, and change the
// INPUT: wind, afternoon humidity, temperature and fuel dryness on the discovery day and the day before, the discovery
// hour, the cause, the reporting agency or the region. Weather changes recompute the Rothermel spread-rate inputs from
// the fuel patch; every change re-runs the 3-member ensemble.
import { fetchPack, type Pack } from '../../lib/nn/pack.ts';
import type { InstrumentContext } from '../types.ts';
import { shell, picker, slider, button, probs, canvas, scheduler, h, label, SCOPE, type Canvas2D } from '../kit.ts';
import { codesOf, isRecorded, lutOf, read, RECORDED, simard, sizeBin, weatherTouched, type ExMeta, type FSMeta, type Reading, type WhatIf } from './net.ts';

// LANDFIRE FBFM40 fuel classes, as firesize.fuel_class, with map colours
const FUEL: { name: string; col: string }[] = [
  { name: 'grass', col: '#d9b54a' }, { name: 'grass-shrub', col: '#b7a64c' }, { name: 'shrub', col: '#9a7b45' },
  { name: 'timber understory', col: '#4f8a52' }, { name: 'timber litter', col: '#2f6b45' }, { name: 'slash', col: '#7d5a3c' },
  { name: 'urban', col: '#7b7f86' }, { name: 'agriculture', col: '#c9d18a' }, { name: 'other non-burnable', col: '#3b4f63' },
];
function fuelClass(code: number) {
  const r: [number, number][] = [[101, 109], [121, 124], [141, 149], [161, 165], [181, 189], [201, 204]];
  for (let k = 0; k < r.length; k++) if (code >= r[k][0] && code <= r[k][1]) return k;
  return code === 91 ? 6 : code === 93 ? 7 : 8;
}
const CAUSE = ['Human', 'Natural', 'Unknown'];
const SHORT = ['<.1', '.1', '.25', '1', '3', '10', '30', '100', '300', '1k', '5k', '20k'];   // lower class edges, acres
const pct = (p: number) => (p < 0.001 ? '<0.1%' : p < 0.1 ? `${(100 * p).toFixed(1)}%` : `${Math.round(100 * p)}%`);
const ac = (a: number) => (a < 1 ? a.toFixed(2) : a < 10 ? a.toFixed(1) : Math.round(a).toLocaleString('en-US'));
const sign = (v: number, d = 1) => (v > 0 ? '+' : v < 0 ? '−' : '±') + Math.abs(v).toFixed(d);

export async function mount(ctx: InstrumentContext) {
  const [weights, examples] = await Promise.all([fetchPack<FSMeta>(ctx.base + 'weights'), fetchPack<ExMeta>(ctx.base + 'examples')]);
  show(ctx, weights, examples);
}

export function show(ctx: InstrumentContext, weights: Pack<FSMeta>, examples: Pack<ExMeta>) {
  const meta = weights.meta, L = lutOf(weights), ign = examples.meta.ignitions;
  const ui = shell(ctx.root, { title: 'FireSize', subtitle: 'will a new wildfire escape initial attack, and how big will it get?',
    params: '3 × 27,788-weight MLPs on 75 inputs' });
  ctx.root.classList.add('fs-inst');
  ctx.root.style.setProperty('--c', ctx.accent);
  ctx.root.append(h('style', {}, `
    .fs-inst .fs-h { font: 10px/1.4 var(--mono, monospace); text-transform: uppercase; letter-spacing: .06em; color: var(--scr-2ink); margin: 14px 0 4px; }
    .fs-inst .fs-h:first-child { margin-top: 0; }
    .fs-inst .fs-who { margin: 0 0 6px; color: var(--scr-ink); font: 11px/1.45 var(--mono, monospace); }
    .fs-inst .fs-size { font: 11px/1.55 var(--mono, monospace); color: var(--scr-2ink); margin: 6px 0 0; }
    .fs-inst .fs-size b { color: var(--scr-ink); font-weight: 400; }
    .fs-inst .fs-size .in { color: var(--c); } .fs-inst .fs-size .out { color: var(--scr-ink); }
    .fs-inst .fs-tbl { width: 100%; border-collapse: collapse; font: 10px/1.5 var(--mono, monospace); color: var(--scr-2ink); }
    .fs-inst .fs-tbl td { padding: 1px 0 1px 6px; white-space: nowrap; } .fs-inst .fs-tbl td:first-child { padding-left: 0; white-space: normal; }
    .fs-inst .fs-tbl td + td { text-align: right; color: var(--scr-ink); } .fs-inst .fs-tbl td.moved { color: var(--c); }
    .fs-inst .fs-note { font-size: 12px; line-height: 1.5; color: var(--scr-2ink); margin: 10px 0 0; }
    .fs-inst .fs-group { font: 10px/1.4 var(--mono, monospace); text-transform: uppercase; letter-spacing: .06em; color: var(--scr-3ink); grid-column: 1 / -1; margin-top: 4px; }
    .fs-inst .fs-row { display: flex; gap: 8px; flex-wrap: wrap; grid-column: 1 / -1; align-items: end; }
    .fs-inst .fs-seg { display: flex; flex-direction: column; gap: 6px; }
    .fs-inst .fs-seg > span { font-size: 12.5px; color: var(--scr-2ink); }
    .fs-inst .fs-seg .inst-pick { padding: 6px 9px; }
    .fs-inst .fs-seg > div { display: flex; gap: 6px; flex-wrap: wrap; }
    .fs-inst .fs-sel select { appearance: none; -webkit-appearance: none; padding: 7px 28px 7px 9px; border-radius: 4px; border: 1px solid var(--scr-rule); background: transparent; color: var(--scr-ink); font-family: var(--mono); font-size: 11.5px; cursor: pointer;
      background-image: linear-gradient(45deg, transparent 50%, var(--scr-3ink) 50%), linear-gradient(135deg, var(--scr-3ink) 50%, transparent 50%); background-position: calc(100% - 15px) 50%, calc(100% - 10px) 50%; background-size: 5px 5px; background-repeat: no-repeat; }
    .fs-inst .fs-sel select:focus-visible { outline: 2px solid var(--c); outline-offset: 2px; }
    .fs-inst .fs-sel option { background: #0d0f0c; color: #ebe6d8; }
    .fs-inst .fs-key { display: flex; flex-wrap: wrap; gap: 4px 14px; padding: 6px 10px; font: 10.5px/1.5 var(--mono, monospace); color: var(--scr-2ink); }
    .fs-inst .fs-key span { display: inline-flex; align-items: center; gap: 5px; }
    .fs-inst .fs-key i { display: inline-block; width: 10px; height: 10px; background: var(--k); }
    .fs-inst .fs-key i.line { height: 0; width: 14px; border-top: 2px solid var(--k); background: none; }
    .fs-inst:not(.compact) .inst-body { grid-template-columns: minmax(0, 1fr); }
    .fs-inst:not(.compact) .inst-readout { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px 22px; align-items: start; }
    @media (max-width: 540px) { .fs-inst .inst-picker { display: grid; grid-template-columns: repeat(2, 1fr); } }
  `));

  let i = 0;
  let codes = codesOf(examples, 0);
  let w: WhatIf = { ...RECORDED };
  let recd: Reading = read(weights, L, ign[0], codes, w);
  let now: Reading = recd;
  const fi = (k: string) => meta.feats.indexOf(k);

  // ---- screen: the fuel patch and the size distribution
  const box = h('div', { role: 'img', 'aria-label': 'Fuel classes in the 3.25 km around the ignition, and the predicted final-size distribution over 12 FPA-FOD classes against the real size' });
  const key = h('div', { class: 'fs-key' });
  const used = () => new Set(Array.from(codes, (c) => fuelClass(c)));
  ui.screen.append(box, key);
  const narrow = ctx.root.getBoundingClientRect().width < 560;
  const plot = canvas(box, ctx.compact || narrow ? 1.25 : 2.5, (c) => draw(c));

  // ---- readout
  const who = h('p', { class: 'fs-who' });
  const escBox = h('div');
  const sizeTxt = h('p', { class: 'fs-size' });
  const feat = h('table', { class: 'fs-tbl' });
  const note = h('p', { class: 'fs-note' });
  ui.readout.append(h('div', {}, h('div', { class: 'fs-h' }, 'This ignition'), who, note),
    h('div', {}, h('div', { class: 'fs-h' }, 'Escape and final size'), escBox, sizeTxt),
    h('div', {}, h('div', { class: 'fs-h' }, 'What the network reads'), feat));

  // ---- controls
  const group = (t: string) => ui.controls.append(h('div', { class: 'fs-group' }, t));
  group('Weather, discovery day and the day before (gridMET)');
  const rec = (k: string) => ign[i].x[fi(k)];
  const sW = slider(ui.controls, {
    label: 'Wind (daily mean)', min: -5, max: 8, step: 0.1, value: 0,
    fmt: (v) => (v === 0 ? `${rec('vs').toFixed(1)} m/s, recorded` : `${Math.max(0, rec('vs') + v).toFixed(1)} m/s (${sign(v)})`),
    hint: 'The Rothermel inputs use 1.45 × the daily mean as the afternoon 10 m wind.',
    onInput: (v) => { w.dWind = v; go(); },
  });
  const sH = slider(ui.controls, {
    label: 'Minimum humidity', min: -30, max: 40, step: 1, value: 0,
    fmt: (v) => (v === 0 ? `${rec('rmin').toFixed(0)} %, recorded` : `${Math.min(100, Math.max(1, rec('rmin') + v)).toFixed(0)} % (${sign(v, 0)})`),
    hint: 'Sets the 1-h fuel moisture (Simard, at Tmax and RHmin) that drives the spread rate.',
    onInput: (v) => { w.dRH = v; go(); },
  });
  const sT = slider(ui.controls, {
    label: 'Maximum temperature', min: -10, max: 10, step: 0.5, value: 0,
    fmt: (v) => (v === 0 ? `${rec('tmmx').toFixed(1)} °C, recorded` : `${(rec('tmmx') + v).toFixed(1)} °C (${sign(v)})`),
    onInput: (v) => { w.dT = v; go(); },
  });
  const sE = slider(ui.controls, {
    label: 'Fuel dryness (ERC)', min: -40, max: 40, step: 1, value: 0,
    fmt: (v) => (v === 0 ? `${rec('erc').toFixed(0)}, recorded` : `${Math.max(0, rec('erc') + v).toFixed(0)} (${sign(v, 0)})`),
    hint: 'BI, 100-h and 1000-h fuel moisture move with it along the dev-fitted slopes.',
    onInput: (v) => { w.dERC = v; go(); },
  });
  const extra: { set: (v: number) => void }[] = [];
  let causePick: { set: (i: number) => void } | null = null, fedPick: { set: (i: number) => void } | null = null;
  let regSel: HTMLSelectElement | null = null;
  if (!ctx.compact) {
    group('The record');
    const hourOf = () => {
      if (rec('hr_miss') === 1) return 'not recorded';
      const hh = ((Math.atan2(rec('hr_sin'), rec('hr_cos')) / (2 * Math.PI)) * 24 + 24) % 24;
      return `${String(Math.floor(hh)).padStart(2, '0')}:${String(Math.round((hh % 1) * 60) % 60).padStart(2, '0')}`;
    };
    extra.push(slider(ui.controls, {
      label: 'Discovery hour (local)', min: -0.5, max: 23.5, step: 0.5, value: -0.5,
      fmt: (v) => (v < 0 ? `${hourOf()}, recorded` : `${String(Math.floor(v)).padStart(2, '0')}:${v % 1 ? '30' : '00'}`),
      onInput: (v) => { w.hour = v < 0 ? null : v; go(); },
    }));
    const row = h('div', { class: 'fs-row' });
    ui.controls.append(row);
    const seg = (title: string) => { const holder = h('div'); row.append(h('div', { class: 'fs-seg' }, h('span', {}, title), holder)); return holder; };
    causePick = picker(seg('Cause'), CAUSE.map((c, k) => ({ label: c, value: k })), (k) => { w.cause = k; go(); });
    fedPick = picker(seg('Reporting agency'), [{ label: 'State / local', value: 0 }, { label: 'Federal', value: 1 }], (k) => { w.fed = k; go(); });
    regSel = h('select', { 'aria-label': 'Region' });
    meta.regions.forEach((r, k) => regSel!.append(h('option', { value: String(k) }, meta.region_names[r])));
    regSel.addEventListener('change', () => { w.region = +regSel!.value; go(); });
    row.append(h('label', { class: 'fs-seg fs-sel' }, h('span', {}, 'Region'), regSel));
  }
  const brow = h('div', { class: 'fs-row' });
  ui.controls.append(brow);
  button(brow, 'Restore the real record', () => reset());

  picker(ui.picker, ign.map((g, k) => ({ label: `${g.name.length > 16 ? g.name.slice(0, 15) + '…' : g.name}`, sub: `${g.state} ${g.year} · ${ac(g.real_acres)} ac`, value: k })), (k) => {
    i = k; codes = codesOf(examples, k); reset();
  });

  function syncRecord() {
    const x = ign[i].x;
    causePick?.set(['c_hum', 'c_nat', 'c_miss'].findIndex((k) => x[fi(k)] === 1));
    fedPick?.set(x[fi('fed')] === 1 ? 1 : 0);
    if (regSel) regSel.value = String(meta.regions.findIndex((r) => x[fi(`r_${r}`)] === 1));
  }

  function reset() {
    w = { ...RECORDED };
    sW.set(0); sH.set(0); sT.set(0); sE.set(0);
    if (extra.length) extra[0].set(-0.5);
    syncRecord();
    recd = read(weights, L, ign[i], codes, w);
    now = recd;
    key.replaceChildren(...[...used()].sort((a, b) => a - b).map((k) => h('span', {}, h('i', { style: `--k:${FUEL[k].col}` }), FUEL[k].name)),
      h('span', {}, h('i', { class: 'line', style: `--k:${ctx.accent}` }), 'widened (released)'),
      h('span', {}, h('i', { class: 'line', style: '--k:rgba(235,230,216,.7)' }), 'raw, T = 1'));
    render(recd.ms);
  }

  const go = scheduler(() => {
    ui.setBusy(true);
    now = read(weights, L, ign[i], codes, w);
    render(now.ms);
    ui.setBusy(false);
  }, 30);

  function draw(c: Canvas2D) {
    const { ctx: g, w: W, h: H } = c;
    g.clearRect(0, 0, W, H);
    const stacked = W / H < 1.6;
    // fuel patch
    const side = stacked ? Math.min(W - 20, H * 0.42) : Math.min(H - 34, W * 0.34);
    const x0 = stacked ? (W - side) / 2 : 12, y0 = 22, cell = side / 13;
    label(c, side < 260 ? '3.25 km fuel patch' : '3.25 km fuel patch (LANDFIRE 2016, 250 m)', x0, 10, SCOPE.text, 'left', 9.5);
    for (let r = 0; r < 13; r++) for (let q = 0; q < 13; q++) {
      g.fillStyle = FUEL[fuelClass(codes[r * 13 + q])].col;
      g.fillRect(x0 + q * cell, y0 + r * cell, Math.ceil(cell), Math.ceil(cell));
    }
    g.strokeStyle = SCOPE.ink; g.lineWidth = 1;
    g.strokeRect(x0 + 4 * cell + 0.5, y0 + 4 * cell + 0.5, 5 * cell, 5 * cell);
    g.lineWidth = 2;
    const cx = x0 + 6.5 * cell, cy = y0 + 6.5 * cell;
    g.beginPath(); g.arc(cx, cy, cell * 0.38, 0, 2 * Math.PI); g.stroke();
    g.beginPath(); g.moveTo(cx - cell * 0.8, cy); g.lineTo(cx + cell * 0.8, cy); g.moveTo(cx, cy - cell * 0.8); g.lineTo(cx, cy + cell * 0.8); g.stroke();
    label(c, 'N ↑', x0 + side + 6, y0 + 6, SCOPE.faint, 'left', 9.5);
    // distribution
    const bx = stacked ? 34 : x0 + side + 46, by = stacked ? y0 + side + 30 : 22;
    const bw = W - bx - 10, bh = (stacked ? H - by : H - by) - 34;
    const nc = meta.class_labels.length, slot = bw / nc;
    const top = 1.08 * Math.max(0.3, ...Array.from(now.p), ...Array.from(now.p1));
    for (const t of [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6].filter((t) => t <= top)) {
      const y = by + bh - (t / top) * bh;
      g.strokeStyle = SCOPE.grid; g.lineWidth = 1; g.beginPath(); g.moveTo(bx, Math.round(y) + 0.5); g.lineTo(bx + bw, Math.round(y) + 0.5); g.stroke();
      label(c, `${Math.round(t * 100)}%`, bx - 4, y, SCOPE.faint, 'right', 9);
    }
    const rb = sizeBin(meta, ign[i].real_acres);
    for (let k = 0; k < nc; k++) {
      const xx = bx + k * slot + 2, ww = slot - 4;
      const hp = (now.p[k] / top) * bh, h1 = (now.p1[k] / top) * bh;
      g.fillStyle = ctx.accent; g.globalAlpha = k >= meta.esc_bin ? 1 : 0.55;
      g.fillRect(xx, by + bh - hp, ww, hp); g.globalAlpha = 1;
      g.strokeStyle = 'rgba(235,230,216,.75)'; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(xx, by + bh - h1); g.lineTo(xx + ww, by + bh - h1); g.stroke();
      label(c, slot < 40 ? SHORT[k] : meta.class_labels[k].replace('>=', '≥'), xx + ww / 2, by + bh + 10, SCOPE.text, 'center', 8.5);
      if (k === rb) {
        g.strokeStyle = SCOPE.ink; g.lineWidth = 3; g.beginPath(); g.moveTo(xx, by + bh + 19); g.lineTo(xx + ww, by + bh + 19); g.stroke();
        label(c, 'real', xx + ww / 2, by + bh + 28, SCOPE.ink, 'center', 9);
      }
    }
    const xe = bx + meta.esc_bin * slot;
    g.strokeStyle = ctx.accent; g.setLineDash([3, 3]); g.lineWidth = 1;
    g.beginPath(); g.moveTo(xe, by - 4); g.lineTo(xe, by + bh); g.stroke(); g.setLineDash([]);
    label(c, '100 acres: escape', xe + 4, by + 2, ctx.accent, 'left', 9);
    label(c, 'final size, acres (FPA-FOD class bins)', bx, by - 12, SCOPE.text, 'left', 9.5);
  }

  function render(ms: number) {
    const g = ign[i];
    const touched = !isRecorded(w);
    const escaped = g.real_acres >= 100;
    const cls = ['c_hum', 'c_nat', 'c_miss'].findIndex((k) => g.x[fi(k)] === 1);
    who.innerHTML = `${g.name} · ${g.date} · ${g.state} · ${['human', 'natural', 'unknown'][cls] ?? 'unknown'} cause (general cause: ${g.cause})`
      + `<br>Fuel at the point: ${g.fuel} · real size <b>${ac(g.real_acres)} acres</b> (class ${g.size_class})`;
    probs(escBox, 'Escapes initial attack (≥ 100 acres)', [
      { label: 'escapes', p: now.pEscape, truth: !touched && escaped },
      { label: 'held', p: 1 - now.pEscape, truth: !touched && !escaped }]);
    const inside = sizeBin(meta, g.real_acres) >= sizeBin(meta, now.lo) && sizeBin(meta, g.real_acres) <= sizeBin(meta, now.hi);
    sizeTxt.innerHTML = `P(≥ 1,000 acres) <b>${pct(now.pLarge)}</b><br>Median <b>${ac(now.median)} acres</b> (raw T = 1) · 80 % range <b>${ac(now.lo)}–${ac(now.hi)}</b>`
      + (touched ? '<br>Your input is not the recorded one, so there is no real outcome to compare.'
        : `<br>Real ${ac(g.real_acres)} acres: <span class="${inside ? 'in' : 'out'}">${inside ? 'inside' : 'outside'} the 80 % range</span>`)
      + `<br>P(escape) without the widening (T = 1): <b>${pct(now.pEscape1)}</b>`;

    const xv = (r: Reading, k: string) => r.x[fi(k)];
    const m1 = (r: Reading) => Math.min(35, Math.max(1, simard(xv(r, 'tmmx'), xv(r, 'rmin'))));
    // [name, the inputs it shows, how to print them]
    const rows: [string, string[], (r: Reading) => string][] = [
      ['wind, daily mean', ['vs'], (r) => `${xv(r, 'vs').toFixed(1)} m/s`],
      ['RHmin', ['rmin'], (r) => `${xv(r, 'rmin').toFixed(0)} %`],
      ['Tmax', ['tmmx'], (r) => `${xv(r, 'tmmx').toFixed(1)} °C`],
      ['1-h fuel moisture (Simard)', ['tmmx', 'rmin'], (r) => `${m1(r).toFixed(1)} %`],
      ['ERC / BI', ['erc', 'bi'], (r) => `${xv(r, 'erc').toFixed(0)} / ${xv(r, 'bi').toFixed(0)}`],
      ['100-h fuel moisture', ['fm100'], (r) => `${xv(r, 'fm100').toFixed(1)} %`],
      ['Rothermel ROS at the point', ['ros_pt'], (r) => `${Math.expm1(xv(r, 'ros_pt')).toFixed(2)} m/min`],
      ['ROS, 1.25 km box (log mean)', ['ros5'], (r) => `${Math.expm1(xv(r, 'ros5')).toFixed(2)} m/min`],
      ['discovery hour', ['hr_sin', 'hr_cos', 'hr_miss'], (r) => (xv(r, 'hr_miss') === 1 ? 'missing'
        : `${(((Math.atan2(xv(r, 'hr_sin'), xv(r, 'hr_cos')) / (2 * Math.PI)) * 24 + 24) % 24).toFixed(1)} h`)],
    ];
    feat.replaceChildren(h('tr', {}, h('td', {}, ''), h('td', {}, 'recorded'), h('td', {}, 'now')),
      ...rows.map(([name, keys, f]) => {
        const moved = keys.some((k) => Math.abs(xv(recd, k) - xv(now, k)) > 1e-9 * Math.max(1, Math.abs(xv(recd, k))));
        return h('tr', {}, h('td', {}, name), h('td', {}, f(recd)), h('td', moved ? { class: 'moved' } : {}, f(now)));
      }));

    const bits: string[] = [];
    if (!touched) {
      bits.push(`Recorded inputs: P(escape) ${pct(now.pEscape)}, as in the exam run (${pct(g.exam_p_escape)}).`);
      bits.push(escaped ? `This fire did escape (${ac(g.real_acres)} acres).` : `This fire was held at ${ac(g.real_acres)} acres.`);
      bits.push('Across all 655,869 exam ignitions FireSize ranked escapes at AUC 0.727 (a GBM fitted on real dev labels: 0.777), and its chances run about twice too high: mean 5.5 % against a real 2.7 %.');
    } else {
      bits.push(`P(escape) ${pct(recd.pEscape)} → ${pct(now.pEscape)}; median ${ac(recd.median)} → ${ac(now.median)} acres.`);
      if (weatherTouched(w)) bits.push(`The spread rate at the point went from ${Math.expm1(xv(recd, 'ros_pt')).toFixed(2)} to ${Math.expm1(xv(now, 'ros_pt')).toFixed(2)} m/min.`
        + (Math.expm1(xv(now, 'ros_pt')) === 0 ? ' The point itself is non-burnable in LANDFIRE (the simulator moved such ignitions up to 180 m), so only the surrounding fuels respond.' : ''));
      if (w.region !== null) bits.push('Region is a learned effect from the simulated ignitions\' borrowed records and the regional reporting floors, not from real outcomes.');
      if (w.fed !== null || w.cause !== null) bits.push('Cause and agency act through the fitted suppression model (response time, line rate, the confine-only share), several of whose parameters sit at their bounds.');
    }
    note.textContent = bits.join(' ');
    plot.redraw();
    ui.stamp(ms, touched ? (weatherTouched(w) ? 'what-if: Rothermel inputs recomputed, 3 networks rerun' : 'what-if: 3 networks rerun') : '12 size classes from 75 inputs');
  }

  reset();
}
