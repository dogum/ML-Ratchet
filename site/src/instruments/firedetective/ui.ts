// FireDetective: pick a real exam fire, see what the satellite recorded (burned area every half day, VIIRS pixels
// and their radiative power) and what the network infers (crown-fire severity, wind read blind, live fuel
// moisture), then change the INPUT: lose pixels to cloud, scale FRP, drier or wetter fuels, more wind, a lower
// canopy, a shorter record. Every change recomputes the features from the raw record and reruns the 3-member CNN.
import { fetchPack, type Pack } from '../../lib/nn/pack.ts';
import type { InstrumentContext } from '../types.ts';
import { shell, picker, slider, button, gauges, canvas, scheduler, h } from '../kit.ts';
import { read, type FDMeta, type Reading } from './net.ts';
import { apply, fireRecord, keepMask, isRecorded, RECORDED, type Examples, type FireRecord, type WhatIf } from './whatif.ts';
import { drawMap, drawSteps, burnRamp, rgb, DAY, NIGHT, FOREST, BURNABLE, NONBURN, LATER, PIXEL_M } from './scope.ts';

export async function mount(ctx: InstrumentContext) {
  const [weights, examples] = await Promise.all([
    fetchPack<FDMeta>(ctx.base + 'weights'), fetchPack<Examples>(ctx.base + 'examples')]);
  show(ctx, weights, examples);
}

const STRATUM: Record<string, string> = { forest_main: 'forest · 2017–21', nonforest_main: 'shrub/grass · 2017–21', recent: 'recent · 2022–23' };

export function show(ctx: InstrumentContext, weights: Pack<FDMeta>, examples: Pack<Examples>) {
  const fires = examples.meta.fires;
  const ui = shell(ctx.root, { title: 'FireDetective', subtitle: 'what drove a real wildfire, read from its satellite record',
    params: `3 × 1-D CNN, ${Math.round(weights.meta.params / 1000)}k weights` });
  ctx.root.classList.add('fd-inst');
  ctx.root.style.setProperty('--c', ctx.accent);
  ctx.root.append(h('style', {}, `
    .fd-inst .fd-h { font: 10px/1.4 var(--mono, monospace); text-transform: uppercase; letter-spacing: .06em; color: var(--scr-2ink); margin: 14px 0 4px; }
    .fd-inst .fd-h:first-child { margin-top: 0; }
    .fd-inst .fd-who { margin: 0 0 6px; color: var(--scr-ink); font: 11px/1.45 var(--mono, monospace); }
    .fd-inst .fd-tbl { width: 100%; border-collapse: collapse; font: 10px/1.5 var(--mono, monospace); color: var(--scr-2ink); }
    .fd-inst .fd-tbl td { padding: 1px 0 1px 6px; white-space: nowrap; } .fd-inst .fd-tbl td:first-child { padding-left: 0; white-space: normal; }
    .fd-inst .fd-tbl td + td { text-align: right; color: var(--scr-ink); } .fd-inst .fd-tbl td.moved { color: var(--c); }
    .fd-inst .fd-note { font-size: 12px; line-height: 1.5; color: var(--scr-2ink); margin: 10px 0 0; }
    .fd-inst .fd-group { font: 10px/1.4 var(--mono, monospace); text-transform: uppercase; letter-spacing: .06em; color: var(--scr-3ink); grid-column: 1 / -1; margin-top: 4px; }
    .fd-inst .fd-row { display: flex; gap: 8px; flex-wrap: wrap; grid-column: 1 / -1; }
    .fd-inst .fd-cv2 { border-top: 1px solid var(--scr-rule); }
    .fd-inst .fd-key { display: flex; gap: 6px 12px; align-items: flex-start; padding: 6px 10px; font: 10.5px/1.5 var(--mono, monospace); color: var(--scr-2ink); }
    .fd-inst .fd-keys { display: flex; flex-wrap: wrap; gap: 4px 14px; flex: 1; }
    .fd-inst .fd-keys > span { display: inline-flex; align-items: center; gap: 5px; }
    .fd-inst .fd-key i { display: inline-block; width: 9px; height: 9px; border-radius: 1px; background: var(--k); box-shadow: inset 0 0 0 1px rgb(235 230 216 / 0.12); }
    .fd-inst .fd-key i.fp { border-radius: 0; background: none; border: 1.5px solid var(--k); box-shadow: 0 0 0 1px rgb(8 9 7 / 0.9); }
    .fd-inst .fd-key em { font-style: normal; color: var(--scr-ink); }
    .fd-inst .fd-ramp { display: inline-flex; gap: 1px; }
    .fd-inst .fd-ramp button { display: block; width: 10px; height: 12px; padding: 0; border: 0; border-radius: 1px; cursor: pointer; }
    .fd-inst .fd-ramp button:disabled { cursor: default; }
    .fd-inst .fd-ramp button.on { outline: 1.5px solid var(--scr-ink); outline-offset: 1px; }
    .fd-inst .fd-ramp button:focus-visible { outline: 1.5px solid var(--c); outline-offset: 1px; }
    .fd-inst .fd-key small { color: var(--scr-3ink); font-size: 9.5px; }
    .fd-inst .fd-key .fd-time { flex-basis: 100%; flex-wrap: wrap; }
    .fd-inst .fd-key .fd-fine { flex-basis: 100%; font-size: 9.5px; color: var(--scr-3ink); }
    .fd-inst .fd-tools { display: flex; gap: 6px; flex-shrink: 0; }
    .fd-inst .fd-tog { padding: 3px 8px; font-size: 9.5px; }
    @media (max-width: 540px) { .fd-inst .fd-key { flex-direction: column; } }
    @media (max-width: 540px) { .fd-inst .inst-picker { display: grid; grid-template-columns: repeat(2, 1fr); } }
  `));

  let rec: FireRecord = fireRecord(examples, 0);
  let w: WhatIf = { ...RECORDED };
  let recd: Reading = read(weights, rec.input);
  let now: Reading = recd;
  let nowInput = rec.input;

  // ---- screen: the fire from above, then the record step by step
  const box1 = h('div', { role: 'img', 'aria-label': 'Map of the burned area coloured by the half-day pass that first mapped it, muted fuels, and VIIRS fire-pixel footprints' });
  const key1 = h('div', { class: 'fd-key' });
  const keyItems = h('div', { class: 'fd-keys' });
  const tools = h('div', { class: 'fd-tools' });
  let full = false, pixels = true, pass: number | null = null;
  const fitBtn = h('button', { type: 'button', class: 'inst-btn fd-tog', 'aria-pressed': 'false',
    title: 'The full frame is the grid the record comes on: the burned area plus about 2.5 km. The network reads fuels within 2 km of the final burned area.' }, 'Full frame');
  const pixBtn = h('button', { type: 'button', class: 'inst-btn fd-tog', 'aria-pressed': 'true' }, 'Hide pixels');
  fitBtn.addEventListener('click', () => { full = !full; fitBtn.textContent = full ? 'Fit fire' : 'Full frame'; fitBtn.setAttribute('aria-pressed', String(full)); plot1.redraw(); });
  pixBtn.addEventListener('click', () => { pixels = !pixels; pixBtn.textContent = pixels ? 'Hide pixels' : 'Show pixels'; pixBtn.setAttribute('aria-pressed', String(pixels)); render(lastMs); });
  tools.append(fitBtn, pixBtn);
  key1.append(keyItems, tools);
  const box2 = h('div', { class: 'fd-cv2', role: 'img', 'aria-label': 'New burned area per half-day pass and total fire radiative power' });
  ui.screen.append(box1, key1, box2);
  const narrow = ctx.root.getBoundingClientRect().width < 560;
  const mapState = () => ({ rec: rec.input, now: nowInput, keep: keepMask(rec, w), frp: w.frp, accent: ctx.accent, full, pixels, pass });
  let lastMs = 0;

  const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const passName = (k: number) => {
    const d = rec.info.steps[k] ?? '';
    return `${Number(d.slice(8, 10))} ${MON[Number(d.slice(5, 7)) - 1] ?? ''} ${rec.input.day[k] ? 'day' : 'night'}`;
  };
  /** The map key: the burn-time ramp with the first and last pass, then fuels and VIIRS footprints. */
  function drawKey() {
    const n = rec.input.masks.n, nNow = nowInput.masks.n, ramp = burnRamp(ctx.accent, n);
    const sw = h('span', { class: 'fd-ramp' });
    ramp.forEach((c, k) => {
      const b = h('button', { type: 'button', style: `background:${k < nNow ? rgb(c) : LATER}`, 'aria-pressed': String(pass === k),
        title: `pass ${k + 1}: ${passName(k)} · click to show only its pixels and the area burned by then`,
        'aria-label': `Pass ${k + 1}, ${passName(k)}` });
      if (pass === k) b.classList.add('on');
      if (k >= nNow) b.disabled = true;
      b.addEventListener('click', () => { pass = pass === k ? null : k; render(lastMs); });
      sw.append(b);
    });
    const item = (cls: string, col: string, txt: string) => h('span', {}, h('i', { class: cls, style: `--k:${col}` }), txt);
    const items: HTMLElement[] = [
      h('span', { class: 'fd-time' }, 'burned in pass', h('em', {}, passName(0)), sw, h('em', {}, passName(n - 1)),
        h('small', {}, pass != null && pass < nNow ? `showing pass ${pass + 1}, ${passName(pass)}: click it again for all` : 'click a pass to step through the growth'))];
    if (nNow < n) items.push(item('sq', LATER, 'burned after the passes read'));
    items.push(item('sq', FOREST, 'forest fuel'), item('sq', BURNABLE, 'grass, shrub'), item('sq', NONBURN, 'non-burnable'));
    if (pixels) {
      items.push(item('fp', DAY, 'VIIRS pixels, day pass'), item('fp', NIGHT, 'night pass'));
      if (w.drop > 0) items.push(item('fp', 'rgba(235,230,216,0.45)', 'lost to the what-if'));
      items.push(h('span', { class: 'fd-fine' }, `drawn at the nominal ${PIXEL_M} m footprint (not in the export; real pixels grow off-nadir), overlaps merged`));
    }
    keyItems.replaceChildren(...items);
  }
  const plot1 = canvas(box1, ctx.compact || narrow ? 1.15 : 1.6, (c) => drawMap(c, mapState()));
  const plot2 = canvas(box2, ctx.compact || narrow ? 2.2 : 3.4, (c) => drawSteps(c, { rec: rec.input, now: nowInput, steps: rec.info.steps, accent: ctx.accent, pass }));

  // ---- readout
  const who = h('p', { class: 'fd-who' });
  const sev = h('div'), wind = h('div'), lfmc = h('div'), lfmcNote = h('p', { class: 'fd-note' });
  const classical = h('table', { class: 'fd-tbl' });
  const simTbl = h('table', { class: 'fd-tbl' });
  const note = h('p', { class: 'fd-note' });
  ui.readout.append(h('div', {}, h('div', { class: 'fd-h' }, 'This fire'), who, note),
    h('div', {}, h('div', { class: 'fd-h' }, 'Burn severity · graded on MTBS'), sev),
    h('div', {}, h('div', { class: 'fd-h' }, 'Wind on the fire days · read with the wind inputs removed'), wind),
    h('div', {}, h('div', { class: 'fd-h' }, 'Live fuel moisture · fails on the exam'), lfmc, lfmcNote),
    h('div', {}, h('div', { class: 'fd-h' }, 'Classical diagnostics, same fire'), classical),
    ...(ctx.compact ? [] : [h('div', {}, h('div', { class: 'fd-h' }, 'Simulator-only readings · no real truth'), simTbl)]));

  // ---- controls
  const group = (t: string) => ui.controls.append(h('div', { class: 'fd-group' }, t));
  group('Satellite (VIIRS pixels; the FEDS perimeters stay as mapped)');
  const dropS = slider(ui.controls, {
    label: 'Pixels lost to cloud or omission', min: 0, max: 0.9, step: 0.05, value: 0,
    fmt: (v) => (v === 0 ? 'none' : `${Math.round(v * 100)} %`),
    hint: 'Each detection is dropped at random (fixed seed). Pixel counts, FRP and the share of pixels outside yesterday’s perimeter change.',
    onInput: (v) => { w.drop = v; go(); },
  });
  const frpS = slider(ui.controls, {
    label: 'Fire radiative power ×', min: -1, max: 1, step: 0.05, value: 0,
    fmt: (v) => (v === 0 ? 'as recorded' : '×' + (10 ** v).toFixed(2)),
    hint: 'Twin test: FireSim fires have too much night FRP (log10 0.86 vs 0.50 per pixel).',
    onInput: (v) => { w.frp = 10 ** v; go(); },
  });
  group('Weather and fuels (gridMET, LANDFIRE)');
  const dryS = slider(ui.controls, {
    label: 'Dead fuels drier', min: -6, max: 6, step: 0.5, value: 0,
    fmt: (v) => (v === 0 ? 'as recorded' : `${v > 0 ? '−' : '+'}${Math.abs(v).toFixed(1)} %-pts`),
    hint: '100-h and 1000-h fuel moisture, floor 1 %. ERC and BI are left as recorded.',
    onInput: (v) => { w.dry = v; go(); },
  });
  const extra: { set: (v: number) => void }[] = [];
  if (!ctx.compact) {
    extra.push(slider(ui.controls, {
      label: 'Wind speed ×', min: 0.25, max: 3, step: 0.05, value: 1,
      fmt: (v) => (v === 1 ? 'as recorded' : '×' + v.toFixed(2)),
      hint: 'The wind reading is made blind, so it cannot see this; severity and moisture can.',
      onInput: (v) => { w.wind = v; go(); },
    }));
    extra.push(slider(ui.controls, {
      label: 'Canopy base height ×', min: 0.3, max: 3, step: 0.05, value: 1,
      fmt: (v) => (v === 1 ? 'as recorded' : '×' + v.toFixed(2)),
      hint: 'Lower canopy bases let surface fire climb into the crowns (Van Wagner).',
      onInput: (v) => { w.cbh = v; go(); },
    }));
  }
  const stepS = slider(ui.controls, {
    label: 'Half-day passes read', min: 1, max: 12, step: 1, value: 12,
    fmt: (v) => (v >= rec.input.masks.n ? `all ${rec.input.masks.n}` : `first ${v}`), hint: 'Trained on up to 12 (six days).',
    onInput: (v) => { w.steps = v; go(); },
  });
  const row = h('div', { class: 'fd-row' });
  ui.controls.append(row);
  button(row, 'Restore the real record', () => reset());

  picker(ui.picker, fires.map((f, i) => ({ label: f.short, sub: `${STRATUM[f.stratum] ?? f.stratum} · ${f.year}`, value: i })), (i) => {
    rec = fireRecord(examples, i); pass = null; reset();
  });

  function reset() {
    w = { ...RECORDED };
    dropS.set(0); frpS.set(0); dryS.set(0); stepS.set(12);
    if (extra.length) { extra[0].set(1); extra[1].set(1); }
    nowInput = rec.input;
    recd = read(weights, rec.input);
    now = recd;
    render(recd.ms);
  }

  const go = scheduler(() => {
    ui.setBusy(true);
    nowInput = apply(rec, w);
    now = read(weights, nowInput);
    render(now.ms);
    ui.setBusy(false);
  }, 30);

  function render(ms: number) {
    const info = rec.info, o = now.out, n = rec.input.masks.n;
    const touched = !isRecorded(w, n);
    const sz = rec.input.masks.ny * rec.input.masks.nx;
    let a1 = 0;
    for (let i = 0; i < sz; i++) a1 += rec.input.masks.data[(n - 1) * sz + i] > 0 ? 1 : 0;
    who.textContent = `${info.name} ${info.year} · ${info.region} · ${STRATUM[info.stratum] ?? info.stratum} · ${n} passes, ${(a1 * 0.0324).toFixed(1)} km² burned`;

    gauges(sev, [
      { label: 'High-severity share of the burned area', value: o.hs, lo: Math.max(0, o.hs - o.hs_sd), hi: o.hs + o.hs_sd, digits: 2, range: [0, 0.5],
        truth: info.truth.hs, truthLabel: 'MTBS' },
      { label: 'Simulated crown-fire share (before the dev map)', value: o.crown, digits: 2, range: [0, 1] }]);
    gauges(wind, [{ label: 'Daily-mean 10 m wind', unit: 'm/s', value: o.wind, lo: o.wind_lo, hi: o.wind_hi, digits: 1, range: [0, 12],
      truth: info.truth.wind, truthLabel: 'gridMET' }]);
    gauges(lfmc, [{ label: 'Live fuel moisture', unit: '%', value: o.lfmc, lo: o.lfmc - o.lfmc_sd, hi: o.lfmc + o.lfmc_sd, digits: 0, range: [40, 250],
      truth: info.truth.lfmc, truthLabel: 'Globe-LFMC' }]);
    lfmcNote.textContent = info.truth.lfmc == null ? 'No Globe-LFMC sample within 25 km and ±10 days of this fire.' : '';

    const c = info.classical, fmt = (v: number | null, d: number) => (v == null ? 'n/a' : v.toFixed(d));
    classical.replaceChildren(
      h('tr', {}, h('td', {}, 'Van Wagner crown-fire potential'), h('td', {}, fmt(c.cfp, 2))),
      h('tr', {}, h('td', {}, 'ERC, maximum over the fire days'), h('td', {}, fmt(c.erc_max, 0))),
      h('tr', {}, h('td', {}, 'Rothermel back-calculated wind (raw, capped at 30)'), h('td', {}, c.wind_bc == null ? 'n/a' : `${c.wind_bc.toFixed(1)} m/s`)));

    if (!ctx.compact) {
      const rows: [string, (r: Reading) => number, (v: number) => string][] = [
        ['spread-rate factor vs Rothermel', (r) => r.out.ros_adj, (v) => '×' + v.toFixed(2)],
        ['afternoon 1-h dead fuel moisture', (r) => r.out.m1, (v) => v.toFixed(1) + ' %'],
        ['wind-driven share of the area', (r) => r.out.wind_regime, (v) => v.toFixed(2)],
        ['P(spotting)', (r) => r.out.p_spot, (v) => v.toFixed(2)],
        ['P(initial-attack line)', (r) => r.out.p_ia, (v) => v.toFixed(2)],
      ];
      simTbl.replaceChildren(h('tr', {}, h('td', {}, ''), h('td', {}, 'recorded'), h('td', {}, 'now')),
        ...rows.map(([name, fn, f]) => {
          const a = fn(recd), b = fn(now), moved = Math.abs(a - b) > 1e-9 * Math.max(1, Math.abs(a));
          return h('tr', {}, h('td', {}, name), h('td', {}, f(a)), h('td', moved ? { class: 'moved' } : {}, f(b)));
        }));
    }

    const bits: string[] = [];
    const r0 = recd.out;
    if (touched) {
      bits.push(`Severity ${r0.hs.toFixed(2)} → ${o.hs.toFixed(2)}, wind ${r0.wind.toFixed(1)} → ${o.wind.toFixed(1)} m/s, live moisture ${r0.lfmc.toFixed(0)} → ${o.lfmc.toFixed(0)} %.`);
      if (w.wind !== 1) bits.push('The wind reading barely moves: it is made with the wind columns blanked, from growth, shape and FRP alone.');
      if (w.drop > 0) bits.push('Fewer pixels look like a weaker fire: the perimeters did not change, only what VIIRS saw.');
      if (w.steps < n) bits.push(`Reading only the first ${Math.min(w.steps, n)} of ${n} passes.`);
      bits.push('The truths are for the recorded fire, not for your changed one.');
    } else {
      bits.push(`On all 944 exam fires: severity in forests ρ 0.58 (FWI 0.54, ERC 0.53: a tie), wind read blind ρ 0.30 (Rothermel back-calculation 0.13) but its level is worse than the dev median, and live moisture fails (ρ 0.17 vs climatology 0.34).`);
    }
    note.textContent = bits.join(' ');

    lastMs = ms;
    drawKey();
    plot1.redraw();
    plot2.redraw();
    ui.stamp(ms, touched ? 'what-if input: features recomputed from the raw record, 3 networks rerun twice' : 'features from the raw record, 3 networks run twice (with and without wind)');
  }

  render(recd.ms);
}
