// BasinDetective: pick a real exam basin, see its gauged decade and the decade the network has to reason about,
// and change the INPUT: a wetter, drier or warmer target decade, a reservoir on the gauged river, a noisier or
// coarser gauge, fewer gauged years. Every change re-computes the 150 features and re-runs the 5-member ensemble.
import { fetchPack, type Pack } from '../../lib/nn/pack.ts';
import type { InstrumentContext } from '../types.ts';
import { shell, picker, slider, button, gauges, probs, canvas, scheduler, h } from '../kit.ts';
import { read, CHANGES, type BDMeta, type Reading } from './net.ts';
import { apply, basinRecord, RECORDED, isRecorded, gaugeTouched, climateTouched, type Examples, type WhatIf, type BasinRecord } from './whatif.ts';
import { drawRecord, drawChanges, fmtChange, NICE } from './scope.ts';

const SHORT: Record<string, string> = {
  camels_04185000: 'Tiffin River, OH', camels_08165300: 'N Fork Guadalupe, TX',
  camels_12388400: 'Revais Creek, MT', camels_14020000: 'Umatilla River, OR',
};

export async function mount(ctx: InstrumentContext) {
  const [weights, examples] = await Promise.all([
    fetchPack<BDMeta>(ctx.base + 'weights'), fetchPack<Examples>(ctx.base + 'examples')]);
  show(ctx, weights, examples);
}

export function show(ctx: InstrumentContext, weights: Pack<BDMeta>, examples: Pack<Examples>) {
  const basins = examples.meta.basins;
  const ui = shell(ctx.root, { title: 'BasinDetective', subtitle: 'what a river’s own gauge record says about a decade it has not seen',
    params: '5 × 110k-weight MLPs on 150 features' });
  ctx.root.classList.add('bd-inst');
  ctx.root.style.setProperty('--c', ctx.accent);
  ctx.root.append(h('style', {}, `
    .bd-inst .bd-h { font: 10px/1.4 var(--mono, monospace); text-transform: uppercase; letter-spacing: .06em; color: var(--scr-2ink); margin: 14px 0 4px; }
    .bd-inst .bd-h:first-child { margin-top: 0; }
    .bd-inst .bd-who { margin: 0 0 6px; color: var(--scr-ink); font: 11px/1.45 var(--mono, monospace); }
    .bd-inst .bd-score { font: 11px/1.5 var(--mono, monospace); color: var(--scr-2ink); margin: 0 0 4px; }
    .bd-inst .bd-score b { color: var(--scr-ink); font-weight: 400; }
    .bd-inst .bd-tbl { width: 100%; border-collapse: collapse; font: 10px/1.5 var(--mono, monospace); color: var(--scr-2ink); }
    .bd-inst .bd-tbl td { padding: 1px 0 1px 6px; white-space: nowrap; } .bd-inst .bd-tbl td:first-child { padding-left: 0; white-space: normal; } .bd-inst .bd-tbl td + td { text-align: right; color: var(--scr-ink); }
    .bd-inst .bd-tbl td.moved { color: var(--c); }
    .bd-inst .bd-note { font-size: 12px; line-height: 1.5; color: var(--scr-2ink); margin: 10px 0 0; }
    .bd-inst .bd-group { font: 10px/1.4 var(--mono, monospace); text-transform: uppercase; letter-spacing: .06em; color: var(--scr-3ink); grid-column: 1 / -1; margin-top: 4px; }
    .bd-inst .bd-row { display: flex; gap: 8px; flex-wrap: wrap; grid-column: 1 / -1; }
    .bd-inst .bd-cv2 { border-top: 1px solid var(--scr-rule); }
    .bd-inst:not(.compact) .inst-body { grid-template-columns: minmax(0, 1fr); }
    .bd-inst:not(.compact) .inst-readout { display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 14px 22px; align-items: start; }
    .bd-inst .bd-key { display: flex; flex-wrap: wrap; gap: 4px 14px; padding: 6px 10px; font: 10.5px/1.5 var(--mono, monospace); color: var(--scr-2ink); }
    .bd-inst .bd-key span { display: inline-flex; align-items: center; gap: 5px; }
    .bd-inst .bd-key i { display: inline-block; width: 14px; border-top: 2px solid var(--k); }
    .bd-inst .bd-key i.dash { border-top-style: dashed; } .bd-inst .bd-key i.dot { border-top-style: dotted; } .bd-inst .bd-key i.band { border-top-width: 7px; opacity: .5; }
    .bd-inst .bd-key i.dia { width: 8px; height: 8px; border: 0; background: var(--k); transform: rotate(45deg); }
    @media (max-width: 540px) { .bd-inst .inst-picker { display: grid; grid-template-columns: repeat(2, 1fr); } }
  `));

  let rec: BasinRecord = basinRecord(examples, 0);
  let w: WhatIf = { ...RECORDED };
  let recd: Reading = read(weights, rec, w);
  let now: Reading = recd;

  // ---- screen: the record, then the six changes
  const box1 = h('div', { role: 'img', 'aria-label': 'Monthly precipitation and temperature for 1989-2008, daily gauged flow 1999-2008, and the predicted, observed and GR4J flow levels for 1989-99' });
  const box2 = h('div', { class: 'bd-cv2', role: 'img', 'aria-label': 'Predicted change of six flow signatures with one- and two-sigma bars, the observed change and GR4J' });
  const key = h('div', { class: 'bd-key' });
  for (const [txt, col, cls] of [['network ±1σ', ctx.accent, 'band'], ['observed 1989–99 (truth)', '#ebe6d8', ''], ['GR4J calibrated on 1999–2008', '#e0a84c', 'dash'],
    ['no change (gauged value)', 'rgba(235,230,216,.6)', 'dot'], ['precipitation', '#6aa7e8', 'band'], ['temperature', '#e0a84c', '']]) {
    key.append(h('span', {}, h('i', { class: cls, style: `--k:${col}` }), txt));
  }
  const key2 = h('div', { class: 'bd-key' }, h('span', {}, h('i', { class: 'band', style: `--k:${ctx.accent}` }), 'network ±1σ, ±2σ'),
    h('span', {}, h('i', { style: '--k:#ebe6d8' }), 'observed change'), h('span', {}, h('i', { class: 'dia', style: '--k:#e0a84c' }), 'GR4J'),
    h('span', {}, '○ network on the recorded input'));
  ui.screen.append(box1, key, box2, key2);
  const recordState = () => {
    const a = apply(rec, w);
    return { info: rec.info, tgtRec: rec.tgt, tgtNow: a.tgt, calRec: rec.cal, qRec: rec.q, qNow: a.q, i0: rec.q.length - a.q.length,
      now, recd, climateTouched: climateTouched(w), gaugeTouched: gaugeTouched(w), accent: ctx.accent };
  };
  const narrow = ctx.root.getBoundingClientRect().width < 560;
  const plot1 = canvas(box1, ctx.compact || narrow ? 1.3 : 2.25, (c) => drawRecord(c, recordState()));
  const plot2 = canvas(box2, ctx.compact || narrow ? 1.25 : 2.6, (c) => drawChanges(c, { info: rec.info, now, recd, touched: !isRecorded(w), climateTouched: climateTouched(w), accent: ctx.accent }));

  // ---- readout
  const who = h('p', { class: 'bd-who' });
  const score = h('p', { class: 'bd-score' });
  const regBox = h('div');
  const hidBox = h('div');
  const feat = h('table', { class: 'bd-tbl' });
  const note = h('p', { class: 'bd-note' });
  ui.readout.append(h('div', {}, h('div', { class: 'bd-h' }, 'This basin'), who, score, note),
    h('div', {}, h('div', { class: 'bd-h' }, 'Reservoir upstream?'), regBox),
    h('div', {}, h('div', { class: 'bd-h' }, 'What the network reads'), feat),
    ...(ctx.compact ? [] : [h('div', {}, h('div', { class: 'bd-h' }, 'Hidden properties · no real truth'), hidBox)]));

  // ---- controls
  const group = (t: string) => ui.controls.append(h('div', { class: 'bd-group' }, t));
  group('Target decade, 1989–99 (what if its weather had been different?)');
  const pS = slider(ui.controls, {
    label: 'Precipitation ×', min: 0.8, max: 1.2, step: 0.01, value: 1,
    fmt: (v) => (v === 1 ? 'as recorded' : '×' + v.toFixed(2)),
    hint: 'Training perturbed the target decade by ×0.88–1.12 in half the simulated basins.',
    onInput: (v) => { w.pmul = v; go(); },
  });
  const tS = slider(ui.controls, {
    label: 'Warming', min: -1, max: 3, step: 0.1, value: 0,
    fmt: (v) => (v === 0 ? 'as recorded' : (v > 0 ? '+' : '−') + Math.abs(v).toFixed(1) + ' °C'),
    hint: 'PET rises 2.5 % per °C, as in the training bank, which only went to ±1 °C.',
    onInput: (v) => { w.dT = v; go(); },
  });
  group('Gauged decade, 1999–2008 (what the network reads)');
  const rS = slider(ui.controls, {
    label: 'Reservoir upstream (linear store)', min: 0, max: 180, step: 5, value: 0,
    fmt: (v) => (v === 0 ? 'none' : `k ${v} d · ${(v / 365).toFixed(2)}× annual flow`),
    hint: 'Storage = k × mean flow, so storage / annual flow = k / 365. Above 36 days the exam calls the basin regulated.',
    onInput: (v) => { w.res_k = v; go(); },
  });
  const extra: { set: (v: number) => void }[] = [];
  if (!ctx.compact) {
    extra.push(slider(ui.controls, {
      label: 'Gauge jitter (AR(1) ρ 0.9)', min: 0, max: 0.3, step: 0.01, value: 0,
      fmt: (v) => (v === 0 ? 'none' : `cv ${v.toFixed(2)}`),
      hint: 'WaterSim’s default gauge model jitters every day (cv 0.08); real records are smoother, which the dev twins found.',
      onInput: (v) => { w.cv = v; go(); },
    }));
    extra.push(slider(ui.controls, {
      label: 'Round to significant figures (cfs)', min: 0, max: 3, step: 1, value: 0,
      fmt: (v) => (v === 0 ? 'as recorded' : `${4 - v} figure${4 - v > 1 ? 's' : ''}`),
      hint: 'USGS publishes 3 figures. Coarser rounding makes flat days and kills recessions.',
      onInput: (v) => { w.sig = v === 0 ? 0 : 4 - v; go(); },
    }));
    extra.push(slider(ui.controls, {
      label: 'Gauged water years read', min: 3, max: 9, step: 1, value: 9,
      fmt: (v) => (v === 9 ? 'all 9' : `last ${v}`), hint: 'Trained on 9.',
      onInput: (v) => { w.years = v; go(); },
    }));
  }
  const row = h('div', { class: 'bd-row' });
  ui.controls.append(row);
  button(row, 'Restore the real record', () => reset());

  picker(ui.picker, basins.map((b, i) => ({ label: SHORT[b.id] ?? b.name, sub: b.stratum, value: i })), (i) => {
    rec = basinRecord(examples, i); reset();
  });

  function reset() {
    w = { ...RECORDED, seed: 1 };
    pS.set(1); tS.set(0); rS.set(0);
    if (extra.length) { extra[0].set(0); extra[1].set(0); extra[2].set(9); }
    recd = read(weights, rec, w);
    now = recd;
    render(recd.ms);
  }

  const go = scheduler(() => {
    ui.setBusy(true);
    now = read(weights, rec, w);
    render(now.ms);
    ui.setBusy(false);
  }, 30);

  function render(ms: number) {
    const info = rec.info;
    const touched = !isRecorded(w);
    who.innerHTML = `${info.name} · ${info.stratum} · ${Math.round(info.area_km2).toLocaleString('en-US')} km²`;
    // basin scorecard on the recorded input
    let closer = 0, inside = 0;
    for (const t of CHANGES) {
      const [mu, sd] = now.get(t), obs = info.observed_change[t];
      if (Math.abs(mu - obs) < Math.abs(info.gr4j_change[t] - obs)) closer++;
      if (Math.abs(mu - obs) <= sd) inside++;
    }
    score.innerHTML = climateTouched(w)
      ? 'Your target decade is not the recorded one, so it has no observed truth: the white ticks are the real 1989–99 decade.'
      : `Closer to the observed change than calibrated GR4J on <b>${closer} of 6</b> signatures; observed inside the 68 % bar on <b>${inside} of 6</b>.`;

    // regulation
    // storage / annual flow: HydroATLAS for the real basin, plus k / 365 for the store you added
    const truthDor = info.dor + w.res_k / 365;
    probs(regBox, 'Storage > 10 % of annual flow', [
      { label: 'regulated', p: now.p, truth: truthDor > 0.1 },
      { label: 'not', p: 1 - now.p, truth: truthDor <= 0.1 }]);
    const [ld, lds] = now.get('log_dor');
    gauges(regBox, [{ label: 'Storage / annual flow (log10)', value: ld, lo: ld - lds, hi: ld + lds, digits: 2, range: [-3, 1],
      truth: truthDor > 0 ? Math.log10(truthDor) : null, truthLabel: w.res_k > 0 ? 'HydroATLAS + your store' : 'HydroATLAS' }]);

    // hidden properties (no real truth)
    if (!ctx.compact) {
      const ex = (tt: string) => { const [m, s] = now.get(tt); return [Math.exp(m), Math.exp(m - s), Math.exp(m + s)]; };
      const [ks, ksl, ksh] = ex('log_kslow'), [sc, scl, sch] = ex('log_soilcap'), [pm, pml, pmh] = ex('log_pmult');
      const [gw, gws] = now.get('gw_share');
      gauges(hidBox, [
        { label: 'Slow-store time constant', unit: 'd', value: ks, lo: ksl, hi: ksh, range: [5, 3000], log: true, digits: 0 },
        { label: 'Groundwater share of flow', value: gw, lo: gw - gws, hi: gw + gws, range: [0, 1], digits: 2 },
        { label: 'Soil water capacity', unit: 'mm', value: sc, lo: scl, hi: sch, range: [20, 2000], log: true, digits: 0 },
        { label: 'True P / forcing P', value: pm, lo: pml, hi: pmh, range: [0.6, 1.6], digits: 2 },
      ]);
    }

    // what the network reads (selected features), recorded -> now
    const n = weights.meta.names;
    const xv = (r: Reading, k: string) => r.x[n.indexOf(k)];
    const rows: [string, (r: Reading) => number, (v: number) => string][] = [
      ['target-decade P vs gauged', (r) => Math.exp(xv(r, 'dt_logP')) - 1, (v) => (v >= 0 ? '+' : '−') + Math.abs(v * 100).toFixed(1) + '%'],
      ['target-decade T vs gauged', (r) => xv(r, 'dt_Tmean'), (v) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2) + ' °C'],
      ['runoff ratio', (r) => xv(r, 'q_rr'), (v) => v.toFixed(3)],
      ['baseflow index (LH)', (r) => xv(r, 'q_bfi_lh'), (v) => v.toFixed(3)],
      ['recession K', (r) => Math.exp(xv(r, 'q_logreck')), (v) => (Number.isFinite(v) ? v.toFixed(1) + ' d' : 'none')],
      ['seasonal damping', (r) => xv(r, 'q_seasdamp'), (v) => v.toFixed(3)],
      ['flat days (|ΔQ| < 1 %)', (r) => xv(r, 'q_flat'), (v) => (v * 100).toFixed(1) + '%'],
      ['Q5 / mean flow', (r) => Math.exp(xv(r, 'q_fdc95')), (v) => v.toFixed(3)],
    ];
    feat.replaceChildren(h('tr', {}, h('td', {}, ''), h('td', {}, 'recorded'), h('td', {}, 'now')),
      ...rows.map(([name, fn, f]) => {
        const a = fn(recd), b = fn(now);
        const moved = Math.abs(a - b) > 1e-9 * Math.max(1, Math.abs(a)) || Number.isNaN(a) !== Number.isNaN(b);
        return h('tr', {}, h('td', {}, name), h('td', {}, f(a)), h('td', moved ? { class: 'moved' } : {}, f(b)));
      }));

    // what to look at
    const [mq] = now.get('d_logq'), [mq0] = recd.get('d_logq');
    const [hd] = now.get('d_hfd'), [hd0] = recd.get('d_hfd');
    const bits: string[] = [];
    if (climateTouched(w)) {
      bits.push(`Mean flow change ${fmtChange('d_logq', mq0)} → ${fmtChange('d_logq', mq)}; half-flow date ${fmtChange('d_hfd', hd0)} → ${fmtChange('d_hfd', hd)}.`);
      if (Math.abs(w.pmul - 1) > 0.12 || w.dT > 1 || w.dT < -1) bits.push('Outside the training range (P ×0.88–1.12, ±1 °C): this is extrapolation.');
      const snowy = now.x[weights.meta.names.indexOf('c_fsnow')] >= 0.15;
      if (w.dT > 0.5 && snowy) bits.push('In a snowy basin warming melts snow earlier: watch the half-flow date move earlier.');
      if (w.dT > 0.5 && !snowy) bits.push('Little snow here: warming acts mostly through higher evaporation (PET +2.5 % per °C).');
    }
    if (w.res_k > 0) {
      const p0 = recd.p;
      bits.push(`A ${w.res_k}-day store holds ${(w.res_k / 365).toFixed(2)}× the annual flow; p(regulated) ${p0.toFixed(2)} → ${now.p.toFixed(2)}.`
        + (now.p < p0 + 0.05 ? ' The network learned reservoirs from WaterSim’s operated ones (SOP, rule curve, Hanasaki); a smooth linear store does not look like them to it. On real basins its reservoir alarm was at chance (exam AUC 0.49).' : ''));
    }
    if (w.cv > 0 || w.sig > 0) bits.push('Gauge texture feeds the baseflow and recession features; the hidden properties move more than the changes.');
    if (w.years < 9) bits.push(`Reading only ${w.years} gauged years: features of the record get noisier and the bars should widen.`);
    if (!touched) bits.push(`Network ${fmtChange('d_logq', mq0)} mean flow; observed ${fmtChange('d_logq', info.observed_change.d_logq)}, GR4J ${fmtChange('d_logq', info.gr4j_change.d_logq)}. On all 619 exam basins it is level with calibrated GR4J on five of six changes and behind it on mean flow (ρ 0.76 vs 0.79), without calibrating anything.`);
    note.textContent = bits.join(' ');

    plot1.redraw();
    plot2.redraw();
    ui.stamp(ms, touched ? 'what-if input: 150 features recomputed, 5 networks rerun' : `${NICE.d_logq[0]} and five more signatures, from 150 features`);
  }

  render(recd.ms);
}
