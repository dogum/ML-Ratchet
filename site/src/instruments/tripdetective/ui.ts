// TripDetective live instrument. Real mode: 13 real VED exam cars (never used in training), up to 8 trips each — pick a
// trip, pool more trips of the same car, cut the logs short, or take channels away from the logger, and the network
// reads the car again. Lab mode: three simulated cars (RoadSim + OBD emulator logs, precomputed in Python) whose mass,
// drag area and control strategy you can change; the network still runs live on the logs.
import type { InstrumentContext } from '../types.ts';
import { fetchPack } from '../../lib/nn/pack.ts';
import { examRaw, labRaw, apply, run as runNet, reading, VEDTYPE, CURB_OFFSET, type TMeta, type XMeta, type Raw, type Flags, type Reading, type Channel } from './net.ts';
import { shell, slider, picker, scheduler, probs, gauges, canvas, gridLines, label, h, fmt, SCOPE, type Canvas2D } from '../kit.ts';

const CYAN = '#45c3cd';
const TYPES: [string, string][] = [['ICE', 'Conventional'], ['HEV', 'Hybrid'], ['PHEV', 'Plug-in hybrid'], ['EV', 'Electric']];
const ANAME: Record<string, string> = {
  conv: 'conventional', hev_powersplit: 'power-split hybrid', hev_parallel_p2: 'parallel (P2) hybrid', hev_series: 'series hybrid',
  phev_powersplit: 'power-split plug-in', phev_series: 'series plug-in', bev: 'battery electric',
};
const SNAME: Record<string, string> = {
  conv_rule: 'shift rule', conv_opt: 'optimal shifting', conv_stopstart: 'stop-start', aecms: 'adaptive ECMS', ecms: 'ECMS',
  powersplit_rule: 'engine-on threshold rule', parallel_rule: 'electric-assist rule', series_thermostat: 'thermostat', series_power_follower: 'power follower',
  phev_cdcs: 'deplete, then sustain', phev_blended: 'blended', bev_regen_series: 'regen first', bev_regen_parallel: 'regen + friction blend', bev_one_pedal: 'one-pedal',
};
const LEDGER = ['aero', 'rolling', 'grade', 'brake', 'powertrain', 'A/C + aux'];
const CHANNELS: { k: Channel; name: string; flag: number }[] = [
  { k: 'fuel', name: 'Fuel rate', flag: 0 }, { k: 'rpm', name: 'Engine rpm', flag: 2 }, { k: 'batt', name: 'HV battery', flag: 3 }, { k: 'aux', name: 'A/C power', flag: 4 },
];
const FULL = 1020; // log-length slider top = whole trip (the longest trip here is 983 s)

const STYLE = `
.td-mode { display: flex; gap: 0; margin: -4px 0 10px; border: 1px solid var(--scr-rule); border-radius: 4px; width: fit-content; max-width: 100%; overflow: hidden; }
.td-mode button { padding: 6px 12px; background: transparent; border: 0; color: var(--scr-2ink); cursor: pointer; font-family: var(--mono); font-variation-settings: 'wdth' 87.5; font-size: 10.5px; letter-spacing: 0.06em; text-transform: uppercase; }
.td-mode button + button { border-left: 1px solid var(--scr-rule); }
.td-mode button[aria-pressed='true'] { color: var(--scr); background: var(--scr-ink); }
.td-chips { display: flex; flex-wrap: wrap; align-items: center; gap: 5px; margin-bottom: 8px; }
.td-chips .ig-l { margin-right: 4px; }
.td-chips .inst-pick { padding: 4px 9px; min-width: 30px; align-items: center; }
.td-toggles { display: flex; flex-wrap: wrap; gap: 6px; align-items: flex-end; }
.td-toggles .ig-l { width: 100%; }
.td-tg { padding: 6px 9px; border-radius: 4px; border: 1px solid var(--c); background: color-mix(in srgb, var(--c) 22%, transparent); color: var(--scr-ink); font-family: var(--mono); font-variation-settings: 'wdth' 87.5; font-size: 10.5px; cursor: pointer; }
.td-tg[aria-pressed='false'] { border-color: var(--scr-rule); background: transparent; color: var(--scr-3ink); text-decoration: line-through; }
.td-tg:disabled { cursor: default; opacity: 0.6; }
.td-truth { font-size: 12px; line-height: 1.45; color: var(--scr-3ink); border: 1px dashed var(--scr-rule); border-radius: 4px; padding: 8px 10px; }
.td-truth b { font-weight: 400; color: var(--scr-ink); }
.td-truth .sim { font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--scr); background: var(--c); padding: 1px 5px; border-radius: 2px; margin-right: 4px; }
.td-kv { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 5px 12px; align-items: baseline; }
.td-kv .v { font-family: var(--mono); font-variation-settings: 'wdth' 87.5; font-size: 12px; color: var(--scr-ink); text-align: right; }
.td-kv .v small { color: var(--scr-3ink); font-size: 10px; margin-left: 4px; }
.td-led { display: flex; flex-direction: column; gap: 6px; }
.td-bar { display: flex; height: 11px; border-radius: 2px; overflow: hidden; background: var(--scr-rule); }
.td-bar span { height: 100%; min-width: 0; transition: width 0.35s var(--ease); }
.td-row { display: grid; grid-template-columns: 62px minmax(0, 1fr); gap: 8px; align-items: center; font-family: var(--mono); font-variation-settings: 'wdth' 87.5; font-size: 9.5px; color: var(--scr-3ink); }
.td-leg { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 3px 10px; font-family: var(--mono); font-variation-settings: 'wdth' 87.5; font-size: 10px; color: var(--scr-2ink); }
.td-leg i { display: inline-block; width: 8px; height: 8px; border-radius: 1px; margin-right: 5px; vertical-align: 0; }
.td-leg em { font-style: normal; color: var(--scr-3ink); }
.td-note { font-size: 11.5px; line-height: 1.45; color: var(--scr-3ink); margin: 0; }
`;

const n0 = (v: number) => Math.round(v).toLocaleString('en-US').replace('-', '−');
const pct = (v: number) => `${Math.round(100 * v)}%`;

export async function mount(ctx: InstrumentContext) {
  const [W, X] = await Promise.all([fetchPack<TMeta>(`${ctx.base}weights`), fetchPack<XMeta>(`${ctx.base}examples`)]);
  const cars = X.meta.cars, LAB = X.meta.lab;
  const LCOL = [CYAN, '#c9c3b3', '#7d8276', '#ff8a7a', ctx.accent, '#4cc38a'];
  if (!document.getElementById('td-style')) document.head.append(h('style', { id: 'td-style' }, STYLE));
  const ui = shell(ctx.root, {
    title: 'TripDetective',
    subtitle: 'what car drove this? read from its trip logs alone — real cars, never used in training',
    params: `${(W.meta.params / 1e6).toFixed(2)} M parameters · matches Python to 3e-5`,
  });

  // ---------------- state ----------------
  const LARCH = ['conv', 'hev_powersplit', 'bev'];
  const st = {
    mode: 'real' as 'real' | 'lab',
    car: 0, trip: 0, k: 1, cut: FULL, drop: new Set<Channel>(),
    arch: 'conv', strat: 'conv_rule', m: 1, c: 1,
  };
  const rawCache = new Map<string, Raw>();
  const tripRaw = (ci: number, ti: number) => {
    const key = `${ci}:${ti}`;
    if (!rawCache.has(key)) rawCache.set(key, examRaw(X, cars[ci].trips[ti]));
    return rawCache.get(key)!;
  };
  const labIndex = (arch: string, strat: string, m: number, c: number) =>
    LAB.configs.findIndex((q) => q.arch === arch && q.strat === strat && q.dm === LAB.mass[m] && q.fc === LAB.cda[c]);

  // ---------------- layout ----------------
  const modeRow = h('div', { class: 'td-mode', role: 'group', 'aria-label': 'Which cars' });
  const realBtn = h('button', { type: 'button', 'aria-pressed': 'true' }, 'Real exam cars');
  const labBtn = h('button', { type: 'button', 'aria-pressed': 'false' }, 'Simulated lab');
  modeRow.append(realBtn, labBtn);
  if (!ctx.compact) ui.picker.before(modeRow);
  const chips = h('div', { class: 'td-chips', role: 'group' });
  ui.screen.before(chips);
  type View = { kind: 'real'; raw: Raw; used: Raw; flags: Flags; usedFlags: Flags; cut: number | null; pooled: number }
    | { kind: 'lab'; raws: Raw[]; base: Raw[]; flags: Flags };
  let view: View | null = null;
  const narrow = () => ctx.root.getBoundingClientRect().width < 560;
  const plot = canvas(ui.screen, ctx.compact ? 2.3 : narrow() ? 1.55 : 2.1, (c) => draw(c));

  const truthBox = h('div', { class: 'td-truth' });
  const typeBox = h('div', {}), gaugeBox = h('div', { style: 'display:flex;flex-direction:column;gap:14px' });
  const kvBox = h('div', { class: 'td-kv' }), ledBox = h('div', { class: 'td-led' });
  ui.readout.append(truthBox, typeBox, gaugeBox, kvBox, ledBox);

  realBtn.addEventListener('click', () => setMode('real'));
  labBtn.addEventListener('click', () => setMode('lab'));

  function setMode(m: 'real' | 'lab') {
    st.mode = m;
    realBtn.setAttribute('aria-pressed', String(m === 'real'));
    labBtn.setAttribute('aria-pressed', String(m === 'lab'));
    if (m === 'real') buildReal(); else buildLab();
    go();
  }

  // ---------------- real mode ----------------
  function buildReal() {
    picker(ui.picker, cars.map((c, i) => ({
      label: `#${c.vid.replace('VED_', '')} · ${c.ved_type}`,
      sub: `${c.weight_lb ? `${c.weight_lb.toLocaleString('en-US')} lb` : '? lb'} · ${c.disp ? `${c.disp} L` : 'no engine'}`,
      value: i,
    })), (i) => { st.car = i; st.trip = 0; st.drop.clear(); buildReal(); go(); }, st.car);
    buildTripChips();
    ui.controls.innerHTML = '';
    const nt = cars[st.car].trips.length;
    st.k = Math.min(st.k, nt);
    slider(ui.controls, {
      label: 'Trips pooled', min: 1, max: nt, step: 1, value: st.k,
      fmt: (v) => (v === 1 ? 'this trip only' : `this + next ${v - 1}`),
      hint: 'more trips of the same car, averaged inside the network',
      onInput: (v) => { st.k = v; go(); },
    });
    if (!ctx.compact) {
      slider(ui.controls, {
        label: 'Keep the first', min: 120, max: FULL, step: 60, value: st.cut,
        fmt: (v) => (v >= FULL ? 'whole trip' : `${v / 60} min`),
        hint: 'of every pooled trip; the rest of the log is thrown away',
        onInput: (v) => { st.cut = v; go(); },
      });
    }
    // which channels the logger had: only those this car's logger recorded can be taken away
    const has = (f: number) => cars[st.car].trips.some((t) => t.flags[f] > 0.5);
    const box = h('div', { class: 'td-toggles' }, h('span', { class: 'ig-l' }, 'Logger channels (tap to remove)'));
    const btns: { k: Channel; b: HTMLButtonElement }[] = [];
    for (const ch of CHANNELS) {
      if (!has(ch.flag) || (ctx.compact && ch.k === 'aux')) continue;
      const b = h('button', { type: 'button', class: 'td-tg' }, ch.name) as HTMLButtonElement;
      b.title = `Give the network the log without ${ch.name.toLowerCase()}, as if the logger never recorded it`;
      b.addEventListener('click', () => { if (st.drop.has(ch.k)) st.drop.delete(ch.k); else st.drop.add(ch.k); paint(); go(); });
      btns.push({ k: ch.k, b });
      box.append(b);
    }
    const paint = () => {
      for (const { k, b } of btns) {
        const off = st.drop.has(k) || (k === 'aux' && st.drop.has('batt'));
        b.setAttribute('aria-pressed', String(!off));
        b.disabled = k === 'aux' && st.drop.has('batt');
      }
    };
    paint();
    box.append(h('span', { class: 'td-note', style: 'width:100%' }, 'Speed, grade and weather are always given.'));
    ui.controls.append(box);
  }
  function buildTripChips() {
    chips.innerHTML = '';
    chips.append(h('span', { class: 'ig-l' }, 'Trip'));
    const c = cars[st.car];
    const el = h('div', { style: 'display:contents' });
    chips.append(el);
    picker(el, c.trips.map((_, i) => ({ label: String(i + 1), value: i })), (i) => { st.trip = i; go(); }, st.trip);
    el.querySelectorAll('button').forEach((b, i) => { b.title = `${c.trips[i].km} km · ${Math.round(c.trips[i].n / 60)} min · ${c.trips[i].T.toFixed(0)} °C`; });
  }

  // ---------------- lab mode ----------------
  function buildLab() {
    picker(ui.picker, LARCH.map((a) => ({ label: `sim · ${ANAME[a]}`, sub: 'RoadSim, 2 real routes', value: a })), (a) => {
      st.arch = a; st.strat = LAB.strat[a][0]; buildLab(); go();
    }, LARCH.indexOf(st.arch));
    chips.innerHTML = '';
    chips.append(h('span', { class: 'ig-l' }, 'Control'));
    const el = h('div', { style: 'display:contents' });
    chips.append(el);
    const ss = LAB.strat[st.arch];
    picker(el, ss.map((s) => ({ label: SNAME[s] ?? s, value: s })), (s) => { st.strat = s; go(); }, Math.max(0, ss.indexOf(st.strat)));
    ui.controls.innerHTML = '';
    slider(ui.controls, {
      label: 'Extra mass', min: 0, max: LAB.mass.length - 1, step: 1, value: st.m,
      fmt: (i) => `${LAB.mass[i] > 0 ? '+' : LAB.mass[i] < 0 ? '−' : '±'}${Math.abs(LAB.mass[i])} kg`,
      hint: 'cargo added to (or taken from) the simulated car',
      onInput: (i) => { st.m = i; go(); },
    });
    slider(ui.controls, {
      label: 'Drag area CdA', min: 0, max: LAB.cda.length - 1, step: 1, value: st.c,
      fmt: (i) => `×${LAB.cda[i].toFixed(1)}`,
      hint: 'the simulated car’s CdA scaled, same everything else',
      onInput: (i) => { st.c = i; go(); },
    });
  }

  // ---------------- run ----------------

  async function run() {
    ui.setBusy(true);
    await new Promise((r) => requestAnimationFrame(r));
    if (st.mode === 'real') runReal(); else runLab();
    ui.setBusy(false);
  }

  function runReal() {
    const c = cars[st.car], nt = c.trips.length, k = Math.min(st.k, nt);
    const cut = st.cut >= FULL ? null : st.cut, drop = [...st.drop];
    const idx = Array.from({ length: k }, (_, i) => (st.trip + i) % nt);
    const trips = idx.map((ti) => apply(tripRaw(st.car, ti), c.trips[ti].flags, drop, cut));
    const t0 = performance.now();
    const out = runNet(W, trips);
    const r = reading(W, out, trips.length);
    const ms = performance.now() - t0;
    const raw = tripRaw(st.car, st.trip);
    view = { kind: 'real', raw, used: trips[0].raw, flags: c.trips[st.trip].flags, usedFlags: trips[0].flags, cut, pooled: k };
    plot.redraw();
    renderReal(r, k);
    const mins = trips.reduce((s, t) => s + t.raw.n, 0) / 60;
    ui.stamp(ms, `${c.vid}: ${k} trip${k > 1 ? 's' : ''}, ${mins.toFixed(0)} min of driving${drop.length ? `, without ${drop.map((d) => CHANNELS.find((q) => q.k === d)!.name.toLowerCase()).join(' + ')}` : ''}`);
  }

  function renderReal(r: Reading, k: number) {
    const c = cars[st.car], t = c.trips[st.trip];
    truthBox.innerHTML = `Published, hidden from the network: <b>${TYPES.find((x) => x[0] === c.ved_type)?.[1] ?? c.ved_type}</b>, `
      + `${c.engine && c.engine !== 'NO DATA' ? `${c.engine}, ` : ''}weight class <b>${c.weight_lb?.toLocaleString('en-US') ?? '?'} lb</b> `
      + `(curb ${c.curb_kg ? n0(c.curb_kg) : '?'} kg, rounded to 500 lb)${c.likely ? `. Likely ${c.likely.split(';')[0]}` : ''}.`
      + `<br>Trip ${st.trip + 1}: ${t.km} km, ${Math.round(t.n / 60)} min, ${t.T.toFixed(0)} °C.`;
    probs(typeBox, `Type · ${k} trip${k > 1 ? 's' : ''}`, TYPES.map(([key, name]) => ({ label: name, p: r.type[key] ?? 0, truth: key === c.ved_type })));
    const isEV = (r.type.EV ?? 0) >= 0.5;
    const gs = [{
      label: 'Curb mass', unit: 'kg', value: r.curb, lo: r.massLo - CURB_OFFSET, hi: r.massHi - CURB_OFFSET,
      truth: c.curb_kg, truthLabel: 'VED class', range: [900, 2800] as [number, number], digits: 0,
    }];
    if (!isEV) gs.push({ label: 'Engine size', unit: 'L', value: r.disp, lo: r.dispLo, hi: r.dispHi, truth: c.disp, truthLabel: 'VED', range: [0.8, 5], digits: 1 });
    gs.push({ label: 'Drag area CdA', unit: 'm²', value: r.CdA, lo: r.CdA * Math.exp(-r.CdAsig), hi: r.CdA * Math.exp(r.CdAsig), truth: null, truthLabel: '', range: [0.4, 1.2], digits: 2 });
    gauges(gaugeBox, gs);
    const F60 = roadForce(r.mass, r.Crr, r.CdA, 60);
    const rows: [string, string][] = [];
    if (isEV) rows.push(['Engine', 'none (electric)']);
    rows.push(['Rolling resistance Crr', `${r.Crr.toFixed(4)}<small>±${pct(r.Crrsig)}</small>`]);
    rows.push(['Road load at 60 mph', `${n0(F60)} N${c.F60_N ? `<small>EPA target ${n0(c.F60_N)} N</small>` : ''}`]);
    const best = TYPES.reduce((a, b) => ((r.type[b[0]] ?? 0) > (r.type[a[0]] ?? 0) ? b : a))[0];
    if (best === 'HEV' || best === 'PHEV') {
      rows.push(['Control', `${SNAME[r.strat[0][0]] ?? r.strat[0][0]}<small>${pct(r.strat[0][1])}</small>`]);
      rows.push(['Engine off while moving', pct(Math.max(0, Math.min(1, r.evShare)))]);
    }
    kv(rows);
    ledger(`Trip ${st.trip + 1}: where the energy went`, [['reading', r.ledger[0].mu, r.ledger[0].sig]],
      'Simulator-trained; no real car measures this split, so it is not graded here.');
  }

  function runLab() {
    const m = st.m, c = st.c;
    const ci = labIndex(st.arch, st.strat, m, c), cfg = LAB.configs[ci];
    const trips = [0, 1].map((r) => labRaw(X, ci, r));
    const t0 = performance.now();
    const out = runNet(W, trips);
    const r = reading(W, out, 2);
    const ms = performance.now() - t0;
    const bi = labIndex(st.arch, st.strat, LAB.mass.indexOf(0), LAB.cda.indexOf(1));
    view = { kind: 'lab', raws: trips.map((t) => t.raw), base: [0, 1].map((q) => labRaw(X, bi, q).raw), flags: trips[0].flags };
    plot.redraw();
    renderLab(r, cfg);
    ui.stamp(ms, `simulated ${ANAME[cfg.arch]}, ${SNAME[cfg.strat]}, 2 routes pooled`);
  }

  function renderLab(r: Reading, cfg: XMeta['lab']['configs'][number]) {
    truthBox.innerHTML = `<span class="sim">simulated</span> A RoadSim ${ANAME[cfg.arch]} driven over two real Ann Arbor routes, logged by the OBD emulator `
      + `(${cfg.arch === 'bev' ? 'HV battery' : 'fuel rate + rpm'}). Logs precomputed in Python; the network reads them live.`;
    probs(typeBox, 'Type · 2 routes', TYPES.map(([key, name]) => ({ label: name, p: r.type[key] ?? 0, truth: key === VEDTYPE[cfg.arch] })));
    const gs = [
      { label: 'Total mass', unit: 'kg', value: r.mass, lo: r.massLo, hi: r.massHi, truth: cfg.mass, truthLabel: 'simulated', range: [1000, 2800] as [number, number], digits: 0 },
      { label: 'Drag area CdA', unit: 'm²', value: r.CdA, lo: r.CdA * Math.exp(-r.CdAsig), hi: r.CdA * Math.exp(r.CdAsig), truth: cfg.CdA, truthLabel: 'simulated', range: [0.3, 1.2] as [number, number], digits: 2 },
    ];
    if (cfg.disp) gs.push({ label: 'Engine size', unit: 'L', value: r.disp, lo: r.dispLo, hi: r.dispHi, truth: cfg.disp, truthLabel: 'simulated', range: [0.8, 5], digits: 1 });
    gauges(gaugeBox, gs);
    const ok = (a: boolean) => (a ? ' ✓' : ' ✗');
    kv([
      ['Layout', `${ANAME[r.arch[0][0]]}${ok(r.arch[0][0] === cfg.arch)}<small>${pct(r.arch[0][1])}</small>`],
      ['Control', `${SNAME[r.strat[0][0]] ?? r.strat[0][0]}${ok(r.strat[0][0] === cfg.strat)}<small>${pct(r.strat[0][1])}</small>`],
      ['Rolling resistance Crr', `${r.Crr.toFixed(4)}<small>simulated ${cfg.Crr.toFixed(4)}</small>`],
    ]);
    ledger('Route 1: where the energy went', [['reading', r.ledger[0].mu, r.ledger[0].sig], ['simulated', cfg.ledger[0], null]],
      'Mass and drag changes show up only weakly: the network leans on what kind of car it is.');
  }

  // ---------------- readout pieces ----------------
  function kv(rows: [string, string][]) {
    kvBox.innerHTML = rows.map(([k, v]) => `<span class="ig-l">${k}</span><span class="v">${v}</span>`).join('');
  }
  function ledger(title: string, bars: [string, number[], number[] | null][], note: string) {
    const bar = (mu: number[]) => {
      const pos = mu.map((x) => Math.max(x, 0)), tot = pos.reduce((a, b) => a + b, 0) || 1;
      return `<div class="td-bar">${pos.map((x, i) => `<span style="width:${((100 * x) / tot).toFixed(1)}%;background:${LCOL[i]}" title="${LEDGER[i]} ${n0(mu[i])} kJ/km"></span>`).join('')}</div>`;
    };
    const [, mu, sig] = bars[0], truth = bars[1]?.[1];
    ledBox.innerHTML = `<span class="ig-l">${title} · kJ/km</span>`
      + bars.map(([name, m]) => (bars.length > 1 ? `<div class="td-row"><span>${name}</span>${bar(m)}</div>` : bar(m))).join('')
      + `<div class="td-leg">${LEDGER.map((k, i) => `<span><i style="background:${LCOL[i]}"></i>${k} ${n0(mu[i])}${sig ? `<em>±${n0(sig[i])}</em>` : ''}${truth ? `<em> (${n0(truth[i])})</em>` : ''}</span>`).join('')}</div>`
      + `<p class="td-note">${truth ? 'In brackets: the simulator’s own ledger. ' : ''}${note}</p>`;
  }

  // ---------------- drawing ----------------
  function series(c: Canvas2D, a: ArrayLike<number>, n: number, x0: number, w: number, y: (v: number) => number, color: string, alpha = 1, dash: number[] = []) {
    const g = c.ctx;
    g.strokeStyle = color; g.globalAlpha = alpha; g.lineWidth = 1.2; g.setLineDash(dash);
    g.beginPath();
    for (let i = 0; i < n; i++) { const x = x0 + (w * i) / Math.max(n - 1, 1); if (i) g.lineTo(x, y(a[i])); else g.moveTo(x, y(a[i])); }
    g.stroke();
    g.setLineDash([]); g.globalAlpha = 1;
  }
  const range = (arrs: ArrayLike<number>[], n: number[], lo0 = 0, hi0 = 0.5) => {
    let lo = lo0, hi = hi0;
    arrs.forEach((a, k) => { for (let i = 0; i < n[k]; i++) { if (a[i] > hi) hi = a[i]; if (a[i] < lo) lo = a[i]; } });
    return [lo, hi];
  };

  /** One log panel: speed (km/h) on top, the logged energy channel below. */
  function panel(c: Canvas2D, x0: number, w: number, top: number, ih: number, raw: Raw, flags: Flags, opts: { vmax: number; elo: number; ehi: number; useFuel: boolean; useBatt: boolean; off: { fuel: boolean; batt: boolean; rpm: boolean }; base?: Raw; title?: string; cut?: number | null }) {
    const g = c.ctx, n = raw.n, r = raw.rows;
    const hTop = ih * 0.46, gap = ih * 0.08, hBot = ih - hTop - gap, yb0 = top + hTop + gap;
    const ys = (v: number) => top + hTop - (hTop * v * 3.6) / opts.vmax;
    const ye = (v: number) => yb0 + hBot - (hBot * (v - opts.elo)) / (opts.ehi - opts.elo || 1);
    gridLines(c, x0, top, w, hTop, 6, 2);
    gridLines(c, x0, yb0, w, hBot, 6, 2);
    // engine off while moving (rpm logged as 0)
    if (flags[2] > 0.5 && !opts.off.rpm) {
      g.fillStyle = CYAN;
      for (let i = 0; i < n; i++) if (r[6][i] < 0.1 && r[0][i] > 1) g.fillRect(x0 + (w * i) / n, yb0 + hBot - 4, Math.max(1, w / n + 0.5), 4);
    }
    series(c, r[0], n, x0, w, ys, SCOPE.ink, 0.9);
    if (opts.elo < 0) { g.strokeStyle = SCOPE.faint; g.beginPath(); g.moveTo(x0, ye(0) + 0.5); g.lineTo(x0 + w, ye(0) + 0.5); g.stroke(); }
    if (opts.base) {
      if (opts.useFuel) series(c, opts.base.rows[5], n, x0, w, ye, SCOPE.faint, 1);
      if (opts.useBatt) series(c, opts.base.rows[7], n, x0, w, ye, SCOPE.faint, 1);
    }
    if (opts.useFuel && flags[0] > 0.5) series(c, r[5], n, x0, w, ye, ctx.accent, opts.off.fuel ? 0.3 : 1, opts.off.fuel ? [3, 3] : []);
    if (opts.useBatt && flags[3] > 0.5) series(c, r[7], n, x0, w, ye, CYAN, opts.off.batt ? 0.3 : 1, opts.off.batt ? [3, 3] : []);
    if (opts.cut && opts.cut < n) {
      const x = x0 + (w * opts.cut) / n;
      g.fillStyle = 'rgba(0,0,0,0.6)';
      g.fillRect(x, top, x0 + w - x, ih);
      g.strokeStyle = ctx.accent; g.beginPath(); g.moveTo(x + 0.5, top); g.lineTo(x + 0.5, top + ih); g.stroke();
      label(c, 'thrown away', Math.min(x + 6, x0 + w - 70), top + 8, ctx.accent, 'left', 9);
    }
    if (opts.title) label(c, opts.title, x0 + w - 4, top + 8, SCOPE.text, 'right', 9);
    return { yb0, hBot, hTop };
  }

  function draw(c: Canvas2D) {
    const { ctx: g, w, h: hh } = c;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, w, hh);
    if (!view) return;
    const padL = 34, padR = 10, top = 12, bot = 18, iw = w - padL - padR, ih = hh - top - bot;
    if (view.kind === 'real') {
      const { raw, flags, usedFlags } = view, n = raw.n;
      const useFuel = flags[0] > 0.5, useBatt = !useFuel && flags[3] > 0.5;
      const showB = flags[3] > 0.5 && useFuel; // plug-ins: fuel and battery on the same axis would mix units; draw battery on its own scale
      const vmax = Math.max(20, Math.ceil(range([raw.rows[0]], [n])[1] * 3.6 / 20) * 20);
      const [elo, ehi] = useFuel ? range([raw.rows[5]], [n]) : range([raw.rows[7]], [n]);
      const off = { fuel: usedFlags[0] < 0.5, batt: usedFlags[3] < 0.5, rpm: usedFlags[2] < 0.5 };
      const p = panel(c, padL, iw, top, ih, raw, flags, { vmax, elo, ehi, useFuel, useBatt, off, cut: view.cut });
      if (showB) {
        const [blo, bhi] = range([raw.rows[7]], [n], 0, 1);
        const yb = (v: number) => p.yb0 + p.hBot - (p.hBot * (v - blo)) / (bhi - blo || 1);
        series(c, raw.rows[7], n, padL, iw, yb, CYAN, off.batt ? 0.25 : 0.75, off.batt ? [3, 3] : []);
      }
      label(c, `${vmax}`, padL - 4, top + 3, SCOPE.faint, 'right', 9);
      label(c, 'km/h', padL - 4, top + p.hTop - 4, SCOPE.faint, 'right', 9);
      label(c, 'speed', padL + 4, top + 8, SCOPE.text, 'left', 9);
      const eName = useFuel ? `fuel ${fmt(ehi, 1)} g/s` : useBatt ? `battery ${fmt(ehi, 0)} kW` : 'no fuel or battery channel';
      label(c, eName, padL + 4, p.yb0 + 8, useFuel ? ctx.accent : useBatt ? CYAN : SCOPE.faint, 'left', 9);
      if (showB) label(c, 'battery kW', padL + 4, p.yb0 + 20, CYAN, 'left', 9);
      if ((useFuel && off.fuel) || (useBatt && off.batt)) label(c, 'removed: not given to the network', padL + iw - 4, p.yb0 + 8, SCOPE.text, 'right', 9);
      if (flags[2] > 0.5 && !off.rpm && iw > 300) label(c, 'cyan strip: engine off while moving', padL + iw - 4, p.yb0 + p.hBot - 7, CYAN, 'right', 9);
      const step = n > 600 ? 300 : 120;
      for (let s = 0; s <= n; s += step) label(c, `${s / 60} min`, padL + (iw * s) / n, hh - 8, SCOPE.faint, s === 0 ? 'left' : 'center', 9);
      if (view.pooled > 1) label(c, `+ ${view.pooled - 1} more trip${view.pooled > 2 ? 's' : ''} pooled`, padL + iw - 4, top + 8, SCOPE.text, 'right', 9);
    } else {
      const { raws, base, flags } = view;
      const useFuel = flags[0] > 0.5, useBatt = !useFuel;
      const ns = raws.map((r) => r.n), tot = ns[0] + ns[1], gap = 14, iw2 = iw - gap;
      const vmax = Math.max(20, Math.ceil(Math.max(...raws.map((r, k) => range([r.rows[0]], [ns[k]])[1])) * 3.6 / 20) * 20);
      const ch = useFuel ? 5 : 7;
      const [elo, ehi] = range([...raws, ...base].map((r) => r.rows[ch]), [...ns, ...ns]);
      const off = { fuel: false, batt: false, rpm: false };
      let x = padL, p = { yb0: 0, hBot: 0, hTop: 0 };
      raws.forEach((r, k) => {
        const wk = (iw2 * ns[k]) / tot;
        p = panel(c, x, wk, top, ih, r, flags, { vmax, elo, ehi, useFuel, useBatt, off, base: base[k], title: `route ${k + 1} · ${Math.round(ns[k] / 60)} min` });
        x += wk + gap;
      });
      label(c, `${vmax}`, padL - 4, top + 3, SCOPE.faint, 'right', 9);
      label(c, 'km/h', padL - 4, top + p.hTop - 4, SCOPE.faint, 'right', 9);
      label(c, useFuel ? `fuel ${fmt(ehi, 1)} g/s` : `battery ${fmt(ehi, 0)} kW`, padL + 4, p.yb0 + 8, useFuel ? ctx.accent : CYAN, 'left', 9);
      if (iw > 300) label(c, 'grey: the car as built (±0 kg, CdA ×1.0)', padL + 4, hh - 8, SCOPE.faint, 'left', 9);
    }
  }

  const go = scheduler(run, 30);
  buildReal();
  await run();
}

/** Road-load force at a speed from mass, Crr and CdA (as the lab page: tyre speed term 0.004 per m/s, air 0.6 kg/m³ × CdA). */
function roadForce(m: number, crr: number, cda: number, mph: number) {
  const v = mph * 0.44704;
  return m * 9.81 * crr * (1 + 0.004 * v) + 0.6 * cda * v * v;
}

