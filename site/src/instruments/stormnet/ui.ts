// StormNet storm replay: pick a whole real storm, scrub or play through it, and watch every forecaster's
// three-hour call against what the ring current actually did. What-if winds rerun the network live.
import { fetchPack, type Pack } from '../../lib/nn/pack.ts';
import type { InstrumentContext } from '../types.ts';
import { shell, picker, slider, button, gauges, canvas, h } from '../kit.ts';
import { predict, replay, stormSeries, windowAt, RECORDED, type Scenario, type StormWeights, type Storms } from './net.ts';
import { drawScope, hourAt, type ScopeState } from './scope.ts';

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

export async function mount(ctx: InstrumentContext) {
  const [weights, storms] = await Promise.all([
    fetchPack<StormWeights>(ctx.base + 'weights'), fetchPack<Storms>(ctx.base + 'storms')]);
  show(ctx, weights, storms);
}

export function show(ctx: InstrumentContext, weights: Pack<StormWeights>, storms: Pack<Storms>) {
  const list = storms.meta.storms;
  const ui = shell(ctx.root, { title: 'StormNet', subtitle: 'whole real storms, replayed hour by hour · three-hour Dst forecast' });
  ctx.root.classList.add('sn-explorer');
  ctx.root.append(h('style', {}, `
    .sn-key { display: flex; flex-wrap: wrap; gap: 6px 14px; padding: 7px 10px; font: 10.5px/1.5 var(--mono, monospace); color: var(--scr-2ink); }
    .sn-key button { all: unset; cursor: pointer; display: inline-flex; align-items: center; gap: 5px; opacity: .45; }
    .sn-key button[aria-pressed="true"], .sn-key span { opacity: 1; }
    .sn-key i { display: inline-block; width: 14px; border-top: 2px solid var(--c); }
    .sn-key i.dash { border-top-style: dashed; } .sn-key i.band { border-top-width: 6px; opacity: .55; }
    .sn-when { margin: 0 0 6px; color: var(--scr-ink); font: 11px/1.4 var(--mono, monospace); }
    .sn-tbl { width: 100%; border-collapse: collapse; font: 11px/1.5 var(--mono, monospace); color: var(--scr-2ink); margin: 8px 0 4px; }
    .sn-tbl td { padding: 2px 0; } .sn-tbl td:last-child { text-align: right; color: var(--scr-ink); }
    .sn-tbl tr.me td { color: var(--c); }
    .sn-h { font: 10px/1.4 var(--mono, monospace); text-transform: uppercase; letter-spacing: .06em; color: var(--scr-2ink); margin: 12px 0 2px; }
    .sn-res { display: grid; grid-template-columns: 4.6em 1fr 4.2em; gap: 3px 8px; align-items: center; font: 10.5px/1.4 var(--mono, monospace); color: var(--scr-2ink); }
    .sn-res b { display: block; height: 7px; background: var(--c); border-radius: 2px; opacity: .8; }
    .sn-res span:last-child { text-align: right; color: var(--scr-ink); }
    .sn-note { font-size: 12px; line-height: 1.5; color: var(--scr-2ink); margin: 8px 0 0; }
    .sn-row { display: flex; gap: 8px; flex-wrap: wrap; }
    @media (max-width: 540px) { .sn-explorer .inst-picker { display: grid; grid-template-columns: repeat(2, 1fr); } }
  `));
  ctx.root.style.setProperty('--c', ctx.accent);

  const nameOf = (s: (typeof list)[number]) => {
    const [y, m] = s.block.split('-').slice(1, 3);
    return (s.block.startsWith('quiet') ? 'Quiet · ' : '') + MONTHS[+m - 1] + ' ' + y;
  };
  let series = stormSeries(storms, 0), replayMs = 0;
  const runReplay = () => { const t = performance.now(); const out = replay(weights, series.rows, series.dst); replayMs = performance.now() - t; return out; };
  let rp = runReplay();
  let k = rp[rp.length - 1].k, what: Scenario = { ...RECORDED };
  const overlays = { physics: true, published: false, persistence: false };
  let timer = 0;

  // ---- screen
  const box = h('div', { role: 'img', 'aria-label': 'Observed Dst through the storm with every hourly three-hour forecast; solar wind Bz, speed and pressure below' });
  const key = h('div', { class: 'sn-key' });
  ui.screen.append(box, key);
  const state = (): ScopeState => ({
    rows: series.rows, dst: series.dst, replay: rp, k, live: reading(), modified: what.bz !== 0 || what.speed !== 1 || what.pressure !== 1,
    show: overlays, startMs: Date.parse(series.info.start_utc), accent: ctx.accent,
  });
  const plot = canvas(box, ctx.compact ? 1.25 : 1.55, c => drawScope(c, state()));
  const cv = box.querySelector('canvas')!;
  const scrub = (ev: PointerEvent) => {
    const r = cv.getBoundingClientRect();
    const target = hourAt({ cv, ctx: cv.getContext('2d')!, w: r.width, h: r.height }, series.dst.length, ev.clientX - r.left);
    const near = rp.reduce((a, b) => (Math.abs(b.k - target) < Math.abs(a.k - target) ? b : a));
    setK(near.k);
  };
  cv.addEventListener('pointerdown', (ev) => { stop(); cv.setPointerCapture(ev.pointerId); scrub(ev); });
  cv.addEventListener('pointermove', (ev) => { if (cv.hasPointerCapture(ev.pointerId)) scrub(ev); });
  cv.style.cursor = 'ew-resize';

  const keyItems: [keyof typeof overlays | null, string, string, string][] = [
    [null, 'Observed Dst', 'rgba(235,230,216,.95)', ''], [null, 'StormNet ±68%', ctx.accent, 'band'],
    ['physics', 'Tuned two-state physics', '#e0a84c', 'dash'], ['published', 'Published O’Brien–McPherron', 'rgba(190,170,235,.9)', 'dash'],
    ['persistence', 'Persistence', 'rgba(235,230,216,.5)', 'dash'],
  ];
  for (const [flag, text, color, cls] of keyItems) {
    const sw = h('i', { class: cls, style: `--c:${color}` });
    if (!flag) { key.append(h('span', {}, sw, text)); continue; }
    const b = h('button', { type: 'button', 'aria-pressed': String(overlays[flag]) }, sw, text);
    b.addEventListener('click', () => { overlays[flag] = !overlays[flag]; b.setAttribute('aria-pressed', String(overlays[flag])); plot.redraw(); });
    key.append(b);
  }

  // ---- readout
  const when = h('p', { class: 'sn-when' });
  const gauge = h('div');
  const tbl = h('table', { class: 'sn-tbl' });
  const resv = h('div', { class: 'sn-res' });
  const board = h('table', { class: 'sn-tbl' });
  const note = h('p', { class: 'sn-note' });
  ui.readout.append(when, gauge, tbl, h('div', { class: 'sn-h' }, 'Ring current at issue'), resv,
    h('div', { class: 'sn-h' }, 'This storm, every hour'), board, note);

  // ---- controls
  const hourSlider = slider(ui.controls, {
    label: 'Forecast issue', min: 0, max: rp.length - 1, step: 1, value: rp.length - 1,
    fmt: v => stamp(rp[Math.min(v, rp.length - 1)]?.k ?? k), onInput: v => { stop(); setK(rp[v].k); },
  });
  const row = h('div', { class: 'sn-row' });
  ui.controls.append(row);
  const playBtn = button(row, '▶ Play the storm', () => (timer ? stop() : play()));
  const bzS = slider(ui.controls, {
    label: 'What if Bz (last 3 h) shifted by', min: -20, max: 20, step: 1, value: 0,
    fmt: v => (v === 0 ? 'as recorded' : (v > 0 ? '+' : '−') + Math.abs(v) + ' nT'),
    hint: 'South (negative) Bz opens the magnetosphere to the solar wind and feeds the ring current.',
    onInput: v => { what.bz = v; update(); },
  });
  const vS = slider(ui.controls, {
    label: 'What if solar-wind speed (last 3 h) ×', min: 0.6, max: 2, step: 0.05, value: 1,
    fmt: v => (v === 1 ? 'as recorded' : '×' + v.toFixed(2)), onInput: v => { what.speed = v; update(); },
  });
  const pS = slider(ui.controls, {
    label: 'What if dynamic pressure (last 3 h) ×', min: 0.5, max: 4, step: 0.1, value: 1,
    fmt: v => (v === 1 ? 'as recorded' : '×' + v.toFixed(1)),
    hint: 'A pressure pulse squeezes the magnetopause and briefly raises Dst before the storm bites.',
    onInput: v => { what.pressure = v; update(); },
  });
  button(row, 'Restore recorded wind', () => { what = { ...RECORDED }; bzS.set(0); vS.set(1); pS.set(1); update(); });

  picker(ui.picker, list.map((s, i) => ({ label: nameOf(s), sub: (s.block.startsWith('quiet') ? 'min ' : 'Dst ') + s.minimum.toFixed(0) + ' nT · ' + s.source.split(' (')[0], value: i })), i => {
    stop(); series = stormSeries(storms, i); rp = runReplay();
    what = { ...RECORDED }; bzS.set(0); vS.set(1); pS.set(1);
    const deepest = rp.reduce((a, b) => (b.truth < a.truth ? b : a));
    hourSlider.input.max = String(rp.length - 1);
    setK(Math.max(rp[0].k, deepest.k - 6));
    scoreboard();
  });

  function stamp(kk: number) {
    const t = new Date(Date.parse(series.info.start_utc) + kk * 3600000).toISOString();
    return t.slice(5, 10).replace('-', '/') + ' ' + t.slice(11, 16) + ' UTC';
  }
  function reading() {
    const w = windowAt(series.rows, k, what)!;
    return predict(weights, w, series.dst[k]);
  }
  function setK(kk: number) {
    k = kk;
    const j = rp.findIndex(r => r.k === k);
    if (j >= 0) hourSlider.set(j);
    update();
  }
  function play() {
    playBtn.textContent = '❚❚ Pause';
    let j = rp.findIndex(r => r.k === k);
    if (j >= rp.length - 1) j = -1;
    timer = window.setInterval(() => {
      j++;
      if (j >= rp.length) { stop(); return; }
      k = rp[j].k; hourSlider.set(j); update();
    }, 90);
  }
  function stop() { if (timer) { clearInterval(timer); timer = 0; playBtn.textContent = '▶ Play the storm'; } }

  function scoreboard() {
    const n = rp.length;
    const mae = (key: 'minimum' | 'classical' | 'published' | 'persistence') => rp.reduce((s, r) => s + Math.abs(r[key] - r.truth), 0) / n;
    const cover = rp.filter(r => Math.abs(r.minimum - r.truth) <= r.sigma).length / n;
    const rows: [string, number, boolean][] = [['StormNet', mae('minimum'), true], ['Tuned physics', mae('classical'), false],
      ['Published physics', mae('published'), false], ['Persistence', mae('persistence'), false]];
    board.replaceChildren(h('tr', {}, h('td', {}, `mean error, ${n} forecasts`), h('td', {}, '')),
      ...rows.map(([name, v, me]) => h('tr', me ? { class: 'me' } : {}, h('td', {}, name), h('td', {}, v.toFixed(1) + ' nT'))),
      h('tr', {}, h('td', {}, 'StormNet inside its 68% bar'), h('td', {}, (100 * cover).toFixed(0) + '%')));
  }

  function update() {
    const t0 = performance.now();
    const r = reading();
    const ms = performance.now() - t0;
    const modified = what.bz !== 0 || what.speed !== 1 || what.pressure !== 1;
    const fut = series.dst.slice(k + 1, k + 4);
    const truth = modified ? null : Math.min(...fut);
    when.textContent = 'Issued ' + stamp(k) + ' · last Dst ' + series.dst[k].toFixed(0) + ' nT';
    const lo = r.minimum - r.sigma, hi = r.minimum + r.sigma;
    gauges(gauge, [{ label: 'Lowest Dst in the next three hours', unit: 'nT', value: r.minimum, lo, hi, truth,
      truthLabel: 'Observed lowest', digits: 0, range: [Math.min(-80, lo - 15, truth ?? 0), Math.max(20, hi + 10, truth ?? 0)] }]);
    tbl.replaceChildren(...([['Tuned two-state physics', r.classical], ['Published O’Brien–McPherron', r.published],
      ['Persistence (last Dst)', r.persistence]] as const).map(([n, v]) => h('tr', {}, h('td', {}, n), h('td', {}, v.toFixed(0) + ' nT'))));
    const scale = Math.max(60, Math.abs(r.fast) + Math.abs(r.slow));
    resv.replaceChildren(
      h('span', {}, 'fast'), h('span', {}, h('b', { style: `width:${Math.min(100, 100 * Math.max(0, -r.fast) / scale)}%` })), h('span', {}, r.fast.toFixed(0) + ' nT'),
      h('span', {}, 'slow'), h('span', {}, h('b', { style: `width:${Math.min(100, 100 * Math.max(0, -r.slow) / scale)}%;opacity:.5` })), h('span', {}, r.slow.toFixed(0) + ' nT'));
    note.textContent = modified
      ? 'What-if wind: the network, the physics and the bar are recomputed; the observed future belongs to the recorded storm, so it is hidden.'
      : 'Fast current (hours) drains first; the slow current (about a day) sets the long tail of recovery. Drag on the chart or press play.';
    plot.redraw();
    ui.stamp(modified ? ms : replayMs, modified ? 'what-if forecast' : `all ${rp.length} hourly forecasts of this storm, each with three physics competitors`);
  }

  setK(Math.max(rp[0].k, rp.reduce((a, b) => (b.truth < a.truth ? b : a)).k - 6));
  scoreboard();
}
