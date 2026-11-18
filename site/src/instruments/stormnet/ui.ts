import { fetchPack, type Pack } from '../../lib/nn/pack.ts';
import type { InstrumentContext } from '../types.ts';
import { shell, picker, slider, button, gauges, canvas, h } from '../kit.ts';
import { frameOf, predict, type Examples, type StormWeights } from './net.ts';
import { drawScope } from './scope.ts';

export async function mount(ctx: InstrumentContext) {
  const [weights, examples] = await Promise.all([
    fetchPack<StormWeights>(ctx.base + 'weights'), fetchPack<Examples>(ctx.base + 'examples')]);
  show(ctx, weights, examples);
}

export function show(ctx: InstrumentContext, weights: Pack<StormWeights>, examples: Pack<Examples>) {
  const records = examples.meta.records;
  const ui = shell(ctx.root, { title: 'StormNet', subtitle: 'Three-hour Dst forecast' });
  ctx.root.classList.add('sn-explorer');
  ctx.root.append(h('style', {}, `
    .sn-key, .sn-when, .sn-facts { color: var(--scr-2ink); font: 11px/1.5 var(--mono, monospace); }
    .sn-key { display: flex; flex-wrap: wrap; gap: 6px 16px; padding: 7px 10px; }
    .sn-key span::before {
      content: ''; display: inline-block; width: 13px; border-top: 2px solid currentColor;
      margin-right: 5px; vertical-align: middle;
    }
    .sn-key .sn-band::before { border-top: 5px solid var(--c); }
    .sn-when { margin: 0 0 5px; color: var(--scr-ink); }
    .sn-facts { display: grid; grid-template-columns: 1fr auto; gap: 4px 12px; margin: 8px 0; }
    .sn-facts dd { margin: 0; color: var(--scr-ink); text-align: right; }
    .sn-note { font-size: 12px; line-height: 1.55; color: var(--scr-2ink); margin: 5px 0; }
    @media (max-width: 540px) {
      .sn-explorer .inst-picker { display: grid; grid-template-columns: repeat(2, 1fr); margin-inline: 0; padding: 0; }
    }
  `));
  let caseId = 0, index = records[0].issues.length - 1, shift = 0;
  let frame = frameOf(examples, caseId, index, shift), reading = predict(weights, frame);
  const chartBox = h('div', { role: 'img', 'aria-label': 'Recorded Dst and forecast interval, with the measured solar wind field below' });
  const observedKey = h('span', {}, 'Recorded Dst');
  ui.screen.append(chartBox, h('div', { class: 'sn-key' }, observedKey, h('span', { class: 'sn-band' }, 'Forecast minimum ±68% band')));
  const plot = canvas(chartBox, ctx.compact ? 1.6 : 1.85, c => drawScope(c, frame, reading, ctx.accent, shift === 0));
  const when = h('p', { class: 'sn-when' });
  const gauge = h('div');
  const facts = h('dl', { class: 'sn-facts' });
  const note = h('p', { class: 'sn-note' });
  ui.readout.append(when, gauge, facts, note);
  const dateOf = (t: string) => t.slice(0, 10) + ' · ' + t.slice(11, 16) + ' UTC';
  const timeSlider = slider(ui.controls, {
    label: 'Forecast issue', min: 0, max: index, step: 1, value: index,
    fmt: v => records[caseId].issues[v]?.slice(11, 16) + ' UTC',
    onInput: v => { index = v; update(); }
  });
  const bzSlider = slider(ui.controls, {
    label: 'Shift Bz over the last 3 hours', min: -12, max: 12, step: 1, value: 0,
    fmt: v => v === 0 ? 'as recorded' : (v > 0 ? '+' : '−') + Math.abs(v) + ' nT',
    hint: 'Negative Bz points south and usually strengthens the ring current.',
    onInput: v => { shift = v; update(); }
  });
  button(ui.controls, 'Restore recording', () => { shift = 0; bzSlider.set(0); update(); });
  picker(ui.picker, records.map((r, i) => ({ label: r.label, sub: r.issues.at(-1)!.slice(0, 10), value: i })), i => {
    caseId = i; index = records[i].issues.length - 1; shift = 0;
    timeSlider.input.max = String(index); timeSlider.set(index); bzSlider.set(0); update();
  });
  function update() {
    const start = performance.now();
    frame = frameOf(examples, caseId, index, shift);
    reading = predict(weights, frame);
    const truth = shift === 0 ? Math.min(...frame.future) : null;
    const low = reading.minimum - reading.sigma, high = reading.minimum + reading.sigma;
    when.textContent = dateOf(records[caseId].issues[index]);
    gauges(gauge, [{ label: 'Next three-hour minimum', unit: 'nT', value: reading.minimum,
      lo: low, hi: high, truth, truthLabel: 'Observed minimum', digits: 1,
      range: [Math.min(-60, low - 5, truth ?? 0), Math.max(20, high + 5, truth ?? 0)] }]);
    facts.replaceChildren();
    for (const [name, val] of [['Last completed Dst', frame.dst], ['Tuned two-state physics', reading.classical]] as const) {
      facts.append(h('dt', {}, name), h('dd', {}, val.toFixed(1) + ' nT'));
    }
    note.textContent = shift === 0 ? 'The white trace continues into the verification period. The band estimates its lowest hourly value.'
      : 'Wind input changed. The observed future belongs to the original recording.';
    observedKey.textContent = shift === 0 ? 'Recorded Dst' : 'Dst through issue';
    plot.redraw(); ui.stamp(performance.now() - start, shift === 0 ? 'recorded wind' : 'modified wind');
  }
  update();
}
