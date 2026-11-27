import type { InstrumentContext } from '../types.ts';
import { canvas, fmt, gauges, gridLines, h, label, picker, scheduler, shell, slider, SCOPE, type Canvas2D } from '../kit.ts';
import { fetchPack } from '../../lib/nn/pack.ts';
import { reading, type Example, type WeightsMeta } from './net.ts';
import { BANDS, FRAMES, preprocess, RATE, type Features } from './pipeline.ts';

export async function mount(ctx: InstrumentContext) {
  const [weights, examples] = await Promise.all([
    fetchPack<WeightsMeta>(ctx.base + 'weights'),
    fetchPack<Example[]>(ctx.base + 'examples'),
  ]);
  const ui = shell(ctx.root, {
    title: 'RoomSense',
    subtitle: 'Reverberation from one clap',
    params: `${weights.meta.parameters.toLocaleString()} parameters`,
  });
  const note = h('p', { class: 'ig-note', role: 'status' });
  const comparison = h('p', { class: 'ig-note' });
  ui.readout.append(comparison, note);
  const player = h('audio', { controls: '', preload: 'none', 'aria-label': 'Play full recording', style: 'max-width:100%;width:100%' });
  const playerWrap = h('div', { style: 'grid-column:1/-1;min-width:0' }, player);
  ui.controls.append(playerWrap);
  const colors = [ctx.accent, '#e8b675', '#c7a2df'];
  let signal = examples.t.wave_0.data;
  let rate = RATE;
  let current: Example | null = examples.meta[0];
  let display: Features | null = null;
  let clipUrl: string | null = null;
  let operation = 0;
  let duration = 2.86;
  const legend = h('div', { style: 'display:flex;flex-wrap:wrap;gap:12px;padding:0 12px 8px;font:11px ui-monospace,monospace;color:#ebe6d8' },
    ...BANDS.map((band, i) => h('span', {}, h('span', { style: `color:${colors[i]}` }, '● '), `${band} Hz`)));
  const plot = canvas(ui.screen, ctx.compact ? 1.85 : 2.25, draw);
  ui.screen.append(legend);
  ui.screen.querySelector('canvas')?.setAttribute('aria-label', 'Energy decay after the clap at 500, 1000 and 2000 Hz');

  const control = slider(ui.controls, {
    label: 'Recording length used', min: 0.7, max: 2.86, step: 0.01, value: duration,
    fmt: v => `${v.toFixed(2)} s`,
    hint: 'Shorten the clip to see how much decay the estimator needs.',
    onInput: value => { duration = value; update(); },
  });
  function setAudio(url: string) {
    if (clipUrl) URL.revokeObjectURL(clipUrl);
    clipUrl = url.startsWith('blob:') ? url : null;
    player.src = url;
  }
  const items = examples.meta.map((row, index) => ({ label: row.label, sub: 'Measured room response', value: index }));
  const picks = picker(ui.picker, items, index => {
    operation++;
    current = examples.meta[index];
    signal = examples.t[`wave_${index}`].data;
    rate = RATE;
    duration = 2.86;
    control.set(duration);
    control.input.disabled = false;
    setAudio(ctx.base + current.filename);
    update();
  });
  setAudio(ctx.base + examples.meta[0].filename);
  if (!ctx.compact) {
    const upload = h('input', { type: 'file', accept: 'audio/*,.wav', 'aria-label': 'Load your mono clap recording' });
    ui.controls.append(h('label', { style: 'grid-column:1/-1' }, h('span', {}, 'Load a mono clap recording'), upload));
    upload.addEventListener('change', async () => {
      const file = upload.files?.[0];
      if (!file) return;
      const version = ++operation;
      try {
        if (file.size > 30 * 1024 * 1024) throw new Error('Choose a recording under 30 MB.');
        const audio = new AudioContext();
        let decoded: AudioBuffer;
        try { decoded = await audio.decodeAudioData(await file.arrayBuffer()); }
        finally { await audio.close(); }
        if (version !== operation) return;
        if (decoded.numberOfChannels !== 1) throw new Error('RoomSense takes a mono recording.');
        current = null;
        picks.set(-1);
        signal = decoded.getChannelData(0);
        rate = decoded.sampleRate;
        control.input.disabled = true;
        setAudio(URL.createObjectURL(file));
        update();
      } catch (error) {
        if (version === operation) note.textContent = error instanceof Error ? error.message : 'Could not read this audio file.';
      } finally {
        upload.value = '';
      }
    });
  }

  function draw(c: Canvas2D) {
    const { ctx: pen, w, h: height } = c;
    pen.fillStyle = SCOPE.bg;
    pen.fillRect(0, 0, w, height);
    const x0 = 38, y0 = 18, width = w - 50, plotHeight = height - 40;
    gridLines(c, x0, y0, width, plotHeight, 5, 4);
    for (let i = 0; i <= 4; i++) label(c, String(i * -20), x0 - 5, y0 + i * plotHeight / 4, SCOPE.text, 'right');
    for (let i = 0; i <= 5; i++) label(c, `${(i * 0.5).toFixed(1)}`, x0 + width * i / 5, y0 + plotHeight + 13,
      SCOPE.text, i === 5 ? 'right' : 'center');
    label(c, 'dB', 4, 8);
    if (!display) return;
    const f = display;
    BANDS.forEach((_, band) => {
      const curve = f.curves.subarray((band + 3) * FRAMES, (band + 4) * FRAMES);
      pen.beginPath();
      pen.strokeStyle = colors[band];
      pen.lineWidth = 1.5;
      for (let i = 0; i < 250; i++) {
        const x = x0 + width * i / 250, y = y0 - plotHeight * curve[i];
        if (i) pen.lineTo(x, y); else pen.moveTo(x, y);
      }
      pen.stroke();
      const classical = f.bandSeconds[band];
      const first = curve.findIndex(v => v <= -5 / 80);
      if (classical === null || first < 0) return;
      pen.beginPath();
      pen.setLineDash([3, 3]);
      pen.moveTo(x0 + width * first / 250, y0 + 5 * plotHeight / 80);
      pen.lineTo(Math.min(x0 + width, x0 + width * (first / 250 + (classical / 3) / 2.5)), y0 + 25 * plotHeight / 80);
      pen.stroke();
      pen.setLineDash([]);
    });
  }

  const update = scheduler(() => {
    const start = performance.now();
    try {
      const samples = current ? signal.subarray(0, Math.round(duration * rate)) : signal;
      const features = preprocess(samples, rate);
      display = features;
      const [rt, volume] = reading(weights, features);
      gauges(ui.readout, [
        { label: 'Reverberation time', unit: 's', ...rt, truth: current?.rt60_s, range: [0, 2.5] },
        { label: 'Volume · experimental', unit: 'm³', ...volume, truth: current?.volume_m3, range: [5, 1500], log: true, digits: 0 },
      ]);
      comparison.textContent = features.classicalSeconds === null
        ? 'Classical fit needs more measurable decay.'
        : `Classical decay fit ${fmt(features.classicalSeconds)} s`;
      note.textContent = current
        ? 'Synthetic clap through a measured room response. Dotted lines show the classical decay slope.'
        : 'Audio processed locally. Accuracy on recorded claps is still being evaluated.';
      ui.stamp(performance.now() - start);
    } catch (error) {
      display = null;
      gauges(ui.readout, []);
      comparison.textContent = '';
      note.textContent = error instanceof Error ? error.message : 'The recording could not be analysed.';
      ui.setStatus('');
    }
    plot.redraw();
  }, 25);
  update();
}

