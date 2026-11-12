// QuakeNet live instrument: four real exam records, the real network running in the browser,
// and three things to change — noise, record length, amplitude — to watch how the reading responds.
import type { InstrumentContext } from '../types.ts';
import { fetchPack, type Pack } from '../../lib/nn/pack.ts';
import { read, L, type QMeta, type QExample } from './net.ts';
import { shell, picker, slider, button, scheduler, gauges, compass, canvas, gridLines, trace, label, rng, SCOPE, type Canvas2D } from '../kit.ts';

// Approximate surface-focus P and S travel times (minutes) vs distance (degrees), for the arrival markers only.
const TT: [number, number, number][] = [[20, 4.6, 8.3], [30, 6.2, 11.2], [40, 7.6, 13.8], [50, 8.8, 16.0], [60, 10.0, 18.1], [70, 11.1, 20.0], [80, 12.1, 21.8], [90, 13.0, 23.4], [100, 13.7, 24.8]];
function sMinusP(deg: number) {
  const d = Math.max(20, Math.min(100, deg));
  const i = Math.min(TT.length - 2, Math.floor((d - 20) / 10));
  const [d0, p0, s0] = TT[i], [, p1, s1] = TT[i + 1], f = (d - d0) / 10;
  return { p: (p0 + f * (p1 - p0)) * 60, s: (s0 + f * (s1 - s0)) * 60 };
}
const SHORT: Record<string, string> = { us6000m0xl: 'Noto, Japan', us10008hu2: 'Atacama, Chile', usb000j0yl: 'Kermadec Is.', usp000jhzj: 'Vanuatu' };

export async function mount(ctx: InstrumentContext) {
  const [W, X] = await Promise.all([
    fetchPack<QMeta>(`${ctx.base}weights`),
    fetchPack<QExample[]>(`${ctx.base}examples`),
  ]);
  const ex = X.meta;
  const ui = shell(ctx.root, { title: 'QuakeNet', subtitle: 'one seismometer, one earthquake — a real record, never used in training', params: '1.44 M parameters · matches Python to 1e-6' });
  const st = { i: 1, noise: 0, cut: L, amp: 0, seed: 7 };
  let view: { x: Float32Array; dist: number } = { x: X.t.x1.data, dist: 60 };

  picker(ui.picker, ex.map((e, i) => ({ label: SHORT[e.id] ?? e.place, sub: `${e.station} · M${e.truth.mw} · ${Math.round(e.truth.depth)} km deep`, value: i })), (i) => { st.i = i; run(); }, st.i);

  const plot = canvas(ui.screen, ctx.compact ? 2.4 : 2.05, (c) => draw(c));
  slider(ui.controls, { label: 'Add noise', min: 0, max: 1.5, step: 0.05, value: 0, fmt: (v) => (v === 0 ? 'none' : `σ ${v.toFixed(2)}`), hint: 'Gaussian, in units of the record’s largest component', onInput: (v) => { st.noise = v; go(); } });
  slider(ui.controls, { label: 'Record length', min: 240, max: L, step: 20, value: L, fmt: (v) => `${Math.round((v - 60) / 60)} min after P`, hint: 'everything after this is zeroed', onInput: (v) => { st.cut = v; go(); } });
  if (!ctx.compact) {
    slider(ui.controls, { label: 'Amplitude at the station', min: -1, max: 1, step: 0.05, value: 0, fmt: (v) => (v === 0 ? 'as recorded' : `×${(10 ** v).toPrecision(2)}`), hint: 'the record’s size enters as one number', onInput: (v) => { st.amp = v; go(); } });
    button(ui.controls, 'New noise draw', () => { st.seed++; go(); });
  }

  function input(): Float32Array {
    const x = new Float32Array(X.t[`x${st.i}`].data);
    if (st.noise > 0) {
      const r = rng(st.seed * 101 + st.i);
      for (let k = 0; k < x.length; k++) x[k] += st.noise * r.n();
    }
    if (st.cut < L) for (let ch = 0; ch < 3; ch++) x.fill(0, ch * L + st.cut, (ch + 1) * L);
    return x;
  }

  async function run() {
    ui.setBusy(true);
    await new Promise((r) => requestAnimationFrame(r));
    const e = ex[st.i];
    const x = input();
    const t0 = performance.now();
    const { r } = read(W, { data: x, shape: [3, L] }, e.ls + st.amp, e.peak);
    const ms = performance.now() - t0;
    view = { x, dist: r.dist };
    plot.redraw();
    const lo = 10 ** (r.logz - r.logz_sig), hi = 10 ** (r.logz + r.logz_sig);
    gauges(ui.readout, [
      { label: 'Magnitude (Mw)', value: r.mw, lo: r.mw - r.mw_sig, hi: r.mw + r.mw_sig, truth: e.truth.mw, truthLabel: `USGS ${e.truth.magType}`, range: [5, 8.5] },
      { label: 'Distance', unit: '°', value: r.dist, lo: r.dist - r.dist_sig, hi: r.dist + r.dist_sig, truth: e.truth.dist, truthLabel: 'USGS', range: [20, 100], digits: 1 },
      { label: 'Depth', unit: 'km', value: r.depth, lo, hi, truth: e.truth.depth, truthLabel: 'USGS', range: [5, 800], digits: 0, log: true },
    ]);
    compass(ui.readout, { label: 'Direction to the quake', deg: r.baz, half: r.half, truth: e.truth.baz });
    ui.stamp(ms, `${e.place}, ${e.time} UTC at ${e.station}`);
    ui.setBusy(false);
  }
  const go = scheduler(run, 30);

  function draw(c: Canvas2D) {
    const { ctx: g, w, h } = c;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, w, h);
    const padL = 22, padR = 10, top = 14, bot = 22;
    const iw = w - padL - padR, ih = h - top - bot;
    gridLines(c, padL, top, iw, ih, 8, 3);
    const rowH = ih / 3;
    const xs = (t: number) => padL + (t / L) * iw;
    // arrival markers implied by the network's distance reading
    const pre = 60;
    const tt = sMinusP(view.dist);
    const sT = pre + (tt.s - tt.p);
    const kmP = view.dist * 111.19;
    const marks: [string, number][] = [['P', pre], ['S', sT], ['Love', pre + kmP / 4.4 - tt.p], ['Rayleigh', pre + kmP / 3.9 - tt.p]];
    for (const [name, t] of marks) {
      if (t < 0 || t > L) continue;
      const x = xs(t);
      g.strokeStyle = name === 'P' ? SCOPE.ink : ctx.accent;
      g.globalAlpha = name === 'P' ? 0.5 : 0.75;
      g.setLineDash(name === 'P' ? [] : [3, 3]);
      g.beginPath(); g.moveTo(x + 0.5, top); g.lineTo(x + 0.5, top + ih); g.stroke();
      g.setLineDash([]); g.globalAlpha = 1;
      label(c, name, x + 4, top + 6, name === 'P' ? SCOPE.text : ctx.accent);
    }
    // traces
    const names = ['Z', 'N', 'E'];
    for (let ch = 0; ch < 3; ch++) {
      const y0 = top + ch * rowH;
      trace(c, view.x, padL, y0, iw, rowH, ch === 0 ? ctx.accent : SCOPE.ink, 0.12, ch === 0 ? 1 : 0.75, ch * L, ch * L + L);
      label(c, names[ch], 6, y0 + rowH / 2, SCOPE.text);
    }
    // cut shading
    if (st.cut < L) {
      const x = xs(st.cut);
      g.fillStyle = 'rgba(0,0,0,0.55)';
      g.fillRect(x, top, padL + iw - x, ih);
      g.strokeStyle = ctx.accent; g.beginPath(); g.moveTo(x + 0.5, top); g.lineTo(x + 0.5, top + ih); g.stroke();
      label(c, 'cut', x + 6, top + ih - 8, ctx.accent);
    }
    for (let m = 0; m <= 40; m += 5) label(c, m === 0 ? 'P' : `${m}m`, xs(pre + m * 60), h - 9, SCOPE.faint, m === 40 ? 'right' : 'center', 9);
  }

  await run();
}
