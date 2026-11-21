// FireCast fire map: a real exam fire-day, the burned area at the afternoon satellite pass, and the network's
// probability that each 90 m cell has burned by the next afternoon pass, against what really burned. Change
// TOMORROW'S WEATHER (wind speed and direction, minimum humidity, 100-h fuel moisture) or take away inputs (the VIIRS
// hot spots, yesterday's perimeter) and the 22 input maps are rebuilt and the U-Net reruns in the browser on the
// map window (up to 240 x 240 cells, 21.6 km), as the published page's lab did.
//
// Drawing: the map zooms to the fire (burned at issue, what really burned next, cells with P >= 0.05, plus 1.5 km).
// Probability is a field, so it is shaded with linear interpolation between cell centres and contoured at
// P = 0.1, 0.3, 0.5 (marching squares). Burned-at-issue, truth, fire line and hot spots are cells and stay cells.
import { fetchPack, type Pack } from '../../lib/nn/pack.ts';
import type { InstrumentContext } from '../types.ts';
import { shell, picker, slider, button, canvas, h, scheduler, fmt, hex, SCOPE, label, type Canvas2D } from '../kit.ts';
import {
  calibrate, day, forward, loadNet, prepare, score, RES,
  type Day, type DaysMeta, type Inputs, type Ops, type Scores, type WeightsMeta,
} from './net.ts';

const STRATUM: Record<string, string> = { small: 'small fire', medium: 'medium fire', large: 'large fire' };
const PERIOD: Record<string, string> = { early: 'early exam 2012-16', main: 'main exam 2017-21', recent: 'recent exam 2022-24' };

const LEVELS = [0.1, 0.3, 0.5];                  // contour levels
const LINE = [{ w: 1, a: 0.7 }, { w: 1.5, a: 0.85 }, { w: 2, a: 1 }];   // css px width, alpha per level
const P_EXT = 0.05;                              // cells at or above this P set the fitted view
const MARGIN = Math.round(1500 / RES);           // 1.5 km of cells around the fire
const BURNED: [number, number, number] = [125, 127, 120];
const BURNED_EDGE = '#c4c6bd';
const FIRELINE: [number, number, number] = [255, 212, 59];
const HOTSPOT: [number, number, number] = [255, 120, 40];

export async function mount(ctx: InstrumentContext) {
  const [w, d] = await Promise.all([fetchPack<WeightsMeta>(ctx.base + 'weights'), fetchPack<DaysMeta>(ctx.base + 'days')]);
  show(ctx, w, d);
}

interface Layers { prob: boolean; yes: boolean; truth: boolean; pix: boolean }
interface Box { y0: number; x0: number; y1: number; x1: number }   // cells, end exclusive

// ---- colour ramp: transparent at P = 0, the accent by 0.3, a light accent at P >= 0.5
type RGB = [number, number, number];
const mix = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
const rgba = (c: RGB, a = 1) => `rgba(${c.map(Math.round).join(',')},${a})`;

function ramp(acc: RGB) {
  const hi = mix(acc, [255, 227, 214], 0.5);
  // t = P / 0.5: the accent fades in up to P = 0.2, then lightens toward a pale accent at P >= 0.5
  const at = (p: number): [RGB, number] => {
    const t = Math.max(0, Math.min(1, p / 0.5));
    if (t <= 0.4) return [acc, 0.9 * (t / 0.4)];
    const u = (t - 0.4) / 0.6;
    return [mix(acc, hi, u), 0.9 + 0.1 * u];
  };
  const css = `linear-gradient(90deg, ${[0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5].map((p) => { const [c, a] = at(p); return `${rgba(c, +a.toFixed(3))} ${p * 200}%`; }).join(', ')})`;
  return { at, css, line: mix(acc, [255, 255, 255], 0.55) as RGB };
}

// ---- marching squares on cell centres (cell (y, x) sits at (x + 0.5, y + 0.5)); flat [x0, y0, x1, y1, ...] in cells
function contour(F: Float32Array, hh: number, ww: number, L: number): Float32Array {
  const out: number[] = [];
  const seg = (ax: number, ay: number, bx: number, by: number) => out.push(ax + 0.5, ay + 0.5, bx + 0.5, by + 0.5);
  for (let y = 0; y + 1 < hh; y++) for (let x = 0; x + 1 < ww; x++) {
    const a = F[y * ww + x], b = F[y * ww + x + 1], c = F[(y + 1) * ww + x + 1], d = F[(y + 1) * ww + x];
    const k = (a >= L ? 8 : 0) | (b >= L ? 4 : 0) | (c >= L ? 2 : 0) | (d >= L ? 1 : 0);
    if (k === 0 || k === 15) continue;
    const T = (): [number, number] => [x + (L - a) / (b - a), y];
    const R = (): [number, number] => [x + 1, y + (L - b) / (c - b)];
    const B = (): [number, number] => [x + (L - d) / (c - d), y + 1];
    const Lf = (): [number, number] => [x, y + (L - a) / (d - a)];
    const s = (p: [number, number], q: [number, number]) => seg(p[0], p[1], q[0], q[1]);
    switch (k) {
      case 1: case 14: s(Lf(), B()); break;
      case 2: case 13: s(B(), R()); break;
      case 3: case 12: s(Lf(), R()); break;
      case 4: case 11: s(T(), R()); break;
      case 6: case 9: s(T(), B()); break;
      case 7: case 8: s(Lf(), T()); break;
      case 5: case 10: {                                   // saddle: the centre value decides which corners join
        const hiC = (a + b + c + d) / 4 >= L;
        if ((k === 5) === hiC) { s(Lf(), T()); s(B(), R()); } else { s(T(), R()); s(Lf(), B()); }
        break;
      }
    }
  }
  return Float32Array.from(out);
}

export function show(ctx: InstrumentContext, wp: Pack<WeightsMeta>, dp: Pack<DaysMeta>) {
  const net = loadNet(wp);
  const list = dp.meta.days;
  const ui = shell(ctx.root, { title: 'FireCast', subtitle: 'a real exam fire-day · where it will have burned by the next afternoon pass',
    params: 'U-Net, 374k weights, float16' });
  ctx.root.classList.add('fcast-inst');
  ctx.root.style.setProperty('--c', ctx.accent);
  const acc = hex(ctx.accent) as RGB;
  const R = ramp(acc);
  ctx.root.append(h('style', {}, `
    .fcast-inst .fx-row { display: flex; flex-wrap: wrap; gap: 6px; }
    .fcast-inst .inst-btn[aria-pressed="true"] { border-color: var(--c); color: var(--scr-ink); background: rgb(235 230 216 / .06); }
    .fcast-inst .inst-btn:focus-visible, .fx-ext button:focus-visible { outline: 2px solid var(--c); outline-offset: 2px; }
    .fx-key { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 14px; padding: 7px 10px; font: 10.5px/1.5 var(--mono, monospace); color: var(--scr-2ink); }
    .fx-key span { display: inline-flex; align-items: center; gap: 5px; }
    .fx-key i { display: inline-block; width: 11px; height: 11px; background: var(--k); box-shadow: inset 0 0 0 1.5px var(--b, transparent); }
    .fx-key svg { display: block; }
    .fx-ramp { position: relative; display: inline-block; width: 136px; height: 22px; margin: 0 6px 0 2px; }
    .fx-ramp b { position: absolute; left: 0; right: 0; top: 1px; height: 9px; background: var(--g), #0d0f0c; outline: 1px solid var(--scr-rule); }
    .fx-ramp s { position: absolute; top: 0; height: 11px; border-left: 1.5px solid var(--l); margin-left: -.75px; }
    .fx-ramp em { position: absolute; top: 12px; transform: translateX(var(--tx, -50%)); font: 9px/1 var(--mono, monospace); font-style: normal; color: var(--scr-3ink); white-space: nowrap; }
    .fx-map { position: relative; }
    .fx-ext { position: absolute; left: 8px; top: 8px; display: flex; z-index: 1; }
    .fx-ext button { padding: 4px 8px; font: 9.5px/1.4 var(--mono, monospace); letter-spacing: .06em; text-transform: uppercase; cursor: pointer;
      color: var(--scr-2ink); background: rgb(13 15 12 / .82); border: 1px solid var(--scr-3ink); }
    .fx-ext button + button { border-left: 0; }
    .fx-ext button:first-child { border-radius: 3px 0 0 3px; } .fx-ext button:last-child { border-radius: 0 3px 3px 0; }
    .fx-ext button:hover { color: var(--scr-ink); }
    .fx-ext button[aria-pressed="true"] { color: var(--scr-ink); border-color: var(--c); background: rgb(13 15 12 / .92); box-shadow: inset 0 -2px 0 var(--c); }
    .fx-h { font: 10px/1.4 var(--mono, monospace); text-transform: uppercase; letter-spacing: .06em; color: var(--scr-2ink); margin: 0 0 3px; }
    .fx-tbl { width: 100%; border-collapse: collapse; font: 10.5px/1.55 var(--mono, monospace); color: var(--scr-2ink); }
    .fx-tbl th { font-weight: 400; text-align: right; color: var(--scr-3ink); font-size: 9.5px; }
    .fx-tbl td { text-align: right; padding: 1px 0 1px 8px; } .fx-tbl td:first-child, .fx-tbl th:first-child { text-align: left; padding-left: 0; }
    .fx-tbl tr.me td { color: var(--c); } .fx-tbl tr.truth td { color: var(--scr-ink); }
    .fx-note { font-size: 12px; line-height: 1.5; color: var(--scr-2ink); margin: 0; }
    .fx-sub { font: 10px/1.45 var(--mono, monospace); color: var(--scr-3ink); margin: 2px 0 0; }
  `));

  // ---- state
  let dy: Day = day(dp, 0);
  let inp: Inputs | null = null, p: Float32Array | null = null, sc: Scores | null = null, rec: Scores | null = null;
  const lay: Layers = { prob: true, yes: false, truth: true, pix: false };
  const ops: Required<Pick<Ops, 'viirs_off' | 'prev_off'>> & { d1: NonNullable<Ops['d1']> } = { viirs_off: false, prev_off: false, d1: {} };
  const wx = { vs: 0, th: 0, rmin: 0, fm100: 0 };
  let fit = true;
  let ext: Box | null = null;             // the fire's extent, grown over this day's runs (reset with the weather)
  let probCv: HTMLCanvasElement | null = null, baseCv: HTMLCanvasElement | null = null;
  let iso: Float32Array[] = [];

  // ---- screen
  const key = h('div', { class: 'fx-key' });
  ui.screen.append(key);
  const mapBox = h('div', { class: 'fx-map' });
  ui.screen.append(mapBox);
  const cv = canvas(mapBox, 1, draw);
  const extG = h('div', { class: 'fx-ext', role: 'group', 'aria-label': 'Map extent' });
  const fitB = h('button', { type: 'button', title: 'Zoom to the fire and its forecast, plus 1.5 km' }, 'Fit fire');
  const fullB = h('button', { type: 'button', title: 'The whole map window the network ran on' }, 'Full window');
  extG.append(fitB, fullB);
  mapBox.append(extG);
  const paintExt = () => { fitB.setAttribute('aria-pressed', String(fit)); fullB.setAttribute('aria-pressed', String(!fit)); };
  fitB.addEventListener('click', () => { fit = true; paintExt(); cv.redraw(); });
  fullB.addEventListener('click', () => { fit = false; paintExt(); cv.redraw(); });
  extG.addEventListener('keydown', (ev) => {        // arrow keys move between the two, as in a segmented control
    if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;
    ev.preventDefault();
    const b = document.activeElement === fitB ? fullB : fitB;
    b.focus(); b.click();
  });
  paintExt();

  function paintKey() {
    const sw = (k: string, b = '') => `<i style="--k:${k}${b ? `;--b:${b}` : ''}"></i>`;
    const ln = (i: number) => `<svg width="22" height="8" aria-hidden="true"><line x1="2" y1="4" x2="20" y2="4" stroke="rgba(0,0,0,.6)" stroke-width="${LINE[i].w + 2}" stroke-linecap="round"/>`
      + `<line x1="2" y1="4" x2="20" y2="4" stroke="${rgba(R.line, LINE[i].a)}" stroke-width="${LINE[i].w}" stroke-linecap="round"/></svg>`;
    const parts: string[] = [];
    if (lay.yes) parts.push(`<span>${sw(rgba(acc, 0.85))}forecast yes, P ≥ ${net.thr.toFixed(2)}</span>`);
    else {
      parts.push(`<span><span class="fx-ramp" style="--g:${R.css};--l:${rgba(R.line)}" aria-hidden="true"><b></b>`
        + `${LEVELS.map((v) => `<s style="left:${v * 200}%"></s>`).join('')}`
        + `<em style="left:0;--tx:0">0</em>${LEVELS.map((v) => `<em style="left:${v * 200}%${v === 0.5 ? ';--tx:-100%' : ''}">${v === 0.5 ? '≥ ' : ''}${v.toFixed(1)}</em>`).join('')}</span>`
        + `P(burned by tomorrow PM)</span>`);
      parts.push(`<span>contours ${LEVELS.map((v, i) => `${ln(i)}${v.toFixed(1)}`).join(' ')}</span>`);
    }
    parts.push(`<span>${sw(rgba(BURNED), BURNED_EDGE)}burned at issue</span>`);
    if (lay.truth) parts.push(`<span>${sw('transparent', SCOPE.ink)}really burned by next pass</span>`);
    parts.push(`<span>${sw(rgba(FIRELINE))}fire line</span>`);
    if (lay.pix) parts.push(`<span>${sw(rgba(HOTSPOT))}VIIRS hot spots</span>`);
    key.innerHTML = parts.join('');
  }
  paintKey();
  const relayer = () => { paintKey(); cv.redraw(); };

  // ---- controls
  const wsS = slider(ui.controls, { label: "Tomorrow's wind speed", min: 0, max: 15, step: 0.5, value: 0, fmt: (v) => `${v.toFixed(1)} m/s`,
    hint: 'gridMET daily mean, 10 m', onInput: (v) => { wx.vs = v; ops.d1.vs = v; ops.d1.th = wx.th; go(); } });
  const wdS = slider(ui.controls, { label: 'Wind from', min: 0, max: 355, step: 5, value: 0, fmt: (v) => `${v.toFixed(0)}°`,
    hint: 'degrees true; the fire spreads the other way', onInput: (v) => { wx.th = v; ops.d1.th = v; ops.d1.vs = wx.vs; go(); } });
  const rhS = ctx.compact ? null : slider(ui.controls, { label: 'Minimum humidity', min: 3, max: 80, step: 1, value: 20, fmt: (v) => `${v.toFixed(0)} %`,
    onInput: (v) => { wx.rmin = v; ops.d1.rmin = v; go(); } });
  const fmS = ctx.compact ? null : slider(ui.controls, { label: '100-h fuel moisture', min: 3, max: 30, step: 0.5, value: 9, fmt: (v) => `${v.toFixed(1)} %`,
    hint: 'dead fuels 2.5-7.5 cm thick', onInput: (v) => { wx.fm100 = v; ops.d1.fm100 = v; go(); } });
  const tog = h('div', { class: 'fx-row' });
  ui.controls.append(tog);
  const pixB = toggle(tog, 'VIIRS hot spots', () => !ops.viirs_off, () => { ops.viirs_off = !ops.viirs_off; go(); });
  const prevB = toggle(tog, "Yesterday's perimeter", () => !ops.prev_off, () => { ops.prev_off = !ops.prev_off; go(); });
  button(tog, 'Recorded weather', () => { reset(); go(); });
  const view = h('div', { class: 'fx-row' });
  if (!ctx.compact) ui.controls.append(view);
  toggle(view, 'Yes/no at 0.10', () => lay.yes, () => { lay.yes = !lay.yes; relayer(); });
  toggle(view, 'Show hot spots', () => lay.pix, () => { lay.pix = !lay.pix; relayer(); });
  toggle(view, 'Show truth', () => lay.truth, () => { lay.truth = !lay.truth; relayer(); });

  function toggle(el: HTMLElement, text: string, on: () => boolean, flip: () => void) {
    const b = button(el, text, () => { flip(); paint(); });
    const paint = () => b.setAttribute('aria-pressed', String(on()));
    paint();
    return { paint, el: b };
  }

  function reset() {
    const d1 = dy.info.d1;
    wx.vs = Math.round(d1.vs * 2) / 2; wx.th = (Math.round(d1.th / 5) * 5) % 360; wx.rmin = Math.round(d1.rmin); wx.fm100 = Math.round(d1.fm100 * 2) / 2;
    wsS.set(wx.vs); wdS.set(wx.th); rhS?.set(wx.rmin); fmS?.set(wx.fm100);
    ops.d1 = {}; ops.viirs_off = false; ops.prev_off = false;
    pixB.paint(); prevB.paint();
    prevB.el.toggleAttribute('disabled', !dy.info.has_prev);
    ext = null;
  }

  // ---- run
  const go = scheduler(async () => {
    ui.setBusy(true);
    await new Promise((r) => setTimeout(r, 0));
    const t0 = performance.now();
    const x = prepare(dy, ops);
    const prob = calibrate(net, forward(net, x.X, x.s));
    const ms = performance.now() - t0;
    inp = x; p = prob;
    sc = score(x.now, dy.nxt, x.inwin, prob, net.thr);
    const recorded = !ops.viirs_off && !ops.prev_off && Object.keys(ops.d1).length === 0;
    if (recorded) rec = sc;
    build(x, prob);
    ui.setBusy(false);
    cv.redraw();
    readout(sc, prob, recorded);
    const [, , hh, ww] = dy.info.window;
    ui.stamp(ms, `${hh} × ${ww} cells (${(hh * 0.09).toFixed(1)} × ${(ww * 0.09).toFixed(1)} km)`);
  }, 120);

  function readout(sc: Scores, p: Float32Array, recorded: boolean) {
    const I = dy.info, e = I.exam;
    const r = (v: number, d = 3) => fmt(v, d);
    const area = (v: number) => (v < 10 ? v.toFixed(2) : v.toFixed(1));
    ui.readout.innerHTML = '';
    const head = h('div');
    head.innerHTML = `<div class="fx-h">${I.name} · ${I.date} · fire-day ${I.k}</div>
      <p class="fx-sub">${STRATUM[I.stratum] ?? I.stratum} · ${PERIOD[I.period] ?? I.period} · ${area(I.area_km2)} km² burned at issue${I.has_prev ? '' : ' · first mapped day'}</p>`;
    const tbl = h('table', { class: 'fx-tbl' });
    tbl.innerHTML = `<tr><th>on this map</th><th>AP</th><th>F1</th><th>new km²</th></tr>
      <tr class="me"><td>FireCast, now</td><td>${r(sc.ap)}</td><td>${r(sc.f1)}</td><td>${area(sc.newPredKm2)}</td></tr>
      ${!recorded && rec ? `<tr><td>FireCast, recorded</td><td>${r(rec.ap)}</td><td>${r(rec.f1)}</td><td>${area(rec.newPredKm2)}</td></tr>` : ''}
      <tr class="truth"><td>really burned</td><td></td><td></td><td>${area(sc.newObsKm2)}</td></tr>`;
    const ex = h('table', { class: 'fx-tbl' });
    ex.innerHTML = `<tr><th>exam run, full grid</th><th>AP</th><th>F1</th></tr>
      <tr><td>FireCast</td><td>${r(e.ap)}</td><td>${r(e.f1)}</td></tr>
      <tr><td>fire-line baseline</td><td>${r(e.fl_prev_ap)}</td><td>${r(e.fl_prev_f1)}</td></tr>`;
    const max = p.reduce((a, b) => Math.max(a, b), 0);
    const note = h('p', { class: 'fx-note' });
    note.innerHTML = `AP ranks the unburned cells within 10 km of the fire (1 = every cell that burned ranked first); F1 scores the yes/no map at the dev threshold ${net.thr.toFixed(2)}. `
      + `Highest probability on the map ${(max * 100).toFixed(0)} %: above about 35 % the forecasts are over-confident (on the exam a 90 % cell burned 61 % of the time). `
      + `Wind turns the spread through two inputs: the 'along / across the wind' offset maps and the weather scalars.`;
    ui.readout.append(head, tbl, ex, note);
  }

  // ---- layers, rebuilt after each run
  function build(x: Inputs, prob: Float32Array) {
    const { hh, ww } = x, n = hh * ww;
    // the shaded and contoured field: the network's P, and 1 on cells already burned at issue (they are burned)
    const F = new Float32Array(n);
    for (let i = 0; i < n; i++) F[i] = x.now[i] ? 1 : prob[i];
    iso = LEVELS.map((L) => contour(F, hh, ww, L));
    const base = new ImageData(ww, hh), pr = new ImageData(ww, hh);
    for (let i = 0; i < n; i++) {
      const k = i * 4, v = x.inwin[i] ? 13 : 8;
      base.data[k] = v; base.data[k + 1] = v + 2; base.data[k + 2] = v - 1; base.data[k + 3] = 255;
      const [c, a] = R.at(F[i]);
      pr.data[k] = c[0]; pr.data[k + 1] = c[1]; pr.data[k + 2] = c[2]; pr.data[k + 3] = Math.round(a * 255);
    }
    baseCv = toCanvas(base); probCv = toCanvas(pr);
    // the fitted view: burned at issue, what really burned next, and P >= 0.05; it only grows while the controls move
    let b: Box | null = ext ? { ...ext } : null;
    for (let y = 0; y < hh; y++) for (let xx = 0; xx < ww; xx++) {
      const i = y * ww + xx;
      if (!(x.now[i] || dy.nxt[i] || prob[i] >= P_EXT)) continue;
      if (!b) b = { y0: y, x0: xx, y1: y + 1, x1: xx + 1 };
      else { b.y0 = Math.min(b.y0, y); b.x0 = Math.min(b.x0, xx); b.y1 = Math.max(b.y1, y + 1); b.x1 = Math.max(b.x1, xx + 1); }
    }
    ext = b;
  }

  function toCanvas(img: ImageData) {
    const off = document.createElement('canvas');
    off.width = img.width; off.height = img.height;
    off.getContext('2d')!.putImageData(img, 0, 0);
    return off;
  }

  /** The cells on screen: the fitted square around the fire, or the whole window. */
  function viewBox(hh: number, ww: number): Box {
    if (!fit || !ext) return { y0: 0, x0: 0, y1: hh, x1: ww };
    const side = Math.max(ext.y1 - ext.y0, ext.x1 - ext.x0) + 2 * MARGIN;
    const vh = Math.min(hh, side), vw = Math.min(ww, side);
    const cy = (ext.y0 + ext.y1) / 2, cx = (ext.x0 + ext.x1) / 2;
    const y0 = Math.max(0, Math.min(hh - vh, Math.round(cy - vh / 2))), x0 = Math.max(0, Math.min(ww - vw, Math.round(cx - vw / 2)));
    return { y0, x0, y1: y0 + vh, x1: x0 + vw };
  }

  // ---- drawing
  function draw(c: Canvas2D) {
    const { ctx: g, w } = c;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, c.w, c.h);
    if (!p || !inp || !probCv || !baseCv) return;
    const inp_ = inp, p_ = p;
    const { hh, ww } = inp_;
    const v = viewBox(hh, ww), vh = v.y1 - v.y0, vw = v.x1 - v.x0;
    const s = Math.min(w / vw, c.h / vh), ox = (w - vw * s) / 2, oy = (c.h - vh * s) / 2;
    // everything on the map is drawn in device pixels, so cell edges land on whole pixels
    const dpr = c.cv.width / c.w;
    const X = (x: number) => Math.round((ox + (x - v.x0) * s) * dpr), Y = (y: number) => Math.round((oy + (y - v.y0) * s) * dpr);
    const fx = (x: number) => (ox + (x - v.x0) * s) * dpr, fy = (y: number) => (oy + (y - v.y0) * s) * dpr;
    g.save();
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.beginPath(); g.rect(X(v.x0), Y(v.y0), X(v.x1) - X(v.x0), Y(v.y1) - Y(v.y0)); g.clip();
    g.imageSmoothingEnabled = false;
    g.drawImage(baseCv, X(0), Y(0), X(ww) - X(0), Y(hh) - Y(0));
    const cells = (on: (i: number) => boolean, fill: string) => {     // row runs of cells, edges on device pixels
      g.fillStyle = fill;
      for (let y = v.y0; y < v.y1; y++) {
        const Y0 = Y(y), Y1 = Y(y + 1);
        for (let x = v.x0; x < v.x1; x++) {
          if (!on(y * ww + x)) continue;
          let e = x + 1;
          while (e < v.x1 && on(y * ww + e)) e++;
          g.fillRect(X(x), Y0, X(e) - X(x), Y1 - Y0);
          x = e;
        }
      }
    };
    if (lay.yes) cells((i) => !inp_.now[i] && p_[i] >= net.thr, rgba(acc, 0.85));
    else {
      g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'low';      // bilinear between cell centres
      g.drawImage(probCv, fx(0), fy(0), fx(ww) - fx(0), fy(hh) - fy(0));
    }
    cells((i) => inp_.now[i] === 1, rgba(BURNED));
    cells((i) => dy.fl[i] === 1, rgba(FIRELINE));
    if (lay.pix) cells((i) => inp_.P[i] > 0 || inp_.A[i] > 0, rgba(HOTSPOT));
    if (!lay.yes) {                       // contours, over a dark halo so they read on any shade
      g.lineCap = 'round'; g.lineJoin = 'round';
      for (let pass = 0; pass < 2; pass++) for (let li = 0; li < LEVELS.length; li++) {
        const sg = iso[li];
        if (!sg?.length) continue;
        g.lineWidth = (LINE[li].w + (pass ? 0 : 2)) * dpr;
        g.strokeStyle = pass ? rgba(R.line, LINE[li].a) : 'rgba(0,0,0,0.55)';
        g.beginPath();
        for (let k = 0; k < sg.length; k += 4) { g.moveTo(fx(sg[k]), fy(sg[k + 1])); g.lineTo(fx(sg[k + 2]), fy(sg[k + 3])); }
        g.stroke();
      }
    }
    const outline = (at: (y: number, x: number) => boolean, color: string, cssW: number, halo = 0) => {
      const path = new Path2D();
      for (let y = v.y0; y <= v.y1; y++) {          // horizontal edges, merged along each row
        let run = -1;
        for (let x = v.x0; x <= v.x1; x++) {
          const edge = x < v.x1 && at(y - 1, x) !== at(y, x);
          if (edge && run < 0) run = x;
          if (!edge && run >= 0) { path.moveTo(X(run), Y(y)); path.lineTo(X(x), Y(y)); run = -1; }
        }
      }
      for (let x = v.x0; x <= v.x1; x++) {          // vertical edges, merged down each column
        let run = -1;
        for (let y = v.y0; y <= v.y1; y++) {
          const edge = y < v.y1 && at(y, x - 1) !== at(y, x);
          if (edge && run < 0) run = y;
          if (!edge && run >= 0) { path.moveTo(X(x), Y(run)); path.lineTo(X(x), Y(y)); run = -1; }
        }
      }
      g.lineCap = 'square'; g.lineJoin = 'miter';
      const stroke = (lw: number, col: string) => {
        g.save();
        if (lw % 2) g.translate(0.5, 0.5);         // odd widths: centre the line on the pixel grid
        g.lineWidth = lw; g.strokeStyle = col; g.stroke(path);
        g.restore();
      };
      const lw = Math.max(1, Math.round(cssW * dpr));
      if (halo) stroke(lw + 2 * Math.max(1, Math.round(halo * dpr)), 'rgba(0,0,0,0.7)');
      stroke(lw, color);
    };
    const inWin = (y: number, x: number) => y >= 0 && x >= 0 && y < hh && x < ww;
    outline((y, x) => inWin(y, x) && inp_.now[y * ww + x] === 1, BURNED_EDGE, 1.5);
    if (lay.truth) outline((y, x) => inWin(y, x) && dy.nxt[y * ww + x] === 1, SCOPE.ink, 2, 1);
    g.restore();

    // wind arrow: where tomorrow's wind pushes the fire (grid frame: the features' spread-toward vector); north is up
    const mx0 = ox, my0 = oy, mx1 = ox + vw * s, my1 = oy + vh * s;
    const e = inp_.s[8], n = inp_.s[9], sp = Math.hypot(e, n);
    const cx = mx1 - 34, cy = my0 + 34;
    g.fillStyle = 'rgba(13,15,12,0.78)'; g.beginPath(); g.arc(cx, cy, 24, 0, 2 * Math.PI); g.fill();
    g.strokeStyle = SCOPE.faint; g.lineWidth = 1; g.beginPath(); g.arc(cx, cy, 22, 0, 2 * Math.PI); g.stroke();
    if (sp > 1e-6) {
      const L = 8 + Math.min(14, sp * 5 * 1.2), ux = e / sp, uy = -n / sp;
      g.strokeStyle = SCOPE.ink; g.fillStyle = SCOPE.ink; g.lineWidth = 2; g.lineCap = 'butt'; g.beginPath();
      g.moveTo(cx - ux * L, cy - uy * L); g.lineTo(cx + ux * L, cy + uy * L); g.stroke();
      g.beginPath(); g.moveTo(cx + ux * (L + 4), cy + uy * (L + 4));
      g.lineTo(cx + ux * (L - 4) - uy * 5, cy + uy * (L - 4) + ux * 5); g.lineTo(cx + ux * (L - 4) + uy * 5, cy + uy * (L - 4) - ux * 5); g.fill();
    }
    g.fillStyle = 'rgba(13,15,12,0.78)'; g.fillRect(cx - 28, cy + 26, 56, 13);
    label(c, `${(sp * 5).toFixed(1)} m/s`, cx, cy + 32.5, SCOPE.text, 'center', 9);
    // scale bar: the longest round length that fits in about a quarter of the map's width
    const pxPerM = s / RES;
    const len = [10000, 5000, 2000, 1000, 500, 250, 100].find((m) => m * pxPerM <= 0.28 * (mx1 - mx0)) ?? 100;
    const bw = len * pxPerM, bx = mx0 + 10, by = my1 - 10;
    const txt = len >= 1000 ? `${len / 1000} km` : `${len} m`;
    g.fillStyle = 'rgba(13,15,12,0.78)'; g.fillRect(bx - 5, by - 21, Math.max(bw, 40) + 10, 27);
    label(c, txt, bx, by - 11, SCOPE.text, 'left', 9);
    g.strokeStyle = SCOPE.ink; g.lineWidth = 2; g.lineCap = 'butt'; g.beginPath();
    g.moveTo(bx, by - 3); g.lineTo(bx, by); g.lineTo(bx + bw, by); g.lineTo(bx + bw, by - 3); g.stroke();
  }

  // ---- picker
  picker(ui.picker, list.map((d, i) => ({ label: d.name, sub: `${d.stratum} · ${d.date}`, value: i })), (i) => {
    dy = day(dp, i); rec = null; reset(); go();
  });
  reset();
  go();
}
