// FireCast fire map: a real exam fire-day, the burned area at the afternoon satellite pass, and the network's
// probability that each 90 m cell has burned by the next afternoon pass, against what really burned. Change
// TOMORROW'S WEATHER (wind speed and direction, minimum humidity, 100-h fuel moisture) or take away inputs (the VIIRS
// hot spots, yesterday's perimeter) and the 22 input maps are rebuilt and the U-Net reruns in the browser on the
// map window (up to 240 x 240 cells, 21.6 km), as the published page's lab did.
import { fetchPack, type Pack } from '../../lib/nn/pack.ts';
import type { InstrumentContext } from '../types.ts';
import { shell, picker, slider, button, canvas, h, scheduler, fmt, hex, SCOPE, label, type Canvas2D } from '../kit.ts';
import {
  calibrate, day, forward, loadNet, prepare, score,
  type Day, type DaysMeta, type Inputs, type Ops, type Scores, type WeightsMeta,
} from './net.ts';

const STRATUM: Record<string, string> = { small: 'small fire', medium: 'medium fire', large: 'large fire' };
const PERIOD: Record<string, string> = { early: 'early exam 2012-16', main: 'main exam 2017-21', recent: 'recent exam 2022-24' };

export async function mount(ctx: InstrumentContext) {
  const [w, d] = await Promise.all([fetchPack<WeightsMeta>(ctx.base + 'weights'), fetchPack<DaysMeta>(ctx.base + 'days')]);
  show(ctx, w, d);
}

interface Layers { prob: boolean; yes: boolean; truth: boolean; pix: boolean }

export function show(ctx: InstrumentContext, wp: Pack<WeightsMeta>, dp: Pack<DaysMeta>) {
  const net = loadNet(wp);
  const list = dp.meta.days;
  const ui = shell(ctx.root, { title: 'FireCast', subtitle: 'a real exam fire-day · where it will have burned by the next afternoon pass',
    params: 'U-Net, 374k weights, float16' });
  ctx.root.classList.add('fcast-inst');
  ctx.root.style.setProperty('--c', ctx.accent);
  ctx.root.append(h('style', {}, `
    .fcast-inst .fx-row { display: flex; flex-wrap: wrap; gap: 6px; }
    .fcast-inst .inst-btn[aria-pressed="true"] { border-color: var(--c); color: var(--scr-ink); background: rgb(235 230 216 / .06); }
    .fx-key { display: flex; flex-wrap: wrap; gap: 4px 14px; padding: 7px 10px; font: 10.5px/1.5 var(--mono, monospace); color: var(--scr-2ink); }
    .fx-key span { display: inline-flex; align-items: center; gap: 5px; }
    .fx-key i { display: inline-block; width: 11px; height: 11px; background: var(--k); border: 1px solid var(--b, transparent); }
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

  // ---- screen
  const key = h('div', { class: 'fx-key' });
  key.innerHTML = `<span><i style="--k:${ctx.accent}"></i>P(burned by tomorrow PM), full at 50 %</span><span><i style="--k:#7d7f78"></i>burned at issue</span>`
    + `<span><i style="--k:transparent;--b:#ebe6d8"></i>really burned next</span><span><i style="--k:#ffd43b"></i>fire line</span>`;
  ui.screen.append(key);
  const cv = canvas(ui.screen, 1, draw);

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
  toggle(view, 'Yes/no at 0.10', () => lay.yes, () => { lay.yes = !lay.yes; cv.redraw(); });
  toggle(view, 'Show hot spots', () => lay.pix, () => { lay.pix = !lay.pix; cv.redraw(); });
  toggle(view, 'Show truth', () => lay.truth, () => { lay.truth = !lay.truth; cv.redraw(); });

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

  // ---- drawing
  function draw(c: Canvas2D) {
    const { ctx: g, w } = c;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, c.w, c.h);
    if (!p || !inp) return;
    const inp_ = inp, p_ = p;
    const { hh, ww } = inp_;
    const s = Math.min(w / ww, c.h / hh), ox = (w - ww * s) / 2, oy = (c.h - hh * s) / 2;
    const img = g.createImageData(ww, hh);
    const [ar, ag, ab] = hex(ctx.accent);
    for (let i = 0; i < hh * ww; i++) {
      const k = i * 4;
      let R = 13, G = 15, B = 12;
      if (!inp_.inwin[i]) { R = 8; G = 9; B = 8; }
      if (inp_.now[i]) { R = 125; G = 127; B = 120; }
      else if (lay.prob || lay.yes) {
        const t = lay.yes ? (p_[i] >= net.thr ? 0.8 : 0) : Math.min(1, 2 * p_[i]) ** 0.7;   // full colour at 0.5
        R += (ar - R) * t; G += (ag - G) * t; B += (ab - B) * t;
        if (!lay.yes && p_[i] > 0.5) { const u = (p_[i] - 0.5) * 1.2; R += (255 - R) * u; G += (235 - G) * u; B += (200 - B) * u; }
      }
      if (dy.fl[i]) { R = 255; G = 212; B = 59; }
      if (lay.pix && (inp_.P[i] > 0 || inp_.A[i] > 0)) { R = 255; G = 120; B = 40; }
      img.data[k] = R; img.data[k + 1] = G; img.data[k + 2] = B; img.data[k + 3] = 255;
    }
    const off = document.createElement('canvas');
    off.width = ww; off.height = hh;
    off.getContext('2d')!.putImageData(img, 0, 0);
    g.imageSmoothingEnabled = false;
    g.drawImage(off, ox, oy, ww * s, hh * s);
    if (lay.truth) {                 // outline of what really burned by the next afternoon pass
      g.strokeStyle = SCOPE.ink; g.lineWidth = 1.2; g.beginPath();
      const at = (y: number, x: number) => y >= 0 && x >= 0 && y < hh && x < ww && dy.nxt[y * ww + x] === 1;
      for (let y = 0; y < hh; y++) for (let x = 0; x < ww; x++) {
        if (!at(y, x)) continue;
        const X0 = ox + x * s, Y0 = oy + y * s;
        if (!at(y - 1, x)) { g.moveTo(X0, Y0); g.lineTo(X0 + s, Y0); }
        if (!at(y + 1, x)) { g.moveTo(X0, Y0 + s); g.lineTo(X0 + s, Y0 + s); }
        if (!at(y, x - 1)) { g.moveTo(X0, Y0); g.lineTo(X0, Y0 + s); }
        if (!at(y, x + 1)) { g.moveTo(X0 + s, Y0); g.lineTo(X0 + s, Y0 + s); }
      }
      g.stroke();
    }
    // wind arrow: where tomorrow's wind pushes the fire (grid frame: the features' spread-toward vector)
    const e = inp_.s[8], n = inp_.s[9], sp = Math.hypot(e, n);
    const cx = ox + ww * s - 34, cy = oy + 34;
    g.strokeStyle = SCOPE.faint; g.lineWidth = 1; g.beginPath(); g.arc(cx, cy, 22, 0, 2 * Math.PI); g.stroke();
    if (sp > 1e-6) {
      const L = 8 + Math.min(14, sp * 5 * 1.2), ux = e / sp, uy = -n / sp;
      g.strokeStyle = SCOPE.ink; g.fillStyle = SCOPE.ink; g.lineWidth = 2; g.beginPath();
      g.moveTo(cx - ux * L, cy - uy * L); g.lineTo(cx + ux * L, cy + uy * L); g.stroke();
      g.beginPath(); g.moveTo(cx + ux * (L + 4), cy + uy * (L + 4));
      g.lineTo(cx + ux * (L - 4) - uy * 5, cy + uy * (L - 4) + ux * 5); g.lineTo(cx + ux * (L - 4) + uy * 5, cy + uy * (L - 4) - ux * 5); g.fill();
    }
    label(c, `${(sp * 5).toFixed(1)} m/s`, cx, cy + 32, SCOPE.text, 'center', 9);
    label(c, `1 km`, ox + 8, oy + hh * s - 18, SCOPE.text, 'left', 9);
    g.strokeStyle = SCOPE.text; g.lineWidth = 2; g.beginPath(); g.moveTo(ox + 8, oy + hh * s - 8); g.lineTo(ox + 8 + s * 1000 / 90, oy + hh * s - 8); g.stroke();
  }

  // ---- picker
  picker(ui.picker, list.map((d, i) => ({ label: d.name, sub: `${d.stratum} · ${d.date}`, value: i })), (i) => {
    dy = day(dp, i); rec = null; reset(); go();
  });
  reset();
  go();
}
