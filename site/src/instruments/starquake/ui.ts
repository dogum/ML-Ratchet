// StarQuake live instrument: three real Kepler red giants from the exam. The browser runs the whole reading chain —
// high-pass, power spectrum, log spectrum and comb map (dsp.ts, a port of pipeline.py), then the network — on the
// exam's own pieces of each light curve. Controls: observing length, added photon noise, and "hear the star".
import type { InstrumentContext } from '../types.ts';
import { fetchPack } from '../../lib/nn/pack.ts';
import { forward, interpret, type SMeta, type SExamples, type Reading } from './net.ts';
import { prepare, dnuS, FC, CEN, NB, NC, NL, L0, makeRebin, type Prepared } from './dsp.ts';
import { starSound, player } from './sonify.ts';
import { shell, picker, slider, button, scheduler, gauges, probs, canvas, gridLines, label, heatmap, rng, hex, h, SCOPE, type Canvas2D } from '../kit.ts';

// slider positions, shortest to longest; the value is the exam's piece key (0 = the whole light curve)
const LENS = [27, 54, 90, 365, 0];
const LEN_NAME: Record<number, string> = { 27: '27 days', 54: '54 days', 90: '90 days', 365: '1 year', 0: '4 years' };
// the exam (420 giants, model.yaml): network vs SYD-style classical pipeline
const EXAM: Record<number, string> = {
  27: 'exam at 27 d: Δν within 4.6% (median) — the classical pipeline 26.8%',
  54: 'between 27 and 90 days the network beats the classical pipeline by a wide margin',
  90: 'exam at 90 d: νmax within 2.2% (median) — the classical pipeline 3.9%',
  365: 'exam at 1 yr: νmax within 1.7% — a tie with the classical pipeline (1.7%)',
  0: 'exam at 4 yr: νmax within 1.5% — the classical pipeline wins here (0.9%)',
};
const PIECE: Record<number, string> = { 27: '27-day', 54: '54-day', 90: '90-day', 365: '1-year' };
const NU = (s: string) => `<span style="text-transform:none">${s}</span>`;
const NAME: Record<string, string> = { '2695975': 'a luminous giant', '7353997': 'a red giant', '7618730': 'a red giant' };
const noiseOf = (v: number) => (v <= 0 ? 0 : 10 ** (1.5 + v));

export async function mount(ctx: InstrumentContext) {
  const [W, X] = await Promise.all([fetchPack<SMeta>(`${ctx.base}weights`), fetchPack<SExamples>(`${ctx.base}examples`)]);
  const stars = X.meta.stars;
  const ui = shell(ctx.root, { title: 'StarQuake', subtitle: 'a red giant’s light curve from Kepler — real stars from the exam, never used in training', params: '410 k parameters · matches Python to 1e-6' });
  const st = { i: 1, len: 4, noise: 0, seed: 3 };
  const narrow = ctx.root.getBoundingClientRect().width < 560;
  // the star's own point-to-point scatter, for the noise slider's label
  const scatter = stars.map((_, i) => {
    const y = X.t[`y${i}`].data, d: number[] = [];
    for (let k = 1; k < y.length; k += 3) if (Number.isFinite(y[k]) && Number.isFinite(y[k - 1])) d.push(Math.abs(y[k] - y[k - 1]));
    d.sort((a, b) => a - b);
    return (1.4826 * d[d.length >> 1]) / Math.SQRT2;
  });

  picker(ui.picker, stars.map((s, i) => ({ label: `KIC ${s.kic}`, sub: `${NAME[s.kic] ?? 'red giant'} · Kp ${s.truth.kp.toFixed(1)} · ${s.truth.R.toFixed(0)} R☉`, value: i })),
    (i) => { st.i = i; noiseSl.set(st.noise); stopSound(); go(); }, st.i);

  const cache = new Map<string, Prepared>();
  let view: { p: Prepared; r: Reading; start: number; n: number } | null = null;
  let playBtn: HTMLButtonElement | null = null;
  const pl = player();
  const LF0 = Math.log10(3), LF1 = Math.log10(283);

  const spec = canvas(ui.screen, narrow ? 1.55 : 2.25, (c) => drawSpectrum(c));
  const maps = canvas(ui.screen, narrow ? 0.95 : 2.4, (c) => drawMaps(c));
  const lenSl = slider(ui.controls, {
    label: 'Observing length', min: 0, max: 4, step: 1, value: st.len, fmt: (v) => LEN_NAME[LENS[Math.round(v)]],
    hint: EXAM[0], onInput: (v) => { st.len = Math.round(v); hintLen.textContent = EXAM[LENS[st.len]]; stopSound(); go(); },
  });
  const hintLen = lenSl.input.parentElement!.querySelector('small')!;
  const noiseSl = slider(ui.controls, {
    label: 'Add photon noise', min: 0, max: 2.5, step: 0.05, value: 0,
    fmt: (v) => { const s = noiseOf(v); return s === 0 ? 'none' : `σ ${s < 1000 ? Math.round(s) : (s / 1000).toFixed(1) + 'k'} ppm · ${(s / scatter[st.i]).toFixed(s < 2 * scatter[st.i] ? 1 : 0)}×`; },
    hint: 'white noise on every 29.4-min point, as for a fainter star (× = the star’s own scatter)', onInput: (v) => { st.noise = v; stopSound(); go(); },
  });
  if (!ctx.compact) {
    button(ui.controls, 'New noise draw', () => { st.seed++; if (st.noise > 0) go(); });
    const cell = h('div', { class: 'inst-slider' });
    ui.controls.append(cell);
    playBtn = button(cell, '♪ Hear the star', () => toggleSound());
    playBtn.style.alignSelf = 'flex-start';
    cell.append(h('small', {}, 'the oscillation hump sped up 10 million times: 1 μHz → 10 Hz'));
  }

  // ---- the reading chain ----
  function input(): { p: Prepared; start: number; n: number } {
    const s = stars[st.i], L = LENS[st.len], [start, n] = s.pieces[String(L)];
    const sigma = noiseOf(st.noise);
    const key = `${st.i}/${L}/${sigma ? `${st.noise}/${st.seed}` : 0}`;
    let p = cache.get(key);
    if (!p) {
      const y = Float64Array.from(X.t[`y${st.i}`].data.subarray(start, start + n));
      if (sigma > 0) {
        const r = rng(st.seed * 7919 + st.i * 101 + L);
        for (let k = 0; k < n; k++) if (Number.isFinite(y[k])) y[k] += sigma * r.n();
      }
      p = prepare(y);
      if (cache.size > 24) cache.clear();
      cache.set(key, p);
    }
    return { p, start, n };
  }

  async function run() {
    ui.setBusy(true);
    await new Promise((r) => requestAnimationFrame(r));
    const s = stars[st.i], t = s.truth;
    const t0 = performance.now();
    const { p, start, n } = input();
    const tPrep = performance.now() - t0;
    const t1 = performance.now();
    const raw = forward(W, p.F1, p.F2, p.s);
    const r = interpret(W, raw, t.teff, t.e_teff);
    const ms = performance.now() - t1;
    view = { p, r, start, n };
    spec.redraw();
    maps.redraw();
    ui.readout.innerHTML = '';
    probs(ui.readout, 'Solar-like oscillations', [{ label: 'detected', p: r.p_osc, truth: true }, { label: 'not detected', p: 1 - r.p_osc }]);
    const lg = (v: number, k: number): [number, number] => [v / k, v * k];
    gauges(ui.readout, [
      { label: `Hump peak ${NU('νmax')}`, unit: 'μHz', value: r.numax, lo: r.numax - r.numax_err, hi: r.numax + r.numax_err, truth: t.numax, truthLabel: 'Yu+ 2018', range: lg(t.numax, 1.3), digits: t.numax < 20 ? 2 : 1, log: true },
      { label: `Comb spacing ${NU('Δν')}`, unit: 'μHz', value: r.dnu, lo: r.dnu - r.dnu_err, hi: r.dnu + r.dnu_err, truth: t.dnu, truthLabel: 'Yu+ 2018', range: lg(t.dnu, 1.3), digits: t.dnu < 2 ? 3 : 2, log: true },
      { label: 'Mass', unit: 'M☉', value: r.mass, lo: r.mass - r.mass_err, hi: r.mass + r.mass_err, truth: t.M, truthLabel: 'Yu+ 2018', range: lg(t.M, 2.2), digits: 2, log: true },
      { label: 'Radius', unit: 'R☉', value: r.radius, lo: r.radius - r.radius_err, hi: r.radius + r.radius_err, truth: t.R, truthLabel: 'Yu+ 2018', range: lg(t.R, 1.7), digits: 1, log: true },
    ]);
    ui.readout.append(h('div', { class: 'ig-foot' }, h('span', {}, `mass and radius by the scaling relations, Teff ${t.teff} ± ${t.e_teff} K`)));
    const py = s.python[String(LENS[st.len])];
    if (st.noise === 0 && py) ui.readout.append(h('div', { class: 'ig-foot' }, h('span', {}, `numpy on the same piece: νmax ${py.numax.toFixed(2)}, Δν ${py.dnu.toFixed(3)}`)));
    const day0 = (start * X.meta.dt) / 86400;
    ui.stamp(ms, `pipeline ${Math.round(tPrep)} ms · KIC ${s.kic}, ${LENS[st.len] ? `the exam’s ${PIECE[LENS[st.len]]} piece from day ${Math.round(day0)}` : `${p.T.toFixed(0)} days of Kepler`} · ${(p.s[1] * 100).toFixed(0)}% duty cycle`);
    ui.setBusy(false);
  }
  const go = scheduler(run, 40);

  // ---- sound ----
  function stopSound() { pl.stop(); if (playBtn) playBtn.textContent = '♪ Hear the star'; }
  function toggleSound() {
    if (pl.playing()) { stopSound(); return; }
    if (!view) return;
    const x = starSound(view.p.f, view.p.P, view.r.numax, view.r.dnu, pl.rate, st.seed);
    if (playBtn) playBtn.textContent = '■ Stop';
    pl.play(x, () => { if (playBtn) playBtn.textContent = '♪ Hear the star'; });
  }

  // ---- drawing ----
  function drawSpectrum(c: Canvas2D) {
    const { ctx: g, w, h: H } = c;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, w, H);
    if (!view) return;
    const s = stars[st.i], t = s.truth, { p, r } = view;
    const padL = 34, padR = 10, top = 22, bot = 22, iw = w - padL - padR, ih = H - top - bot;
    const [y0, y1] = [Math.floor(s.f1range[0] - 0.3), Math.ceil(s.f1range[1] + 0.5)];
    const xs = (f: number) => padL + ((Math.log10(f) - LF0) / (LF1 - LF0)) * iw;
    const ys = (lp: number) => top + ih - ((lp - y0) / (y1 - y0)) * ih;
    gridLines(c, padL, top, iw, ih, 0, y1 - y0);
    for (const f of narrow ? [3, 10, 30, 100] : [3, 5, 10, 20, 50, 100, 200]) {
      const x = Math.round(xs(f)) + 0.5;
      g.strokeStyle = SCOPE.grid; g.beginPath(); g.moveTo(x, top); g.lineTo(x, top + ih); g.stroke();
      label(c, String(f), x, H - 11, SCOPE.faint, 'center', 9);
    }
    label(c, 'μHz', padL + iw, H - 11, SCOPE.faint, 'right', 9);
    const every = ih / (y1 - y0) < 28 ? 2 : 1;
    for (let d = y0; d <= y1; d++) if ((d - y0) % every === 0) label(c, `10${sup(d)}`, padL - 5, ys(d), SCOPE.faint, 'right', 9);
    // the native spectrum: min-max per pixel column (faint), the comb's texture
    const { f, P } = p, df = f[1] - f[0];
    g.strokeStyle = SCOPE.ink; g.globalAlpha = 0.16; g.lineWidth = 1;
    g.beginPath();
    for (let px = 0; px < iw; px++) {
      const fa = 10 ** (LF0 + (px / iw) * (LF1 - LF0)), fb = 10 ** (LF0 + ((px + 1) / iw) * (LF1 - LF0));
      const a = Math.max(0, Math.floor((fa - f[0]) / df)), b = Math.min(f.length - 1, Math.max(a, Math.ceil((fb - f[0]) / df)));
      let lo = Infinity, hi = -Infinity;
      for (let k = a; k <= b; k++) { if (P[k] < lo) lo = P[k]; if (P[k] > hi) hi = P[k]; }
      if (!(hi > 0)) continue;
      const ya = ys(Math.log10(Math.max(lo, 1e-6))), yb = ys(Math.log10(hi));
      g.moveTo(padL + px + 0.5, Math.min(top + ih, ya)); g.lineTo(padL + px + 0.5, Math.max(top, Math.min(yb, ya - 0.8)));
    }
    g.stroke(); g.globalAlpha = 1;
    // the network's numax and its range
    const xa = xs(Math.max(3, r.numax - r.numax_err)), xb = xs(Math.min(283, r.numax + r.numax_err)), xn = xs(r.numax);
    g.fillStyle = ctx.accent; g.globalAlpha = 0.16; g.fillRect(xa, top, Math.max(1.5, xb - xa), ih); g.globalAlpha = 1;
    // the 1-D branch's input: log power on 1024 log bins
    g.save(); g.beginPath(); g.rect(padL, top, iw, ih); g.clip();
    g.strokeStyle = ctx.accent; g.lineWidth = 1; g.globalAlpha = 0.75; g.beginPath();
    for (let i = 0; i < NB; i++) { const x = xs(FC[i]), y = ys(p.F1[i]); if (i) g.lineTo(x, y); else g.moveTo(x, y); }
    g.stroke(); g.globalAlpha = 1;
    // smoothed (96 log bins): the granulation background and the oscillation hump on it
    const NS = 96, se = Float64Array.from({ length: NS + 1 }, (_, i) => 3 * (283 / 3) ** (i / NS)), sm = makeRebin(f, P)(se);
    g.strokeStyle = SCOPE.ink; g.lineWidth = 2; g.lineJoin = 'round'; g.beginPath();
    for (let i = 0; i < NS; i++) { const x = xs(Math.sqrt(se[i] * se[i + 1])), y = ys(Math.log10(Math.max(sm[i], 1e-6))); if (i) g.lineTo(x, y); else g.moveTo(x, y); }
    g.stroke(); g.restore();
    label(c, 'log spectrum · the 1-D input', padL + iw - 4, top + 18, ctx.accent, 'right', 9);
    g.strokeStyle = ctx.accent; g.lineWidth = 1.2; g.beginPath(); g.moveTo(Math.round(xn) + 0.5, top); g.lineTo(Math.round(xn) + 0.5, top + ih); g.stroke();
    // its comb: numax + k dnu
    g.strokeStyle = ctx.accent; g.lineWidth = 1.2;
    for (let k = -6; k <= 6; k++) {
      const fk = r.numax + k * r.dnu;
      if (fk < 3 || fk > 283) continue;
      const x = Math.round(xs(fk)) + 0.5;
      g.globalAlpha = 0.85 - Math.abs(k) * 0.1;
      g.beginPath(); g.moveTo(x, top); g.lineTo(x, top + (k === 0 ? 9 : 5)); g.stroke();
    }
    g.globalAlpha = 1;
    const right = xn > padL + iw * 0.62;
    const txt = `network νmax ${r.numax.toFixed(1)} · comb Δν ${r.dnu.toFixed(2)}`;
    g.font = '10px "Martian Mono Variable", ui-monospace, monospace';
    const tw = g.measureText(txt).width, tx = Math.max(padL, Math.min(padL + iw - tw, right ? xn - 6 - tw : xn + 6));
    label(c, txt, tx, top - 10, ctx.accent, 'left', 10);
    // published numax
    const xt = Math.round(xs(t.numax)) + 0.5;
    g.fillStyle = SCOPE.ink;
    g.beginPath(); g.moveTo(xt, top + ih - 7); g.lineTo(xt - 4, top + ih); g.lineTo(xt + 4, top + ih); g.closePath(); g.fill();
    label(c, 'Yu+18', xt + (right ? -7 : 7), top + ih - 6, SCOPE.text, right ? 'right' : 'left', 9);
    label(c, 'ppm²/μHz', padL + 4, top + 7, SCOPE.faint, 'left', 9);
  }

  function drawMaps(c: Canvas2D) {
    const { ctx: g, w, h: H } = c;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, w, H);
    if (!view) return;
    const t = stars[st.i].truth, { p, r } = view;
    // two panels: side by side, or stacked on a phone
    const top = 22, bot = 22, padL = 34, padR = 10, gap = 30 + 22;
    const pw = narrow ? w - padL - padR : (w - padL - padR - gap) / 2;
    const ih = narrow ? (H - 2 * (top + bot)) / 2 : H - top - bot;
    const [ex0, ey0] = narrow ? [padL, top + ih + bot + top] : [padL + pw + gap, top];
    // ---- comb map: the 2-D branch's input ----
    const x0 = padL, LC0 = Math.log10(CEN[0]), LC1 = Math.log10(CEN[NC - 1]);
    const half = (Math.log10(CEN[1]) - LC0) / 2;
    const yc = (f: number) => top + ih - ((Math.log10(f) - (LC0 - half)) / (LC1 - LC0 + 2 * half)) * ih;
    const lag0 = L0 / 100, lag1 = (L0 + NL - 1) / 100, dl = 0.005;
    const xc = (q: number) => x0 + ((q - (lag0 - dl)) / (lag1 - lag0 + 2 * dl)) * pw;
    heatmap(c, p.F2, NC, NL, x0, top, pw, ih, ctx.accent, 0, 0.45);
    label(c, 'COMB MAP · the 2-D input', x0, top - 11, SCOPE.text, 'left', 9);
    for (const f of [10, 30, 100]) label(c, String(f), x0 - 5, yc(f), SCOPE.faint, 'right', 9);
    for (const q of narrow ? [0.6, 1] : [0.6, 0.8, 1, 1.2]) label(c, q === 1 ? '1' : q.toFixed(1), xc(q), top + ih + 11, SCOPE.faint, 'center', 9);
    label(c, 'lag ÷ expected Δν', x0 + pw, top + ih + 11, SCOPE.faint, 'right', 9);
    // the network's reading and the published one
    const qn = r.dnu / dnuS(r.numax), qt = t.dnu / dnuS(t.numax);
    g.strokeStyle = ctx.accent; g.setLineDash([3, 3]); g.lineWidth = 1;
    g.beginPath(); g.moveTo(x0, Math.round(yc(r.numax)) + 0.5); g.lineTo(x0 + pw, Math.round(yc(r.numax)) + 0.5); g.stroke(); g.setLineDash([]);
    if (qt > lag0 && qt < lag1) {
      const X0 = xc(qt), Y0 = yc(t.numax);
      g.strokeStyle = SCOPE.ink; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(X0 - 5, Y0 - 5); g.lineTo(X0 + 5, Y0 + 5); g.moveTo(X0 - 5, Y0 + 5); g.lineTo(X0 + 5, Y0 - 5); g.stroke();
    }
    if (qn > lag0 - 0.05 && qn < lag1 + 0.05) {
      g.strokeStyle = ctx.accent; g.lineWidth = 2;
      g.beginPath(); g.arc(xc(qn), yc(r.numax), 6, 0, 2 * Math.PI); g.stroke();
    }
    // ---- échelle diagram at the network's dnu ----
    echelle(c, p, r.numax, r.dnu, ex0, ey0, pw, ih);
    label(c, 'ÉCHELLE · folded at the network’s Δν', ex0, ey0 - 11, SCOPE.text, 'left', 9);
    label(c, 'frequency mod Δν', ex0 + pw, ey0 + ih + 11, SCOPE.faint, 'right', 9);
  }

  function echelle(c: Canvas2D, p: Prepared, numax: number, dnu: number, x0: number, y0: number, pw: number, ih: number) {
    const { ctx: g } = c;
    const ROWS = 9, COLS = 64, n0 = Math.round(numax / dnu - ROWS / 2); // nine orders centred on the network's numax
    const rb = makeRebin(p.f, p.P), fmax = p.f[p.f.length - 1];
    const data = new Float64Array(ROWS * COLS), edges = new Float64Array(COLS + 1);
    const rowsF: number[] = [];
    for (let rI = 0; rI < ROWS; rI++) {
      const fs = (n0 + rI) * dnu;
      rowsF.push(fs);
      if (fs < p.f[0] || fs + dnu > fmax) continue;
      for (let k = 0; k <= COLS; k++) edges[k] = fs + (k * dnu) / COLS;
      const v = rb(edges), srt = Float64Array.from(v).sort(), med = srt[COLS >> 1] || 1;
      for (let k = 0; k < COLS; k++) data[rI * COLS + k] = v[k] / med;
    }
    // roll the columns so the strongest ridge sits at 0.4 of the width (display only: the fold is circular)
    const colSum = new Float64Array(COLS);
    for (let rI = 0; rI < ROWS; rI++) for (let k = 0; k < COLS; k++) colSum[k] += data[rI * COLS + k];
    let best = 0, bestV = -Infinity;
    for (let k = 0; k < COLS; k++) { let v = 0; for (let d = -2; d <= 2; d++) v += colSum[(k + d + COLS) % COLS]; if (v > bestV) { bestV = v; best = k; } }
    const roll = (best - Math.round(0.4 * COLS) + COLS) % COLS;
    const all = Float64Array.from(data).sort(), hi = Math.max(3, all[Math.floor(all.length * 0.985)]);
    const img = g.createImageData(COLS, ROWS), [ar, ag, ab] = hex(ctx.accent);
    for (let rI = 0; rI < ROWS; rI++) for (let k = 0; k < COLS; k++) {
      const v = data[rI * COLS + ((k + roll) % COLS)], tt = Math.max(0, Math.min(1, (v - 0.6) / (hi - 0.6))) ** 0.8;
      const o = ((ROWS - 1 - rI) * COLS + k) * 4, wv = Math.max(0, tt - 0.85) * 4;
      img.data[o] = 13 + (ar - 13) * tt + (255 - ar) * wv;
      img.data[o + 1] = 15 + (ag - 15) * tt + (255 - ag) * wv;
      img.data[o + 2] = 12 + (ab - 12) * tt + (255 - ab) * wv;
      img.data[o + 3] = 255;
    }
    const off = document.createElement('canvas');
    off.width = COLS; off.height = ROWS;
    off.getContext('2d')!.putImageData(img, 0, 0);
    g.imageSmoothingEnabled = false;
    g.drawImage(off, x0, y0, pw, ih);
    g.imageSmoothingEnabled = true;
    // rows labelled by their starting frequency; expected ridges of the universal pattern
    for (let rI = 0; rI < ROWS; rI += 2) label(c, rowsF[rI].toFixed(rowsF[rI] < 20 ? 1 : 0), x0 - 5, y0 + ih - ((rI + 0.5) / ROWS) * ih, SCOPE.faint, 'right', 9);
  }

  await run();
}

function sup(n: number): string {
  const m: Record<string, string> = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  return String(n).split('').map((ch) => m[ch] ?? ch).join('');
}
