// SwellCast live instrument: real NOAA buoy spectra around six hurricane swells, the real network running in the
// browser, and the dispersion ridge its reading implies drawn on the spectrogram it read. Controls change the input:
// noise on every spectral estimate, the buoy feed stopping early (reading in real time), and sliding the window.
import type { InstrumentContext } from '../types.ts';
import { fetchPack } from '../../lib/nn/pack.ts';
import { read, slice, ridgeSlope, NT, NF, DT_H, type SMeta, type SExamples, type Reading } from './net.ts';
import { shell, picker, slider, button, scheduler, compass, canvas, label, rng, fmt, h, SCOPE, type Canvas2D } from '../kit.ts';

const HALF = (NT * DT_H) / 2; // 120 h
const F0 = 0.03, F1 = 0.2; // the frequency grid's cell edges, Hz
const MEDIAN_GUESS = { D: 2364, t0: -48 }; // the dev median ("dumb baseline") from the model card

const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
function utc(hour: number, withHour = true) {
  const d = new Date(Math.round(hour) * 3600e3);
  const hh = String(d.getUTCHours()).padStart(2, '0');
  return `${d.getUTCDate()} ${MON[d.getUTCMonth()]}` + (withHour ? ` ${hh}h` : '');
}
const km = (v: number) => (Number.isFinite(v) ? Math.round(v).toLocaleString('en-US') : '—');
const signed = (v: number) => `${v > 0 ? '+' : v < 0 ? '−' : '±'}${Math.abs(Math.round(v))}`;

function rgb(c: string): [number, number, number] {
  const m = /^#?([0-9a-f]{6})$/i.exec(c.trim());
  const s = m ? m[1] : '0e9aa7';
  return [parseInt(s.slice(0, 2), 16), parseInt(s.slice(2, 4), 16), parseInt(s.slice(4, 6), 16)];
}

/** one readout row in the kit's gauge markup, with free-form text for the value, range and foot */
function gaugeRow(o: { label: string; value: string; unit?: string; range?: string; pos: (v: number) => number; at: number; lo?: number; hi?: number; tick?: number; tickTitle?: string; footL?: string; footR?: string; footClass?: string }) {
  const pc = (v: number) => `${(Math.max(0, Math.min(1, o.pos(v))) * 100).toFixed(2)}%`;
  const row = h('div', { class: 'inst-gauge' });
  row.innerHTML = `
    <div class="ig-top"><span class="ig-l">${o.label}</span>
      <span class="ig-v"><b>${o.value}</b>${o.unit ? `<em>${o.unit}</em>` : ''}${o.range ? `<small>${o.range}</small>` : ''}</span></div>
    <div class="ig-track">
      ${o.lo != null && o.hi != null ? `<span class="ig-band" style="left:${pc(o.lo)};right:calc(100% - ${pc(o.hi)})"></span>` : ''}
      <span class="ig-dot" style="left:${pc(o.at)}"></span>
      ${o.tick != null ? `<span class="ig-truth" style="left:${pc(o.tick)}" title="${o.tickTitle ?? 'truth'}"></span>` : ''}
    </div>
    ${o.footL || o.footR ? `<div class="ig-foot"><span>${o.footL ?? ''}</span><span class="${o.footClass ?? ''}">${o.footR ?? ''}</span></div>` : ''}`;
  return row;
}

export async function mount(ctx: InstrumentContext) {
  const [W, X] = await Promise.all([
    fetchPack<SMeta>(`${ctx.base}weights`),
    fetchPack<SExamples>(`${ctx.base}examples`),
  ]);
  const { pad, examples: ex } = X.meta;
  const Tw = NT + 2 * pad;
  const ui = shell(ctx.root, {
    title: 'SwellCast',
    subtitle: 'one wave buoy hears a hurricane thousands of km away — real NOAA spectra, never used in training',
    params: '296 k parameters · matches Python to 3e-6',
  });
  const st = { i: 0, noise: 0, end: HALF, shift: 0, seed: 3, anomaly: true };
  const acc = rgb(ctx.accent);

  // clean colour scale per example (so added noise looks like noise, not a re-stretch)
  const scale = ex.map((_, i) => {
    const E = X.t[`x${i}`].data, vals: number[] = [], an: number[] = [];
    for (let j = 0; j < NF; j++) {
      const col: number[] = [];
      for (let t = pad; t < pad + NT; t++) { const e = E[t * NF + j]; if (Number.isFinite(e)) col.push(Math.log10(Math.max(e, 0.005))); }
      col.sort((a, b) => a - b);
      const med = col.length ? col[col.length >> 1] : -2.3;
      for (const v of col) { vals.push(v); an.push(v - med); }
    }
    const q = (a: number[], p: number) => { a.sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.floor(p * a.length))]; };
    return { lo: q(vals, 0.03), hi: q(vals, 0.995), alo: 0, ahi: Math.max(1, q(an, 0.995)) };
  });

  picker(ui.picker, ex.map((e, i) => ({
    label: `${e.name} ${e.year}`,
    sub: `${e.buoy} · ${e.place}`,
    value: i,
  })), (i) => { st.i = i; go(); }, st.i);

  let view: { x: Float32Array; r: Reading | null } = { x: new Float32Array(3 * NT * NF), r: null };
  const small = ctx.root.getBoundingClientRect().width < 560;
  const plot = canvas(ui.screen, small ? 1.2 : ctx.compact ? 1.9 : 1.75, (c) => draw(c));

  slider(ui.controls, {
    label: 'Spectral noise', min: 0, max: 1, step: 0.05, value: 0,
    fmt: (v) => (v === 0 ? 'none' : `×÷ ${(10 ** v).toFixed(1)}`),
    hint: 'every energy reading scaled by a random factor', onInput: (v) => { st.noise = v; go(); },
  });
  slider(ui.controls, {
    label: 'Buoy feed ends', min: -96, max: HALF, step: DT_H, value: HALF,
    fmt: (v) => (v >= HALF ? 'full 10 days' : `${signed(v)} h`),
    hint: 'later hours are missing — reading in real time', onInput: (v) => { st.end = v; go(); },
  });
  let shiftHint: HTMLElement | null = null;
  if (!ctx.compact) {
    const s = slider(ui.controls, {
      label: 'Slide the window', min: -pad * DT_H, max: pad * DT_H, step: DT_H, value: 0,
      fmt: (v) => (v === 0 ? 'centred' : `${signed(v)} h`),
      hint: 'trained on arrivals within ±36 h of centre', onInput: (v) => { st.shift = v; go(); },
    });
    shiftHint = s.input.parentElement!.querySelector('small');
    const btns = h('div', { style: 'display:flex;gap:8px;flex-wrap:wrap;grid-column:1/-1' });
    ui.controls.append(btns);
    button(btns, 'New noise draw', () => { st.seed++; if (st.noise > 0) go(); });
    const tog = button(btns, 'Show: energy', () => { st.anomaly = !st.anomaly; tog.textContent = st.anomaly ? 'Show: energy' : 'Show: anomaly'; plot.redraw(); });
  }

  // noise is drawn once per (example, seed) over the whole exported span, so sliding the window keeps it attached
  let noiseKey = '', noiseArr = new Float32Array(0);
  function noiseFor(i: number) {
    const k = `${i}:${st.seed}`;
    if (k !== noiseKey) {
      const r = rng(st.seed * 7919 + i * 31 + 1);
      noiseArr = new Float32Array(Tw * NF);
      for (let q = 0; q < noiseArr.length; q++) noiseArr[q] = r.n();
      noiseKey = k;
    }
    return noiseArr;
  }

  function input(): Float32Array {
    const sh = st.shift / DT_H;
    const x = slice(X.t[`x${st.i}`], pad, sh);
    if (st.noise > 0) {
      const n = noiseFor(st.i), off = (pad + sh) * NF;
      for (let q = 0; q < NT * NF; q++) if (Number.isFinite(x[q])) x[q] *= 10 ** (st.noise * n[off + q]);
    }
    if (st.end < HALF) {
      const t0 = Math.max(0, Math.ceil((st.end + HALF) / DT_H));
      for (let ch = 0; ch < 3; ch++) x.fill(NaN, (ch * NT + t0) * NF, (ch + 1) * NT * NF);
    }
    return x;
  }

  async function run() {
    ui.setBusy(true);
    await new Promise((r) => requestAnimationFrame(r));
    const e = ex[st.i];
    const x = input();
    const t0 = performance.now();
    const { r } = read(W, x);
    const ms = performance.now() - t0;
    view = { x, r };
    plot.redraw();
    readout(r);
    if (shiftHint) shiftHint.textContent = `trained on arrivals within ±36 h of centre · ${e.raw ? 'real hours slide in' : 'no data beyond'}`;
    ui.stamp(ms, `Hurricane ${e.name} → buoy ${e.buoy}, window centred ${utc(e.centre_hour + st.shift)} UTC ${e.year} · on the exam a median guess (${km(MEDIAN_GUESS.D)} km, ${signed(MEDIAN_GUESS.t0)} h) still beats it on distance and time`);
    ui.setBusy(false);
  }
  const go = scheduler(run, 30);

  function readout(r: Reading) {
    const e = ex[st.i], el = ui.readout;
    el.innerHTML = '';
    const thr = W.meta.det_thr;
    el.append(gaugeRow({
      label: 'Swell arriving?', value: r.detected ? 'yes' : 'no', range: `score ${r.p_swell.toFixed(2)}`,
      pos: (v) => v, at: r.p_swell, tick: thr, tickTitle: 'threshold (10% false alarms on quiet records)',
      footL: `threshold <b>${thr.toFixed(2)}</b> · truth: it arrived`, footR: r.detected ? 'detected' : 'missed', footClass: r.detected ? 'in' : 'out',
    }));
    const lD = (v: number) => (Math.log10(v) - Math.log10(500)) / (Math.log10(12000) - Math.log10(500));
    const dlo = r.D_km * 10 ** -r.logD_sig, dhi = r.D_km * 10 ** r.logD_sig;
    const inD = e.truth.D >= dlo && e.truth.D <= dhi;
    el.append(gaugeRow({
      label: 'Distance', value: km(r.D_km), unit: 'km', range: `${km(dlo)}–${km(dhi)}`,
      pos: lD, at: r.D_km, lo: dlo, hi: dhi, tick: e.truth.D, tickTitle: 'IBTrACS best track',
      footL: `IBTrACS <b>${km(e.truth.D)} km</b>`, footR: inD ? 'inside the 68% bar' : 'outside the bar', footClass: inD ? 'in' : 'out',
    }));
    // storm time on a fixed axis: hours from the exported window's own centre, so the truth stays put while you slide
    const tAbs = st.shift + r.t0, lo = tAbs - r.t0_sig, hi = tAbs + r.t0_sig;
    const inT = e.truth.t0 >= lo && e.truth.t0 <= hi;
    el.append(gaugeRow({
      label: 'Storm peak', value: utc(e.centre_hour + tAbs), unit: 'UTC', range: `± ${Math.round(r.t0_sig)} h`,
      pos: (v) => (v + 240) / 360, at: tAbs, lo, hi, tick: e.truth.t0, tickTitle: 'IBTrACS peak wind',
      footL: `IBTrACS <b>${utc(e.centre_hour + e.truth.t0)}</b>`, footR: inT ? 'inside the 68% bar' : 'outside the bar', footClass: inT ? 'in' : 'out',
    }));
    compass(el, { label: 'Swell comes from', deg: r.dir, half: r.half, truth: e.truth.dir });
    const lH = (v: number) => (Math.log10(v) - Math.log10(2)) / (Math.log10(25) - Math.log10(2));
    const f = 10 ** r.loghs_sig;
    el.append(gaugeRow({
      label: 'Storm wave height', value: fmt(r.hs, 1), unit: 'm', range: `${fmt(r.hs / f, 1)}–${fmt(r.hs * f, 1)}`,
      pos: lH, at: r.hs, lo: r.hs / f, hi: r.hs * f,
      footL: 'no truth — best tracks record wind, not waves',
    }));
  }

  // ---- the spectrogram ------------------------------------------------------------------------
  function draw(c: Canvas2D) {
    const { ctx: g, w, h: H } = c;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, w, H);
    const narrow = w < 520;
    const padL = narrow ? 34 : 42, padR = narrow ? 30 : 38, top = narrow ? 36 : 26, bot = 36;
    const iw = w - padL - padR, ih = H - top - bot;
    const xs = (t: number) => padL + ((t + HALF) / (2 * HALF)) * iw;
    const ys = (f: number) => top + (1 - (f - F0) / (F1 - F0)) * ih;
    const x = view.x, sc = scale[st.i];

    // hatch for missing data, then the cells on top (missing cells transparent)
    g.save();
    g.beginPath(); g.rect(padL, top, iw, ih); g.clip();
    g.fillStyle = '#181a16'; g.fillRect(padL, top, iw, ih);
    g.strokeStyle = 'rgba(235,230,216,0.1)'; g.lineWidth = 1; g.beginPath();
    for (let k = -ih; k < iw; k += 7) { g.moveTo(padL + k, top + ih); g.lineTo(padL + k + ih, top); }
    g.stroke();
    const img = g.createImageData(NT, NF);
    const med = new Float32Array(NF);
    if (st.anomaly) for (let j = 0; j < NF; j++) {
      const col: number[] = [];
      for (let t = 0; t < NT; t++) { const e = x[t * NF + j]; if (Number.isFinite(e)) col.push(Math.log10(Math.max(e, 0.005))); }
      col.sort((a, b) => a - b);
      const n = col.length;
      med[j] = n ? (n % 2 ? col[n >> 1] : (col[(n >> 1) - 1] + col[n >> 1]) / 2) : 0;
    }
    const [lo, hi] = st.anomaly ? [sc.alo, sc.ahi] : [sc.lo, sc.hi];
    for (let t = 0; t < NT; t++) for (let j = 0; j < NF; j++) {
      const e = x[t * NF + j], k = ((NF - 1 - j) * NT + t) * 4;
      if (!Number.isFinite(e)) continue;
      const v = Math.log10(Math.max(e, 0.005)) - (st.anomaly ? med[j] : 0);
      const u = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
      // ramp: scope black -> accent (at 0.65) -> warm white
      const a = Math.min(1, u / 0.65), b = Math.max(0, (u - 0.65) / 0.35);
      img.data[k] = 13 + (acc[0] - 13) * a + (246 - acc[0]) * b;
      img.data[k + 1] = 15 + (acc[1] - 15) * a + (240 - acc[1]) * b;
      img.data[k + 2] = 12 + (acc[2] - 12) * a + (222 - acc[2]) * b;
      img.data[k + 3] = 255;
    }
    const off = document.createElement('canvas');
    off.width = NT; off.height = NF;
    off.getContext('2d')!.putImageData(img, 0, 0);
    g.imageSmoothingEnabled = false;
    g.drawImage(off, padL, top, iw, ih);

    // feed-ends line
    if (st.end < HALF) {
      const xe = xs(Math.max(-HALF, Math.ceil((st.end + HALF) / DT_H) * DT_H - HALF));
      g.strokeStyle = ctx.accent; g.lineWidth = 1.5; g.setLineDash([4, 3]);
      g.beginPath(); g.moveTo(xe + 0.5, top); g.lineTo(xe + 0.5, top + ih); g.stroke(); g.setLineDash([]);
      label(c, 'feed ends', xe + 6, top + 10, ctx.accent);
      if (xe < padL + iw - 70) label(c, 'no data yet', xe + 6, top + 24, SCOPE.faint);
    }
    // missing-margin label when sliding an exam window
    if (st.shift !== 0 && !ex[st.i].raw) {
      const right = st.shift > 0, edge = xs(right ? HALF - st.shift : -HALF - st.shift);
      label(c, 'no data', right ? (edge + padL + iw) / 2 : (padL + edge) / 2, top + ih / 2, SCOPE.text, 'center');
    }

    // ridges: f = g (t - t0) / (4 pi D)
    const r = view.r, e = ex[st.i];
    const tL = -HALF, tR = HALF;
    const line = (D: number, t0: number) => { const k = ridgeSlope(D); g.beginPath(); g.moveTo(xs(tL), ys(k * (tL - t0))); g.lineTo(xs(tR), ys(k * (tR - t0))); };
    if (r) {
      // the 68% distance range as a fan
      const k1 = ridgeSlope(r.D_km * 10 ** -r.logD_sig), k2 = ridgeSlope(r.D_km * 10 ** r.logD_sig);
      g.beginPath();
      g.moveTo(xs(r.t0), ys(0));
      g.lineTo(xs(tR + 400), ys(k1 * (tR + 400 - r.t0)));
      g.lineTo(xs(tR + 400), ys(k2 * (tR + 400 - r.t0)));
      g.closePath();
      g.fillStyle = 'rgba(235,230,216,0.07)'; g.fill();
    }
    // true storm ridge (fainter, dashed)
    g.setLineDash([6, 5]); g.lineWidth = 1.6; g.strokeStyle = 'rgba(235,230,216,0.75)';
    line(e.truth.D, e.truth.t0 - st.shift); g.stroke(); g.setLineDash([]);
    if (r) {
      line(r.D_km, r.t0);
      g.lineWidth = 5; g.strokeStyle = 'rgba(0,0,0,0.55)'; g.stroke();
      g.lineWidth = 2.2; g.strokeStyle = '#f6f0de'; g.stroke();
    }
    g.restore();

    // storm-peak markers on the time axis (where each ridge would reach zero frequency)
    const tri = (t: number, col: string) => {
      if (t < -HALF || t > HALF) return;
      const xx = xs(t), yy = top + ih + 1;
      g.fillStyle = col; g.beginPath(); g.moveTo(xx, yy); g.lineTo(xx - 4, yy + 6); g.lineTo(xx + 4, yy + 6); g.closePath(); g.fill();
    };
    tri(e.truth.t0 - st.shift, 'rgba(235,230,216,0.6)');
    if (r) tri(r.t0, '#f6f0de');

    // axes
    g.strokeStyle = SCOPE.grid; g.lineWidth = 1; g.strokeRect(padL + 0.5, top + 0.5, iw - 1, ih - 1);
    const step = narrow ? 60 : 24;
    for (let t = -HALF; t <= HALF; t += step) {
      const xx = xs(t);
      g.strokeStyle = SCOPE.faint; g.beginPath(); g.moveTo(xx + 0.5, top + ih); g.lineTo(xx + 0.5, top + ih + 4); g.stroke();
      label(c, t === 0 ? '0' : `${signed(t)}`, xx, top + ih + 12, t === 0 ? SCOPE.text : SCOPE.faint, 'center', 9);
    }
    label(c, 'hours from window centre', padL + iw / 2, H - 7, SCOPE.faint, 'center', 9);
    for (const f of [0.05, 0.1, 0.15]) {
      const yy = ys(f);
      label(c, f.toFixed(2), padL - 5, yy, SCOPE.faint, 'right', 9);
      label(c, `${Math.round(1 / f)}s`, padL + iw + 5, yy, SCOPE.faint, 'left', 9);
    }
    label(c, 'Hz', padL - 5, top + ih - 6, SCOPE.faint, 'right', 9);

    // legend
    const ly = narrow ? 10 : 12;
    let lx = padL;
    g.lineWidth = 2.2; g.strokeStyle = '#f6f0de';
    g.beginPath(); g.moveTo(lx, ly); g.lineTo(lx + 18, ly); g.stroke();
    const netTxt = r ? `network ridge · ${km(r.D_km)} km` : 'network ridge';
    label(c, netTxt, lx + 24, ly, SCOPE.ink, 'left', 10);
    lx += 24 + g.measureText(netTxt).width + 18;
    let ly2 = ly;
    if (narrow) { lx = padL; ly2 = ly + 15; }
    g.setLineDash([6, 5]); g.lineWidth = 1.6; g.strokeStyle = 'rgba(235,230,216,0.75)';
    g.beginPath(); g.moveTo(lx, ly2); g.lineTo(lx + 18, ly2); g.stroke(); g.setLineDash([]);
    const tTxt = `storm (IBTrACS) · ${km(e.truth.D)} km`;
    label(c, tTxt, lx + 24, ly2, SCOPE.text, 'left', 10);
    const used = lx + 24 + g.measureText(tTxt).width + 16;
    const note = `f = g(t − t₀) / 4πD · colour: ${st.anomaly ? 'energy vs each frequency’s median' : 'log energy'}`;
    g.font = '9px "Martian Mono Variable", ui-monospace, monospace';
    if (padL + iw - g.measureText(note).width > used) label(c, note, padL + iw, ly, SCOPE.faint, 'right', 9);
  }

  await run();
}
