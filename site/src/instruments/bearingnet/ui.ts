// BearingNet live instrument: 14 real exam recordings (CWRU and Paderborn), the whole reading pipeline — kurtogram band,
// envelope spectra, fault-aligned strips — and the network, all running in the browser. Tell it the wrong shaft speed
// and watch the fault combs slide off the peaks; add noise and watch the impacts drown.
import type { InstrumentContext } from '../types.ts';
import { fetchPack } from '../../lib/nn/pack.ts';
import { addNoise, recordSpectra, runNet, orders, DF, NV, NH, NB, NCH, V, type BMeta, type BExample, type Geom, type Spectra, type Reading } from './net.ts';
import { shell, picker, slider, button, scheduler, probs, gauges, canvas, label, heatmap, rng, h, SCOPE, type Canvas2D, type Gauge } from '../kit.ts';

const NAME: Record<string, string> = { N: 'healthy', OR: 'outer race', IR: 'inner race', B: 'ball' };
const COMB = { BPFO: '#e5a54b', BPFI: '#5fc4ae', BSF: '#cf8fd8' } as const;
const FAULT_OF: Record<string, keyof typeof COMB | null> = { N: null, OR: 'BPFO', IR: 'BPFI', B: 'BSF' };
const OTHER_BEARING: Record<number, Geom> = { // "tell it the wrong bearing": the other bearing size in the collection
  9: { n: 8, d: 0.2656 * 25.4, D: 1.122 * 25.4 },   // SKF 6203
  8: { n: 9, d: 0.3126 * 25.4, D: 1.537 * 25.4 },   // SKF 6205
};

function mix(a: string, b: string, t: number) {
  const p = (c: string) => [1, 3, 5].map((k) => parseInt(c.slice(k, k + 2), 16));
  const A = p(a), B = p(b);
  return `#${A.map((v, i) => Math.round(v + (B[i] - v) * t).toString(16).padStart(2, '0')).join('')}`;
}

export async function mount(ctx: InstrumentContext) {
  const [W, X] = await Promise.all([fetchPack<BMeta>(`${ctx.base}weights`), fetchPack<BExample[]>(`${ctx.base}examples`)]);
  const ex = X.meta;
  const ui = shell(ctx.root, {
    title: 'BearingNet',
    subtitle: 'one accelerometer on a real test rig — is the bearing healthy, or which part is cracked?',
    params: '438 k parameters · matches Python to 1e-5',
  });
  const ink = mix(ctx.accent, SCOPE.ink, 0.45);
  const st = { i: 2, err: 0, sigma: 0, seed: 7, wrong: false };
  const narrow = ctx.root.getBoundingClientRect().width < 560;

  // spectra depend on the recording and the noise only; a speed or bearing change re-reads them (cheap)
  let cache: { key: string; sps: Spectra[] } | null = null;
  let view: { sps: Spectra[]; feats: { F: Float32Array }[]; r: Reading; rpm: number; g: Geom } | null = null;

  const loadOf = (e: BExample) => (e.rig === 'CWRU' ? `${e.cond.split('_').pop()} hp` : e.cond.includes('M01') ? '0.1 N·m' : '0.7 N·m');
  const items = ex.map((e, i) => (narrow
    ? { label: `${e.truth} · ${e.file}${e.rig === 'CWRU' ? '' : e.cond.includes('M01') ? '/M01' : '/M07'}`, value: i }
    : { label: `${e.rig === 'CWRU' ? 'CWRU' : 'PU'} ${e.file}`, sub: `${NAME[e.truth]}${e.size_mm ? ` · ${e.size_mm.toFixed(2)} mm` : ` · ${loadOf(e)}`}`, value: i }));
  const shown = ctx.compact ? [0, 2, 4, 7, 10, 13].map((k) => items[k]) : items;
  picker(ui.picker, shown, (i) => { st.i = i; spd.set(0); st.err = 0; st.wrong = false; go(); }, shown.findIndex((it) => it.value === st.i));

  const plot = canvas(ui.screen, narrow ? 1.0 : ctx.compact ? 1.55 : 1.6, (c) => draw(c));
  const spd = slider(ui.controls, {
    label: 'Speed you tell it', min: -8, max: 8, step: 0.5, value: 0,
    fmt: (v) => { const e = ex[st.i], r = told(v); return `${Math.round(r)} rpm ${pct(r / e.rpm_true - 1)}`; },
    hint: '% vs the true speed · centre = nameplate · trained for errors within ±5%',
    onInput: (v) => { st.err = v; go(); },
  });
  slider(ui.controls, {
    label: 'Add noise', min: 0, max: 5, step: 0.1, value: 0,
    fmt: (v) => (v === 0 ? 'none' : `σ ${v.toFixed(1)} × rms`),
    hint: 'white Gaussian, in units of the recording’s own rms',
    onInput: (v) => { st.sigma = v; go(); },
  });
  let wrongBtn: HTMLButtonElement | null = null;
  if (!ctx.compact) {
    const row = h('div', { style: 'display:flex;gap:8px;flex-wrap:wrap;align-self:end' });
    ui.controls.append(row);
    wrongBtn = button(row, 'Tell it the wrong bearing', () => { st.wrong = !st.wrong; go(); });
    wrongBtn.setAttribute('aria-pressed', 'false');
    button(row, 'New noise draw', () => { st.seed++; if (st.sigma > 0) go(); });
  }
  const note = h('div', { style: 'font-family:var(--mono);font-variation-settings:"wdth" 87.5;font-size:10.5px;line-height:1.55;color:var(--scr-3ink);letter-spacing:0.02em' });
  const nRight = ex.filter((e) => e.python.condition === e.truth).length;

  function told(v = st.err) { return ex[st.i].rpm_nominal * (1 + v / 100); }
  function pct(f: number) { const s = (f * 100).toFixed(1); return `${f >= 0 ? '+' : ''}${s === '-0.0' ? '0.0' : s}%`; }
  function geom(): Geom { const g = ex[st.i].bearing; return st.wrong ? OTHER_BEARING[g.n] ?? g : g; }


  async function run() {
    ui.setBusy(true);
    await new Promise((r) => requestAnimationFrame(r));
    const e = ex[st.i];
    const t0 = performance.now();
    const key = `${st.i}|${st.sigma}|${st.seed}`;
    if (cache?.key !== key) {
      const x = addNoise(X.t[`x${st.i}`].data, st.sigma, rng(st.seed * 101 + st.i).n);
      cache = { key, sps: recordSpectra(x) };
    }
    const rpm = told(), g = geom();
    const { feats, reading: r } = runNet(W, cache.sps, g, rpm);
    const ms = performance.now() - t0;
    view = { sps: cache.sps, feats, r, rpm, g };
    plot.redraw();
    if (wrongBtn) {
      wrongBtn.setAttribute('aria-pressed', String(st.wrong));
      wrongBtn.textContent = st.wrong ? 'Undo the wrong bearing' : 'Tell it the wrong bearing';
      wrongBtn.style.borderColor = st.wrong ? ctx.accent : '';
    }
    spd.set(st.err);

    probs(ui.readout, 'Condition', W.meta.classes.map((c, k) => ({ label: NAME[c], p: r.p[k], truth: c === e.truth })));
    const gs: Gauge[] = [{
      label: 'Shaft speed', unit: 'rpm', value: r.rpm, lo: r.rpm - r.rpm_sig, hi: r.rpm + r.rpm_sig,
      truth: e.rpm_true, truthLabel: 'measured', range: [e.rpm_true * 0.91, e.rpm_true * 1.09], digits: 0,
    }];
    if (!ctx.compact) {
      const s = r.log10_size_sig;
      gs.push({
        label: 'Defect size (experimental)', unit: 'mm', value: r.size_mm, lo: r.size_mm / 10 ** s, hi: r.size_mm * 10 ** s,
        truth: e.size_mm || null, truthLabel: 'seeded defect', range: [0.05, 5], digits: 2, log: true,
      });
    }
    gauges(ui.readout, gs);
    const right = r.condition === e.truth;
    note.innerHTML =
      `<div style="color:var(--scr-2ink)">Reading: <b style="font-weight:400;color:${right ? '#7fd6a6' : '#ff8a7a'}">${NAME[r.condition]} — ${right ? 'right' : 'wrong'}</b>` +
      ` · truth ${NAME[e.truth]}</div>` +
      (st.wrong ? `<div style="color:#e5a54b">Told the wrong bearing: ${g.n} balls, d/D ${(g.d / g.D).toFixed(3)} (really ${e.bearing.n}, ${(e.bearing.d / e.bearing.D).toFixed(3)}) — every fault comb moves.</div>` : '') +
      `<div>Told ${Math.round(rpm)} rpm (${pct(rpm / e.rpm_true - 1)} vs true ${Math.round(e.rpm_true)}); the net moved it ${pct(r.rpm / rpm - 1)}.</div>` +
      `<div>At the nameplate speed it reads <b style="font-weight:400;color:var(--scr-ink)">${nRight} of these ${ex.length}</b> recordings right — the same in Python.` +
      `${ctx.compact ? '' : ' Its size estimate never beat a median guess on the exam.'}</div>`;
    ui.readout.append(note);
    ui.stamp(ms, `front end + network on ${r.n_windows} windows of 2.7 s · ${e.name}`);
    ui.setBusy(false);
  }
  const go = scheduler(run, 30);

  function draw(c: Canvas2D) {
    const { ctx: g, w, h: H } = c;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, w, H);
    if (!view) return;
    const e = ex[st.i], v = view;
    const padL = narrow ? 38 : 46, padR = 10, iw = w - padL - padR;
    const laneH = narrow ? 10 : 11, lanesTop = 8, specTop = lanesTop + 3 * laneH + 6;
    const axisH = 18, titleH = 16, axis2H = 14;
    const rest = H - specTop - axisH - titleH - axis2H - 4;
    const specH = Math.round(rest * 0.58), heatTop = specTop + specH + axisH + titleH, heatH = rest - specH;

    // ---- envelope spectrum (kurtogram band), mean over windows, linear frequency axis --------------------
    const o = orders(e.bearing); // the axis span is fixed per recording, so the combs visibly move against it
    const fmax = Math.ceil((o.BPFI * e.rpm_nominal / 60) * 3.6 / 50) * 50;
    const nb = Math.min(v.sps[0].ses[4].length - 1, Math.ceil(fmax / DF));
    const spec = new Float64Array(nb + 1);
    for (const sp of v.sps) for (let k = 1; k <= nb; k++) spec[k] += sp.ses[4][k] / v.sps.length;
    const lg = Array.from(spec, (s) => Math.log10(s + 1e-12)).slice(1);
    const sorted = [...lg].sort((a, b) => a - b);
    const lo = sorted[Math.floor(sorted.length * 0.25)] - 0.15, hi = sorted[sorted.length - 1] + 0.05;
    const xf = (f: number) => padL + (f / fmax) * iw;
    const yv = (l: number) => specTop + specH - Math.max(0, Math.min(1, (l - lo) / (hi - lo))) * specH;
    // grid + Hz ticks
    const step = (fmax > 400 ? 100 : 50) * (narrow ? 2 : 1);
    g.strokeStyle = SCOPE.grid; g.lineWidth = 1; g.beginPath();
    for (let f = 0; f <= fmax; f += step) { const x = Math.round(xf(f)) + 0.5; g.moveTo(x, specTop); g.lineTo(x, specTop + specH); }
    for (let j = 0; j <= 4; j++) { const y = Math.round(specTop + (specH * j) / 4) + 0.5; g.moveTo(padL, y); g.lineTo(padL + iw, y); }
    g.stroke();
    for (let f = 0; f <= fmax; f += step) {
      const last = f + step > fmax;
      label(c, last ? `${f} Hz` : `${f}`, last ? xf(f) + 2 : xf(f), specTop + specH + 9, SCOPE.faint, last ? 'right' : 'center', 9);
    }

    // comb positions at the speed it was told (what the network assumes), for the geometry it was told
    const og = orders(v.g), fr = v.rpm / 60;
    const combs: [keyof typeof COMB, number][] = [['BPFO', og.BPFO * fr], ['BPFI', og.BPFI * fr], ['BSF', og.BSF * fr]];
    const top = FAULT_OF[v.r.condition];
    // shaft harmonics: faint ticks on the axis
    g.strokeStyle = SCOPE.text; g.globalAlpha = 0.6; g.beginPath();
    for (let k = 1; k * fr <= fmax; k++) { const x = Math.round(xf(k * fr)) + 0.5; g.moveTo(x, specTop + specH - 5); g.lineTo(x, specTop + specH); }
    g.stroke(); g.globalAlpha = 1;
    for (const [name, f0] of combs) {
      const col = COMB[name];
      g.strokeStyle = col; g.globalAlpha = name === top ? 0.55 : 0.22; g.setLineDash([2, 3]); g.beginPath();
      for (let k = 1; k * f0 <= fmax; k++) { const x = Math.round(xf(k * f0)) + 0.5; g.moveTo(x, specTop); g.lineTo(x, specTop + specH); }
      g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
    }
    // the spectrum: per-pixel max, filled
    const cols = Math.max(1, Math.floor(iw));
    const ys = new Float64Array(cols);
    for (let px = 0; px < cols; px++) {
      const a = Math.max(0, Math.floor((px / cols) * nb)), b = Math.max(a + 1, Math.floor(((px + 1) / cols) * nb));
      let m = -Infinity;
      for (let k = a; k < b && k < lg.length; k++) if (lg[k] > m) m = lg[k];
      ys[px] = yv(m);
    }
    g.beginPath(); g.moveTo(padL, specTop + specH);
    for (let px = 0; px < cols; px++) g.lineTo(padL + px + 0.5, ys[px]);
    g.lineTo(padL + cols, specTop + specH); g.closePath();
    g.fillStyle = mix(ctx.accent, SCOPE.ink, 0.1); g.globalAlpha = 0.28; g.fill(); g.globalAlpha = 1;
    g.beginPath();
    for (let px = 0; px < cols; px++) px ? g.lineTo(padL + px + 0.5, ys[px]) : g.moveTo(padL + 0.5, ys[0]);
    g.strokeStyle = ink; g.lineWidth = 1.1; g.stroke();
    const sk = v.sps.map((s) => s.band);
    const same = sk.every((b) => b[0] === sk[0][0] && b[1] === sk[0][1]);
    const bandTxt = same ? `${(sk[0][0] / 1000).toFixed(2)}–${(sk[0][1] / 1000).toFixed(2)} kHz` : 'varies by window';
    tag(c, narrow ? `band ${bandTxt}` : `envelope spectrum · kurtogram band ${bandTxt}`, padL + 6, specTop + 10);

    // lanes: comb teeth at the told speed; triangles where the true fault sits (true speed, true bearing)
    const tf = FAULT_OF[e.truth], ot = orders(e.bearing), frT = e.rpm_true / 60;
    combs.forEach(([name, f0], j) => {
      const y = lanesTop + j * laneH + laneH / 2, col = COMB[name], on = name === top;
      label(c, name, 4, y, on ? col : mix(col, '#0d0f0c', 0.35), 'left', narrow ? 8.5 : 9.5);
      g.fillStyle = col; g.globalAlpha = on ? 1 : 0.55;
      for (let k = 1; k * f0 <= fmax; k++) g.fillRect(Math.round(xf(k * f0)) - 1, y - 3, 2, 7);
      g.globalAlpha = 1;
      if (name === tf) {
        const ft = ot[name] * frT;
        g.strokeStyle = SCOPE.ink; g.lineWidth = 1;
        for (let k = 1; k * ft <= fmax; k++) {
          const x = xf(k * ft);
          g.beginPath(); g.moveTo(x - 3.5, y - 4.5); g.lineTo(x + 3.5, y - 4.5); g.lineTo(x, y + 1.5); g.closePath(); g.stroke();
        }
      }
    });
    tag(c, tf ? `▽ true ${tf}${narrow ? '' : ' (true speed)'}` : narrow ? 'truth: healthy' : 'truth: healthy, no fault comb', padL + iw - 6, specTop + 10, 'right');

    // ---- what the network reads: all 21 strips (mean over windows), on hypothesis-aligned log-frequency axes ----
    label(c, narrow ? 'network input · 21 strips' : 'what the network reads · 5 bands × 4 fault-aligned axes + raw',
      padL, heatTop - 8, SCOPE.text, 'left', 9);
    const D = new Float32Array(NCH * NV), order: number[] = [];
    for (let hh = 0; hh < NH; hh++) for (let b = 0; b < NB; b++) order.push(b * NH + hh);
    order.push(NCH - 1);
    order.forEach((src, row) => {
      const dst = (NCH - 1 - row) * NV; // heatmap() draws row 0 at the bottom
      for (const f of v.feats) for (let k = 0; k < NV; k++) D[dst + k] += f.F[src * NV + k] / v.feats.length;
    });
    const gap = 2, rh = (heatH - gap * 4) / NCH;
    // draw each group separately so the groups get a small gap
    for (let gi = 0; gi < 5; gi++) {
      const r0 = gi * NB, n = gi < 4 ? NB : 1;
      const sub = new Float32Array(n * NV);
      for (let r = 0; r < n; r++) sub.set(D.subarray((NCH - 1 - (r0 + r)) * NV, (NCH - r0 - r) * NV), (n - 1 - r) * NV);
      const y0 = heatTop + gi * gap + r0 * rh;
      heatmap(c, sub, n, NV, padL, y0, iw, n * rh, ink, -0.3, 1.9);
      const nm = ['shaft', 'BPFO', 'BPFI', 'BSF', 'raw'][gi];
      const col = gi >= 1 && gi <= 3 ? COMB[nm as keyof typeof COMB] : SCOPE.text;
      label(c, nm, 4, y0 + (n * rh) / 2, col, 'left', narrow ? 8.5 : 9.5);
    }
    // harmonic guides 1x..6x (where the network expects a comb) and the true position (shifted by the speed error)
    const xv = (vv: number) => padL + ((vv + 2) / 4.6) * iw;
    g.strokeStyle = SCOPE.ink; g.globalAlpha = 0.45; g.setLineDash([2, 3]); g.beginPath();
    for (let k = 1; k <= 6; k++) { const x = Math.round(xv(Math.log2(k))) + 0.5; g.moveTo(x, heatTop); g.lineTo(x, heatTop + heatH); }
    g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
    for (let k = 1; k <= 6; k++) if (!narrow || k <= 4) label(c, `${k}×`, xv(Math.log2(k)), heatTop + heatH + 8, SCOPE.faint, 'center', 9);
    label(c, '½×', xv(-1), heatTop + heatH + 8, SCOPE.faint, 'center', 9);
    const shift = Math.log2(e.rpm_true / v.rpm);
    if (Math.abs(shift) > 0.004) {
      g.fillStyle = SCOPE.ink;
      for (let k = 1; k <= 4; k++) {
        const x = xv(Math.log2(k) + shift);
        g.beginPath(); g.moveTo(x - 3.5, heatTop - 1); g.lineTo(x + 3.5, heatTop - 1); g.lineTo(x, heatTop + 4); g.closePath(); g.fill();
      }
      label(c, `▼ true ${narrow ? '' : '1×–4×: shifted '}${shift > 0 ? '+' : ''}${(shift * 64).toFixed(1)} cells`, padL + iw, heatTop - 8, SCOPE.ink, 'right', 9);
    }
  }

  await run();
}

/** A label on a dark plate, so it stays legible over the spectrum. */
function tag(c: Canvas2D, text: string, x: number, y: number, align: CanvasTextAlign = 'left') {
  const g = c.ctx;
  g.font = '9px "Martian Mono Variable", ui-monospace, monospace';
  const w = g.measureText(text).width + 8, x0 = align === 'right' ? x - w + 4 : x - 4;
  g.fillStyle = 'rgba(13,15,12,0.82)';
  g.fillRect(x0, y - 7, w, 14);
  label(c, text, x, y, SCOPE.text, align, 9);
}
