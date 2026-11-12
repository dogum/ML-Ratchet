// ChirpNet live instrument: four real LIGO detections, both detectors' whitened strain, and the released cascade —
// a black-hole finder that proposes and a two-stream coincidence checker that confirms — running in the browser.
// Controls change the input: slide Livingston against Hanford, swap one detector for its own off-source noise,
// push the source farther away (fresh noise, signal shrinks), and move the 2 s window the network reads.
import type { InstrumentContext } from '../types.ts';
import { fetchPack } from '../../lib/nn/pack.ts';
import { read, farPerHour, peContext, peSample, physical, quantiles, chirpFreq, FS, WIN, type CMeta, type CExamples, type Reading } from './net.ts';
import { display, TF_ROWS, TF_FLO, TF_FHI } from './dsp.ts';
import { shell, picker, slider, button, scheduler, gauges, canvas, gridLines, trace, label, heatmap, rng, h, SCOPE, type Canvas2D } from '../kit.ts';

const T0 = -3, T1 = 2;                      // displayed span, seconds from LIGO's merger time
const NS = (T1 - T0) * FS;                  // samples in the span
const SLIDES = [0, 5, 10, 20, 50, 100, 200, 500, 1000];   // L1 delay steps (ms)
const OFF = 4 * FS;                         // "off-source": a detector's own data 4 s earlier
const SCAN = Array.from({ length: 13 }, (_, i) => -2 + i / 4);   // window centres for the live scan (s from merger)
const SWEET = 1.6;                          // default window: merger 1.6 s in, where the finder times it best
const N_PE = 600;
const TF_COLS = 250;

const STYLE = `
.cn-casc{display:flex;flex-direction:column;gap:9px;padding:12px 12px 11px;border:1px solid var(--scr-rule);border-radius:4px;background:rgb(0 0 0 / .22)}
.cn-row{display:grid;grid-template-columns:1fr auto;align-items:baseline;gap:2px 10px}
.cn-row .cn-n{font-family:var(--mono);font-variation-settings:'wdth' 87.5;font-size:15px;color:var(--scr-ink)}
.cn-row small{grid-column:1/-1;font-family:var(--mono);font-variation-settings:'wdth' 87.5;font-size:9.5px;color:var(--scr-3ink);letter-spacing:.02em}
.cn-bar{grid-column:1/-1;position:relative;height:6px;background:var(--scr-rule);border-radius:3px}
.cn-bar span{position:absolute;top:0;bottom:0;left:0;border-radius:3px;background:var(--scr-3ink);transition:width .3s var(--ease)}
.cn-bar.on span{background:var(--c);box-shadow:0 0 8px var(--c)}
.cn-bar i{position:absolute;top:-3px;bottom:-3px;width:0;border-left:1.5px dashed var(--scr-2ink)}
.cn-verdict{display:flex;align-items:baseline;justify-content:space-between;gap:8px;padding-top:9px;border-top:1px solid var(--scr-rule)}
.cn-verdict b{font-family:var(--display);font-weight:400;font-size:25px;line-height:1;color:var(--scr-ink)}
.cn-verdict b.yes{color:var(--c);text-shadow:0 0 18px color-mix(in srgb,var(--c) 55%,transparent)}
.cn-verdict b.veto{color:#ff8a7a}
.cn-verdict span{font-family:var(--mono);font-variation-settings:'wdth' 87.5;font-size:10px;color:var(--scr-3ink);text-align:right;line-height:1.45}
.cn-verdict span em{font-style:normal;color:var(--scr-ink)}
.cn-sub{font-family:var(--mono);font-variation-settings:'wdth' 87.5;font-size:9.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--scr-3ink);margin:2px 0 -6px}
.cn-dets{display:flex;flex-direction:column;gap:6px}
.cn-dets .inst-picker{margin:0}
.cn-dets .inst-pick{padding:5px 9px}
`;

export async function mount(ctx: InstrumentContext) {
  const [W, X] = await Promise.all([fetchPack<CMeta>(`${ctx.base}weights`), fetchPack<CExamples>(`${ctx.base}examples`)]);
  const M = W.meta, EV = X.meta.events;
  if (!document.getElementById('cn-style')) document.head.append(h('style', { id: 'cn-style' }, STYLE));

  const ui = shell(ctx.root, {
    title: 'ChirpNet',
    subtitle: 'two LIGO detectors, one black-hole merger — real strain the network never trained on',
    params: '1.81 M parameters · matches Python to 1e-6',
  });
  const st = { i: 0, slide: 0, far: 1, centre: 1 - SWEET, dets: 0, seed: 3 };
  let streamKey = '';
  let S: { a: number; H: Float32Array; L: Float32Array; bH: Float32Array; bL: Float32Array; tf: Float32Array } | null = null;
  const scan = { finder: new Float32Array(SCAN.length).fill(NaN), stat: new Float32Array(SCAN.length).fill(NaN), veto: new Uint8Array(SCAN.length), done: 0, ms: 0 };
  let cur: { r: Reading; w: number; post: ReturnType<typeof physical> | null } | null = null;

  picker(ui.picker, EV.map((e, i) => ({
    label: e.id,
    sub: `${e.date} · ℳ ${e.truth.mc_det.toFixed(0)} M☉ · ${e.truth.dl} Mpc`,
    value: i,
  })), (i) => { st.i = i; go(); }, st.i);

  const narrow = ctx.root.getBoundingClientRect().width < 640;
  const plot = canvas(ui.screen, narrow ? 0.95 : ctx.compact ? 1.35 : 1.5, (c) => draw(c));

  slider(ui.controls, {
    label: 'Time-slide Livingston', min: 0, max: SLIDES.length - 1, step: 1, value: 0,
    fmt: (v) => (v === 0 ? 'as recorded' : `L1 +${SLIDES[v]} ms`),
    hint: 'a real wave reaches both sites within 10 ms', onInput: (v) => { st.slide = SLIDES[v]; go(); },
  });
  slider(ui.controls, {
    label: 'Source distance', min: 1, max: 3, step: 0.05, value: 1,
    fmt: (v) => (v === 1 ? 'as observed' : `×${v.toFixed(2)} distance`),
    hint: 'fresh noise added; the chirp shrinks by 1/×', onInput: (v) => { st.far = v; go(); },
  });
  slider(ui.controls, {
    label: 'Window position', min: -2, max: 1, step: 1 / 32, value: st.centre,
    fmt: (v) => { const m = 1 - v; return m < 0 ? 'after the merger' : m > 2 ? 'before the merger' : `merger ${m.toFixed(2)} s in`; },
    hint: 'which 2 s of strain the network reads', onInput: (v) => { st.centre = v; go(); },
  });
  if (!ctx.compact) {
    const box = h('div', { class: 'cn-dets' }, h('span', { class: 'inst-sl-top' }, h('span', {}, 'Detectors')));
    const pk = h('div', { class: 'inst-picker', role: 'group', 'aria-label': 'Detectors' });
    box.append(pk);
    ui.controls.append(box);
    picker(pk, [{ label: 'H1 + L1', value: 0 }, { label: 'H1 alone', value: 1 }, { label: 'L1 alone', value: 2 }], (v) => { st.dets = v; go(); }, 0);
    box.append(h('small', { style: 'font-size:11px;color:var(--scr-3ink)' }, 'alone = partner swapped for its own data 4 s earlier'));
    button(ui.controls, 'New noise draw', () => { st.seed++; go(); });
  }

  // readout containers (gauges() clears only its own container)
  const casc = h('div', { class: 'cn-casc' });
  const g1 = h('div', { class: 'inst-readout' });
  const peHead = h('div', { class: 'cn-sub' }, 'posterior flow · 600 samples · 68%');
  const g2 = h('div', { class: 'inst-readout' });
  ui.readout.append(casc, g1);
  if (!ctx.compact) ui.readout.append(peHead, g2);

  // ---------------------------------------------------------------- input streams
  function streams() {
    const key = `${st.i}|${st.slide}|${st.far}|${st.dets}|${st.seed}`;
    if (key === streamKey && S) return false;
    streamKey = key;
    const e = EV[st.i];
    const Hs = X.t[`h${st.i}`].data, Ls = X.t[`l${st.i}`].data;
    const a = Math.round((e.t_ev + T0) * FS);
    const hOff = st.dets === 2 ? OFF : 0, lOff = (st.dets === 1 ? OFF : 0) + Math.round((st.slide / 1000) * FS);
    const sig = Math.sqrt(st.far * st.far - 1), norm = 1 / st.far;
    const r = rng(st.seed * 7919 + st.i * 31 + 1);
    const H = new Float32Array(NS), L = new Float32Array(NS);
    const at = (arr: Float32Array, k: number) => arr[Math.min(arr.length - 1, Math.max(0, k))];
    for (let k = 0; k < NS; k++) {
      H[k] = at(Hs, a + k - hOff);
      L[k] = at(Ls, a + k - lOff);
    }
    if (sig > 0) for (let k = 0; k < NS; k++) { H[k] = (H[k] + sig * r.n()) * norm; L[k] = (L[k] + sig * r.n()) * norm; }
    const d = display([H, L], FS, TF_COLS);
    S = { a, H, L, bH: d.bands[0], bL: d.bands[1], tf: d.tf };
    return true;
  }
  const windowAt = (centre: number) => {
    const w = Math.max(0, Math.min(NS - WIN, Math.round((centre - 1 - T0) * FS)));
    const data = new Float32Array(2 * WIN);
    data.set(S!.H.subarray(w, w + WIN), 0);
    data.set(S!.L.subarray(w, w + WIN), WIN);
    return { w, x: { data, shape: [2, WIN] } };
  };

  // ---------------------------------------------------------------- scan state
  let scanGen = 0;
  async function runScan() {
    const gen = ++scanGen;
    scan.finder.fill(NaN); scan.stat.fill(NaN); scan.veto.fill(0); scan.done = 0;
    let busy = 0;
    for (let k = 0; k < SCAN.length; k++) {
      await new Promise((res) => setTimeout(res, 0));
      if (gen !== scanGen) return;
      const t = performance.now();
      const r = read(W, windowAt(SCAN[k]).x);
      busy += performance.now() - t;
      scan.finder[k] = r.finder[0]; scan.stat[k] = r.stat; scan.veto[k] = r.vetoed ? 1 : 0; scan.done = k + 1;
      if (k % 3 === 2 || k === SCAN.length - 1) plot.redraw();
    }
    scan.ms = busy;
    stamp();
  }

  // ---------------------------------------------------------------- the run
  let lastMs = 0;
  function stamp() {
    const e = EV[st.i];
    const sc = scan.done === SCAN.length ? ` · scan of ${SCAN.length} windows ${(scan.ms / 1000).toFixed(1)} s` : '';
    ui.stamp(lastMs, `${e.id}, GPS ${e.gps.toFixed(2)}, LIGO Hanford + Livingston${sc}`);
  }

  async function run() {
    ui.setBusy(true);
    await new Promise((r) => requestAnimationFrame(r));
    const e = EV[st.i];
    const changed = streams();
    const t0 = performance.now();
    const { w, x } = windowAt(st.centre);
    const r = read(W, x);
    let post: ReturnType<typeof physical> | null = null;
    if (!ctx.compact) {
      const z = new Float32Array(N_PE * 6), g = rng(12345);
      for (let k = 0; k < z.length; k++) z[k] = g.n();
      // as the pipeline does: the posterior network gets a window re-centred on the finder's merger time
      const pw = Math.max(0, Math.min(NS - WIN, Math.round(w + (r.t_m - X.meta.place) * FS)));
      const px = new Float32Array(2 * WIN);
      px.set(S!.H.subarray(pw, pw + WIN), 0); px.set(S!.L.subarray(pw, pw + WIN), WIN);
      post = physical(peSample(W, peContext(W, { data: px, shape: [2, WIN] }, e.asd), z));
    }
    lastMs = performance.now() - t0;
    cur = { r, w, post };
    readouts();
    plot.redraw();
    stamp();
    ui.setBusy(false);
    if (changed) runScan();
  }
  const go = scheduler(run, 40);

  function readouts() {
    if (!cur) return;
    const e = EV[st.i], { r } = cur, T = e.truth;
    const thr = M.threshold;
    const pos = (v: number, a: number, b: number) => `${(Math.max(0, Math.min(1, (v - a) / (b - a))) * 100).toFixed(1)}%`;
    const det = r.stat >= thr;
    const far = farPerHour(M, r.stat);
    const farTxt = det ? (far.bound ? `FAR < ${far.far.toFixed(2)} /h` : `FAR ${far.far.toFixed(1)} /h`) : far.floor ? 'as loud as everyday noise' : `FAR ${far.far.toFixed(0)} /h`;
    const verdict = r.vetoed ? '<b class="veto">Vetoed</b>' : det ? '<b class="yes">Merger</b>' : '<b>Noise</b>';
    const why = r.vetoed ? 'the detectors disagree' : det ? `louder than all ${M.bg_n.toLocaleString('en-US')} time-slid noise windows` : `below the ${thr.toFixed(1)} threshold`;
    casc.innerHTML = `
      <div class="ig-l">The cascade · 2 s window</div>
      <div class="cn-row"><span class="ig-l" style="color:var(--scr-2ink)">1 · finder proposes</span><span class="cn-n">${r.finder[0].toFixed(1)}</span>
        <div class="cn-bar${r.finder[0] > 0 ? ' on' : ''}"><span style="width:${pos(r.finder[0], -10, 15)}"></span><i style="left:${pos(0, -10, 15)}" title="checker runs above 0"></i></div>
        <small>${r.finder[0] > 0 ? 'above 0 → ask the checker' : '≤ 0 → nothing to check'}</small></div>
      <div class="cn-row"><span class="ig-l" style="color:var(--scr-2ink)">2 · checker confirms</span><span class="cn-n">${r.check == null ? '—' : r.check.toFixed(1)}</span>
        <div class="cn-bar${r.check != null && r.check >= M.veto ? ' on' : ''}"><span style="width:${r.check == null ? '0%' : pos(r.check, -8, 10)}"></span><i style="left:${pos(M.veto, -8, 10)}" title="veto level"></i></div>
        <small>${r.check == null ? 'not run' : r.check >= M.veto ? `same chirp in H1 and L1 (veto below ${M.veto.toFixed(0)})` : `below ${M.veto.toFixed(0)}: veto, statistic −${M.drop}`}</small></div>
      <div class="cn-verdict">${verdict}<span>statistic <em>${r.stat.toFixed(1)}</em> vs ${thr.toFixed(1)}<br>${why}<br>${farTxt}</span></div>`;

    const tNet = (S!.a + cur.w) / FS + r.t_m - e.t_ev;
    gauges(g1, [
      { label: 'Chirp mass (detector frame)', unit: 'M☉', value: r.mc, lo: r.mc_lo, hi: r.mc_hi, truth: T.mc_det, truthLabel: 'LIGO GWTC-1', range: [5, 60], digits: 1, log: true },
      { label: 'Merger time vs LIGO', unit: 'ms', value: tNet * 1000, lo: (tNet - r.t_sig) * 1000, hi: (tNet + r.t_sig) * 1000, truth: 0, truthLabel: 'LIGO', range: [-300, 300], digits: 0 },
    ]);
    if (cur.post) {
      const p = cur.post, q = (a: Float32Array) => quantiles(a, [0.16, 0.5, 0.84]);
      const [dl0, dl1, dl2] = q(p.dl), [a0, a1, a2] = q(p.m1), [b0, b1, b2] = q(p.m2);
      gauges(g2, [
        { label: 'Distance', unit: 'Mpc', value: dl1, lo: dl0, hi: dl2, truth: T.dl * st.far, truthLabel: st.far === 1 ? 'LIGO' : `LIGO ×${st.far.toFixed(2)}`, range: [100, 6000], digits: 0, log: true },
        { label: 'Heavier black hole m₁ (source)', unit: 'M☉', value: a1, lo: a0, hi: a2, truth: T.m1, truthLabel: 'LIGO', range: [3, 90], digits: 1, log: true },
        { label: 'Lighter black hole m₂ (source)', unit: 'M☉', value: b1, lo: b0, hi: b2, truth: T.m2, truthLabel: 'LIGO', range: [3, 90], digits: 1, log: true },
      ]);
      peHead.textContent = det ? 'posterior flow · 600 samples · 68%' : 'posterior flow · assumes a merger is there';
    }
  }

  // ---------------------------------------------------------------- drawing
  function draw(c: Canvas2D) {
    const { ctx: g, w, h: H } = c;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, w, H);
    if (!S || !cur) return;
    const r = cur.r;
    const padL = 34, padR = 10, top = 16, bot = 20, iw = w - padL - padR;
    const ih = H - top - bot;
    const small = w < 520;
    const hr = Math.round(ih * 0.135), gap = 8;
    const yH = top, yL = yH + hr, yTF = yL + hr + gap, hTF = Math.round(ih * 0.33), yS = yTF + hTF + gap + 10, hS = top + ih - yS;
    const X = (t: number) => padL + ((t - T0) / (T1 - T0)) * iw;
    const tAt = (k: number) => T0 + k / FS;

    // time-frequency map (H1 + L1 constant-Q energy)
    heatmap(c, S.tf, TF_ROWS, TF_COLS, padL, yTF, iw, hTF, ctx.accent, 0, 7);
    const fY = (f: number) => yTF + hTF - (Math.log(f / TF_FLO) / Math.log(TF_FHI / TF_FLO)) * hTF;
    g.strokeStyle = 'rgba(235,230,216,0.10)'; g.lineWidth = 1;
    for (const f of [32, 64, 128, 256]) { const y = Math.round(fY(f)) + 0.5; g.beginPath(); g.moveTo(padL, y); g.lineTo(padL + iw, y); g.stroke(); label(c, String(f), padL - 4, y, SCOPE.faint, 'right', 9); }
    label(c, 'Hz', padL - 4, yTF + 6, SCOPE.faint, 'right', 9);

    // strain rows
    gridLines(c, padL, yH, iw, 2 * hr, 10, 2);
    trace(c, S.bH, padL, yH, iw, hr, ctx.accent, 0.2, 1);
    trace(c, S.bL, padL, yL, iw, hr, SCOPE.ink, 0.2, 0.8);
    label(c, 'H1', 6, yH + hr / 2, ctx.accent);
    label(c, 'L1', 6, yL + hr / 2, SCOPE.ink);
    const lNote = st.dets === 1 ? (small ? 'L1: noise 4 s earlier' : 'L1: its own noise from 4 s earlier') : st.slide ? `L1 delayed ${st.slide} ms` : '';
    const hNote = st.dets === 2 ? (small ? 'H1: noise 4 s earlier' : 'H1: its own noise from 4 s earlier') : '';
    if (lNote) label(c, lNote, padL + iw - 4, yL + 8, '#ffb4a8', 'right', 9);
    if (hNote) label(c, hNote, padL + iw - 4, yH + 8, '#ffb4a8', 'right', 9);

    // the window the network reads: dim everything outside it
    const w0 = tAt(cur.w), w1 = w0 + WIN / FS;
    g.fillStyle = 'rgba(0,0,0,0.5)';
    g.fillRect(padL, yH, Math.max(0, X(w0) - padL), yTF + hTF - yH);
    g.fillRect(X(w1), yH, Math.max(0, padL + iw - X(w1)), yTF + hTF - yH);
    g.strokeStyle = ctx.accent; g.lineWidth = 1; g.globalAlpha = 0.7;
    g.strokeRect(Math.round(X(w0)) + 0.5, yH - 4.5, Math.round(X(w1) - X(w0)), yTF + hTF - yH + 8);
    g.globalAlpha = 1;
    if (X(0) - X(w0) > 120 || X(0) < X(w0)) label(c, 'network reads 2 s', X(w0) + 4, yH - 9, ctx.accent, 'left', 9);

    // LIGO merger time
    g.strokeStyle = SCOPE.ink; g.globalAlpha = 0.55; g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(X(0) + 0.5, yH); g.lineTo(X(0) + 0.5, yS + hS); g.stroke();
    g.setLineDash([]); g.globalAlpha = 1;
    label(c, 'LIGO merger', X(0) + 4, yH - 9, SCOPE.text, 'left', 9);

    // the network's merger time and chirp: f(t) for its chirp mass, up to roughly the last stable orbit
    const tNet = w0 + r.t_m;
    const live = r.stat >= M.threshold;
    g.fillStyle = ctx.accent; g.globalAlpha = 0.18;
    g.fillRect(X(tNet - r.t_sig), yTF, Math.max(2, X(tNet + r.t_sig) - X(tNet - r.t_sig)), hTF);
    g.globalAlpha = 1;
    const eta = r.q / (1 + r.q) ** 2, Mt = r.mc / eta ** 0.6, fmax = Math.min(TF_FHI, (4400 / Mt) * 1.1);
    g.save();
    g.beginPath(); g.rect(padL, yTF, iw, hTF); g.clip();
    g.strokeStyle = '#ffffff'; g.lineWidth = 2; g.setLineDash([6, 4]); g.globalAlpha = live ? 0.95 : 0.35;
    g.shadowColor = ctx.accent; g.shadowBlur = 8;
    g.beginPath();
    let started = false;
    for (let t = tNet - 3; t < tNet - 0.001; t += 0.002) {
      const f = chirpFreq(r.mc, tNet - t);
      if (f < TF_FLO) continue;
      if (f > fmax) break;
      if (!started) { g.moveTo(X(t), fY(f)); started = true; } else g.lineTo(X(t), fY(f));
    }
    g.stroke();
    g.restore();
    label(c, `network chirp, ℳ ${r.mc.toFixed(1)} M☉`, padL + 6, yTF + 10, live ? '#ffffff' : SCOPE.text, 'left', 9);

    // live scan of the statistic
    const lo = -12, hi = 16, Y = (v: number) => yS + hS - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * hS;
    gridLines(c, padL, yS, iw, hS, 10, 4);
    g.fillStyle = 'rgba(235,230,216,0.05)';
    g.fillRect(padL, Y(M.bg_max), iw, Y(lo) - Y(M.bg_max));
    label(c, small ? 'loudest time-slid noise' : `loudest of ${M.bg_n.toLocaleString('en-US')} time-slid noise windows`, padL + 6, Y(M.bg_max) + 8, SCOPE.faint, 'left', 9);
    g.strokeStyle = ctx.accent; g.globalAlpha = 0.6; g.setLineDash([4, 3]);
    g.beginPath(); g.moveTo(padL, Math.round(Y(M.threshold)) + 0.5); g.lineTo(padL + iw, Math.round(Y(M.threshold)) + 0.5); g.stroke();
    g.setLineDash([]); g.globalAlpha = 1;
    label(c, `threshold ${M.threshold.toFixed(0)}`, padL + iw - 4, Y(M.threshold) + 8, ctx.accent, 'right', 9);
    for (const v of [-8, 0, 8]) label(c, String(v), padL - 4, Y(v), SCOPE.faint, 'right', 9);
    if (!small || scan.done === SCAN.length) label(c, 'statistic', padL + 6, yS + 7, SCOPE.text, 'left', 9);
    const line = (arr: Float32Array, color: string, width: number, alpha: number) => {
      g.strokeStyle = color; g.lineWidth = width; g.globalAlpha = alpha; g.beginPath();
      let on = false;
      for (let k = 0; k < SCAN.length; k++) {
        if (!Number.isFinite(arr[k])) { on = false; continue; }
        const x = X(SCAN[k]), y = Y(arr[k]);
        if (on) g.lineTo(x, y); else { g.moveTo(x, y); on = true; }
      }
      g.stroke(); g.globalAlpha = 1; g.lineWidth = 1;
    };
    line(scan.finder, SCOPE.ink, 1, 0.35);
    line(scan.stat, ctx.accent, 2, 1);
    for (let k = 0; k < scan.done; k++) {
      if (!scan.veto[k]) continue;
      const x = X(SCAN[k]), y = Y(lo) - 5;
      g.strokeStyle = '#ff8a7a'; g.lineWidth = 1.5; g.beginPath(); g.moveTo(x - 3, y - 3); g.lineTo(x + 3, y + 3); g.moveTo(x + 3, y - 3); g.lineTo(x - 3, y + 3); g.stroke(); g.lineWidth = 1;
    }
    if (scan.veto.some((v) => v)) label(c, small ? '× vetoed (−30)' : '× vetoed by the checker (−30)', padL + iw - 4, Y(lo) - 14, '#ff8a7a', 'right', 9);
    if (scan.done < SCAN.length) label(c, `scanning ${scan.done}/${SCAN.length}…`, padL + iw - 4, yS + 7, ctx.accent, 'right', 9);
    else if (!small) label(c, 'grey: finder alone · colour: after the checker', padL + iw - 4, yS + 7, SCOPE.faint, 'right', 9);
    // the current window on the scan
    const xc = X(w0 + 1);
    g.strokeStyle = ctx.accent; g.globalAlpha = 0.5; g.beginPath(); g.moveTo(xc + 0.5, yS); g.lineTo(xc + 0.5, yS + hS); g.stroke(); g.globalAlpha = 1;
    g.fillStyle = ctx.accent; g.shadowColor = ctx.accent; g.shadowBlur = 10;
    g.beginPath(); g.arc(xc, Y(r.stat), 4, 0, 2 * Math.PI); g.fill(); g.shadowBlur = 0;

    // time axis
    for (let t = T0; t <= T1; t += small ? 1 : 0.5) {
      const txt = t === 0 ? '0' : `${t > 0 ? '+' : '−'}${Math.abs(t).toFixed(1)}${t === T1 ? ' s' : ''}`;
      label(c, txt, X(t), H - 9, SCOPE.faint, t === T1 ? 'right' : t === T0 ? 'left' : 'center', 9);
    }
  }

  await run();
}
