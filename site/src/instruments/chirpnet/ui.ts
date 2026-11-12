// ChirpNet live instrument: five real LIGO detections and both detectors' whitened strain.
// Black holes (GW150914 … GW170104): the released cascade — a finder that proposes and a two-stream coincidence
// checker that confirms — plus the posterior flow. Neutron stars (GW170817): ChirpNetBNS, a time-frequency 2-D CNN
// that reads 16 s at a time and has no coincidence checker.
// Controls change the input: slide Livingston against Hanford, swap one detector for its own off-source data,
// push the source farther away (fresh noise, signal shrinks), move the window the network reads, and for GW170817
// put Livingston's famous glitch back in and switch the 25-sigma auto-gating on or off.
import type { InstrumentContext } from '../types.ts';
import { fetchPack, type Pack } from '../../lib/nn/pack.ts';
import { read, farPerHour, peContext, peSample, physical, quantiles, chirpFreq, type CMeta, type CExamples } from './net.ts';
import { bnsRead, bnsFar, tfMap, gate, type BMeta, type BEvent } from './bns.ts';
import { toRd, type Rd } from './reading.ts';
import { display, TF_ROWS, TF_FLO, TF_FHI } from './dsp.ts';
import { shell, picker, slider, button, scheduler, gauges, canvas, gridLines, trace, label, heatmap, rng, h, SCOPE, type Canvas2D } from '../kit.ts';

type Meta = CMeta & { bns: BMeta };
type Ex = CExamples & { bns: BEvent };

interface Mode {
  bns: boolean; fs: number; win: number; T0: number; T1: number;
  slides: number[]; scan: number[]; centre: number; cmin: number; cmax: number; cstep: number;
  threshold: number; bgMax: number; bgN: number; bgHours: number;
  flo: number; fhi: number; tick: number; ylo: number; yhi: number; yticks: number[];
}

const N_PE = 600;
const TF_COLS = 250;
const RED = '#ff8a7a', PINK = '#ffb4a8';

const STYLE = `
.cn-casc{display:flex;flex-direction:column;gap:9px;padding:12px 12px 11px;border:1px solid var(--scr-rule);border-radius:4px;background:rgb(0 0 0 / .22)}
.cn-row{display:grid;grid-template-columns:1fr auto;align-items:baseline;gap:2px 10px}
.cn-row .cn-n{font-family:var(--mono);font-variation-settings:'wdth' 87.5;font-size:15px;color:var(--scr-ink)}
.cn-row small{grid-column:1/-1;font-family:var(--mono);font-variation-settings:'wdth' 87.5;font-size:9.5px;color:var(--scr-3ink);letter-spacing:.02em}
.cn-row.off .cn-n,.cn-row.off .ig-l{color:var(--scr-3ink)!important}
.cn-bar{grid-column:1/-1;position:relative;height:6px;background:var(--scr-rule);border-radius:3px}
.cn-bar span{position:absolute;top:0;bottom:0;left:0;border-radius:3px;background:var(--scr-3ink);transition:width .3s var(--ease)}
.cn-bar.on span{background:var(--c);box-shadow:0 0 8px var(--c)}
.cn-bar i{position:absolute;top:-3px;bottom:-3px;width:0;border-left:1.5px dashed var(--scr-2ink)}
.cn-verdict{display:flex;align-items:baseline;justify-content:space-between;gap:8px;padding-top:9px;border-top:1px solid var(--scr-rule)}
.cn-verdict b{font-family:var(--display);font-weight:400;font-size:25px;line-height:1;color:var(--scr-ink)}
.cn-verdict b.yes{color:var(--c);text-shadow:0 0 18px color-mix(in srgb,var(--c) 55%,transparent)}
.cn-verdict b.veto{color:${RED}}
.cn-verdict span{font-family:var(--mono);font-variation-settings:'wdth' 87.5;font-size:10px;color:var(--scr-3ink);text-align:right;line-height:1.45}
.cn-verdict span em{font-style:normal;color:var(--scr-ink)}
.cn-sub{font-family:var(--mono);font-variation-settings:'wdth' 87.5;font-size:9.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--scr-3ink);margin:2px 0 -6px}
.cn-note{font-family:var(--mono);font-variation-settings:'wdth' 87.5;font-size:10px;line-height:1.5;color:var(--scr-3ink);border-left:2px solid var(--scr-rule);padding-left:9px}
.cn-note b{font-weight:400;color:var(--scr-2ink)}
.cn-dets{display:flex;flex-direction:column;gap:6px}
.cn-dets .inst-picker{margin:0}
.cn-dets .inst-pick{padding:5px 9px}
.cn-dets small{font-size:11px;color:var(--scr-3ink)}
`;

export async function mount(ctx: InstrumentContext) {
  const [W, X] = await Promise.all([fetchPack<Meta>(`${ctx.base}weights`), fetchPack<Ex>(`${ctx.base}examples`)]);
  const M = W.meta, EV = X.meta.events, BE = X.meta.bns, BM = M.bns;
  if (!document.getElementById('cn-style')) document.head.append(h('style', { id: 'cn-style' }, STYLE));

  const BBH: Mode = {
    bns: false, fs: 2048, win: 4096, T0: -3, T1: 2,
    slides: [0, 5, 10, 20, 50, 100, 200, 500, 1000],
    scan: Array.from({ length: 13 }, (_, i) => -2 + i / 4), centre: 1 - 1.6, cmin: -2, cmax: 1, cstep: 1 / 32,
    threshold: M.threshold, bgMax: M.bg_max, bgN: M.bg_n, bgHours: M.bg_live_s / 3600,
    flo: TF_FLO, fhi: TF_FHI, tick: 0.5, ylo: -12, yhi: 16, yticks: [-8, 0, 8],
  };
  const BNS: Mode = {
    bns: true, fs: BE.fs, win: BE.win, T0: -30, T1: 12,
    slides: [0, 10, 50, 100, 250, 500, 1000, 2000, 4000],
    scan: Array.from({ length: 27 }, (_, i) => -22 + i), centre: 8 - BE.place, cmin: -22, cmax: 4, cstep: 0.25,
    threshold: BM.threshold, bgMax: BM.bg_max, bgN: BM.bg_n, bgHours: BM.bg_live_s / 3600,
    flo: 24, fhi: 450, tick: 5, ylo: -6, yhi: 16, yticks: [0, 5, 10, 15],
  };
  const half = (m: Mode) => m.win / m.fs / 2;

  const ui = shell(ctx.root, {
    title: 'ChirpNet',
    subtitle: 'two LIGO detectors, one merger — real strain the networks never trained on',
    params: '2.11 M parameters in 4 networks · matches Python to 1e-6',
  });
  const st = { i: 0, slide: 0, far: 1, centre: BBH.centre, dets: 0, seed: 3, raw: false, gating: true };
  let mode = BBH;
  const isBns = () => st.i === EV.length;

  // ---------------------------------------------------------------- state shared with drawing
  let streamKey = '';
  let S: { a: number; NS: number; H: Float32Array; L: Float32Array; bH: Float32Array; bL: Float32Array; tf: Float32Array; rows: number; cols: number; gates: number } | null = null;
  const scan = { finder: new Float32Array(0), stat: new Float32Array(0), veto: new Uint8Array(0), done: 0, ms: 0, worker: false };
  let cur: { r: Rd; w: number; post: ReturnType<typeof physical> | null } | null = null;

  const events = [
    ...EV.map((e, i) => ({ label: e.id, sub: `${e.date} · ℳ ${e.truth.mc_det.toFixed(0)} M☉ · ${e.truth.dl} Mpc`, value: i })),
    { label: `${BE.id} · neutron stars`, sub: `${BE.date} · ℳ ${BE.truth.mc_det.toFixed(3)} M☉ · ${BE.truth.dl} Mpc`, value: EV.length },
  ];
  picker(ui.picker, events, (i) => { st.i = i; setMode(); go(); }, st.i);

  const narrow = ctx.root.getBoundingClientRect().width < 640;
  const plot = canvas(ui.screen, narrow ? 0.95 : ctx.compact ? 1.35 : 1.5, (c) => draw(c));

  const sSlide = slider(ui.controls, {
    label: 'Time-slide Livingston', min: 0, max: 8, step: 1, value: 0,
    fmt: (v) => { const ms = mode.slides[v]; return v === 0 ? 'as recorded' : ms >= 1000 ? `L1 +${ms / 1000} s` : `L1 +${ms} ms`; },
    hint: 'a real wave reaches both sites within 10 ms', onInput: (v) => { st.slide = v; go(); },
  });
  slider(ui.controls, {
    label: 'Source distance', min: 1, max: 3, step: 0.05, value: 1,
    fmt: (v) => (v === 1 ? 'as observed' : `×${v.toFixed(2)} distance`),
    hint: 'fresh noise added; the chirp shrinks by 1/×', onInput: (v) => { st.far = v; go(); },
  });
  const sWin = slider(ui.controls, {
    label: 'Window position', min: BBH.cmin, max: BBH.cmax, step: BBH.cstep, value: st.centre,
    fmt: (v) => { const hh = half(mode), m = hh - v; return m < 0 ? 'after the merger' : m > 2 * hh ? 'before the merger' : `merger ${m.toFixed(mode.bns ? 1 : 2)} s in`; },
    hint: 'which 2 s of strain the network reads', onInput: (v) => { st.centre = v; go(); },
  });
  const winHint = sWin.input.parentElement!.querySelector('small');

  let glitchBox: HTMLElement | null = null;
  let setDetHint = (_t: string) => {};
  if (!ctx.compact) {
    const box = h('div', { class: 'cn-dets' }, h('span', { class: 'inst-sl-top' }, h('span', {}, 'Detectors')));
    const pk = h('div', { class: 'inst-picker', role: 'group', 'aria-label': 'Detectors' });
    box.append(pk);
    ui.controls.append(box);
    picker(pk, [{ label: 'H1 + L1', value: 0 }, { label: 'H1 alone', value: 1 }, { label: 'L1 alone', value: 2 }], (v) => { st.dets = v; go(); }, 0);
    const detHint = h('small', {}, 'alone = partner swapped for its own data 4 s earlier');
    box.append(detHint);
    glitchBox = h('div', { class: 'cn-dets' }, h('span', { class: 'inst-sl-top' }, h('span', {}, 'Livingston data · gating')));
    const pg = h('div', { class: 'inst-picker', role: 'group', 'aria-label': 'Livingston data' });
    const pg2 = h('div', { class: 'inst-picker', role: 'group', 'aria-label': 'Auto-gating' });
    glitchBox.append(pg, pg2, h('small', {}, 'raw L1 holds a loud glitch 1.1 s before the merger; gating zeroes > 25σ'));
    picker(pg, [{ label: 'cleaned', value: false }, { label: 'raw, with glitch', value: true }], (v) => { st.raw = v; go(); }, 0);
    picker(pg2, [{ label: 'gating on', value: true }, { label: 'off', value: false }], (v) => { st.gating = v; go(); }, 0);
    ui.controls.append(glitchBox);
    button(ui.controls, 'New noise draw', () => { st.seed++; go(); });
    setDetHint = (t) => { detHint.textContent = t; };
  }

  // readout containers (gauges() clears only its own container)
  const casc = h('div', { class: 'cn-casc' });
  const g1 = h('div', { class: 'inst-readout' });
  const peHead = h('div', { class: 'cn-sub' }, 'posterior flow · 600 samples · 68%');
  const g2 = h('div', { class: 'inst-readout' });
  const note = h('div', { class: 'cn-note' });
  ui.readout.append(casc, g1);
  if (!ctx.compact) ui.readout.append(peHead, g2, note);

  function setMode() {
    const m = isBns() ? BNS : BBH;
    if (m !== mode) {
      mode = m;
      st.centre = m.centre;
      sWin.input.min = String(m.cmin); sWin.input.max = String(m.cmax); sWin.input.step = String(m.cstep);
      sWin.set(st.centre);
      sSlide.set(st.slide);
      if (winHint) winHint.textContent = m.bns ? 'which 16 s of strain the network reads' : 'which 2 s of strain the network reads';
      scan.finder = new Float32Array(m.scan.length).fill(NaN);
      scan.stat = new Float32Array(m.scan.length).fill(NaN);
      scan.veto = new Uint8Array(m.scan.length);
    }
    if (glitchBox) glitchBox.style.display = m.bns ? '' : 'none';
    peHead.style.display = g2.style.display = m.bns ? 'none' : '';
    note.style.display = m.bns ? '' : 'none';
    setDetHint(m.bns ? `alone = partner swapped for its own data ${BE.off} s earlier` : 'alone = partner swapped for its own data 4 s earlier');
  }
  setMode();
  scan.finder = new Float32Array(mode.scan.length).fill(NaN);
  scan.stat = new Float32Array(mode.scan.length).fill(NaN);
  scan.veto = new Uint8Array(mode.scan.length);

  // ---------------------------------------------------------------- input streams
  function streams() {
    const key = `${st.i}|${st.slide}|${st.far}|${st.dets}|${st.seed}|${st.raw}|${st.gating}`;
    if (key === streamKey && S) return false;
    streamKey = key;
    const m = mode, fs = m.fs, NS = (m.T1 - m.T0) * fs;
    const slideS = Math.round((m.slides[st.slide] / 1000) * fs);
    let Hs: Float32Array, Ls: Float32Array, a: number, hOff = 0, lOff = slideS;
    if (m.bns) {
      Hs = X.t[st.dets === 2 ? 'n_Hoff' : 'n_H'].data;
      Ls = X.t[st.dets === 1 ? 'n_Loff' : st.raw ? 'n_Lr' : 'n_L'].data;
      a = Math.round((BE.t_ev + m.T0) * fs);
    } else {
      const e = EV[st.i];
      Hs = X.t[`h${st.i}`].data; Ls = X.t[`l${st.i}`].data;
      a = Math.round((e.t_ev + m.T0) * fs);
      if (st.dets === 2) hOff = 4 * fs;
      if (st.dets === 1) lOff += 4 * fs;
    }
    const sig = Math.sqrt(st.far * st.far - 1), norm = 1 / st.far;
    const r = rng(st.seed * 7919 + st.i * 31 + 1);
    let H: Float32Array = new Float32Array(NS), L: Float32Array = new Float32Array(NS);
    const at = (arr: Float32Array, k: number) => arr[Math.min(arr.length - 1, Math.max(0, k))];
    for (let k = 0; k < NS; k++) { H[k] = at(Hs, a + k - hOff); L[k] = at(Ls, a + k - lOff); }
    if (sig > 0) for (let k = 0; k < NS; k++) { H[k] = (H[k] + sig * r.n()) * norm; L[k] = (L[k] + sig * r.n()) * norm; }
    let gates = 0;
    if (m.bns && st.gating) { const gh = gate(H), gl = gate(L); H = gh.x; L = gl.x; gates = gh.gates.length + gl.gates.length; }
    if (m.bns) {
      // the network's own front end over the whole span (64 log-spaced bands, 1/16 s frames): H1 + L1 band power,
      // smoothed over 6 frames (0.375 s) for the eye and divided by each band's median
      const d = display([H, L], fs, 0, false);
      const cols = Math.floor(NS / BM.hop), A = tfMap(W, H, cols), B = tfMap(W, L, cols), rows = A.length / cols;
      const tf = new Float32Array(A.length), row = new Float32Array(cols), sm = 6;
      for (let q = 0; q < rows; q++) {
        for (let c = 0; c < cols; c++) row[c] = 0.5 * (Math.expm1(A[q * cols + c]) + Math.expm1(B[q * cols + c]));
        let acc = 0;
        const out = tf.subarray(q * cols, (q + 1) * cols);
        for (let c = 0; c < cols + sm; c++) {
          if (c < cols) acc += row[c];
          if (c >= sm) acc -= row[c - sm];
          const j = c - (sm >> 1);
          if (j >= 0 && j < cols) out[j] = acc / Math.min(sm, c + 1, cols + sm - c);
        }
        const med = Float32Array.from(out).sort()[cols >> 1] || 1;
        for (let c = 0; c < cols; c++) out[c] /= med;
      }
      S = { a, NS, H, L, bH: d.bands[0], bL: d.bands[1], tf, rows, cols, gates };
    } else {
      const d = display([H, L], fs, TF_COLS);
      S = { a, NS, H, L, bH: d.bands[0], bL: d.bands[1], tf: d.tf, rows: TF_ROWS, cols: TF_COLS, gates };
    }
    return true;
  }
  const startAt = (centre: number) => Math.max(0, Math.min(S!.NS - mode.win, Math.round((centre - half(mode) - mode.T0) * mode.fs)));
  const windowAt = (centre: number) => {
    const w = startAt(centre), win = mode.win;
    const data = new Float32Array(2 * win);
    data.set(S!.H.subarray(w, w + win), 0);
    data.set(S!.L.subarray(w, w + win), win);
    return { w, x: { data, shape: [2, win] } };
  };
  const readWin = (x: { data: Float32Array; shape: number[] }): Rd => (mode.bns ? toRd(bnsRead(W, x)) : toRd(read(W, x)));

  // ---------------------------------------------------------------- the live scan: in a worker when possible
  let scanGen = 0;
  let worker: Worker | null = null, workerReady = false, markUp = () => {};
  const workerUp = new Promise<void>((r) => { markUp = r; });
  try {
    worker = new Worker(new URL('./scan.worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = (ev) => {
      const d = ev.data;
      if (d.type === 'ready') { workerReady = true; markUp(); return; }
      if (d.type === 'error') { console.warn('chirpnet scan worker:', d.message); worker = null; markUp(); if (scan.done < scan.stat.length) { streamKey = ''; go(); } return; }
      if (d.gen !== scanGen) return;
      if (d.type === 'pt') {
        const r = d.r as Rd;
        scan.finder[d.k] = r.finder; scan.stat[d.k] = r.stat; scan.veto[d.k] = r.vetoed ? 1 : 0; scan.done = Math.max(scan.done, d.k + 1);
        if (d.k % 3 === 2 || d.k === scan.stat.length - 1) plot.redraw();
      } else if (d.type === 'done') { scan.ms = d.ms; scan.worker = true; stamp(); plot.redraw(); }
    };
    worker.onerror = () => { worker = null; workerReady = false; markUp(); };
    worker.postMessage({ type: 'init', base: new URL(ctx.base, location.href).href });
  } catch { worker = null; }

  async function runScan() {
    const gen = ++scanGen, m = mode;
    scan.finder.fill(NaN); scan.stat.fill(NaN); scan.veto.fill(0); scan.done = 0; scan.ms = 0;
    const starts = m.scan.map(startAt);
    if (worker && !workerReady) await Promise.race([workerUp, new Promise((r) => setTimeout(r, 4000))]);
    if (gen !== scanGen) return;
    if (worker && workerReady) {
      worker.postMessage({ type: 'scan', gen, mode: m.bns ? 'bns' : 'bbh', H: S!.H, L: S!.L, starts, win: m.win });
      return;
    }
    let busy = 0;
    for (let k = 0; k < starts.length; k++) {
      await new Promise((res) => setTimeout(res, 0));
      if (gen !== scanGen || m !== mode) return;
      const t = performance.now();
      const r = readWin(windowAt(m.scan[k]).x);
      busy += performance.now() - t;
      scan.finder[k] = r.finder; scan.stat[k] = r.stat; scan.veto[k] = r.vetoed ? 1 : 0; scan.done = k + 1;
      if (k % 3 === 2 || k === starts.length - 1) plot.redraw();
    }
    scan.ms = busy; scan.worker = false;
    stamp();
  }

  // ---------------------------------------------------------------- the run
  let lastMs = 0;
  function stamp() {
    const n = mode.scan.length;
    const sc = scan.done === n && scan.ms > 0 ? ` · scan of ${n} windows ${(scan.ms / 1000).toFixed(1)} s${scan.worker ? ' (worker)' : ''}` : '';
    const id = mode.bns ? `${BE.id}, GPS ${BE.gps.toFixed(2)}, ChirpNetBNS` : `${EV[st.i].id}, GPS ${EV[st.i].gps.toFixed(2)}`;
    ui.stamp(lastMs, `${id}, LIGO Hanford + Livingston${sc}`);
  }

  async function run() {
    ui.setBusy(true);
    await new Promise((r) => requestAnimationFrame(r));
    const changed = streams();
    const t0 = performance.now();
    const { w, x } = windowAt(st.centre);
    const r = readWin(x);
    let post: ReturnType<typeof physical> | null = null;
    if (!ctx.compact && !mode.bns) {
      const e = EV[st.i];
      const z = new Float32Array(N_PE * 6), g = rng(12345);
      for (let k = 0; k < z.length; k++) z[k] = g.n();
      // as the pipeline does: the posterior network gets a window re-centred on the finder's merger time
      const pw = Math.max(0, Math.min(S!.NS - mode.win, Math.round(w + (r.t_m - X.meta.place) * mode.fs)));
      const px = new Float32Array(2 * mode.win);
      px.set(S!.H.subarray(pw, pw + mode.win), 0); px.set(S!.L.subarray(pw, pw + mode.win), mode.win);
      post = physical(peSample(W, peContext(W, { data: px, shape: [2, mode.win] }, e.asd), z));
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

  // ---------------------------------------------------------------- readouts
  const pos = (v: number, a: number, b: number) => `${(Math.max(0, Math.min(1, (v - a) / (b - a))) * 100).toFixed(1)}%`;
  function readouts() {
    if (!cur) return;
    const { r } = cur, m = mode, thr = m.threshold;
    const det = r.stat >= thr;
    const far = m.bns ? bnsFar(BM, r.stat) : farPerHour(M, r.stat);
    const farTxt = det ? (far.bound ? `FAR < ${far.far.toFixed(2)} /h` : `FAR ${far.far.toFixed(1)} /h`) : far.floor ? 'as loud as everyday noise' : `FAR ${far.far.toFixed(0)} /h`;
    const verdict = r.vetoed ? '<b class="veto">Vetoed</b>' : det ? '<b class="yes">Merger</b>' : '<b>Noise</b>';
    const louder = `louder than all ${m.bgN.toLocaleString('en-US')} time-slid noise windows (${m.bgHours.toFixed(1)} h)`;
    const why = r.vetoed ? 'the detectors disagree' : det ? louder : `below the ${thr.toFixed(1)} threshold`;
    const tNet = (startAt(st.centre) / m.fs + m.T0) + r.t_m;
    if (m.bns) {
      casc.innerHTML = `
        <div class="ig-l">ChirpNetBNS · 16 s window</div>
        <div class="cn-row"><span class="ig-l" style="color:var(--scr-2ink)">1 · time-frequency CNN</span><span class="cn-n">${r.stat.toFixed(1)}</span>
          <div class="cn-bar${det ? ' on' : ''}"><span style="width:${pos(r.stat, -5, 16)}"></span><i style="left:${pos(thr, -5, 16)}" title="threshold"></i></div>
          <small>64 bands × 256 frames per detector → 2-D residual CNN</small></div>
        <div class="cn-row off"><span class="ig-l">2 · coincidence checker</span><span class="cn-n">none</span>
          <small>not built for neutron stars: nothing asks whether H1 and L1 agree</small></div>
        <div class="cn-verdict">${verdict}<span>statistic <em>${r.stat.toFixed(1)}</em> vs ${thr.toFixed(1)}<br>${why}<br>${farTxt}</span></div>`;
      gauges(g1, [
        { label: 'Chirp mass (detector frame)', unit: 'M☉', value: r.mc, lo: r.mc_lo, hi: r.mc_hi, truth: BE.truth.mc_det, truthLabel: 'LIGO GWTC-1', range: [0.9, 1.9], digits: 3 },
        { label: 'Merger time vs LIGO', unit: 's', value: tNet, lo: tNet - r.t_sig, hi: tNet + r.t_sig, truth: 0, truthLabel: 'LIGO', range: [-4, 4], digits: 2 },
      ]);
      const gTxt = S!.gates ? `${S!.gates} gate${S!.gates > 1 ? 's' : ''} applied` : st.gating ? 'nothing to gate' : 'gating off';
      note.innerHTML = `<b>Released rung scores GW170817 14.4 and reads ℳ 1.1365 M☉</b> (Python, 0.25 s scan): 5% light, its 68% bar misses LIGO's 1.1975. `
        + `Raw L1: 12.0 gated, 9.8 ungated. Here: ${gTxt}.`;
    } else {
      casc.innerHTML = `
        <div class="ig-l">The cascade · 2 s window</div>
        <div class="cn-row"><span class="ig-l" style="color:var(--scr-2ink)">1 · finder proposes</span><span class="cn-n">${r.finder.toFixed(1)}</span>
          <div class="cn-bar${r.finder > 0 ? ' on' : ''}"><span style="width:${pos(r.finder, -10, 15)}"></span><i style="left:${pos(0, -10, 15)}" title="checker runs above 0"></i></div>
          <small>${r.finder > 0 ? 'above 0 → ask the checker' : '≤ 0 → nothing to check'}</small></div>
        <div class="cn-row"><span class="ig-l" style="color:var(--scr-2ink)">2 · checker confirms</span><span class="cn-n">${r.check == null ? '—' : r.check.toFixed(1)}</span>
          <div class="cn-bar${r.check != null && r.check >= M.veto ? ' on' : ''}"><span style="width:${r.check == null ? '0%' : pos(r.check, -8, 10)}"></span><i style="left:${pos(M.veto, -8, 10)}" title="veto level"></i></div>
          <small>${r.check == null ? 'not run' : r.check >= M.veto ? `same chirp in H1 and L1 (veto below ${M.veto.toFixed(0)})` : `below ${M.veto.toFixed(0)}: veto, statistic −${M.drop}`}</small></div>
        <div class="cn-verdict">${verdict}<span>statistic <em>${r.stat.toFixed(1)}</em> vs ${thr.toFixed(1)}<br>${why}<br>${farTxt}</span></div>`;
      const T = EV[st.i].truth;
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
  }

  // ---------------------------------------------------------------- drawing
  function draw(c: Canvas2D) {
    const { ctx: g, w, h: H } = c;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, w, H);
    if (!S || !cur) return;
    const m = mode, r = cur.r;
    const padL = 34, padR = 10, top = 16, bot = 20, iw = w - padL - padR;
    const ih = H - top - bot;
    const small = w < 520;
    const hr = Math.round(ih * (m.bns ? 0.11 : 0.135)), gap = 8;
    const yH = top, yL = yH + hr, yTF = yL + hr + gap, hTF = Math.round(ih * (m.bns ? 0.39 : 0.33)), yS = yTF + hTF + gap + 10, hS = top + ih - yS;
    const X = (t: number) => padL + ((t - m.T0) / (m.T1 - m.T0)) * iw;
    const fY = (f: number) => yTF + hTF - (Math.log(f / m.flo) / Math.log(m.fhi / m.flo)) * hTF;

    // time-frequency map: H1 + L1 (black holes: constant-Q energy; neutron stars: the network's own front end)
    heatmap(c, S.tf, S.rows, S.cols, padL, yTF, iw, hTF, ctx.accent, m.bns ? 0.9 : 0, m.bns ? 3.4 : 7);
    g.strokeStyle = 'rgba(235,230,216,0.10)'; g.lineWidth = 1;
    for (const f of [32, 64, 128, 256]) { const y = Math.round(fY(f)) + 0.5; g.beginPath(); g.moveTo(padL, y); g.lineTo(padL + iw, y); g.stroke(); label(c, String(f), padL - 4, y, SCOPE.faint, 'right', 9); }
    label(c, 'Hz', padL - 4, yTF + 6, SCOPE.faint, 'right', 9);

    // strain rows (band-passed 30-350 Hz for the eye)
    gridLines(c, padL, yH, iw, 2 * hr, 10, 2);
    trace(c, S.bH, padL, yH, iw, hr, ctx.accent, 0.2, 1);
    trace(c, S.bL, padL, yL, iw, hr, SCOPE.ink, 0.2, 0.8);
    label(c, 'H1', 6, yH + hr / 2, ctx.accent);
    label(c, 'L1', 6, yL + hr / 2, SCOPE.ink);
    const offTxt = m.bns ? `${BE.off} s` : '4 s';
    const ms = m.slides[st.slide];
    const lNote = st.dets === 1 ? (small ? `L1: data ${offTxt} earlier` : `L1: its own data from ${offTxt} earlier`)
      : [st.slide ? `L1 delayed ${ms >= 1000 ? `${ms / 1000} s` : `${ms} ms`}` : '', m.bns && st.raw ? 'raw (glitch)' : ''].filter(Boolean).join(' · ');
    const hNote = st.dets === 2 ? (small ? `H1: data ${offTxt} earlier` : `H1: its own data from ${offTxt} earlier`) : '';
    if (lNote) label(c, lNote, padL + iw - 4, yL + 8, PINK, 'right', 9);
    if (hNote) label(c, hNote, padL + iw - 4, yH + 8, PINK, 'right', 9);

    // the window the network reads: dim everything outside it
    const w0 = cur.w / m.fs + m.T0, w1 = w0 + m.win / m.fs;
    g.fillStyle = 'rgba(0,0,0,0.5)';
    g.fillRect(padL, yH, Math.max(0, X(w0) - padL), yTF + hTF - yH);
    g.fillRect(X(w1), yH, Math.max(0, padL + iw - X(w1)), yTF + hTF - yH);
    g.strokeStyle = ctx.accent; g.lineWidth = 1; g.globalAlpha = 0.7;
    g.strokeRect(Math.round(X(w0)) + 0.5, yH - 4.5, Math.round(X(w1) - X(w0)), yTF + hTF - yH + 8);
    g.globalAlpha = 1;
    const wl = m.bns ? 'network reads 16 s' : 'network reads 2 s';
    if (X(0) - X(w0) > 120 || X(0) < X(w0)) label(c, wl, X(w0) + 4, yH - 9, ctx.accent, 'left', 9);

    // LIGO merger time
    g.strokeStyle = SCOPE.ink; g.globalAlpha = 0.55; g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(X(0) + 0.5, yH); g.lineTo(X(0) + 0.5, yS + hS); g.stroke();
    g.setLineDash([]); g.globalAlpha = 1;
    label(c, 'LIGO merger', X(0) + 4, yH - 9, SCOPE.text, 'left', 9);

    // chirp tracks f(t) (leading order): the network's reading, and for GW170817 LIGO's too
    const tNet = w0 + r.t_m;
    const live = r.stat >= m.threshold;
    g.fillStyle = ctx.accent; g.globalAlpha = 0.18;
    g.fillRect(X(tNet - r.t_sig), yTF, Math.max(2, X(tNet + r.t_sig) - X(tNet - r.t_sig)), hTF);
    g.globalAlpha = 1;
    const track = (mc: number, tc: number, fmax: number, color: string, alpha: number, glow: boolean, dash: number[], width = 2) => {
      g.save();
      g.beginPath(); g.rect(padL, yTF, iw, hTF); g.clip();
      g.strokeStyle = color; g.lineWidth = width; g.setLineDash(dash); g.globalAlpha = alpha;
      if (glow) { g.shadowColor = ctx.accent; g.shadowBlur = 8; }
      g.beginPath();
      let on = false;
      const dt = (m.T1 - m.T0) / 1500;
      for (let t = m.T0; t < tc - 0.001; t += dt) {
        const f = chirpFreq(mc, tc - t);
        if (f < m.flo) continue;
        if (f > fmax) break;
        if (!on) { g.moveTo(X(t), fY(f)); on = true; } else g.lineTo(X(t), fY(f));
      }
      g.stroke();
      g.restore();
    };
    if (m.bns) {
      // thin and sparse, so the chirp in the data stays visible underneath
      track(BE.truth.mc_det, 0, m.fhi, SCOPE.ink, 0.55, false, [1.5, 6], 1.2);
      track(r.mc, tNet, m.fhi, '#ffffff', live ? 0.7 : 0.3, false, [5, 7], 1.2);
      label(c, `network ℳ ${r.mc.toFixed(3)}`, padL + 6, yTF + 10, live ? '#ffffff' : SCOPE.text, 'left', 9);
      label(c, `LIGO ℳ ${BE.truth.mc_det.toFixed(4)} · dotted`, padL + 6, yTF + 23, SCOPE.text, 'left', 9);
    } else {
      const eta = r.q / (1 + r.q) ** 2, Mt = r.mc / eta ** 0.6;
      track(r.mc, tNet, Math.min(m.fhi, (4400 / Mt) * 1.1), '#ffffff', live ? 0.95 : 0.35, true, [6, 4]);
      label(c, `network chirp, ℳ ${r.mc.toFixed(1)} M☉`, padL + 6, yTF + 10, live ? '#ffffff' : SCOPE.text, 'left', 9);
    }

    // live scan of the statistic
    const lo = m.ylo, hi = m.yhi, Y = (v: number) => yS + hS - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * hS;
    gridLines(c, padL, yS, iw, hS, 10, 4);
    g.fillStyle = 'rgba(235,230,216,0.05)';
    g.fillRect(padL, Y(m.bgMax), iw, Y(lo) - Y(m.bgMax));
    label(c, small ? 'loudest time-slid noise' : `loudest of ${m.bgN.toLocaleString('en-US')} time-slid noise windows`, padL + 6, Y(m.bgMax) + 8, SCOPE.faint, 'left', 9);
    g.strokeStyle = ctx.accent; g.globalAlpha = 0.6; g.setLineDash([4, 3]);
    g.beginPath(); g.moveTo(padL, Math.round(Y(m.threshold)) + 0.5); g.lineTo(padL + iw, Math.round(Y(m.threshold)) + 0.5); g.stroke();
    g.setLineDash([]); g.globalAlpha = 1;
    label(c, `threshold ${m.threshold}`, padL + iw - 4, Y(m.threshold) + 8, ctx.accent, 'right', 9);
    for (const v of m.yticks) label(c, String(v), padL - 4, Y(v), SCOPE.faint, 'right', 9);
    const n = m.scan.length;
    if (!small || scan.done === n) label(c, m.bns && !small ? 'statistic (centre of each 16 s window)' : 'statistic', padL + 6, yS + 7, SCOPE.text, 'left', 9);
    const line = (arr: Float32Array, color: string, width: number, alpha: number) => {
      g.strokeStyle = color; g.lineWidth = width; g.globalAlpha = alpha; g.beginPath();
      let on = false;
      for (let k = 0; k < arr.length; k++) {
        if (!Number.isFinite(arr[k])) { on = false; continue; }
        const x = X(m.scan[k]), y = Y(arr[k]);
        if (on) g.lineTo(x, y); else { g.moveTo(x, y); on = true; }
      }
      g.stroke(); g.globalAlpha = 1; g.lineWidth = 1;
    };
    if (!m.bns) line(scan.finder, SCOPE.ink, 1, 0.35);
    line(scan.stat, ctx.accent, 2, 1);
    for (let k = 0; k < scan.done; k++) {
      if (!scan.veto[k]) continue;
      const x = X(m.scan[k]), y = Y(lo) - 5;
      g.strokeStyle = RED; g.lineWidth = 1.5; g.beginPath(); g.moveTo(x - 3, y - 3); g.lineTo(x + 3, y + 3); g.moveTo(x + 3, y - 3); g.lineTo(x - 3, y + 3); g.stroke(); g.lineWidth = 1;
    }
    if (scan.veto.some((v) => v)) label(c, small ? '× vetoed (−30)' : '× vetoed by the checker (−30)', padL + iw - 4, Y(lo) - 14, RED, 'right', 9);
    if (scan.done < n) label(c, `scanning ${scan.done}/${n}…`, padL + iw - 4, yS + 7, ctx.accent, 'right', 9);
    else if (!small) label(c, m.bns ? 'no checker: nothing can veto' : 'grey: finder alone · colour: after the checker', padL + iw - 4, yS + 7, SCOPE.faint, 'right', 9);
    // the current window on the scan
    const xc = X(w0 + half(m));
    g.strokeStyle = ctx.accent; g.globalAlpha = 0.5; g.beginPath(); g.moveTo(xc + 0.5, yS); g.lineTo(xc + 0.5, yS + hS); g.stroke(); g.globalAlpha = 1;
    g.fillStyle = ctx.accent; g.shadowColor = ctx.accent; g.shadowBlur = 10;
    g.beginPath(); g.arc(xc, Y(r.stat), 4, 0, 2 * Math.PI); g.fill(); g.shadowBlur = 0;

    // time axis
    const step = m.tick * (small ? 2 : 1);
    for (let t = Math.ceil(m.T0 / step) * step; t <= m.T1 + 1e-9; t += step) {
      const end = Math.abs(t - m.T1) < 1e-9, first = Math.abs(t - m.T0) < 1e-9;
      const txt = t === 0 ? '0' : `${t > 0 ? '+' : '−'}${m.bns ? Math.abs(t) : Math.abs(t).toFixed(1)}${end ? ' s' : ''}`;
      label(c, txt, X(t), H - 9, SCOPE.faint, end ? 'right' : first ? 'left' : 'center', 9);
    }
    if (!(Math.abs(Math.round(m.T1 / step) * step - m.T1) < 1e-9)) label(c, 's', padL + iw, H - 9, SCOPE.faint, 'right', 9);
  }

  await run();
}
