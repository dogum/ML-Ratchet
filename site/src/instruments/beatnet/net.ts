// BeatNet in the browser: a line-for-line port of models/beatnet/pipeline.py (band-pass, decimation, beat windows,
// local QRS scale, RR features) and beatnet.py forward() / predict_features(). No DOM.
// NOT A MEDICAL DEVICE: research and teaching only.
import { conv1d, gelu_, add_, stride1d, linear, vgelu } from '../../lib/nn/ops.ts';
import type { Pack } from '../../lib/nn/pack.ts';
import type { Tensor } from '../../lib/nn/tensor.ts';
import { rng } from '../kit.ts';

export interface BMeta {
  n_blocks: number; strides: number[]; n_head: number; cin: number; ctr: number; qmu: number; qsd: number;
  classes: string[]; logprior: number[]; qrs_cal: { a: number; b: number; sig_factor: number };
  sos: number[][]; zi: number[][]; half: number; nb: number; fs0: number; fs: number;
}
export interface BExample {
  record: number; why: string; start_s: number; fs: number; show: [number, number]; ref: string;
  python: { label: string; hr: number };
}
export const FS0 = 360;
export const ADC = 200;

/** The excerpt as recorded, in mV (float32, exactly as the Python examples store it). */
export function ecgOf(X: Pack, i: number): Float32Array {
  const a = X.t[`ecg${i}`].data, x = new Float32Array(a.length);
  for (let k = 0; k < a.length; k++) x[k] = a[k] / ADC;
  return x;
}

/** White "muscle" noise (σ mV, seeded) and a two-tone respiration-like baseline wander (amplitude mV) — export_instrument.py perturb(). */
export function perturb(ecg: Float32Array, o: { noise?: number; wander?: number; seed?: number }): Float64Array {
  const x = Float64Array.from(ecg);
  const noise = o.noise ?? 0, wander = o.wander ?? 0;
  if (noise > 0) { const r = rng(o.seed ?? 1); for (let k = 0; k < x.length; k++) x[k] += noise * r.n(); }
  if (wander > 0) {
    for (let k = 0; k < x.length; k++) {
      const t = k / FS0;
      x[k] += (wander * (Math.sin(2 * Math.PI * 0.21 * t + 0.6) + 0.5 * Math.sin(2 * Math.PI * 0.07 * t + 2.1))) / 1.5;
    }
  }
  return x;
}

// ---- front end -------------------------------------------------------------------------------
/** scipy.signal.sosfilt with zi = sosfilt_zi(sos) * x[0] (what sosfiltfilt does), in place, float64. */
function sosfilt(y: Float64Array, sos: number[][], zi: number[][]) {
  const x0 = y[0];
  for (let s = 0; s < sos.length; s++) {
    const [b0, b1, b2, , a1, a2] = sos[s];
    let z0 = zi[s][0] * x0, z1 = zi[s][1] * x0;
    for (let i = 0; i < y.length; i++) {
      const xi = y[i], yi = b0 * xi + z0;
      z0 = b1 * xi - a1 * yi + z1;
      z1 = b2 * xi - a2 * yi;
      y[i] = yi;
    }
  }
}

/** pipeline.filt(): zero-phase 0.5–40 Hz band-pass (sosfiltfilt, odd padding of 3·(2·sections+1)) at 360 Hz, then every 2nd sample. */
export function filt(x: Float64Array, m: BMeta): Float64Array {
  const n = x.length, pad = 3 * (2 * m.sos.length + 1);
  const e = new Float64Array(n + 2 * pad);
  for (let i = 0; i < pad; i++) { e[i] = 2 * x[0] - x[pad - i]; e[n + pad + i] = 2 * x[n - 1] - x[n - 2 - i]; }
  e.set(x, pad);
  sosfilt(e, m.sos, m.zi);
  e.reverse();
  sosfilt(e, m.sos, m.zi);
  e.reverse();
  const out = new Float64Array(Math.ceil(n / 2));
  for (let i = 0; i < out.length; i++) out[i] = e[pad + 2 * i];
  return out;
}

const rhe = (x: number) => { const f = Math.floor(x), d = x - f; return d === 0.5 ? (f % 2 === 0 ? f : f + 1) : Math.round(x); }; // numpy round-half-even
function medianF(a: ArrayLike<number>): number {
  const s = Float64Array.from(a).sort(), n = s.length;
  return n % 2 ? s[(n - 1) >> 1] : Math.fround(0.5 * (s[n / 2 - 1] + s[n / 2]));
}
function median(a: ArrayLike<number>): number {
  const s = Float64Array.from(a).sort(), n = s.length;
  return n % 2 ? s[(n - 1) >> 1] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

export interface BeatFeatures { w: Float32Array; rr: number[]; scale: number; ptp: number }

/** pipeline.features() for the beats `keep` (indices into pos; positions in samples at 360 Hz) of the filtered 180 Hz signal y.
 *  (cin = 1: the neighbour template is not an input of the released network, so it is not computed.) */
export function features(y: Float64Array, pos: ArrayLike<number>, keep: number[], m: BMeta): BeatFeatures[] {
  const n = pos.length, H = m.half, L = 2 * H, q = Math.floor(0.08 * m.fs), ratio = m.fs / FS0;
  const W: Float32Array[] = [], ptp = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const p = rhe(pos[i] * ratio), w = new Float32Array(L);
    for (let j = 0; j < L; j++) { const k = p - H + j; w[j] = k >= 0 && k < y.length ? y[k] : 0; }
    let mx = -Infinity, mn = Infinity;
    for (let j = H - q; j < H + q; j++) { if (w[j] > mx) mx = w[j]; if (w[j] < mn) mn = w[j]; }
    ptp[i] = mx - mn;
    W.push(w);
  }
  const rr: number[] = [];
  for (let i = 1; i < n; i++) rr.push(pos[i] / FS0 - pos[i - 1] / FS0);
  const pre = [n > 1 ? rr[0] : 1, ...rr], post = [...rr, n > 1 ? rr[rr.length - 1] : 1];
  const out: BeatFeatures[] = [];
  for (const i of keep) {
    const lo = Math.max(0, i - m.nb), hi = Math.min(n, i + m.nb + 1);
    let nb: number[] = [];
    for (let j = lo; j < hi; j++) if (j !== i) nb.push(j);
    if (!nb.length) nb = [i];
    const s = Math.max(medianF(nb.map((j) => ptp[j])), 0.05), sf = Math.fround(s);
    const k0 = Math.max(0, i - 10), k1 = Math.min(n - 1, i + 10);
    let loc = 1, irr = 0;
    if (k1 > k0) { let sm = 0; for (let j = k0; j < k1; j++) sm += rr[j]; loc = sm / (k1 - k0); }
    if (k1 - k0 > 2) { const d: number[] = []; for (let j = k0 + 1; j < k1; j++) d.push(Math.abs(Math.log(rr[j]) - Math.log(rr[j - 1]))); irr = median(d); }
    const w = new Float32Array(L);
    for (let j = 0; j < L; j++) w[j] = W[i][j] / sf;
    const f = [Math.log(pre[i] / loc), Math.log(post[i] / loc), Math.log(loc / 0.8), Math.log(pre[i] / post[i]), irr * 5, Math.log(Math.fround(ptp[i] / sf))];
    out.push({ w, rr: f.map((v) => (Number.isFinite(v) ? Math.fround(v) : 0)), scale: s, ptp: ptp[i] });
  }
  return out;
}

// ---- network ---------------------------------------------------------------------------------
/** BeatNet.forward() for one beat: x [cin, 450] window, r the six rhythm features -> 7 raw outputs (4 logits, QRS mu, log-sigma, AF logit). */
export function forward(W: Pack<BMeta>, x: Tensor, r: ArrayLike<number>): Float32Array {
  const t = W.t, m = W.meta;
  let h = x;
  m.strides.forEach((s, i) => {
    let y = gelu_(conv1d(h, t[`b${i}_c1_w`], t[`b${i}_c1_b`], s));
    y = conv1d(y, t[`b${i}_c2_w`], t[`b${i}_c2_b`]);
    h = gelu_(add_(y, conv1d(s > 1 ? stride1d(h, s) : h, t[`b${i}_sk_w`], t[`b${i}_sk_b`])));
  });
  const [C, L] = h.shape, mid = L >> 1, k = m.ctr >> 1, nc = 2 * k + 1;
  let z: Float32Array = new Float32Array(C * nc + 2 * C + r.length);
  for (let c = 0; c < C; c++) for (let l = 0; l < nc; l++) z[c * nc + l] = h.data[c * L + mid - k + l];
  for (let c = 0; c < C; c++) {
    let s = 0, mx = -Infinity;
    for (let l = 0; l < L; l++) { const v = h.data[c * L + l]; s += v; if (v > mx) mx = v; }
    z[C * nc + c] = s / L; z[C * nc + C + c] = mx;
  }
  for (let j = 0; j < r.length; j++) z[C * nc + 2 * C + j] = r[j];
  for (let j = 0; j < m.n_head; j++) {
    z = linear(z, t[`h${j}_w`], t[`h${j}_b`]);
    if (j < m.n_head - 1) vgelu(z);
  }
  return z;
}

// ---- fast network: the same arithmetic as forward(), restructured for speed ------------------
// Each convolution is an im2col + register-tiled matrix product (4 output channels × 2 positions per inner loop,
// accumulating in float64). One beat is 5.6 M multiply-adds; this runs them ~3× faster than the plain loop.
function convFast(x: Float32Array, C: number, L: number, w: Tensor, b: Tensor, stride: number): [Float32Array, number, number] {
  const [O, , K] = w.shape, wd = w.data, bd = b.data, p = K >> 1, Lo = Math.floor((L + 2 * p - K) / stride) + 1, CK = C * K;
  const col = new Float32Array((Lo + 1) * CK);
  for (let j = 0; j < Lo; j++) {
    const cb = j * CK, s0 = j * stride - p;
    for (let c = 0; c < C; c++) {
      const xb = c * L, ob = cb + c * K;
      for (let k = 0; k < K; k++) { const s = s0 + k; col[ob + k] = s >= 0 && s < L ? x[xb + s] : 0; }
    }
  }
  const out = new Float32Array(O * Lo);
  let o = 0;
  for (; o + 4 <= O; o += 4) {
    const w0 = o * CK, w1 = w0 + CK, w2 = w1 + CK, w3 = w2 + CK;
    const r0 = o * Lo, r1 = r0 + Lo, r2 = r1 + Lo, r3 = r2 + Lo;
    for (let j = 0; j < Lo; j += 2) {
      const c0 = j * CK, c1 = c0 + CK;
      let a0 = 0, a1 = 0, a2 = 0, a3 = 0, d0 = 0, d1 = 0, d2 = 0, d3 = 0;
      for (let i = 0; i < CK; i++) {
        const u = col[c0 + i], v = col[c1 + i], p0 = wd[w0 + i], p1 = wd[w1 + i], p2 = wd[w2 + i], p3 = wd[w3 + i];
        a0 += p0 * u; a1 += p1 * u; a2 += p2 * u; a3 += p3 * u;
        d0 += p0 * v; d1 += p1 * v; d2 += p2 * v; d3 += p3 * v;
      }
      out[r0 + j] = a0 + bd[o]; out[r1 + j] = a1 + bd[o + 1]; out[r2 + j] = a2 + bd[o + 2]; out[r3 + j] = a3 + bd[o + 3];
      if (j + 1 < Lo) { out[r0 + j + 1] = d0 + bd[o]; out[r1 + j + 1] = d1 + bd[o + 1]; out[r2 + j + 1] = d2 + bd[o + 2]; out[r3 + j + 1] = d3 + bd[o + 3]; }
    }
  }
  for (; o < O; o++) {
    for (let j = 0; j < Lo; j++) {
      let a = 0;
      for (let i = 0; i < CK; i++) a += wd[o * CK + i] * col[j * CK + i];
      out[o * Lo + j] = a + bd[o];
    }
  }
  return [out, O, Lo];
}

/** forward() for one beat, fast path: window [450], rhythm features [6] -> 7 raw outputs. */
export function forwardFast(W: Pack<BMeta>, x: Float32Array, r: ArrayLike<number>): Float32Array {
  const t = W.t, m = W.meta;
  let h = x, C = 1, L = x.length;
  m.strides.forEach((s, i) => {
    const [y1, O, Lo] = convFast(h, C, L, t[`b${i}_c1_w`], t[`b${i}_c1_b`], s);
    gelu_({ data: y1, shape: [O, Lo] });
    const [y2] = convFast(y1, O, Lo, t[`b${i}_c2_w`], t[`b${i}_c2_b`], 1);
    const [sk] = convFast(h, C, L, t[`b${i}_sk_w`], t[`b${i}_sk_b`], s); // 1×1 conv on x[..., ::s]
    for (let j = 0; j < y2.length; j++) y2[j] += sk[j];
    h = gelu_({ data: y2, shape: [O, Lo] }).data; C = O; L = Lo;
  });
  const mid = L >> 1, k = m.ctr >> 1, nc = 2 * k + 1;
  let z: Float32Array = new Float32Array(C * nc + 2 * C + r.length);
  for (let c = 0; c < C; c++) {
    for (let l = 0; l < nc; l++) z[c * nc + l] = h[c * L + mid - k + l];
    let sm = 0, mx = -Infinity;
    for (let l = 0; l < L; l++) { const v = h[c * L + l]; sm += v; if (v > mx) mx = v; }
    z[C * nc + c] = sm / L; z[C * nc + C + c] = mx;
  }
  for (let j = 0; j < r.length; j++) z[C * nc + 2 * C + j] = r[j];
  for (let j = 0; j < m.n_head; j++) {
    z = linear(z, t[`h${j}_w`], t[`h${j}_b`]);
    if (j < m.n_head - 1) vgelu(z);
  }
  return z;
}

export interface BeatReading { p: number[]; cls: number; qrs: number; qrsSig: number; af: number; raw: Float32Array }

/** predict_features() post-processing for one beat: prior-adjusted class probabilities, calibrated QRS width ± σ (ms), AF probability. */
export function read(W: Pack<BMeta>, f: BeatFeatures): BeatReading {
  return readout(W.meta, forward(W, { data: f.w, shape: [1, f.w.length] }, f.rr));
}
export function readout(m: BMeta, o: Float32Array): BeatReading {
  const z = [0, 1, 2, 3].map((i) => o[i] + (m.logprior?.[i] ?? 0)), mx = Math.max(...z);
  const e = z.map((v) => Math.exp(v - mx)), s = e.reduce((a, b) => a + b, 0), p = e.map((v) => v / s);
  const c = m.qrs_cal ?? { a: 0, b: 1, sig_factor: 1 };
  const q = Math.exp(o[4] * m.qsd + m.qmu), qs = Math.exp(Math.min(2, Math.max(-4, o[5]))) * m.qsd * q;
  return { p, cls: p.indexOf(Math.max(...p)), qrs: c.a + c.b * q, qrsSig: qs * c.b * c.sig_factor, af: o.length > 6 ? 1 / (1 + Math.exp(-o[6])) : NaN, raw: o };
}

/** Indices of the beats inside the shown minute (the 30 s either side are context for the neighbour scale and RR features). */
export function shownBeats(pos: ArrayLike<number>, show: [number, number]): number[] {
  const k: number[] = [];
  for (let i = 0; i < pos.length; i++) if (pos[i] >= show[0] && pos[i] < show[1]) k.push(i);
  return k;
}
