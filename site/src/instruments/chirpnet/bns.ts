// ChirpNetBNS in the browser: a line-for-line port of models/chirpnet/chirpnet.py ChirpNetBNS (the fixed
// time-frequency front end + the 2-D residual CNN) and of chirpnet.gate (25-sigma auto-gating).
import { gelu_, add_, linear, vgelu } from '../../lib/nn/ops.ts';
import type { Pack } from '../../lib/nn/pack.ts';
import type { Tensor } from '../../lib/nn/tensor.ts';
import { conv2dFast } from './fastconv.ts';
import { fft } from './dsp.ts';

export const FSB = 1024, WINB = 16384;

export interface BMeta {
  conv: [[number, number], [number, number]][]; nfft: number; hop: number; frames: number; pnorm: number;
  threshold: number; bg_top: number[]; bg_n: number; bg_live_s: number; bg_max: number;
}
export interface BEvent {
  id: string; date: string; gps: number; t_ev: number; s0: number; fs: number; win: number;
  pre: number; seg: number; off: number; place: number; gates: Record<string, [number, number][]>;
  truth: { mc_det: number; mc_det_lo: number; mc_det_hi: number; m1: number; m2: number; dl: number; snr: number };
  python: { gps: number; score: number; chirp_mass: number; chirp_mass_68: [number, number]; raw_gated: number; raw_ungated: number };
}

const clip = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));

// ---- 384-point real DFT as three interleaved 128-point FFTs (384 = 3 x 128), bins 0..192
const NF = 384, NQ = 128;
const TW = (() => {
  const c = new Float64Array(3 * (NF / 2 + 1)), s = new Float64Array(3 * (NF / 2 + 1));
  for (let r = 0; r < 3; r++) for (let k = 0; k <= NF / 2; k++) { const a = (-2 * Math.PI * r * k) / NF; c[r * (NF / 2 + 1) + k] = Math.cos(a); s[r * (NF / 2 + 1) + k] = Math.sin(a); }
  return { c, s };
})();
const HANN = Float64Array.from({ length: NF }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / NF)); // periodic, as scipy get_window('hann')
const qr = [new Float64Array(NQ), new Float64Array(NQ), new Float64Array(NQ)], qi = [new Float64Array(NQ), new Float64Array(NQ), new Float64Array(NQ)];

/** |rfft(frame * hann)|^2 for one 384-sample frame starting at xp[off]; writes 193 powers into P. */
function power384(xp: Float64Array, off: number, P: Float64Array) {
  for (let r = 0; r < 3; r++) {
    const re = qr[r], im = qi[r];
    for (let m = 0; m < NQ; m++) { const i = 3 * m + r; re[m] = xp[off + i] * HANN[i]; im[m] = 0; }
    fft(re, im);
  }
  const K = NF / 2 + 1;
  for (let k = 0; k < K; k++) {
    const km = k & (NQ - 1);
    let sr = 0, si = 0;
    for (let r = 0; r < 3; r++) {
      const c = TW.c[r * K + k], s = TW.s[r * K + k], a = qr[r][km], b = qi[r][km];
      sr += a * c - b * s; si += a * s + b * c;
    }
    P[k] = sr * sr + si * si;
  }
}

/** tf_map for one detector: x [n] -> [64 bands, frames] log(1 + band power); reflect-padded by nfft/2 like numpy.
 *  `frames` defaults to the network's 256; the display uses as many as the span holds. */
export function tfMap(W: Pack<{ bns: BMeta }>, x: ArrayLike<number>, frames?: number): Float32Array {
  const m = W.meta.bns, M = W.t.n_band_matrix, nb = M.shape[0], K = M.shape[1];
  const n = x.length, h = m.nfft >> 1, hop = m.hop;
  const nfr = frames ?? m.frames;
  const xp = new Float64Array(n + 2 * h);
  for (let i = 0; i < n; i++) xp[h + i] = x[i];
  for (let i = 1; i <= h; i++) { xp[h - i] = x[i]; xp[h + n - 1 + i] = x[n - 1 - i]; }
  const out = new Float32Array(nb * nfr), P = new Float64Array(K);
  for (let f = 0; f < nfr; f++) {
    power384(xp, f * hop, P);
    for (let b = 0; b < nb; b++) {
      let s = 0;
      for (let k = 0; k < K; k++) s += M.data[b * K + k] * P[k];
      out[b * nfr + f] = Math.log1p(s / m.pnorm);
    }
  }
  return out;
}

/** Raw outputs (7) for one 16 s window x [2, 16384] whitened [H1, L1] at 1024 Hz. */
export function bnsForward(W: Pack<{ bns: BMeta }>, x: Tensor): Float32Array {
  const t = W.t, m = W.meta.bns, n = x.shape[1];
  const a = tfMap(W, x.data.subarray(0, n)), b = tfMap(W, x.data.subarray(n, 2 * n));
  const tf = new Float32Array(a.length * 2);
  tf.set(a, 0); tf.set(b, a.length);
  const nb = t.n_band_matrix.shape[0];
  const L = (i: number, h: Tensor) => conv2dFast(h, t[`n_w${i}`], t[`n_b${i}`], m.conv[i][0], m.conv[i][1]);
  let h = gelu_(L(0, { data: tf, shape: [2, nb, m.frames] }));
  for (let i = 1; i < m.conv.length; i += 3) {
    let y = gelu_(L(i, h));
    y = L(i + 1, y);
    h = gelu_(add_(y, L(i + 2, h)));
  }
  const C = h.shape[0], N = h.data.length / C, g = new Float32Array(2 * C);
  for (let c = 0; c < C; c++) {
    let s = 0, mx = -Infinity;
    for (let i = c * N; i < (c + 1) * N; i++) { const v = h.data[i]; s += v; if (v > mx) mx = v; }
    g[c] = s / N; g[C + c] = mx;
  }
  return linear(vgelu(linear(g, t.n_lw0, t.n_lb0)), t.n_lw1, t.n_lb1);
}

export interface BReading { o: Float32Array; stat: number; mc: number; mc_lo: number; mc_hi: number; q: number; t_m: number; t_sig: number }
export function bnsRead(W: Pack<{ bns: BMeta }>, x: Tensor): BReading {
  const o = bnsForward(W, x), sl = Math.exp(clip(o[2], -7, 3));
  return { o, stat: o[0], mc: Math.exp(o[1]), mc_lo: Math.exp(o[1] - sl), mc_hi: Math.exp(o[1] + sl), q: clip(o[3], 0.05, 1), t_m: o[5], t_sig: Math.exp(clip(o[6], -7, 3)) };
}

/** chirpnet.gate: zero the whitened data around any excursion beyond `thresh` sigma (with its Tukey edges, exactly
 *  as written there). Returns the gated copy and the gates (seconds). */
export function gate(w: Float32Array, thresh = 25, half = 0.25, taper = 0.1, fs = FSB): { x: Float32Array; gates: [number, number][] } {
  const bad: number[] = [];
  for (let i = 0; i < w.length; i++) if (Math.abs(w[i]) > thresh) bad.push(i);
  if (!bad.length) return { x: w, gates: [] };
  const hs = Math.trunc(half * fs), tp = Math.trunc(taper * fs), n = w.length;
  const spans: [number, number][] = [];
  let start = bad[0], prev = bad[0];
  for (let j = 1; j < bad.length; j++) { const i = bad[j]; if (i - prev > hs) { spans.push([start, prev]); start = i; } prev = i; }
  spans.push([start, prev]);
  const mask = new Float32Array(n).fill(1);
  const r = Float64Array.from({ length: tp }, (_, i) => 0.5 + 0.5 * Math.cos(tp > 1 ? (Math.PI * i) / (tp - 1) : 0));
  for (const [a, b] of spans) {
    const lo = Math.max(0, a - hs), hi = Math.min(n, b + hs);
    mask.fill(0, lo, hi);
    if (tp) {
      const l0 = Math.max(0, lo - tp);
      for (let i = l0; i < lo; i++) mask[i] = Math.min(mask[i], r[tp - 1 - (i - l0)]);
      const h1 = Math.min(n, hi + tp);
      for (let i = hi; i < h1; i++) mask[i] = Math.min(mask[i], r[i - hi]);
    }
  }
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = w[i] * mask[i];
  return { x, gates: spans.map(([a, b]) => [a / fs, b / fs]) };
}

export function bnsFar(m: BMeta, score: number) {
  let n = 0;
  while (n < m.bg_top.length && m.bg_top[n] >= score) n++;
  return { far: (n + 1) / (m.bg_live_s / 3600), bound: n === 0, floor: n === m.bg_top.length };
}
