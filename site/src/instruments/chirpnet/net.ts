// ChirpNet in the browser: a line-for-line port of models/chirpnet/chirpnet.py —
// the black-hole finder (ChirpNet.forward, arch 'cnn'), the two-stream coincidence checker (arch '2s'),
// the cascade that joins them (ChirpNet.statistic / predict) and the posterior flow (ChirpPE.sample).
import { gelu_, add_, linear, vgelu, concat0 } from '../../lib/nn/ops.ts';
import { conv1dFast as conv1d, linearBatch } from './fastconv.ts';
import type { Pack } from '../../lib/nn/pack.ts';
import type { Tensor } from '../../lib/nn/tensor.ts';

export const FS = 2048, WIN = 4096;
const G_MSUN_C3 = 4.925490947e-6;

export interface CMeta {
  conv: [number, number][]; threshold: number; veto: number; drop: number;
  pe_layers: number; pe_mu: number[]; pe_sd: number[]; pe_params: string[];
  bg_top: number[]; bg_n: number; bg_live_s: number; bg_max: number;
}
export interface CEvent {
  id: string; date: string; gps: number; t_ev: number; s0: number; asd: number[];
  truth: {
    mc_src: number; mc_src_lo: number; mc_src_hi: number; z: number; mc_det: number; mc_det_lo: number; mc_det_hi: number;
    m1: number; m2: number; q: number; dl: number; dl_lo: number; dl_hi: number; chi: number; snr: number;
  };
  python: { gps: number; score: number; coincidence: number; chirp_mass: number; chirp_mass_68: [number, number] };
}
export interface CExamples { fs: number; win: number; pre: number; seg: number; place: number; events: CEvent[] }

const clip = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));

/** The shared residual trunk (ChirpNet._trunk): conv+GELU, then blocks of conv, GELU, conv + strided 1x1 skip, GELU. */
function trunk(W: Pack<CMeta>, p: string, x: Tensor): Tensor {
  const t = W.t, cv = W.meta.conv;
  const L = (i: number, h: Tensor) => conv1d(h, t[`${p}w${i}`], t[`${p}b${i}`], cv[i][0], cv[i][1]);
  let h = gelu_(L(0, x));
  for (let i = 1; i < cv.length; i += 3) {
    let y = gelu_(L(i, h));
    y = L(i + 1, y);
    h = gelu_(add_(y, L(i + 2, h)));
  }
  return h;
}

/** Per-channel mean and max of h [C, n], written into mean[0:C] and max[0:C]. */
function meanMax(h: Tensor, mean: Float32Array, max: Float32Array) {
  const [C, n] = h.shape;
  for (let c = 0; c < C; c++) {
    let s = 0, m = -Infinity;
    for (let i = c * n; i < (c + 1) * n; i++) { const v = h.data[i]; s += v; if (v > m) m = v; }
    mean[c] = s / n; max[c] = m;
  }
}

/** Finder: x [2, 4096] whitened [H1, L1] -> 7 raw outputs
 *  (score, log chirp mass, log sigma, mass ratio, log sigma q, merger time in the window, log sigma t). */
export function finder(W: Pack<CMeta>, x: Tensor): Float32Array {
  const t = W.t;
  const h = trunk(W, 'f_', x);
  const C = h.shape[0];
  const f = new Float32Array(2 * C);
  meanMax(h, f.subarray(0, C), f.subarray(C));
  return linear(vgelu(linear(f, t.f_lw0, t.f_lb0)), t.f_lw1, t.f_lb1);
}

/** Coincidence checker (two-stream): the same trunk reads each detector alone, then an interaction stage
 *  compares them position by position ([h, l, h*l, |h-l|] -> 1x1 mix -> residual block -> pooled). */
export function checker(W: Pack<CMeta>, x: Tensor): Float32Array {
  const t = W.t, n = x.shape[1];
  const h = trunk(W, 'c_', { data: x.data.subarray(0, n), shape: [1, n] });
  const l = trunk(W, 'c_', { data: x.data.subarray(n, 2 * n), shape: [1, n] });
  const [C, Lh] = h.shape, N = C * Lh;
  const prod = new Float32Array(N), dif = new Float32Array(N);
  for (let j = 0; j < N; j++) { const a = h.data[j], b = l.data[j]; prod[j] = a * b; dif[j] = Math.abs(a - b); }
  const cat = concat0([h, l, { data: prod, shape: [C, Lh] }, { data: dif, shape: [C, Lh] }]);
  const z0 = gelu_(conv1d(cat, t.c_mixw, t.c_mixb, 1, 0));
  let y = gelu_(conv1d(z0, t.c_p1w, t.c_p1b, 1, t.c_p1w.shape[2] >> 1));
  y = conv1d(y, t.c_p2w, t.c_p2b, 1, t.c_p2w.shape[2] >> 1);
  const zz = gelu_(add_(y, conv1d(z0, t.c_psw, t.c_psb, 1, 0)));
  const g = new Float32Array(4 * C), junk = new Float32Array(C);
  meanMax(zz, g.subarray(0, C), g.subarray(C, 2 * C));
  meanMax(h, junk, g.subarray(2 * C, 3 * C));
  meanMax(l, junk, g.subarray(3 * C, 4 * C));
  return linear(vgelu(linear(g, t.c_lw0, t.c_lb0)), t.c_lw1, t.c_lb1);
}

export interface Reading {
  finder: Float32Array; check: number | null; stat: number; vetoed: boolean;
  mc: number; mc_lo: number; mc_hi: number; q: number; q_sig: number; t_m: number; t_sig: number;
}

/** ChirpNet.statistic + predict for one window: the checker runs only where the finder scores above `floor` (0);
 *  if it scores below the veto level the statistic drops by `drop` — the cascade can only lower a score. */
export function read(W: Pack<CMeta>, x: Tensor, floor = 0): Reading {
  const o = finder(W, x);
  let check: number | null = null, stat = o[0], vetoed = false;
  if (o[0] > floor) {
    check = checker(W, x)[0];
    if (check < W.meta.veto) { stat -= W.meta.drop; vetoed = true; }
  }
  const sl = Math.exp(clip(o[2], -7, 3));
  return {
    finder: o, check, stat, vetoed,
    mc: Math.exp(o[1]), mc_lo: Math.exp(o[1] - sl), mc_hi: Math.exp(o[1] + sl),
    q: clip(o[3], 0.05, 1), q_sig: Math.exp(clip(o[4], -7, 3)), t_m: o[5], t_sig: Math.exp(clip(o[6], -7, 3)),
  };
}

/** far_per_hour: (background windows at least this loud + 1) / livetime. `bound`: none louder (an upper limit).
 *  Only the loudest background values ship; below them the rate is a lower limit (`floor`). */
export function farPerHour(m: CMeta, score: number): { far: number; bound: boolean; floor: boolean } {
  let n = 0;
  while (n < m.bg_top.length && m.bg_top[n] >= score) n++;
  return { far: (n + 1) / (m.bg_live_s / 3600), bound: n === 0, floor: n === m.bg_top.length };
}

// ---------------------------------------------------------------- ChirpNet-PE (DINGO-style RealNVP flow)
/** Context vector: CNN embedding of the window (mean | max) + both detectors' ASD summary -> 2-layer MLP. */
export function peContext(W: Pack<CMeta>, x: Tensor, asd: ArrayLike<number>): Float32Array {
  const t = W.t;
  const h = trunk(W, 'p_', x);
  const C = h.shape[0];
  const f = new Float32Array(2 * C + asd.length);
  meanMax(h, f.subarray(0, C), f.subarray(C, 2 * C));
  f.set(asd as ArrayLike<number>, 2 * C);
  return linear(vgelu(linear(f, t.p_cw0, t.p_cb0)), t.p_cw1, t.p_cb1);
}

/** Push latent draws z [n, 6] through the inverse flow -> samples [n, 6] (log mc, q, chi_eff, log10 dL, cos iota, t_c). */
export function peSample(W: Pack<CMeta>, ctx: Float32Array, z: Float32Array): Float32Array {
  const t = W.t, m = W.meta, D = 6, n = z.length / D, Cn = ctx.length;
  const out = new Float32Array(z);
  const zm = new Float32Array(n * D);
  for (let k = m.pe_layers - 1; k >= 0; k--) {
    const mask = t[`p_m${k}`].data;
    const w0 = t[`p_f${k}w0`], w1 = t[`p_f${k}w1`], w2 = t[`p_f${k}w2`];
    const H = w0.shape[0], I0 = w0.shape[1];
    // first layer on [z*m, ctx]: the context part is the same for every draw, so fold it into the bias once
    const cb = new Float32Array(H), w0z = new Float32Array(H * D);
    for (let o = 0; o < H; o++) {
      let s = t[`p_f${k}b0`].data[o];
      for (let i = 0; i < Cn; i++) s += w0.data[o * I0 + D + i] * ctx[i];
      cb[o] = s;
      for (let d = 0; d < D; d++) w0z[o * D + d] = w0.data[o * I0 + d];
    }
    for (let r = 0; r < n; r++) for (let d = 0; d < D; d++) zm[r * D + d] = out[r * D + d] * mask[d];
    const h0 = vgelu(linearBatch(zm, n, D, w0z, H, cb));
    const h1 = vgelu(linearBatch(h0, n, H, w1.data, w1.shape[0], t[`p_f${k}b1`].data));
    const o = linearBatch(h1, n, w1.shape[0], w2.data, w2.shape[0], t[`p_f${k}b2`].data);
    const O = w2.shape[0];
    for (let r = 0; r < n; r++) {
      for (let d = 0; d < D; d++) {
        const mm = mask[d], zi = r * D + d;
        const s = 2.5 * Math.tanh(o[r * O + d] / 2.5) * (1 - mm), tt = o[r * O + D + d] * (1 - mm);
        out[zi] = out[zi] * mm + (1 - mm) * ((out[zi] - tt) * Math.exp(-s));
      }
    }
  }
  for (let r = 0; r < n; r++) for (let d = 0; d < D; d++) out[r * D + d] = out[r * D + d] * m.pe_sd[d] + m.pe_mu[d];
  return out;
}

// Planck 2015 flat LambdaCDM: luminosity distance (Mpc) -> redshift (chirpnet._dl_to_z)
const DLZ = (() => {
  const N = 5001, z = new Float64Array(N), dl = new Float64Array(N), H0 = 67.74, Om = 0.3075, c = 299792.458;
  let dc = 0, prev = 1;
  for (let i = 0; i < N; i++) {
    z[i] = (2.5 * i) / (N - 1);
    const e = 1 / Math.sqrt(Om * (1 + z[i]) ** 3 + (1 - Om));
    if (i > 0) dc += 0.5 * (e + prev) * (z[i] - z[i - 1]);
    prev = e;
    dl[i] = (1 + z[i]) * dc * (c / H0);
  }
  return { z, dl };
})();
export function redshift(dlMpc: number): number {
  const { z, dl } = DLZ;
  if (dlMpc <= 0) return 0;
  let lo = 0, hi = dl.length - 1;
  if (dlMpc >= dl[hi]) return z[hi];
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (dl[mid] <= dlMpc) lo = mid; else hi = mid; }
  return z[lo] + ((dlMpc - dl[lo]) / (dl[hi] - dl[lo])) * (z[hi] - z[lo]);
}

/** ChirpPE.weigh: per-sample physical parameters, source-frame masses via the sampled distance's redshift. */
export function physical(S: Float32Array) {
  const n = S.length / 6;
  const f = () => new Float32Array(n);
  const mc = f(), q = f(), chi = f(), dl = f(), m1 = f(), m2 = f(), mcs = f();
  for (let i = 0; i < n; i++) {
    mc[i] = Math.exp(S[i * 6]); q[i] = clip(S[i * 6 + 1], 0.05, 1); chi[i] = clip(S[i * 6 + 2], -1, 1); dl[i] = 10 ** S[i * 6 + 3];
    const zz = redshift(dl[i]), eta = q[i] / (1 + q[i]) ** 2, M = mc[i] / eta ** 0.6;
    m1[i] = M / (1 + q[i]) / (1 + zz); m2[i] = (M * q[i]) / (1 + q[i]) / (1 + zz); mcs[i] = mc[i] / (1 + zz);
  }
  return { mc, q, chi, dl, m1, m2, mcs };
}

/** Linear-interpolated quantiles (numpy's default). */
export function quantiles(a: ArrayLike<number>, ps: number[]): number[] {
  const s = Float64Array.from(a).sort();
  return ps.map((p) => {
    const x = p * (s.length - 1), i = Math.floor(x), fr = x - i;
    return i + 1 < s.length ? s[i] * (1 - fr) + s[i + 1] * fr : s[i];
  });
}

/** Leading-order frequency (Hz) of the (2,2) chirp a time tau (s) before coalescence (chirpnet._chirp_track). */
export function chirpFreq(mc: number, tau: number): number {
  return (1 / Math.PI) * (5 / (256 * Math.max(tau, 1e-4))) ** 0.375 * (mc * G_MSUN_C3) ** -0.625;
}
