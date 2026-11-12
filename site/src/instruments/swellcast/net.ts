// SwellCast in the browser: a line-for-line port of models/swellcast/swellcast.py features(), forward() and predict().
import { conv2d, gelu_, add_, linear, vgelu } from '../../lib/nn/ops.ts';
import type { Pack } from '../../lib/nn/pack.ts';
import type { Tensor } from '../../lib/nn/tensor.ts';

export interface SMeta {
  n_blocks: number; n_head: number; mu_y: number[]; sd_y: number[];
  sigma_factor: { logD: number; t0: number; dir: number; loghs: number };
  det_thr: number;
}
export interface SExample {
  name: string; buoy: string; basin: string; year: number; raw: boolean; place: string;
  centre_hour: number; centre: string;
  truth: { D: number; t0: number; dir: number };
}
export interface SExamples { pad: number; nt: number; fgrid: number[]; dt_h: number; examples: SExample[] }

export const NT = 80, NF = 34, DT_H = 3;
export const fgrid = (j: number) => 0.0325 + 0.005 * j;
/** deep-water dispersion: the ridge rises at df/dt = g / (4 pi D), in Hz per hour for D in km */
export const ridgeSlope = (D_km: number) => (9.81 * 3600) / (4 * Math.PI * D_km * 1e3);

/** numpy nanmedian of the finite values (NaN if none) */
function nanmedian(v: number[]): number {
  if (!v.length) return NaN;
  v.sort((a, b) => a - b);
  const n = v.length, m = n >> 1;
  return n % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

/**
 * features(): window [3, 80, 34] = [E m^2/Hz, r1 cos a1, r1 sin a1] (time-major, NaN = missing)
 * -> [5, 34, 80] (frequency x time): normalised log energy, anomaly from the per-frequency median, C, S, mask.
 */
export function features(x: Float32Array): Tensor {
  const out = new Float32Array(5 * NF * NT);
  const L = new Float32Array(NT * NF);
  const ok = new Uint8Array(NT * NF);
  for (let i = 0; i < NT * NF; i++) {
    const e = x[i];
    if (Number.isFinite(e)) { ok[i] = 1; L[i] = Math.log10(Math.max(e, 0.005)); }
  }
  const P = NF * NT;
  for (let j = 0; j < NF; j++) {
    const col: number[] = [];
    for (let t = 0; t < NT; t++) if (ok[t * NF + j]) col.push(L[t * NF + j]);
    let med = Math.fround(nanmedian(col));
    if (!Number.isFinite(med)) med = -2.3;
    for (let t = 0; t < NT; t++) {
      const i = t * NF + j, o = j * NT + t;
      if (!ok[i]) continue;
      const c = x[NT * NF + i], s = x[2 * NT * NF + i];
      out[o] = (L[i] + 0.5) / 1.2;
      out[P + o] = (L[i] - med) / 0.6;
      out[2 * P + o] = Number.isFinite(c) ? c : 0;
      out[3 * P + o] = Number.isFinite(s) ? s : 0;
      out[4 * P + o] = 1;
    }
  }
  return { data: out, shape: [5, NF, NT] };
}

/** Raw network outputs (9 numbers) for features f [5, 34, 80]. */
export function forward(W: Pack<SMeta>, f: Tensor): Float32Array {
  const t = W.t, m = W.meta;
  let h = gelu_(conv2d(f, t.stem_w, t.stem_b));
  for (let i = 0; i < m.n_blocks; i++) {
    let y = gelu_(conv2d(h, t[`b${i}_c1_w`], t[`b${i}_c1_b`], 2));
    y = conv2d(y, t[`b${i}_c2_w`], t[`b${i}_c2_b`]);
    h = gelu_(add_(y, conv2d(h, t[`b${i}_sk_w`], t[`b${i}_sk_b`], 2)));
  }
  h = gelu_(conv2d(h, t.red_w, t.red_b));
  let z: Float32Array = h.data;
  for (let j = 0; j < m.n_head; j++) {
    z = linear(z, t[`h${j}_w`], t[`h${j}_b`]);
    if (j < m.n_head - 1) vgelu(z);
  }
  return z;
}

export interface Reading {
  p_swell: number; detected: boolean;
  D_km: number; logD: number; logD_sig: number;
  t0: number; t0_sig: number;
  hs: number; loghs: number; loghs_sig: number;
  dir: number; kappa: number; half: number;
}

/** predict() for one window, with the dev-fitted widening factors (calibrated=True). */
export function read(W: Pack<SMeta>, x: Float32Array): { r: Reading; raw: Float32Array } {
  const o = forward(W, features(x));
  const { mu_y: mu, sd_y: sd, sigma_factor: f, det_thr } = W.meta;
  const F = [f.logD ?? 1, f.t0 ?? 1, f.loghs ?? 1];
  const val = (k: number) => o[k] * sd[k] + mu[k];
  const sig = (k: number) => Math.exp(Math.min(3, Math.max(-6, o[3 + k]))) * sd[k] * F[k];
  const kappa = Math.hypot(o[6], o[7]) / (f.dir ?? 1) ** 2;
  const p = 1 / (1 + Math.exp(-o[8]));
  const logD = val(0), loghs = val(2);
  const r: Reading = {
    p_swell: p, detected: p > (det_thr ?? 0.5),
    D_km: 10 ** logD, logD, logD_sig: sig(0),
    t0: val(1), t0_sig: sig(1),
    hs: 10 ** loghs, loghs, loghs_sig: sig(2),
    dir: ((Math.atan2(o[6], o[7]) * 180) / Math.PI + 360) % 360,
    kappa, half: Math.min(180, 57.29577951308232 / Math.sqrt(Math.max(kappa, 1e-3))),
  };
  return { r, raw: o };
}

/** The 80-step window starting `shift` steps after the exported window's own start (wide array [3, nt + 2 pad, 34]). */
export function slice(wide: Tensor, pad: number, shift: number): Float32Array {
  const [, Tw] = wide.shape;
  const out = new Float32Array(3 * NT * NF);
  const a = pad + shift;
  for (let c = 0; c < 3; c++) out.set(wide.data.subarray((c * Tw + a) * NF, (c * Tw + a + NT) * NF), c * NT * NF);
  return out;
}
