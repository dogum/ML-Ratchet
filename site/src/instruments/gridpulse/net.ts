// GridPulse in the browser: a line-for-line port of models/gridpulse/gridpulse.py (_features, forward, read_windows)
// and of perturb() in models/gridpulse/web/export_instrument.py (what each control does to the input). No DOM.
import { linear, erf } from '../../lib/nn/ops.ts';
import type { Pack } from '../../lib/nn/pack.ts';

export const PRE = 20, POST = 60, LEN = PRE + POST + 1; // window: 81 samples, t = -20 .. +60 s around the trip

export interface GMeta { ctx: boolean; n_params: number; names: string[]; pre: number; post: number; layers: number }
export interface GExample {
  id: string; t: string; demand_MW: number; E_true_GVAs: number; loss_label_MW: number | null; tier: string | null;
  nadir_Hz: number; settle_Hz: number; source: string; python_N60: Record<string, number>;
}

/** The instrument's controls (see perturb() in the exporter). */
export interface Perturb {
  N?: number; // seconds after the trip the network sees (2..60)
  demand?: boolean; // false: transmission demand unknown, the network runs without it
  noise?: number; // scale on a real quiet NESO slice (mHz), added to the whole window
}

export function perturb(w: ArrayLike<number>, demand: number, noise: ArrayLike<number>, p: Perturb) {
  const out = Float64Array.from(w);
  if (p.noise) for (let i = 0; i < out.length; i++) out[i] += (p.noise * noise[i]) / 1000;
  return { w: out, N: Math.trunc(p.N ?? 60), d: (p.demand ?? true) ? demand : NaN };
}

/** GridPulse._features for one GB window (f0 = 50 Hz): masked window ×10, is_gb, N/60, log(N)/4, log demand, flag. */
export function features(w: ArrayLike<number>, N: number, demand: number, ctx: boolean): Float64Array {
  const x = new Float64Array(LEN + 3 + (ctx ? 2 : 0));
  for (let i = 0; i < LEN; i++) x[i] = i - PRE <= N ? w[i] * 10 : 0; // * (50 / f0) = 1 for GB
  x[LEN] = 1; x[LEN + 1] = N / 60; x[LEN + 2] = Math.log(N) / 4;
  if (ctx) {
    const k = Number.isFinite(demand);
    x[LEN + 3] = k ? Math.log(demand / 30000) : 0; x[LEN + 4] = k ? 1 : 0;
  }
  return x;
}

const geluExact = (v: Float64Array) => { for (let i = 0; i < v.length; i++) v[i] = 0.5 * v[i] * (1 + erf(v[i] / Math.SQRT2)); return v; };
const clip = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

/** GridPulse.forward: raw outputs in target space, [mu x7, sigma x7] (sigma already widened by the dev factors). */
export function forward(W: Pack<GMeta>, w: ArrayLike<number>, N: number, demand: number): Float64Array {
  const t = W.t, xm = t.xm.data, xs = t.xs.data;
  const x = features(w, N, demand, W.meta.ctx);
  let h: ArrayLike<number> = Float64Array.from(x, (v, i) => (v - xm[i]) / xs[i]);
  for (let i = 0; i < W.meta.layers; i++) h = geluExact(Float64Array.from(linear(h, t[`b${i}_w`], t[`b${i}_b`])));
  const mu = linear(h, t.mu_w, t.mu_b), ls = linear(h, t.ls_w, t.ls_b);
  const ym = t.ym.data, ys = t.ys.data, wd = t.widen.data, n = mu.length;
  const out = new Float64Array(2 * n);
  for (let k = 0; k < n; k++) { out[k] = mu[k] * ys[k] + ym[k]; out[n + k] = Math.exp(clip(ls[k], -6, 3)) * ys[k] * wd[k]; }
  return out;
}

export interface Reading {
  loss: [number, number, number]; // MW lost: value, lo, hi (1 sigma)
  E: [number, number, number]; // inertia, GVA s
  beta: [number, number, number]; // MW/Hz
  head: number; ffr: number; // MW
  nadir: [number, number]; // Hz deviation from the pre-event mean, sigma
  settle: [number, number];
  raw: Float64Array;
}

/** GridPulse.read_windows for GB: physical units. */
export function read(W: Pack<GMeta>, w: ArrayLike<number>, N: number, demand: number): Reading {
  const o = forward(W, w, N, demand), n = 7, e = Math.exp;
  const tri = (i: number, s = 1): [number, number, number] => [e(o[i]) * s, e(o[i] - o[n + i]) * s, e(o[i] + o[n + i]) * s];
  const k = 1 / 10; // f0 / 50 / 10
  return {
    loss: tri(0), E: tri(1, 1e-3), beta: tri(2), head: e(o[3]), ffr: e(o[4]) - 100,
    nadir: [o[5] * k, o[n + 5] * k], settle: [o[6] * k, o[n + 6] * k], raw: o,
  };
}
