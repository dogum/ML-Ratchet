// BasinDetective in the browser: the 5-member MLP ensemble (port of basindetective/model.py, Detective.forward)
// and the readable outputs of basindetective.predict(). Weights from site/public/instruments/basindetective/weights.
// No DOM.
import type { Pack } from '../../lib/nn/pack.ts';
import { EPSQ, featurize, type Sigs } from './features.ts';
import { apply, type BasinRecord, type WhatIf } from './whatif.ts';

export interface BDMeta {
  names: string[]; targets: string[]; n_members: number; n_layers: number; cls_temp: number;
  units: Record<string, string>; version: string;
}

/** erf to ~1e-15 (series below 3, Lentz continued fraction above), from the shipped page's detective.js. */
export function erf(x: number): number {
  const s = x < 0 ? -1 : 1;
  x = Math.abs(x);
  if (x < 3) {
    let sum = x, term = x, n = 0;
    const x2 = x * x;
    while (Math.abs(term) > 1e-17 * Math.abs(sum) && n < 200) { n++; term *= -x2 / n; sum += term / (2 * n + 1); }
    return (s * 2) / Math.sqrt(Math.PI) * sum;
  }
  const tiny = 1e-300;
  let f = x, C = x, D = 0;
  for (let i = 1; i < 300; i++) {
    const an = i / 2;
    D = x + an * D; if (Math.abs(D) < tiny) D = tiny; D = 1 / D;
    C = x + an / C; if (Math.abs(C) < tiny) C = tiny;
    const delta = C * D; f *= delta; if (Math.abs(delta - 1) < 1e-16) break;
  }
  return s * (1 - Math.exp(-x * x) / Math.sqrt(Math.PI) / f);
}
const gelu = (v: number) => 0.5 * v * (1 + erf(v / Math.SQRT2));

function dense(x: Float64Array, w: Float32Array, b: Float32Array, nout: number, act: boolean) {
  const nin = x.length, o = new Float64Array(nout);
  for (let i = 0; i < nout; i++) {
    let s = 0;
    const off = i * nin;
    for (let j = 0; j < nin; j++) s += w[off + j] * x[j];
    s += b[i];
    o[i] = act ? gelu(s) : s;
  }
  return o;
}

export interface Raw { mu: number[]; sd: number[]; p: number }

/** Detective.forward for one feature vector: ensemble mean, sqrt(mean variance + spread) x sig_scale, tempered p. */
export function forward(pack: Pack<BDMeta>, x: ArrayLike<number>): Raw {
  const { t, meta } = pack;
  const nt = meta.targets.length, K = meta.n_members;
  const med = t.x_med.data, iqr = t.x_iqr.data;
  const z = new Float64Array(x.length);
  for (let j = 0; j < x.length; j++) {
    let v = (x[j] - med[j]) / iqr[j];
    v = Math.max(-6, Math.min(6, v));
    z[j] = Number.isFinite(v) ? v : 0;
  }
  const mus: number[][] = [], vars: number[][] = [], ps: number[] = [];
  for (let k = 0; k < K; k++) {
    const W = (n: string) => t[`m${k}_${n}`];
    let h = z;
    for (let l = 0; l < meta.n_layers; l++) h = dense(h, W(`w${l}`).data, W(`b${l}`).data, W(`b${l}`).shape[0], true);
    const o = dense(h, W('wo').data, W('bo').data, 2 * nt + 1, false);
    const ymu = W('y_mu').data, ysd = W('y_sd').data;
    const mu: number[] = [], vr: number[] = [];
    for (let j = 0; j < nt; j++) {
      mu.push(o[j] * ysd[j] + ymu[j]);
      const sd = Math.exp(Math.max(-7, Math.min(5, o[nt + j]))) * ysd[j];
      vr.push(sd * sd);
    }
    mus.push(mu); vars.push(vr); ps.push(1 / (1 + Math.exp(-o[2 * nt])));
  }
  const sc = t.sig_scale.data;
  const mu: number[] = [], sd: number[] = [];
  for (let j = 0; j < nt; j++) {
    let m = 0, v = 0, sp = 0;
    for (let k = 0; k < K; k++) { m += mus[k][j]; v += vars[k][j]; }
    m /= K; v /= K;
    for (let k = 0; k < K; k++) sp += (mus[k][j] - m) ** 2;
    mu.push(m);
    sd.push(Math.sqrt(v + sp / K) * sc[j]);
  }
  let p = ps.reduce((a, b) => a + b, 0) / K;
  if (meta.cls_temp !== 1) {
    const lg = Math.log(Math.min(Math.max(p, 1e-9), 1 - 1e-9) / Math.max(1 - p, 1e-9));
    p = 1 / (1 + Math.exp(-lg / meta.cls_temp));
  }
  return { mu, sd, p };
}

export const CHANGES = ['d_logq', 'd_rr', 'd_bfi', 'd_logq5', 'd_logq95', 'd_hfd'] as const;
export type Change = (typeof CHANGES)[number];
export const LEVEL: Record<Change, keyof Sigs> = { d_logq: 'q_mean', d_rr: 'runoff_ratio', d_bfi: 'bfi_lh', d_logq5: 'q5', d_logq95: 'q95', d_hfd: 'hfd_mean' };
export const LOGT = new Set<Change>(['d_logq', 'd_logq5', 'd_logq95']);

/** Level in the target decade from a calibration level and a change (basindetective.predict). */
export const level = (t: Change, cal: number, d: number) => (LOGT.has(t) ? Math.exp(Math.log(cal + EPSQ) + d) - EPSQ : cal + d);

export interface Reading extends Raw {
  x: Float64Array;
  sig: Sigs;
  get: (t: string) => [number, number];
  ms: number;
}

/** The whole pipeline the instrument runs on every change: what-if -> features -> network. */
export function read(pack: Pack<BDMeta>, rec: BasinRecord, w: WhatIf): Reading {
  const t0 = performance.now();
  const { cal, q, tgt } = apply(rec, w);
  const { x, names, sig } = featurize(cal, q, tgt);
  if (names.length !== pack.meta.names.length || names.some((n, i) => n !== pack.meta.names[i])) throw new Error('feature list mismatch');
  const r = forward(pack, x);
  const idx = Object.fromEntries(pack.meta.targets.map((tt, j) => [tt, j]));
  return { ...r, x, sig, get: (tt: string) => [r.mu[idx[tt]], r.sd[idx[tt]]], ms: performance.now() - t0 };
}
