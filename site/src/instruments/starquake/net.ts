// StarQuake in the browser: a line-for-line port of models/starquake/starquake.py Net.forward() and read_ppm().
// Two branches: 1-D on the log power spectrum, 2-D on the comb map, plus two scalars (length, duty cycle).
import { conv1d, conv2d, gelu_, add_, linear, vgelu } from '../../lib/nn/ops.ts';
import type { Pack } from '../../lib/nn/pack.ts';
import type { Tensor } from '../../lib/nn/tensor.ts';
import { NB, NC, NL } from './dsp.ts';

export interface SMeta { n1: number; n2: number; n_head: number; mu_y: number[]; sd_y: number[]; widen?: number[] }
export interface Truth { numax: number; e_numax: number; dnu: number; e_dnu: number; teff: number; e_teff: number; M: number; R: number; kp: number; phase: number }
export interface Star { kic: string; truth: Truth; f1range: [number, number]; pieces: Record<string, [number, number]>; python: Record<string, Record<string, number>> }
export interface SExamples { stars: Star[]; dt: number; lens: number[] }

/** The head's five raw outputs: ln numax, ln dnu (standardised), their log-sigmas, and the detection logit. */
export function forward(W: Pack<SMeta>, F1: Float32Array, F2: Float32Array, s: [number, number]): Float32Array {
  const t = W.t, m = W.meta;
  const x1 = new Float32Array(NB);
  for (let i = 0; i < NB; i++) x1[i] = (F1[i] - 2.5) / 2;
  let h: Tensor = gelu_(conv1d({ data: x1, shape: [1, NB] }, t.s1_w, t.s1_b, 2, 3));
  for (let i = 0; i < m.n1; i++) {
    let a = gelu_(conv1d(h, t[`r1_${i}_c1_w`], t[`r1_${i}_c1_b`], 2, 3));
    a = conv1d(a, t[`r1_${i}_c2_w`], t[`r1_${i}_c2_b`], 1, 3);
    h = gelu_(add_(a, conv1d(h, t[`r1_${i}_sk_w`], t[`r1_${i}_sk_b`], 2, 0)));
  }
  const h1 = h.data;
  const x2 = new Float32Array(NC * NL);
  for (let i = 0; i < x2.length; i++) x2[i] = F2[i] * 4;
  let g: Tensor = gelu_(conv2d({ data: x2, shape: [1, NC, NL] }, t.s2_w, t.s2_b, [1, 2], 1));
  for (let i = 0; i < m.n2; i++) {
    let a = gelu_(conv2d(g, t[`r2_${i}_c1_w`], t[`r2_${i}_c1_b`], [2, 2], 1));
    a = conv2d(a, t[`r2_${i}_c2_w`], t[`r2_${i}_c2_b`], [1, 1], 1);
    g = gelu_(add_(a, conv2d(g, t[`r2_${i}_sk_w`], t[`r2_${i}_sk_b`], [2, 2], 0)));
  }
  let v: Float32Array = new Float32Array(h1.length + g.data.length + 2);
  v.set(h1, 0); v.set(g.data, h1.length); v[v.length - 2] = s[0]; v[v.length - 1] = s[1];
  for (let j = 0; j < m.n_head; j++) {
    v = linear(v, t[`h${j}_w`], t[`h${j}_b`]);
    if (j < m.n_head - 1) vgelu(v);
  }
  return v;
}

/** Scaling relations (no Delta-nu correction), solar references of Yu et al. 2018 — starquake.scaling(). */
export function scaling(numax: number, dnu: number, teff: number, eN = 0, eD = 0, eT = 0) {
  const a = numax / 3090, b = dnu / 135.1, t = teff / 5777;
  const M = a ** 3 * b ** -4 * t ** 1.5, R = a * b ** -2 * t ** 0.5;
  const fn = eN / numax, fd = eD / dnu, ft = eT / teff;
  return { mass: M, mass_err: M * Math.hypot(3 * fn, 4 * fd, 1.5 * ft), radius: R, radius_err: R * Math.hypot(fn, 2 * fd, 0.5 * ft) };
}

export interface Reading { p_osc: number; numax: number; numax_err: number; dnu: number; dnu_err: number; mass: number; mass_err: number; radius: number; radius_err: number }

/** read_ppm()'s post-processing: the raw outputs -> numax, dnu with (widened) error bars, detection, mass and radius. */
export function interpret(W: Pack<SMeta>, v: Float32Array, teff: number, eTeff: number): Reading {
  const m = W.meta, wd = m.widen ?? [1, 1];
  const ln = [v[0] * m.sd_y[0] + m.mu_y[0], v[1] * m.sd_y[1] + m.mu_y[1]];
  const sg = [Math.exp(Math.min(3, Math.max(-7, v[2]))) * m.sd_y[0], Math.exp(Math.min(3, Math.max(-7, v[3]))) * m.sd_y[1]];
  const numax = Math.exp(ln[0]), dnu = Math.exp(ln[1]);
  const numax_err = numax * sg[0] * wd[0], dnu_err = dnu * sg[1] * wd[1];
  const p_osc = 1 / (1 + Math.exp(-Math.min(30, Math.max(-30, v[4]))));
  return { p_osc, numax, numax_err, dnu, dnu_err, ...scaling(numax, dnu, teff, numax_err, dnu_err, eTeff) };
}
