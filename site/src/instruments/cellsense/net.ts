// CellSense in the browser: a line-for-line port of models/cellsense/cellsense.py — features() (NaN fill, masks,
// the 24 mV smoothing), CellSenseNet.forward() and the routing in CellSense.diagnose().
import { conv1d, gelu_, add_, globalPool, linear, vcat, vgelu } from '../../lib/nn/ops.ts';
import type { Pack } from '../../lib/nn/pack.ts';
import type { Tensor } from '../../lib/nn/tensor.ts';

export interface NetMeta {
  blocks: [number, number, number][];
  labels: string[];
  extra: number;
  smooth?: number;
  mu_y: number[];
  sd_y: number[];
}
export interface CMeta { health: NetMeta; life: NetMeta; partial: NetMeta }
export type NetName = 'health' | 'life' | 'partial';
const PREFIX: Record<NetName, string> = { health: 'h', life: 'l', partial: 'p' };

export interface CExamples {
  cell: string; policy: string; charge_time: number; cycle_life: number; cycles: number[];
  measured_q_total: (number | null)[]; measured_ir_mohm: (number | null)[]; note: string;
}

export const L = 250;

/** Gaussian smoothing along the voltage axis with the edges held (cellsense.gsmooth), one row. */
export function gsmooth(a: Float64Array, sigma: number): Float64Array {
  if (!sigma) return a;
  const r = Math.floor(4 * sigma + 0.5);
  const k = new Float64Array(2 * r + 1);
  let s = 0;
  for (let i = -r; i <= r; i++) { k[i + r] = Math.exp(-0.5 * (i / sigma) ** 2); s += k[i + r]; }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  const n = a.length, out = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    let v = 0;
    for (let i = -r; i <= r; i++) v += k[i + r] * a[Math.max(0, Math.min(n - 1, j + i))];
    out[j] = v;
  }
  return out;
}

/** Carry the last finite value forward (and the first one back); all-missing -> default. Returns values and mask. */
function fill(a: ArrayLike<number> | null, def: number): [Float64Array, Float64Array] {
  const out = new Float64Array(L), m = new Float64Array(L);
  if (!a) { out.fill(def); m.fill(1); return [out, m]; }
  let first = -1;
  for (let i = 0; i < L; i++) if (Number.isFinite(a[i])) { m[i] = 1; if (first < 0) first = i; }
  if (first < 0) { out.fill(def); return [out, m]; }
  let last = a[first];
  for (let i = 0; i < L; i++) { if (Number.isFinite(a[i])) last = a[i]; out[i] = i < first ? a[first] : last; }
  return [out, m];
}

/** features(Q1, T1, Q0, T0, smooth): the 8 input channels x 250 voltage levels. */
export function features(Q1: ArrayLike<number>, T1: ArrayLike<number>, Q0: ArrayLike<number> | null, T0: ArrayLike<number> | null, smooth = 0): Tensor {
  let has = 0;
  if (Q0) for (let i = 0; i < L; i++) if (Number.isFinite(Q0[i])) { has = 1; break; }
  let [q1, m1] = fill(Q1, 0), [t1] = fill(T1, 30);
  // no reference: Q0 = zeros, T0 = 30 (masked out by has = 0 anyway)
  let [q0, m0] = Q0 ? fill(Q0, 0) : [new Float64Array(L), new Float64Array(L).fill(1)];
  let [t0] = Q0 && T0 ? fill(T0, 30) : [new Float64Array(L).fill(30)];
  if (smooth) { q1 = gsmooth(q1, smooth); t1 = gsmooth(t1, smooth); q0 = gsmooth(q0, smooth); t0 = gsmooth(t0, smooth); }
  const x = new Float32Array(8 * L);
  for (let i = 0; i < L; i++) {
    x[i] = q1[i];
    x[L + i] = q0[i] * has;
    x[2 * L + i] = (q1[i] - q0[i]) * 10 * has;
    x[3 * L + i] = (t1[i] - 30) / 10;
    x[4 * L + i] = ((t0[i] - 30) / 10) * has;
    x[5 * L + i] = m1[i];
    x[6 * L + i] = m0[i] * has;
    x[7 * L + i] = has;
  }
  for (let i = 0; i < x.length; i++) if (!Number.isFinite(x[i])) x[i] = 0;
  return { data: x, shape: [8, L] };
}

/** CellSenseNet.forward for one curve: raw outputs (k means, then k log-sigmas). */
export function forward(W: Pack<CMeta>, net: NetName, X: Tensor, S: ArrayLike<number> | null = null): Float32Array {
  const m = W.meta[net], p = PREFIX[net], t = (k: string) => W.t[`${p}_${k}`];
  let h = gelu_(conv1d(X, t('stem_w'), t('stem_b'), 1, 3));
  m.blocks.forEach(([st, p1, p2], i) => {
    let y = gelu_(conv1d(h, t(`b${i}c1_w`), t(`b${i}c1_b`), st, p1));
    y = conv1d(y, t(`b${i}c2_w`), t(`b${i}c2_b`), 1, p2);
    h = gelu_(add_(y, conv1d(h, t(`b${i}sk_w`), t(`b${i}sk_b`), st, 0)));
  });
  const { mean, max } = globalPool(h);
  const z = m.extra && S ? vcat(mean, max, S) : vcat(mean, max);
  return linear(vgelu(linear(z, t('h0_w'), t('h0_b'))), t('h1_w'), t('h1_b'));
}

/** _run(): de-standardise the means and turn the log-sigmas into 68% error bars. */
export function decode(m: NetMeta, o: ArrayLike<number>): Record<string, number> {
  const k = m.labels.length, r: Record<string, number> = {};
  m.labels.forEach((n, j) => {
    r[n] = o[j] * m.sd_y[j] + m.mu_y[j];
    r[`${n}_sig`] = Math.exp(Math.min(3, Math.max(-7, o[k + j]))) * m.sd_y[j];
  });
  return r;
}

export const lifeS = (cycle: number, chargeMin: number) => [Math.log10(Math.max(cycle, 1)) / 3.5, chargeMin / 15, 1];

export interface Diagnosis extends Record<string, number> { partial: number }

/**
 * CellSense.diagnose for one discharge. A curve that stops before 2.0 V (NaN at the end) goes to the partial-discharge
 * network. The lifetime network runs only on full discharges here (diagnose() would also run it on a cut curve, which
 * it never learned: only the partial-discharge network was trained on curves cut short).
 */
export function diagnose(W: Pack<CMeta>, Q: ArrayLike<number>, T: ArrayLike<number>, Q0: ArrayLike<number> | null, T0: ArrayLike<number> | null,
  cycle: number, chargeMin: number): Diagnosis {
  const part = !Number.isFinite(Q[L - 1]);
  const net: NetName = part ? 'partial' : 'health';
  const m = W.meta[net];
  const r = decode(m, forward(W, net, features(Q, T, Q0, T0, m.smooth ?? 0))) as Diagnosis;
  r.partial = part ? 1 : 0;
  if (!part) {
    const ml = W.meta.life;
    const l = decode(ml, forward(W, 'life', features(Q, T, Q0, T0, ml.smooth ?? 0), lifeS(cycle, chargeMin)));
    for (const k of ['log_life', 'log_life_sig', 'left', 'left_sig']) r[k] = l[k];
    r.life = 10 ** r.log_life;
    r.life_lo = 10 ** (r.log_life - r.log_life_sig);
    r.life_hi = 10 ** (r.log_life + r.log_life_sig);
  }
  return r;
}
