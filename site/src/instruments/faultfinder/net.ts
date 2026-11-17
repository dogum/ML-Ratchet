// FaultFinder in the browser: a port of models/faultfinder/faultfinder.py (featurize, forward, predict_raw), started
// from the release's training/ffnet.js, and of perturb() in models/faultfinder/web/export_instrument.py (what each
// control does to the trend log). No DOM.
import { conv1d, relu_, globalPool, linear, vcat } from '../../lib/nn/ops.ts';
import type { Pack } from '../../lib/nn/pack.ts';
import type { Tensor } from '../../lib/nn/tensor.ts';

export const RAWC = ['T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_sa_sp', 'T_zone', 'damper_oa', 'valve_cc', 'valve_hc', 'fan_speed',
  'fan_power', 'fan_on', 'occ'] as const;
export const R: Record<string, number> = Object.fromEntries(RAWC.map((c, i) => [c, i]));
export const NT = 96, NF = 35, NC = 19;
export const TEMPS = ['T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_zone'];

export interface FMeta { temperature: number; classes: string[]; labels: Record<string, string>; n_params: number; channels: string[] }
export interface FDay {
  id: string; source: string; facility: string; dataset: string; unit: string; date: string; fault: string;
  severity: string | null; diag_class: string | null; simulable: boolean;
  rule: { name: string; flag: boolean; score: number | null; threshold: number; rules_fired: string[] };
  page_p: number; python_p: number; credit: string; drawn: string; has: boolean[];
}

/** The instrument's controls (see perturb() in the exporter). */
export interface Perturb {
  sensor?: string; // which temperature gets the offset
  offset?: number; // K
  damper?: number | null; // outdoor-air damper stuck at this fraction (mixing-box balance), or none
  noise?: number; // K, times the day's fixed standard-normal pattern on the five measured temperatures
  drop_ma?: boolean; // mixed-air sensor removed
}

const fin = (v: number) => Number.isFinite(v);

/** perturb(): x [13 * 96] raw day (NaN = missing), nz [5 * 96]. Returns a new Float64Array. */
export function perturb(x: ArrayLike<number>, nz: ArrayLike<number>, p: Perturb): Float64Array {
  const o = Float64Array.from(x);
  const row = (c: string) => R[c] * NT;
  if (p.damper != null) {
    const f = p.damper, on = row('fan_on'), oa = row('T_oa'), ra = row('T_ra'), ma = row('T_ma');
    for (let i = 0; i < NT; i++) {
      const run = fin(o[on + i]) ? o[on + i] > 0.5 : true;
      if (run && fin(o[oa + i]) && fin(o[ra + i])) o[ma + i] = o[ra + i] + f * (o[oa + i] - o[ra + i]);
    }
  }
  if (p.offset && p.sensor) { const s = row(p.sensor); for (let i = 0; i < NT; i++) o[s + i] += p.offset; }
  if (p.noise) TEMPS.forEach((c, j) => { const s = row(c); for (let i = 0; i < NT; i++) o[s + i] += p.noise! * nz[j * NT + i]; });
  if (p.drop_ma) { const s = row('T_ma'); for (let i = 0; i < NT; i++) o[s + i] = NaN; }
  return o;
}

function fill(x: ArrayLike<number>, off: number): Float64Array { // np.interp over missing samples, flat ends
  const idx: number[] = [];
  for (let i = 0; i < NT; i++) if (fin(x[off + i])) idx.push(i);
  const o = new Float64Array(NT);
  if (!idx.length) return o;
  let a = 0;
  for (let i = 0; i < NT; i++) {
    const v = x[off + i];
    if (fin(v)) { o[i] = v; continue; }
    if (i <= idx[0]) { o[i] = x[off + idx[0]]; continue; }
    if (i >= idx[idx.length - 1]) { o[i] = x[off + idx[idx.length - 1]]; continue; }
    while (idx[a + 1] < i) a++;
    const i0 = idx[a], i1 = idx[a + 1];
    o[i] = x[off + i0] + ((x[off + i1] - x[off + i0]) * (i - i0)) / (i1 - i0);
  }
  return o;
}
function pct(v: number[], q: number) { // np.percentile, linear
  const s = v.slice().sort((a, b) => a - b), r = (q / 100) * (s.length - 1), lo = Math.floor(r), hi = Math.ceil(r);
  return s[lo] + (s[hi] - s[lo]) * (r - lo);
}
const clip = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));

/** faultfinder.featurize: raw day [13 * 96] -> features [35, 96]. */
export function featurize(x: ArrayLike<number>): Tensor {
  const f = new Float32Array(NF * NT), avail: boolean[] = [];
  for (let c = 0; c < 13; c++) { let any = false; for (let i = 0; i < NT; i++) if (fin(x[c * NT + i])) { any = true; break; } avail.push(any); }
  const X = RAWC.map((_, c) => fill(x, c * NT));
  const fpv: number[] = [];
  for (let i = 0; i < NT; i++) { const v = x[R.fan_power * NT + i]; if (fin(v)) fpv.push(v); }
  if (fpv.length) { const p = pct(fpv, 95); for (let i = 0; i < NT; i++) X[R.fan_power][i] = p > 1e-9 ? X[R.fan_power][i] / p : 0; }
  let k = 0;
  const set = (rw: number, fn: (i: number) => number) => { for (let i = 0; i < NT; i++) f[rw * NT + i] = fn(i); };
  for (const c of ['T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_sa_sp', 'T_zone']) { if (avail[R[c]]) set(k, (i) => (X[R[c]][i] - 20.0) / 10.0); k++; }
  for (const [a, b, s] of [['T_ma', 'T_ra', 5], ['T_sa', 'T_ma', 5], ['T_sa', 'T_sa_sp', 3], ['T_oa', 'T_ra', 10], ['T_zone', 'T_ra', 3]] as [string, string, number][]) {
    if (avail[R[a]] && avail[R[b]]) set(k, (i) => clip((X[R[a]][i] - X[R[b]][i]) / s, -5, 5));
    k++;
  }
  const frm = new Float32Array(NT);
  if (avail[R.T_oa] && avail[R.T_ma] && avail[R.T_ra]) {
    set(k, (i) => {
      const den = X[R.T_oa][i] - X[R.T_ra][i];
      if (Math.abs(den) > 3) { frm[i] = 1; return clip((X[R.T_ma][i] - X[R.T_ra][i]) / den, -0.5, 1.5); }
      return 0;
    });
  }
  k++;
  for (const c of ['damper_oa', 'valve_cc', 'valve_hc', 'fan_speed', 'fan_power', 'fan_on', 'occ']) { if (avail[R[c]]) set(k, (i) => clip(X[R[c]][i], -0.2, 2.0)); k++; }
  set(k, (i) => Math.sin((i / NT) * 2 * Math.PI)); set(k + 1, (i) => Math.cos((i / NT) * 2 * Math.PI)); k += 2;
  for (let c = 0; c < 13; c++) set(k + c, (i) => (fin(x[c * NT + i]) ? 1 : 0));
  set(k + 13, (i) => frm[i]);
  return { data: f, shape: [NF, NT] };
}

function pool2(x: Tensor): Tensor { // max over pairs, odd tail dropped
  const [C, L] = x.shape, L2 = L >> 1, o = new Float32Array(C * L2);
  for (let c = 0; c < C; c++) for (let l = 0; l < L2; l++) o[c * L2 + l] = Math.max(x.data[c * L + 2 * l], x.data[c * L + 2 * l + 1]);
  return { data: o, shape: [C, L2] };
}

/** faultfinder.forward + the calibrated detection: [19 logits, 19 parameters, energy, P(fault)]. */
export function forward(W: Pack<FMeta>, x: ArrayLike<number>): Float64Array {
  const t = W.t;
  let h = relu_(conv1d(featurize(x), t.c1_W, t.c1_b));
  h = pool2(relu_(conv1d(h, t.c2_W, t.c2_b)));
  h = pool2(relu_(conv1d(h, t.c3_W, t.c3_b)));
  h = relu_(conv1d(h, t.c4_W, t.c4_b));
  const g = globalPool(h);
  const z = linear(vcat(g.mean, g.max), t.fc_W, t.fc_b);
  for (let i = 0; i < z.length; i++) if (z[i] < 0) z[i] = 0;
  const lo = linear(z, t.cls_W, t.cls_b), pa = linear(z, t.par_W, t.par_b), wa = linear(z, t.waste_W, t.waste_b);
  let m = -Infinity;
  for (let i = 1; i < NC; i++) m = Math.max(m, lo[i]);
  let s = 0;
  for (let i = 1; i < NC; i++) s += Math.exp(lo[i] - m);
  const d = m + Math.log(s) - lo[0];
  const out = new Float64Array(2 * NC + 2);
  out.set(lo, 0); out.set(pa, NC); out[2 * NC] = wa[0]; out[2 * NC + 1] = 1 / (1 + Math.exp(-d / W.meta.temperature));
  return out;
}

export interface Reading { p: number; diag: { cls: string; label: string; p: number }[]; energy: number; raw: Float64Array }

/** predict_raw for one day: P(fault) and the diagnosis softmax(fault logits / T), sorted. */
export function read(W: Pack<FMeta>, x: ArrayLike<number>): Reading {
  const o = forward(W, x), T = W.meta.temperature;
  const z = Array.from(o.subarray(1, NC), (v) => v / T), m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
  const diag = e.map((v, i) => ({ cls: W.meta.classes[i + 1], label: W.meta.labels[W.meta.classes[i + 1]], p: v / s })).sort((a, b) => b.p - a.p);
  return { p: o[2 * NC + 1], diag, energy: o[2 * NC], raw: o };
}
