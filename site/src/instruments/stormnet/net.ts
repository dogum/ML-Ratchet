// StormNet in the browser: the released readout plus every competitor it was examined against.
// Mirrors models/stormnet/{stormnet.py,stormphysics.py}; parity cases in site/public/instruments/stormnet/reference.json.
import type { Pack } from '../../lib/nn/pack.ts';

export interface StormWeights {
  mode: string;
  version?: string;
  physics: number[];        // the anchor (dev-tuned two-state)
  classical: number[];      // dev-tuned two-state, as examined
  single: number[];         // dev-tuned single-reservoir O'Brien–McPherron
  published: number[];      // published O'Brien–McPherron coefficients
  wind_law: number[];
  widen_factor: number;
  widen_by_dst?: { edges: number[]; factors: number[] };
}

export interface StormInfo {
  label: string; source: string; block: string; minimum: number; hours: number;
  start_utc: string; issue_from: number; length: number;
}
export interface Storms { storms: StormInfo[] }

/** One hour of solar wind: GSM Bz (nT), bulk speed (km/s), dynamic pressure (nPa). */
export type WindRow = [number, number, number];

export interface Scenario { bz: number; speed: number; pressure: number }
export const RECORDED: Scenario = { bz: 0, speed: 1, pressure: 1 };

export interface Reading {
  minimum: number; sigma: number;            // StormNet: next-three-hour minimum Dst and its 68% half-width
  classical: number; single: number; published: number; persistence: number;
  path: number[];                            // the anchor's three hourly values
  fast: number; slow: number;                // ring-current reservoirs at issue (nT, negative = stronger)
}

export function stormSeries(pack: Pack<Storms>, i: number) {
  const wind = pack.t['wind' + i].data, dst = pack.t['dst' + i].data;
  const info = pack.meta.storms[i];
  const n = info.length;
  const rows: WindRow[] = Array.from({ length: n }, (_, j) => [wind[3 * j], wind[3 * j + 1], wind[3 * j + 2]]);
  return { info, rows, dst: Array.from(dst) };
}

/** The 36-hour wind window ending at hour k, with a what-if applied to its last three hours. */
export function windowAt(rows: WindRow[], k: number, s: Scenario = RECORDED): WindRow[] | null {
  if (k < 35) return null;
  const w = rows.slice(k - 35, k + 1).map((r, j): WindRow => j >= 33
    ? [r[0] + s.bz, r[1] * s.speed, r[2] * s.pressure] : [r[0], r[1], r[2]]);
  return w.every(r => r.every(Number.isFinite) && r[1] > 0 && r[2] > 0) ? w : null;
}

function step(state: number, field: number, gain: number, scale: number, share: number) {
  const tau = scale * 2.4 * Math.exp(9.74 / (4.69 + field));
  return Math.exp(-1 / tau) * state - gain * share * Math.max(field - .49, 0) * tau * -Math.expm1(-1 / tau);
}

export function projected(e: number[], p: number[], dst: number, params: number[], law: number[]) {
  const [gain, tf, ts, share, b, c] = params;
  let fast = 0, slow = 0;
  for (const value of e) {
    fast = step(fast, value, gain, tf, share);
    slow = step(slow, value, gain, ts, 1 - share);
  }
  const total = fast + slow;
  const fraction = Math.abs(total) > 1e-8 ? fast / total : share;
  let field = e[e.length - 1], pressure = p[p.length - 1];
  const corrected = dst - b * pressure + c;
  fast = fraction * corrected;
  slow = (1 - fraction) * corrected;
  const startFast = fast, startSlow = slow;
  const path: number[] = [];
  for (let i = 0; i < 3; i++) {
    field = law[0] + law[1] * (field - law[0]);
    pressure = law[2] + law[3] * (pressure - law[2]);
    fast = step(fast, field, gain, tf, share);
    slow = step(slow, field, gain, ts, 1 - share);
    path.push(fast + slow + b * pressure - c);
  }
  return { path, total, fraction, minimum: Math.min(...path), fast: startFast, slow: startSlow };
}

const meanTail = (a: number[], n: number) => a.slice(-n).reduce((s, v) => s + v, 0) / n;

export function drivers(wind: WindRow[]) {
  return { e: wind.map(([bz, speed]) => Math.max(-bz, 0) * speed / 1000), p: wind.map(r => Math.sqrt(r[2])) };
}

export function predict(weights: Pack<StormWeights>, wind: WindRow[], dst: number, calibrated = true): Reading {
  const { meta, t } = weights;
  const { e, p } = drivers(wind);
  const x = [dst, e[35], p[35], wind[35][1] / 1000, wind[35][0]];
  let anchor = dst;
  const phys = projected(e, p, dst, meta.physics, meta.wind_law);
  if (meta.mode === 'history') {
    anchor = phys.minimum;
    for (const n of [3, 12, 36]) x.push(meanTail(e, n), meanTail(p, n));
    x.push(phys.fraction, phys.total, anchor - dst, phys.path[2] - phys.path[0]);
  }
  const z = x.map((v, i) => Math.max(-8, Math.min(8, (v - t.mean.data[i]) / t.scale.data[i])));
  const h = [1, ...z];
  const size = t.bias.data.length;
  for (let j = 0; j < size; j++) {
    let a = t.bias.data[j];
    for (let i = 0; i < z.length; i++) a += z[i] * t.kernel.data[i * size + j];
    h.push(Math.tanh(a));
  }
  const dot = (key: string) => h.reduce((sum, v, i) => sum + v * t[key].data[i], 0);
  let sigma = Math.exp(Math.max(-1, Math.min(5, dot('spread'))));
  if (calibrated) {
    const by = meta.widen_by_dst;
    if (by) {
      let k = 0;
      while (k < by.edges.length && dst >= by.edges[k]) k++;
      sigma *= by.factors[k];
    } else sigma *= meta.widen_factor;
  }
  return {
    minimum: anchor + dot('location'), sigma,
    classical: projected(e, p, dst, meta.classical, meta.wind_law).minimum,
    single: projected(e, p, dst, meta.single, meta.wind_law).minimum,
    published: projected(e, p, dst, meta.published, meta.wind_law).minimum,
    persistence: dst, path: phys.path, fast: phys.fast, slow: phys.slow,
  };
}

/** Every eligible issue hour of a storm, run through all forecasters (a few ms for 280 hours). */
export function replay(weights: Pack<StormWeights>, rows: WindRow[], dst: number[]) {
  const out: (Reading & { k: number; truth: number })[] = [];
  for (let k = 35; k + 3 < dst.length; k++) {
    const w = windowAt(rows, k);
    const future = dst.slice(k + 1, k + 4);
    if (!w || !Number.isFinite(dst[k]) || !future.every(Number.isFinite)) continue;
    out.push({ ...predict(weights, w, dst[k]), k, truth: Math.min(...future) });
  }
  return out;
}
