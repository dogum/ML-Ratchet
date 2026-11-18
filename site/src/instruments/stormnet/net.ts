import type { Pack } from '../../lib/nn/pack.ts';

export interface StormWeights {
  mode: string;
  physics: number[];
  classical: number[];
  wind_law: number[];
  widen_factor: number;
}

export interface StormRecord { label: string; issues: string[]; block: string }
export interface Examples { records: StormRecord[] }
export interface Frame { wind: number[][]; dst: number; past: number[]; future: number[] }
export interface Reading { minimum: number; sigma: number; classical: number }

export function frameOf(pack: Pack<Examples>, i: number, frame: number, shift: number): Frame {
  const values = (key: string, width: number) => Array.from(
    pack.t[key + i].data.subarray(frame * width, (frame + 1) * width));
  const flat = values('wind', 36 * 3);
  const wind = Array.from({ length: 36 }, (_, j) =>
    [flat[3 * j] + (j >= 33 ? shift : 0), flat[3 * j + 1], flat[3 * j + 2]]);
  return { wind, dst: values('last_dst', 1)[0], past: values('past_dst', 36), future: values('future_dst', 3) };
}

function step(state: number, field: number, gain: number, scale: number, share: number) {
  const tau = scale * 2.4 * Math.exp(9.74 / (4.69 + field));
  return Math.exp(-1 / tau) * state - gain * share * Math.max(field - .49, 0) * tau * -Math.expm1(-1 / tau);
}

function projected(e: number[], p: number[], dst: number, params: number[], law: number[]) {
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
  const path: number[] = [];
  for (let i = 0; i < 3; i++) {
    field = law[0] + law[1] * (field - law[0]);
    pressure = law[2] + law[3] * (pressure - law[2]);
    fast = step(fast, field, gain, tf, share);
    slow = step(slow, field, gain, ts, 1 - share);
    path.push(fast + slow + b * pressure - c);
  }
  return { path, total, fraction, minimum: Math.min(...path) };
}

const meanTail = (a: number[], n: number) => a.slice(-n).reduce((s, v) => s + v, 0) / n;

export function predict(weights: Pack<StormWeights>, frame: Pick<Frame, 'wind' | 'dst'>): Reading {
  const { wind, dst } = frame, { meta, t } = weights;
  const e = wind.map(([bz, speed]) => Math.max(-bz, 0) * speed / 1000);
  const p = wind.map((r) => Math.sqrt(r[2]));
  const x = [dst, e[35], p[35], wind[35][1] / 1000, wind[35][0]];
  let anchor = dst;
  const classical = projected(e, p, dst, meta.classical, meta.wind_law);
  if (meta.mode === 'history') {
    const phys = projected(e, p, dst, meta.physics, meta.wind_law);
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
  return { minimum: anchor + dot('location'), sigma: Math.exp(Math.max(-1, Math.min(5, dot('spread')))) * meta.widen_factor,
    classical: classical.minimum };
}
