// Parity hook for site/tests/parity.test.ts: apply the case's control settings to the real exam unit, compute the
// features in JS, run the network, and return the vector the exporter wrote: 12 means and 12 widened sigmas in
// target space, then the class probabilities (use 12, heat 5, cool 4).
import { forward, inputs, CH, type Hourly, type Perturb, type Unit, type WMeta } from './net.ts';
import type { Pack } from '../../lib/nn/pack.ts';

let W: Pack<WMeta> | null = null;
let X: Pack<Unit[]> | null = null;

export function hourlyUnit(X: Pack<Unit[]>, i: number): Hourly | null {
  const u = X.meta[i];
  if (!u.channels) return null;
  const t = (k: string) => X.t[`${k}${i}`].data;
  const meters: Hourly['meters'] = {}, z: Hourly['z'] = {};
  for (const c of CH) if (X.t[`m${i}_${c}`]) { meters[c] = X.t[`m${i}_${c}`].data; z[c] = X.t[`z${i}_${c}`].data; }
  return { meters, z, T: t('T'), G: t('G'), tm: { hr: t('hr'), dow: t('dow'), day: t('day'), month: t('mon') }, lat: u.lat! };
}

export async function run(c: { example: number; variant: string; p: Perturb }, load: (name: string) => Pack<any>) {
  W ??= load('weights');
  X ??= load('examples');
  const { mode, X: x } = inputs(hourlyUnit(X, c.example), X.meta[c.example], c.p);
  return Array.from(forward(W, mode, x));
}
