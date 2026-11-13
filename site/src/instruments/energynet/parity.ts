// Parity hook for site/tests/parity.test.ts: apply the case's control settings to the real exam trip, run the
// features and the ensemble, return the same vector the exporter wrote (features, members' raw outputs, predict()).
import { predict, perturb, parityVector, FIELDS, type EMeta, type EExample, type Perturb, type Trip } from './net.ts';
import type { Pack } from '../../lib/nn/pack.ts';

let W: Pack<EMeta> | null = null;
let X: Pack<EExample[]> | null = null;

export function tripOf(X: Pack<EExample[]>, i: number): Trip {
  const t = {} as Record<string, Float32Array>;
  for (const k of FIELDS) t[k] = X.t[`t${i}_${k}`].data;
  return t as unknown as Trip;
}

export async function run(c: { example: number; variant: string; p: Perturb }, load: (name: string) => Pack<any>) {
  W ??= load('weights');
  X ??= load('examples');
  const { trip, veh } = perturb(tripOf(X, c.example), X.meta[c.example].vehicle, c.p);
  return parityVector(predict(W, trip, veh));
}
