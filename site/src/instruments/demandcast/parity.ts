// Parity hook for site/tests/parity.test.ts: apply the case's control settings to the real exam day, re-run the
// building-stock physics, the wind/solar tables and the three networks, return the vector the exporter wrote.
import { predict, dayOf, parityVector, type DMeta, type DDay, type Perturb } from './net.ts';
import type { Pack } from '../../lib/nn/pack.ts';

let W: Pack<DMeta> | null = null;
let X: Pack<DDay[]> | null = null;

export async function run(c: { example: number; variant: string; p: Perturb }, load: (name: string) => Pack<any>) {
  W ??= load('weights');
  X ??= load('examples');
  return parityVector(predict(W, dayOf(X, c.example), c.p));
}
