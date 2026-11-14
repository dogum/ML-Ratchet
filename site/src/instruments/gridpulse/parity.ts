// Parity hook for site/tests/parity.test.ts: apply the case's control settings to the real GB exam window, run the
// network, return the same vector the exporter wrote (raw outputs: 7 means and 7 widened sigmas in target space).
import { forward, perturb, type GMeta, type GExample, type Perturb } from './net.ts';
import type { Pack } from '../../lib/nn/pack.ts';

let W: Pack<GMeta> | null = null;
let X: Pack<GExample[]> | null = null;

export async function run(c: { example: number; variant: string; p: Perturb }, load: (name: string) => Pack<any>) {
  W ??= load('weights');
  X ??= load('examples');
  const i = c.example;
  const { w, N, d } = perturb(X.t[`w${i}`].data, X.meta[i].demand_MW, X.t[`n${i}`].data, c.p);
  return Array.from(forward(W, w, N, d));
}
