// Parity hook for site/tests/parity.test.ts: apply the case's control settings to the real exam day, run the network,
// return the same vector the exporter wrote (19 logits, 19 parameters, the energy output, P(fault)).
import { forward, perturb, type FMeta, type FDay, type Perturb } from './net.ts';
import type { Pack } from '../../lib/nn/pack.ts';

let W: Pack<FMeta> | null = null;
let X: Pack<FDay[]> | null = null;

export async function run(c: { example: number; variant: string; p: Perturb }, load: (name: string) => Pack<any>) {
  W ??= load('weights');
  X ??= load('examples');
  const i = c.example;
  return Array.from(forward(W, perturb(X.t[`x${i}`].data, X.t[`n${i}`].data, c.p)));
}
