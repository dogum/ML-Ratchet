// Parity hook for site/tests/parity.test.ts: apply the case's control settings to the real exam day, run the network,
// return the same vector the exporter wrote (19 logits, 19 parameters, the energy output, P(fault)). Cases with
// refs: true read the day with its reference days (v1.1), as the instrument does by default.
import { forward, perturb, type FMeta, type FDay, type Perturb } from './net.ts';
import type { Pack } from '../../lib/nn/pack.ts';

let W: Pack<FMeta> | null = null;
let X: Pack<FDay[]> | null = null;

export async function run(c: { example: number; variant: string; p: Perturb; refs?: boolean }, load: (name: string) => Pack<any>) {
  W ??= load('weights');
  X ??= load('examples');
  const i = c.example;
  const refs = c.refs ? X.meta[i].refs.map((_, j) => X!.t[`r${i}_${j}`].data) : [];
  return Array.from(forward(W, perturb(X.t[`x${i}`].data, X.t[`n${i}`].data, c.p), refs));
}
