// Parity hook for site/tests/parity.test.ts: run one reference case, return the raw network outputs.
import { features, forward, slice, NT, NF, type SMeta, type SExamples } from './net.ts';
import type { Pack } from '../../lib/nn/pack.ts';

let W: Pack<SMeta> | null = null;
let X: Pack<SExamples> | null = null;

export async function run(c: { example: number; variant: string }, load: (name: string) => Pack<any>) {
  W ??= load('weights');
  X ??= load('examples');
  const wide = X.t[`x${c.example}`], pad = X.meta.pad;
  const x = slice(wide, pad, c.variant === 'shift24' ? 8 : 0);
  if (c.variant === 'end24') for (let ch = 0; ch < 3; ch++) x.fill(NaN, (ch * NT + 48) * NF, (ch + 1) * NT * NF);
  return forward(W, features(x));
}
