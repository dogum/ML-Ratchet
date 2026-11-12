// Parity hook for site/tests/parity.test.ts: one reference case = one exam piece of one star's light curve.
// The whole chain runs here as in the browser: light curve -> pipeline (dsp.ts) -> network (net.ts) -> raw outputs.
import { forward, type SMeta } from './net.ts';
import { prepare } from './dsp.ts';
import type { Pack } from '../../lib/nn/pack.ts';

let W: Pack<SMeta> | null = null;
let X: Pack | null = null;

export async function run(c: { example: number; variant: string; start: number; n: number }, load: (name: string) => Pack<any>) {
  W ??= load('weights');
  X ??= load('examples');
  const y = Float64Array.from(X.t[`y${c.example}`].data.subarray(c.start, c.start + c.n));
  const p = prepare(y);
  return forward(W, p.F1, p.F2, p.s);
}
