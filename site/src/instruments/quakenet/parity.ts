// Parity hook for site/tests/parity.test.ts: run one reference case, return the raw network outputs.
import { forward, L, type QMeta } from './net.ts';
import type { Pack } from '../../lib/nn/pack.ts';

let W: Pack<QMeta> | null = null;
let X: Pack | null = null;

export async function run(c: { example: number; variant: string; s: [number, number] }, load: (name: string) => Pack<any>) {
  W ??= load('weights');
  X ??= load('examples');
  const x = X.t[`x${c.example}`];
  const data = new Float32Array(x.data);
  if (c.variant === 'cut900') for (let ch = 0; ch < 3; ch++) data.fill(0, ch * L + 900, (ch + 1) * L);
  return forward(W, { data, shape: x.shape }, c.s);
}
