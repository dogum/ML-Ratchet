// Parity hook for site/tests/parity.test.ts: rebuild one reference case's input in JS (faults + channels()), run the network.
import { channels, forward, inject, fortnight, type SMeta, type SWindow, type Faults } from './net.ts';
import type { Pack } from '../../lib/nn/pack.ts';

let W: Pack<SMeta> | null = null;
let X: Pack<SWindow[]> | null = null;

export async function run(c: { example: number; variant: string; home: boolean; inj: Faults }, load: (name: string) => Pack<any>) {
  W ??= load('weights');
  X ??= load('examples');
  const m = X.meta[c.example], s = fortnight(X, c.example);
  const p = inject(s, m.n, m.ceiling, c.inj ?? {});
  return forward(W, channels(p, s, c.home ? m.ref.home : m.ref.poa, c.home));
}
