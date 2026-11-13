// Parity hook for site/tests/parity.test.ts: rebuild one reference case's input exactly as the UI does, run the network,
// return the raw outputs (arch, strat, mu, sig, then each window's ledger mu and sig).
import { examRaw, labRaw, apply, run as runNet, flat, type TMeta, type XMeta } from './net.ts';
import type { Pack } from '../../lib/nn/pack.ts';

let W: Pack<TMeta> | null = null;
let X: Pack<XMeta> | null = null;

interface Case { example: number; variant: string; set: 'exam' | 'lab'; trips?: number[]; drop?: string[]; cut?: number | null }

export async function run(c: Case, load: (name: string) => Pack<any>) {
  W ??= load('weights');
  X ??= load('examples');
  if (c.set === 'lab') return flat(runNet(W, [0, 1].map((r) => labRaw(X!, c.example, r))).h);
  const car = X.meta.cars[c.example];
  const trips = (c.trips ?? [0]).map((i) => {
    const t = car.trips[i];
    return apply(examRaw(X!, t), t.flags, c.drop ?? [], c.cut ?? null);
  });
  return flat(runNet(W, trips).h);
}
