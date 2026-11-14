// Parity hook for site/tests/parity.test.ts: cut the real exam unit's record as the case's controls say, run the
// features, the ensemble, the pooling and the readings, and return the same vector the exporter wrote.
import { read, series, parityVector, type PMeta, type XMeta, type Control, type Series } from './net.ts';
import type { Pack } from '../../lib/nn/pack.ts';

let W: Pack<PMeta> | null = null;
let X: Pack<XMeta> | null = null;
const S: Record<number, Series> = {};

export async function run(c: { example: number; variant: string; p: Control }, load: (name: string) => Pack<any>) {
  W ??= load('weights');
  X ??= load('examples');
  S[c.example] ??= series(X, c.example);
  const { r } = read(W, S[c.example], X.meta.units[c.example], c.p);
  return parityVector(W.meta, r);
}
