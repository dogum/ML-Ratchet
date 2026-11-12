// Parity hook for site/tests/parity.test.ts: fold, bin and normalise one star's points (with the case's deterministic
// change: secondary eclipse, odd/even depths, third light or a bigger eclipsing body), run the network both ways in
// time, return the 24 raw outputs.
import { readStar, points, type TMeta, type TExample } from './net.ts';
import type { Pack } from '../../lib/nn/pack.ts';

let W: Pack<TMeta> | null = null;
let X: Pack<TExample[]> | null = null;

export async function run(c: { example: number; variant: string; sec?: number; oe?: number; depth?: number }, load: (name: string) => Pack<any>) {
  W ??= load('weights');
  X ??= load('examples');
  const e = X.meta[c.example];
  return readStar(W, e, points(X, c.example), { sec: c.sec ?? 0, oe: c.oe ?? 0, depth: c.depth ?? 1 }).raw;
}
