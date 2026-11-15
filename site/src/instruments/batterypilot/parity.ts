// Parity hook for site/tests/parity.test.ts: rebuild the features from the real exam node-year, run the network at the
// case's hours (raw z knots) and the battery over the case's month (state of charge), for the case's battery settings.
import { battery, parityVector, pathOf, type BPMeta, type Path, type XMeta } from './net.ts';
import type { Pack } from '../../lib/nn/pack.ts';

let W: Pack<BPMeta> | null = null;
let X: Pack<XMeta> | null = null;
const paths: Record<number, Path> = {};

export async function run(c: { example: number; variant: string; d: number; eta: number; cost: number; hours: number[]; window: [number, number] },
  load: (name: string) => Pack<any>) {
  W ??= load('weights');
  X ??= load('examples');
  paths[c.example] ??= pathOf(X!, c.example);
  return parityVector(W!, paths[c.example], battery(c.d, c.eta, c.cost), c.hours, c.window);
}
