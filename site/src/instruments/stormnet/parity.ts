import type { Pack } from '../../lib/nn/pack.ts';
import { frameOf, predict, type Examples, type StormWeights } from './net.ts';

export function run(c: { example: number; frame: number; shift: number }, load: (s: string) => Pack<unknown>) {
  const frame = frameOf(load('examples') as Pack<Examples>, c.example, c.frame, c.shift);
  const r = predict(load('weights') as Pack<StormWeights>, frame);
  return [r.minimum, r.sigma, r.classical];
}
