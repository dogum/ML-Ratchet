// Parity hook for site/tests/parity.test.ts: the eight real exam ignitions under recorded and what-if inputs.
// Each case re-runs the what-if, the Rothermel inputs and the ensemble; outputs = 12 widened probabilities, 12 raw
// (T = 1), P(escape), P(large), median acres, the 80 % range (lo, hi) and the 75 network inputs, as written by
// models/firesize/web/export_instrument.py.
import type { Pack } from '../../lib/nn/pack.ts';
import { codesOf, lutOf, read, type ExMeta, type FSMeta, type WhatIf } from './net.ts';

export function run(c: WhatIf & { example: number }, load: (s: string) => Pack<unknown>) {
  const weights = load('weights') as Pack<FSMeta>, ex = load('examples') as Pack<ExMeta>;
  const w: WhatIf = { dWind: c.dWind, dRH: c.dRH, dT: c.dT, dERC: c.dERC, hour: c.hour, cause: c.cause, fed: c.fed, region: c.region };
  const r = read(weights, lutOf(weights), ex.meta.ignitions[c.example], codesOf(ex, c.example), w);
  return [...r.p, ...r.p1, r.pEscape, r.pLarge, r.median, r.lo, r.hi, ...r.x];
}
