// Parity hook for site/tests/parity.test.ts: the four real exam basins under recorded and what-if inputs.
// Each case re-runs the feature port AND the ensemble; outputs = 11 means, 11 sigmas, p_regulated, 150 features
// (NaN = -999), as written by models/basindetective/web/export_instrument.py.
import type { Pack } from '../../lib/nn/pack.ts';
import { read, type BDMeta } from './net.ts';
import { basinRecord, type Examples, type WhatIf } from './whatif.ts';

export function run(c: WhatIf & { example: number }, load: (s: string) => Pack<unknown>) {
  const rec = basinRecord(load('examples') as Pack<Examples>, c.example);
  const r = read(load('weights') as Pack<BDMeta>, rec, c);
  return [...r.mu, ...r.sd, r.p, ...Array.from(r.x, (v) => (Number.isNaN(v) ? -999 : v))];
}
