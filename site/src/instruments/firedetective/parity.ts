// Parity hook for site/tests/parity.test.ts: the four real exam fires under recorded and what-if inputs. Each case
// re-runs the feature port AND the ensemble; outputs = 13 readable outputs, the 12 x 25 sequence features, the
// wind-blind copy's wind columns and flag (12 x 4) and the 17 static features, as written by
// models/firedetective/web/export_instrument.py.
import type { Pack } from '../../lib/nn/pack.ts';
import { HIDDEN_COL, NF, T_STEPS, WIND_COLS } from './features.ts';
import { OUTS, read, type FDMeta } from './net.ts';
import { apply, fireRecord, type Examples, type WhatIf } from './whatif.ts';

export function run(c: WhatIf & { example: number }, load: (s: string) => Pack<unknown>) {
  const rec = fireRecord(load('examples') as Pack<Examples>, c.example);
  const r = read(load('weights') as Pack<FDMeta>, apply(rec, c));
  const hid: number[] = [];
  for (let k = 0; k < T_STEPS; k++) for (const j of [...WIND_COLS, HIDDEN_COL]) hid.push(r.seqH[k * NF + j]);
  return [...OUTS.map((k) => r.out[k]), ...r.seq, ...hid, ...r.st];
}
