// Parity hook for site/tests/parity.test.ts: one recorded moment with SOC / demand / coolant / preview set as in the
// case; returns [log(s + 0.05), logit P(on), s with SOC feedback, P(on), chosen candidate index].
import { loadNet, decide, cands, type Net, type HMeta, type WMeta } from './net.ts';
import type { Pack } from '../../lib/nn/pack.ts';

let net: Net | null = null;
let X: Pack<HMeta> | null = null;

export async function run(c: { example: number; moment: number; soc: number; d: number; t: number; L: number }, load: (name: string) => Pack<any>) {
  net ??= loadNet(load('weights') as Pack<WMeta>);
  X ??= load('examples') as Pack<HMeta>;
  const trip = X.meta.trips[c.example];
  const m = trip.moments[c.moment];
  const r = decide(net, X.meta, trip, m, cands(X, c.example, c.moment, c.d, c.t), { soc: c.soc, d: c.d, t: c.t, L: c.L });
  return [r.y[0], r.y[1], r.s, r.pon, r.j];
}
