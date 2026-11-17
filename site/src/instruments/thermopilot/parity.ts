// Parity hook for site/tests/parity.test.ts: ThermoPilot closed-loop on the browser house for one week of one real
// exam twin, climate and tariff, at the case's comfort weight, forecast error and forecast bias. Returns the hourly
// heating setpoints, cooling setpoints and indoor temperatures, the bill and the kelvin-hours outside the band, then
// the raw logits at the logged hours.
import { ctxOf, houseOf, loadNet, runPilot, type Net, type WMeta, type XMeta } from './net.ts';
import type { Pack } from '../../lib/nn/pack.ts';

let net: Net | null = null;
let X: Pack<XMeta> | null = null;

export async function run(c: { example: number; run: number; lam_mul: number; fe: number; bias: number; log_hours: number[] },
  load: (name: string) => Pack<any>) {
  net ??= loadNet(load('weights') as Pack<WMeta>);
  X ??= load('examples') as Pack<XMeta>;
  const house = houseOf(X, c.example);
  const home = X.meta.homes[Math.floor(c.example / (X.meta.climates.length * X.meta.weeks.length))];
  const ctx = ctxOf(X, house, c.run, home);
  const r = runPilot(net, house, ctx, X.t.z.data, { lamMul: c.lam_mul, fe: c.fe, bias: c.bias }, 168, c.log_hours);
  const out: number[] = [...r.h, ...r.c, ...r.T, r.bill, r.DH];
  for (const lg of r.logits) out.push(...lg);
  return out;
}
