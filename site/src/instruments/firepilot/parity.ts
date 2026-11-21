// Parity hook for site/tests/parity.test.ts: firepilot.py's 31 features, the value network's mean and sigma, km of
// line built, the connectivity proxy and the number of line cells, for fixed plans on the three example fires.
import type { Pack } from '../../lib/nn/pack.ts';
import { Context, fire, predict, valueNet, type FiresMeta, type NetMeta, type Plan } from './net.ts';

export function run(c: { fire: number; rate: number; plan: Plan }, load: (s: string) => Pack<unknown>) {
  const ctx = new Context(fire(load('fires') as Pack<FiresMeta>, c.fire), c.rate);
  const { x, pe } = ctx.features(c.plan);
  const { mean, sigma } = predict(valueNet(load('weights') as Pack<NetMeta>), x);
  return [...x, mean, sigma, pe.km, pe.s_proxy, pe.L.idx.length];
}
