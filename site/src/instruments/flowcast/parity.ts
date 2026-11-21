// Parity hook for site/tests/parity.test.ts: FlowCast (ungauged and gauged nowcast weights) against flowcast.py on
// the three exam basins, recorded and with what-ifs on the forcing and the attributes.
import type { Pack } from '../../lib/nn/pack.ts';
import { basin, loadNet, predict, prepare, type BasinsMeta, type Ops, type WeightsMeta } from './net.ts';

export function run(c: { basin: number; weights: string; ops: Ops; lead?: number; outage?: [number, number] | null },
  load: (s: string) => Pack<unknown>) {
  const pack = load('basins') as Pack<BasinsMeta>;
  const b = basin(pack, c.basin);
  const net = loadNet(load(c.weights) as Pack<WeightsMeta>);
  let gauge;
  if (c.weights === 'nowcast') {
    const obs = new Float64Array(b.obs);
    if (c.outage) for (let t = c.outage[0]; t <= c.outage[1]; t++) obs[t] = NaN;
    gauge = { obs, lead: c.lead ?? 1 };
  }
  const n = b.info.n_target, step = pack.meta.step;
  const f = predict(net, prepare(net, b, c.ops, gauge), n);
  const out = Array.from(f.st);
  for (let i = 0; i < n; i += step) out.push(f.mean[i]);
  for (let i = 0; i < n; i += step) out.push(f.q[i * 5]);
  for (let i = 0; i < n; i += step) out.push(f.q[i * 5 + 4]);
  return out;
}
