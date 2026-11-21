// Parity hook for site/tests/parity.test.ts: FireCast against firecast.py on the three exam fire-days of
// examples/, recorded and with what-ifs on tomorrow's weather, the VIIRS hot spots and yesterday's perimeter.
// Outputs: the 22 scalar inputs, then the logits on rows/cols start::step of the map window.
import type { Pack } from '../../lib/nn/pack.ts';
import { day, forward, loadNet, prepare, type DaysMeta, type Ops, type WeightsMeta } from './net.ts';

export function run(c: { day: number; ops: Ops }, load: (s: string) => Pack<unknown>) {
  const pack = load('days') as Pack<DaysMeta>;
  const net = loadNet(load('weights') as Pack<WeightsMeta>);
  const inp = prepare(day(pack, c.day), c.ops);
  const lg = forward(net, inp.X, inp.s);
  const { step, start } = pack.meta;
  const out = Array.from(inp.s);
  for (let r = start; r < inp.hh; r += step) for (let q = start; q < inp.ww; q += step) out.push(lg[r * inp.ww + q]);
  return out;
}
