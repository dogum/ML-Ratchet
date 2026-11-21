// Parity hook for site/tests/parity.test.ts: ReservoirPilot, SOP, the sim-tuned rule and WaterData's DP against the
// model's own Python on the six exam dams, recorded and with what-ifs on the input.
import type { Pack } from '../../lib/nn/pack.ts';
import { components, damSeries, makeTask, pilotNet, runDP, runPilot, runRule, runSOP, type Components, type DamsMeta, type WhatIf } from './net.ts';

const DAYS = [0, 365, 1000, 2000, -1];
const all = (c: Components) => [c.J, c.J_short, c.J_flood, c.shortage_ratio, c.shortage_days, c.flood_days, c.max_excess,
  c.S_mean, c.dS_end, c.empty, c.full];

export function run(c: { dam: number; whatif: WhatIf; dp: boolean }, load: (s: string) => Pack<unknown>) {
  const s = damSeries(load('dams') as Pack<DamsMeta>, c.dam);
  const task = makeTask(s, c.whatif);
  const p = runPilot(pilotNet(load('weights')), task);
  const n = task.I.length;
  const out = [...all(components(p, task)), ...DAYS.map(d => p.R[d < 0 ? n + d : d] / task.Qmax), p.H[1000], p.E[1000]];
  const r = components(runRule(task), task), o = components(runSOP(task), task);
  out.push(r.J, r.S_mean, r.dS_end, o.J, o.S_mean, o.dS_end);
  if (c.dp) { const d = components(runDP(task), task); out.push(d.J, d.S_mean); }
  return out;
}
