// Parity hook for site/tests/parity.test.ts: SnowSense's bands and annual readings and WaterSim's uncalibrated
// energy-balance and degree-day SWE against Python, for recorded and what-if weather at the three exam stations.
import type { Pack } from '../../lib/nn/pack.ts';
import { degreeDay, energyBalance, makeWindow, read, type Scenario, type SnowWeights, type Stations } from './net.ts';

const every = (a: Float64Array, k: number) => Array.from(a).filter((_, i) => i % k === 0);
const maxOf = (a: Float64Array) => a.reduce((m, v) => Math.max(m, v), -Infinity);

export function run(c: { station: number; wy: number; scenario: Scenario }, load: (s: string) => Pack<unknown>) {
  const st = load('stations') as Pack<Stations>;
  const w = makeWindow(st, c.station, c.wy, c.scenario);
  const r = read(load('weights') as Pack<SnowWeights>, w);
  const eb = energyBalance(w, st.meta.eb_default), dd = degreeDay(w, st.meta.dd_default);
  return [r.peak, r.peakDay, r.melt, r.peakLo68, r.peakHi68, r.peakLo90, r.peakHi90, r.meltLo68, r.meltHi68,
    r.meltLo90, r.meltHi90, maxOf(eb), maxOf(dd), ...every(r.swe, 10), ...every(r.lo68, 30), ...every(r.hi90, 30),
    ...every(eb, 10), ...every(dd, 10)];
}
