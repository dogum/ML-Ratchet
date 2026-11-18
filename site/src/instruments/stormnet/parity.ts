// Parity hook for site/tests/parity.test.ts: StormNet and its three physics competitors against Python on
// storm-gallery issue hours, recorded and with what-if winds.
import type { Pack } from '../../lib/nn/pack.ts';
import { predict, stormSeries, windowAt, type StormWeights, type Storms } from './net.ts';

export function run(c: { storm: number; k: number; bz: number; speed: number; pressure: number },
  load: (s: string) => Pack<unknown>) {
  const s = stormSeries(load('storms') as Pack<Storms>, c.storm);
  const w = windowAt(s.rows, c.k, { bz: c.bz, speed: c.speed, pressure: c.pressure })!;
  const r = predict(load('weights') as Pack<StormWeights>, w, s.dst[c.k]);
  return [r.minimum, r.sigma, r.classical, r.single, r.published, r.fast, r.slow];
}
