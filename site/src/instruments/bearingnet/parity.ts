// Parity hook for site/tests/parity.test.ts: run one reference case through the whole browser pipeline
// (noise -> windows -> kurtogram band -> envelope spectra -> strips -> network) and return every window's raw outputs.
import { addNoise, recordSpectra, runNet, type BMeta, type BExample } from './net.ts';
import { rng } from '../kit.ts';
import type { Pack } from '../../lib/nn/pack.ts';

let W: Pack<BMeta> | null = null;
let X: Pack<BExample[]> | null = null;

export async function run(c: { example: number; variant: string; rpm: number; sigma: number; seed: number }, load: (name: string) => Pack<any>) {
  W ??= load('weights');
  X ??= load('examples');
  const e = X.meta[c.example];
  const r = rng(c.seed);
  const x = addNoise(X.t[`x${c.example}`].data, c.sigma, r.n);
  const { outs } = runNet(W, recordSpectra(x), e.bearing, c.rpm);
  const flat: number[] = [];
  for (const o of outs) flat.push(...o);
  return flat;
}
