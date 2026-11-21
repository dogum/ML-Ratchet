import type { Pack } from '../../lib/nn/pack.ts';
import { forward } from './net.ts';
import { preprocess, RATE } from './pipeline.ts';

export function run(c: { example: number; variant: string }, load: (name: string) => Pack<any>) {
  const weights = load('weights');
  const { t } = load('examples');
  const wave = t[`wave_${c.example}`].data;
  const crop = c.variant === 'short-tail';
  const at48k = c.variant === 'resample48k';
  const original = crop ? wave.subarray(0, Math.round(0.9 * RATE)) : wave;
  const samples = at48k ? Float32Array.from({ length: original.length * 3 }, (_, i) => {
    const j = Math.floor(i / 3);
    return original[j] + (i % 3) / 3 * (original[Math.min(original.length - 1, j + 1)] - original[j]);
  }) : original;
  const f = preprocess(samples, at48k ? 48000 : RATE);
  return [...forward(weights, f), f.classicalSeconds ?? -1, f.onsetSeconds, f.snrDb, ...f.bandSeconds.map(v => v ?? -1)];
}

