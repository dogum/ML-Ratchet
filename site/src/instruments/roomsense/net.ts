import type { Pack } from '../../lib/nn/pack.ts';
import { conv1d, linear, relu_, vcat } from '../../lib/nn/ops.ts';
import { FRAMES, FALLBACK, median, type Features } from './pipeline.ts';

export interface WeightsMeta { widen: [number, number]; parameters: number }
export interface Example {
  label: string;
  rt60_s: number;
  volume_m3: number | null;
  filename: string;
}
export function forward(pack: Pack<WeightsMeta>, features: Features) {
  const w = pack.t;
  const x = relu_(conv1d({ data: features.curves, shape: [6, FRAMES] }, w['conv1.weight'], w['conv1.bias'], 4, 3));
  const y = relu_(conv1d(x, w['conv2.weight'], w['conv2.bias'], 4, 2));
  const hidden = linear(vcat(y.data, features.context), w['hidden.weight'], w['hidden.bias']);
  for (let i = 0; i < hidden.length; i++) hidden[i] = Math.max(0, hidden[i]);
  const raw = linear(hidden, w['output.weight'], w['output.bias']);
  const valid = Array.from(features.context.slice(0, 3)).filter((_, i) => features.context[i + 3] > 0);
  raw[0] += valid.length >= 2 ? median(valid) : Math.log(FALLBACK);
  raw[1] += Math.log(150);
  for (let i = 2; i < 4; i++) {
    raw[i] = Math.max(raw[i], 0) + Math.log1p(Math.exp(-Math.abs(raw[i]))) + 0.035;
  }
  return raw;
}

export function reading(pack: Pack<WeightsMeta>, features: Features) {
  const raw = forward(pack, features);
  return [0, 1].map(i => {
    const span = raw[i + 2] * pack.meta.widen[i];
    return { value: Math.exp(raw[i]), lo: Math.exp(raw[i] - span), hi: Math.exp(raw[i] + span) };
  });
}

