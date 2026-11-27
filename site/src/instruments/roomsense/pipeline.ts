/** Mono clap preprocessing mirrored from models/roomsense/pipeline.py. */
export const RATE = 16_000;
export const BANDS = [500, 1000, 2000];
export const FRAMES = 256;
const HOP = RATE / 100;
const EARLY_HOP = RATE / 1000;
const TINY = 1e-20;
export const FALLBACK = 0.65;

export interface Features {
  curves: Float32Array;
  context: Float32Array;
  onsetSeconds: number;
  snrDb: number;
  classicalSeconds: number | null;
  bandSeconds: (number | null)[];
}

export function median(xs: readonly number[]) {
  const ordered = [...xs].sort((a, b) => a - b);
  const k = Math.floor(ordered.length / 2);
  return ordered.length % 2 ? ordered[k] : (ordered[k - 1] + ordered[k]) / 2;
}

const average = (values: ArrayLike<number>) => {
  let total = 0;
  for (let i = 0; i < values.length; i++) total += values[i];
  return total / values.length;
};

function resample(samples: Float64Array, rate: number) {
  if (rate === RATE) return samples;
  const ratio = rate / RATE;
  const cutoff = Math.min(1, 1 / ratio);
  const radius = Math.ceil(12 / cutoff);
  const result = new Float64Array(Math.floor(samples.length / ratio));
  for (let i = 0; i < result.length; i++) {
    const p = i * ratio;
    let sum = 0;
    let total = 0;
    for (let j = Math.floor(p) - radius + 1; j <= Math.floor(p) + radius; j++) {
      const delta = j - p;
      if (Math.abs(delta) >= radius) continue;
      const t = Math.PI * cutoff * delta;
      const sinc = t === 0 ? 1 : Math.sin(t) / t;
      const weight = sinc * (0.5 + 0.5 * Math.cos(Math.PI * delta / radius));
      total += weight;
      sum += weight * samples[Math.max(0, Math.min(samples.length - 1, j))];
    }
    result[i] = sum / total;
  }
  return result;
}

function bandFilter(samples: Float64Array, frequency: number) {
  const angle = 2 * Math.PI * frequency / RATE;
  const alpha = Math.sin(angle) / (2 * Math.sqrt(2));
  const b = alpha / (1 + alpha);
  const a1 = -2 * Math.cos(angle) / (1 + alpha);
  const a2 = (1 - alpha) / (1 + alpha);
  let z1 = 0;
  let z2 = 0;
  return samples.map(x => {
    const y = b * x + z1;
    z1 = z2 - a1 * y;
    z2 = -b * x - a2 * y;
    return y;
  });
}

function powerFrames(samples: Float64Array, hop: number, count: number) {
  const frames = new Float64Array(count);
  for (let i = 0; i < Math.min(samples.length, hop * count); i++) {
    frames[Math.floor(i / hop)] += samples[i] ** 2 / hop;
  }
  return frames;
}

function energyDecay(power: Float64Array, noise: number) {
  const threshold = Math.max(4 * noise, Math.max(...power) * 1e-8, TINY);
  let stop = power.length;
  while (stop > 0 && power[stop - 1] < threshold) stop--;
  let total = 0;
  const energy = new Float64Array(power.length);
  for (let t = stop - 1; t >= 0; t--) {
    total += power[t] - noise;
    energy[t] = total;
  }
  const scale = Math.max(energy[0], TINY);
  let previous = 1;
  const db = energy.map(v => {
    previous = Math.max(1e-8, Math.min(previous, v / scale));
    return 10 * Math.log10(previous);
  });
  return { db, stop };
}

function estimateT20(db: Float64Array, stop: number) {
  const indices = [];
  for (let i = 0; i < stop - 1; i++) {
    if (db[i] <= -5 && db[i] >= -25) indices.push(i);
  }
  if (indices.length < 4 || db[indices[indices.length - 1]] > -23 || db[indices[0]] < -9) return null;
  const x = indices.map(i => i * HOP / RATE);
  const y = indices.map(i => db[i]);
  const xMean = average(x);
  const yMean = average(y);
  let xx = 0;
  let xy = 0;
  let yy = 0;
  for (let i = 0; i < indices.length; i++) {
    const dx = x[i] - xMean;
    const dy = y[i] - yMean;
    xx += dx * dx;
    xy += dx * dy;
    yy += dy * dy;
  }
  const reading = -60 / (xy / xx);
  return xy < 0 && xy * xy / (xx * yy) >= 0.85 && reading >= 0.08 && reading <= 6 ? reading : null;
}

export function preprocess(input: ArrayLike<number>, rate: number): Features {
  if (rate < 8000 || rate > 96000 || input.length < 0.45 * rate || input.length > 30 * rate) {
    throw new Error('Provide 0.45 to 30 seconds of audio at 8 to 96 kHz.');
  }
  let clipped = 0;
  for (let i = 0; i < input.length; i++) {
    if (!Number.isFinite(input[i])) throw new Error('The recording contains invalid samples.');
    if (Math.abs(input[i]) >= 0.999) clipped++;
  }
  if (clipped >= 3) throw new Error('The recording is clipped.');
  const mean = average(input);
  const samples = resample(Float64Array.from(input, v => v - mean), rate);
  const powers = powerFrames(samples, EARLY_HOP, Math.floor(samples.length / EARLY_HOP));
  let peak = 0;
  for (let i = 1; i < powers.length; i++) if (powers[i] > powers[peak]) peak = i;
  let begin = peak;
  while (begin > Math.max(0, peak - 8) && powers[begin - 1] > 0.015 * powers[peak]) begin--;
  const onset = begin * EARLY_HOP;
  if (onset < RATE / 20 || samples.length - onset < RATE * 0.4) {
    throw new Error('Leave at least 50 ms before the clap and 400 ms after it.');
  }
  const preEnd = onset - RATE / 100;
  const preStart = Math.max(0, preEnd - RATE / 4);
  const noise = average(samples.subarray(preStart, preEnd).map(v => v * v));
  const snrDb = 10 * Math.log10(Math.max(powers[peak], TINY) / Math.max(noise, TINY));
  if (powers[peak] < 1e-12 || snrDb < 15) throw new Error('The clap is too quiet relative to the background.');
  const segmentStart = Math.max(0, preStart - RATE / 50);
  const segment = samples.subarray(segmentStart, onset + HOP * FRAMES);
  const curves = new Float32Array(6 * FRAMES);
  const bandSeconds = BANDS.map((frequency, i) => {
    const filtered = bandFilter(segment, frequency);
    const bandNoise = average(filtered.subarray(preStart - segmentStart, preEnd - segmentStart).map(v => v * v));
    const tail = filtered.subarray(onset - segmentStart);
    const early = powerFrames(tail, EARLY_HOP, FRAMES);
    const max = Math.max(...early, TINY);
    curves.set(early.map(power => Math.log10(Math.max(1e-6, power / max)) / 6), i * FRAMES);
    const { db, stop } = energyDecay(powerFrames(tail, HOP, Math.floor(tail.length / HOP)), bandNoise);
    curves.fill(-1, (i + 3) * FRAMES, (i + 4) * FRAMES);
    curves.set(db.map(v => v / 80), (i + 3) * FRAMES);
    return estimateT20(db, stop);
  });
  const valid = bandSeconds.filter((v): v is number => v !== null);
  const context = Float32Array.from([...bandSeconds.map(v => Math.log(v ?? FALLBACK)), ...bandSeconds.map(v => Number(v !== null))]);
  return { curves, context, snrDb, onsetSeconds: onset / RATE,
    classicalSeconds: valid.length >= 2 ? median(valid) : null, bandSeconds };
}

