// "Hear the star": the measured power spectrum around the oscillation hump, played as sound.
// Every frequency is multiplied by 10 million (1 uHz -> 10 Hz), which is the same as playing the light curve
// 10^7 times faster: a 70 uHz red giant hums at 700 Hz, and its comb of modes (spacing dnu) becomes a chord of
// tones dnu x 10 Hz apart. The amplitude at each pitch is the measured excess power over the local background,
// so a short or noisy light curve sounds blurred and hissy — exactly what the network has to read.
import { fft2, makeRebin } from './dsp.ts';
import { rng } from '../kit.ts';

export const SPEEDUP = 1e7;

/** Mono samples at `sr`: the band numax +- 3 sigma_env of (f, P), background-flattened, random phases. */
export function starSound(f: Float64Array, P: Float64Array, numax: number, dnu: number, sr: number, seed = 1): Float32Array {
  const N = 1 << 18; // ~5.5-6 s at 44.1-48 kHz
  const fwhm = 0.66 * Math.pow(numax, 0.88), sig = fwhm / 2.355;
  const lo = Math.max(f[0], numax - 3.2 * sig), hi = Math.min(f[f.length - 1], numax + 3.2 * sig);
  const df = f[1] - f[0];
  // background: median power in windows two orders wide (most of each order is between modes), /ln2 -> mean of chi^2_2
  const step = Math.max(dnu / 2, df), grid: number[] = [], bg: number[] = [];
  for (let c = lo - dnu; c <= hi + dnu; c += step) {
    const a = Math.max(0, Math.floor((c - dnu - f[0]) / df)), b = Math.min(f.length, Math.ceil((c + dnu - f[0]) / df) + 1);
    if (b - a < 3) continue;
    const w = Float64Array.from(P.subarray(a, b)).sort();
    grid.push(c); bg.push(w[w.length >> 1] / Math.LN2);
  }
  const bgAt = (x: number) => {
    if (!grid.length) return 1;
    if (x <= grid[0]) return bg[0];
    for (let i = 1; i < grid.length; i++) if (x <= grid[i]) { const t = (x - grid[i - 1]) / (grid[i] - grid[i - 1]); return bg[i - 1] * (1 - t) + bg[i] * t; }
    return bg[bg.length - 1];
  };
  const ex = new Float64Array(f.length);
  for (let k = 0; k < f.length; k++) {
    if (f[k] < lo || f[k] > hi) continue;
    const env = Math.exp(-0.5 * ((f[k] - numax) / sig) ** 2);
    ex[k] = Math.max(0, P[k] / bgAt(f[k]) - 1) * env;
  }
  // onto the audio FFT grid (area-preserving, so a coarse spectrum spreads each peak over its own resolution)
  const dfa = sr / N, nf = N / 2, edges = new Float64Array(nf + 1);
  for (let k = 0; k <= nf; k++) edges[k] = ((k - 0.5) * dfa) / (SPEEDUP * 1e-6);
  const pa = makeRebin(f, ex)(edges);
  const re = new Float64Array(N), im = new Float64Array(N), r = rng(seed);
  for (let k = 1; k < nf; k++) {
    const a = Math.sqrt(Math.max(0, pa[k])), ph = 2 * Math.PI * r.u();
    re[k] = a * Math.cos(ph); im[k] = a * Math.sin(ph);
    re[N - k] = re[k]; im[N - k] = -im[k];
  }
  fft2(re, im, true);
  let peak = 1e-12;
  for (let i = 0; i < N; i++) peak = Math.max(peak, Math.abs(re[i]));
  const out = new Float32Array(N), fade = Math.round(sr * 0.08);
  for (let i = 0; i < N; i++) {
    const g = Math.min(1, i / fade, (N - 1 - i) / fade);
    out[i] = (0.7 * re[i] * g) / peak;
  }
  return out;
}

/** One-voice player: play(samples) stops anything already playing. */
export function player() {
  let ac: AudioContext | null = null, src: AudioBufferSourceNode | null = null;
  return {
    get rate() { ac ??= new AudioContext(); return ac.sampleRate; },
    playing: () => src != null,
    async play(x: Float32Array, onEnd: () => void) {
      this.stop();
      ac ??= new AudioContext();
      if (ac.state === 'suspended') await ac.resume();
      const buf = ac.createBuffer(1, x.length, ac.sampleRate);
      buf.copyToChannel(x as Float32Array<ArrayBuffer>, 0);
      const s = ac.createBufferSource();
      s.buffer = buf;
      s.connect(ac.destination);
      s.onended = () => { if (src === s) { src = null; onEnd(); } };
      src = s;
      s.start();
    },
    stop() { if (src) { const s = src; src = null; try { s.stop(); } catch { /* already stopped */ } } },
  };
}
