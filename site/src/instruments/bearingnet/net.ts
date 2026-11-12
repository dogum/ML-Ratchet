// BearingNet in the browser: a line-for-line port of models/bearingnet/pipeline.py (the front end: kurtogram band,
// five squared-envelope spectra, fault-aligned log-frequency strips) and bearingnet.py forward() + read().
// No DOM, so the parity test runs it in Node.
import { conv1d, gelu_, add_, stride1d, globalPool, linear, vcat, vgelu } from '../../lib/nn/ops.ts';
import type { Pack } from '../../lib/nn/pack.ts';
import type { Tensor } from '../../lib/nn/tensor.ts';

export const FS = 12000, NWIN = 32768, DF = FS / NWIN, NV = 295, NB = 5, NH = 4, NCH = NB * NH + 1;
export const V = Float64Array.from({ length: NV }, (_, i) => -2 + i / 64); // log2(f / f_hypothesis)
export const HYP = ['shaft', 'BPFO', 'BPFI', 'BSF'] as const;
const FIXED: [number, number][] = [[500, 5700], [500, 2000], [2000, 3800], [3800, 5700]];
const HANN = Float64Array.from({ length: NWIN }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (NWIN - 1))); // np.hanning

export interface Geom { n: number; d: number; D: number; phi?: number }
export interface BMeta {
  arch: string; n_blocks: number; n_head: number; sp_scale: number; sz_mu: number; sz_sd: number; classes: string[];
  calib: { speed: number; size: number; temp: number; nbias: number };
}
export interface BExample {
  name: string; rig: string; file: string; cond: string; truth: string; bearing: Geom;
  rpm_nominal: number; rpm_true: number; size_mm: number; n: number;
  orders: { FTF: number; BPFO: number; BPFI: number; BSF: number };
  python: { condition: string; p: Record<string, number>; rpm: number; rpm_sig: number; size_mm: number };
}

// ---- DSP ----------------------------------------------------------------------------------------
const tw: Record<number, [Float64Array, Float64Array]> = {};
/** In-place radix-2 complex FFT (inverse divides by n, as numpy). */
export function fft(re: Float64Array, im: Float64Array, inv = false) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let b = n >> 1;
    for (; j & b; b >>= 1) j ^= b;
    j ^= b;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  let w = tw[n];
  if (!w) {
    w = [new Float64Array(n / 2), new Float64Array(n / 2)];
    for (let k = 0; k < n / 2; k++) { w[0][k] = Math.cos((2 * Math.PI * k) / n); w[1][k] = -Math.sin((2 * Math.PI * k) / n); }
    tw[n] = w;
  }
  const [wr0, wi0] = w, s = inv ? -1 : 1;
  for (let len = 2; len <= n; len <<= 1) {
    const h = len >> 1, st = n / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < h; k++) {
        const wr = wr0[k * st], wi = s * wi0[k * st], a = i + k, b = a + h;
        const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
      }
    }
  }
  if (inv) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

function rfftAbs(x: Float64Array): Float64Array {
  const n = x.length, re = Float64Array.from(x), im = new Float64Array(n);
  fft(re, im);
  const o = new Float64Array(n / 2 + 1);
  for (let i = 0; i <= n / 2; i++) o[i] = Math.hypot(re[i], im[i]);
  return o;
}

export function orders(g: Geom) {
  const r = (g.d / g.D) * Math.cos(((g.phi ?? 0) * Math.PI) / 180);
  return { FTF: 0.5 * (1 - r), BPFO: (g.n / 2) * (1 - r), BPFI: (g.n / 2) * (1 + r), BSF: (g.D / (2 * g.d)) * (1 - r * r) };
}

/** pipeline.windows(): start indices of up to maxWin half-overlapping windows. */
export function windows(n: number, maxWin = 8): number[] {
  if (n < NWIN) return [0];
  let st: number[] = [];
  for (let s = 0; s <= n - NWIN; s += NWIN / 2) st.push(s);
  if (st[st.length - 1] !== n - NWIN) st.push(n - NWIN);
  if (st.length > maxWin) {
    const m = st.length - 1;
    st = Array.from({ length: maxWin }, (_, i) => st[Math.round((i * m) / (maxWin - 1))]);
  }
  return st;
}

/** pipeline.sk_band(): spectral kurtosis from a 256-point STFT (periodic Hann, hop 64); the band of highest mean SK. */
function skBand(x: Float64Array, lo = 500, hi = 5700): [number, number] {
  const N = 256, hop = 64, nf = Math.floor((x.length - N) / hop) + 1, K = N / 2 + 1;
  const win = Float64Array.from({ length: N }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N));
  const m1 = new Float64Array(K), m2 = new Float64Array(K), re = new Float64Array(N), im = new Float64Array(N);
  for (let f = 0; f < nf; f++) {
    for (let i = 0; i < N; i++) { re[i] = x[f * hop + i] * win[i]; im[i] = 0; }
    fft(re, im);
    for (let k = 0; k < K; k++) { const p = re[k] * re[k] + im[k] * im[k]; m1[k] += p; m2[k] += p * p; }
  }
  const sk = new Float64Array(K);
  for (let k = 0; k < K; k++) { const a = m1[k] / nf, b = m2[k] / nf; sk[k] = b / Math.max(a * a, 1e-30) - 2; }
  let best = -1e9, band: [number, number] = [lo, hi];
  for (const bw of [600, 1200, 2400]) {
    for (let c = lo + bw / 2; c < hi - bw / 2 + 1; c += 150) {
      let s = 0, cnt = 0;
      for (let k = 0; k < K; k++) { const f = (k * FS) / N; if (f >= c - bw / 2 && f <= c + bw / 2) { s += sk[k]; cnt++; } }
      s /= cnt;
      if (s > best) { best = s; band = [c - bw / 2, c + bw / 2]; }
    }
  }
  return band;
}

/** Squared-envelope spectrum of the band [lo, hi] from the window's FFT; also the band signal's kurtosis (x2). */
function envSpec(Xre: Float64Array, Xim: Float64Array, lo: number, hi: number) {
  const re = new Float64Array(NWIN), im = new Float64Array(NWIN), T = 60;
  for (let i = 0; i <= NWIN / 2; i++) {
    const f = i * DF, g = Math.min(1, Math.max(0, (f - lo + T) / T)) * Math.min(1, Math.max(0, (hi + T - f) / T));
    re[i] = 2 * Xre[i] * g; im[i] = 2 * Xim[i] * g;
  }
  fft(re, im, true);
  const e = new Float64Array(NWIN);
  let m = 0, m2 = 0;
  for (let i = 0; i < NWIN; i++) { e[i] = re[i] * re[i] + im[i] * im[i]; m += e[i]; m2 += e[i] * e[i]; }
  m /= NWIN; m2 /= NWIN;
  const k = m2 / Math.max(m * m, 1e-30);
  for (let i = 0; i < NWIN; i++) e[i] = (e[i] - m) * HANN[i];
  return { ses: rfftAbs(e), k };
}

/** pipeline.strip(): read `spec` at f0 * 2**V (max over the bins inside each 1/64-octave cell, else interpolation). */
export function strip(spec: Float64Array, f0: number): Float64Array {
  const out = new Float64Array(NV), L = spec.length, a = 2 ** (-1 / 128), b = 2 ** (1 / 128);
  for (let i = 0; i < NV; i++) {
    const fc = f0 * 2 ** V[i], u = fc / DF, j = Math.floor(u);
    let v = j >= L - 1 ? spec[L - 1] : u <= 0 ? spec[0] : spec[j] + (spec[j + 1] - spec[j]) * (u - j);
    const lo = Math.min(Math.max(Math.ceil((fc * a) / DF), 1), L - 1), hi = Math.min(Math.max(Math.floor((fc * b) / DF), 1), L - 1);
    for (let q = lo; q <= hi; q++) if (spec[q] > v) v = spec[q];
    out[i] = v;
  }
  return out;
}

function median(a: Float64Array) {
  const s = Float64Array.from(a).sort(), n = s.length;
  return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

/** The speed-independent part of one window: five envelope spectra, the raw spectrum, band kurtoses, the SK band. */
export interface Spectra { ses: Float64Array[]; raw: Float64Array; ks: number[]; band: [number, number] }

export function spectra(xin: ArrayLike<number>, start = 0): Spectra {
  let mu = 0;
  for (let i = 0; i < NWIN; i++) mu += xin[start + i];
  mu /= NWIN;
  const x = new Float64Array(NWIN);
  for (let i = 0; i < NWIN; i++) x[i] = xin[start + i] - mu;
  const re = Float64Array.from(x), im = new Float64Array(NWIN);
  fft(re, im);
  const band = skBand(x), ses: Float64Array[] = [], ks: number[] = [];
  for (const [lo, hi] of [...FIXED, band]) { const r = envSpec(re, im, lo, hi); ses.push(r.ses); ks.push(r.k); }
  const xr = new Float64Array(NWIN);
  for (let i = 0; i < NWIN; i++) xr[i] = x[i] * HANN[i];
  return { ses, raw: rfftAbs(xr), ks, band };
}

/** pipeline.window_features() from precomputed spectra: the (21, 295) strip map and 8 scalars for shaft speed fNom (Hz). */
export function features(sp: Spectra, g: Geom, fNom: number): { F: Float32Array; sc: Float32Array } {
  const o = orders(g), fh = [fNom, o.BPFO * fNom, o.BPFI * fNom, o.BSF * fNom], F = new Float32Array(NCH * NV);
  const put = (row: number, spec: Float64Array, f0: number) => {
    const s = strip(spec, f0);
    for (let i = 0; i < NV; i++) s[i] = Math.log10(s[i] + 1e-12);
    const md = median(s);
    for (let i = 0; i < NV; i++) F[row * NV + i] = Math.min(5, Math.max(-2, Math.fround(s[i] - md)));
  };
  for (let b = 0; b < NB; b++) for (let h = 0; h < NH; h++) put(b * NH + h, sp.ses[b], fh[h]);
  put(NCH - 1, sp.raw, fNom);
  const r = (g.d / g.D) * Math.cos(((g.phi ?? 0) * Math.PI) / 180);
  const sc = Float32Array.from([...sp.ks.map((k) => Math.log10(Math.max(k, 1e-3) / 2)), Math.log2(fNom / 25), g.n / 10, 4 * r]);
  return { F, sc };
}

// ---- network (arch 'base': one trunk over all 21 strips, then an MLP head) -------------------------
function trunk(W: Pack<BMeta>, x: Tensor): { mean: Float32Array; max: Float32Array } {
  const t = W.t;
  let h = gelu_(conv1d(x, t.stem_w, t.stem_b, 2));
  for (let i = 0; i < W.meta.n_blocks; i++) {
    let y = gelu_(conv1d(h, t[`b${i}_c1_w`], t[`b${i}_c1_b`], 2));
    y = conv1d(y, t[`b${i}_c2_w`], t[`b${i}_c2_b`]);
    h = gelu_(add_(y, conv1d(stride1d(h, 2), t[`b${i}_sk_w`], t[`b${i}_sk_b`])));
  }
  return globalPool(h);
}

/** Raw outputs (8): 4 class logits (N, OR, IR, B), speed (mean, log sigma), size (mean, log sigma). */
export function forward(W: Pack<BMeta>, F: Float32Array, sc: Float32Array): Float32Array {
  const { mean, max } = trunk(W, { data: F, shape: [NCH, NV] });
  let z = vcat(mean, max, sc);
  for (let j = 0; j < W.meta.n_head; j++) {
    z = linear(z, W.t[`h${j}_w`], W.t[`h${j}_b`]);
    if (j < W.meta.n_head - 1) vgelu(z);
  }
  return z;
}

export interface Reading { condition: string; p: number[]; rpm: number; rpm_sig: number; size_mm: number; log10_size_sig: number; n_windows: number }

/** BearingNet.read() post-processing over the windows' raw outputs (calibrated). rpmTold: the speed you told it. */
export function decode(meta: BMeta, outs: Float32Array[], rpmTold: number): Reading {
  const cal = meta.calib, n = outs.length, lp = [0, 0, 0, 0];
  for (const o of outs) {
    const l = [0, 1, 2, 3].map((k) => o[k] / cal.temp), m = Math.max(...l);
    const ls = Math.log(l.reduce((s, v) => s + Math.exp(v - m), 0));
    for (let k = 0; k < 4; k++) lp[k] += (l[k] - m - ls) / n;
  }
  lp[0] += cal.nbias;
  const mx = Math.max(...lp), e = lp.map((v) => Math.exp(v - mx)), s = e.reduce((a, b) => a + b, 0), p = e.map((v) => v / s);
  let sw = 0, spw = 0, sw2 = 0, lzw = 0;
  for (const o of outs) {
    const sp = o[4] * meta.sp_scale, sps = Math.exp(Math.min(3, Math.max(-5, o[5]))) * meta.sp_scale, w = 1 / sps ** 2;
    sw += w; spw += sp * w;
    const lz = o[6] * meta.sz_sd + meta.sz_mu, lzs = Math.exp(Math.min(3, Math.max(-5, o[7]))) * meta.sz_sd, w2 = 1 / lzs ** 2;
    sw2 += w2; lzw += lz * w2;
  }
  const rpm = rpmTold * 2 ** (spw / sw), spsR = Math.sqrt(n / sw) * cal.speed;
  return {
    condition: meta.classes[p.indexOf(Math.max(...p))], p, rpm, rpm_sig: rpm * spsR * Math.LN2,
    size_mm: 10 ** (lzw / sw2), log10_size_sig: Math.sqrt(n / sw2) * cal.size, n_windows: n,
  };
}

/** Everything for one recording: spectra per window (cacheable), then features + network for a told speed. */
export function recordSpectra(x: ArrayLike<number>, maxWin = 8): Spectra[] {
  return windows(x.length, maxWin).map((s) => spectra(x, s));
}

export function runNet(W: Pack<BMeta>, sps: Spectra[], g: Geom, rpmTold: number) {
  const feats = sps.map((sp) => features(sp, g, rpmTold / 60));
  const outs = feats.map((f) => forward(W, f.F, f.sc));
  return { feats, outs, reading: decode(W.meta, outs, rpmTold) };
}

/** Add white Gaussian noise of sigma = sigmaRms x rms(x), drawn from `n` (kit rng) — float64, as the exporter. */
export function addNoise(x: ArrayLike<number>, sigmaRms: number, n: () => number): Float64Array {
  const y = Float64Array.from(x as ArrayLike<number>);
  if (sigmaRms <= 0) return y;
  let mu = 0;
  for (let i = 0; i < y.length; i++) mu += y[i];
  mu /= y.length;
  let v = 0;
  for (let i = 0; i < y.length; i++) v += (y[i] - mu) ** 2;
  const sd = sigmaRms * Math.sqrt(v / y.length);
  for (let i = 0; i < y.length; i++) y[i] += sd * n();
  return y;
}
