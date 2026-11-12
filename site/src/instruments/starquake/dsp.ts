// StarQuake's reading pipeline in the browser: a line-for-line port of models/starquake/pipeline.py
// (fill_short -> highpass -> psd -> features, scalars), as the original page's training/sqnet.js was.
// Float64 throughout, like numpy. No DOM.

export const DT = 1765.4637; // Kepler long cadence (s)
export const NYQ = 1e6 / (2 * DT); // 283.2 uHz
export const NB = 1024, NC = 48, STEP = 100, NW = 700, L0 = 56, NL = 110;
export const EDGES = Float64Array.from({ length: NB + 1 }, (_, i) => Math.exp(Math.log(3) + ((Math.log(283) - Math.log(3)) * i) / NB));
export const FC = Float64Array.from({ length: NB }, (_, i) => Math.sqrt(EDGES[i] * EDGES[i + 1]));
export const CEN = Float64Array.from({ length: NC }, (_, j) => Math.exp(Math.log(6) + ((Math.log(260) - Math.log(6)) * j) / (NC - 1)));
/** Expected large separation for a given numax (Mosser et al. 2012). */
export const dnuS = (numax: number) => 0.276 * Math.pow(numax, 0.751);

// ---- FFT: radix-2 in place, Bluestein for any length ----
export function fft2(re: Float64Array, im: Float64Array, inv: boolean) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let b = n >> 1;
    for (; j & b; b >>= 1) j ^= b;
    j ^= b;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inv ? 2 : -2) * Math.PI) / len, h = len >> 1;
    const tw = new Float64Array(2 * h); // exact twiddles per stage (no recurrence drift)
    for (let k = 0; k < h; k++) { tw[2 * k] = Math.cos(ang * k); tw[2 * k + 1] = Math.sin(ang * k); }
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < h; k++) {
        const a = i + k, b = a + h, cr = tw[2 * k], ci = tw[2 * k + 1];
        const xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
      }
    }
  }
}

/** Forward DFT of a real sequence of any length (Bluestein). Returns [re, im] of length n. */
export function fftReal(x: Float64Array): [Float64Array, Float64Array] {
  const n = x.length;
  if ((n & (n - 1)) === 0) { const r = Float64Array.from(x), i = new Float64Array(n); fft2(r, i, false); return [r, i]; }
  let m = 1;
  while (m < 2 * n - 1) m <<= 1;
  const cr = new Float64Array(n), ci = new Float64Array(n);
  for (let k = 0; k < n; k++) { const a = (Math.PI * ((k * k) % (2 * n))) / n; cr[k] = Math.cos(a); ci[k] = -Math.sin(a); }
  const ar = new Float64Array(m), ai = new Float64Array(m), br = new Float64Array(m), bi = new Float64Array(m);
  for (let k = 0; k < n; k++) { ar[k] = x[k] * cr[k]; ai[k] = x[k] * ci[k]; }
  br[0] = cr[0]; bi[0] = -ci[0];
  for (let k = 1; k < n; k++) { br[k] = br[m - k] = cr[k]; bi[k] = bi[m - k] = -ci[k]; }
  fft2(ar, ai, false); fft2(br, bi, false);
  for (let k = 0; k < m; k++) { const r = ar[k] * br[k] - ai[k] * bi[k], i = ar[k] * bi[k] + ai[k] * br[k]; ar[k] = r; ai[k] = i; }
  fft2(ar, ai, true);
  const or = new Float64Array(n), oi = new Float64Array(n);
  for (let k = 0; k < n; k++) { const r = ar[k] / m, i = ai[k] / m; or[k] = r * cr[k] - i * ci[k]; oi[k] = r * ci[k] + i * cr[k]; }
  return [or, oi];
}

// ---- processing (pipeline.py) ----
function median(a: Float64Array): number {
  const s = Float64Array.from(a).sort();
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

/** Linearly interpolate gaps of <= maxgap cadences (fill_short). */
export function fillShort(y: Float64Array, maxgap = 3): Float64Array {
  const n = y.length, out = Float64Array.from(y);
  let i = 0;
  while (i < n) {
    if (Number.isFinite(y[i])) { i++; continue; }
    let j = i;
    while (j < n && !Number.isFinite(y[j])) j++;
    if (i > 0 && j < n && j - i <= maxgap) {
      const a = y[i - 1], b = y[j];
      for (let k = i; k < j; k++) out[k] = a + ((b - a) * (k - (i - 1))) / (j - (i - 1));
    }
    i = j;
  }
  return out;
}

/** 10-day running-mean high-pass, then a 5-sigma (MAD) clip. */
export function highpass(y: Float64Array, days = 10, clip = 5): Float64Array {
  const n = y.length, w = Math.round((days * 86400) / DT) | 1, h = (w - 1) >> 1;
  const cx = new Float64Array(n + 1), cm = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) { const ok = Number.isFinite(y[i]); cx[i + 1] = cx[i] + (ok ? y[i] : 0); cm[i + 1] = cm[i] + (ok ? 1 : 0); }
  const out = new Float64Array(n);
  let nv = 0;
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(y[i])) { out[i] = NaN; continue; }
    const a = Math.max(0, i - h), b = Math.min(n, i + h + 1);
    out[i] = y[i] - (cx[b] - cx[a]) / Math.max(cm[b] - cm[a], 1);
    nv++;
  }
  const vals = new Float64Array(nv);
  for (let i = 0, k = 0; i < n; i++) if (Number.isFinite(out[i])) vals[k++] = out[i];
  const med = median(vals);
  for (let k = 0; k < nv; k++) vals[k] = Math.abs(vals[k] - med);
  const mad = 1.4826 * median(vals);
  for (let i = 0; i < n; i++) if (Number.isFinite(out[i]) && Math.abs(out[i] - med) > clip * mad) out[i] = NaN;
  return out;
}

/** Single-sided power density (ppm^2/uHz), Parseval-calibrated for the duty cycle; frequencies in uHz. */
export function psd(y: Float64Array): { f: Float64Array; P: Float64Array } {
  const n = y.length;
  let s = 0, nv = 0;
  for (let i = 0; i < n; i++) if (Number.isFinite(y[i])) { s += y[i]; nv++; }
  const mu = s / Math.max(nv, 1), x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = Number.isFinite(y[i]) ? y[i] - mu : 0;
  const [R, I] = fftReal(x);
  const nf = Math.floor(n / 2), f = new Float64Array(nf), P = new Float64Array(nf), c = ((2 * DT) / Math.max(nv, 1)) * 1e-6;
  for (let k = 1; k <= nf; k++) { f[k - 1] = (k * 1e6) / (n * DT); P[k - 1] = c * (R[k] * R[k] + I[k] * I[k]); }
  return { f, P };
}

/** Area-preserving rebin (np.interp on the cumulative sum): returns a function of the bin edges. */
export function makeRebin(f: Float64Array, P: Float64Array) {
  const n = f.length, df = f[1] - f[0], e0 = f[0] - df / 2, C = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) C[i + 1] = C[i] + P[i] * df;
  const cum = (x: number) => {
    const t = (x - e0) / df;
    if (t <= 0) return 0;
    if (t >= n) return C[n];
    const i = Math.floor(t);
    return C[i] + (C[i + 1] - C[i]) * (t - i);
  };
  return (edges: ArrayLike<number>) => {
    const out = new Float64Array(edges.length - 1);
    let prev = cum(edges[0]);
    for (let i = 1; i < edges.length; i++) { const c = cum(edges[i]); out[i - 1] = (c - prev) / (edges[i] - edges[i - 1]); prev = c; }
    return out;
  };
}

function solve3(A: number[][], b: number[]): number[] {
  const M = A.map((r, i) => r.concat([b[i]]));
  for (let c = 0; c < 3; c++) {
    let p = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < 3; r++) if (r !== c) { const k = M[r][c] / M[c][c]; for (let q = c; q < 4; q++) M[r][q] -= k * M[c][q]; }
  }
  return [M[0][3] / M[0][0], M[1][3] / M[1][1], M[2][3] / M[2][2]];
}

/** F1: log10 power on 1024 log bins (3-283 uHz). F2: the 48 x 110 comb map. */
export function features(f: Float64Array, P: Float64Array): { F1: Float32Array; F2: Float32Array } {
  const rb = makeRebin(f, P), b = rb(EDGES), F1 = new Float32Array(NB);
  for (let i = 0; i < NB; i++) F1[i] = Math.log10(Math.max(b[i], 1e-6));
  const F2 = new Float32Array(NC * NL), fmax = f[f.length - 1], t = new Float64Array(NW);
  for (let i = 0; i < NW; i++) t[i] = -1 + (2 * i) / (NW - 1);
  const e = new Float64Array(NW + 1), ec = new Float64Array(NW + 1), ok = new Uint8Array(NW), z = new Float64Array(NW);
  for (let j = 0; j < NC; j++) {
    const d = dnuS(CEN[j]);
    let nok = 0;
    for (let i = 0; i <= NW; i++) { e[i] = CEN[j] + ((i - NW / 2) * d) / STEP; ec[i] = Math.min(Math.max(e[i], f[0]), fmax); }
    for (let i = 0; i < NW; i++) { ok[i] = e[i] > f[0] && e[i + 1] < fmax ? 1 : 0; nok += ok[i]; }
    if (nok < NW / 2) continue;
    const y = rb(ec), A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], r = [0, 0, 0];
    for (let i = 0; i < NW; i++) {
      y[i] = Math.log(Math.max(y[i], 1e-12));
      if (!ok[i]) continue;
      const v = [t[i] * t[i], t[i], 1];
      for (let a = 0; a < 3; a++) { r[a] += v[a] * y[i]; for (let c = 0; c < 3; c++) A[a][c] += v[a] * v[c]; }
    }
    const co = solve3(A, r);
    let mz = 0;
    for (let i = 0; i < NW; i++) { z[i] = ok[i] ? Math.exp(y[i] - (co[0] * t[i] * t[i] + co[1] * t[i] + co[2])) - 1 : 0; if (ok[i]) mz += z[i]; }
    mz /= nok;
    for (let i = 0; i < NW; i++) if (ok[i]) z[i] -= mz;
    let r0 = 0;
    for (let i = 0; i < NW; i++) r0 += z[i] * z[i];
    r0 = Math.max(r0, 1e-12);
    for (let l = 0; l < NL; l++) {
      const lag = L0 + l;
      let s = 0;
      for (let i = 0; i + lag < NW; i++) s += z[i] * z[i + lag];
      F2[j * NL + l] = s / r0;
    }
  }
  return { F1, F2 };
}

/** Observing length (days) and duty cycle -> the network's two scalars. */
export function scalars(y: Float64Array): { s: [number, number]; T: number } {
  let first = -1, last = -1, n = 0;
  for (let i = 0; i < y.length; i++) if (Number.isFinite(y[i])) { if (first < 0) first = i; last = i; n++; }
  const T = first < 0 ? 1 : ((last - first + 1) * DT) / 86400;
  return { s: [Math.fround(Math.log10(Math.max(T, 1)) - 2.5), Math.fround(n / y.length)], T }; // float32, as in Python
}

export interface Prepared { F1: Float32Array; F2: Float32Array; s: [number, number]; f: Float64Array; P: Float64Array; T: number }

/** ppm light curve on the cadence grid (NaN in gaps) -> network inputs, plus the spectrum for drawing. */
export function prepare(y: Float64Array): Prepared {
  const yh = highpass(fillShort(y));
  const { f, P } = psd(yh);
  const { F1, F2 } = features(f, P);
  const { s, T } = scalars(yh);
  return { F1, F2, s, f, P, T };
}
