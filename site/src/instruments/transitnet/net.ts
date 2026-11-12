// TransitNet in the browser: the last step of models/transitnet/pipeline.py (views + scalars: fold, bin, measure the
// depth and noise, normalise) and a line-for-line port of transitnet.py Net.forward / TransitNet.predict_views.
// The slow steps (detrending, ephemeris refinement, flare clipping) ran in Python; the exporter kept every detrended
// 2-minute point: dt (days from the nearest transit), y (relative flux - 1), and whether the transit number is odd.
import { conv1d, gelu_, linear, vgelu, vcat, softmax } from '../../lib/nn/ops.ts';
import type { Pack } from '../../lib/nn/pack.ts';
import type { Tensor } from '../../lib/nn/tensor.ts';

export const NL = 201, NS = 101, NG = 401, NO = 101;
export const CLASSES = ['planet', 'eb', 'blend', 'nothing'] as const;
export type ClassId = (typeof CLASSES)[number];

export interface TMeta { branches: Record<string, [string, string?][]>; n_head: number; mu_y: number[]; sd_y: number[]; residual: number }
export interface Star { teff: number | null; logg: number | null; rad: number | null; tmag: number; rho: number | null; have: number }
export interface TExample {
  id: string; name: string; kind: 'planet' | 'FP'; truth: ClassId;
  P: number; dur: number; duration_hours: number; n_sectors: number; ntr: number; n_points: number;
  star: Star; tic: Record<string, number | null>;
  published: { k?: number; b?: number; aR?: number; T14_h?: number; disposition?: string; comments?: string };
}

export interface Points { dt: Float32Array; y: Float32Array; odd: Float32Array; tpl: Float32Array }

export function points(X: Pack, i: number): Points {
  return { dt: X.t[`dt${i}`].data, y: X.t[`y${i}`].data, odd: X.t[`odd${i}`].data, tpl: X.t[`tpl${i}`].data };
}

// ---- changing the light curve ------------------------------------------------------------------
/** The star's mean dip at x durations from mid-transit (linear interpolation on the local grid, zero outside). */
export function tplAt(T: Float32Array, x: number): number {
  const u = ((x + 2.5) * NL) / 5 - 0.5;
  const i = Math.floor(u), f = u - i;
  const a = i >= 0 && i < NL ? T[i] : 0, b = i + 1 >= 0 && i + 1 < NL ? T[i + 1] : 0;
  return a * (1 - f) + b * f;
}

/** numpy's np.mod for floats (result has the sign of the divisor). */
const pmod = (a: number, b: number) => { const m = a % b; return m !== 0 && (m < 0) !== (b < 0) ? m + b : m; };

export interface Change {
  sec?: number;   // secondary eclipse at phase 0.5, as a fraction of the primary's depth
  oe?: number;    // odd transits deeper by oe, even ones shallower by oe
  depth?: number; // < 1: third light (an unresolved star 1/depth - 1 times as bright); > 1: a bigger eclipsing body
  noise?: number; // extra white noise per point (relative flux), on top of what the star already has
  photon?: number; // the star's own point noise; with depth < 1 the third light's photon noise is added back
  gauss?: () => number;
}

/** pipeline-equivalent perturbation, as export_instrument.perturb (plus the noise the parity cases leave out). */
export function perturb(p: Points, P: number, dur: number, c: Change): Float64Array {
  const { dt, y, odd, tpl } = p;
  const n = y.length, out = new Float64Array(n);
  const sec = c.sec ?? 0, oe = c.oe ?? 0, depth = c.depth ?? 1;
  const extra = c.noise ?? 0;
  // third light: depth -> depth/(1+c) but relative photon noise only drops by sqrt(1+c), so put the difference back
  const ph = depth < 1 && c.photon ? c.photon * Math.sqrt(Math.max(0, depth - depth * depth)) : 0;
  const sig = Math.sqrt(extra * extra + ph * ph);
  for (let i = 0; i < n; i++) {
    const d = dt[i], x = d / dur, t1 = tplAt(tpl, x);
    let S = t1 * (1 + oe * (odd[i] > 0 ? 1 : -1));
    if (sec) S = S + sec * tplAt(tpl, (pmod(d, P) - 0.5 * P) / dur);
    const base = y[i] - t1 + S;
    let v = depth < 1 ? base * depth : base + (depth - 1) * S;
    if (sig > 0 && c.gauss) v += sig * c.gauss();
    out[i] = v;
  }
  return out;
}

// ---- pipeline.views and pipeline.scalars ---------------------------------------------------------
function binned(x: ArrayLike<number>, y: ArrayLike<number>, lo: number, hi: number, nb: number, sel?: (i: number) => boolean) {
  const s = new Float64Array(nb), c = new Float64Array(nb);
  for (let i = 0; i < x.length; i++) {
    if (sel && !sel(i)) continue;
    const b = Math.floor(((x[i] - lo) / (hi - lo)) * nb);
    if (b >= 0 && b < nb) { s[b] += y[i]; c[b] += 1; }
  }
  const mu = new Float64Array(nb);
  for (let b = 0; b < nb; b++) mu[b] = c[b] > 0 ? s[b] / c[b] : NaN;
  return { mu, cnt: c };
}

function median(a: Float64Array): number {
  const b = Float64Array.from(a).sort();
  const n = b.length;
  return n ? (n % 2 ? b[(n - 1) / 2] : 0.5 * (b[n / 2 - 1] + b[n / 2])) : NaN;
}

export interface Views { local: Tensor; secondary: Tensor; glob: Tensor; oddeven: Tensor }
export interface Info { depth: number; scale: number; noise_bin: number; noise_pt: number; ntr: number }

/** pipeline.views for already-folded points: y is relative flux - 1. */
export function views(p: Points, y: Float64Array, P: number, dur: number, ntr: number): { V: Views; info: Info } {
  const n = y.length, x = new Float64Array(n), x2 = new Float64Array(n), ph = new Float64Array(n), f = new Float64Array(n), r = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const d = p.dt[i];
    x[i] = d / dur;
    x2[i] = (pmod(d, P) - 0.5 * P) / dur;
    ph[i] = d / P;
    f[i] = 1 + y[i];
    r[i] = f[i] - 1;
  }
  const loc = binned(x, r, -2.5, 2.5, NL);
  const sec = binned(x2, r, -2.5, 2.5, NS);
  const glo = binned(ph, r, -0.5, 0.5, NG);
  const ov = binned(x, r, -1.5, 1.5, NO, (i) => p.odd[i] > 0);
  const ev = binned(x, r, -1.5, 1.5, NO, (i) => !(p.odd[i] > 0));
  let ds = 0, dn = 0;
  for (let b = 0; b < NL; b++) {
    const xc = b * 0.025 + -2.5; // np.linspace(-2.5, 2.5, 201)
    if (Math.abs(xc) < 0.3 && loc.cnt[b] > 0) { ds += loc.mu[b]; dn++; }
  }
  const depth0 = dn ? -ds / dn : 0;
  const med = median(f);
  const dev = new Float64Array(n);
  for (let i = 0; i < n; i++) dev[i] = Math.abs(f[i] - med);
  const noisePt = 1.4826 * median(dev);
  const filled = Float64Array.from(Array.from(loc.cnt).filter((v) => v > 0));
  const ppb = Math.max(filled.length ? median(filled) : 1, 1);
  const noiseBin = noisePt / Math.sqrt(ppb);
  const depth = Number.isFinite(depth0) ? depth0 : 0;
  const scale = Math.max(depth, 3 * noiseBin, 2e-5);
  const pack = (rows: ArrayLike<number>[], L: number, norm: boolean[]): Tensor => {
    const o = new Float32Array(rows.length * L);
    rows.forEach((row, j) => { for (let l = 0; l < L; l++) { const v = row[l]; o[j * L + l] = norm[j] ? (Number.isFinite(v) ? v / scale : 0) : v > 0 ? 1 : 0; } });
    return { data: o, shape: [rows.length, L] };
  };
  const V: Views = {
    local: pack([loc.mu, loc.cnt], NL, [true, false]),
    secondary: pack([sec.mu, sec.cnt], NS, [true, false]),
    glob: pack([glo.mu, glo.cnt], NG, [true, false]),
    oddeven: pack([ov.mu, ev.mu, ov.cnt, ev.cnt], NO, [true, true, false, false]),
  };
  return { V, info: { depth, scale, noise_bin: noiseBin, noise_pt: noisePt, ntr } };
}

export function scalars(P: number, dur: number, info: Info, st: Star): Float32Array {
  const ok = st.have > 0;
  return Float32Array.from([
    Math.log10(P), Math.log10(dur * 24), Math.log10(info.scale), Math.log10(info.noise_bin / info.scale), Math.log10(Math.max(info.ntr, 1)),
    (st.tmag - 10) / 3, ok ? (st.teff! - 5500) / 1500 : 0, ok ? (st.logg! - 4.3) / 0.5 : 0, ok ? Math.log10(st.rho!) : 0, st.have,
  ]);
}

// ---- the network ----------------------------------------------------------------------------------
function maxpool2(x: Tensor): Tensor {
  const [C, L] = x.shape, Lo = L >> 1, o = new Float32Array(C * Lo);
  for (let c = 0; c < C; c++) for (let l = 0; l < Lo; l++) o[c * Lo + l] = Math.max(x.data[c * L + 2 * l], x.data[c * L + 2 * l + 1]);
  return { data: o, shape: [C, Lo] };
}

function branch(W: Pack<TMeta>, name: string, x: Tensor): Float32Array {
  for (const op of W.meta.branches[name]) {
    if (op[0] === 'conv') x = gelu_(conv1d(x, W.t[`${op[1]}_w`], W.t[`${op[1]}_b`]));
    else x = maxpool2(x);
  }
  const [C, L] = x.shape, out = new Float32Array(2 * C);
  for (let c = 0; c < C; c++) {
    let s = 0, m = -Infinity;
    for (let l = 0; l < L; l++) { const v = x.data[c * L + l]; s += v; if (v > m) m = v; }
    out[c] = s / L; out[C + c] = m;
  }
  return out;
}

/** Net.forward: 12 raw outputs (4 class logits, 4 parameter means, 4 log sigmas). */
export function forward(W: Pack<TMeta>, V: Views, s: ArrayLike<number>): Float32Array {
  let z = vcat(branch(W, 'loc', V.local), branch(W, 'sec', V.secondary), branch(W, 'glo', V.glob), branch(W, 'oe', V.oddeven), s);
  for (let j = 0; j < W.meta.n_head; j++) {
    z = linear(z, W.t[`h${j}_w`], W.t[`h${j}_b`]);
    if (j < W.meta.n_head - 1) vgelu(z);
  }
  return z;
}

const flipT = (t: Tensor): Tensor => {
  const [C, L] = t.shape, o = new Float32Array(t.data.length);
  for (let c = 0; c < C; c++) for (let l = 0; l < L; l++) o[c * L + l] = t.data[c * L + L - 1 - l];
  return { data: o, shape: [C, L] };
};
export const flip = (V: Views): Views => ({ local: flipT(V.local), secondary: flipT(V.secondary), glob: flipT(V.glob), oddeven: flipT(V.oddeven) });

export interface Reading {
  p: Record<ClassId, number>;
  log_k: number; log_k_sig: number; b: number; b_sig: number; log_aR: number; log_aR_sig: number; log_T14: number; log_T14_sig: number;
  k: number; k_lo: number; k_hi: number; b_lo: number; b_hi: number; aR: number; aR_lo: number; aR_hi: number; T14_h: number;
  /** the model-free estimates the network corrects: sqrt(depth scale), P / (pi T14), the catalogue duration */
  k0: number; aR0: number;
}

/** TransitNet.predict_views for one star: both directions in time averaged. */
export function predict(W: Pack<TMeta>, V: Views, s: Float32Array): { r: Reading; raw: Float32Array } {
  const o1 = forward(W, V, s), o2 = forward(W, flip(V), s);
  const { mu_y: mu, sd_y: sd, residual } = W.meta;
  const p1 = softmax(o1.subarray(0, 4)), p2 = softmax(o2.subarray(0, 4));
  const p = Object.fromEntries(CLASSES.map((c, j) => [c, (p1[j] + p2[j]) / 2])) as Record<ClassId, number>;
  const base = residual ? [0.5 * s[2], 0, s[0] - Math.log10(Math.PI) - s[1] + Math.log10(24), s[1]] : [0, 0, 0, 0];
  const clip = (v: number) => Math.max(-6, Math.min(3, v));
  const m: number[] = [], g: number[] = [];
  for (let j = 0; j < 4; j++) {
    const a = o1[4 + j], b = o2[4 + j];
    m.push(((a + b) / 2) * sd[j] + mu[j] + base[j]);
    g.push(Math.sqrt(((Math.exp(2 * clip(o1[8 + j])) + Math.exp(2 * clip(o2[8 + j]))) / 2) * sd[j] ** 2 + ((a - b) / 2) ** 2 * sd[j] ** 2));
  }
  const r: Reading = {
    p, log_k: m[0], log_k_sig: g[0], b: m[1], b_sig: g[1], log_aR: m[2], log_aR_sig: g[2], log_T14: m[3], log_T14_sig: g[3],
    k: 10 ** m[0], k_lo: 10 ** (m[0] - g[0]), k_hi: 10 ** (m[0] + g[0]),
    b_lo: Math.max(0, m[1] - g[1]), b_hi: m[1] + g[1],
    aR: 10 ** m[2], aR_lo: 10 ** (m[2] - g[2]), aR_hi: 10 ** (m[2] + g[2]), T14_h: 10 ** m[3],
    k0: 10 ** base[0], aR0: 10 ** base[2],
  };
  return { r, raw: vcat(o1, o2) };
}

/** Everything from the folded points to the reading. */
export function readStar(W: Pack<TMeta>, e: TExample, p: Points, c: Change) {
  const y = perturb(p, e.P, e.dur, c);
  const { V, info } = views(p, y, e.P, e.dur, e.ntr);
  const s = scalars(e.P, e.dur, info, e.star);
  const { r, raw } = predict(W, V, s);
  return { y, V, info, s, r, raw };
}
