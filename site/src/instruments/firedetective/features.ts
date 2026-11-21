// Port of models/firedetective/feats.py: seq_features, static_features and area_mask_from, from a fire's raw
// observations (cumulative FEDS burned-area masks on a 180 m grid, VIIRS detections, gridMET days, LANDFIRE layers)
// to the 12 x 25 sequence and the 17 static features the network reads. No DOM.
//
// Every feature is stored as float32, as in Python. Means follow numpy's pairwise summation (float32 accumulators
// for float32 layers), percentiles numpy's linear method and rounding numpy's half-to-even, so the features match
// feats.py to the last float32 bit except where numpy calls BLAS (np.cov in the shape moments).

export const T_STEPS = 12;
export const RES = 180.0;
export const SEQ_NAMES = ['valid', 'is_day', 'logA', 'dlogA', 'dr_km', 'logN', 'logFRP', 'logFRPmean', 'frac_new', 'elong',
  'gx', 'gy', 'tmmx', 'rmin', 'rmax', 'vs', 'u_from', 'v_from', 'fm100', 'fm1000', 'erc', 'bi', 'pr', 'vpd', 'wind_hidden'];
export const NF = SEQ_NAMES.length;
export const WIND_COLS = [15, 16, 17];
export const HIDDEN_COL = 24;
export const STAT_NAMES = ['f_GR', 'f_GS', 'f_SH', 'f_TU', 'f_TL', 'f_SB', 'f_NB', 'cc', 'ch', 'cbh', 'cbd', 'slope', 'slope90',
  'elev', 'doy_s', 'doy_c', 'n_steps'];
const range = (a: number, b: number) => Array.from({ length: b - a }, (_, i) => a + i);
export const GROUPS: Record<string, number[]> = {
  GR: [...range(101, 110), 1, 2, 3], GS: range(121, 125), SH: [...range(141, 150), 4, 5, 6, 7],
  TU: range(161, 166), TL: [...range(181, 190), 8, 9, 10], SB: [...range(201, 205), 11, 12, 13],
};
const GROUP_OF = (() => {
  const m = new Map<number, string>();
  for (const [g, codes] of Object.entries(GROUPS)) for (const c of codes) m.set(c, g);
  return m;
})();
/** Fuel group of an FBFM40 code: GR, GS, SH, TU, TL, SB or NB (anything else). */
export const fuelGroup = (code: number) => GROUP_OF.get(code) ?? 'NB';

export interface Dets { row: number[]; col: number[]; frp: number[] }
export interface Wx { tmmx: number; rmin: number; rmax: number; vs: number; th: number; fm100: number; fm1000: number;
  erc: number; bi: number; pr: number; vpd: number }
export interface Layers { fuel: ArrayLike<number>; cc: ArrayLike<number>; ch: ArrayLike<number>; cbh: ArrayLike<number>;
  cbd: ArrayLike<number>; slope: ArrayLike<number>; elev: ArrayLike<number> }
/** Cumulative burned masks, step-major: masks[k * ny * nx + r * nx + c] > 0. */
export interface Masks { n: number; ny: number; nx: number; data: ArrayLike<number> }
export interface FireInput { masks: Masks; dets: Dets[]; day: boolean[]; wx: Wx[]; layers: Layers; doy: number; gamma: number }

const f32 = Math.fround;

/** numpy's pairwise summation (loops_utils.h: unrolled by 8, blocks of 128), float32 or float64 accumulators. */
export function pairwiseSum(a: ArrayLike<number>, single: boolean, lo = 0, n = a.length - lo): number {
  const F = single ? f32 : (v: number) => v;
  if (n < 8) {
    let r = 0;
    for (let i = 0; i < n; i++) r = F(r + a[lo + i]);
    return r;
  }
  if (n <= 128) {
    const r = new Array<number>(8);
    for (let j = 0; j < 8; j++) r[j] = F(a[lo + j]);
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] = F(r[j] + a[lo + i + j]);
    let res = F(F(F(r[0] + r[1]) + F(r[2] + r[3])) + F(F(r[4] + r[5]) + F(r[6] + r[7])));
    for (; i < n; i++) res = F(res + a[lo + i]);
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return F(pairwiseSum(a, single, lo, n2) + pairwiseSum(a, single, lo + n2, n - n2));
}
/** ndarray.mean(): float32 arrays reduce in float32, float64 arrays in float64. */
const mean = (a: ArrayLike<number>, single: boolean) => (single ? f32(pairwiseSum(a, true) / a.length) : pairwiseSum(a, false) / a.length);

/** np.percentile(a, q) with the default linear method (numpy's _lerp). */
export function percentile(a: ArrayLike<number>, q: number): number {
  const s = Float64Array.from(a).sort();
  const vi = (s.length - 1) * (q / 100);
  const i0 = Math.floor(vi), i1 = Math.min(i0 + 1, s.length - 1);
  const g = vi - i0, x = s[i0], y = s[i1], d = f32(y - x);   // b - a in the array's dtype (float32 layers)
  return g >= 0.5 ? y - d * (1 - g) : x + d * g;
}

/** np.round: half to even. */
export function roundHalfEven(x: number): number {
  const fl = Math.floor(x), d = x - fl;
  if (d > 0.5) return fl + 1;
  if (d < 0.5) return fl;
  return fl % 2 === 0 ? fl : fl + 1;
}

/** feats._moments: centroid (km) and elongation sqrt(l1 / l2) of a mask's cells; NaN centroid below 3 cells. */
function moments(m: ArrayLike<boolean | number>, ny: number, nx: number, rk: number): [number, number, number] {
  const xs: number[] = [], ys: number[] = [];
  for (let r = 0; r < ny; r++) for (let c = 0; c < nx; c++) if (m[r * nx + c]) { xs.push(c * rk); ys.push(-r * rk); }
  const n = xs.length;
  if (n < 3) return [NaN, NaN, 1.0];
  const cx = mean(xs, false), cy = mean(ys, false);
  let sxx = 0, syy = 0, sxy = 0;
  for (let i = 0; i < n; i++) { const dx = xs[i] - cx, dy = ys[i] - cy; sxx += dx * dx; syy += dy * dy; sxy += dx * dy; }
  const inv = 1 / (n - 1);   // np.cov multiplies by 1 / (n - 1)
  const a = sxx * inv + (rk * rk) / 12, d = syy * inv + (rk * rk) / 12, b = sxy * inv;
  // eigenvalues of the symmetric 2 x 2 [[a, b], [b, d]]
  const h = (a + d) / 2, q = Math.hypot((a - d) / 2, b);
  const lo = h - q, hi = h + q;
  return [cx, cy, Math.sqrt(hi / Math.max(lo, 1e-9))];
}

const stepMask = (M: Masks, k: number) => {
  const sz = M.ny * M.nx, out = new Uint8Array(sz);
  for (let i = 0; i < sz; i++) out[i] = M.data[k * sz + i] > 0 ? 1 : 0;
  return out;
};

/** feats.seq_features: [T_STEPS * NF] float32 values (row-major), zero rows past the record. */
export function seqFeatures(M: Masks, dets: Dets[], day: boolean[], wx: Wx[], windHidden: boolean, gammaDeg: number): Float64Array {
  const F = new Float64Array(T_STEPS * NF);
  const n = Math.min(M.n, T_STEPS), rk = RES / 1000, cellKm2 = rk * rk, ny = M.ny, nx = M.nx;
  let prevA = 0;
  let prev = new Uint8Array(ny * nx);
  let pcx: number | null = null, pcy = 0;
  for (let k = 0; k < n; k++) {
    const m = stepMask(M, k);
    let cnt = 0;
    for (let i = 0; i < m.length; i++) cnt += m[i];
    const A = cnt * cellKm2;
    const o = k * NF;
    const set = (j: number, v: number) => { F[o + j] = f32(v); };
    set(0, 1.0);
    set(1, day[k] ? 1 : 0);
    set(2, Math.log10(A + 0.01));
    set(3, k > 0 ? Math.log10(A + 0.01) - Math.log10(prevA + 0.01) : 0.0);
    set(4, Math.sqrt(A / Math.PI) - Math.sqrt(prevA / Math.PI));
    const d = dets[k], N = d.frp.length;
    set(5, Math.log10(1 + N));
    set(6, Math.log10(1 + pairwiseSum(d.frp, false)));
    set(7, N ? Math.log10(1 + pairwiseSum(d.frp, false) / N) : 0.0);
    if (N && k > 0) {
      let fresh = 0;
      for (let j = 0; j < N; j++) {
        const rr = Math.min(Math.max(roundHalfEven(d.row[j]), 0), ny - 1);
        const cc = Math.min(Math.max(roundHalfEven(d.col[j]), 0), nx - 1);
        fresh += prev[rr * nx + cc] ? 0 : 1;
      }
      set(8, fresh / N);
    } else if (N) set(8, 1.0);
    const [cx, cy, el] = moments(m, ny, nx, rk);
    set(9, Number.isFinite(el) ? Math.log(el) : 0.0);
    const fresh = new Uint8Array(m.length);
    let nnew = 0;
    for (let i = 0; i < m.length; i++) { fresh[i] = m[i] && !prev[i] ? 1 : 0; nnew += fresh[i]; }
    if (k > 0 && nnew >= 2 && pcx !== null) {
      const [nx_, ny_] = moments(fresh, ny, nx, rk);
      if (Number.isFinite(nx_)) { set(10, (nx_ - pcx) / 10.0); set(11, (ny_ - pcy) / 10.0); }
    }
    if (Number.isFinite(cx)) { pcx = cx; pcy = cy; }
    const w = wx[k];
    if (w) {
      set(12, w.tmmx / 30.0);
      set(13, w.rmin / 100.0);
      set(14, w.rmax / 100.0);
      const th = (w.th + gammaDeg) * (Math.PI / 180);
      set(15, w.vs / 10.0);
      set(16, (w.vs * Math.sin(th)) / 10.0);
      set(17, (w.vs * Math.cos(th)) / 10.0);
      set(18, w.fm100 / 20.0);
      set(19, w.fm1000 / 20.0);
      set(20, w.erc / 100.0);
      set(21, w.bi / 100.0);
      set(22, Math.log1p(Math.max(w.pr, 0.0)));
      set(23, w.vpd / 3.0);
    }
    if (windHidden) {
      for (const j of WIND_COLS) set(j, 0.0);
      set(HIDDEN_COL, 1.0);
    }
    prevA = A;
    prev = m;
  }
  return F;
}

/** feats.area_mask_from: cells within 2 km (11 cells, Euclidean) of the last step's burned area. */
export function areaMask(M: Masks): Uint8Array {
  const ny = M.ny, nx = M.nx, out = new Uint8Array(ny * nx);
  if (!M.n) return out;
  const last = stepMask(M, Math.min(M.n, T_STEPS) - 1);
  const r = Math.round(2000.0 / RES), r2 = r * r;
  for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    if (!last[y * nx + x]) continue;
    for (let dy = -r; dy <= r; dy++) {
      const yy = y + dy;
      if (yy < 0 || yy >= ny) continue;
      for (let dx = -r; dx <= r; dx++) {
        const xx = x + dx;
        if (xx >= 0 && xx < nx && dx * dx + dy * dy <= r2) out[yy * nx + xx] = 1;
      }
    }
  }
  return out;
}

/** feats.static_features: 17 float32 values. */
export function staticFeatures(L: Layers, area: Uint8Array, doy: number, nSteps: number): Float64Array {
  const S = new Float64Array(STAT_NAMES.length);
  const cells = area.length;
  let sel: number[] = [];
  for (let i = 0; i < cells; i++) if (area[i] && L.fuel[i] > 0) sel.push(i);
  if (!sel.length) for (let i = 0; i < cells; i++) if (area[i]) sel.push(i);
  const groups = ['GR', 'GS', 'SH', 'TU', 'TL', 'SB', 'NB'];
  if (sel.length) {
    const counts: Record<string, number> = {};
    for (const i of sel) { const g = fuelGroup(L.fuel[i]); counts[g] = (counts[g] ?? 0) + 1; }
    groups.forEach((g, j) => { S[j] = f32((counts[g] ?? 0) / sel.length); });
  }
  const forest = sel.filter((i) => { const g = fuelGroup(L.fuel[i]); return (g === 'TU' || g === 'TL' || g === 'SB') && L.cc[i] > 0; });
  if (forest.length) {
    const pick = (a: ArrayLike<number>) => forest.map((i) => a[i]);
    S[7] = f32(mean(pick(L.cc), true) / 100.0);
    S[8] = f32(mean(pick(L.ch), true) / 30.0);
    S[9] = f32(mean(pick(L.cbh), true) / 10.0);
    S[10] = f32(mean(pick(L.cbd), true) * 10.0);
  }
  const sl = sel.length ? sel.map((i) => L.slope[i]) : [0];
  S[11] = f32(mean(sl, sel.length > 0) / 30.0);
  S[12] = f32(percentile(sl, 90) / 45.0);
  S[13] = sel.length ? f32(mean(sel.map((i) => L.elev[i]), true) / 3000.0) : 0;
  S[14] = f32(Math.sin((2 * Math.PI * doy) / 365.25));
  S[15] = f32(Math.cos((2 * Math.PI * doy) / 365.25));
  S[16] = f32(nSteps / T_STEPS);
  return S;
}

/** pipeline.features: (seq, wind-blind seq, static) for the first 12 steps of a fire. */
export function featurize(f: FireInput) {
  const n = Math.min(f.masks.n, T_STEPS);
  const M: Masks = { ...f.masks, n };
  const seq = seqFeatures(M, f.dets, f.day, f.wx, false, f.gamma);
  const seqH = seqFeatures(M, f.dets, f.day, f.wx, true, f.gamma);
  const st = staticFeatures(f.layers, areaMask(M), f.doy, n);
  return { seq, seqH, st };
}
