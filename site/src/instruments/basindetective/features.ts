// BasinDetective's input features in the browser: a line-by-line port of models/basindetective/basindetective/
// features.py (sigs, _climate, featurize) and of the numpy/pandas calls it makes (np.quantile 'linear' with
// numpy's _lerp, np.median, np.std, np.corrcoef, np.polyfit degree 1, pandas rolling(7).mean, interpolate).
// Every feature is rounded to float32 at the end, as featurize() returns float32. No DOM: the parity test runs it.

export const EPSQ = 0.01;
const FDC_P = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99];
const FDC_NAMES = ['01', '05', '10', '20', '30', '50', '70', '80', '90', '95', '99'];
const CKEYS = ['logP', 'logPET', 'arid', 'Tmean', 'fsnow', 'fwarm', 'wetfrac', 'p99rel', 'pseas', 'pcv'] as const;

/** Daily forcing with its calendar: month 1-12 and water year (year + 1 from October) for every day. */
export interface Days { P: Float64Array; T: Float64Array; E: Float64Array; month: Int32Array; wy: Int32Array }

/** Calendar of `n` consecutive days from an ISO date (UTC). */
export function calendar(start: string, n: number): { month: Int32Array; wy: Int32Array } {
  const month = new Int32Array(n), wy = new Int32Array(n);
  const t0 = Date.parse(start + 'T00:00:00Z');
  for (let i = 0; i < n; i++) {
    const d = new Date(t0 + i * 86400000);
    const m = d.getUTCMonth() + 1;
    month[i] = m;
    wy[i] = d.getUTCFullYear() + (m >= 10 ? 1 : 0);
  }
  return { month, wy };
}

// ---- numpy equivalents ---------------------------------------------------------------------------
const sum = (a: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s; };
export const mean = (a: ArrayLike<number>) => sum(a) / a.length;
function std(a: ArrayLike<number>) {
  const m = mean(a);
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - m; s += d * d; }
  return Math.sqrt(s / a.length);
}
const finite = (a: ArrayLike<number>) => Array.from(a).filter((v) => !Number.isNaN(v));
const nanmean = (a: ArrayLike<number>) => { const f = finite(a); return f.length ? mean(f) : NaN; };
const nanstd = (a: ArrayLike<number>) => { const f = finite(a); return f.length ? std(f) : NaN; };
const sorted = (a: ArrayLike<number>) => Float64Array.from(a).sort();

/** np.quantile(a, q) with the default 'linear' method, on an already sorted array. */
export function quantileSorted(s: ArrayLike<number>, q: number) {
  const n = s.length;
  const vi = (n - 1) * q;
  if (vi >= n - 1) return s[n - 1];
  const lo = Math.floor(vi);
  const a = s[lo], b = s[lo + 1], t = vi - lo;
  const d = b - a;
  return t >= 0.5 ? b - d * (1 - t) : a + d * t;
}
function medianSorted(s: ArrayLike<number>) {
  const n = s.length;
  if (n === 0) return NaN;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}
const median = (a: ArrayLike<number>) => medianSorted(sorted(a));
const nanmedian = (a: ArrayLike<number>) => median(finite(a));

/** features.corr: Pearson r (np.corrcoef), 0 when either side is constant. */
function corr(a: ArrayLike<number>, b: ArrayLike<number>) {
  if (std(a) === 0 || std(b) === 0) return 0;
  const n = a.length, ma = mean(a), mb = mean(b);
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
  const r = (sab / (n - 1)) / Math.sqrt(saa / (n - 1)) / Math.sqrt(sbb / (n - 1));
  return Math.max(-1, Math.min(1, r));
}
const clip = (v: number, lo: number, hi: number) => (Number.isNaN(v) ? NaN : Math.min(hi, Math.max(lo, v)));

/** pandas Series.rolling(7).mean().values[6:] */
function rolling7(a: ArrayLike<number>) {
  const n = a.length, out = new Float64Array(Math.max(0, n - 6));
  for (let i = 6; i < n; i++) { let s = 0; for (let k = i - 6; k <= i; k++) s += a[k]; out[i - 6] = s / 7; }
  return out;
}

/** Linear interpolation of NaN runs strictly inside the series; at most `limit` values per run (forward). */
function interpInside(q: Float64Array, limit = Infinity) {
  const o = Float64Array.from(q);
  let i = 0;
  while (i < o.length) {
    if (!Number.isNaN(o[i])) { i++; continue; }
    let j = i;
    while (j < o.length && Number.isNaN(o[j])) j++;
    if (i > 0 && j < o.length) {
      const a = o[i - 1], b = o[j], span = j - i + 1;
      for (let k = i; k < j && k - i < limit; k++) o[k] = a + ((b - a) * (k - i + 1)) / span;
    }
    i = j;
  }
  return o;
}

// ---- signature filters (features._lh, _eck, _rec_pairs) ------------------------------------------
function lyneHollick(q: Float64Array, alpha: number, passes: number) {
  let b = Float64Array.from(q);
  const n = q.length;
  for (let p = 0; p < passes; p++) {
    const x = p % 2 === 0 ? Float64Array.from(b) : Float64Array.from(b).reverse();
    const out = new Float64Array(n);
    let qf = 0;
    out[0] = x[0];
    for (let t = 1; t < n; t++) {
      qf = alpha * qf + ((1 + alpha) / 2) * (x[t] - x[t - 1]);
      if (qf < 0) qf = 0;
      out[t] = x[t] - Math.min(qf, x[t]);
    }
    b = p % 2 === 0 ? out : out.reverse();
  }
  return b;
}
function eckhardt(q: Float64Array, a: number, bfimax: number) {
  const n = q.length, b = new Float64Array(n);
  b[0] = q[0] * bfimax;
  for (let t = 1; t < n; t++) b[t] = Math.min(q[t], ((1 - bfimax) * a * b[t - 1] + (1 - a) * bfimax * q[t]) / (1 - a * bfimax));
  return b;
}
function bfi(q: Float64Array, method: 'lh' | 'eck') {
  const s0 = q.some(Number.isNaN) ? Float64Array.from(finite(interpInside(q, 3))) : q;
  if (s0.length < 365) return NaN;
  const b = method === 'lh' ? lyneHollick(s0, 0.925, 3) : eckhardt(s0, 0.98, 0.8);
  const tot = sum(s0);
  return tot > 0 ? sum(b) / tot : NaN;
}
function recession(v: Float64Array): [number, number, number] {
  const n = v.length - 1, minLen = 5, skip = 2;
  const dq = new Float64Array(n), qm = new Float64Array(n), dec = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    dq[i] = v[i + 1] - v[i];
    qm[i] = 0.5 * (v[i + 1] + v[i]);
    dec[i] = dq[i] < 0 && Number.isFinite(dq[i]) && qm[i] > 0 ? 1 : 0;
  }
  const xs: number[] = [], ys: number[] = [];
  let i = 0;
  while (i < n) {
    if (dec[i]) {
      let j = i;
      while (j < n && dec[j]) j++;
      if (j - i >= minLen) for (let k = i + skip; k < j; k++) { xs.push(qm[k]); ys.push(-dq[k]); }
      i = j;
    } else i++;
  }
  const lx: number[] = [], ly: number[] = [], ratio: number[] = [];
  for (let k = 0; k < xs.length; k++) if (xs[k] > 0 && ys[k] > 0) { lx.push(Math.log(xs[k])); ly.push(Math.log(ys[k])); ratio.push(xs[k] / ys[k]); }
  if (xs.length === 0 || lx.length < 20) return [NaN, NaN, NaN];
  // np.polyfit(log x, log y, 1): least squares slope and intercept
  const mx = mean(lx), my = mean(ly);
  let sxy = 0, sxx = 0;
  for (let k = 0; k < lx.length; k++) { sxy += (lx[k] - mx) * (ly[k] - my); sxx += (lx[k] - mx) ** 2; }
  const slope = sxy / sxx, icpt = my - slope * mx;
  return [Math.exp(icpt), slope, median(ratio)];
}

export interface Sigs {
  q_mean: number; q5: number; q95: number; high_q_freq: number; low_q_freq: number; zero_q_freq: number;
  slope_fdc: number; bfi_lh: number; bfi_eckhardt: number; rec_a: number; rec_b: number; rec_k_days: number;
  hfd_mean: number; runoff_ratio: number; elasticity_p: number;
}

/** features.sigs (the signatures the network reads; aridity and frac_snow are not features and are skipped). */
export function sigs(q: Float64Array, P: Float64Array, wyAll: Int32Array): Sigs | null {
  const idx: number[] = [];
  for (let i = 0; i < q.length; i++) if (Number.isFinite(q[i])) idx.push(i);
  const qq = Float64Array.from(idx, (i) => q[i]);
  if (qq.length < 365) return null;
  const s = sorted(qq);
  const out = {} as Sigs;
  out.q_mean = mean(qq);
  out.q5 = quantileSorted(s, 0.05);
  out.q95 = quantileSorted(s, 0.95);
  const med = medianSorted(s);
  let hi = 0, lo = 0, zero = 0;
  for (const v of qq) { if (v > 9 * med) hi++; if (v < 0.2 * out.q_mean) lo++; if (v <= 0) zero++; }
  out.high_q_freq = (hi / qq.length) * 365.25;
  out.low_q_freq = (lo / qq.length) * 365.25;
  out.zero_q_freq = zero / qq.length;
  const q33 = Math.log(Math.max(quantileSorted(s, 0.67), 1e-6));
  const q66 = Math.log(Math.max(quantileSorted(s, 0.34), 1e-6));
  out.slope_fdc = (q66 - q33) / (0.66 - 0.33);
  out.bfi_lh = bfi(q, 'lh');
  out.bfi_eckhardt = bfi(q, 'eck');
  [out.rec_a, out.rec_b, out.rec_k_days] = recession(q);
  const wy = Int32Array.from(idx, (i) => wyAll[i]);
  const years = Array.from(new Set(wy)).sort((a, b) => a - b);
  const hfd: number[] = [], full: number[] = [];
  for (const y of years) {
    const g: number[] = [];
    for (let k = 0; k < qq.length; k++) if (wy[k] === y) g.push(qq[k]);
    if (g.length >= 330) {
      const c = new Float64Array(g.length);
      let acc = 0;
      for (let k = 0; k < g.length; k++) { acc += g[k]; c[k] = acc; }
      const half = c[c.length - 1] / 2;
      let j = 0;
      while (j < c.length && c[j] < half) j++;
      hfd.push(j);
      full.push(y);
    }
  }
  out.hfd_mean = hfd.length ? mean(hfd) : NaN;
  const Pv = Float64Array.from(idx, (i) => P[i]);
  const ps = sum(finite(Pv));
  out.runoff_ratio = ps > 0 ? sum(qq) / ps : NaN;
  out.elasticity_p = NaN;
  if (full.length >= 3) {
    const aq = full.map((y) => { let a = 0; for (let k = 0; k < qq.length; k++) if (wy[k] === y) a += qq[k]; return a; });
    const ap = full.map((y) => { let a = 0; for (let k = 0; k < qq.length; k++) if (wy[k] === y && !Number.isNaN(Pv[k])) a += Pv[k]; return a; });
    if (std(ap) > 0) {
      const maq = mean(aq), map = mean(ap);
      const r = aq.map((v, k) => { const dp = ap[k] - map; return Math.abs(dp) > 1e-9 ? (v - maq) / dp : NaN; });
      out.elasticity_p = (nanmedian(r) * map) / maq;
    }
  }
  return out;
}

interface Climate { c: Record<(typeof CKEYS)[number], number>; Pm: number[]; Em: number[]; Tm: number[] }

/** features._climate */
export function climate(d: Days): Climate {
  const { P, T, E, month, wy } = d;
  const pm = Math.max(mean(P), 1e-3), em = Math.max(mean(E), 1e-3);
  const Pm: number[] = [], Em: number[] = [], Tm: number[] = [];
  for (let m = 1; m <= 12; m++) {
    let sp = 0, se = 0, st = 0, c = 0;
    for (let i = 0; i < P.length; i++) if (month[i] === m) { sp += P[i]; se += E[i]; st += T[i]; c++; }
    Pm.push(sp / c / pm); Em.push(se / c / em); Tm.push(st / c);
  }
  let tot = 0, snow = 0, warm = 0, wet = 0;
  for (let i = 0; i < P.length; i++) { tot += P[i]; if (T[i] < 0) snow += P[i]; if (T[i] > 10) warm += P[i]; if (P[i] > 1.0) wet++; }
  const ys = Array.from(new Set(wy)).sort((a, b) => a - b);
  const ap = ys.map((y) => { let a = 0; for (let i = 0; i < P.length; i++) if (wy[i] === y) a += P[i]; return a; });
  return {
    c: {
      logP: Math.log(pm), logPET: Math.log(em), arid: em / pm, Tmean: mean(T),
      fsnow: snow / Math.max(tot, 1e-9), fwarm: warm / Math.max(tot, 1e-9), wetfrac: wet / P.length,
      p99rel: quantileSorted(sorted(P), 0.99) / pm,
      pseas: std(Pm) > 0 && std(Em) > 0 ? corr(Pm, Em) : 0,
      pcv: ys.length > 1 ? std(ap) / Math.max(mean(ap), 1e-9) : 0,
    },
    Pm, Em, Tm,
  };
}

/** features.featurize: calibration forcing + calibration Q (NaN = missing) + target forcing -> 150 features. */
export function featurize(cal: Days, qcal: Float64Array, tgt: Days): { x: Float64Array; names: string[]; sig: Sigs } {
  const f = new Map<string, number>();
  const cc = climate(cal), tc = climate(tgt);
  for (const k of CKEYS) f.set(`c_${k}`, cc.c[k]);
  for (const k of CKEYS) {
    f.set(`t_${k}`, tc.c[k]);
    if (k !== 'wetfrac') f.set(`dt_${k}`, tc.c[k] - cc.c[k]);
  }
  f.set('dt_wetfrac', tc.c.wetfrac - cc.c.wetfrac);
  for (let m = 0; m < 12; m++) {
    f.set(`c_Pm${m}`, cc.Pm[m]);
    f.set(`c_Em${m}`, cc.Em[m]);
    f.set(`c_Tm${m}`, cc.Tm[m]);
    f.set(`dt_Pm${m}`, tc.Pm[m] * Math.exp(tc.c.logP - cc.c.logP) - cc.Pm[m]);
    f.set(`dt_Tm${m}`, tc.Tm[m] - cc.Tm[m]);
  }
  const qv = qcal;
  const s = sigs(qv, cal.P, cal.wy);
  if (!s) throw new Error('fewer than 365 days of flow');
  const okIdx: number[] = [];
  for (let i = 0; i < qv.length; i++) if (Number.isFinite(qv[i])) okIdx.push(i);
  const qq = Float64Array.from(okIdx, (i) => qv[i]);
  const qm = Math.max(mean(qq), 1e-4);
  f.set('q_missing', 1 - okIdx.length / qv.length);
  f.set('q_logmean', Math.log(qm + EPSQ));
  f.set('q_rr', s.runoff_ratio);
  f.set('q_logrr', Number.isNaN(s.runoff_ratio) ? NaN : Math.log(Math.max(s.runoff_ratio, 1e-3)));
  f.set('q_bfi_lh', s.bfi_lh);
  f.set('q_bfi_eck', s.bfi_eckhardt);
  f.set('q_logreck', Number.isFinite(s.rec_k_days) ? Math.log(clip(s.rec_k_days, 0.3, 3000)) : NaN);
  f.set('q_recb', Number.isFinite(s.rec_b) ? clip(s.rec_b, -1, 5) : NaN);
  f.set('q_slopefdc', clip(s.slope_fdc, -1, 30));
  f.set('q_hfd', s.hfd_mean / 365.0);
  f.set('q_logq5', Math.log(s.q5 + EPSQ));
  f.set('q_logq95', Math.log(s.q95 + EPSQ));
  f.set('q_highfreq', s.high_q_freq / 365.25);
  f.set('q_lowfreq', s.low_q_freq / 365.25);
  f.set('q_zerofreq', s.zero_q_freq);
  f.set('q_elast', clip(s.elasticity_p, -2, 6));
  const sq = sorted(qq);
  FDC_P.forEach((p, j) => f.set(`q_fdc${FDC_NAMES[j]}`, Math.log(quantileSorted(sq, 1 - p) + 1e-4) - Math.log(qm)));
  for (let m = 0; m < 12; m++) {
    let a = 0, c = 0;
    for (let k = 0; k < okIdx.length; k++) if (cal.month[okIdx[k]] === m + 1) { a += qq[k]; c++; }
    f.set(`q_m${m}`, c ? a / c / qm : NaN);
  }
  // day-to-day dynamics on the gap-filled series (interpolate inside, then bfill, ffill)
  const qf = interpInside(qv);
  { let first = qf.findIndex((v) => !Number.isNaN(v)); for (let i = 0; i < first; i++) qf[i] = qf[first];
    for (let i = 1; i < qf.length; i++) if (Number.isNaN(qf[i])) qf[i] = qf[i - 1]; }
  const lqf = Float64Array.from(qf, (v) => Math.log(v + 1e-3));
  const P = cal.P, n = P.length;
  for (const k of [0, 1, 2, 3, 5, 10, 30]) f.set(`q_pq${k}`, corr(P.subarray(0, n - k), qf.subarray(k)));
  for (const k of [1, 3, 10, 30, 90]) f.set(`q_ac${k}`, corr(lqf.subarray(0, n - k), lqf.subarray(k)));
  let absd = 0, flat = 0, rise = 0;
  for (let i = 1; i < n; i++) {
    const d = qf[i] - qf[i - 1];
    absd += Math.abs(d);
    if (Math.abs(d) < 0.01 * qf[i] + 1e-6) flat++;
    if (d > 0) rise++;
  }
  f.set('q_flash', absd / Math.max(sum(qf), 1e-9));
  f.set('q_flat', flat / (n - 1));
  f.set('q_rise', rise / (n - 1));
  const q7 = rolling7(qf), p7 = rolling7(P);
  let q7min = Infinity;
  for (const v of q7) if (v < q7min) q7min = v;
  f.set('q_min7', Math.log(q7min + 1e-3) - Math.log(qm));
  f.set('q_cv7_over_p', std(q7) / qm / Math.max(std(p7) / Math.max(mean(P), 1e-3), 1e-3));
  const ys = Array.from(new Set(cal.wy)).sort((a, b) => a - b);
  const aq = ys.map((y) => { const g: number[] = []; for (let i = 0; i < n; i++) if (cal.wy[i] === y) g.push(qv[i]); return nanmean(g); });
  const ap = ys.map((y) => { let a = 0, c = 0; for (let i = 0; i < n; i++) if (cal.wy[i] === y) { a += P[i]; c++; } return a / c; });
  f.set('q_anncv', nanstd(aq) / qm);
  const aqm = nanmean(aq);
  f.set('q_annpcorr', ys.length > 2 ? corr(aq.map((v) => (Number.isNaN(v) ? aqm : v)), ap) : 0);
  const qmon = Array.from({ length: 12 }, (_, m) => f.get(`q_m${m}`)!);
  const pmean = Math.exp(cc.c.logP), emean = Math.exp(cc.c.logPET);
  const wmon = cc.Pm.map((v, m) => Math.max(v * pmean - cc.Em[m] * emean * 0.5, 0) + 1e-3);
  const wm = mean(wmon);
  f.set('q_seasamp', nanstd(qmon));
  f.set('q_seasdamp', nanstd(qmon) / Math.max(std(wmon.map((v) => v / wm)), 1e-3));
  const names = Array.from(f.keys());
  const x = Float64Array.from(names, (k) => { const v = f.get(k)!; return Number.isFinite(v) ? Math.fround(v) : NaN; });
  return { x, names, sig: s };
}
