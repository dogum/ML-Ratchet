// PlantDetective in the browser: a pure port of models/plantdetective/{pdcore.features, plantdetective.window_vector,
// Detective.windows, ens_forward, pool, Detective.infer}. No DOM, so the parity test runs it in Node.
// The feature code follows the release's own JS port (training/pd.js) line by line; percentiles follow numpy's
// 'linear' method exactly, and the histogram edges come from Python (np.linspace) rather than being recomputed.
import type { Pack } from '../../lib/nn/pack.ts';

export interface PMeta {
  members: number;
  factors: Record<string, number>;
  techs: string[];
  reg: string[];
  n_params: number;
  flex: string[];
  loadbins: number[];
  run_edges: number[];
  zbins: number[];
  ctx_keys: string[];
  hot_h: number;
  warm_h: number;
  part_min: number;
}
export interface Ctx {
  pm_ct?: number | null; pm_st?: number | null; pm_ca?: number | null; fuel_coal?: number | null; chp?: number | null;
  np_own?: number | null; np_st_share?: number | null; minload?: number | null; cold_h?: number | null;
  online_year?: number | null; hr923?: Record<string, number>;
}
export interface Unit {
  id: string; name: string; unit: string; tech: string; years: string;
  context: Ctx;
  truth: Record<string, number | null>;
  flex_true: string;
  python: { tech: Record<string, number>; estimates: Record<string, number>; flex: string };
}
export interface XMeta { t0: number; n: number; units: Unit[] }
export type Span = 'all' | 'year' | 'quarter' | 'month';
export interface Control { span: Span; start?: number; ctx: boolean; price: boolean }
export interface Estimate { value: number; lo68: number; hi68: number; lo95: number; hi95: number }
export interface Reading {
  tech: number[];           // after the EIA-860 prime-mover rule (if context is given)
  techOutputOnly: number[]; // the network's pooled probabilities
  est: Record<string, Estimate>;
  capacityMW: number;
  flex: string;
  windows: [number, number][];
  raw: { X: Float32Array[]; pc: number[]; mm: number[]; ss: number[] };
}

const LENGTH_H: Record<string, number> = { month: 30 * 24, quarter: 91 * 24 };

// ---------------------------------------------------------------- numpy helpers
/** numpy.percentile(a, qs) with the default 'linear' method, on one sorted copy. */
function quantiles(a: ArrayLike<number>, qs: number[]): number[] {
  const n = a.length;
  if (!n) return qs.map(() => NaN);
  const s = Float64Array.from(a).sort();
  return qs.map((qq) => {
    const q = qq / 100;
    const vi = n * q + (1 + q * (1 - 1 - 1)) - 1; // numpy _compute_virtual_index, alpha = beta = 1
    let lo = Math.floor(vi);
    let g = vi - lo;
    if (lo < 0) { lo = 0; g = 0; }
    if (lo > n - 1) { lo = n - 1; g = 0; }
    const hi = Math.min(lo + 1, n - 1);
    const A = s[lo], B = s[hi], d = B - A;
    return g >= 0.5 ? B - d * (1 - g) : A + d * g; // numpy _lerp
  });
}
const pct = (a: ArrayLike<number>, q: number) => quantiles(a, [q])[0];
function mean(a: ArrayLike<number>): number {
  if (!a.length) return NaN;
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
}
/** numpy.histogram counts (last bin closed on the right). */
function hist(a: ArrayLike<number>, edges: number[]): number[] {
  const h = new Array(edges.length - 1).fill(0);
  const last = edges.length - 1;
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (!(v >= edges[0] && v <= edges[last])) continue;
    let k = last - 1;
    for (let j = 0; j < last; j++) if (v < edges[j + 1]) { k = j; break; }
    h[k]++;
  }
  return h;
}
function runs(on: Uint8Array) {
  const s: number[] = [], L: number[] = [], v: boolean[] = [];
  for (let i = 0; i < on.length; i++) {
    if (i === 0 || on[i] !== on[i - 1]) { s.push(i); L.push(1); v.push(on[i] === 1); } else L[L.length - 1]++;
  }
  return { s, L, v };
}
const fin = (v: number, fill: number) => (Number.isFinite(v) ? v : fill);

// ---------------------------------------------------------------- features (pdcore.features)
export function features(M: PMeta, x: Float64Array, hour: Int32Array | null, dow: Int32Array | null,
  price: Float64Array | null, gas: Float64Array | null, ctx: Record<string, number> | null): number[] {
  const n = x.length;
  const miss = new Uint8Array(n), on = new Uint8Array(n), inter = new Uint8Array(n);
  const xv = new Float64Array(n);
  let nMiss = 0;
  for (let i = 0; i < n; i++) {
    const m = Number.isFinite(x[i]) ? 0 : 1;
    miss[i] = m; nMiss += m;
    xv[i] = m ? 0 : x[i];
    on[i] = xv[i] > 0.5 ? 1 : 0;
  }
  let nInter = 0, nOnKnown = 0;
  for (let i = 0; i < n; i++) {
    inter[i] = on[i] && i > 0 && on[i - 1] && i < n - 1 && on[i + 1] ? 1 : 0;
    nInter += inter[i];
    if (!miss[i] && on[i]) nOnKnown++;
  }
  const nKnown = n - nMiss;
  const f: number[] = [];
  const base: number[] = [];
  for (let i = 0; i < n; i++) if (nInter >= 12 ? inter[i] : on[i]) base.push(xv[i]);
  const pcap = Math.max(base.length ? pct(base, 99.5) : 1.0, 1.0);
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) y[i] = xv[i] / pcap;
  f.push(Math.log(n / 720.0), nMiss / n, nKnown ? nOnKnown / nKnown : 0.0, Math.log(pcap / 100.0), nInter < 12 ? 1 : 0);
  let yi: number[] = [];
  for (let i = 0; i < n; i++) if (inter[i]) yi.push(y[i]);
  if (!yi.length) yi = [0];
  const h = hist(yi.map((v) => Math.min(Math.max(v, 0), 1.0999)), M.loadbins);
  const hs = Math.max(h.reduce((a, b) => a + b, 0), 1);
  h.forEach((c) => f.push(c / hs));
  quantiles(yi, [0.5, 1, 2, 5, 10, 25, 50, 75, 90, 99]).forEach((v) => f.push(fin(v, 0)));
  const my = mean(yi);
  let ss = 0, near = 0;
  for (const v of yi) { ss += (v - my) ** 2; if (Math.abs(v - 1) < 0.03) near++; }
  f.push(yi.length > 1 ? Math.sqrt(ss / yi.length) : 0, near / yi.length);
  // first / second / last hour of each run
  const R = runs(on);
  const st: number[] = [];
  R.s.forEach((s, i) => { if (R.v[i] && s > 0) st.push(s); });
  const first = st.map((s) => y[s]);
  const second = st.filter((s) => s + 1 < n).map((s) => y[s + 1]);
  const last: number[] = [];
  R.s.forEach((s, i) => { if (R.v[i] && s + R.L[i] < n) last.push(y[s + R.L[i] - 1]); });
  for (const a of [first, second, last]) quantiles(a, [10, 50, 90]).forEach((v) => f.push(fin(v, -1)));
  // ramps between interior hours
  let dy: number[] = [];
  for (let i = 0; i < n - 1; i++) if (inter[i] && inter[i + 1]) dy.push(y[i + 1] - y[i]);
  if (!dy.length) dy = [0];
  quantiles(dy, [0.1, 1, 5, 25, 75, 95, 99, 99.9]).forEach((v) => f.push(fin(v, 0)));
  let small = 0, big = 0, sabs = 0;
  for (const d of dy) { const a = Math.abs(d); if (a < 0.01) small++; if (a > 0.1) big++; sabs += a; }
  f.push(small / dy.length, big / dy.length, Math.log1p((sabs / dy.length) * 100));
  // complete run and off-spell lengths
  const cm = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) cm[i + 1] = cm[i] + miss[i];
  const up: number[] = [], dn: number[] = [];
  R.s.forEach((s, i) => {
    if (i === 0 || i === R.s.length - 1 || cm[s + R.L[i]] - cm[s] !== 0) return;
    (R.v[i] ? up : dn).push(R.L[i]);
  });
  const nk = Math.max(nKnown, 1);
  for (const a of [up, dn]) {
    hist(a, M.run_edges).forEach((c) => f.push((c * 1000.0) / nk));
    f.push(a.length ? Math.log(pct(a, 50)) : -1.0, a.length >= 3 ? Math.log(pct(a, 5)) : -1.0, a.length >= 3 ? 1 : 0);
  }
  const nst = st.length;
  f.push(Math.log1p((nst * 8760.0) / nk), Math.log1p(nst));
  const prev: number[] = [];
  R.s.forEach((s, i) => { if (R.v[i] && s > 0 && i - 1 > 0) prev.push(R.L[i - 1]); });
  if (prev.length) {
    let a = 0, b = 0, c = 0;
    for (const p of prev) { if (p < M.hot_h) a++; if (p >= M.hot_h && p < M.warm_h) b++; if (p >= M.warm_h) c++; }
    f.push(a / prev.length, b / prev.length, c / prev.length, 1.0);
  } else f.push(0, 0, 0, 0);
  let sx = 0;
  for (let i = 0; i < n; i++) sx += xv[i];
  f.push(sx / (pcap * nk));
  // diurnal and weekly pattern
  if (hour && dow) {
    for (let b = 0; b < 8; b++) {
      let k = 0, c = 0, k2 = 0, c2 = 0;
      for (let i = 0; i < n; i++) if (!miss[i] && Math.floor(hour[i] / 3) === b) {
        k++; c += on[i];
        if (inter[i]) { k2++; c2 += y[i]; }
      }
      f.push(k ? c / k : 0.0, k2 ? c2 / k2 : 0.0);
    }
    let a1 = 0, b1 = 0, a2 = 0, b2 = 0;
    for (let i = 0; i < n; i++) if (!miss[i]) {
      if (dow[i] === 5 || dow[i] === 6) { b1++; a1 += on[i]; } else { b2++; a2 += on[i]; }
    }
    f.push(b1 ? a1 / b1 : 0.0, b2 ? a2 / b2 : 0.0);
  } else for (let i = 0; i < 18; i++) f.push(0);
  // price context: the market implied heat rate price / gas
  const ZB = M.zbins, nz = ZB.length - 1;
  if (price && gas) {
    const z = new Float64Array(n), zb = new Int32Array(n), ok = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      z[i] = price[i] / Math.max(gas[i], 0.5);
      ok[i] = !miss[i] && Number.isFinite(z[i]) ? 1 : 0;
      let k = 0;
      for (let j = 1; j < ZB.length - 1; j++) if (z[i] >= ZB[j]) k = j;
      zb[i] = k;
    }
    for (let b = 0; b < nz; b++) {
      let cnt = 0, con = 0, ci = 0, sy = 0;
      for (let i = 0; i < n; i++) if (ok[i] && zb[i] === b) { cnt++; con += on[i]; if (inter[i]) { ci++; sy += y[i]; } }
      f.push(cnt / n, cnt >= 5 ? con / cnt : -1.0, ci >= 5 ? sy / ci : -1.0);
    }
    const zo: number[] = [], zf: number[] = [];
    for (let i = 0; i < n; i++) if (ok[i]) (on[i] ? zo : zf).push(z[i]);
    const cl = (v: number) => Math.log(Math.min(Math.max(v, 1), 100));
    f.push(zo.length ? cl(pct(zo, 50)) : 0.0, zf.length ? cl(pct(zf, 50)) : 0.0, zo.length >= 10 ? cl(pct(zo, 10)) : 0.0, 1.0);
    const zz: number[] = [], yy: number[] = [];
    for (let i = 0; i < n; i++) if (ok[i] && inter[i]) { zz.push(z[i]); yy.push(y[i]); }
    let r = 0;
    if (zz.length > 30) {
      const mz = mean(zz), m2 = mean(yy);
      let vz = 0, vy = 0, cv = 0;
      for (let i = 0; i < zz.length; i++) { const a = zz[i] - mz, b = yy[i] - m2; vz += a * a; vy += b * b; cv += a * b; }
      const sz = Math.sqrt(vz / zz.length), sy = Math.sqrt(vy / yy.length);
      if (sy > 1e-6 && sz > 1e-6) r = cv / Math.sqrt(vz * vy);
    }
    f.push(r);
  } else for (let i = 0; i < 3 * nz + 5; i++) f.push(0);
  if (ctx) { for (const k of M.ctx_keys) f.push(fin(ctx[k] ?? 0, 0)); f.push(1.0); }
  else for (let i = 0; i <= M.ctx_keys.length; i++) f.push(0);
  return f.map((v) => fin(v, 0));
}

/** pdcore.context_vector: normalise the raw EIA-860/923 facts for the feature vector. */
function contextVector(c: Ctx & { hr923_window: number | null; ref_year: number }, pcap: number): Record<string, number> {
  const npo = c.np_own || 0.0, nps = c.np_st_share || 0.0, ml = c.minload, cold = c.cold_h, hr = c.hr923_window;
  return {
    pm_ct: c.pm_ct ?? 0, pm_st: c.pm_st ?? 0, pm_ca: c.pm_ca ?? 0, fuel_coal: c.fuel_coal ?? 0, chp: c.chp || 0.0,
    lnp_own: Math.log(Math.max(npo, 1.0) / 100.0), lnp_st: Math.log1p(nps / 100.0),
    r_own: npo > 0 ? pcap / npo : 0.0, r_blk: npo > 0 ? pcap / (npo + nps) : 0.0,
    minload_frac: ml && npo > 0 ? ml / (npo + nps) : 0.0, minload_has: ml ? 1 : 0,
    cold_lh: cold != null ? Math.log1p(cold) : 0.0, cold_has: cold != null ? 1 : 0,
    hr923: hr ? Math.log(hr / 10.0) : 0.0, hr923_has: hr ? 1 : 0,
    age: ((c.ref_year || 2020) - (c.online_year || 2000)) / 30.0,
  };
}

const yearOf = (t: number) => new Date(t * 1000).getUTCFullYear();
export const monthOf = (t: number) => { const d = new Date(t * 1000); return (d.getUTCFullYear() - 1970) * 12 + d.getUTCMonth(); };

/** plantdetective.window_vector: the 171 network inputs of one window (170 features + the net-output flag). */
export function windowVector(M: PMeta, x: Float64Array, t: Float64Array, price: Float64Array | null, gas: Float64Array | null,
  ctx: Ctx | null, utcOffset = -6): Float32Array {
  const n = x.length;
  const hour = new Int32Array(n), dow = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const tl = t[i] + utcOffset * 3600;
    hour[i] = Math.floor(tl / 3600) % 24;
    dow[i] = (Math.floor(tl / 86400) + 3) % 7;
  }
  let cv: Record<string, number> | null = null;
  if (ctx) {
    const yrs = [...new Set(Array.from(t, yearOf))].sort((a, b) => a - b);
    const hr = ctx.hr923 ?? {};
    const hh = yrs.map((y) => hr[String(y)]).filter((v) => v);
    const xo: number[] = [];
    for (let i = 0; i < n; i++) if (Number.isFinite(x[i]) && x[i] > 0) xo.push(x[i]);
    const pc = Math.max(xo.length > 10 ? pct(xo, 99.5) : 1.0, 1.0);
    cv = contextVector({ ...ctx, hr923_window: hh.length ? mean(hh) : null, ref_year: mean(yrs) }, pc);
  }
  const v = features(M, x, hour, dow, price, gas, cv);
  v.push(0); // net = False: CAMPD gross output
  return Float32Array.from(v);
}

/** Detective.windows: calendar years with >= 300 days; else ~quarters; else the whole series (>= 30 days). */
export function windows(t: Float64Array): [number, number][] {
  const n = t.length;
  const out: [number, number][] = [];
  const first = new Map<number, number>(), count = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const y = yearOf(t[i]);
    if (!first.has(y)) first.set(y, i);
    count.set(y, (count.get(y) ?? 0) + 1);
  }
  for (const y of [...first.keys()].sort((a, b) => a - b)) if (count.get(y)! >= 300 * 24) out.push([first.get(y)!, count.get(y)!]);
  if (out.length) return out;
  if (n >= 80 * 24) {
    const k = Math.floor(n / (91 * 24)) || 1, L = Math.floor(n / k);
    return Array.from({ length: k }, (_, j) => [j * L, L] as [number, number]);
  }
  return n >= 30 * 24 ? [[0, n]] : [];
}

// ---------------------------------------------------------------- network
export function memberForward(W: Pack<PMeta>, i: number, X: Float32Array) {
  const g = (k: string) => W.t[`m${i}_${k}`];
  const mu = g('mu').data, sd = g('sd').data;
  let h: Float64Array = new Float64Array(X.length);
  for (let j = 0; j < X.length; j++) h[j] = (X[j] - mu[j]) / sd[j];
  for (let l = 0; l < 4; l++) {
    const w = g(`l${l}w`), b = g(`l${l}b`).data;
    const [O, I] = w.shape, wd = w.data;
    const o = new Float64Array(O);
    for (let r = 0; r < O; r++) {
      let s = b[r];
      const off = r * I;
      for (let c = 0; c < I; c++) s += wd[off + c] * h[c];
      o[r] = l < 3 ? Math.max(s, 0) + 0.01 * Math.min(s, 0) : s; // leaky ReLU between layers
    }
    h = o;
  }
  const nr = W.meta.reg.length;
  const lg = Array.from(h.slice(0, 4)), mx = Math.max(...lg);
  const e = lg.map((v) => Math.exp(v - mx)), se = e.reduce((a, b) => a + b, 0);
  return {
    p: e.map((v) => v / se),
    m: Array.from(h.slice(4, 4 + nr)),
    s: Array.from(h.slice(4 + nr, 4 + 2 * nr), (v) => Math.exp(Math.min(Math.max(v, -6), 4))),
  };
}

/** ens_forward: mean probabilities and means; variance = mean member variance + spread of member means. */
export function ensemble(W: Pack<PMeta>, X: Float32Array) {
  const outs = Array.from({ length: W.meta.members }, (_, i) => memberForward(W, i, X));
  const k = outs.length, nr = W.meta.reg.length;
  const p = [0, 1, 2, 3].map((c) => outs.reduce((a, o) => a + o.p[c], 0) / k);
  const m: number[] = [], s: number[] = [];
  for (let j = 0; j < nr; j++) {
    const mj = outs.reduce((a, o) => a + o.m[j], 0) / k;
    const v = outs.reduce((a, o) => a + o.s[j] ** 2, 0) / k + outs.reduce((a, o) => a + (o.m[j] - mj) ** 2, 0) / k;
    m.push(mj); s.push(Math.sqrt(v));
  }
  return { p, m, s };
}

/** pool: technology by summed log-probabilities, targets by precision-weighted mean, sigma = mean window sigma. */
function pool(P: { p: number[]; m: number[]; s: number[] }[]) {
  const lp = [0, 1, 2, 3].map((c) => P.reduce((a, o) => a + Math.log(Math.min(Math.max(o.p[c], 1e-6), 1)), 0));
  const mx = Math.max(...lp), e = lp.map((v) => Math.exp(v - mx)), se = e.reduce((a, b) => a + b, 0);
  const nr = P[0].m.length, m: number[] = [], s: number[] = [];
  for (let j = 0; j < nr; j++) {
    let sw = 0, sm = 0, sg = 0;
    for (const o of P) { const w = 1 / o.s[j] ** 2; sw += w; sm += w * o.m[j]; sg += o.s[j]; }
    m.push(sm / sw); s.push(sg / P.length);
  }
  return { pc: e.map((v) => v / se), mm: m, ss: s };
}

const sg = (a: number) => 1 / (1 + Math.exp(-a));
export function natural(v: number[]): Record<string, number> {
  return {
    pmin_frac: sg(v[0]), ramp_up: Math.exp(v[1]), ramp_dn: Math.exp(v[2]), starts_yr: Math.max(Math.expm1(v[3]), 0.0),
    min_up: Math.exp(v[4]), med_up: Math.exp(v[5]), min_dn: Math.exp(v[6]), hot: sg(v[7]), warm: sg(v[8]), cold: sg(v[9]),
    hr_full: Math.exp(v[10]), hr_min: Math.exp(v[10] + v[11]), cf: sg(v[12]),
  };
}

/** Technology likelihood from the EIA-860 prime mover and fuel (null if unknown). */
function techRule(c: Ctx | null): number[] | null {
  if (!c) return null;
  if (c.pm_ca) return [0.01, 0.96, 0.02, 0.01];
  if (c.pm_ct) return [0.01, 0.15, 0.83, 0.01];
  if (c.pm_st) return c.fuel_coal ? [0.96, 0.01, 0.01, 0.02] : [0.03, 0.01, 0.01, 0.95];
  return null;
}

export function flexClass(starts: number, cf: number, medUp: number) {
  if (!Number.isFinite(starts) || !Number.isFinite(cf)) return 'unknown';
  if (cf < 0.15 && (!Number.isFinite(medUp) || medUp < 12)) return 'peaking';
  if (starts >= 50) return 'cycling';
  if (starts >= 12) return 'load-following';
  return 'baseload';
}

/** Detective.infer on one stretch of hourly output. */
export function infer(W: Pack<PMeta>, x: Float64Array, t: Float64Array, price: Float64Array | null, gas: Float64Array | null,
  ctx: Ctx | null): Reading {
  const M = W.meta;
  const wins = windows(t);
  if (!wins.length) throw new Error('need at least 30 days of hourly output');
  const X = wins.map(([s, L]) => windowVector(M, x.subarray(s, s + L), t.subarray(s, s + L),
    price ? price.subarray(s, s + L) : null, gas ? gas.subarray(s, s + L) : null, ctx));
  const { pc, mm, ss } = pool(X.map((v) => ensemble(W, v)));
  const rule = techRule(ctx);
  let tech = pc.slice();
  if (rule) { tech = tech.map((p, i) => p * rule[i]); const s = tech.reduce((a, b) => a + b, 0); tech = tech.map((p) => p / s); }
  const sf = M.reg.map((k, j) => ss[j] * (M.factors[k] ?? 1.0));
  const mid = natural(mm);
  const est: Record<string, Estimate> = {};
  M.reg.forEach((k, j) => {
    const at = (d: number) => { const v = mm.slice(); v[j] += d; return natural(v)[k]; };
    const a = at(-sf[j]), b = at(sf[j]), c = at(-2 * sf[j]), d = at(2 * sf[j]);
    est[k] = { value: mid[k], lo68: Math.min(a, b), hi68: Math.max(a, b), lo95: Math.min(c, d), hi95: Math.max(c, d) };
  });
  const xo: number[] = [];
  for (let i = 0; i < x.length; i++) if (Number.isFinite(x[i]) && x[i] > 0) xo.push(x[i]);
  return {
    tech, techOutputOnly: pc, est,
    capacityMW: xo.length > 10 ? pct(xo, 99.5) : NaN,
    flex: flexClass(mid.starts_yr, mid.cf, mid.med_up),
    windows: wins, raw: { X, pc, mm, ss },
  };
}

// ---------------------------------------------------------------- the controls (export_instrument.cut)
export interface Series { x: Float64Array; t: Float64Array; price: Float64Array; gas: Float64Array }

export function series(X: Pack<XMeta>, i: number): Series {
  const n = X.meta.n, t = new Float64Array(n);
  for (let k = 0; k < n; k++) t[k] = X.meta.t0 + 3600 * k;
  return { x: Float64Array.from(X.t[`x${i}`].data), t, price: Float64Array.from(X.t.price.data), gas: Float64Array.from(X.t.gas.data) };
}

/** The window control -> [start, hours]. month = 30 days and quarter = 91 days from the first hour of the chosen UTC
 * month (start = months since January 1970); year = the chosen calendar year (UTC); all = the whole record. */
export function cut(t: Float64Array, c: Control): [number, number] {
  const n = t.length;
  if (c.span === 'all') return [0, n];
  if (c.span === 'year') {
    let s = -1, L = 0;
    for (let i = 0; i < n; i++) if (yearOf(t[i]) === c.start) { if (s < 0) s = i; L++; }
    return [s, L];
  }
  let s = 0;
  while (s < n && monthOf(t[s]) !== c.start) s++;
  return [s, Math.min(LENGTH_H[c.span], n - s)];
}

export function read(W: Pack<PMeta>, S: Series, u: Unit, c: Control): { r: Reading; start: number; hours: number } {
  const [s, L] = cut(S.t, c);
  const e = s + L;
  const r = infer(W, S.x.subarray(s, e), S.t.subarray(s, e), c.price ? S.price.subarray(s, e) : null,
    c.price ? S.gas.subarray(s, e) : null, c.ctx ? u.context : null);
  return { r, start: s, hours: L };
}

/** The vector the exporter wrote for each reference case. */
export function parityVector(M: PMeta, r: Reading): number[] {
  const out: number[] = [];
  for (const v of r.raw.X) out.push(...v);
  out.push(...r.raw.pc, ...r.raw.mm, ...r.raw.ss, ...r.tech);
  for (const k of M.reg) { const e = r.est[k]; out.push(e.value, e.lo68, e.hi68, e.lo95, e.hi95); }
  out.push(Number.isFinite(r.capacityMW) ? r.capacityMW : -1, M.flex.indexOf(r.flex)); // NaN capacity -> -1, as exported
  return out;
}
