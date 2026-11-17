// BuildingDetective in the browser. No DOM.
// - Features: a port of models/buildingdetective/features.py (feat_hourly, monthly_from_hourly, feat_monthly,
//   feat_annual), taken from the release's own JS port (training/detective.js, verified against Python to 2e-7).
//   Like Python, the feature vector is rounded to float32 before the network.
// - Network: Net.raw / decode in models/buildingdetective/buildingdetective.py (2 GELU layers, Gaussian heads with
//   the dev widening, three softmax heads with the dev-fitted temperatures).
// - perturb(): what each instrument control does to the input, mirroring perturb() in
//   models/buildingdetective/web/export_instrument.py.
import { erf } from '../../lib/nn/ops.ts';
import type { Pack } from '../../lib/nn/pack.ts';

export const CH = ['elec', 'fuel', 'dheat', 'dcool'] as const;
export type Channel = (typeof CH)[number];
export type Mode = 'hourly' | 'monthly' | 'annual';
const TB: number[] = []; for (let t = -21; t < 37; t += 3) TB.push(t);
const NB = TB.length - 1;
const TAUS = [0, 12, 24, 48, 96, 192];
export const HB = 18.33;
const N_C_H = 3 + 2 * NB + 168 + 48 + 6 + 12 + 10 + 8;
const N_C_M = 3 + 12 + 6;

export interface WMeta { n_params: Record<Mode, number>; uses: string[]; heat: string[]; cool: string[]; reg: string[] }
export interface Unit {
  id: string; kind: 'bdg2' | 'hue' | 'recs' | 'cbecs';
  truth: { use?: string; floor_area_m2?: number | null; year_built?: number | null; meters?: string[]; heat?: string; codes?: string; year_range?: [number, number] };
  python: { floor_area_m2: number; year_built: number; UA_W_K: number; Tb_heat_C: number };
  tz?: string; lat?: number; channels?: Channel[]; n?: number; start_local?: string;
  kwh?: Partial<Record<Channel, number>>; hdd?: number; cdd?: number;
}
export interface Time { hr: ArrayLike<number>; dow: ArrayLike<number>; day: ArrayLike<number>; month: ArrayLike<number> }
export type Meters = Partial<Record<Channel, Float64Array>>;

const isF = Number.isFinite;
function nanmean(a: ArrayLike<number>) { let s = 0, n = 0; for (let i = 0; i < a.length; i++) if (isF(a[i])) { s += a[i]; n++; } return n ? s / n : NaN; }
function nanstd(a: ArrayLike<number>) { const m = nanmean(a); let s = 0, n = 0; for (let i = 0; i < a.length; i++) if (isF(a[i])) { s += (a[i] - m) ** 2; n++; } return n ? Math.sqrt(s / n) : NaN; }
function percentile(arr: number[], p: number) { // numpy 'linear'
  const a = Float64Array.from(arr).sort(); const n = a.length; if (!n) return NaN;
  const r = (p / 100) * (n - 1); const lo = Math.floor(r), hi = Math.ceil(r); return a[lo] + (a[hi] - a[lo]) * (r - lo);
}
function corr(a: number[], b: number[]) {
  let n = 0, sa = 0, sb = 0;
  for (let i = 0; i < a.length; i++) if (isF(a[i]) && isF(b[i])) { n++; sa += a[i]; sb += b[i]; }
  if (n < 10) return 0;
  const ma = sa / n, mb = sb / n; let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < a.length; i++) if (isF(a[i]) && isF(b[i])) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
  const d = Math.sqrt(saa * sbb); return d > 0 ? sab / d : 0;
}
function fillBins(v: number[]): number[] { // np.interp over the finite bins, edge hold; all-NaN -> 0
  const x: number[] = [], y: number[] = []; v.forEach((e, i) => { if (isF(e)) { x.push(i); y.push(e); } });
  if (!x.length) return v.map(() => 0);
  return v.map((_, i) => {
    if (i <= x[0]) return y[0]; if (i >= x[x.length - 1]) return y[y.length - 1];
    let k = 0; while (x[k + 1] < i) k++; return y[k] + ((y[k + 1] - y[k]) * (i - x[k])) / (x[k + 1] - x[k]);
  });
}
function ewma(x: Float64Array, tau: number) { // scipy lfilter([a], [1, -(1-a)], x, zi=[x0 (1-a)])
  if (tau === 0) return Float64Array.from(x);
  const a = 1 - Math.exp(-1 / tau); const y = new Float64Array(x.length); let acc = x[0];
  for (let i = 0; i < x.length; i++) { acc = a * x[i] + (1 - a) * acc; y[i] = acc; }
  return y;
}
function lsq(cols: number[], X: number[][], E: number[]) { // least squares on columns of [1, xh, xc]; all-zero columns get 0
  const k = cols.filter((c) => X[c].some((v) => v !== 0));
  const m = k.length, A: number[][] = [], b: number[] = [];
  for (let i = 0; i < m; i++) {
    A.push([]); let s = 0; for (let t = 0; t < E.length; t++) s += X[k[i]][t] * E[t]; b.push(s);
    for (let j = 0; j < m; j++) { let q = 0; for (let t = 0; t < E.length; t++) q += X[k[i]][t] * X[k[j]][t]; A[i].push(q); }
  }
  for (let i = 0; i < m; i++) {
    let p = i; for (let r = i + 1; r < m; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
    [A[i], A[p]] = [A[p], A[i]]; [b[i], b[p]] = [b[p], b[i]];
    for (let r = i + 1; r < m; r++) { const f = A[r][i] / A[i][i]; for (let c = i; c < m; c++) A[r][c] -= f * A[i][c]; b[r] -= f * b[i]; }
  }
  const x = new Array(m).fill(0);
  for (let i = m - 1; i >= 0; i--) { let s = b[i]; for (let c = i + 1; c < m; c++) s -= A[i][c] * x[c]; x[i] = s / A[i][i]; }
  const coef = [0, 0, 0]; k.forEach((c, i) => { coef[c] = x[i]; }); return coef;
}
/** Change-point fits whose best balance point lies beyond every day (all days on the heating side, or all on the
 * cooling side). There the fit is degenerate: every such balance point fits equally well, and numpy and JS pick
 * different ones by rounding noise, so the reading can differ from the Python release. Reset by inputs(). */
export const diag = { degenerate: 0 };

/** features.prism: E = b + h max(Tbh - T, 0) + c max(T - Tbc, 0), h, c >= 0. Returns [b, h, c, Tbh, Tbc, r2]. */
export function prism(Td0: ArrayLike<number>, Ed0: ArrayLike<number>): number[] {
  const Td: number[] = [], Ed: number[] = [];
  for (let i = 0; i < Td0.length; i++) if (isF(Td0[i]) && isF(Ed0[i])) { Td.push(Td0[i]); Ed.push(Ed0[i]); }
  if (Td.length < 8 || nanstd(Ed) === 0) return [Ed.length ? nanmean(Ed) : 0, 0, 0, 15, 20, 0];
  const me = nanmean(Ed); let sst = 1e-12; for (const e of Ed) sst += (e - me) ** 2;
  let best: [number, number[], number, number] | null = null; const one = Td.map(() => 1);
  for (let i = 0; 4 + 1.5 * i < 23; i++) {
    const tbh = 4 + 1.5 * i; const xh = Td.map((t) => Math.max(tbh - t, 0));
    const c0 = Math.max(tbh, 10);
    for (let j = 0; c0 + 1.5 * j < 29; j++) {
      const tbc = c0 + 1.5 * j; const xc = Td.map((t) => Math.max(t - tbc, 0)); const X = [one, xh, xc];
      let coef = lsq([0, 1, 2], X, Ed);
      if (coef[1] < 0 || coef[2] < 0) {
        const cols = [0].concat([1, 2].filter((q) => coef[q] >= 0));
        coef = lsq(cols, X, Ed); coef = [coef[0], Math.max(coef[1], 0), Math.max(coef[2], 0)];
      }
      let sse = 0; for (let t = 0; t < Ed.length; t++) { const r = coef[0] + coef[1] * xh[t] + coef[2] * xc[t] - Ed[t]; sse += r * r; }
      if (best === null || sse < best[0]) best = [sse, coef, tbh, tbc];
    }
  }
  const [sse, coef, tbh, tbc] = best!;
  const tmax = Math.max(...Td), tmin = Math.min(...Td);
  if ((coef[1] > 0 && tbh >= tmax) || (coef[2] > 0 && tbc <= tmin)) diag.degenerate++;
  return [coef[0], coef[1], coef[2], coef[1] > 0 ? tbh : 15, coef[2] > 0 ? tbc : 20, 1 - sse / sst];
}

/** Local-day index per hour (days are monotone, so insertion order = numpy's sorted unique order). */
function days(tm: Time) {
  const n = tm.day.length, map = new Map<number, number>(), inv = new Int32Array(n);
  for (let i = 0; i < n; i++) { const d = tm.day[i]; let k = map.get(d); if (k === undefined) { k = map.size; map.set(d, k); } inv[i] = k; }
  return { inv, nd: map.size };
}

/** features.feat_hourly */
export function featHourly(meters: Meters, T: Float64Array, GHI: ArrayLike<number>, tm: Time, lat: number): Float32Array {
  const n = T.length, hr = tm.hr, dow = tm.dow;
  const { inv: dinv, nd } = days(tm);
  const cnt = new Float64Array(nd), Tds = new Float64Array(nd), wds = new Float64Array(nd);
  for (let i = 0; i < n; i++) { cnt[dinv[i]]++; Tds[dinv[i]] += isF(T[i]) ? T[i] : 0; wds[dinv[i]] += dow[i] < 5 ? 1 : 0; }
  const Td = Array.from(Tds, (s, d) => s / Math.max(cnt[d], 1));
  const ddwd = Array.from(wds, (s, d) => s / Math.max(cnt[d], 1) > 0.5);
  let hdd = 0, cdd = 0; for (const t of Td) { hdd += Math.max(HB - t, 0); cdd += Math.max(t - HB, 0); }
  hdd /= nd; cdd /= nd;
  const occ = new Array(NB).fill(0);
  for (const t0 of Td) {
    const t = Math.min(Math.max(t0, TB[0] + 1e-6), TB[NB] - 1e-6);
    let b = -1; for (let k = 0; k < TB.length; k++) if (t >= TB[k]) b = k;
    occ[Math.min(Math.max(b, 0), NB - 1)] += 1;
  }
  const out: number[] = [nanmean(T) / 10, nanstd(T) / 10, hdd / 10, cdd / 10, nanmean(GHI) / 200, lat / 90, nd / 365, Math.log10(Math.max(nd, 1)) / 3]
    .concat(occ.map((c) => c / nd));
  const mT = nanmean(T); const Tf = Float64Array.from(T, (v) => (isF(v) ? v : mT));
  const Tew = TAUS.map((tau) => { const e = ewma(Tf, tau); const s = new Float64Array(nd); for (let i = 0; i < n; i++) s[dinv[i]] += e[i]; return Array.from(s, (v, d) => v / Math.max(cnt[d], 1)); });
  const Tdev = new Float64Array(n); // centred 24-h mean, window [i-12, i+11]
  { const cs = new Float64Array(n + 1); for (let i = 0; i < n; i++) cs[i + 1] = cs[i] + Tf[i];
    for (let i = 0; i < n; i++) { const a = Math.max(0, i - 12), b = Math.min(n, i + 12); Tdev[i] = Tf[i] - (cs[b] - cs[a]) / (b - a); } }
  const Th = new Float64Array(n); for (let i = 0; i < n; i++) Th[i] = Td[dinv[i]];
  for (const ch of CH) {
    const x = meters[ch];
    let tot = 0, any = false; if (x) for (let i = 0; i < x.length; i++) if (isF(x[i])) { tot += Math.abs(x[i]); any = true; }
    if (!x || !any || tot <= 0) { for (let k = 0; k < N_C_H; k++) out.push(0); continue; }
    const mu = nanmean(x);
    if (!isF(mu) || mu <= 0) { for (let k = 0; k < N_C_H; k++) out.push(0); continue; }
    const v = Array.from(x, (e) => isF(e)); const xn = Float64Array.from(x, (e) => e / mu);
    let vm = 0; for (const b of v) vm += b ? 1 : 0; vm /= n;
    const f: number[] = [1, vm, Math.log10(mu)];
    const dv = new Float64Array(nd), es = new Float64Array(nd);
    for (let i = 0; i < n; i++) if (v[i]) { dv[dinv[i]]++; es[dinv[i]] += xn[i]; }
    const Ed = Array.from(es, (s, d) => (dv[d] < 12 ? NaN : s / Math.max(dv[d], 1)));
    for (const want of [true, false]) { // daily load in daily-temperature bins, weekdays then weekends
      const s = new Float64Array(NB), c = new Float64Array(NB);
      for (let d = 0; d < nd; d++) {
        if (ddwd[d] !== want || !isF(Ed[d])) continue;
        let b = -1; for (let k = 0; k < TB.length; k++) if (Td[d] >= TB[k]) b = k; // np.digitize - 1
        if (b >= 0 && b < NB) { s[b] += Ed[d]; c[b]++; }
      }
      for (const q of fillBins(Array.from(s, (q, k) => (c[k] > 0 ? q / Math.max(c[k], 1) : NaN)))) f.push(q);
    }
    const how = (i: number) => dow[i] * 24 + hr[i];
    const ps = new Float64Array(168), pc = new Float64Array(168);
    for (let i = 0; i < n; i++) if (v[i]) { ps[how(i)] += xn[i]; pc[how(i)]++; }
    const pw = fillBins(Array.from(ps, (q, k) => (pc[k] > 0 ? q / Math.max(pc[k], 1) : NaN)));
    for (const q of pw) f.push(q);
    for (const cold of [true, false]) { // weekday hourly profile on cold (< 5 C) and hot (> 22 C) days
      const s = new Float64Array(24), c = new Float64Array(24);
      for (let i = 0; i < n; i++) { const sel = (cold ? Th[i] < 5 : Th[i] > 22) && dow[i] < 5 && v[i]; if (sel) { s[hr[i]] += xn[i]; c[hr[i]]++; } }
      const p = Array.from(s, (q, k) => (c[k] > 0 ? q / Math.max(c[k], 1) : NaN));
      if (p.every(isF)) for (const q of p) f.push(q);
      else for (let hh = 0; hh < 24; hh++) { const a: number[] = []; for (let d = 0; d < 5; d++) a.push(pw[d * 24 + hh]); f.push(nanmean(a)); }
    }
    const pr = prism(Td, Ed);
    f.push(pr[0], pr[1] * 10, pr[2] * 10, pr[3] / 10, pr[4] / 10, pr[5]);
    for (const hot of [false, true]) { // daily load vs EWMA-smoothed temperature: the thermal-mass signal
      const idx: number[] = []; for (let d = 0; d < nd; d++) if (hot ? Td[d] > 20 : Td[d] < 10) idx.push(d);
      for (const tw of Tew) f.push(idx.length > 10 ? corr(idx.map((d) => Ed[d]), idx.map((d) => tw[d])) : 0);
    }
    const dev = Float64Array.from(xn, (q, i) => q - pw[how(i)]);
    for (const cold of [true, false]) {
      for (const lag of [0, 2, 4, 8, 16]) {
        const a: number[] = [], b: number[] = [];
        for (let i = lag; i < n; i++) { const sel = (cold ? Th[i] < 5 : Th[i] > 22) && v[i]; if (sel) { a.push(dev[i]); b.push(Tdev[i - lag]); } }
        f.push(a.length > 48 ? corr(a, b) : 0);
      }
    }
    const xv: number[] = []; for (let i = 0; i < n; i++) if (v[i]) xv.push(xn[i]);
    const nightA: number[] = [], dayA: number[] = [];
    for (let i = 0; i < n; i++) if (v[i] && dow[i] < 5) { if (hr[i] <= 4) nightA.push(xn[i]); if (hr[i] >= 10 && hr[i] <= 15) dayA.push(xn[i]); }
    const night = nightA.length ? nanmean(nightA) : NaN, day_ = dayA.length ? nanmean(dayA) : NaN;
    const Ewd: number[] = [], Ewe: number[] = []; for (let d = 0; d < nd; d++) if (isF(Ed[d])) (ddwd[d] ? Ewd : Ewe).push(Ed[d]);
    const wkd = Ewd.length ? nanmean(Ewd) : 1, wke = Ewe.length ? nanmean(Ewe) : wkd;
    let z = 0; for (const q of xv) if (q <= 1e-6) z++;
    let ad = 0, an = 0; if (xv.length > 2) for (let i = 1; i < n; i++) { const q = Math.abs(xn[i] - xn[i - 1]); if (isF(q)) { ad += q; an++; } }
    f.push(percentile(xv, 1), percentile(xv, 50), percentile(xv, 99) / 3,
      isF(night) && isF(day_) ? night / Math.max(day_, 1e-3) : 1, wke / Math.max(wkd, 1e-3), nanstd(Ed), z / xv.length,
      xv.length > 2 ? (an ? ad / an : 0) : 0);
    for (const q of f) out.push(Math.min(Math.max(isF(q) ? q : 0, -20), 20));
  }
  return Float32Array.from(out);
}

export interface Monthly { mm: Meters; Tm: number[]; Hm: number[]; Cm: number[]; nd: number[] }
/** features.monthly_from_hourly (no billing shift): monthly mean kW per meter, monthly mean T, HDD and CDD per day. */
export function monthlyFromHourly(meters: Meters, T: Float64Array, tm: Time): Monthly {
  const n = T.length; const mT = nanmean(T);
  const { inv: dinv, nd: nd0 } = days(tm);
  const cnt = new Float64Array(nd0), ts = new Float64Array(nd0), ms = new Float64Array(nd0);
  for (let i = 0; i < n; i++) { cnt[dinv[i]]++; ts[dinv[i]] += isF(T[i]) ? T[i] : mT; ms[dinv[i]] += tm.month[i] - 1; }
  const nd = new Array(12).fill(0), Tm = new Array(12).fill(0), Hm = new Array(12).fill(0), Cm = new Array(12).fill(0);
  for (let d = 0; d < nd0; d++) {
    const t = ts[d] / Math.max(cnt[d], 1); const m = Math.round(ms[d] / Math.max(cnt[d], 1));
    nd[m]++; Tm[m] += t; Hm[m] += Math.max(HB - t, 0); Cm[m] += Math.max(t - HB, 0);
  }
  for (let m = 0; m < 12; m++) { const q = Math.max(nd[m], 1); Tm[m] /= q; Hm[m] /= q; Cm[m] /= q; }
  const mm: Meters = {};
  for (const ch of CH) {
    const x = meters[ch]; if (!x) continue;
    const s = new Float64Array(12), c = new Float64Array(12);
    for (let i = 0; i < n; i++) if (isF(x[i])) { s[tm.month[i] - 1] += x[i]; c[tm.month[i] - 1]++; }
    mm[ch] = Float64Array.from(s, (q, m) => (c[m] > 24 * 5 ? q / Math.max(c[m], 1) : NaN));
  }
  return { mm, Tm, Hm, Cm, nd };
}

/** features.feat_monthly */
export function featMonthly(M: Monthly): Float32Array {
  const { mm, Tm, Hm, Cm, nd } = M; const ok = nd.map((q) => q > 0);
  const tok = Tm.filter((_, m) => ok[m]); let sn = 0, sh = 0, sc = 0;
  for (let m = 0; m < 12; m++) { sn += nd[m]; sh += Hm[m] * nd[m]; sc += Cm[m] * nd[m]; }
  const out = Tm.map((q, m) => (ok[m] ? q / 10 : 0)).concat(Hm.map((q, m) => (ok[m] ? q / 10 : 0)),
    [nanmean(tok) / 10, nanstd(tok) / 10, sh / Math.max(sn, 1) / 10, sc / Math.max(sn, 1) / 10]);
  for (const ch of CH) {
    const x = mm[ch]; let s = 0, any = false; if (x) for (const q of x) if (isF(q)) { s += q; any = true; }
    if (!x || !any || s <= 0) { for (let k = 0; k < N_C_M; k++) out.push(0); continue; }
    const mu = nanmean(x); if (mu <= 0) { for (let k = 0; k < N_C_M; k++) out.push(0); continue; }
    const xn = Array.from(x, (q) => q / mu); const f = [1, Array.from(x).filter(isF).length / 12, Math.log10(mu)];
    for (const q of xn) f.push(isF(q) ? q : 0);
    const pr = prism(Tm, xn); f.push(pr[0], pr[1] * 10, pr[2] * 10, pr[3] / 10, pr[4] / 10, pr[5]);
    for (const q of f) out.push(Math.min(Math.max(isF(q) ? q : 0, -20), 20));
  }
  return Float32Array.from(out);
}

/** features.feat_annual: annual kWh per meter, annual HDD and CDD (base 18.33 C). */
export function featAnnual(kwh: Partial<Record<Channel, number>>, hdd: number, cdd: number): Float32Array {
  const f = [hdd / 3000, cdd / 1500, Math.log10(hdd + 50), Math.log10(cdd + 50)];
  for (const ch of CH) { const e = kwh[ch]; if (e === undefined || !isF(e) || e <= 0) f.push(0, 0); else f.push(1, Math.log10(e) - 5); }
  return Float32Array.from(f);
}

/** Annual totals from hourly meters, as the exam made them (training/realfeats.py). */
export function annualFromHourly(meters: Meters, T: Float64Array, tm: Time) {
  const kwh: Partial<Record<Channel, number>> = {};
  for (const ch of CH) {
    const x = meters[ch]; if (!x) continue;
    let nv = 0; for (const q of x) if (isF(q)) nv++;
    kwh[ch] = nv / x.length > 0.2 ? nanmean(x) * 8760 : 0;
  }
  const { inv, nd } = days(tm); const mT = nanmean(T);
  const s = new Float64Array(nd), c = new Float64Array(nd);
  for (let i = 0; i < T.length; i++) { s[inv[i]] += isF(T[i]) ? T[i] : mT; c[inv[i]]++; }
  let hdd = 0, cdd = 0; for (let d = 0; d < nd; d++) { const t = s[d] / c[d]; hdd += Math.max(HB - t, 0); cdd += Math.max(t - HB, 0); }
  return { kwh, hdd, cdd };
}

// ---------------------------------------------------------------- controls
export interface Perturb {
  mode?: Mode; // which network reads it (survey units: annual only)
  months?: number; // keep the first k local months of meter data (1-12)
  elec?: boolean; // electricity meter only
  dT?: number; // K added to the outdoor temperature
  noise?: number; // multiplicative meter noise: x * exp(noise * z)
}
export interface Hourly { meters: Partial<Record<Channel, ArrayLike<number>>>; z: Partial<Record<Channel, ArrayLike<number>>>; T: ArrayLike<number>; G: ArrayLike<number>; tm: Time; lat: number }

/** The perturbed input series (hourly units). */
export function perturbSeries(u: Hourly, p: Perturb) {
  const T = Float64Array.from(u.T, (v) => v + (p.dT ?? 0));
  const m: Meters = {}; const k = p.months ?? 12;
  for (const ch of CH) {
    const x = u.meters[ch]; if (!x || (p.elec && ch !== 'elec')) continue;
    const z = u.z[ch]!, o = new Float64Array(x.length);
    for (let i = 0; i < x.length; i++) {
      let v = x[i];
      if (p.noise) v = v * Math.exp(p.noise * z[i]);
      if (k < 12 && u.tm.month[i] > k) v = NaN;
      o[i] = v;
    }
    m[ch] = o;
  }
  return { T, meters: m };
}

/** perturb() of the exporter: the feature vector the chosen network reads. */
export function inputs(u: Hourly | null, unit: Unit, p: Perturb): { mode: Mode; X: Float32Array; series?: { T: Float64Array; meters: Meters } } {
  diag.degenerate = 0;
  if (!u) {
    const kwh: Partial<Record<Channel, number>> = {};
    for (const [c, v] of Object.entries(unit.kwh ?? {})) if (!p.elec || c === 'elec') kwh[c as Channel] = v;
    return { mode: 'annual', X: featAnnual(kwh, unit.hdd!, unit.cdd!) };
  }
  const mode = p.mode ?? 'hourly';
  const s = perturbSeries(u, p);
  if (mode === 'hourly') return { mode, X: featHourly(s.meters, s.T, u.G, u.tm, u.lat), series: s };
  if (mode === 'monthly') return { mode, X: featMonthly(monthlyFromHourly(s.meters, s.T, u.tm)), series: s };
  const a = annualFromHourly(s.meters, s.T, u.tm);
  return { mode, X: featAnnual(a.kwh, a.hdd, a.cdd), series: s };
}

// ---------------------------------------------------------------- network
function softmax(v: number[], T: number) { const m = Math.max(...v); const e = v.map((q) => Math.exp((q - m) / T)); const s = e.reduce((a, b) => a + b, 0); return e.map((q) => q / s); }

/** Net.raw + the temperature softmax: [mu x12, sigma x12 (widened), p(use) x12, p(heat) x5, p(cool) x4]. */
export function forward(W: Pack<WMeta>, mode: Mode, X: ArrayLike<number>): Float64Array {
  const t = (k: string) => W.t[`${mode}_${k}`];
  const xm = t('xm').data, xs = t('xs').data;
  let h: Float64Array = Float64Array.from(X, (v, i) => (v - xm[i]) / xs[i]);
  for (const l of [0, 1]) {
    const w = t(`W${l}`), b = t(`b${l}`).data, [O, I] = w.shape, wd = w.data, o = new Float64Array(O);
    for (let j = 0; j < O; j++) { let s = b[j]; const base = j * I; for (let i = 0; i < I; i++) s += wd[base + i] * h[i]; o[j] = 0.5 * s * (1 + erf(s / Math.SQRT2)); }
    h = o;
  }
  const lin = (w: string, b: string) => {
    const ww = t(w), bb = t(b).data, [O, I] = ww.shape, out = new Array<number>(O);
    for (let j = 0; j < O; j++) { let s = bb[j]; for (let i = 0; i < I; i++) s += ww.data[j * I + i] * h[i]; out[j] = s; }
    return out;
  };
  const r = lin('Wr', 'br'), lg = lin('Wc', 'bc');
  const ym = t('ym').data, ys = t('ys').data, wd = t('widen').data, temp = t('temp').data, nr = ym.length;
  const nu = W.meta.uses.length, nh = W.meta.heat.length;
  const out = new Float64Array(2 * nr + lg.length);
  for (let j = 0; j < nr; j++) { out[j] = r[j] * ys[j] + ym[j]; out[nr + j] = Math.exp(Math.min(Math.max(r[nr + j], -6), 3)) * ys[j] * wd[j]; }
  const p = softmax(lg.slice(0, nu), temp[0]).concat(softmax(lg.slice(nu, nu + nh), temp[1]), softmax(lg.slice(nu + nh), temp[2]));
  out.set(p, 2 * nr);
  return out;
}

export interface Reading {
  v: Record<string, [number, number, number]>; // value, lo, hi at 1 sigma, physical units
  use: number[]; heat: number[]; cool: number[];
  raw: Float64Array;
}
const OUT: [string, string, 'log' | 'lin'][] = [['floor_area_m2', 'logA', 'log'], ['year_built', 'year', 'lin'], ['UA_W_K', 'logUA', 'log'],
  ['UA_W_m2K', 'logUAm2', 'log'], ['tau_h', 'logtau', 'log'], ['Tb_heat_C', 'Tb_heat', 'lin'], ['Tb_cool_C', 'Tb_cool', 'lin'],
  ['S_heat_W_K', 'logS_heat', 'log'], ['S_cool_W_K', 'logS_cool', 'log'], ['base_share', 'base_share', 'lin'],
  ['op_hours', 'op_hours', 'lin'], ['weekend_ratio', 'weekend', 'lin']];

/** decode(): physical units, as Detective.read_* returns them. */
export function read(W: Pack<WMeta>, mode: Mode, X: ArrayLike<number>): Reading {
  const o = forward(W, mode, X), n = OUT.length, v: Reading['v'] = {};
  OUT.forEach(([name, , kind], j) => {
    const m = o[j], s = o[n + j];
    v[name] = kind === 'log' ? [10 ** m, 10 ** (m - s), 10 ** (m + s)] : [m, m - s, m + s];
  });
  const nu = W.meta.uses.length, nh = W.meta.heat.length, b = 2 * n;
  return { v, use: Array.from(o.slice(b, b + nu)), heat: Array.from(o.slice(b + nu, b + nu + nh)), cool: Array.from(o.slice(b + nu + nh)), raw: o };
}
