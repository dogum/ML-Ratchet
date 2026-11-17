// LoadCast in the browser: a line-for-line port of models/loadcast/loadcast.py (weather_features, calendar_features,
// annual_climate, history_hourly, np_forward and LoadCast.forecast) as driven by frame() / predict() / cp_baseline() in
// models/loadcast/web/export_instrument.py. Two MLPs (cold: 64 inputs, warm: 83), 3 x 192 SiLU, Gaussian head.
// No DOM.
import type { Pack } from '../../lib/nn/pack.ts';

export const METERS = ['electricity', 'chilledwater', 'steam', 'hotwater', 'gas'] as const;
export const SCHED = ['unknown', 'standard', 'extended', '24x7'] as const;
export const VARIANTS = ['full', 'w8', 'w2'] as const; // history: local 2016, its last 8 weeks, its last 2 weeks
export type Mode = 'cold' | 'warm';
export type Variant = (typeof VARIANTS)[number];
export type Sched = (typeof SCHED)[number];
const KNOTS = [4.4, 12.8, 18.3, 26.7, 32.2];
const N = 9480; // stored hours per site: 720 h lead-in from 2 Dec 2016, then local 2017
const NT = 8760;

export interface WMeta {
  widen: Record<string, number>; n_params: Record<Mode, number>; z90: number; qmin: number;
}
export interface UMeta {
  id: string; bid: string; site: number; use: string; area: number; yb: number | null; meter: string; meters: string[];
  hvac: string; cz: string; cp: { kind: string; params: Record<string, number> };
}
export interface ExamRow { n: number; cold: number; warm_full: number; warm_w8: number; warm_w2: number; dumb: number; towt: number; cp: number; cov_warm: number; cov_cold: number }
export interface XMeta {
  lead: number;
  sites: { name: string; tz: string; lat: number; country: string; start_utc: string }[];
  units: UMeta[];
  exam: Record<string, ExamRow>;
  exam_n: number;
}

/** Features over a site's whole stored frame. dT (K) shifts the outdoor temperature (dewpoint capped at it);
 *  closedWk marks that week of 2017 as a holiday. */
export interface Frame { T: Float64Array; Tday: Float64Array; T24: Float64Array; wx: Float64Array; cal: Float64Array; tow: Int32Array; clim: number[] }

const rollPast = (x: Float64Array, n: number) => {
  const c = new Float64Array(x.length + 1), o = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) c[i + 1] = c[i] + x[i];
  for (let i = 1; i <= x.length; i++) { const lo = Math.max(i - n, 0); o[i - 1] = (c[i] - c[lo]) / (i - lo); }
  return o;
};

export function frame(X: Pack<XMeta>, s: number, dT = 0, closedWk: number | null = null): Frame {
  const lead = X.meta.lead, wxi = X.t.wx.data, cal = X.t.cal.data;
  const at = (q: number, i: number) => wxi[(s * 4 + q) * N + i], ca = (q: number, i: number) => cal[(s * 5 + q) * N + i];
  const day = X.t.day.data.subarray(s * N, (s + 1) * N), doy = X.t.doy.data.subarray(s * N, (s + 1) * N);
  const T = new Float64Array(N), dew = new Float64Array(N), ghi = new Float64Array(N), wind = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    T[i] = at(0, i) / 100 + dT; dew[i] = Math.min(at(1, i) / 100, T[i]); ghi[i] = at(2, i); wind[i] = at(3, i) / 100;
  }
  const T24 = rollPast(T, 24), T168 = rollPast(T, 168), g24 = rollPast(ghi, 24), Tew = new Float64Array(N);
  let e = T[0];
  for (let i = 0; i < N; i++) { e += (1 / 72) * (T[i] - e); Tew[i] = e; }
  const nd = day[N - 1] + 1, dmax = new Float64Array(nd).fill(-Infinity), dsum = new Float64Array(nd), dcnt = new Float64Array(nd);
  for (let i = 0; i < N; i++) { const d = day[i]; dmax[d] = Math.max(dmax[d], T[i]); dsum[d] += T[i]; dcnt[d]++; }
  const wx = new Float64Array(N * 10), Tday = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const o = i * 10, d = day[i];
    wx[o] = T[i] / 10; wx[o + 1] = T24[i] / 10; wx[o + 2] = Tew[i] / 10; wx[o + 3] = T168[i] / 10; wx[o + 4] = dmax[d] / 10;
    wx[o + 5] = dew[i] / 10; wx[o + 6] = Math.max(dew[i] - 12, 0) / 5; wx[o + 7] = ghi[i] / 500; wx[o + 8] = g24[i] / 200; wx[o + 9] = wind[i] / 5;
    Tday[i] = dsum[d] / dcnt[d];
  }
  const cf = new Float64Array(N * 16), tow = new Int32Array(N);
  const c0 = closedWk == null ? -1 : lead + 168 * closedWk, c1 = c0 + 168;
  for (let i = 0; i < N; i++) {
    const o = i * 16, h = ca(0, i), dw = ca(1, i), mo = ca(3, i), md = ca(4, i);
    cf[o] = Math.sin((2 * Math.PI * h) / 24); cf[o + 1] = Math.cos((2 * Math.PI * h) / 24);
    cf[o + 2] = Math.sin((4 * Math.PI * h) / 24); cf[o + 3] = Math.cos((4 * Math.PI * h) / 24);
    cf[o + 4 + dw] = 1;
    cf[o + 11] = i >= c0 && i < c1 ? 1 : ca(2, i);
    cf[o + 12] = (mo === 12 && md >= 23) || (mo === 1 && md <= 2) ? 1 : 0;
    cf[o + 13] = Math.sin((2 * Math.PI * doy[i]) / 365.25); cf[o + 14] = Math.cos((2 * Math.PI * doy[i]) / 365.25);
    cf[o + 15] = (mo === 6 && md >= 20) || mo === 7 || (mo === 8 && md <= 20) ? 1 : 0;
    tow[i] = dw * 24 + h;
  }
  // annual climate of the target year (annual_climate on local 2017)
  let sum = 0, hdd = 0, cdd = 0;
  for (let i = lead; i < N; i++) sum += T[i];
  const ndays = Math.floor((N - lead) / 24);
  for (let d = 0; d < ndays; d++) {
    let m = 0;
    for (let k = 0; k < 24; k++) m += T[lead + d * 24 + k];
    m /= 24; hdd += Math.max(18 - m, 0); cdd += Math.max(m - 18, 0);
  }
  const clim = [sum / (N - lead) / 10, (hdd * 365) / ndays / 3000, (cdd * 365) / ndays / 1000];
  return { T, Tday, T24, wx, cal: cf, tow, clim };
}

interface Net { mu: Float32Array; sd: Float32Array; L: { w: Float32Array; b: Float32Array; o: number; n: number }[]; hw: Float32Array; hb: Float32Array; nin: number }
const nets = new WeakMap<object, Record<Mode, Net>>();
function netOf(W: Pack<WMeta>, mode: Mode): Net {
  let r = nets.get(W);
  if (!r) {
    const mk = (m: Mode): Net => {
      const t = W.t, L: Net['L'] = [];
      for (let i = 0; i < 40; i++) {
        const w = t[`${m}_body_${i}_weight`];
        if (w) L.push({ w: w.data, b: t[`${m}_body_${i}_bias`].data, o: w.shape[0], n: w.shape[1] });
      }
      return { mu: t[`${m}_mu_x`].data, sd: t[`${m}_sd_x`].data, L, hw: t[`${m}_head_weight`].data, hb: t[`${m}_head_bias`].data, nin: t[`${m}_mu_x`].shape[0] };
    };
    r = { cold: mk('cold'), warm: mk('warm') };
    nets.set(W, r);
  }
  return r[mode];
}

export interface Forecast { kW: Float64Array; sd: Float64Array; k: number }

/** kW and sigma (kW, dev-widened unless calibrated = false) for target hours [a, b) of local 2017. */
export function predict(W: Pack<WMeta>, X: Pack<XMeta>, F: Frame, u: number, a: number, b: number,
  mode: Mode = 'cold', variant: Variant = 'full', sched: Sched = 'unknown', calibrated = true): Forecast {
  const lead = X.meta.lead, um = X.meta.units[u], net = netOf(W, mode), nin = net.nin;
  const st = Float64Array.from(X.t.static.data.subarray(u * 38, (u + 1) * 38));
  st[34] = F.clim[0]; st[35] = F.clim[1]; st[36] = F.clim[2];
  if (sched !== 'unknown') { for (let k = 30; k < 34; k++) st[k] = 0; st[30 + SCHED.indexOf(sched)] = 1; }
  const v = VARIANTS.indexOf(variant);
  const hS = X.t.H_S.data[u * 3 + v];
  const coef = X.t.H_coef.data.subarray((u * 3 + v) * 174, (u * 3 + v + 1) * 174);
  const hcp = X.t.H_cp.data.subarray((u * 3 + v) * 3, (u * 3 + v + 1) * 3);
  const prof = X.t.H_prof.data.subarray((u * 3 + v) * 168, (u * 3 + v + 1) * 168);
  const scal = X.t.H_scal.data.subarray((u * 3 + v) * 14, (u * 3 + v + 1) * 14);
  const Q = mode === 'warm' ? Math.max(hS, W.meta.qmin) : 1;
  const unit = X.t.unit.data[u];
  const k = calibrated ? W.meta.widen[`${mode}_${variant}_${um.meter}`] ?? 1 : 1;
  const kW = new Float64Array(b - a), sd = new Float64Array(b - a);
  const x = new Float64Array(nin), hin = new Float64Array(nin), bufs = net.L.map((l) => new Float64Array(l.o));
  for (let t = a; t < b; t++) {
    const i = lead + t;
    let j = 0;
    for (let q = 0; q < 38; q++) x[j++] = st[q];
    for (let q = 0; q < 10; q++) x[j++] = F.wx[i * 10 + q];
    for (let q = 0; q < 16; q++) x[j++] = F.cal[i * 16 + q];
    let base = 0;
    if (mode === 'warm') {
      const T = F.T[i], tw = F.tow[i], Td = F.Tday[i];
      let towt = coef[tw];
      const seg = [Math.min(T, KNOTS[0])];
      for (let q = 0; q < 4; q++) seg.push(Math.min(Math.max(T, KNOTS[q]), KNOTS[q + 1]) - KNOTS[q]);
      seg.push(Math.max(T - KNOTS[4], 0));
      for (let q = 0; q < 6; q++) towt += seg[q] * coef[168 + q];
      x[j++] = towt;
      x[j++] = hcp[0] + hcp[1] * Math.max(15 - Td, 0) + hcp[2] * Math.max(Td - 18, 0);
      x[j++] = prof[tw];
      for (let q = 0; q < 14; q++) x[j++] = scal[q];
      x[j++] = Math.max(F.T24[i] - scal[8] * 10, 0) / 10;
      x[j++] = Math.max(scal[7] * 10 - F.T24[i], 0) / 10;
      base = (towt * hS) / Q;
    }
    // np_forward
    for (let q = 0; q < nin; q++) hin[q] = (x[q] - net.mu[q]) / net.sd[q];
    let h: Float64Array = hin;
    net.L.forEach((l, li) => {
      const o = bufs[li];
      for (let r = 0; r < l.o; r++) {
        let s = l.b[r];
        const off = r * l.n;
        for (let c = 0; c < l.n; c++) s += l.w[off + c] * h[c];
        o[r] = s / (1 + Math.exp(-Math.min(60, Math.max(-60, s))));
      }
      h = o;
    });
    const n = h.length;
    let o0 = net.hb[0], o1 = net.hb[1];
    for (let c = 0; c < n; c++) { o0 += net.hw[c] * h[c]; o1 += net.hw[n + c] * h[c]; }
    const ls = Math.min(4, Math.max(-7, o1));
    kW[t - a] = Math.max(base + o0, 0) * Q * unit;
    sd[t - a] = Math.exp(ls) * Q * unit * k;
  }
  return { kW, sd, k };
}

/** BuildingData's change-point model fitted on 2016 daily means, times the 2016 time-of-week shape (kW), all of 2017. */
export function cpBaseline(X: Pack<XMeta>, F: Frame, u: number): Float64Array {
  const { kind, params: p } = X.meta.units[u].cp, lead = X.meta.lead;
  const shape = X.t.cp_shape.data.subarray(u * 168, (u + 1) * 168), out = new Float64Array(NT);
  for (let t = 0; t < NT; t++) {
    const Td = F.Tday[lead + t];
    let d: number;
    if (kind === '2P') d = p.b0 + p.b1 * Td;
    else if (kind === '3PH') d = p.base + p.slope * Math.max(p.Tb - Td, 0);
    else if (kind === '3PC') d = p.base + p.slope * Math.max(Td - p.Tb, 0);
    else if (kind === '4P') d = p.base + p.slope_lo * Math.min(Td - p.Tb, 0) + p.slope_hi * Math.max(Td - p.Tb, 0);
    else d = p.base + p.heat_slope * Math.max(p.Tb_heat - Td, 0) + p.cool_slope * Math.max(Td - p.Tb_cool, 0);
    out[t] = Math.max(d * shape[F.tow[lead + t]], 0);
  }
  return out;
}

/** The real 2017 meter (kW, NaN = missing). */
export const truth = (X: Pack<XMeta>, u: number) => X.t.y.data.subarray(u * NT, (u + 1) * NT);

/** ASHRAE Guideline 14 CV(RMSE) and NMBE (%, p = 0) and the share inside the 90 % band, over finite hours. */
export function scores(y: ArrayLike<number>, f: ArrayLike<number>, sd?: ArrayLike<number>, z90 = 1.6449) {
  let n = 0, sy = 0, se = 0, sb = 0, inb = 0;
  for (let i = 0; i < y.length; i++) {
    if (!Number.isFinite(y[i])) continue;
    n++; sy += y[i]; se += (y[i] - f[i]) ** 2; sb += y[i] - f[i];
    if (sd && y[i] >= Math.max(f[i] - z90 * sd[i], 0) && y[i] <= f[i] + z90 * sd[i]) inb++;
  }
  const ym = sy / n;
  return { cv: (100 * Math.sqrt(se / n)) / ym, nmbe: (100 * sb) / n / ym, cov: sd ? inb / n : NaN, mean: ym, n };
}
