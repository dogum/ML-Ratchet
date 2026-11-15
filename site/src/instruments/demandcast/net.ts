// DemandCast in the browser: a line-for-line port of models/demandcast/demandcast.py (three MLPs, SiLU, softplus
// sigma, dev widening, solar night mask) and of physics() / inputs() / interp() in
// models/demandcast/web/export_instrument.py (the building-stock physics stepped over the 38 h from issue to the end of
// day D, the network inputs it feeds, wind and solar physics tabulated over weather multipliers). No DOM.
import { erf, linear } from '../../lib/nn/ops.ts';
import type { Pack } from '../../lib/nn/pack.ts';

export const BAS = ['ERCO', 'CISO', 'PJM', 'MISO', 'SWPP', 'ISNE', 'NYIS', 'BPAT'];
export const WX = ['T', 'q', 'ghi', 'clr', 'btm', 'res', 'com', 'ind', 'occ', 'gr'] as const;
type Kind = 'd' | 'w' | 's';

export type Params = Record<string, number>;
export interface DMeta {
  sigma_scale: Record<Kind, number>;
  n_params: Record<Kind, number>;
  inputs: Record<Kind, number>;
  MW: number[]; // wind-speed multipliers the wind physics is tabulated at
  MS: number[]; // irradiance multipliers the solar physics is tabulated at
  params: Record<string, Params>; // dev-fitted building-stock parameters per grid
  floor: Record<Kind, number>;
}
export interface DDay {
  ba: string; day: string; tag: string;
  st: { T30: number; L1: number; L2: number; G: number }; // filter states at issue
  sd: number; // mean physics demand over the input week (feature scale)
  hasw: boolean; hass: boolean; capw: number; caps: number;
  soff?: number; // v1.1: solar reporting offset (h) the day's 168 h history shows at issue (read in Python)
  T_mean: number; T_min: number; T_max: number;
}

/** The instrument's controls (perturb() in the exporter). */
export interface Perturb {
  dT?: number; // K added to the 38 h after issue; specific humidity follows at constant relative humidity
  hum?: number; // x specific humidity
  sun?: number; // x irradiance (capped at max(1.05 x clear sky, actual)); rooftop PV and utility solar follow
  wind?: number; // x wind speed at every weather point
}

/** One day's inputs as the exporter stored them (views into the examples pack). */
export interface Day {
  meta: DDay;
  wx: Record<(typeof WX)[number], Float32Array>; // 38 h each
  xd: Float32Array; truth: { y: Float32Array; df: Float32Array; pers: Float32Array; prior: Float32Array };
  xw: Float32Array; wtab: Float32Array[]; ws: Float32Array; yw: Float32Array;
  xs: Float32Array; stab: Float32Array[]; ys: Float32Array; night: Float32Array;
}

const row = (p: Pack, name: string, i: number, n: number) => p.t[name].data.subarray(i * n, (i + 1) * n);

export function dayOf(X: Pack<DDay[]>, i: number): Day {
  const nw = X.t.wtab.shape[1], ns = X.t.stab.shape[1];
  const wx = {} as Day['wx'];
  WX.forEach((k, j) => { wx[k] = X.t.wx.data.subarray((i * WX.length + j) * 38, (i * WX.length + j + 1) * 38); });
  const tr = (j: number) => X.t.truth.data.subarray((i * 4 + j) * 24, (i * 4 + j + 1) * 24);
  return {
    meta: X.meta[i], wx, xd: row(X, 'xd', i, 329), truth: { y: tr(0), df: tr(1), pers: tr(2), prior: tr(3) },
    xw: row(X, 'xw', i, 164), wtab: Array.from({ length: nw }, (_, k) => X.t.wtab.data.subarray((i * nw + k) * 24, (i * nw + k + 1) * 24)),
    ws: row(X, 'ws', i, 24), yw: row(X, 'yw', i, 24),
    xs: row(X, 'xs', i, 156), stab: Array.from({ length: ns }, (_, k) => X.t.stab.data.subarray((i * ns + k) * 24, (i * ns + k + 1) * 24)),
    ys: row(X, 'ys', i, 24), night: row(X, 'night', i, 24),
  };
}

// ---- building-stock physics (training/buildings.py, stepped from the states at issue) ------------------------------
const Phi = (z: number) => 0.5 * (1 + erf(z / Math.SQRT2));
const phi = (z: number) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
const clip = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const rcA = (tau: number) => 1 - Math.exp(-1 / Math.max(tau, 0.05));

export interface Phys { base: number[]; light: number[]; heat: number[]; cool: number[]; btm: number[]; total: number[]; T: number[]; q: number[]; ghi: number[] }

export function physics(d: Day, p: Params, dT = 0, hum = 1, sun = 1): Phys {
  const w = d.wx, st = d.meta.st;
  const a30 = rcA(240), a1 = rcA(p.tau1), a2 = rcA(p.tau2), ag = rcA(2.0);
  let T30 = st.T30, L1 = st.L1, L2 = st.L2, G = st.G;
  const shInd = Math.max(1 - p.sh_res - p.sh_com, 0), eo = Math.max(1 - p.e_res - p.e_hp, 0);
  const o: Phys = { base: [], light: [], heat: [], cool: [], btm: [], total: [], T: [], q: [], ghi: [] };
  for (let t = 0; t < 38; t++) {
    const T = w.T[t] + dT;
    const q = w.q[t] * hum * Math.exp(0.065 * dT);
    const g = Math.min(w.ghi[t] * sun, Math.max(w.clr[t] * 1.05, w.ghi[t]));
    const bcf = Math.min(1, Math.max(0, w.btm[t] * sun));
    T30 += a30 * (T - T30);
    const season = 1 + (p.wh * (15 - T30)) / 10, P0 = p.P0 * w.gr[t];
    const res = w.res[t], com = w.com[t], ind = w.ind[t], occ = w.occ[t];
    const base = P0 * (p.sh_res * res + p.sh_com * com + shInd * ind) * season;
    const dark = 1 - Math.min(1, Math.max(0, g / 1000 / 0.25));
    const light = P0 * p.light * (0.5 * res + 0.5 * com) * dark;
    const Tsa = T + (p.a_sol * g) / 1000;
    L1 += a1 * (Tsa - L1); L2 += a2 * (Tsa - L2); G += ag * (g / 1000 - G);
    const Teff = p.f1 * L1 + (1 - p.f1) * L2 + p.g_int * (0.6 + 0.4 * occ) + p.g_win * G;
    const mh = p.mu_h - p.setback * (1 - occ), mc = p.mu_c + p.setup * (1 - occ), UA = p.ua * P0;
    const zh = (mh - Teff) / p.sd_h, zc = (Teff - mc) / p.sd_c;
    const heatTh = UA * ((mh - Teff) * Phi(zh) + p.sd_h * phi(zh));
    const coolTh = UA * ((Teff - mc) * Phi(zc) + p.sd_c * phi(zc) + p.lam * Math.max(q - p.q_in, 0) * Phi(zc));
    const heat = heatTh * (p.e_res + p.e_hp / clip(2.6 + 0.08 * T, 1, 4.5) + 0.03 * eo);
    const cool = (coolTh * p.ac) / clip(4.2 - 0.07 * (T - 25), 2, 5.5);
    const btm = p.btm * P0 * bcf;
    if (t >= 14) {
      o.base.push(base); o.light.push(light); o.heat.push(heat); o.cool.push(cool); o.btm.push(btm);
      o.total.push(base + light + heat + cool - btm); o.T.push(T); o.q.push(q); o.ghi.push(g);
    }
  }
  return o;
}

/** Linear interpolation between tabulated multipliers (clamped at the ends), as interp() in the exporter. */
export function interp(tab: Float32Array[], grid: number[], m: number): number[] {
  const step = grid[1] - grid[0];
  const k = Math.max(0, Math.min(grid.length - 2, Math.floor((m - grid[0]) / step)));
  const f = Math.max(0, Math.min(1, (m - grid[k]) / (grid[k + 1] - grid[k])));
  return Array.from(tab[k], (v, h) => v * (1 - f) + tab[k + 1][h] * f);
}

// ---- the three MLPs (demandcast._forward) ---------------------------------------------------------------------------
const softplus = (v: number) => (v > 0 ? v + Math.log1p(Math.exp(-v)) : Math.log1p(Math.exp(v)));

/** Raw outputs: mu (log demand ratio, or CF = network + physics prior) and sigma (softplus + floor, not yet widened). */
export function forward(W: Pack<DMeta>, kind: Kind, x: ArrayLike<number>, prior?: ArrayLike<number>) {
  const t = W.t, xm = t[`${kind}_xm`].data, xs = t[`${kind}_xs`].data;
  let z: ArrayLike<number> = Float64Array.from({ length: xm.length }, (_, i) => (x[i] - xm[i]) / xs[i]);
  for (let i = 0; t[`${kind}_body_${i}_weight`]; i += 2) {
    const o = linear(z, t[`${kind}_body_${i}_weight`], t[`${kind}_body_${i}_bias`]);
    for (let j = 0; j < o.length; j++) o[j] = o[j] / (1 + Math.exp(-o[j])); // SiLU
    z = o;
  }
  const mu = Array.from(linear(z, t[`${kind}_mu_weight`], t[`${kind}_mu_bias`]), (v, h) => (kind === 'd' ? v : v + prior![h]));
  const sig = Array.from(linear(z, t[`${kind}_ls_weight`], t[`${kind}_ls_bias`]), (v) => softplus(v) + W.meta.floor[kind]);
  return { mu, sig };
}

export interface Reading {
  phys: Phys;
  d: { mu: number[]; sig: number[]; mw: number[]; logSigma: number[]; lo95: number[]; hi95: number[] };
  w: { prior: number[]; mu: number[]; sig: number[]; cf: number[]; sigma: number[]; lo95: number[]; hi95: number[] } | null;
  s: { prior: number[]; mu: number[]; sig: number[]; cf: number[]; sigma: number[]; lo95: number[]; hi95: number[] } | null;
}

/** inputs() + DemandCast.predict_demand / predict_cf for one day and one set of control settings. */
export function predict(W: Pack<DMeta>, d: Day, p: Perturb = {}): Reading {
  const dT = p.dT ?? 0, hum = p.hum ?? 1, sun = p.sun ?? 1, wind = p.wind ?? 1;
  const M = W.meta, prm = M.params[d.meta.ba];
  const ph = physics(d, prm, dT, hum, sun);
  const s = d.meta.sd, xd = Float64Array.from(d.xd);
  (['base', 'light', 'heat', 'cool', 'btm'] as const).forEach((c, j) => { for (let h = 0; h < 24; h++) xd[100 + 24 * j + h] = ph[c][h] / s; });
  for (let h = 0; h < 24; h++) {
    xd[220 + h] = ph.total[h] / s - 1; xd[244 + h] = ph.T[h] / 10; xd[268 + h] = ph.q[h] / 10; xd[292 + h] = ph.ghi[h] / 1000;
  }
  const nd = forward(W, 'd', xd), kd = M.sigma_scale.d;
  const ls = nd.sig.map((v) => v * kd);
  const out: Reading = {
    phys: ph,
    d: { mu: nd.mu, sig: nd.sig, mw: nd.mu.map((v, h) => ph.total[h] * Math.exp(v)), logSigma: ls,
      lo95: nd.mu.map((v, h) => ph.total[h] * Math.exp(v - 1.96 * ls[h])), hi95: nd.mu.map((v, h) => ph.total[h] * Math.exp(v + 1.96 * ls[h])) },
    w: null, s: null,
  };
  if (d.meta.hasw) {
    const pt = interp(d.wtab, M.MW, wind), lo = interp(d.wtab, M.MW, 0.9 * wind), hi = interp(d.wtab, M.MW, 1.1 * wind);
    const xw = Float64Array.from(d.xw);
    for (let h = 0; h < 24; h++) { xw[h] = pt[h]; xw[24 + h] = hi[h] - lo[h]; xw[136 + h] = (d.ws[h] * wind) / 10; }
    const n = forward(W, 'w', xw, pt), k = M.sigma_scale.w;
    const cf = n.mu.map((v) => clip(v, 0, 1.2)), sg = n.sig.map((v) => v * k);
    out.w = { prior: pt, mu: n.mu, sig: n.sig, cf, sigma: sg, lo95: cf.map((v, h) => Math.max(v - 1.96 * sg[h], 0)), hi95: cf.map((v, h) => v + 1.96 * sg[h]) };
  }
  if (d.meta.hass) {
    const pt = interp(d.stab, M.MS, sun), lo = interp(d.stab, M.MS, 0.8 * sun), hi = interp(d.stab, M.MS, 1.2 * sun);
    const xs = Float64Array.from(d.xs);
    for (let h = 0; h < 24; h++) {
      xs[h] = pt[h]; xs[24 + h] = hi[h] - lo[h];
      const clr = d.wx.clr[14 + h] / 1000;
      xs[128 + h] = clr > 0.02 ? clip((d.wx.ghi[14 + h] * sun) / 1000 / Math.max(clr, 1e-3), 0, 1.2) : 0;
    }
    // v1.1 (demandcast.align_solar_x): put the last 24 h of reported solar back on the physics clock
    const r = d.meta.soff ?? 0;
    if (r !== 0) {
      const a = xs.slice(56, 80), kap = xs[55];
      for (let j = 0; j < 24; j++) {
        const src = j + r;
        xs[56 + j] = src >= 0 && src < 24 ? a[src] : kap * xs[80 + j];
      }
    }
    const n = forward(W, 's', xs, pt), k = M.sigma_scale.s;
    const night = (h: number) => d.night[h] > 0.5;
    // ... then move the forecast onto the reported clock: out[h] = in[h - r] (demandcast._shift)
    const shift = (v: number[], fill: number) => v.map((_, h) => (h - r >= 0 && h - r < 24 ? v[h - r] : fill));
    const cf = shift(n.mu.map((v, h) => (night(h) ? 0 : clip(v, 0, 1.2))), 0);
    const sg = shift(n.sig.map((v, h) => (night(h) ? M.floor.s : v * k)), M.floor.s);
    out.s = { prior: pt, mu: n.mu, sig: n.sig, cf, sigma: sg, lo95: cf.map((v, h) => Math.max(v - 1.96 * sg[h], 0)), hi95: cf.map((v, h) => v + 1.96 * sg[h]) };
  }
  return out;
}

/** The parity vector written by the exporter's outputs(). */
export function parityVector(r: Reading): number[] {
  const v = [...r.phys.total, ...r.d.mu, ...r.d.sig, ...r.d.mw, ...r.d.lo95, ...r.d.hi95];
  for (const o of [r.w, r.s]) if (o) v.push(...o.prior, ...o.mu, ...o.sig, ...o.cf, ...o.lo95, ...o.hi95);
  return v;
}
