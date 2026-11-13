// HybridPilot in the browser: a line-for-line port of models/hybridpilot/hybridpilot.py — MLP.__call__/s_pon,
// the slider-driven entries of features(), RoadSim's battery model (components.Battery: ocv, r0, limits, current)
// and HybridPilotCore.decide(). Plain doubles throughout, like numpy's float64. No DOM.
import type { Pack } from '../../lib/nn/pack.ts';

export interface Batt { ocv: number[]; R0_25: number; Ea: number; v_min: number; v_max: number; P_dis_max: number; P_chg_max: number; E_nom: number; mass: number }
export interface Moment {
  k: number; v_kmh: number; u: number; soc: number; T_eng: number; T_bat: number; v_rc: number;
  eng_on: boolean; t_on: number; t_off: number; t_cat: number; moving: boolean; heat_high: boolean; P_aux: number;
  x0: number[]; prev60: number[]; D: number[]; T: number[]; d_rec: number; t_rec: number;
  j_rec: number; s_rec: number; pon_rec: number;
}
export interface Run { v: string; s: string; fuel_g: number; E_kWh: number; soc_end: number; pct_dp: number; extrap: boolean; starts_km: number; L100: number; dist_km: number }
export interface Trip {
  id: string; vehicle: string; short: string; arch: string; likely: string; N: number; dist_km: number; T_route: number;
  logged_fuel_g: number; soc0: number; mass_kg: number; Pe: number; E_kWh: number; soc_lo: number; soc_hi: number;
  ref: number; phev: boolean; ctrl: number[]; start_W: number; cat_warm_s: number; batt: Batt; runs: Run[]; moments: Moment[];
}
export interface Policy { Kp: number; min_on_s: number; min_off_s: number; s_scale: number; s_bias: number; preview_s: number; nav: boolean; onoff: string }
export interface HMeta {
  trips: Trip[]; strategies: string[]; temps: number[]; masses: number[]; previews: number[];
  policy: Policy; n_feat: number; bins: [number, number][]; note: string;
}
export interface WMeta { layers: number[]; params: number }

const T0K = 273.15;

// ---------------------------------------------------------------------------------------------- network
export interface Net { mu: Float64Array; sd: Float64Array; W: Float64Array[]; b: Float64Array[]; dims: number[] }

export function loadNet(P: Pack<WMeta>): Net {
  const f = (k: string) => Float64Array.from(P.t[k].data);
  const W: Float64Array[] = [], b: Float64Array[] = [], dims = [P.t.W0.shape[0]];
  for (let i = 0; P.t[`W${i}`]; i++) { W.push(f(`W${i}`)); b.push(f(`b${i}`)); dims.push(P.t[`W${i}`].shape[1]); }
  return { mu: f('mu'), sd: f('sd'), W, b, dims };
}

/** MLP.__call__: (x - mu)/sd -> [x @ W + b, tanh] x 2 -> x @ W + b. Returns [log(s + 0.05), logit P(on)]. */
export function forward(net: Net, x: ArrayLike<number>): [number, number] {
  let h = new Float64Array(net.dims[0]);
  for (let i = 0; i < h.length; i++) h[i] = (x[i] - net.mu[i]) / net.sd[i];
  for (let l = 0; l < net.W.length; l++) {
    const I = net.dims[l], O = net.dims[l + 1], W = net.W[l], b = net.b[l];
    const o = new Float64Array(O);
    for (let i = 0; i < I; i++) { const hi = h[i]; if (hi === 0) continue; const row = i * O; for (let j = 0; j < O; j++) o[j] += hi * W[row + j]; }
    for (let j = 0; j < O; j++) o[j] = l < net.W.length - 1 ? Math.tanh(o[j] + b[j]) : o[j] + b[j];
    h = o;
  }
  return [h[0], h[1]];
}

const clamp = (v: number, a: number, b: number) => Math.min(Math.max(v, a), b);
/** MLP.s_pon */
export const sOf = (y: [number, number]) => Math.exp(clamp(y[0], -5, 3)) - 0.05;
export const pOn = (y: [number, number]) => 1 / (1 + Math.exp(-clamp(y[1], -30, 30)));

/** HybridPilotCore.equivalence: the network's s, then SOC feedback s <- s + Kp (SOC_ref - SOC). */
export function equivalence(pol: Policy, trip: Trip, sNet: number, soc: number): number {
  let s = Math.max(sNet * pol.s_scale + pol.s_bias, 0);
  const e = trip.ref - soc;
  if (pol.Kp && (!trip.phev || e > -0.03)) s = Math.max(s + pol.Kp * e, 0);
  return s;
}

// ---------------------------------------------------------------------------------------------- battery (RoadSim)
export function ocv(b: Batt, soc: number): number {
  const n = b.ocv.length, f = soc * (n - 1);
  if (f <= 0) return b.ocv[0];
  if (f >= n - 1) return b.ocv[n - 1];
  const i = Math.floor(f), w = f - i;
  return b.ocv[i] + (b.ocv[i + 1] - b.ocv[i]) * w;
}
export function r0(b: Batt, soc: number, T: number): number {
  let f = Math.exp(b.Ea / 8.314 * (1.0 / (T + T0K) - 1.0 / 298.15));
  if (soc < 0.2) f *= 1.0 + 1.5 * (0.2 - soc);
  return b.R0_25 * f;
}
/** terminal power limits (W): [max discharge, max charge > 0] */
export function limits(b: Batt, soc: number, T: number, v_rc: number, lo: number, hi: number): [number, number] {
  const V = ocv(b, soc) - v_rc, R = r0(b, soc, T);
  const I_dis = Math.max((V - b.v_min) / R, 0);
  let p_dis = Math.min(V * I_dis - R * I_dis * I_dis, V * V / (4 * R) * 0.95, b.P_dis_max);
  const I_chg = Math.max((b.v_max - V) / R, 0);
  let p_chg = Math.min(V * I_chg + R * I_chg * I_chg, b.P_chg_max);
  if (T < 0) { p_chg *= Math.max(0.1, 1.0 + T / 20.0); p_dis *= Math.max(0.3, 1.0 + T / 40.0); }
  if (soc > hi) p_chg *= Math.max(0.0, 1.0 - (soc - hi) / 0.03);
  if (soc < lo) p_dis *= Math.max(0.0, 1.0 - (lo - soc) / 0.03);
  return [p_dis, p_chg];
}
/** chemical battery power OCV * I for terminal power P (the strategy's chem_power) */
export function chem(b: Batt, P: number, soc: number, T: number, v_rc: number): number {
  const V = ocv(b, soc) - v_rc, R = r0(b, soc, T);
  const disc = Math.max(V * V - 4.0 * R * P, 0.0);
  return ocv(b, soc) * ((V - Math.sqrt(disc)) / (2.0 * R));
}

// ---------------------------------------------------------------------------------------------- features
export interface Setting { soc: number; d: number; t: number; L: number }

/** features() for a recorded moment with SOC, demand level d, coolant level t and preview length L (s) changed.
 *  Everything else (vehicle, timers, speed history) is the recorded vector x0. */
export function features(meta: HMeta, trip: Trip, m: Moment, set: Setting) {
  const lo = trip.soc_lo, hi = trip.soc_hi, mid = 0.5 * (lo + hi), half = 0.5 * (hi - lo), Pe = trip.Pe;
  const { soc, L } = set;
  const T = m.T[set.t];
  const x = m.x0.slice();
  const [p_dis, p_chg] = limits(trip.batt, soc, m.T_bat, m.v_rc, lo, hi);
  const need_heat = m.heat_high && T < 55.0;
  const cat = T < 35.0 && m.t_cat < trip.cat_warm_s;
  x[12] = (soc - mid) / half; x[13] = (soc - lo) / (hi - lo); x[14] = (soc - trip.ref) * 10.0;
  x[18] = T / 100.0; x[21] = need_heat ? 1 : 0; x[22] = cat ? 1 : 0;
  x[23] = p_dis / Pe; x[24] = p_chg / Pe; x[29] = m.D[set.d] / Pe;
  meta.bins.forEach(([, b], i) => {
    for (let j = 0; j < 4; j++) x[37 + 4 * i + j] = L >= b ? m.prev60[4 * i + j] : 0;
    x[57 + i] = L >= b ? 1 : 0;
  });
  const E_av = Math.max(soc - trip.ref, 0.0) * trip.batt.E_nom / 3.6e6;
  x[65] = E_av / (trip.batt.mass / 1000.0);
  const forced = need_heat || (cat && m.moving) || soc < lo - 0.02;
  return { x, p_dis, p_chg, need_heat, cat, forced };
}

// ---------------------------------------------------------------------------------------------- decision
export interface Cands { PF: ArrayLike<number>; PB: ArrayLike<number>; OK: ArrayLike<number>; ON: ArrayLike<number> }

/** candidate controls RoadSim scored at moment mi for demand level d and coolant level t (battery limits not applied) */
export function cands(X: Pack<HMeta>, ti: number, mi: number, d: number, t: number): Cands {
  const pf = X.t[`t${ti}_pf`], [, D, T, n] = pf.shape;
  const o = ((mi * D + d) * T + t) * n;
  const sl = (k: string) => X.t[`t${ti}_${k}`].data.subarray(o, o + n);
  return { PF: sl('pf'), PB: sl('pb'), OK: sl('ok'), ON: sl('onc') };
}

export interface Decision {
  y: [number, number]; sNet: number; s: number; pon: number; j: number;
  J: number[]; ok: boolean[]; on: boolean[]; chemW: number[]; x: number[]; forced: boolean; need_heat: boolean; cat: boolean;
}

/** HybridPilotCore.decide over this second's candidates: minimise P_fuel + s * P_chem (+ start fuel from off). */
export function decide(net: Net, meta: HMeta, trip: Trip, m: Moment, c: Cands, set: Setting): Decision {
  const f = features(meta, trip, m, set);
  const pol = meta.policy;
  const y = forward(net, f.x);
  const sNet = sOf(y), pon = pOn(y);
  const s = equivalence(pol, trip, sNet, set.soc);
  const n = c.PF.length;
  const ok: boolean[] = [], on: boolean[] = [], chemW: number[] = [];
  let anyOkOn = false, anyOkOff = false;
  for (let i = 0; i < n; i++) {
    const pb = c.PB[i];
    ok.push(!!c.OK[i] && pb <= f.p_dis + 1e-6 && pb >= -f.p_chg - 1e-6);
    on.push(!!c.ON[i]);
    chemW.push(chem(trip.batt, pb, set.soc, m.T_bat, m.v_rc));
    if (ok[i] && on[i]) anyOkOn = true;
    if (ok[i] && !on[i]) anyOkOff = true;
  }
  let sel = ok.slice();
  if (f.forced && anyOkOn) sel = sel.map((v, i) => v && on[i]);
  if (m.eng_on && m.t_on < pol.min_on_s && anyOkOn) sel = sel.map((v, i) => v && on[i]);
  if (!m.eng_on && m.t_off < pol.min_off_s && anyOkOff) sel = sel.map((v, i) => v && !on[i]);
  if (!sel.some(Boolean)) sel = ok.slice();
  const J: number[] = [];
  let j = -1, best = Infinity;
  for (let i = 0; i < n; i++) {
    let Ji = c.PF[i] + s * chemW[i];
    if (!m.eng_on) Ji += on[i] ? trip.start_W : 0;
    J.push(Ji);
    if (sel[i] && Ji < best) { best = Ji; j = i; }
  }
  return { y, sNet, s, pon, j, J, ok, on, chemW, x: f.x, forced: f.forced, need_heat: f.need_heat, cat: f.cat };
}

/** s over the whole recorded trip (every traction second of the base run), with SOC feedback, run live. */
export function tripS(net: Net, meta: HMeta, trip: Trip, X: Pack<HMeta>, ti: number): { k: Int32Array; s: Float64Array; sNet: Float64Array } {
  const xs = X.t[`t${ti}_xs`], soc = X.t[`t${ti}_xsoc`].data, kk = X.t[`t${ti}_xk`].data;
  const [n, F] = xs.shape;
  const k = new Int32Array(n), s = new Float64Array(n), sNet = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const y = forward(net, xs.data.subarray(i * F, (i + 1) * F));
    k[i] = kk[i]; sNet[i] = sOf(y); s[i] = equivalence(meta.policy, trip, sNet[i], soc[i]);
  }
  return { k, s, sNet };
}
