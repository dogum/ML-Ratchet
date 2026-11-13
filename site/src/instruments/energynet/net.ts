// EnergyNet in the browser: a line-for-line port of models/energynet/{features.py, model_np.py, energynet.py}
// (route features, road-load integrals, the generic physics estimate, the masked vehicle vector, the 5-member ensemble)
// and of perturb() in models/energynet/web/export_instrument.py (what each control does to the inputs). No DOM.
import { linear } from '../../lib/nn/ops.ts';
import type { Pack } from '../../lib/nn/pack.ts';

export const ARCHS = ['conv', 'hev_parallel_p2', 'hev_series', 'hev_powersplit', 'phev_powersplit', 'phev_series', 'bev'];
export const CLASSES = ['car', 'suv', 'pickup', 'van', 'bus', 'truck_md', 'truck_hd'];
const TRANS = ['AT', 'DCT', 'CVT', 'AMT', 'single', 'eCVT'];
const HD = ['bus', 'truck_md', 'truck_hd'];
// class road-load priors: mass kg, CdA m2, Crr, Crr speed coef, rotating mass fraction
const CLS: Record<string, [number, number, number, number, number]> = {
  car: [1650, 0.62, 0.0085, 0.004, 0.025], suv: [2000, 0.85, 0.009, 0.004, 0.025], pickup: [2450, 1.1, 0.0095, 0.004, 0.03],
  van: [2600, 1.25, 0.009, 0.004, 0.03], bus: [15000, 6.0, 0.007, 0.002, 0.05], truck_md: [10000, 5.0, 0.0072, 0.002, 0.05],
  truck_hd: [30000, 5.9, 0.0062, 0.0015, 0.04],
};
const SURF = [1.0, 0.95, 1.12, 1.6, 2.2];
const SPD_E = [0.5, 4.17, 8.33, 13.89, 19.44, 25.0, 30.56];
const ACC_E = [-1.5, -0.5, -0.1, 0.1, 0.5, 1.5];
const GRD_E = [-0.06, -0.03, -0.01, 0.01, 0.03, 0.06];
const NR = 90, NF = 12, NV = 47;
const MI = 1609.344, LHV = 0.042;
export const COMPONENTS = ['aero', 'rolling', 'friction_brakes', 'aux_hvac', 'powertrain_losses'] as const;

export const FIELDS = ['v', 'grade', 'T_amb', 'wind_speed', 'wind_dir', 'heading', 'rho_air', 'surface'] as const;
export type Trip = Record<(typeof FIELDS)[number], ArrayLike<number>>;

export interface Vehicle {
  class: string; architecture: string;
  mass_kg?: number | null; mass_kind?: string | null;
  road_load_ABC?: { A_N: number; B_N_per_mps: number; C_N_per_mps2: number } | null;
  displacement_L?: number | null; max_power_kW?: number | null; fuel?: string | null;
  transmission?: string | null; n_gears?: number | null; battery_kWh?: number | null;
  label_city?: number | null; label_hwy?: number | null;
  label_city_kWh_per_100mi?: number | null; label_hwy_kWh_per_100mi?: number | null;
  soc_start?: number | null; thermal?: string | null; obd_maf?: boolean | null; dyno?: boolean | null;
}

export interface EMeta {
  members: string[]; mode: string; n_params: number;
  phys_k: { ice: number[]; hev: number[]; bev: number[] };
  sigma_factors: Record<string, number>;
}
export interface EExample {
  id: string; stratum: string; vehicle: Vehicle; measured_MJ: number;
  baselines_MJ: { label: number; moves_like: number; class_median: number };
  exam_prediction_MJ: number; n: number; km: number; T_mean: number; climb_m: number;
  python: { energy_MJ: number; value: number; unit: string; lo68: number; hi68: number; sigma_log: number; physics_estimate_MJ: number };
}

/** The instrument's controls (see perturb() in the exporter). */
export interface Perturb {
  dT?: number; // air temperature shift, C (air density follows at constant pressure)
  wind?: [number, number] | null; // [speed m/s, direction relative to heading: 0 head, 90 cross, 180 tail] replaces station wind
  grade?: number; // added to the grade (fraction)
  mass?: number; // multiplier on mass_kg
  soc?: number | null; // start charge 0..1
  thermal?: string | null; // 'warm' | 'cold'
}

const pmod = (a: number, n: number) => ((a % n) + n) % n;

export function perturb(trip: Trip, veh: Vehicle, p: Perturb): { trip: Trip; veh: Vehicle } {
  const t = {} as Record<(typeof FIELDS)[number], Float64Array>;
  for (const k of FIELDS) t[k] = Float64Array.from(trip[k]);
  const N = t.v.length;
  if (p.dT) {
    for (let i = 0; i < N; i++) {
      const T0 = t.T_amb[i], T1 = T0 + p.dT;
      t.rho_air[i] = (t.rho_air[i] * (T0 + 273.15)) / (T1 + 273.15);
      t.T_amb[i] = T1;
    }
  }
  if (p.wind != null) {
    const [ws, rel] = p.wind;
    for (let i = 0; i < N; i++) { t.wind_speed[i] = ws; t.wind_dir[i] = pmod(t.heading[i] + rel, 360); }
  }
  if (p.grade) for (let i = 0; i < N; i++) t.grade[i] += p.grade;
  const v: Vehicle = { ...veh };
  if (p.mass != null) v.mass_kg = (veh.mass_kg as number) * p.mass;
  if (p.soc != null) v.soc_start = p.soc;
  if (p.thermal != null) v.thermal = p.thermal;
  return { trip: t, veh: v };
}

// ---- features.py ----------------------------------------------------------------------------------
export interface Prep { v: Float64Array; a: Float64Array; grade: Float64Array; T: Float64Array; wh: Float64Array; wc: Float64Array; ws: Float64Array; rho: Float64Array; surf: Int32Array; N: number }

const dig = (x: number, e: number[]) => { let i = 0; while (i < e.length && x >= e[i]) i++; return i; }; // np.digitize, increasing bins
const sum = (a: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s; };
const mean = (a: ArrayLike<number>) => sum(a) / a.length;
const clip = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

/** prep(): trip of 1 Hz arrays (no gaps: the exported trips have none) -> the arrays every feature uses. */
export function prep(tr: Trip): Prep {
  const v = Float64Array.from(tr.v, (x) => (Number.isFinite(x) ? x : 0)), N = v.length;
  const a = new Float64Array(N);
  if (N > 1) { a[0] = v[1] - v[0]; a[N - 1] = v[N - 1] - v[N - 2]; for (let i = 1; i < N - 1; i++) a[i] = (v[i + 1] - v[i - 1]) / 2; }
  const arr = (k: (typeof FIELDS)[number], d: number) => (tr[k] ? Float64Array.from(tr[k]) : new Float64Array(N).fill(d));
  const grade = arr('grade', 0).map((g) => clip(g, -0.2, 0.2));
  const T = arr('T_amb', 20), ws = arr('wind_speed', 0), wd = arr('wind_dir', 0), hd = arr('heading', 0);
  let rho = arr('rho_air', 0);
  if (rho.every((r) => r <= 0)) rho = T.map((t) => 101325.0 / (287.05 * (t + 273.15))); // elevation 0
  const wh = new Float64Array(N), wc = new Float64Array(N);
  for (let i = 0; i < N; i++) { const r = ((wd[i] - hd[i]) * Math.PI) / 180; wh[i] = ws[i] * Math.cos(r); wc[i] = Math.abs(ws[i] * Math.sin(r)); }
  const surf = tr.surface ? Int32Array.from(tr.surface, (s) => clip(Math.trunc(Number.isFinite(s) ? s : 0), 0, 4)) : new Int32Array(N);
  return { v, a, grade, T, wh, wc, ws, rho, surf, N };
}

export function routeVec(P: Prep): Float64Array {
  const { v, a, grade, N } = P, R = new Float64Array(NR);
  const dist = Math.max(sum(v), 1), dur = N;
  let nm = 0, a2 = 0, stops = 0, pke = 0, v3 = 0, fast = 0;
  for (let i = 0; i < N; i++) { if (v[i] > 0.5) { nm++; a2 += a[i] * a[i]; } v3 += v[i] ** 3; if (v[i] > 25) fast++; }
  for (let i = 1; i < N; i++) { if (v[i] < 0.5 && v[i - 1] >= 0.5) stops++; pke += Math.max(0, v[i] * v[i] - v[i - 1] * v[i - 1]); }
  R[0] = Math.log(dur / 600); R[1] = Math.log(dist / 5000); R[2] = dist / dur / 20; R[3] = 1 - nm / N;
  R[4] = Math.min(stops / (dist / 1000), 10) / 3; R[5] = v[0] / 20; R[6] = v[N - 1] / 20;
  R[7] = nm ? Math.sqrt(a2 / nm) : 0; R[8] = pke / 2 / dist; R[9] = v3 / dist / 400;
  const H = new Float64Array(56);
  for (let i = 0; i < N; i++) H[dig(v[i], SPD_E) * 7 + dig(a[i], ACC_E)] += 1;
  for (let k = 0; k < 56; k++) R[10 + k] = H[k] / dur;
  const G = new Float64Array(7);
  let up = 0, dn = 0, net = 0;
  for (let i = 0; i < N; i++) { G[dig(grade[i], GRD_E)] += v[i]; const dh = v[i] * grade[i]; if (dh > 0) up += dh; else dn -= dh; net += dh; }
  for (let k = 0; k < 7; k++) R[66 + k] = G[k] / dist;
  const km = dist / 1000;
  R[73] = up / km / 20; R[74] = dn / km / 20; R[75] = net / km / 20;
  const T = P.T;
  let tmin = Infinity, tmax = -Infinity, c0 = 0, c25 = 0;
  for (let i = 0; i < N; i++) { const t = T[i]; tmin = Math.min(tmin, t); tmax = Math.max(tmax, t); if (t < 0) c0++; if (t > 25) c25++; }
  R[76] = mean(T) / 20; R[77] = tmin / 20; R[78] = tmax / 20; R[79] = c0 / N; R[80] = c25 / N;
  let hw = 0, cw = 0;
  for (let i = 0; i < N; i++) { hw += P.wh[i] * v[i]; cw += P.wc[i] * v[i]; }
  R[81] = hw / dist / 5; R[82] = cw / dist / 5; R[83] = mean(P.ws) / 5; R[84] = (mean(P.rho) - 1.2) * 5;
  const sc = [0, 0, 0, 0, 0];
  for (let i = 0; i < N; i++) sc[P.surf[i]]++;
  R[85] = sc[2] / N; R[86] = sc[4] / N; R[87] = sc[3] / N; R[88] = sc[1] / N; R[89] = fast / N;
  return R;
}

/** Per-second road-load powers (W): aero, rolling, grade, inertia — wheel_power(). */
export function wheelPower(P: Prep, m: number, CdA: number, Crr: number, cv: number, rot: number, hd: boolean) {
  const N = P.N, Kt = hd ? 0.01 : 0.008;
  const pa = new Float64Array(N), pr = new Float64Array(N), pg = new Float64Array(N), pi = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const v = P.v[i], fT = 1 / Math.max(0.6, 1 + Kt * (P.T[i] - 20)), vr = v + P.wh[i], th = Math.atan(P.grade[i]);
    pa[i] = 0.5 * P.rho[i] * CdA * vr * Math.abs(vr) * v;
    pr[i] = (v > 0.1 ? m * 9.81 * Math.cos(th) * Crr * (1 + cv * v) * fT * SURF[P.surf[i]] : 0) * v;
    pg[i] = m * 9.81 * Math.sin(th) * v;
    pi[i] = m * (1 + rot) * P.a[i] * v;
  }
  return { pa, pr, pg, pi };
}

export function physInts(P: Prep, m: number, CdA: number, Crr: number, cv: number, rot: number, hd: boolean): Float64Array {
  const { pa, pr, pg, pi } = wheelPower(P, m, CdA, Crr, cv, rot, hd);
  const o = new Float64Array(8);
  for (let i = 0; i < P.N; i++) {
    const p = pa[i] + pr[i] + pg[i] + pi[i];
    if (p > 0) o[0] += p; else o[1] += p;
    o[2] += pa[i]; o[3] += pr[i];
    if (pg[i] > 0) o[4] += pg[i]; else o[5] += pg[i];
    if (pi[i] > 0) o[6] += pi[i]; else o[7] += pi[i];
  }
  for (let k = 0; k < 8; k++) o[k] /= 1e6;
  return o;
}

/** Generic physics energy estimate (MJ) -> [E, E_hev_form, E_bev_form, frac_charge_depleting]. */
export function ePhys(K: EMeta['phys_k'], ph: Float64Array, arch: string, m: number, dur: number, soc: number | null, cap: number | null, hd: boolean): [number, number, number, number] {
  const Ep = ph[0], En = -ph[1], s = Math.pow(m / 1500, 0.7), ki = K.ice, kh = K.hev, kb = K.bev;
  let ei = Ep / ki[0] + (ki[1] * s * dur) / 1e6;
  let eh = Math.max((Ep - kh[1] * En) / kh[0], (0.2 * Ep) / kh[0]) + (kh[2] * s * dur) / 1e6;
  let eb = Math.max(Ep / kb[0] - kb[1] * En, 0.3 * Ep) + (kb[2] * s * dur) / 1e6;
  ei = Math.max(ei, 1e-3); eh = Math.max(eh, 1e-3); eb = Math.max(eb, 1e-3);
  if (arch === 'conv') return [ei, eh, eb, 0];
  if (arch === 'bev') return [eb, eh, eb, 1];
  if (arch.startsWith('phev')) {
    let f = 0.5;
    if (soc != null) { const u = cap ? 0.8 * cap : hd ? 100 : 8; f = clip((soc * u * 3.6) / eb, 0, 1); }
    return [f * eb + (1 - f) * eh, eh, eb, f];
  }
  return [eh, eh, eb, 0];
}

const mjkmMpg = (mpg: number | null | undefined, diesel: boolean) => (mpg ? ((diesel ? 832 : 745) * 3.785411784 * LHV) / ((mpg * MI) / 1000) : null);
const mjkmKwh = (k: number | null | undefined) => (k ? (k * 3.6) / (MI / 10) : null);

/** EnergyNet.features(): the 149 network inputs, the physics estimate E (MJ), distance (m), duration (s). */
export function features(K: EMeta['phys_k'], trip: Trip, veh: Vehicle) {
  const arch = veh.architecture, cls = veh.class, hd = HD.includes(cls);
  const [mc, Cc, Rc, cvc, rot] = CLS[cls];
  const P = prep(trip);
  const kind = veh.mass_kind ?? 'exact';
  const m0 = veh.mass_kg;
  const ml = !m0 ? 0 : kind === 'exact' ? 2 : 1;
  const m = m0 ? +m0 : mc;
  const rl = ml === 2 ? veh.road_load_ABC ?? null : null;
  let cda: number, crr: number, cv: number;
  if (rl) { cda = (2 * rl.C_N_per_mps2) / 1.1767; crr = rl.A_N / (m * 9.81); cv = clip(rl.B_N_per_mps / rl.A_N, -0.01, 0.03); }
  else { cda = Cc * Math.sqrt(m / mc); crr = Rc; cv = cvc; }
  const ph = physInts(P, m, cda, crr, cv, rot, hd);
  const soc = veh.soc_start ?? null, cap = veh.battery_kWh ?? null;
  const dur = P.N, dist = sum(P.v);
  const [E, Eh, Eb, f] = ePhys(K, ph, arch, m, dur, soc, cap, hd);
  const diesel = veh.fuel === 'diesel';
  let lf: [number | null, number | null] | null = null, le: [number | null, number | null] | null = null;
  if (arch === 'bev') { if (veh.label_city) le = [mjkmKwh(veh.label_city), mjkmKwh(veh.label_hwy)]; }
  else {
    if (veh.label_city) lf = [mjkmMpg(veh.label_city, diesel), mjkmMpg(veh.label_hwy, diesel)];
    if (veh.label_city_kWh_per_100mi) le = [mjkmKwh(veh.label_city_kWh_per_100mi), mjkmKwh(veh.label_hwy_kWh_per_100mi)];
  }
  // phys_vec
  const Fv = new Float64Array(NF), Ep = Math.max(ph[0], 1e-6);
  Fv[0] = Math.log(E); Fv[1] = Math.log(Ep / E); Fv[2] = Math.log((-ph[1] + 1e-4) / E);
  Fv[3] = ph[2] / Ep; Fv[4] = ph[3] / Ep; Fv[5] = ph[4] / Ep; Fv[6] = ph[5] / Ep; Fv[7] = ph[6] / Ep;
  Fv[8] = Math.log((Ep * 1e6) / (m * Math.max(dist, 1)) / 0.1); Fv[9] = Math.log(Eh / E); Fv[10] = Math.log(Eb / E); Fv[11] = f;
  // vehicle_vec
  const V = new Float64Array(NV);
  V[ARCHS.indexOf(arch)] = 1; V[7 + CLASSES.indexOf(cls)] = 1;
  V[14] = Math.log(m / 1500); V[15] = ml === 2 ? 1 : 0; V[16] = ml === 1 ? 1 : 0;
  if (rl) { V[17] = Math.log(cda / 0.7); V[18] = Math.log(crr / 0.009); V[19] = 1; }
  if (veh.displacement_L) { V[20] = Math.log(veh.displacement_L / 2); V[21] = 1; }
  if (veh.max_power_kW) { V[22] = Math.log(veh.max_power_kW / 100); V[23] = 1; }
  V[24] = diesel ? 1 : 0;
  if (veh.transmission && TRANS.includes(veh.transmission)) V[25 + TRANS.indexOf(veh.transmission)] = 1;
  if (veh.n_gears) { V[31] = veh.n_gears / 10; V[32] = 1; }
  if (cap) { V[33] = Math.log(cap / 10); V[34] = 1; }
  const vmean = dist / Math.max(dur, 1);
  const w = clip((48.3 - vmean / 0.44704) / (48.3 - 21.2), 0, 1);
  let i = 35;
  for (const lab of [lf, le]) {
    if (lab && lab[0] && lab[1]) { const mix = w * lab[0] + (1 - w) * lab[1]; V[i] = Math.log((mix * (dist / 1000)) / E); V[i + 1] = Math.log(lab[0] / lab[1]); V[i + 2] = 1; }
    i += 3;
  }
  if (soc != null) { V[41] = soc; V[42] = 1; }
  V[43] = veh.thermal === 'warm' ? 1 : 0; V[44] = veh.thermal === 'cold' ? 1 : 0; V[45] = veh.obd_maf ? 1 : 0; V[46] = veh.dyno ? 1 : 0;
  const x = new Float64Array(NR + NF + NV);
  x.set(routeVec(P), 0); x.set(Fv, NR); x.set(V, NR + NF);
  return { x, E, dist, dur, ph, P, m, cda, crr, cv, rot, hd };
}

// ---- model_np.py ----------------------------------------------------------------------------------
const geluTanh = (v: Float32Array) => {
  for (let i = 0; i < v.length; i++) { const x = v[i]; v[i] = 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))); }
  return v;
};

/** One member's raw outputs: log(E/E_phys), raw log sigma, 5 log components (ENet.forward before exp). */
export function member(W: Pack<EMeta>, j: number, x: ArrayLike<number>): Float32Array {
  const t = W.t, mu = t[`m${j}_x_mu`].data, sd = t[`m${j}_x_sd`].data;
  let h: ArrayLike<number> = Float64Array.from(x, (v, i) => clip((v - mu[i]) / sd[i], -8, 8));
  for (const k of [0, 2, 4, 6]) h = geluTanh(linear(h, t[`m${j}_body_${k}_weight`], t[`m${j}_body_${k}_bias`]));
  return linear(h, t[`m${j}_head_weight`], t[`m${j}_head_bias`]);
}

export const group = (arch: string) => (arch === 'conv' ? 'ICE' : arch === 'bev' ? 'EV' : arch.startsWith('phev') ? 'PHEV' : 'HEV');

export interface Prediction {
  energy_MJ: number; sigma_log: number; physics_estimate_MJ: number; value: number; unit: string; lo68: number; hi68: number;
  components_MJ: number[]; members: number[][]; x: Float64Array; distance_km: number; duration_s: number;
}

/** EnergyNet.predict(): ensemble in log space, sigma^2 = mean member sigma^2 + spread of the means, dev widening factor. */
export function predict(W: Pack<EMeta>, trip: Trip, veh: Vehicle): Prediction {
  const meta = W.meta, n = meta.members.length;
  const f = features(meta.phys_k, trip, veh);
  const lE = Math.log(f.E);
  const outs: number[][] = [];
  const L: number[] = [], S2: number[] = [], C = new Float64Array(5);
  for (let j = 0; j < n; j++) {
    const o = member(W, j, f.x);
    const ls = clip(o[1], -5, 2);
    outs.push([o[0], ls, o[2], o[3], o[4], o[5], o[6]]);
    const lm = meta.mode === 'corr' ? o[0] + lE : o[0];
    L.push(Math.log(Math.max(Math.exp(lm), 1e-9)));
    S2.push(Math.exp(2 * ls));
    for (let k = 0; k < 5; k++) C[k] += Math.log(Math.max(Math.exp(o[2 + k]) * f.E, 1e-9)) / n;
  }
  const Lm = mean(L);
  const varL = L.reduce((s, l) => s + (l - Lm) ** 2, 0) / n;
  const e = Math.exp(Lm), s = Math.sqrt(mean(S2) + varL);
  const arch = veh.architecture, cls = veh.class, K = meta.sigma_factors;
  let k: number;
  if (cls === 'bus') k = K[arch === 'bev' ? 'HD_ZTBus_leg' : 'HD_Mersin'] ?? 1;
  else if (HD.includes(cls)) k = K.HD_ICCT ?? 1;
  else k = K[`VED_${group(arch)}`] ?? 1;
  const S = s * k;
  const [unit, u] = arch === 'bev' ? ['kWh', 1 / 3.6] : arch.startsWith('phev') ? ['MJ', 1] : ['g fuel', 1 / LHV];
  return {
    energy_MJ: e, sigma_log: S, physics_estimate_MJ: f.E, value: e * u, unit, lo68: e * u * Math.exp(-S), hi68: e * u * Math.exp(S),
    components_MJ: Array.from(C, Math.exp), members: outs, x: f.x, distance_km: f.dist / 1000, duration_s: f.dur,
  };
}

/** The parity vector written by the exporter's outputs(). */
export function parityVector(r: Prediction): number[] {
  return [...r.x, ...r.members.flat(), r.energy_MJ, r.sigma_log, r.physics_estimate_MJ, r.value, r.lo68, r.hi68, ...r.components_MJ];
}
