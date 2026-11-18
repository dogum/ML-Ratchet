// SnowSense in the browser: the released GRU (models/snowsense/snowsense.py), its bands and annual readings, and
// WaterSim's uncalibrated degree-day and energy-balance snow routines (shared/water/watersim/watersim/snow.py) run on
// the same weather. Pure functions, no DOM; checked against Python by site/tests/parity.test.ts.
import type { Pack } from '../../lib/nn/pack.ts';

export interface SnowWeights {
  k: { daily: number; peak: number; melt: number };
  layers: number; hidden: number; window_days: number; warmup_days: number;
}
export interface StationInfo {
  id: string; name: string; state: string; region: string; lat: number; elev_m: number;
  start: string; days: number; wys: number[];
}
export interface Stations {
  stations: StationInfo[];
  storm_days: number; gap_days: number; gap_months: (number | null)[]; lapse: number;
  dd_default: Record<string, number>; eb_default: Record<string, number>;
  exam: { n: number; nse: number; eb: number; dd: number; clim: number; dd_cal: number };
}

/** A what-if on the INPUT weather. */
export interface Scenario {
  warm: number;      // deg C added to T_mean, T_min, T_max on every day
  pscale: number;    // precipitation multiplier
  dz: number;        // station moved up (+) or down (m)
  lapse: boolean;    // temperatures follow the move at 6.5 C/km (false: only the elevation input changes)
  drop: boolean;     // remove the wettest three days between 1 Oct and 30 Jun
  gap: number;       // 0 = none, else index into gap_months: a 30-day gauge outage starting on the 1st
}
export const RECORDED: Scenario = { warm: 0, pscale: 1, dz: 0, lapse: true, drop: false, gap: 0 };
export const isRecorded = (s: Scenario) =>
  s.warm === 0 && s.pscale === 1 && s.dz === 0 && !s.drop && s.gap === 0;

const L = 427, WARM = 61, DAY = 864e5;
const utc = (y: number, m: number, d = 1) => Date.UTC(y, m - 1, d);

export function wyDays(wy: number) { return Math.round((utc(wy, 10) - utc(wy - 1, 10)) / DAY); }
export function dayOfYear(ms: number) {
  const d = new Date(ms);
  return Math.floor((ms - Date.UTC(d.getUTCFullYear(), 0, 1)) / DAY) + 1;
}

export interface Window {
  wy: number; t0: number; nwy: number;                 // t0: UTC ms of 1 Aug (window day 0)
  P: Float64Array; Tm: Float64Array; Tn: Float64Array; Tx: Float64Array; mP: Float64Array; mT: Float64Array;
  elev: number; lat: number;
  storm: number;                                        // first day of the removed storm, or -1
  gap: [number, number] | null;                         // window days [a, b) of the injected outage
}

function series(pack: Pack<Stations>, i: number, key: string) { return pack.t[key + i].data; }

/** Observed end-of-day pillow SWE for one water year (NaN = no reading). */
export function observed(pack: Pack<Stations>, i: number, wy: number) {
  const st = pack.meta.stations[i];
  const i0 = Math.round((utc(wy - 1, 10) - Date.parse(st.start + 'T00:00:00Z')) / DAY);
  return Float64Array.from(series(pack, i, 'obs').subarray(i0, i0 + wyDays(wy)));
}

/** The wettest `n`-day run between 1 Oct and 30 Jun (window days 61 .. 333). */
export function stormStart(P: ArrayLike<number>, n: number) {
  let best = -1, at = WARM;
  for (let a = WARM; a <= WARM + 273 - n; a++) {
    let t = 0;
    for (let j = 0; j < n; j++) t += P[a + j];
    if (t > best + 1e-9) { best = t; at = a; }
  }
  return at;
}

export function gapStart(wy: number, month: number) {
  const y = month >= 8 ? wy - 1 : wy;
  return Math.round((utc(y, month) - utc(wy - 1, 8)) / DAY);
}

/** One water-year window of the station's weather with a what-if applied (mirrors web/export_instrument.py). */
export function makeWindow(pack: Pack<Stations>, i: number, wy: number, s: Scenario = RECORDED): Window {
  const m = pack.meta, st = m.stations[i];
  const t0 = utc(wy - 1, 8);
  const i0 = Math.round((t0 - Date.parse(st.start + 'T00:00:00Z')) / DAY);
  const cut = (k: string) => Float64Array.from(series(pack, i, k).subarray(i0, i0 + L));
  const P = cut('P'), Tm = cut('T_mean'), Tn = cut('T_min'), Tx = cut('T_max'), mP = cut('mP'), mT = cut('mT');
  let storm = -1, gap: [number, number] | null = null;
  if (s.drop) {
    storm = stormStart(P, m.storm_days);
    for (let j = 0; j < m.storm_days; j++) P[storm + j] = 0;
  }
  if (s.gap) {
    const a = gapStart(wy, m.gap_months[s.gap]!), G = m.gap_days, b = a + G;
    gap = [a, b];
    for (let d = a; d < b; d++) { P[d] = 0; mP[d] = 1; mT[d] = 1; }
    for (const v of [Tm, Tn, Tx]) {
      const lo = v[a - 1], hi = v[b];
      for (let j = 1; j <= G; j++) v[a + j - 1] = lo + ((hi - lo) * j) / (G + 1);
    }
    for (let d = a; d < b; d++) { Tn[d] = Math.min(Tn[d], Tm[d]); Tx[d] = Math.max(Tx[d], Tm[d]); }
  }
  const dT = s.warm + (s.lapse ? (m.lapse * s.dz) / 1000 : 0);
  for (let d = 0; d < L; d++) { P[d] *= s.pscale; Tm[d] += dT; Tn[d] += dT; Tx[d] += dT; }
  return { wy, t0, nwy: wyDays(wy), P, Tm, Tn, Tx, mP, mT, elev: st.elev_m + s.dz, lat: st.lat, storm, gap };
}

// ---- the network ---------------------------------------------------------------------------------------
const f32 = Math.fround;
const sig = (x: number) => 1 / (1 + Math.exp(-x));
const softplus = (x: number) => Math.log1p(Math.exp(-Math.abs(x))) + Math.max(x, 0);

/** Input channels exactly as snowsense.features (float32). */
export function features(w: Window): Float64Array {
  const X = new Float64Array(L * 10);
  for (let t = 0; t < L; t++) {
    const doy = dayOfYear(w.t0 + t * DAY);
    const ang = (2 * Math.PI * (doy - 1)) / 365.25;
    const o = t * 10;
    X[o] = f32(Math.log1p(Math.max(w.P[t], 0)) / 2);
    X[o + 1] = f32(w.Tm[t] / 10); X[o + 2] = f32(w.Tn[t] / 10); X[o + 3] = f32(w.Tx[t] / 10);
    X[o + 4] = f32(Math.sin(ang)); X[o + 5] = f32(Math.cos(ang));
    X[o + 6] = w.mP[t]; X[o + 7] = w.mT[t];
    X[o + 8] = f32((w.elev - 2200) / 800); X[o + 9] = f32((w.lat - 43) / 8);
  }
  return X;
}

/** GRU forward: (L, 10) inputs -> (L, 5) quantiles of z = sqrt(SWE)/10 [q05, q16, q50, q84, q95]. */
export function forward(W: Pack<SnowWeights>, X: Float64Array): Float64Array {
  const t = W.t, H = W.meta.hidden, T = X.length / 10;
  const Wi0 = t['inp.weight'].data, b0 = t['inp.bias'].data;
  let h = new Float64Array(T * H);
  for (let s = 0; s < T; s++) for (let i = 0; i < H; i++) {
    let a = b0[i];
    for (let j = 0; j < 10; j++) a += Wi0[i * 10 + j] * X[s * 10 + j];
    h[s * H + i] = Math.tanh(a);
  }
  const gi = new Float64Array(3 * H), gh = new Float64Array(3 * H), hs = new Float64Array(H);
  for (let l = 0; l < W.meta.layers; l++) {
    const Wi = t[`gru.weight_ih_l${l}`].data, Wh = t[`gru.weight_hh_l${l}`].data;
    const bi = t[`gru.bias_ih_l${l}`].data, bh = t[`gru.bias_hh_l${l}`].data;
    const out = new Float64Array(T * H);
    hs.fill(0);
    for (let s = 0; s < T; s++) {
      const xo = s * H;
      for (let r = 0; r < 3 * H; r++) {
        let a = bi[r], c = bh[r];
        const ro = r * H;
        for (let j = 0; j < H; j++) { a += Wi[ro + j] * h[xo + j]; c += Wh[ro + j] * hs[j]; }
        gi[r] = a; gh[r] = c;
      }
      for (let j = 0; j < H; j++) {
        const r = sig(gi[j] + gh[j]);
        const z = sig(gi[H + j] + gh[H + j]);
        const n = Math.tanh(gi[2 * H + j] + r * gh[2 * H + j]);
        hs[j] = (1 - z) * n + z * hs[j];
        out[xo + j] = hs[j];
      }
    }
    h = out;
  }
  const Wo = t['head.weight'].data, bo = t['head.bias'].data;
  const Q = new Float64Array(T * 5), o = new Float64Array(5);
  for (let s = 0; s < T; s++) {
    for (let k = 0; k < 5; k++) {
      let a = bo[k];
      for (let j = 0; j < H; j++) a += Wo[k * H + j] * h[s * H + j];
      o[k] = a;
    }
    const d0 = softplus(o[1]) + 1e-3, d1 = softplus(o[2]) + 1e-3, d2 = softplus(o[3]) + 1e-3, d3 = softplus(o[4]) + 1e-3;
    const q = s * 5;
    Q[q] = o[0] - d0 - d1; Q[q + 1] = o[0] - d0; Q[q + 2] = o[0]; Q[q + 3] = o[0] + d2; Q[q + 4] = o[0] + d2 + d3;
  }
  return Q;
}

/** First day after the peak below 5 mm (waterdata.metrics.melt_out), or -1. */
export function meltOut(s: ArrayLike<number>, thresh = 5) {
  let im = 0;
  for (let i = 1; i < s.length; i++) if (s[i] > s[im]) im = i;
  for (let i = im; i < s.length; i++) if (s[i] < thresh) return i;
  return -1;
}
function argmax(s: ArrayLike<number>) { let im = 0; for (let i = 1; i < s.length; i++) if (s[i] > s[im]) im = i; return im; }
function maxOf(s: ArrayLike<number>) { let m = -Infinity; for (let i = 0; i < s.length; i++) if (s[i] > m) m = s[i]; return m; }

export interface Reading {
  swe: Float64Array; lo68: Float64Array; hi68: Float64Array; lo90: Float64Array; hi90: Float64Array;
  peak: number; peakDay: number; melt: number;                   // day indices from 1 Oct; melt -1 = none
  peakLo68: number; peakHi68: number; peakLo90: number; peakHi90: number;
  meltLo68: number; meltHi68: number; meltLo90: number; meltHi90: number;
}

/** SnowSense's reading of one window, post-processed exactly as SnowSense.predict (water-year days only). */
export function read(W: Pack<SnowWeights>, w: Window): Reading {
  const Q = forward(W, features(w)), k = W.meta.k, n = w.nwy;
  const band = (j: number, kk = 1) => {
    const out = new Float64Array(n);
    for (let d = 0; d < n; d++) {
      const q = (WARM + d) * 5, m = Q[q + 2];
      out[d] = 100 * Math.max(m + kk * (Q[q + j] - m), 0) ** 2;
    }
    return out;
  };
  const swe = band(2);
  return {
    swe, lo68: band(1, k.daily), hi68: band(3, k.daily), lo90: band(0, k.daily), hi90: band(4, k.daily),
    peak: maxOf(swe), peakDay: argmax(swe), melt: meltOut(swe),
    peakLo68: maxOf(band(1, k.peak)), peakHi68: maxOf(band(3, k.peak)),
    peakLo90: maxOf(band(0, k.peak)), peakHi90: maxOf(band(4, k.peak)),
    meltLo68: meltOut(band(1, k.melt)), meltHi68: meltOut(band(3, k.melt)),
    meltLo90: meltOut(band(0, k.melt)), meltHi90: meltOut(band(4, k.melt)),
  };
}

// ---- WaterSim snow physics (snow-only, one band at the station, Hargreaves radiation, from 1 Aug with no snow) --
const LAMBDA_F = 0.334, LAMBDA_V = 2.501, LAMBDA_S = LAMBDA_F + LAMBDA_V, C_ICE = 2.102e-3, C_WATER = 4.186e-3;
const CP_AIR = 1.013e-3, SIGMA = 4.903e-9, VK = 0.41, GSC = 0.082;

function snowFraction(T: number, TT: number, TTI: number) {
  if (TTI <= 0) return T <= TT ? 1 : 0;
  const f = (TT + 0.5 * TTI - T) / TTI;
  return f < 0 ? 0 : f > 1 ? 1 : f;
}
const esatKpa = (T: number) => 0.6108 * Math.exp((17.27 * T) / (T + 237.3));
const esatIce = (T: number) => 0.61115 * Math.exp((22.452 * T) / (T + 272.55));

function extraterrestrial(lat: number, doy: number) {
  const phi = lat * (Math.PI / 180);
  const dr = 1 + 0.033 * Math.cos((2 * Math.PI * doy) / 365);
  const delta = 0.409 * Math.sin((2 * Math.PI * doy) / 365 - 1.39);
  const x = Math.max(-1, Math.min(1, -Math.tan(phi) * Math.tan(delta)));
  const ws = Math.acos(x);
  const ra = (((24 * 60) / Math.PI) * GSC * dr) * (ws * Math.sin(phi) * Math.sin(delta) + Math.cos(phi) * Math.cos(delta) * Math.sin(ws));
  return Math.max(ra, 0);
}

/** Uncalibrated WaterSim degree-day (HBV-light) SWE for the water-year days. */
export function degreeDay(w: Window, p: Record<string, number>) {
  const { TT, TTI, CFMAX, SFCF, CFR, CWH } = p;
  let ice = 0, liq = 0;
  const out = new Float64Array(w.nwy);
  for (let t = 0; t < WARM + w.nwy; t++) {
    const P = w.P[t], T = w.Tm[t];
    const fs = snowFraction(T, TT, TTI);
    const snow = P * fs * SFCF, rain = P * (1 - fs);
    ice += snow;
    if (T > TT) {
      let melt = CFMAX * (T - TT);
      if (melt > ice) melt = ice;
      ice -= melt; liq += melt;
    } else {
      let refr = CFR * CFMAX * (TT - T);
      if (refr > liq) refr = liq;
      liq -= refr; ice += refr;
    }
    if (ice > 0) {
      liq += rain;
      let o = liq - CWH * ice;
      if (o < 0) o = 0;
      liq -= o;
    } else { liq = 0; ice = 0; }
    if (t >= WARM) out[t - WARM] = ice + liq;
  }
  return out;
}

/** Uncalibrated WaterSim daily energy balance (Walter et al. 2005 / UEB style) SWE for the water-year days. */
export function energyBalance(w: Window, p: Record<string, number>) {
  const { TT, TTI, SFCF, CWH, ALB_MAX: amax, ALB_MIN: amin, ALB_DECAY: adec, Z0: z0, TURB: turb, GHEAT: gheat, CANOPY: canopy } = p;
  const z = w.elev, pa = 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26;
  let ice = 0, liq = 0, cc = 0, age = 0;
  const out = new Float64Array(w.nwy);
  for (let t = 0; t < WARM + w.nwy; t++) {
    const P = w.P[t], T = w.Tm[t];
    const ra = extraterrestrial(w.lat, dayOfYear(w.t0 + t * DAY));
    const rso = (0.75 + 2e-5 * z) * ra;
    const Rs = (((0.16 * Math.sqrt(Math.max(w.Tx[t] - w.Tn[t], 0))) * ra) / 0.0864) * 0.0864;
    const ea = (esatKpa(w.Tn[t]) * 1000) / 1000;
    const u10 = 2;
    const fs = snowFraction(T, TT, TTI);
    const snow = P * fs * SFCF, rain = P * (1 - fs);
    if (ice + liq + snow <= 0) {
      ice = 0; liq = 0; cc = 0; age = 0;
      if (t >= WARM) out[t - WARM] = 0;
      continue;
    }
    ice += snow;
    if (snow > 0) { let r = snow / 5; if (r > 1) r = 1; age = age * (1 - r); }
    const Ts = T < 0 ? T : 0;
    cc += C_ICE * snow * -Ts;
    const alb = amin + (amax - amin) * Math.exp(-age / adec);
    const swe = ice + liq;
    const albEff = 0.2 + (alb - 0.2) * (1 - Math.exp(-swe / 15));
    const sw = (1 - albEff) * Rs * canopy;
    const TaK = T + 273.15, TsK = Ts + 273.15;
    let epsC = 1.24 * ((ea * 10) / TaK) ** (1 / 7);
    if (epsC > 1) epsC = 1;
    let cf = 0.5;
    if (rso > 0) { cf = 1 - Rs / rso; if (cf < 0) cf = 0; if (cf > 1) cf = 1; }
    let eps = (1 - 0.84 * cf) * epsC + 0.84 * cf;
    eps = canopy * eps + (1 - canopy) * 1;
    const lwin = eps * SIGMA * TaK ** 4, lwout = 0.98 * SIGMA * TsK ** 4;
    const lz = Math.log(2 / z0);
    let u2 = (u10 * lz) / Math.log(10 / z0);
    if (u2 < 0.3) u2 = 0.3;
    const rA = (lz * lz) / (VK * VK * u2);
    const ri = (9.81 * 2 * (T - Ts)) / (TaK * u2 * u2);
    const fst = ri > 0 ? 1 / (1 + 10 * ri) : 1;
    const rho = (pa * 1000) / (287.05 * TaK);
    const ch = ((turb * fst) / rA) * 86400;
    const Hs = rho * CP_AIR * (T - Ts) * ch;
    const emass = ((rho * 0.622 * (ea - esatIce(Ts))) / pa) * ch;
    const LE = (liq > 0 ? LAMBDA_V : LAMBDA_S) * emass;
    const Qr = C_WATER * rain * (T > 0 ? T : 0);
    const Q = sw + lwin - lwout + Hs + LE + gheat + Qr;
    liq += rain;
    if (Q >= 0) {
      const d = Q < cc ? Q : cc;
      cc -= d;
      let melt = (Q - d) / LAMBDA_F;
      if (melt > ice) melt = ice;
      ice -= melt; liq += melt;
    } else {
      let refr = -Q / LAMBDA_F;
      if (refr > liq) refr = liq;
      liq -= refr; ice += refr;
      cc += -(Q + refr * LAMBDA_F);
    }
    const ccmax = C_ICE * ice * -Ts;
    if (cc > ccmax) cc = ccmax;
    if (cc > 0 && liq > 0) {
      let r2 = cc / LAMBDA_F;
      if (r2 > liq) r2 = liq;
      liq -= r2; ice += r2; cc -= r2 * LAMBDA_F;
    }
    if (emass < 0) {
      let loss = -emass;
      const a = loss < liq ? loss : liq;
      liq -= a; loss -= a;
      const b = loss < ice ? loss : ice;
      ice -= b;
    } else ice += emass;
    if (ice > 0) {
      let o = liq - CWH * ice;
      if (o < 0) o = 0;
      liq -= o;
    } else { liq = 0; ice = 0; cc = 0; }
    age += T > 0 ? 3 : 1;
    if (t >= WARM) out[t - WARM] = ice + liq;
  }
  return out;
}

// ---- scoring helpers for the readout ---------------------------------------------------------------------
export function nse(obs: ArrayLike<number>, sim: ArrayLike<number>) {
  let n = 0, mean = 0;
  for (let i = 0; i < obs.length; i++) if (Number.isFinite(obs[i])) { n++; mean += obs[i]; }
  if (n < 30) return NaN;
  mean /= n;
  let se = 0, sv = 0;
  for (let i = 0; i < obs.length; i++) if (Number.isFinite(obs[i])) { se += (sim[i] - obs[i]) ** 2; sv += (obs[i] - mean) ** 2; }
  return 1 - se / sv;
}

/** Observed annual numbers as the exam scored them: >= 300 readings and a peak of at least 25 mm. */
export function observedYear(obs: Float64Array) {
  let n = 0;
  for (const v of obs) if (Number.isFinite(v)) n++;
  const filled = Float64Array.from(obs, v => (Number.isFinite(v) ? v : 0));
  const peak = maxOf(filled), ok = n >= 300 && peak >= 25;
  return { ok, n, peak, peakDay: argmax(filled), melt: ok ? meltOut(filled) : -1 };
}

/** Share of Oct-Jun precipitation falling as snow by WaterSim's default degree-day partition (TT 0, TTI 2 C). */
export function snowShare(w: Window) {
  let s = 0, p = 0;
  for (let t = WARM; t < WARM + 273; t++) { p += w.P[t]; s += w.P[t] * snowFraction(w.Tm[t], 0, 2); }
  return p > 0 ? s / p : NaN;
}
export function precipTotal(w: Window) {
  let p = 0;
  for (let t = WARM; t < WARM + w.nwy; t++) p += w.P[t];
  return p;
}
export { snowFraction };
