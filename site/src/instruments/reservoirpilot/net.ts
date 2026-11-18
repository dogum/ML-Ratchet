// ReservoirPilot in the browser: the causal release policy, WaterData's daily mass balance and cost J, SOP, the
// sim-tuned classical rule and WaterData's perfect-foresight DP. No DOM, so the parity test runs it in Node.
// Mirrors models/reservoirpilot/reservoirpilot.py (Pilot, sop, tuned_rule, cost_components) and
// shared/water/data/scripts/run_baselines_reservoir.py (dp, numpy's linspace / unique / interp arithmetic kept).
// The shipped page's port (models/reservoirpilot/training/engine.js) was the starting point.
import type { Pack } from '../../lib/nn/pack.ts';

export const K = 0.0864;                       // hm3 per (m3/s x day)
const TAUS = [5, 20, 60];

export interface DamInfo {
  res_id: string; label: string; main_use: string; use_name: string; start: string; n: number;
  Qmax: number; cap: number; S0: number; cap_years: number;
  exam_J: { pilot: number; sop: number; rule: number; historical: number; dp: number };
  dp_J: number;                                 // WaterData's DP re-run on the stored excerpt (float32 inflow)
}
export interface DamsMeta { dams: DamInfo[]; exam: Record<string, number> }

/** One operating task: what WaterData's exam hands a policy. */
export interface Task {
  I: Float64Array;          // true inflow (m3/s): drives the mass balance
  seen: Float64Array;       // inflow the causal policies read (= I unless a gauge error is added)
  doy: Int16Array;          // day of year, 1..366
  Dcurve: Float64Array;     // demand calendar D(doy), m3/s (calibration half)
  Icurve: Float64Array;     // inflow climatology (calibration half)
  Qmax: number; cap: number; S0: number;
}

export interface Run { R: Float64Array; S: Float64Array; H?: Float64Array; E?: Float64Array }

export interface Components {
  J: number; J_short: number; J_flood: number; shortage_ratio: number; shortage_days: number;
  flood_days: number; max_excess: number; S_mean: number; dS_end: number; empty: number; full: number;
}

// ---- data ----------------------------------------------------------------------------------------
/** Day of year (1..366) of `n` consecutive days from an ISO date, as pandas' day_of_year. */
export function doyOf(start: string, n: number): Int16Array {
  const out = new Int16Array(n);
  const t0 = Date.parse(start + 'T00:00:00Z');
  for (let i = 0; i < n; i++) {
    const d = new Date(t0 + i * 86400000);
    out[i] = Math.round((d.getTime() - Date.UTC(d.getUTCFullYear(), 0, 1)) / 86400000) + 1;
  }
  return out;
}

export interface DamSeries { info: DamInfo; I: Float64Array; Dcurve: Float64Array; Icurve: Float64Array; Robs: Float64Array; Sobs: Float64Array; doy: Int16Array }

export function damSeries(pack: Pack<DamsMeta>, i: number): DamSeries {
  const f = (k: string) => Float64Array.from(pack.t[`${k}${i}`].data);
  const info = pack.meta.dams[i];
  return { info, I: f('I'), Dcurve: f('Dcurve'), Icurve: f('Icurve'), Robs: f('Robs'), Sobs: f('Sobs'), doy: doyOf(info.start, info.n) };
}

// ---- what-ifs on the input -----------------------------------------------------------------------
export interface WhatIf {
  inflow: number;      // x the whole scored decade's inflow (a drier or wetter decade than the calibration one)
  pulse: number;       // flood pulse peak, x Qmax (0 = none)
  pulseDay: number;    // day index of the pulse peak
  capacity: number;    // x capacity (start storage scaled with it: same starting fill)
  gauge: number;       // storage-gauge error sigma, % of capacity: back-calculated inflow error the policies read
  seed: number;
  demand: number;      // x the demand calendar
}
export const RECORDED: WhatIf = { inflow: 1, pulse: 0, pulseDay: 0, capacity: 1, gauge: 0, seed: 1, demand: 1 };
export const isRecorded = (w: WhatIf) => w.inflow === 1 && w.pulse === 0 && w.capacity === 1 && w.gauge === 0 && w.demand === 1;

/** Flood hydrograph added to the inflow: q(k) = peak (k/3) e^(1 - k/3) for k = 0..30 days after onset (peak 3 days in). */
export const PULSE_LEN = 31;
export function pulseShape(k: number) { return (k / 3) * Math.exp(1 - k / 3); }

/** mulberry32 + Box-Muller (the kit's rng), so a gauge-noise draw is repeatable per seed and in Python. */
export function gaussians(seed: number, n: number): Float64Array {
  let a = seed >>> 0;
  const u = () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let x = 0; while (x === 0) x = u();
    const y = u();
    out[i] = Math.sqrt(-2 * Math.log(x)) * Math.cos(2 * Math.PI * y);
  }
  return out;
}

export function makeTask(s: DamSeries, w: WhatIf): Task {
  const n = s.info.n, q = s.info.Qmax, cap = s.info.cap * w.capacity;
  const I = new Float64Array(n);
  for (let t = 0; t < n; t++) I[t] = s.I[t] * w.inflow;
  if (w.pulse > 0) {
    const t0 = w.pulseDay - 3;
    for (let k = 0; k < PULSE_LEN; k++) { const t = t0 + k; if (t >= 0 && t < n) I[t] += w.pulse * q * pulseShape(k); }
  }
  let seen = I;
  if (w.gauge > 0) {
    // ResOpsUS inflow is back-calculated, I = R + dS/dt: a storage gauge error e_t ~ N(0, sigma cap) becomes (e_t - e_t-1)/K
    const e = gaussians(w.seed, n + 1), sg = (w.gauge / 100) * cap;
    seen = new Float64Array(n);
    for (let t = 0; t < n; t++) seen[t] = I[t] + (sg * e[t + 1] - sg * e[t]) / K;
  }
  const Dcurve = w.demand === 1 ? s.Dcurve : s.Dcurve.map(v => v * w.demand);
  return { I, seen, doy: s.doy, Dcurve, Icurve: s.Icurve, Qmax: q, cap, S0: s.info.S0 * w.capacity };
}

// ---- calendar and network ------------------------------------------------------------------------
function fwdMean(curve: Float64Array, w: number) {      // mean over doy..doy+w-1 (wrapping), for every doy
  const c = new Float64Array(366 + w); c.set(curve); c.set(curve.subarray(0, w), 366);
  const cs = new Float64Array(c.length + 1);
  for (let i = 0; i < c.length; i++) cs[i + 1] = cs[i] + c[i];
  const out = new Float64Array(366);
  for (let i = 0; i < 366; i++) out[i] = (cs[i + w] - cs[i]) / w;
  return out;
}

export function demandOf(task: Task) {
  const D = new Float64Array(task.doy.length);
  for (let t = 0; t < D.length; t++) D[t] = task.Dcurve[task.doy[t] - 1];
  return D;
}

export interface PilotNet { mu: Float64Array; sd: Float64Array; W: Float64Array[]; b: Float64Array[]; dims: number[][] }

export function pilotNet(pack: Pack<unknown>): PilotNet {
  const g = (k: string) => Float64Array.from(pack.t[k].data);
  const W: Float64Array[] = [], b: Float64Array[] = [], dims: number[][] = [];
  for (let l = 0; pack.t[`W${l}`]; l++) { W.push(g(`W${l}`)); b.push(g(`b${l}`)); dims.push(pack.t[`W${l}`].shape); }
  return { mu: g('mu'), sd: g('sd'), W, b, dims };
}

/** The MLP: (x - mu)/sd, tanh hidden layers, linear output. W stored [in, out] row-major (numpy x @ W). */
export function mlp(net: PilotNet, x: Float64Array, buf: Float64Array[]): Float64Array {
  let h = buf[0];
  for (let i = 0; i < x.length; i++) h[i] = (x[i] - net.mu[i]) / net.sd[i];
  for (let l = 0; l < net.W.length; l++) {
    const [nin, nout] = net.dims[l], W = net.W[l], bb = net.b[l], o = buf[l + 1], last = l === net.W.length - 1;
    for (let j = 0; j < nout; j++) o[j] = 0;
    for (let i = 0; i < nin; i++) { const hi = h[i], r = i * nout; for (let j = 0; j < nout; j++) o[j] += hi * W[r + j]; }
    for (let j = 0; j < nout; j++) o[j] = last ? o[j] + bb[j] : Math.tanh(o[j] + bb[j]);
    h = o;
  }
  return h;
}

const sig = (z: number) => 1 / (1 + Math.exp(-z));

/** ReservoirPilot over the whole record (Pilot.run): features from what an operator knows today, h and e, mass balance. */
export function runPilot(net: PilotNet, task: Task): Run & { H: Float64Array; E: Float64Array } {
  const n = task.I.length, I = task.I, Is = task.seen, cap = task.cap, q = Math.max(task.Qmax, 1e-9), qK = q * K;
  const d30 = fwdMean(task.Dcurve, 30), d90 = fwdMean(task.Dcurve, 90), i30 = fwdMean(task.Icurve, 30), i90 = fwdMean(task.Icurve, 90);
  const R = new Float64Array(n), S = new Float64Array(n), H = new Float64Array(n), E = new Float64Array(n);
  const buf = [new Float64Array(15), ...net.dims.map(d => new Float64Array(d[1]))];
  const x = new Float64Array(15), as = Math.asinh;
  const e0 = i30[task.doy[0] - 1];
  let e1 = e0, e2 = e0, e3 = e0, s = task.S0;
  for (let t = 0; t < n; t++) {
    const j = task.doy[t] - 1, D = task.Dcurve[j], Dn30 = d30[j], Dn90 = d90[j], Ic30 = i30[j], Ic90 = i90[j], It = Is[t];
    e1 = e1 + (It - e1) / TAUS[0]; e2 = e2 + (It - e2) / TAUS[1]; e3 = e3 + (It - e3) / TAUS[2];
    x[0] = s / cap; x[1] = as(It / q); x[2] = as(e1 / q); x[3] = as(e2 / q); x[4] = as(e3 / q);
    x[5] = D / q; x[6] = Dn30 / q; x[7] = Math.log(cap / qK) / 3.0; x[8] = as((cap - s) / qK) / 3.0;
    x[9] = as((s + It * K - D * K - cap) / qK); x[10] = as(s / (Dn30 * K + 1e-3 * qK) / 10.0);
    x[11] = as(Ic30 / q); x[12] = as(Ic90 / q); x[13] = as((s + 90 * K * (Ic90 - Dn90) - cap) / qK) / 3.0;
    x[14] = as((s + 30 * K * (Ic30 - Dn30)) / qK) / 3.0;
    const o = mlp(net, x, buf);
    const h = sig(o[0]), e = sig(o[1]);
    let r = h * D + 1.25 * e * Math.max(q - D, 0.0);
    const avail = s + I[t] * K;
    r = Math.min(Math.max(r, 0.0), Math.max(avail, 0) / K);
    let s1 = avail - r * K;
    if (s1 > cap) { r += (s1 - cap) / K; s1 = cap; }
    s = Math.max(s1, 0.0); R[t] = r; S[t] = s; H[t] = h; E[t] = e;
  }
  return { R, S, H, E };
}

/** Standard operating policy: release demand when the water is there, spill when full. */
export function runSOP(task: Task): Run {
  const n = task.I.length, I = task.I, cap = task.cap, R = new Float64Array(n), S = new Float64Array(n);
  let s = task.S0;
  for (let t = 0; t < n; t++) {
    const D = task.Dcurve[task.doy[t] - 1], avail = s + I[t] * K;
    let r = Math.min(D, Math.max(avail, 0) / K), s1 = avail - r * K;
    if (s1 > cap) { r += (s1 - cap) / K; s1 = cap; }
    s = Math.max(s1, 0.0); R[t] = r; S[t] = s;
  }
  return { R, S };
}

/** The classical comparator (rung c2, tuned on the simulator's g4 generator): SOP + flood pool + linear hedging. */
export const RULE_G4 = { f: 0.6, tau: 3.0, H: 10.0 };
export function runRule(task: Task, p = RULE_G4): Run {
  const n = task.I.length, I = task.I, Is = task.seen, cap = task.cap, q = task.Qmax;
  const R = new Float64Array(n), S = new Float64Array(n);
  let s = task.S0;
  for (let t = 0; t < n; t++) {
    const D = task.Dcurve[task.doy[t] - 1];
    let r = D * Math.min(1.0, (s + Is[t] * K) / Math.max(p.H * D * K, 1e-12));
    if (D < q) r = Math.max(r, Math.min(q, D + (s - p.f * cap) / (p.tau * K)));
    const avail = s + I[t] * K;
    r = Math.min(Math.max(r, 0.0), Math.max(avail, 0) / K);
    let s1 = avail - r * K;
    const sp = Math.max(s1 - cap, 0.0); r += sp / K; s1 -= sp;
    s = Math.max(s1, 0.0); R[t] = r; S[t] = s;
  }
  return { R, S };
}

// ---- WaterData's perfect-foresight DP (run_baselines_reservoir.dp), numpy arithmetic kept --------------
function linspace(a: number, b: number, num: number, out: number[]) {
  const div = num - 1, delta = b - a, step = delta / div;
  for (let i = 0; i < num; i++) out.push(i === num - 1 ? b : (step === 0 ? (i / div) * delta : i * step) + a);
}

/** Knows the whole decade's inflow; 201 storage levels, 21 + 21 release candidates a day, ends at least as full as it
 *  started (penalty 1000 x deficit / capacity). An approximate lower bound on J, not a policy anyone can run. */
export function runDP(task: Task, nlev = 201, nr = 21): Run {
  const n = task.I.length, I = task.I, q = task.Qmax, cap = task.cap, S0 = task.S0, D = demandOf(task);
  const levs: number[] = []; linspace(0, cap, nlev, levs);
  const lev = Float64Array.from(levs);
  const step = cap / (nlev - 1);
  let V = new Float64Array(nlev), V2 = new Float64Array(nlev);
  for (let i = 0; i < nlev; i++) V[i] = 1000 * Math.max(0, S0 - lev[i]) / cap;
  const slopes = new Float64Array(nlev - 1);
  const pol = new Float32Array(n * nlev);
  const Rc = new Float64Array(2 * nr);
  const cand: number[] = [];
  const interpV = (x: number) => {
    if (x < lev[0]) return V[0];
    if (x > lev[nlev - 1]) return V[nlev - 1];
    let j = Math.min(nlev - 1, Math.max(0, Math.floor(x / step)));
    while (j + 1 < nlev && lev[j + 1] <= x) j++;
    while (j > 0 && lev[j] > x) j--;
    if (j === nlev - 1) return V[j];
    if (lev[j] === x) return V[j];
    return slopes[j] * (x - lev[j]) + V[j];
  };
  for (let t = n - 1; t >= 0; t--) {
    const Dt = D[t];
    cand.length = 0; linspace(0, Dt, nr, cand); linspace(Dt, Math.max(q, Dt), nr, cand);
    cand.sort((a, b) => a - b);
    let nc = 0;
    for (const v of cand) if (nc === 0 || v !== Rc[nc - 1]) Rc[nc++] = v;
    for (let j = 0; j < nlev - 1; j++) slopes[j] = (V[j + 1] - V[j]) / (lev[j + 1] - lev[j]);
    const IK = I[t] * K, dd = Math.max(Dt, 1e-6);
    for (let i = 0; i < nlev; i++) {
      const avail = lev[i] + IK, amax = Math.max(avail, 0) / K;
      let best = Infinity, bj = 0;
      for (let j = 0; j < nc; j++) {
        let r = Math.min(Rc[j], amax), s1 = avail - r * K;
        const spill = Math.max(0, s1 - cap);
        r = r + spill / K; s1 = Math.min(Math.max(s1 - spill, 0), cap);
        const a = Math.max(0, Dt - r) / dd, b = Math.max(0, r - q) / q;
        const c = (a * a + 10 * (b * b)) + interpV(s1);
        if (c < best) { best = c; bj = j; }
      }
      V2[i] = best; pol[t * nlev + i] = Rc[bj];
    }
    const tmp = V; V = V2; V2 = tmp;
  }
  const R = new Float64Array(n), S = new Float64Array(n);
  let s = S0;
  for (let t = 0; t < n; t++) {
    // np.interp(S, lev, pol[t])
    const p = pol.subarray(t * nlev, (t + 1) * nlev);
    let r: number;
    if (s <= lev[0]) r = p[0];
    else if (s >= lev[nlev - 1]) r = p[nlev - 1];
    else {
      let j = Math.min(nlev - 2, Math.floor(s / step));
      while (j + 1 < nlev && lev[j + 1] <= s) j++;
      while (j > 0 && lev[j] > s) j--;
      r = lev[j] === s ? p[j] : ((p[j + 1] - p[j]) / (lev[j + 1] - lev[j])) * (s - lev[j]) + p[j];
    }
    const avail = s + I[t] * K;
    r = Math.min(r, Math.max(avail, 0) / K);
    let s1 = avail - r * K;
    if (s1 > cap) { r += (s1 - cap) / K; s1 = cap; }
    R[t] = r; s = Math.max(s1, 0.0); S[t] = s;
  }
  return { R, S };
}

// ---- WaterData's cost J and every component (reservoirpilot.cost_components) ------------------------
export function components(run: Run, task: Task): Components {
  const { R, S } = run, n = R.length, q = task.Qmax, cap = task.cap, D = demandOf(task);
  let jt = 0, js = 0, jf = 0, sh = 0, sd = 0, nshort = 0, fd = 0, mx = 0, sm = 0, emp = 0, ful = 0;
  for (let t = 0; t < n; t++) {
    const a = Math.max(0, D[t] - R[t]) / Math.max(D[t], 1e-6), b = Math.max(0, R[t] - q) / q;
    jt += a * a + 10 * (b * b); js += a * a; jf += 10 * (b * b); sh += Math.max(0, D[t] - R[t]); sd += D[t];
    if (R[t] < 0.999 * D[t]) nshort++;
    if (R[t] > q) fd++;
    mx = Math.max(mx, Math.max(0, R[t] - q));
    sm += S[t]; if (S[t] < 0.01 * cap) emp++; if (S[t] > 0.999 * cap) ful++;
  }
  return {
    J: jt / n, J_short: js / n, J_flood: jf / n, shortage_ratio: sh / Math.max(sd, 1e-12), shortage_days: nshort / n,
    flood_days: (fd / n) * 365.25, max_excess: mx / q, S_mean: sm / n / cap, dS_end: (S[n - 1] - task.S0) / cap,
    empty: emp / n, full: ful / n,
  };
}
