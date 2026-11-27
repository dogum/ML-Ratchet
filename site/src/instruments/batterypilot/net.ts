// BatteryPilot in the browser: a pure port of models/batterypilot/batterypilot.py — the causal features (build,
// rolling_scale, _rollmax), the day-ahead backbone (value_curves, v1.3), the value-curve network (PolicyNet.z_knots /
// values, plain or hybrid), the battery (Battery: c_lin and the rainflow wear of degradation_cost), the offer-curve
// executor (s_star, step_offer) and GridData's daily-spread rule.
// No DOM, so site/tests/parity.test.ts can run it in Node.
import type { Pack } from '../../lib/nn/pack.ts';
import type { Tensor } from '../../lib/nn/tensor.ts';

export interface BPMeta {
  n_params: number; knots: number[]; H_DA: number; H_RT: number; H_DF: number; publish_hour: number; n_feat: number;
  layers: number; f16_max_abs_z_diff: number;
  hybrid?: boolean; min_step?: number;   // v1.3: the network reads the backbone curve and outputs a residual on it
  plan_lag_h?: number;   // v1.3.1: the backbone's wear cost uses the temperature this many hours earlier (causal_wear)
}
export interface Annual { bp: number; pf: number; rolling: number; spread: number; bp_cycles: number }
export interface ExMeta {
  id: string; market: string; node: string; label: string; year: number; n: number; t0: string; tz: string;
  month_start: number[]; annual: Record<string, Annual>; peak: { i: number; rt: number; utc: string };
}
export interface XMeta { examples: ExMeta[]; durations: number[]; etas: number[]; costs: number[] }

export interface Path {
  rt: Float32Array; da: Float32Array; df: Float32Array; T: Float32Array;
  hour: Float32Array; day: Float32Array; dow: Float32Array; doy: Float32Array;
  ps: Float64Array; dmax: Float64Array; // causal price scale (7-day median DA) and 30-day max of the demand forecast
}

const R_GAS = 8.314;

// ---- battery (Battery in batterypilot.py) -------------------------------------------------------------------------
export interface Battery {
  P: number; d: number; E: number; eta: number; etac: number; etad: number; lo: number; hi: number; Enom: number;
  cost: number; crate: number; tf: (T: number) => number; phi: (depth: number) => number; clin: (T: number) => number;
}

export function battery(duration: number, eta = 0.85, cost = 130): Battery {
  const P = 1.0, E = P * duration, lo = 0.05, hi = 0.95, Enom = E / (hi - lo);
  const N100 = 6000, k = 1.3, Ea = 31700, b = 370.3, kappa = 0.25, crate = P / Enom;
  const etac = Math.sqrt(eta), etad = Math.sqrt(eta);
  const f0 = Math.exp(-(Ea - b * 0.5) / (R_GAS * (25.0 + 1.5 + 273.15)));
  const tf = (T: number) => {
    const Tc = 25.0 + kappa * (T - 25.0) + 3.0 * crate;
    return Math.exp(-(Ea - b * crate) / (R_GAS * (Tc + 273.15))) / f0;
  };
  const phi = (d: number) => Math.pow(d, k) / N100;
  const perCycle = 1000.0 * cost * Enom * phi(E / Enom); // $ per full usable cycle at 25 C
  return { P, d: duration, E, eta, etac, etad, lo, hi, Enom, cost, crate, tf, phi, clin: (T) => (perCycle * tf(T)) / (E * etad) };
}

// ---- causal features (build in batterypilot.py) -------------------------------------------------------------------
function median(a: number[]) {
  const s = Float64Array.from(a).sort();
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

/** rolling_scale: median DA of the trailing week, recomputed every 24 h, floor $5. */
export function rollingScale(da: ArrayLike<number>, hist = 168): Float64Array {
  const n = da.length, out = new Float64Array(n);
  let last = 25.0;
  for (let t = 0; t < n; t++) {
    if (t % 24 === 0) {
      const a = Math.max(0, t - hist);
      const w: number[] = [];
      for (let i = a; i < t; i++) if (Number.isFinite(da[i])) w.push(da[i]);
      if (t - a >= 24 && w.length) last = median(w);
      else {
        const w2: number[] = [];
        for (let i = t; i < Math.min(n, t + 24); i++) if (Number.isFinite(da[i])) w2.push(da[i]);
        if (w2.length) last = median(w2);
      }
    }
    out[t] = Math.max(last, 5.0);
  }
  return out;
}

/** _rollmax: trailing max over the last w+1 values, inclusive (monotone deque). */
export function rollMax(x: ArrayLike<number>, w: number): Float64Array {
  const n = x.length, out = new Float64Array(n), q = new Int32Array(n);
  let h = 0, t = 0;
  for (let i = 0; i < n; i++) {
    while (t > h && x[q[t - 1]] <= x[i]) t--;
    q[t++] = i;
    while (q[h] < i - w) h++;
    out[i] = x[q[h]];
  }
  return out;
}

export function pathOf(X: Pack<XMeta>, i: number): Path {
  const g = (k: string) => X.t[`x${i}_${k}`].data;
  const p = { rt: g('rt'), da: g('da'), df: g('df'), T: g('T'), hour: g('hour'), day: g('day'), dow: g('dow'), doy: g('doy') } as Path;
  p.ps = rollingScale(p.da);
  p.dmax = rollMax(p.df, 720);
  return p;
}

/** The 135 features for decision hour t (same order as build()). */
export function features(m: BPMeta, p: Path, t: number, dur: number, cl: number, out: Float64Array) {
  const n = p.rt.length, ps = p.ps[t];
  const c = (i: number) => (i < 0 ? 0 : i > n - 1 ? n - 1 : i);
  let j = 0;
  const known = new Uint8Array(m.H_DA);
  for (let o = 0; o < m.H_DA; o++) {
    const i = c(t + o), dd = p.day[i] - p.day[t];
    known[o] = dd <= 0 || (dd === 1 && p.hour[t] >= m.publish_hour) ? 1 : 0;
    out[j++] = known[o] ? Math.asinh(p.da[i] / ps) : 0;
  }
  for (let o = 0; o < m.H_DA; o++) out[j++] = known[o];
  for (let o = -m.H_RT; o < 0; o++) out[j++] = Math.asinh(p.rt[c(t + o)] / ps);
  for (let o = -6; o < 0; o++) { const i = c(t + o); out[j++] = Math.asinh(p.rt[i] / ps) - Math.asinh(p.da[i] / ps); }
  const dm = Math.max(p.dmax[t], 1);
  for (let o = 0; o < m.H_DF; o++) out[j++] = p.df[c(t + o)] / dm - 0.8;
  out[j++] = (p.T[c(t - 1)] - 20) / 10;
  const hh = (2 * Math.PI * p.hour[t]) / 24, dy = (2 * Math.PI * p.doy[t]) / 365.25;
  out[j++] = Math.sin(hh); out[j++] = Math.cos(hh); out[j++] = Math.sin(dy); out[j++] = Math.cos(dy);
  out[j++] = p.dow[t] >= 5 ? 1 : 0;
  out[j++] = Math.log2(dur); out[j++] = cl / ps; out[j++] = Math.log(ps / 30);
  for (let i = 0; i < j; i++) if (!Number.isFinite(out[i])) out[i] = Number.isNaN(out[i]) ? 0 : out[i] > 0 ? 10 : -10;
  return j;
}

// ---- the day-ahead backbone (value_curves in batterypilot.py) ------------------------------------------------------
/** SOC up to which the decreasing curve lam (on the grid) is >= y (piecewise linear). */
function cross(lam: Float64Array, y: number, dg: number): number {
  let s = 0;
  for (let k = 0; k < lam.length - 1; k++) {
    const a = lam[k], b = lam[k + 1];
    const f = a - b <= 1e-12 ? (a >= y ? 1 : 0) : Math.min(Math.max((a - y) / Math.max(a - b, 1e-12), 0), 1);
    s += f * dg;
  }
  return s;
}

/** Linear interpolation on the uniform grid 0..1 (N+1 points), clamped at the ends (np.interp). */
function interpU(x: number, f: Float64Array): number {
  const N = f.length - 1;
  if (x <= 0) return f[0];
  if (x >= 1) return f[N];
  const u = x * N, i = Math.min(Math.floor(u), N - 1), w = u - i;
  return f[i] + w * (f[i + 1] - f[i]);
}

/** Backbone value curves ($ per stored MWh at the SOC knots) for hours [0, b): a backward dynamic programme over the
 * DA prices known at each re-plan (the first hour, then 14:00 local), as backward_pass / value_curves. */
export function backbone(knots: number[], p: Path, bat: Battery, b: number, N = 200, lag = 0): Float64Array {
  const K = knots.length, out = new Float64Array(b * K), n = p.da.length;
  const E = bat.P * bat.d, eta = Math.sqrt(bat.eta), up = (bat.P * eta) / E, dn = bat.P / (eta * E), dg = 1 / N;
  const grid = new Float64Array(N + 1);
  for (let i = 0; i <= N; i++) grid[i] = i / N;
  // local days in order of appearance: first index and end of each
  const first: number[] = [], pos = new Map<number, number>();
  for (let t = 0; t < n; t++) if (!pos.has(p.day[t])) { pos.set(p.day[t], first.length); first.push(t); }
  const ends = first.slice(1).concat([n]);
  const replan = [0];
  for (let t = 1; t < n; t++) if (p.hour[t] === 14) replan.push(t);
  let lam: Float64Array[] = [], W = new Float64Array(N + 1), Wn = new Float64Array(N + 1);
  const tgt = new Float64Array(N + 1), g = new Float64Array(N + 1);
  for (let j = 0; j < replan.length && replan[j] < b; j++) {
    const t0 = replan[j], dj = pos.get(p.day[t0])!;
    const kend = t0 > 0 || p.hour[t0] >= 14 ? ends[Math.min(dj + 1, ends.length - 1)] : ends[dj];
    const t1 = j + 1 < replan.length ? replan[j + 1] : n, H = kend - t0;
    lam = Array.from({ length: H + 1 }, () => new Float64Array(N + 1));
    W.fill(0);
    for (let k = H - 1; k >= 0; k--) {
      const pr = p.da[t0 + k], c = bat.clin(p.T[Math.max(t0 + k - lag, 0)]), nxt = lam[k + 1];   // causal_wear
      const slo = cross(nxt, pr / eta, dg), shi = cross(nxt, eta * (pr - c), dg);
      for (let i = 0; i <= N; i++) {
        let x = Math.min(Math.max(grid[i], slo), Math.max(shi, slo));
        x = Math.min(Math.max(x, grid[i] - dn), grid[i] + up);
        x = Math.min(Math.max(x, 0), 1);
        tgt[i] = x;
        const d = x - grid[i];
        const r = d > 0 ? (-pr * d * E) / eta : (pr - c) * -d * E * eta;
        Wn[i] = r + interpU(x, W);
      }
      g[0] = (Wn[1] - Wn[0]) / dg / E;
      g[N] = (Wn[N] - Wn[N - 1]) / dg / E;
      for (let i = 1; i < N; i++) g[i] = (Wn[i + 1] - Wn[i - 1]) / (2 * dg) / E;
      const L = lam[k];
      L[0] = g[0];
      for (let i = 1; i <= N; i++) L[i] = Math.min(L[i - 1], g[i]);
      const tmp = W; W = Wn; Wn = tmp;
    }
    for (let i = 0; i < Math.min(t1 - t0, H) && t0 + i < b; i++)
      for (let k = 0; k < K; k++) out[(t0 + i) * K + k] = interpU(knots[k], lam[i + 1]);
  }
  return out;
}

// ---- the network (PolicyNet) -------------------------------------------------------------------------------------
export function makeNet(W: Pack<BPMeta>) {
  const mu = W.t.x_mu.data, sd = W.t.x_sd.data, L = W.meta.layers;
  const layers: Tensor[][] = [];
  for (let l = 0; l < L; l++) layers.push([W.t[`W${l}`], W.t[`b${l}`]]);
  const bufs = layers.map(([w]) => new Float64Array(w.shape[0]));
  const x0 = new Float64Array(mu.length);
  const K = W.meta.knots.length;
  /** z knots: asinh(value / price scale) at the SOC knots, decreasing, clipped to ±12. */
  return (x: ArrayLike<number>, z: Float64Array = new Float64Array(K)) => {
    for (let i = 0; i < x0.length; i++) x0[i] = (x[i] - mu[i]) / sd[i];
    let h: Float64Array = x0;
    for (let l = 0; l < L; l++) {
      const [w, b] = layers[l], o = bufs[l], [O, I] = w.shape, wd = w.data, bd = b.data;
      for (let r = 0; r < O; r++) {
        let s = bd[r];
        const base = r * I;
        for (let q = 0; q < I; q++) s += wd[base + q] * h[q];
        o[r] = l < L - 1 ? s / (1 + Math.exp(-s)) : s;
      }
      h = o;
    }
    if (W.meta.hybrid) {
      // residual on the backbone knots (the last K inputs), each knot at least min_step below the last
      const ms = W.meta.min_step ?? 0, off = x0.length - K;
      for (let k = 0; k < K; k++) {
        const u = x[off + k] + h[k];
        z[k] = k === 0 ? u : ms > 0 ? Math.min(z[k - 1] - ms, u) : Math.min(z[k - 1], u);
      }
    } else {
      z[0] = h[0];
      for (let k = 1; k < K; k++) {
        const v = h[k], sp = v > 30 ? v : Math.log1p(Math.exp(v));
        z[k] = z[k - 1] - (sp + 0.01);
      }
    }
    for (let k = 0; k < K; k++) z[k] = Math.min(Math.max(z[k], -12), 12);
    return z;
  };
}

// ---- the offer-curve executor (s_star, step_offer) ---------------------------------------------------------------
export function sStar(knots: number[], v: ArrayLike<number>, y: number) {
  let s = 0;
  for (let k = 0; k < knots.length - 1; k++) {
    const a = v[k], b = v[k + 1];
    s += Math.min(Math.max((a - y) / Math.max(a - b, 1e-9), 0), 1) * (knots[k + 1] - knots[k]);
  }
  return s;
}

export function stepOffer(knots: number[], bat: Battery, soc: number, v: ArrayLike<number>, price: number, cl: number): [number, number, number] {
  const shi = sStar(knots, v, bat.etad * (price - cl)), slo = sStar(knots, v, price / bat.etac);
  let tgt = Math.min(Math.max(soc, slo), Math.max(shi, slo));
  tgt = Math.min(Math.max(tgt, soc - bat.P / (bat.etad * bat.E)), soc + (bat.P * bat.etac) / bat.E);
  tgt = Math.min(Math.max(tgt, 0), 1);
  const d = tgt - soc;
  return [tgt, d > 0 ? (d * bat.E) / bat.etac : 0, d < 0 ? -d * bat.E * bat.etad : 0];
}

export interface Run { z: Float64Array; V: Float64Array; soc: Float64Array; ch: Float64Array; dis: Float64Array; cl: Float64Array }

/** BatteryPilot over hours [a, b), starting empty: the network for every hour, then the offer rule against RT. */
export function runWindow(W: Pack<BPMeta>, p: Path, bat: Battery, a: number, b: number, net = makeNet(W)): Run {
  const K = W.meta.knots.length, n = b - a, knots = W.meta.knots;
  const x = new Float64Array(W.t.x_mu.data.length), zt = new Float64Array(K);
  const bb = W.meta.hybrid ? backbone(knots, p, bat, b, 200, W.meta.plan_lag_h ?? 0) : null;
  const r: Run = { z: new Float64Array(n * K), V: new Float64Array(n * K), soc: new Float64Array(n + 1), ch: new Float64Array(n), dis: new Float64Array(n), cl: new Float64Array(n) };
  for (let i = 0; i < n; i++) {
    const t = a + i, cl = bat.clin(p.T[t]);
    r.cl[i] = cl;
    features(W.meta, p, t, bat.d, cl, x);
    if (bb) for (let k = 0; k < K; k++) x[W.meta.n_feat + k] = Math.asinh(bb[t * K + k] / p.ps[t]);
    net(x, zt);
    for (let k = 0; k < K; k++) { r.z[i * K + k] = zt[k]; r.V[i * K + k] = Math.sinh(zt[k]) * p.ps[t]; }
    const [s, c, d] = stepOffer(knots, bat, r.soc[i], r.V.subarray(i * K, i * K + K), p.rt[t], cl);
    r.soc[i + 1] = s; r.ch[i] = c; r.dis[i] = d;
  }
  return r;
}

// ---- GridData's daily-spread rule (daily_spread) and settlement with rainflow wear (settle) -----------------------
export function dailySpread(p: Path, bat: Battery, a: number, b: number) {
  const n = b - a, ch = new Float64Array(n), dis = new Float64Array(n), k = Math.round(bat.d);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j < n && p.day[a + j] === p.day[a + i]) j++;
    const m = j - i;
    if (m >= 2 * k) {
      const pr = Array.from({ length: m }, (_, q) => p.da[a + i + q]);
      const idx = pr.map((_, q) => q);
      const cheap = [...idx].sort((u, v) => pr[u] - pr[v] || u - v).slice(0, k).sort((u, v) => u - v);
      const mx = cheap[cheap.length - 1];
      const later = [...idx].sort((u, v) => pr[v] - pr[u] || u - v).filter((q) => q > mx).slice(0, k);
      if (later.length === k) {
        const out = k * bat.P * bat.etac * bat.etad;
        let clm = 0;
        for (let q = 0; q < m; q++) clm += bat.clin(p.T[a + i + q]);
        clm /= m;
        const mL = later.reduce((s, q) => s + pr[q], 0) / k, mC = cheap.reduce((s, q) => s + pr[q], 0) / k;
        if (mL * out - clm * out > mC * k * bat.P) {
          for (const q of cheap) ch[i + q] = bat.P;
          for (const q of later) dis[i + q] = Math.min(out / k, bat.P);
        }
      }
    }
    i = j;
  }
  const soc = new Float64Array(n + 1);
  for (let q = 0; q < n; q++) soc[q + 1] = soc[q] + (bat.etac * ch[q] - dis[q] / bat.etad) / bat.E;
  return { ch, dis, soc };
}

/** Rainflow wear ($) of a SOC trace (fraction of usable), throughput-weighted temperature factor per cycle. */
export function wearCost(bat: Battery, soc: ArrayLike<number>, T: ArrayLike<number>, a: number) {
  if (bat.cost <= 0) return 0;
  const N = soc.length, x = new Float64Array(N);
  for (let i = 0; i < N; i++) x[i] = bat.lo + Math.min(Math.max(soc[i], 0), 1) * (bat.hi - bat.lo);
  const cthr = new Float64Array(N), cft = new Float64Array(N);
  for (let i = 0; i < N - 1; i++) { const d = Math.abs(x[i + 1] - x[i]); cthr[i + 1] = cthr[i] + d; cft[i + 1] = cft[i] + d * bat.tf(T[a + i]); }
  // turning points
  const ti: number[] = [0];
  let last = 0;
  for (let i = 1; i < N - 1; i++) {
    const d0 = x[i] - x[i - 1], d1 = x[i + 1] - x[i];
    if (d0 !== 0) last = d0;
    if (last * d1 < 0) ti.push(i);
  }
  if (N > 1) ti.push(N - 1);
  let cost = 0;
  const add = (r: number, cnt: number, i0: number, i1: number) => {
    if (!(r > 0)) return;
    const lo = Math.min(i0, i1), hi = Math.max(i0, i1), w = cthr[hi] - cthr[lo];
    const f = w > 1e-12 ? (cft[hi] - cft[lo]) / w : 1;
    cost += cnt * bat.phi(r) * f * 1000 * bat.cost * bat.Enom;
  };
  const st: number[] = [];
  for (const i of ti) {
    st.push(i);
    while (st.length >= 3) {
      const X = Math.abs(x[st[st.length - 1]] - x[st[st.length - 2]]), Y = Math.abs(x[st[st.length - 2]] - x[st[st.length - 3]]);
      if (X < Y) break;
      if (st.length === 3) { add(Y, 0.5, st[0], st[1]); st.shift(); }
      else { add(Y, 1.0, st[st.length - 3], st[st.length - 2]); const l = st.pop()!; st.pop(); st.pop(); st.push(l); }
    }
  }
  for (let q = 0; q < st.length - 1; q++) add(Math.abs(x[st[q + 1]] - x[st[q]]), 0.5, st[q], st[q + 1]);
  return cost;
}

export interface Settled { gross: number; wear: number; net: number; cycles: number }
export function settle(bat: Battery, p: Path, a: number, ch: ArrayLike<number>, dis: ArrayLike<number>, soc: ArrayLike<number>): Settled {
  let gross = 0, out = 0;
  for (let i = 0; i < ch.length; i++) { gross += p.rt[a + i] * (dis[i] - ch[i]); out += dis[i] / bat.etad; }
  const wear = wearCost(bat, soc, p.T, a);
  return { gross, wear, net: gross - wear, cycles: out / bat.E };
}

/** Parity vector: raw z knots at the given hours, then the state of charge over the window. */
export function parityVector(W: Pack<BPMeta>, p: Path, bat: Battery, hours: number[], win: [number, number]) {
  const net = makeNet(W), K = W.meta.knots.length, x = new Float64Array(W.t.x_mu.data.length), z = new Float64Array(K);
  const bb = W.meta.hybrid ? backbone(W.meta.knots, p, bat, Math.max(...hours) + 1, 200, W.meta.plan_lag_h ?? 0) : null;
  const out: number[] = [];
  for (const t of hours) {
    features(W.meta, p, t, bat.d, bat.clin(p.T[t]), x);
    if (bb) for (let k = 0; k < K; k++) x[W.meta.n_feat + k] = Math.asinh(bb[t * K + k] / p.ps[t]);
    net(x, z);
    out.push(...z);
  }
  const r = runWindow(W, p, bat, win[0], win[1], net);
  out.push(...r.soc);
  return out;
}
