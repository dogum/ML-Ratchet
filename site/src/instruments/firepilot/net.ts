// FirePilot in the browser: a port of firepilot.py (Frame, line_cells, build_times, Context.scenario_features,
// Context.plan_eval and ValueNet), the value network that ranks containment-line plans, and the planner's search.
// It follows numpy's arithmetic where firepilot.py relies on it: float32 line costs and their cumulative sum
// (float32 cost raster x a Python float), numpy's pairwise sums, numpy's float remainder, degrees and radians.
// The search (planBrowser) uses the kit's seeded random numbers, not numpy's, so its pick can differ from Python's;
// everything it scores is checked against Python in reference.json.
import type { Pack } from '../../lib/nn/pack.ts';

export const HORIZON = 72.0;
export const FL_HOLD = 2.4;
const D_MIN = 90.0, D_MAX = 8000.0;
const DEG = 180.0 / Math.PI, RAD = Math.PI / 180.0;
const f32 = Math.fround;

export interface Plan { mode: number; phi_c: number; omega: number; d_h: number; d_f: number }
export const BASELINES: Record<string, Plan> = {
  direct_head: { mode: 0, phi_c: 0, omega: 90, d_h: 90, d_f: 90 },
  indirect_1km: { mode: 0, phi_c: 0, omega: 90, d_h: 1000, d_f: 1000 },
  indirect_2km: { mode: 0, phi_c: 0, omega: 90, d_h: 2000, d_f: 2000 },
  indirect_4km: { mode: 0, phi_c: 0, omega: 90, d_h: 4000, d_f: 4000 },
  anchor_flank: { mode: 1, phi_c: 0, omega: 180, d_h: 90, d_f: 90 },
};

export interface BudgetMeta { firepilot_plan: Plan; truth_saved: Record<string, number> }
export interface FireMeta {
  id: string; name: string; period: string; k: number; res: number; head: number; ny: number; nx: number;
  a0_ha: number; forecast_none_ha: number; budgets: Record<string, BudgetMeta>;
}
export interface FiresMeta {
  fires: FireMeta[]; cost_classes: number[]; exam_means: Record<string, Record<string, number>>;
  horizon_h: number; fl_hold_m: number;
}
export interface NetMeta { mu: number[]; sd: number[]; sig_scale: number; params: number; version: string }

export interface Fire {
  meta: FireMeta; ny: number; nx: number; res: number; head: number;
  burned0: Uint8Array; arr: Float64Array; fl: Float64Array; cost: Float64Array; clockT: Float64Array; clockW: Float64Array;
}

/** One example fire, as Context.from_store reads it (arrival >= 9e3 h means never). */
export function fire(p: Pack<FiresMeta>, i: number): Fire {
  const m = p.meta.fires[i], t = p.t, cls = p.meta.cost_classes;
  const arr = Float64Array.from(t[`a${i}`].data, (v) => (v >= 9e3 ? NaN : v));
  return {
    meta: m, ny: m.ny, nx: m.nx, res: m.res, head: m.head,
    burned0: Uint8Array.from(t[`b${i}`].data), arr, fl: Float64Array.from(t[`f${i}`].data),
    cost: Float64Array.from(t[`c${i}`].data, (k) => cls[k]),
    clockT: Float64Array.from(t[`t${i}`].data), clockW: Float64Array.from(t[`w${i}`].data),
  };
}

// ---- numpy arithmetic ----------------------------------------------------------------------------
/** numpy / Python float remainder for b > 0 (fmod, then shifted into [0, b)). */
function rem(a: number, b: number) {
  const m = a % b;
  return m ? (m < 0 ? m + b : m) : 0;
}
export const wrap180 = (a: number) => rem(a + 180.0, 360.0) - 180.0;
const sq = (x: number) => x * x;

/** numpy's pairwise summation (add.reduce on a contiguous 1-D array); `single` accumulates in float32. */
function pairwise(a: ArrayLike<number>, i0: number, n: number, single: boolean): number {
  const r = single ? f32 : (x: number) => x;
  if (n < 8) {
    let s = 0;
    for (let i = 0; i < n; i++) s = r(s + a[i0 + i]);
    return s;
  }
  if (n <= 128) {
    const p = [0, 1, 2, 3, 4, 5, 6, 7].map((j) => a[i0 + j]);
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) p[j] = r(p[j] + a[i0 + i + j]);
    let s = r(r(r(p[0] + p[1]) + r(p[2] + p[3])) + r(r(p[4] + p[5]) + r(p[6] + p[7])));
    for (; i < n; i++) s = r(s + a[i0 + i]);
    return s;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return r(pairwise(a, i0, n2, single) + pairwise(a, i0 + n2, n - n2, single));
}
const sum32 = (a: ArrayLike<number>) => pairwise(a, 0, a.length, true);
const sum64 = (a: ArrayLike<number>) => pairwise(a, 0, a.length, false);

/** Exact Euclidean distance (in cells) to the nearest true cell (Felzenszwalb & Huttenlocher), as scipy's EDT. */
function edt(mask: Uint8Array, ny: number, nx: number) {
  const N = Math.max(ny, nx), INF = 1e20;
  const f = new Float64Array(N), d = new Float64Array(N), v = new Int32Array(N), z = new Float64Array(N + 1);
  const g = new Float64Array(ny * nx);
  const dt1 = (n: number) => {
    let k = 0; v[0] = 0; z[0] = -INF; z[1] = INF;
    for (let q = 1; q < n; q++) {
      let s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
      while (s <= z[k]) { k--; s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]); }
      k++; v[k] = q; z[k] = s; z[k + 1] = INF;
    }
    k = 0;
    for (let q = 0; q < n; q++) { while (z[k + 1] < q) k++; d[q] = (q - v[k]) * (q - v[k]) + f[v[k]]; }
  };
  for (let c = 0; c < nx; c++) {
    for (let r = 0; r < ny; r++) f[r] = mask[r * nx + c] ? 0 : INF;
    dt1(ny);
    for (let r = 0; r < ny; r++) g[r * nx + c] = d[r];
  }
  for (let r = 0; r < ny; r++) {
    for (let c = 0; c < nx; c++) f[c] = g[r * nx + c];
    dt1(nx);
    for (let c = 0; c < nx; c++) g[r * nx + c] = Math.sqrt(d[c]);
  }
  return g;
}

export interface Lines { idx: number[]; tb: Float64Array; km: number; dl: number[]; n4: number }
export interface PlanEval {
  s_proxy: number; s_proxy3: number; km: number; f: number[]; L: Lines;
  won: Uint8Array; hold: Uint8Array; shadow: number;
}

/** What the planner knows about one fire at one budget (firepilot.Context.from_store). */
export class Context {
  readonly n: number; readonly ny: number; readonly nx: number; readonly res: number; readonly rate: number;
  readonly fire: Fire; readonly D: Float64Array; readonly Phi: Float64Array; readonly fburn: Uint8Array;
  readonly a0: number; readonly af: number; readonly gf: number; readonly scn: number[];
  private byD: Uint32Array; private Ds: Float64Array; private fbList: Uint32Array;

  constructor(fire: Fire, rate: number) {
    const { ny, nx, res, burned0, arr } = fire;
    this.fire = fire; this.ny = ny; this.nx = nx; this.res = res; this.rate = rate;
    const n = (this.n = ny * nx);
    // Frame: distance from the burned area (m, scipy's EDT of ~burned0), angle about its centroid from the head (deg)
    this.D = edt(burned0, ny, nx);
    for (let i = 0; i < n; i++) this.D[i] *= res;
    let sr = 0, sc = 0, nb = 0;
    for (let i = 0; i < n; i++) if (burned0[i]) { sr += Math.floor(i / nx); sc += i % nx; nb++; }
    const r0 = sr / nb, c0 = sc / nb;
    this.Phi = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const r = Math.floor(i / nx), c = i % nx;
      const az = Math.atan2(c - c0, -(r - r0)) * DEG;
      this.Phi[i] = rem(az - fire.head + 180.0, 360.0) - 180.0;
    }
    this.fburn = new Uint8Array(n);
    let af = 0;
    const fb: number[] = [];
    for (let i = 0; i < n; i++) { const a = arr[i]; if (a === a && a <= HORIZON) { this.fburn[i] = 1; af++; fb.push(i); } }
    this.fbList = Uint32Array.from(fb);
    this.a0 = nb; this.af = af; this.gf = Math.max(af - nb, 1);
    this.byD = Uint32Array.from({ length: n }, (_, i) => i).sort((a, b) => this.D[a] - this.D[b] || a - b);
    this.Ds = Float64Array.from(this.byD, (i) => this.D[i]);
    this.scn = this.scenarioFeatures();
  }

  private scenarioFeatures(): number[] {
    const { n, ny, nx, res, gf } = this, { arr, burned0, fl, cost, clockW } = this.fire;
    const cell = (res * res) / 1e4;
    const g = [12, 24, 48].map((h) => { let k = 0; for (let i = 0; i < n; i++) if (!burned0[i] && arr[i] === arr[i] && arr[i] <= h) k++; return k / gf; });
    const fln: number[] = [], cn: number[] = [];
    let edge = 0;
    for (let i = 0; i < n; i++) {
      if (this.fburn[i] && !burned0[i]) { fln.push(fl[i]); cn.push(cost[i]); }
      const r = Math.floor(i / nx), c = i % nx;
      if (this.fburn[i] && (r === 0 || r === ny - 1 || c === 0 || c === nx - 1)) edge = 1;
    }
    let over = 0;
    for (const v of fln) if (v > FL_HOLD) over++;
    const srt = fln.slice().sort((a, b) => a - b), m = srt.length;
    const med = m ? (m % 2 ? srt[(m - 1) / 2] : (srt[m / 2 - 1] + srt[m / 2]) / 2) : 0;
    const req0 = Math.sqrt((this.a0 * (res * res)) / Math.PI), req1 = Math.sqrt((this.af * (res * res)) / Math.PI);
    const Wend = clockW[clockW.length - 1];
    return [Math.log10(this.a0 * cell + 1), Math.log10((this.af - this.a0) * cell + 1), g[0], g[1], g[2],
      Math.log10(this.rate), Wend / 36.0, (req1 - req0) / HORIZON / 100.0,
      m ? over / m : 0.0, m ? Math.log10(med + 0.1) : -1.0, edge,
      cn.length ? f32(sum32(cn) / cn.length) : 1.0,
      Math.log10((this.rate * Wend) / Math.max(2 * Math.PI * req1, 1.0) + 1e-3)];
  }

  private dOf(i: number, p: Plan) {
    const dl = wrap180(this.Phi[i] - p.phi_c);
    const w = sq((1.0 + Math.cos(dl * RAD)) / 2.0);
    return Math.max(p.d_f + (p.d_h - p.d_f) * w, this.res);
  }

  /** firepilot.line_cells + build_times: the 8-connected line, its cells' build completion times (h) and km built. */
  lines(p: Plan): Lines {
    const { ny, nx, res, D, Phi } = this, { burned0, cost, clockT: T, clockW: W } = this.fire;
    const memo = new Int8Array(this.n);                       // 0 unknown, 1 outside the plan region, 2 inside
    const isOut = (i: number) => { let v = memo[i]; if (!v) { v = D[i] >= this.dOf(i, p) ? 1 : 2; memo[i] = v; } return v === 1; };
    // only cells with D within [min offset, max offset + 2 cells] can be on the line: scan them in order of D
    const dmax = Math.max(p.d_h, p.d_f, res) + 2.0 * res, dmin = Math.max(Math.min(p.d_h, p.d_f), res) - 1e-9;
    let hi = 0, top = this.Ds.length;
    while (hi < top) { const k = (hi + top) >> 1; if (this.Ds[k] <= dmax) hi = k + 1; else top = k; }
    let lo = 0, b = hi;
    while (lo < b) { const k = (lo + b) >> 1; if (this.Ds[k] < dmin) lo = k + 1; else b = k; }
    const idx: number[] = [];
    let n4 = 0;
    for (let q = lo; q < hi; q++) {
      const i = this.byD[q], r = Math.floor(i / nx), c = i % nx;
      if (burned0[i] || !isOut(i)) continue;
      if (Math.abs(wrap180(Phi[i] - p.phi_c)) > p.omega) continue;
      let any8 = false, any4 = false;
      for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
        if (!dr && !dc) continue;
        const rr = r + dr, cc = c + dc;
        if (rr < 0 || rr >= ny || cc < 0 || cc >= nx) continue;
        if (!isOut(rr * nx + cc)) { any8 = true; if (!dr || !dc) any4 = true; }
      }
      if (any4) n4++;
      if (any8 && r > 0 && r < ny - 1 && c > 0 && c < nx - 1) idx.push(i);
    }
    idx.sort((a, b2) => a - b2);
    const L = idx.length, tb = new Float64Array(L).fill(Infinity);
    if (!L) return { idx, tb, km: 0, dl: [], n4 };
    const cellLen = f32(((res * (Math.PI / 4.0)) * n4) / Math.max(L, 1));
    const dl = idx.map((i) => wrap180(Phi[i] - p.phi_c));
    const cst = idx.map((i) => f32(cost[i] * cellLen));           // float32 x float32, as numpy does
    const r2 = f32(this.rate / 2.0), Wend = W[W.length - 1];
    for (const side of [true, false]) {
      const s: number[] = [];
      for (let k = 0; k < L; k++) if ((dl[k] >= 0) === side) s.push(k);
      const key = (k: number) => (p.mode === 0 ? Math.abs(dl[k]) : -Math.abs(dl[k]));
      s.sort((a, b2) => key(a) - key(b2) || a - b2);               // a stable argsort
      let cum = 0;
      for (const k of s) {
        cum = f32(cum + cst[k]);                                   // float32 cumsum
        const w = f32(cum / r2);
        let t = Infinity;
        if (w <= Wend) { let a = 0, e = W.length; while (a < e) { const m = (a + e) >> 1; if (W[m] < w) a = m + 1; else e = m; } t = T[Math.min(a, T.length - 1)]; }
        tb[k] = t > HORIZON ? Infinity : t;
      }
    }
    const built: number[] = [];
    for (let k = 0; k < L; k++) if (Number.isFinite(tb[k])) built.push(cst[k]);
    return { idx, tb, km: f32(sum32(built) / 1000.0), dl, n4 };
  }

  /** Context.plan_eval: the 18 plan-vs-forecast features and the connectivity proxies. */
  planEval(p: Plan): PlanEval {
    const L = this.lines(p), { idx, tb } = L, nl = idx.length;
    if (!nl) return { s_proxy: 0, s_proxy3: 0, km: 0, f: new Array(18).fill(0), L, won: new Uint8Array(0), hold: new Uint8Array(0), shadow: 0 };
    const { n, nx, ny } = this, { arr, fl, burned0 } = this.fire;
    let builtN = 0, wonR = 0, reachedN = 0, reachNotHold = 0, effN = 0, tfin = -Infinity;
    const eff = new Uint8Array(nl), eff3 = new Uint8Array(nl), won = new Uint8Array(nl), hold = new Uint8Array(nl);
    for (let k = 0; k < nl; k++) {
      let a = arr[idx[k]];
      if (!(a === a)) a = 1e9;
      const b = Number.isFinite(tb[k]), hd = fl[idx[k]] <= FL_HOLD, reached = a <= HORIZON;
      const w = b && tb[k] < a, w3 = b && tb[k] < a - 3.0;
      if (b) { builtN++; tfin = Math.max(tfin, tb[k]); }
      if (w && reached) wonR++;
      if (reached) reachedN++;
      if (reached && !hd) reachNotHold++;
      won[k] = w ? 1 : 0; hold[k] = hd ? 1 : 0;
      eff[k] = w && hd ? 1 : 0; eff3[k] = w3 && hd ? 1 : 0;
      if (eff[k]) effN++;
    }
    if (!builtN) tfin = HORIZON;
    // area of the forecast burn still 8-connected to the detected fire once the effective line cells block it
    const sp = [eff, eff3].map((e) => {
      const blk = new Uint8Array(n);
      for (let k = 0; k < nl; k++) if (e[k]) blk[idx[k]] = 1;
      const seen = new Uint8Array(n), st: number[] = [];
      let cnt = 0;
      for (const i of this.fbList) if (burned0[i] && !blk[i]) { seen[i] = 1; st.push(i); }
      while (st.length) {
        const i = st.pop()!;
        cnt++;
        const r = Math.floor(i / nx), c = i % nx;
        for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
          const rr = r + dr, cc = c + dc;
          if (rr < 0 || rr >= ny || cc < 0 || cc >= nx) continue;
          const j = rr * nx + cc;
          if (!seen[j] && this.fburn[j] && !blk[j]) { seen[j] = 1; st.push(j); }
        }
      }
      const ap = cnt || this.a0;
      return Math.min(Math.max((this.af - ap) / this.gf, 0), 1);
    });
    // the line's shadow: forecast growth beyond the plan contour inside the sector, and how late the fire gets there
    const shT: number[] = [];
    for (const i of this.fbList) {
      const dl = wrap180(this.Phi[i] - p.phi_c);
      if (this.D[i] >= this.dOf(i, p) && Math.abs(dl) <= p.omega) shT.push(arr[i]);
    }
    const sSh = shT.length / this.gf, tSh = shT.length ? sum64(shT) / shT.length / HORIZON : 1.0, eFrac = effN / Math.max(nl, 1);
    const f = [p.mode, Math.cos(p.phi_c * RAD), Math.sin(p.phi_c * RAD), p.omega / 180.0,
      Math.log10(p.d_h) - 3, Math.log10(p.d_f) - 3, Math.log10(L.km + 0.1), builtN / nl,
      wonR / Math.max(reachedN, 1), reachNotHold / nl, reachedN / nl, tfin / HORIZON, sSh, tSh, sSh * eFrac, eFrac,
      sp[0], sp[1]];
    return { s_proxy: sp[0], s_proxy3: sp[1], km: L.km, f, L, won, hold, shadow: shT.length };
  }

  features(p: Plan) { const pe = this.planEval(p); return { x: this.scn.concat(pe.f), pe }; }
}

// ---- the value network ---------------------------------------------------------------------------
export interface ValueNet { W: Float32Array[]; b: Float32Array[]; dims: number[][]; mu: number[]; sd: number[]; sigScale: number; params: number }

export function valueNet(p: Pack<NetMeta>): ValueNet {
  const t = p.t;
  return {
    W: [0, 1, 2].map((i) => t[`W${i}`].data), b: [0, 1, 2].map((i) => t[`b${i}`].data),
    dims: [0, 1, 2].map((i) => t[`W${i}`].shape), mu: p.meta.mu, sd: p.meta.sd, sigScale: p.meta.sig_scale, params: p.meta.params,
  };
}

/** ValueNet.__call__ for one plan: mean predicted share of the no-action growth stopped, and its sigma. */
export function predict(net: ValueNet, x: number[]) {
  let h = x.map((v, i) => (v - net.mu[i]) / net.sd[i]);
  for (let l = 0; l < 3; l++) {
    const W = net.W[l], b = net.b[l], [nin, nout] = net.dims[l], o = new Array<number>(nout);
    for (let j = 0; j < nout; j++) {
      let s = 0;
      for (let i = 0; i < nin; i++) s += h[i] * W[i * nout + j];
      s += b[j];
      o[j] = l < 2 ? Math.tanh(s) : s;
    }
    h = o;
  }
  const mean = 1.0 / (1.0 + Math.exp(-h[0]));
  const sig = Math.exp(Math.min(Math.max(h[1], -6), 2));
  return { mean, sigma: Math.sqrt(sig * sig) * net.sigScale };
}

// ---- the search ------------------------------------------------------------------------------------
type Rng = { u: () => number; n: () => number };

function samplePlan(r: Rng): Plan {
  const mode = r.u() < 0.5 ? 1 : 0;
  const phi = wrap180(r.u() < 0.75 ? 50 * r.n() : r.u() * 360 - 180);
  const omega = 15 + r.u() * 165, lo = Math.log(D_MIN), hi = Math.log(D_MAX);
  const d_h = Math.exp(lo + r.u() * (hi - lo)), d_f = r.u() < 0.6 ? Math.exp(lo + r.u() * (hi - lo)) : d_h;
  return { mode, phi_c: phi, omega, d_h, d_f };
}

/** firepilot.candidate_set: the 5 baselines, the 40-plan grid and n prior draws. */
export function candidates(r: Rng, n: number): Plan[] {
  const c: Plan[] = Object.values(BASELINES).map((p) => ({ ...p }));
  for (const d of [500, 1000, 2000, 3000, 5000]) for (const om of [60, 90, 120, 180]) for (const mode of [0, 1]) c.push({ mode, phi_c: 0, omega: om, d_h: d, d_f: d });
  for (let k = 0; k < n; k++) c.push(samplePlan(r));
  return c;
}

const clampLog = (v: number) => Math.exp(Math.min(Math.max(v, Math.log(D_MIN)), Math.log(D_MAX)));

/** firepilot.plan_value with the same search sizes (205 candidates, then 2 CEM rounds of 48 around the top 6 on the
 * network) but the kit's seeded random numbers instead of numpy's, so the pick can differ from Python's. */
export function planBrowser(ctx: Context, net: ValueNet, r: Rng, n = 160, refine = 2, nRef = 48, top = 6) {
  const scored = candidates(r, n).map((p) => ({ p, ...predict(net, ctx.features(p).x) }));
  for (let it = 0; it < refine; it++) {
    const best = scored.slice().sort((a, b) => b.mean - a.mean).slice(0, top);   // the top 6 of everything scored so far
    for (const b of best) for (let k = 0; k < nRef / top; k++) {
      const v = b.p, mode = r.u() < 0.85 ? v.mode : 1 - v.mode;
      const q: Plan = { mode, phi_c: wrap180(v.phi_c + 25 * r.n()), omega: Math.min(Math.max(v.omega + 20 * r.n(), 5), 180),
        d_h: clampLog(Math.log(v.d_h) + 0.35 * r.n()), d_f: clampLog(Math.log(v.d_f) + 0.35 * r.n()) };
      scored.push({ p: q, ...predict(net, ctx.features(q).x) });
    }
  }
  scored.sort((a, b) => b.mean - a.mean);
  return { best: scored[0], n: scored.length };
}
