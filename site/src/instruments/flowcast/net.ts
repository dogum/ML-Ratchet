// FlowCast in the browser: a pure port of models/flowcast/flowcast.py (climate statics from the forcing, static
// vector, daily features, one 64-unit LSTM run in 365-day blocks with a 365-day warm-up each, mean + 5 monotone
// quantiles, widen-only band factors), plus the what-ifs the instrument applies to the input.
// Parity cases: site/public/instruments/flowcast/reference.json (written by models/flowcast/web/export_instrument.py).
import type { Pack } from '../../lib/nn/pack.ts';

export const WARM = 365;
const fr = Math.fround;

export interface WeightsMeta { k_widen: number[]; rung: string; statics: string[] }
export interface Attrs {
  area_km2: number; elev_mean_m: number; elev_range_m: number; slope_mkm: number; forest_frac: number;
  soil_porosity: number; ksat_cmh: number; carbonate_frac: number; reservoir_storage_frac: number;
  irrigation_frac: number; urban_frac: number; elev_min_m: number; elev_max_m: number;
}
export interface BasinInfo {
  id: string; name: string; stratum: string; attrs: Attrs; start: string; end: string; forcing_start: string;
  n_target: number; lat: number | null; lon: number | null; decade_nse: Record<string, number>;
}
export interface BasinsMeta { basins: BasinInfo[]; step: number }

export interface Net {
  H: number; nin: number; nd: number; wih: Float64Array; whh: Float64Array; b: Float64Array;
  hw: Float64Array; hb: Float64Array; mu: Float64Array; sd: Float64Array; k50: number; k90: number; nowcast: boolean;
}

export function loadNet(p: Pack<WeightsMeta>): Net {
  const t = p.t;
  const H = t.w_hh.shape[1], nin = t.w_ih.shape[1];
  const b = new Float64Array(4 * H);
  for (let i = 0; i < 4 * H; i++) b[i] = t.b_ih.data[i] + t.b_hh.data[i];
  return {
    H, nin, nd: nin - 18, wih: Float64Array.from(t.w_ih.data), whh: Float64Array.from(t.w_hh.data), b,
    hw: Float64Array.from(t.head_w.data), hb: Float64Array.from(t.head_b.data),
    mu: Float64Array.from(t.st_mu.data), sd: Float64Array.from(t.st_sd.data),
    k50: p.meta.k_widen[0], k90: p.meta.k_widen[1], nowcast: nin - 18 === 14,
  };
}

/** One exam basin: forcing rows (P, PET, T_mean, T_min, T_max, SRAD, vp, wind) from 365 days before the first
 *  predicted day to one day after the last; day of year of each row; observed flow (mm/day) on the predicted days. */
export interface Basin { info: BasinInfo; F: Float64Array; rows: number; doy: Int32Array; obs: Float64Array }

export function basin(p: Pack<BasinsMeta>, i: number): Basin {
  const F = p.t['F' + i], info = p.meta.basins[i];
  return { info, F: Float64Array.from(F.data), rows: F.shape[0], doy: Int32Array.from(p.t['doy' + i].data),
    obs: Float64Array.from(p.t['obs' + i].data) };
}

// ---- what-ifs on the input ------------------------------------------------------------------
export interface Ops {
  pmul?: number; tshift?: number; petmul?: number;
  drop?: [number, number];          // P = 0 on forcing rows r0..r1
  storm?: [number, number];         // [row, mm]: 60 % on the row, 40 % on the next
  elev?: number; area?: number; forest?: number;
  freeze?: boolean;                 // climate statics and output scale from the recorded forcing
}

export function applyOps(b: Basin, ops: Ops): { F: Float64Array; attrs: Attrs } {
  const F = new Float64Array(b.F);
  const n = b.rows;
  const pm = ops.pmul ?? 1, dt = ops.tshift ?? 0, em = ops.petmul ?? 1;
  for (let r = 0; r < n; r++) {
    const o = r * 8;
    if (pm !== 1) F[o] = fr(F[o] * pm);
    if (dt !== 0) { F[o + 2] = fr(F[o + 2] + dt); F[o + 3] = fr(F[o + 3] + dt); F[o + 4] = fr(F[o + 4] + dt); }
    if (em !== 1) F[o + 1] = fr(F[o + 1] * em);
  }
  if (ops.drop) for (let r = Math.max(0, ops.drop[0]); r <= Math.min(n - 1, ops.drop[1]); r++) F[r * 8] = 0;
  if (ops.storm && ops.storm[1] !== 0) {
    const [r, mm] = ops.storm;
    if (r >= 0 && r + 1 < n) { F[r * 8] = fr(F[r * 8] + 0.6 * mm); F[(r + 1) * 8] = fr(F[(r + 1) * 8] + 0.4 * mm); }
  }
  const a = { ...b.info.attrs };
  if (ops.elev) { a.elev_mean_m += ops.elev; a.elev_min_m += ops.elev; a.elev_max_m += ops.elev; }
  if (ops.area !== undefined && ops.area !== 1) a.area_km2 *= ops.area;
  if (ops.forest !== undefined) a.forest_frac = ops.forest;
  return { F, attrs: a };
}

// ---- preprocessing, as flowcast.py ----------------------------------------------------------
export interface Clim { p_mean: number; pet_mean: number; aridity: number; frac_snow: number; hp_freq: number; lp_freq: number; t_mean: number }

/** climate_stats over every forcing row (the window being predicted, its warm-up and the extra day). */
export function climate(F: Float64Array, n: number): Clim {
  let sp = 0, se = 0, st = 0, ss = 0;
  for (let r = 0; r < n; r++) { const P = F[r * 8], T = F[r * 8 + 2]; sp += P; se += F[r * 8 + 1]; st += T; if (T < 0) ss += P; }
  const pm = Math.max(sp / n, 0.05), em = Math.max(se / n, 0.05);
  let hp = 0, lp = 0;
  for (let r = 0; r < n; r++) { const P = F[r * 8]; if (P > 5 * pm) hp++; if (P < 1.0) lp++; }
  return { p_mean: pm, pet_mean: em, aridity: em / pm, frac_snow: ss / Math.max(sp, 1e-6), hp_freq: hp / n, lp_freq: lp / n, t_mean: st / n };
}

/** static_vector(spec_attrs(attrs), clim) for attributes already in FlowCast units (as examples.json gives them). */
export function staticVector(a: Attrs, c: Clim): Float64Array {
  const range = Number.isFinite(a.elev_range_m) ? a.elev_range_m : (a.elev_max_m - a.elev_min_m);
  const v = [Math.log10(a.area_km2), a.elev_mean_m / 1000, range / 1000, Math.log10(a.slope_mkm), a.forest_frac,
    a.soil_porosity, Math.log10(a.ksat_cmh), a.carbonate_frac, Math.min(a.reservoir_storage_frac, 2.0),
    a.irrigation_frac, a.urban_frac, Math.log(c.p_mean), c.pet_mean, Math.log(c.aridity), c.frac_snow, c.hp_freq,
    c.lp_freq, c.t_mean / 10];
  return Float64Array.from(v, fr);
}

// float32 day-of-year features (dyn_features computes them in float32)
const TWO_PI = fr(2 * Math.PI), YEAR = fr(365.25);
const SIN = new Float64Array(367), COS = new Float64Array(367);
for (let d = 0; d <= 366; d++) { const w = fr(fr(TWO_PI * d) / YEAR); SIN[d] = fr(Math.sin(w)); COS[d] = fr(Math.cos(w)); }

/** The 9 weather features of every row (dyn_features without the day-of-year pair). */
function weather(F: Float64Array, n: number): Float64Array {
  const X = new Float64Array(n * 9);
  for (let r = 0; r < n; r++) {
    const o = r * 8, P = Math.max(F[o], 0), x = r * 9;
    X[x] = fr(P / 10); X[x + 1] = fr(Math.log1p(P)); X[x + 2] = fr(F[o + 1] / 5);
    X[x + 3] = fr(F[o + 2] / 10); X[x + 4] = fr(F[o + 3] / 10); X[x + 5] = fr(F[o + 4] / 10);
    X[x + 6] = fr(F[o + 5] / 150); X[x + 7] = fr(F[o + 6] / 1000); X[x + 8] = fr(F[o + 7] / 3);
  }
  return X;
}

const sig = (x: number) => 1 / (1 + Math.exp(-x));
const softplus = (x: number) => Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));

/** Everything one forward pass needs that does not change from block to block. */
export interface Prepared {
  X: Float64Array; doy: Int32Array; rows: number; zs: Float64Array; pm: number; clim: Clim; st: Float64Array;
  qn: Float64Array | null; lead: number;
}

export function prepare(net: Net, b: Basin, ops: Ops = {}, gauge?: { obs: Float64Array; lead: number }): Prepared {
  const { F, attrs } = applyOps(b, ops);
  const clim = climate(ops.freeze ? b.F : F, b.rows);
  const st = staticVector(attrs, clim);
  const H4 = 4 * net.H, zs = new Float64Array(H4);
  for (let r = 0; r < H4; r++) {
    let a = net.b[r];
    const o = r * net.nin + net.nd;
    for (let j = 0; j < 18; j++) a += net.wih[o + j] * ((st[j] - net.mu[j]) / net.sd[j]);
    zs[r] = a;
  }
  let qn: Float64Array | null = null;
  if (gauge && net.nowcast) {                       // observed flow / p_mean on every row (NaN where unknown)
    qn = new Float64Array(b.rows).fill(NaN);
    for (let t = 0; t < gauge.obs.length; t++) qn[WARM + t] = gauge.obs[t] / clim.p_mean;
  }
  return { X: weather(F, b.rows), doy: b.doy, rows: b.rows, zs, pm: clim.p_mean, clim, st, qn, lead: gauge?.lead ?? 1 };
}

/** Run the block that starts at target day a for WARM + nOut steps; raw (unwidened, unscaled) outputs for
 *  target days a .. a + nOut - 1. */
function block(net: Net, p: Prepared, a: number, nOut: number) {
  const { H, nin, nd, wih, whh, hw, hb } = net;
  const H4 = 4 * H, L = WARM + nOut;
  const h = new Float64Array(H), c = new Float64Array(H), g = new Float64Array(H4), x = new Float64Array(nd);
  const mean = new Float64Array(nOut), q = new Float64Array(nOut * 5);
  const d0 = p.doy[a] - 1, lead = p.lead, lf = fr(lead / 7);
  for (let t = 0; t < L; t++) {
    const r = a + t, o = r * 9;
    for (let j = 0; j < 9; j++) x[j] = p.X[o + j];
    const d = ((d0 + t) % 365) + 1;
    x[9] = SIN[d]; x[10] = COS[d];
    if (nd === 14) {
      const lag = p.qn && t >= lead ? p.qn[r - lead] : NaN;
      const ok = Number.isFinite(lag);
      x[11] = ok ? fr(Math.log1p(Math.max(lag, 0) * 5)) : 0; x[12] = ok ? 1 : 0; x[13] = lf;
    }
    for (let k = 0; k < H4; k++) {
      let s = p.zs[k];
      const oi = k * nin, oh = k * H;
      for (let j = 0; j < nd; j++) s += wih[oi + j] * x[j];
      for (let j = 0; j < H; j++) s += whh[oh + j] * h[j];
      g[k] = s;
    }
    for (let j = 0; j < H; j++) {
      const ig = sig(g[j]), fg = sig(g[H + j]), gg = Math.tanh(g[2 * H + j]), og = sig(g[3 * H + j]);
      c[j] = fg * c[j] + ig * gg;
      h[j] = og * Math.tanh(c[j]);
    }
    if (t >= WARM) {
      const i = t - WARM;
      const out = [0, 0, 0, 0, 0, 0];
      for (let k = 0; k < 6; k++) { let s = hb[k]; for (let j = 0; j < H; j++) s += hw[k * H + j] * h[j]; out[k] = s; }
      mean[i] = softplus(out[0]);
      let acc = softplus(out[1]) * 0.2;
      q[i * 5] = acc;
      for (let k = 2; k < 6; k++) { acc += softplus(out[k]); q[i * 5 + k - 1] = acc; }
    }
  }
  return { mean, q };
}

export interface Forecast { mean: Float64Array; q: Float64Array; pm: number; clim: Clim; st: Float64Array }

/** widen() about the median, clip at 0, then scale by the mean precipitation (mm/day). */
function finish(net: Net, mean: Float64Array, q: Float64Array, pm: number) {
  const n = mean.length;
  for (let i = 0; i < n; i++) {
    const o = i * 5, m = q[o + 2];
    const q0 = m - net.k90 * (m - q[o]), q1 = m - net.k50 * (m - q[o + 1]);
    const q3 = m + net.k50 * (q[o + 3] - m), q4 = m + net.k90 * (q[o + 4] - m);
    q[o] = Math.max(q0, 0) * pm; q[o + 1] = Math.max(q1, 0) * pm; q[o + 2] = Math.max(m, 0) * pm;
    q[o + 3] = Math.max(q3, 0) * pm; q[o + 4] = Math.max(q4, 0) * pm;
    mean[i] *= pm;
  }
}

/** FlowCast.predict over every target day of the basin (blocks of 365 days, each with its own warm-up). */
export function predict(net: Net, p: Prepared, nTarget: number): Forecast {
  const mean = new Float64Array(nTarget), q = new Float64Array(nTarget * 5);
  for (let a = 0; a < nTarget; a += 365) {
    const n = Math.min(365, nTarget - a);
    const r = block(net, p, a, n);
    mean.set(r.mean, a); q.set(r.q, a * 5);
  }
  finish(net, mean, q, p.pm);
  return { mean, q, pm: p.pm, clim: p.clim, st: p.st };
}

/** One target day's forecast only (runs its block up to that day): the same number predict() gives for it. */
export function predictDay(net: Net, p: Prepared, day: number) {
  const a = Math.floor(day / 365) * 365;
  const r = block(net, p, a, day - a + 1);
  const mean = r.mean.slice(-1), q = r.q.slice(-5);
  finish(net, mean, q, p.pm);
  return { mean: mean[0], q: Array.from(q) };
}

// ---- scores ---------------------------------------------------------------------------------
export function scores(obs: ArrayLike<number>, sim: ArrayLike<number>, lo?: ArrayLike<number>, hi?: ArrayLike<number>, i0 = 0, i1 = obs.length) {
  let n = 0, so = 0, ss = 0;
  for (let i = i0; i < i1; i++) if (Number.isFinite(obs[i]) && Number.isFinite(sim[i])) { n++; so += obs[i]; ss += sim[i]; }
  if (n < 3) return { nse: NaN, kge: NaN, ratio: NaN, cover: NaN, n };
  const mo = so / n, ms = ss / n;
  let num = 0, den = 0, vo = 0, vs = 0, cov = 0, inside = 0;
  for (let i = i0; i < i1; i++) {
    if (!(Number.isFinite(obs[i]) && Number.isFinite(sim[i]))) continue;
    const e = sim[i] - obs[i];
    num += e * e; den += (obs[i] - mo) ** 2;
    vo += (obs[i] - mo) ** 2; vs += (sim[i] - ms) ** 2; cov += (obs[i] - mo) * (sim[i] - ms);
    if (lo && hi && obs[i] >= lo[i] && obs[i] <= hi[i]) inside++;
  }
  const r = cov / Math.sqrt(vo * vs || 1e-12), alpha = Math.sqrt(vs / (vo || 1e-12)), beta = ms / (mo || 1e-12);
  return { nse: 1 - num / (den || 1e-12), kge: 1 - Math.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2),
    ratio: beta, cover: lo && hi ? inside / n : NaN, n };
}

/** The p1 water-yield prior (training/src/simbank.p_multiplier without its scatter): the simulator's true
 *  precipitation is ERA5-Land x m, while the network reads ERA5-Land. */
export function waterYield(a: Attrs, c: Clim) {
  const z = -0.088 + 0.058 * a.elev_mean_m / 1000 + 0.153 * c.frac_snow + 0.397 * Math.log(c.p_mean / 3);
  return Math.min(2.5, Math.max(0.4, Math.exp(z)));
}
