// SunDoctor in the browser: a line-for-line port of models/sundoctor/sundoctor.py — channels() (the 12 input channels,
// including expected_power), SunDoctorNet.forward() and predict() — plus the instrument's input faults (inject()).
import { conv1d, conv2d, gelu_, add_, linear, vgelu, concat0 } from '../../lib/nn/ops.ts';
import type { Pack } from '../../lib/nn/pack.ts';
import type { Tensor } from '../../lib/nn/tensor.ts';

export const D = 14, T = 96, N = D * T;

export interface SMeta {
  conv2d: [number, number][][][]; tconv: number[]; stem: [number, number][];
  reg: string[]; scale: number[]; glob: string[]; gscale: number[]; days: number;
}
export interface SWindow {
  farm: 'ca' | 'co'; label: string; system: string; name: string; desc: string; dates: string[];
  dc_kw: number; mount: 'fixed' | 'tracker'; n: number; unit: string; ceiling: number; ceiling_note: string;
  ref: { poa: number | null; home: number | null };
  key: { dead?: (number | null)[]; snow?: (number | null)[]; stuck?: (number | null)[]; outage?: (number | null)[] };
}
/** One fortnight's arrays, each [14 days × 96 solar-time steps] (power and irradiance per kW of nameplate / kW m⁻²). */
export interface SData { p: Float32Array; pu: Float32Array; g: Float32Array; ta: Float32Array; cz: Float32Array; ca: Float32Array; gc: Float32Array }

/** clean, healthy, unclipped AC output per kW of DC (PVWatts + ASHRAE IAM + Faiman), as sundoctor.expected_power(). */
export function expectedPower(g: number, ta: number, ca: number): number {
  const m = Math.min(1, Math.max(0, 1 - 0.05 * (1 / Math.min(1, Math.max(ca, 1e-3)) - 1)));
  const geff = g * 1000 * (0.25 + 0.75 * m), tc = ta + (g * 1000) / (25 + 6.84 * 2);
  const pdc = (geff / 1000) * (1 - 0.0037 * (tc - 25));
  if (!(pdc > 0)) return 0;
  const z = Math.max(pdc, 1e-6);
  const eta = (0.96 / 0.9637) * (-0.0162 * z - 0.0059 / z + 0.9858);
  return Math.max(eta * pdc, 0);
}

/** The 12 input channels [12, 14, 96], exactly as sundoctor.channels(). */
export function channels(p: ArrayLike<number>, s: SData, ref: number | null, home: boolean): Tensor {
  const x = new Float32Array(12 * N);
  const has = ref != null && Number.isFinite(ref) && ref > 0, r = has ? ref! : 0;
  for (let i = 0; i < N; i++) {
    const pi = p[i], g = s.g[i], t0 = Number.isFinite(s.ta[i]) ? s.ta[i] : 15, ca = s.ca[i];
    const pc = expectedPower(s.gc[i], t0, ca);
    const valid = home ? Number.isFinite(pi) : Number.isFinite(pi) && Number.isFinite(g);
    const g0 = home || !valid ? 0 : g;
    x[i] = valid ? pi : 0;
    x[N + i] = g0;
    x[2 * N + i] = home ? 0 : 1;
    x[3 * N + i] = s.gc[i];
    x[4 * N + i] = valid ? (home ? pc : expectedPower(Math.max(g0, 0), t0, ca)) : 0;
    x[5 * N + i] = pc;
    x[6 * N + i] = t0 / 40;
    x[7 * N + i] = s.cz[i];
    x[8 * N + i] = ca;
    x[9 * N + i] = valid ? 1 : 0;
    x[10 * N + i] = r;
    x[11 * N + i] = has ? 1 : 0;
  }
  return { data: x, shape: [12, D, T] };
}

export interface Faults { ceil?: number | null; kill?: number | null; snow?: number | null }

/**
 * The instrument's faults, on the input (models/sundoctor/web/export_instrument.py inject() is the same):
 * kill — from that day one of the n strings (combiners) is dead: the unclipped power ×(n−1)/n, then the inverter clips it;
 * ceil — a smaller inverter: output clipped at this fraction of nameplate;
 * snow — that day buried all day (zero), the next day until solar noon, when it slides off.
 */
export function inject(s: SData, n: number, native: number, f: Faults): Float32Array {
  const x = new Float32Array(s.p);
  if (f.kill != null) for (let i = f.kill * T; i < N; i++) x[i] = Math.min(s.pu[i] * ((n - 1) / n), native);
  if (f.ceil != null) for (let i = 0; i < N; i++) x[i] = Math.min(x[i], f.ceil);
  if (f.snow != null) {
    for (let i = f.snow * T; i < (f.snow + 1) * T; i++) if (Number.isFinite(x[i])) x[i] = 0;
    if (f.snow + 1 < D) for (let i = (f.snow + 1) * T; i < (f.snow + 1) * T + 48; i++) if (Number.isFinite(x[i])) x[i] = 0;
  }
  return x;
}

/** SunDoctorNet.forward(): raw outputs, day [19 × 14] row-major then the 4 fortnight-wide numbers. */
export function forward(W: Pack<SMeta>, x: Tensor): Float32Array {
  const t = W.t, M = W.meta;
  const [ss, sp] = M.stem;
  let h = gelu_(conv2d(x, t.s_w, t.s_b, ss, sp));
  M.conv2d.forEach((cfg, i) => {
    const [[s1, p1], [s2, p2], [s3, p3]] = cfg;
    let y = gelu_(conv2d(h, t[`b${i}c1_w`], t[`b${i}c1_b`], s1, p1));
    y = conv2d(y, t[`b${i}c2_w`], t[`b${i}c2_b`], s2, p2);
    h = gelu_(add_(y, conv2d(h, t[`b${i}sk_w`], t[`b${i}sk_b`], s3, p3)));
  });
  // mean and max over time of day -> [2C, days]
  const [C, H, Wd] = h.shape;
  const mm = new Float32Array(2 * C * H);
  for (let c = 0; c < C; c++) for (let r = 0; r < H; r++) {
    let s = 0, m = -Infinity;
    const b = (c * H + r) * Wd;
    for (let j = 0; j < Wd; j++) { const v = h.data[b + j]; s += v; if (v > m) m = v; }
    mm[c * H + r] = s / Wd; mm[(C + c) * H + r] = m;
  }
  let z = gelu_(conv1d({ data: mm, shape: [2 * C, H] }, t.p_w, t.p_b));
  M.tconv.forEach((dil, i) => {
    const y = gelu_(conv1d(z, t[`t${i}c1_w`], t[`t${i}c1_b`], 1, 'same', dil));
    z = gelu_(add_(conv1d(y, t[`t${i}c2_w`], t[`t${i}c2_b`]), z));
  });
  const day = conv1d(z, t.d_w, t.d_b);
  const Z = z.shape[0], L = z.shape[1];
  const gv = new Float32Array(2 * Z);
  for (let c = 0; c < Z; c++) {
    let s = 0, m = -Infinity;
    for (let j = 0; j < L; j++) { const v = z.data[c * L + j]; s += v; if (v > m) m = v; }
    gv[c] = s / L; gv[Z + c] = m;
  }
  const g = linear(vgelu(linear(gv, t.g0_w, t.g0_b)), t.g1_w, t.g1_b);
  const out = new Float32Array(day.data.length + g.length);
  out.set(day.data); out.set(g, day.data.length);
  return out;
}

export interface Diagnosis {
  /** per day, as SunDoctor.predict() then diagnose(): losses clipped to [0, 1], stuck = 0 on a fixed mount */
  pf: number[]; frac: number[]; frac_sig: number[]; soil: number[]; shade: number[]; clip: number[]; outage: number[];
  snow: number[]; stuck: number[]; step_p: number[]; snow_sig: number[]; outage_sig: number[];
  eta0: number; ac_limit: number; ac_limit_sig: number;
}

/** predict() for one fortnight, with diagnose()'s clipping of the loss shares. */
export function diagnose(W: Pack<SMeta>, x: Tensor, mount: string): { d: Diagnosis; raw: Float32Array } {
  const o = forward(W, x), M = W.meta, n = M.reg.length;
  const row = (k: number) => Array.from(o.subarray(k * D, (k + 1) * D));
  const val = (name: string) => row(M.reg.indexOf(name)).map((v) => v * M.scale[M.reg.indexOf(name)]);
  const sig = (name: string) => row(n + M.reg.indexOf(name)).map((v) => Math.exp(Math.min(3, Math.max(-6, v))) * M.scale[M.reg.indexOf(name)]);
  const c01 = (a: number[]) => a.map((v) => Math.min(1, Math.max(0, v)));
  const G = o.subarray(D * (2 * n + 1));
  const d: Diagnosis = {
    pf: val('pf'), frac: val('frac'), frac_sig: sig('frac'), soil: c01(val('soil')), shade: c01(val('shade')), clip: c01(val('clip')),
    outage: c01(val('outage')), snow: c01(val('snow')), stuck: mount === 'tracker' ? c01(val('stuck')) : new Array(D).fill(0),
    step_p: row(2 * n).map((v) => 1 / (1 + Math.exp(-v))), snow_sig: sig('snow'), outage_sig: sig('outage'),
    eta0: G[0] * M.gscale[0], ac_limit: G[1] * M.gscale[1], ac_limit_sig: Math.exp(Math.min(3, Math.max(-6, G[3]))) * M.gscale[1],
  };
  return { d, raw: o };
}

export const CAUSES = ['dead', 'soiling', 'stuck', 'shading', 'snow', 'clipping', 'outage'] as const;
export type Cause = (typeof CAUSES)[number];

export interface Losses { frac: number; soil: number; stuck: number; shade: number; snow: number; clip: number; outage: number }

/** Shares of the healthy array's possible energy, undone in the simulator's order (as diagnose()'s top chain). */
export function chain(v: Losses): Record<Cause, number> {
  const x1 = Math.min(1, Math.max(0, v.frac)), x2 = x1 * (1 - v.soil), x3 = x2 * (1 - v.stuck), x4 = x3 * (1 - v.shade);
  const x5 = x4 * (1 - v.snow), x6 = x5 * (1 - v.clip), x7 = x6 * (1 - v.outage);
  return { dead: 1 - x1, soiling: x1 - x2, stuck: x2 - x3, shading: x3 - x4, snow: x4 - x5, clipping: x5 - x6, outage: x6 - x7 };
}

export const shares = (d: Diagnosis, i: number) => chain({ frac: d.frac[i], soil: d.soil[i], stuck: d.stuck[i], shade: d.shade[i], snow: d.snow[i], clip: d.clip[i], outage: d.outage[i] });

/** One exported fortnight's arrays from the examples pack. */
export function fortnight(X: Pack<SWindow[]>, i: number): SData {
  const g = (k: string) => X.t[`${k}${i}`].data;
  return { p: g('p'), pu: g('pu'), g: g('g'), ta: g('ta'), cz: g('cz'), ca: g('ca'), gc: g('gc') };
}
