// TripDetective in the browser: a line-for-line port of models/tripdetective/tripdetective.py — net_input(), scalars()
// (the 16 classical road-load estimates), encode() (1-D CNN per trip window with a validity mask), heads() (pooled car
// heads and the per-window energy ledger) and the read_car() post-processing. No DOM.
import { conv1d, gelu_, linear, vgelu, softmax } from '../../lib/nn/ops.ts';
import type { Pack } from '../../lib/nn/pack.ts';
import type { Tensor } from '../../lib/nn/tensor.ts';

export const L = 1024;
export const NRAW = 11; // v, grade[%], T_amb, headwind, rho_air, fuel_rate[g/s], rpm/1000, batt[kW], soc, aux[kW], valid
export const NSC = 16;
export const LED_SCALE = [200, 150, 150, 150, 1500, 150];
export const CURB_OFFSET = 136;
export const VEDTYPE: Record<string, string> = {
  conv: 'ICE', hev_powersplit: 'HEV', hev_parallel_p2: 'HEV', hev_series: 'HEV', phev_powersplit: 'PHEV', phev_series: 'PHEV', bev: 'EV',
};

export interface TMeta {
  n_blocks: number; strides: number[]; feat: number; params: number;
  archs: string[]; strats: string[]; reg: string[]; ledger: string[];
  sigk: { mass_car?: number; disp_car?: number; mass_trip?: number; disp_trip?: number };
}

/** One trip window in the model's raw units: rows of RAW, each Float32Array of length n. flags = FLAGS vector. */
export interface Raw { rows: Float32Array[]; n: number; }
export type Flags = number[]; // [has_fuel, fuel_est, has_rpm, has_batt, has_aux]

// ---------------- examples pack ----------------
export interface TripMeta { id: string; n: number; km: number; T: number; rho: number; flags: Flags; ch: Record<string, number | null>; valid: number; }
export interface CarMeta {
  vid: string; ved_type: string; arch: string; arch_conf: string; weight_lb: number | null; curb_kg: number | null; disp: number | null;
  engine: string; trans: string; likely: string; F60_N: number | null; trips: TripMeta[];
}
export interface LabConfig { arch: string; strat: string; dm: number; fc: number; mass: number; CdA: number; Crr: number; disp: number | null; ledger: number[][]; logs: Record<string, number>[]; }
export interface XMeta {
  cars: CarMeta[];
  lab: { routes: { id: string; n: number }[]; configs: LabConfig[]; mass: number[]; cda: number[]; strat: Record<string, string[]> };
  scales: Record<string, number>; labScales: Record<string, number>;
}
const ROW: Record<string, number> = { v: 0, grade: 1, hw: 3, fuel: 5, rpm: 6, batt: 7, soc: 8, aux: 9 };

/** int16 values stored as little-endian byte pairs in a u8 tensor (pack.ts hands u8 over as floats 0..255). */
function int16At(b: Float32Array, i: number) {
  const u = b[2 * i] + 256 * b[2 * i + 1];
  return u >= 32768 ? u - 65536 : u;
}

/** A real exam trip, decoded exactly as training/webdata.py compact() + the exporter: q / scale, then float32. */
export function examRaw(X: Pack<XMeta>, t: TripMeta): Raw {
  const n = t.n, q = X.t.q16.data, sc = X.meta.scales;
  const rows = Array.from({ length: NRAW }, () => new Float32Array(n));
  for (const [k, r] of Object.entries(ROW)) {
    const off = t.ch[k];
    if (off == null) continue;
    for (let i = 0; i < n; i++) rows[r][i] = int16At(q, off + i) / sc[k];
  }
  rows[2].fill(t.T); rows[4].fill(t.rho);
  const va = X.t.valid.data;
  for (let i = 0; i < n; i++) rows[10][i] = va[t.valid + i] > 0.5 ? 1 : 0;
  return { rows, n };
}

/** A simulated lab log over one route (precomputed RoadSim + OBD emulator in Python), and the logger's flags. */
export function labRaw(X: Pack<XMeta>, ci: number, r: number): { raw: Raw; flags: Flags } {
  const cfg = X.meta.lab.configs[ci], R = X.t[`route${r}`], n = R.shape[1], q = X.t.q16.data, sc = X.meta.labScales;
  const rows = Array.from({ length: NRAW }, () => new Float32Array(n));
  for (let k = 0; k < 5; k++) rows[k].set(R.data.subarray(k * n, (k + 1) * n));
  for (const [k, off] of Object.entries(cfg.logs[r])) for (let i = 0; i < n; i++) rows[ROW[k]][i] = int16At(q, off + i) / sc[k];
  rows[10].fill(1);
  return { raw: { rows, n }, flags: cfg.arch === 'bev' ? [0, 0, 0, 1, 0] : [1, 1, 1, 0, 0] };
}

export type Channel = 'fuel' | 'rpm' | 'batt' | 'aux';
/** Take channels away from the logger, and/or keep only the first `cut` seconds (the exporter's apply()). */
export function apply(raw: Raw, flags: Flags, drop: string[] = [], cut: number | null = null): { raw: Raw; flags: Flags } {
  const fl = [...flags];
  const n = cut ? Math.min(cut, raw.n) : raw.n;
  const rows = raw.rows.map((r) => r.slice(0, n));
  const zero = (i: number) => rows[i].fill(0);
  if (drop.includes('fuel')) { zero(5); fl[0] = 0; fl[1] = 0; }
  if (drop.includes('rpm')) { zero(6); fl[2] = 0; }
  if (drop.includes('batt')) { zero(7); zero(8); fl[3] = 0; zero(9); fl[4] = 0; }
  if (drop.includes('aux')) { zero(9); fl[4] = 0; }
  return { raw: { rows, n }, flags: fl };
}

// ---------------- features ----------------
/** net_input(): the 22 network channels [22, L] and the number of real samples. */
export function netInput(raw: Raw, fl: Flags): { x: Tensor; n: number } {
  const n = Math.min(raw.n, L), C = 22, x = new Float32Array(C * L), r = raw.rows;
  const v = r[0];
  for (let i = 0; i < n; i++) {
    const a = n > 2 && i > 0 && i < n - 1 ? Math.fround((v[i + 1] - v[i - 1]) / 2) : 0;
    const va = r[10][i];
    x[i] = v[i] / 15; x[L + i] = a / 1.5; x[2 * L + i] = r[1][i] / 3; x[3 * L + i] = (r[2][i] - 10) / 12; x[4 * L + i] = r[3][i] / 5;
    x[5 * L + i] = (r[4][i] - 1.22) * 10; x[6 * L + i] = r[5][i] * va; x[7 * L + i] = Math.log1p(Math.max(r[5][i], 0) * 4) * va;
    x[8 * L + i] = r[6][i] * va; x[9 * L + i] = (r[7][i] / 20) * va; x[10 * L + i] = (r[8][i] - 0.5) * 2 * fl[3]; x[11 * L + i] = r[9][i] / 2;
    x[12 * L + i] = va; x[13 * L + i] = 1;
    for (let k = 0; k < 5; k++) x[(14 + k) * L + i] = fl[k];
    const va2 = Math.fround(v[i] + r[3][i]);
    x[19 * L + i] = (v[i] * (1.03 * a + (9.81 * r[1][i]) / 100)) / 10;
    x[20 * L + i] = (0.5 * r[4][i] * va2 * Math.abs(va2) * v[i]) / 2000;
    x[21 * L + i] = (9.81 * v[i]) / 200;
  }
  return { x: { data: x, shape: [C, L] }, n };
}

function smooth5(x: Float64Array) { // np.convolve(x, ones(5)/5, 'same')
  const n = x.length, y = new Float64Array(n);
  for (let i = 0; i < n; i++) { let s = 0; for (let k = -2; k <= 2; k++) { const j = i + k; if (j >= 0 && j < n) s += x[j] * 0.2; } y[i] = s; }
  return y;
}
function pctl(a: number[], p: number) { // np.percentile, linear
  const s = Float64Array.from(a).sort(), N = s.length, pos = (p / 100) * (N - 1), lo = Math.floor(pos), hi = Math.min(lo + 1, N - 1);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}
function solve(A: number[][], b: number[]) { // Gaussian elimination with partial pivoting
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  return M.map((r, i) => r[n] / r[i]);
}

/** scalars(): 16 classical numbers per trip window — a least-squares road-load fit of the logged source power, idle fuel and
 * rpm, high-load fuel, rpm per speed in cruise, engine-off share, energy per km, battery power percentiles. */
export function scalars(raw: Raw, fl: Flags): Float32Array {
  const n = Math.min(raw.n, L), r = raw.rows, out = new Float32Array(NSC);
  const v = Float64Array.from(r[0].subarray(0, n)), ok = (i: number) => r[10][i] > 0.5;
  const a = new Float64Array(n);
  if (n > 2) for (let i = 1; i < n - 1; i++) a[i] = (v[i + 1] - v[i - 1]) / 2;
  const P = new Float64Array(n);
  let have = false;
  if (fl[0] > 0.5) { for (let i = 0; i < n; i++) P[i] += 0.25 * r[5][i] * 42600; have = true; }
  if (fl[3] > 0.5) { for (let i = 0; i < n; i++) P[i] += r[7][i] * 1000; have = true; }
  if (have) {
    const g = Float64Array.from({ length: n }, (_, i) => r[1][i] / 100);
    const Ps = smooth5(P), vs = smooth5(v), as = smooth5(a), gs = smooth5(g);
    const rows: number[][] = [], y: number[] = [];
    for (let i = 0; i < n; i++) if (ok(i) && vs[i] > 3 && as[i] > -0.3 && Ps[i] > 0) {
      const vw = v[i] + r[3][i];
      rows.push([vs[i] * (1.03 * as[i] + 9.81 * gs[i]), 9.81 * vs[i], 0.5 * r[4][i] * vw * Math.abs(vw) * vs[i], 1]);
      y.push(Ps[i]);
    }
    const N = rows.length;
    if (N >= 30) {
      const K = [0, 1, 2, 3];
      const sc = K.map((k) => Math.sqrt(rows.reduce((s, q) => s + q[k] * q[k], 0) / N) + 1e-9);
      const A = K.map((p) => K.map((q) => rows.reduce((s, row) => s + (row[p] / sc[p]) * (row[q] / sc[q]), 0) + (p === q ? 1e-3 * N : 0)));
      const bb = K.map((p) => rows.reduce((s, row, i) => s + (row[p] / sc[p]) * y[i], 0));
      const b = solve(A, bb).map((x, k) => x / sc[k]);
      const res = y.map((x, i) => x - rows[i].reduce((s, c, k) => s + c * b[k], 0));
      const variance = (u: number[]) => { const m = u.reduce((s, x) => s + x, 0) / u.length; return u.reduce((s, x) => s + (x - m) * (x - m), 0) / u.length; };
      const r2 = 1 - variance(res) / Math.max(variance(y), 1e-9);
      out[0] = Math.asinh(b[0] / 500); out[1] = Math.asinh(b[1] / 10); out[2] = Math.asinh(b[2] / 0.3); out[3] = Math.asinh(b[3] / 2000);
      out[4] = Math.max(-1, Math.min(1, r2)); out[5] = Math.log1p(N) / 7;
    }
  }
  let km = 0;
  for (let i = 0; i < n; i++) if (ok(i)) km += v[i];
  km = Math.max(km / 1000, 0.05);
  const mv: number[] = [];
  for (let i = 0; i < n; i++) if (ok(i) && v[i] > 2) mv.push(i);
  if (fl[0] > 0.5) {
    const idle: number[] = [];
    for (let i = 0; i < n; i++) if (ok(i) && v[i] < 0.3 && (fl[2] > 0.5 ? r[6][i] > 0.3 : r[5][i] > 0)) idle.push(r[5][i]);
    if (idle.length >= 5) out[6] = pctl(idle, 50) * 3;
    if (mv.length >= 10) out[8] = pctl(mv.map((i) => r[5][i]), 90) / 2;
    let s = 0;
    for (let i = 0; i < n; i++) if (ok(i)) s += r[5][i];
    out[12] = s / km / 100;
  }
  if (fl[2] > 0.5) {
    const idle: number[] = [], cr: number[] = [];
    for (let i = 0; i < n; i++) {
      if (ok(i) && v[i] < 0.3 && r[6][i] > 0.3) idle.push(r[6][i]);
      if (ok(i) && v[i] > 11 && v[i] < 20 && r[6][i] > 0.3) cr.push(r[6][i] / v[i]);
    }
    if (idle.length >= 5) out[7] = pctl(idle, 50);
    if (cr.length >= 5) out[9] = pctl(cr, 50) * 10;
    if (mv.length >= 10) out[10] = mv.filter((i) => r[6][i] < 0.1).length / mv.length;
    out[11] = 1;
  }
  if (fl[3] > 0.5) {
    let s = 0;
    for (let i = 0; i < n; i++) if (ok(i)) s += r[7][i];
    out[13] = (s / 3600 / km) * 5;
    if (mv.length >= 10) { const pb = mv.map((i) => r[7][i]); out[14] = pctl(pb, 90) / 50; out[15] = pctl(pb, 10) / 50; }
  }
  return out;
}

/** windows(): trips longer than L seconds are cut into near-equal windows (each >= 120 s) and pooled like trips. */
export function windows(n: number, Lw = L, minLen = 120): [number, number][] {
  if (n <= Lw) return [[0, n]];
  const k = Math.ceil(n / Lw), e = Array.from({ length: k + 1 }, (_, i) => Math.trunc((i * n) / k));
  const out: [number, number][] = [];
  for (let i = 0; i < k; i++) if (e[i + 1] - e[i] >= minLen) out.push([e[i], e[i + 1]]);
  return out;
}
export function slice(raw: Raw, s: number, e: number): Raw { return { rows: raw.rows.map((r) => r.subarray(s, e)), n: e - s }; }

// ---------------- network ----------------
/** encode(): one trip window -> its embedding. */
export function encode(W: Pack<TMeta>, raw: Raw, fl: Flags): Float32Array {
  const t = W.t, m = W.meta;
  const { x, n } = netInput(raw, fl);
  const T = Math.min(L, Math.max(32, Math.ceil(n / 32) * 32));
  const C0 = x.shape[0];
  const h0 = new Float32Array(C0 * T);
  for (let c = 0; c < C0; c++) h0.set(x.data.subarray(c * L, c * L + T), c * T);
  let h: Tensor = { data: h0, shape: [C0, T] };
  let mk = new Float32Array(T);
  mk.fill(1, 0, n);
  for (let i = 0; i < m.n_blocks; i++) {
    h = gelu_(conv1d(h, t[`c${i}_w`], t[`c${i}_b`], m.strides[i]));
    if (i > 0) {
      const m2 = new Float32Array(Math.ceil(mk.length / 2));
      for (let j = 0; j < m2.length; j++) m2[j] = Math.max(mk[2 * j], 2 * j + 1 < mk.length ? mk[2 * j + 1] : 0);
      mk = m2;
    }
    const [C, Lh] = h.shape, Lk = Math.min(Lh, mk.length), d = new Float32Array(C * Lk);
    for (let c = 0; c < C; c++) for (let j = 0; j < Lk; j++) d[c * Lk + j] = h.data[c * Lh + j] * mk[j];
    h = { data: d, shape: [C, Lk] };
  }
  const [C, Lh] = h.shape;
  let cnt = 0;
  for (let j = 0; j < Lh; j++) cnt += mk[j];
  const f = new Float32Array(2 * C + 2 + (m.feat >= 3 ? NSC : 0));
  for (let c = 0; c < C; c++) {
    let s = 0, mx = -Infinity;
    for (let j = 0; j < Lh; j++) { const v = h.data[c * Lh + j] * mk[j]; s += v; const u = v + (mk[j] - 1) * 1e4; if (u > mx) mx = u; }
    f[c] = s / Math.max(cnt, 1); f[C + c] = mx;
  }
  const dur = n / 1024;
  f[2 * C] = dur; f[2 * C + 1] = Math.log(dur + 1e-3);
  if (m.feat >= 3) f.set(scalars(raw, fl), 2 * C + 2);
  return linear(vgelu(linear(f, t.trip0_w, t.trip0_b)), t.trip1_w, t.trip1_b);
}

export interface Heads { arch: number[]; strat: number[]; mu: number[]; sig: number[]; led: { mu: number[]; sig: number[] }[]; }

/** heads(): pool the window embeddings of one car, then architecture, strategy, regressions, per-window ledger. */
export function heads(W: Pack<TMeta>, E: Float32Array[]): Heads {
  const t = W.t, D = E[0].length, z0 = new Float32Array(D + 1);
  for (let i = 0; i < D; i++) { let s = 0; for (const e of E) s += e[i]; z0[i] = s / E.length; }
  z0[D] = Math.log(E.length);
  const z = vgelu(linear(vgelu(linear(z0, t.car0_w, t.car0_b)), t.car1_w, t.car1_b));
  const arch = softmax(linear(z, t.arch_w, t.arch_b)), strat = softmax(linear(z, t.strat_w, t.strat_b));
  const r = linear(z, t.reg_w, t.reg_b), nr = r.length / 2;
  const clipExp = (x: number) => Math.exp(Math.min(3, Math.max(-6, x)));
  const mu = Array.from(r.subarray(0, nr)), sig = Array.from(r.subarray(nr), clipExp);
  const led = E.map((e) => {
    const u = new Float32Array(2 * D);
    u.set(e, 0); u.set(z, D);
    const o = linear(vgelu(linear(u, t.led0_w, t.led0_b)), t.led1_w, t.led1_b);
    return { mu: LED_SCALE.map((s, i) => o[i] * s), sig: LED_SCALE.map((s, i) => clipExp(o[6 + i]) * s) };
  });
  return { arch, strat, mu, sig, led };
}

/** The raw outputs the parity test compares: arch, strat, mu, sig, then each window's ledger mu and sig. */
export function flat(h: Heads): number[] {
  return [...h.arch, ...h.strat, ...h.mu, ...h.sig, ...h.led.flatMap((l) => [...l.mu, ...l.sig])];
}

export interface Trip { raw: Raw; flags: Flags; }

/** Run the network on several trips of one car (every window of every trip pooled), as read_car(). */
export function run(W: Pack<TMeta>, trips: Trip[]): { h: Heads; tripOf: number[]; len: number[] } {
  const E: Float32Array[] = [], tripOf: number[] = [], len: number[] = [];
  trips.forEach((tr, k) => {
    for (const [s, e] of windows(tr.raw.n)) { E.push(encode(W, slice(tr.raw, s, e), tr.flags)); tripOf.push(k); len.push(e - s); }
  });
  return { h: heads(W, E), tripOf, len };
}

export interface Reading {
  type: Record<string, number>; arch: [string, number][]; strat: [string, number][];
  mass: number; massLo: number; massHi: number; curb: number;
  disp: number; dispLo: number; dispHi: number;
  CdA: number; CdAsig: number; Crr: number; Crrsig: number;
  evShare: number; Pon: number; vEv: number;
  ledger: { mu: number[]; sig: number[] }[]; // per trip, kJ/km
}

/** read_car()'s post-processing: physical units and dev-fitted error-bar widening. */
export function reading(W: Pack<TMeta>, out: ReturnType<typeof run>, nTrips: number): Reading {
  const M = W.meta, { h } = out, k = M.sigk ?? {};
  const km = (nTrips > 1 ? k.mass_car : k.mass_trip) ?? 1, kd = (nTrips > 1 ? k.disp_car : k.disp_trip) ?? 1;
  const m = 1800 * Math.exp(h.mu[0]), d = 2.5 * Math.exp(h.mu[1]);
  const type: Record<string, number> = {};
  M.archs.forEach((a, i) => { type[VEDTYPE[a]] = (type[VEDTYPE[a]] ?? 0) + h.arch[i]; });
  const ledger = Array.from({ length: nTrips }, (_, t) => {
    const js = out.tripOf.map((tt, j) => (tt === t ? j : -1)).filter((j) => j >= 0);
    const tot = js.reduce((s, j) => s + out.len[j], 0);
    const mu = LED_SCALE.map((_, i) => js.reduce((s, j) => s + (out.len[j] / tot) * h.led[j].mu[i], 0));
    const sig = LED_SCALE.map((_, i) => Math.sqrt(js.reduce((s, j) => s + ((out.len[j] / tot) * h.led[j].sig[i]) ** 2, 0)));
    return { mu, sig };
  });
  const sorted = (names: string[], p: number[]) => names.map((a, i) => [a, p[i]] as [string, number]).sort((x, y) => y[1] - x[1]);
  return {
    type, arch: sorted(M.archs, h.arch), strat: sorted(M.strats, h.strat),
    mass: m, massLo: m * Math.exp(-h.sig[0] * km), massHi: m * Math.exp(h.sig[0] * km), curb: m - CURB_OFFSET,
    disp: d, dispLo: d * Math.exp(-h.sig[1] * kd), dispHi: d * Math.exp(h.sig[1] * kd),
    CdA: 0.75 * Math.exp(h.mu[2]), CdAsig: h.sig[2], Crr: 0.008 * Math.exp(h.mu[3]), Crrsig: h.sig[3],
    evShare: h.mu[4], Pon: Math.expm1(h.mu[5] + 2), vEv: (h.mu[6] + 1.5) * 10, ledger,
  };
}
