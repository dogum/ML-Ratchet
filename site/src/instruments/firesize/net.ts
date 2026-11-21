// FireSize in the browser: the what-if, the Rothermel inputs and the 3-member MLP ensemble, ported from
// models/firesize/firesize.py (FireSize.probs, quantile, predict; simard_emc, ros_mmin, make_features' ROS columns)
// and the what-if of models/firesize/web/export_instrument.py. No DOM.
import type { Pack } from '../../lib/nn/pack.ts';

export interface FSMeta {
  feats: string[]; temp: number; n_ens: number; n_layers: number;
  edges_ac: number[]; E: number[]; class_labels: string[]; esc_bin: number; large_bin: number;
  regions: string[]; region_names: Record<string, string>; causes: string[]; fcls: string[]; wxf: string[];
  erc_coupling: { bi: number; fm100: number; fm1000: number };
  lut_codes: number[]; lut_m1: number[]; lut_u10: number[]; version: string;
}

export interface Ignition {
  fod_id: number; fire_name: string; name: string; year: number; state: string; date: string; cause: string;
  fuel: string; size_class: string; size_bin: number; real_acres: number; exam_p_escape: number; x: number[];
  release: { p_escape: number; p_large: number; median: number; lo: number; hi: number };
}
export interface ExMeta { ignitions: Ignition[] }

/** A change to the input. null = keep the record. */
export interface WhatIf {
  dWind: number; dRH: number; dT: number; dERC: number;
  hour: number | null; cause: number | null; fed: number | null; region: number | null;
}
export const RECORDED: WhatIf = { dWind: 0, dRH: 0, dT: 0, dERC: 0, hour: null, cause: null, fed: null, region: null };
export const weatherTouched = (w: WhatIf) => w.dWind !== 0 || w.dRH !== 0 || w.dT !== 0 || w.dERC !== 0;
export const isRecorded = (w: WhatIf) => !weatherTouched(w) && w.hour === null && w.cause === null && w.fed === null && w.region === null;

/** Simard (1968) equilibrium moisture content (%), as firesize.simard_emc. */
export function simard(tc: number, rh: number) {
  const T = tc * 1.8 + 32, H = Math.min(100, Math.max(0, rh));
  return H < 10 ? 0.03229 + 0.281073 * H - 0.000578 * H * T
    : H < 50 ? 2.22749 + 0.160107 * H - 0.01478 * T
      : 21.0606 + 0.005565 * H * H - 0.00035 * H * T - 0.483199 * H;
}

/** np.interp(v, g, arange(len(g))): a fractional index into an increasing grid, clamped at both ends. */
function fracIndex(v: number, g: number[]) {
  if (Number.isNaN(v)) return NaN;
  if (v <= g[0]) return 0;
  const n = g.length;
  if (v >= g[n - 1]) return n - 1;
  let j = 0;
  while (v >= g[j + 1]) j++;
  return j + (v - g[j]) / (g[j + 1] - g[j]);
}

export interface Lut { row: Map<number, number>; data: Float32Array; nM: number; nU: number; m1: number[]; u10: number[] }
export function lutOf(pack: Pack<FSMeta>): Lut {
  const { meta, t } = pack;
  return { row: new Map(meta.lut_codes.map((c, i) => [c, i])), data: t.lut.data, nM: meta.lut_m1.length, nU: meta.lut_u10.length,
    m1: meta.lut_m1, u10: meta.lut_u10 };
}

/** Rothermel head ROS (m/min) from FireSim's behave() table: fuel code x 1-h moisture x 10 m wind, bilinear. */
export function ros(L: Lut, code: number, m1: number, u10: number) {
  const r = L.row.get(code);
  if (r === undefined) throw new Error(`fuel code ${code} is not in the exported table`);
  const i = fracIndex(m1, L.m1), j = fracIndex(u10, L.u10);
  const i0 = Math.floor(i), j0 = Math.floor(j), i1 = Math.min(i0 + 1, L.nM - 1), j1 = Math.min(j0 + 1, L.nU - 1);
  const fi = i - i0, fj = j - j0;
  const g = (a: number, b: number) => L.data[(r * L.nM + a) * L.nU + b];
  return (1 - fi) * (1 - fj) * g(i0, j0) + fi * (1 - fj) * g(i1, j0) + (1 - fi) * fj * g(i0, j1) + fi * fj * g(i1, j1);
}

const clampMoist = (v: number) => Math.min(35, Math.max(1, v));

/** The what-if on the recorded inputs (export_instrument.apply). codes: 169 FBFM40 codes, row-major, ignition at 84. */
export function apply(pack: Pack<FSMeta>, L: Lut, x0: ArrayLike<number>, codes: ArrayLike<number>, w: WhatIf): Float64Array {
  const { meta } = pack;
  const idx = (k: string) => { const j = meta.feats.indexOf(k); if (j < 0) throw new Error(`no feature ${k}`); return j; };
  const x = Float64Array.from(x0);
  const get = (k: string) => x[idx(k)];
  const set = (k: string, v: number) => { x[idx(k)] = v; };
  if (w.dWind !== 0) for (const k of ['vs', 'vs_m1']) set(k, Math.max(0, get(k) + w.dWind));
  if (w.dRH !== 0) for (const k of ['rmin', 'rmin_m1']) set(k, Math.min(100, Math.max(1, get(k) + w.dRH)));
  if (w.dT !== 0) for (const k of ['tmmx', 'tmmx_m1']) set(k, get(k) + w.dT);
  if (w.dERC !== 0) {
    const d = w.dERC, c = meta.erc_coupling;
    for (const s of ['', '_m1']) {
      set('erc' + s, Math.max(0, get('erc' + s) + d));
      set('bi' + s, Math.max(0, get('bi' + s) + c.bi * d));
      set('fm100' + s, Math.max(1, get('fm100' + s) + c.fm100 * d));
      set('fm1000' + s, Math.max(1, get('fm1000' + s) + c.fm1000 * d));
    }
  }
  if (weatherTouched(w)) {
    const m1 = clampMoist(simard(get('tmmx'), get('rmin'))), u = get('vs') * 1.45;
    let s5 = 0, s13 = 0, pt = 0;
    for (let r = 0; r < 13; r++) for (let c = 0; c < 13; c++) {
      const v = Math.log1p(ros(L, codes[r * 13 + c], m1, u));
      s13 += v;
      if (r >= 4 && r <= 8 && c >= 4 && c <= 8) s5 += v;
      if (r === 6 && c === 6) pt = v;
    }
    set('ros_pt', pt); set('ros5', s5 / 25); set('ros13', s13 / 169);
    const m1b = clampMoist(simard(get('tmmx_m1'), get('rmin_m1')));
    set('ros_pt_m1', Math.log1p(ros(L, codes[84], m1b, get('vs_m1') * 1.45)));
  }
  if (w.hour !== null) { set('hr_sin', Math.sin((2 * Math.PI * w.hour) / 24)); set('hr_cos', Math.cos((2 * Math.PI * w.hour) / 24)); set('hr_miss', 0); }
  if (w.cause !== null) ['c_hum', 'c_nat', 'c_miss'].forEach((k, i) => set(k, i === w.cause ? 1 : 0));
  if (w.fed !== null) set('fed', w.fed);
  if (w.region !== null) meta.regions.forEach((r, i) => set(`r_${r}`, i === w.region ? 1 : 0));
  return x;
}

/** FireSize.probs for one input vector: the ensemble mean of tempered softmaxes. */
export function probs(pack: Pack<FSMeta>, x: ArrayLike<number>, T: number): Float64Array {
  const { t, meta } = pack;
  const F = meta.feats.length, mu = t.mu.data, sd = t.sd.data, fill = t.fill.data;
  const z = new Float64Array(F);
  for (let j = 0; j < F; j++) { const v = Number.isFinite(x[j]) ? x[j] : fill[j]; z[j] = (v - mu[j]) / sd[j]; }
  const nc = meta.class_labels.length;
  const out = new Float64Array(nc);
  for (let e = 0; e < meta.n_ens; e++) {
    let h: Float64Array = z;
    for (let k = 0; k < meta.n_layers; k++) {
      const W = t[`e${e}_L${k}_w`], b = t[`e${e}_L${k}_b`].data, w = W.data;
      const nout = W.shape[0], nin = W.shape[1], o = new Float64Array(nout);
      for (let i = 0; i < nout; i++) {
        let s = 0;
        const off = i * nin;
        for (let j = 0; j < nin; j++) s += h[j] * w[off + j];
        s += b[i];
        o[i] = k < meta.n_layers - 1 ? Math.max(s, 0) : s;
      }
      h = o;
    }
    let mx = -Infinity;
    for (let i = 0; i < nc; i++) { h[i] = h[i] / T; if (h[i] > mx) mx = h[i]; }
    let sum = 0;
    const p = new Float64Array(nc);
    for (let i = 0; i < nc; i++) { p[i] = Math.exp(h[i] - mx); sum += p[i]; }
    for (let i = 0; i < nc; i++) out[i] += p[i] / sum;
  }
  for (let i = 0; i < nc; i++) out[i] /= meta.n_ens;
  return out;
}

/** FireSize.quantile: the level's point, interpolated in log10 acres inside its class (open ends ±0.6 decade). */
export function quantile(E: number[], p: ArrayLike<number>, level: number) {
  let c = 0, k = -1, cprev = 0;
  for (let i = 0; i < p.length; i++) { const n = c + p[i]; if (n >= level) { k = i; cprev = c; break; } c = n; }
  if (k < 0) { k = 0; cprev = 0; }          // np.argmax on an all-false row
  const f = Math.min(1, Math.max(0, (level - cprev) / Math.max(p[k], 1e-12)));
  return 10 ** (E[k] + f * (E[k + 1] - E[k]));
}

export interface Reading {
  x: Float64Array; p: Float64Array; p1: Float64Array;
  pEscape: number; pLarge: number; pEscape1: number; median: number; lo: number; hi: number; ms: number;
}

/** What the instrument runs on every change: what-if -> Rothermel inputs -> ensemble (widened and T = 1). */
export function read(pack: Pack<FSMeta>, L: Lut, ign: Ignition, codes: ArrayLike<number>, w: WhatIf): Reading {
  const t0 = typeof performance !== 'undefined' ? performance.now() : 0;
  const { meta } = pack;
  const x = apply(pack, L, ign.x, codes, w);
  const p = probs(pack, x, meta.temp), p1 = probs(pack, x, 1);
  let pe = 0, pl = 0, pe1 = 0;
  for (let k = meta.esc_bin; k < p.length; k++) { pe += p[k]; pe1 += p1[k]; }
  for (let k = meta.large_bin; k < p.length; k++) pl += p[k];
  const ms = typeof performance !== 'undefined' ? performance.now() - t0 : 0;
  return { x, p, p1, pEscape: pe, pLarge: pl, pEscape1: pe1, median: quantile(meta.E, p1, 0.5),
    lo: quantile(meta.E, p, 0.1), hi: quantile(meta.E, p, 0.9), ms };
}

/** The ignition's 169 fuel codes from the examples pack. */
export function codesOf(ex: Pack<ExMeta>, i: number) {
  return ex.t.codes.data.subarray(i * 169, (i + 1) * 169);
}

/** FPA-FOD size class index of a size in acres (np.searchsorted(EDGES_AC, a, 'right')). */
export function sizeBin(meta: FSMeta, acres: number) {
  let b = 0;
  while (b < meta.edges_ac.length && meta.edges_ac[b] <= acres) b++;
  return b;
}
