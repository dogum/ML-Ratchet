// What-ifs on BasinDetective's INPUT, mirrored exactly in models/basindetective/web/export_instrument.py (apply):
// fewer gauged years, a linear reservoir on the gauged flow, AR(1) gauge jitter, rounding to significant figures in
// cfs, and a wetter/drier, warmer/cooler target decade. No DOM.
import type { Pack } from '../../lib/nn/pack.ts';
import { calendar, type Days } from './features.ts';

export interface BasinInfo {
  id: string; name: string; stratum: string; area_km2: number;
  cal_start: string; tgt_start: string; n_cal: number; n_tgt: number;
  observed_change: Record<string, number>; gr4j_change: Record<string, number>; dor: number;
  expected: Record<string, number>;
  cal_signatures: Record<string, number>;
  release: { mu: number[]; sd: number[]; p: number };
}
export interface Examples { basins: BasinInfo[] }

export interface WhatIf {
  years: number;   // gauged water years kept (the last N), 9 = all
  res_k: number;   // linear-reservoir time constant (days), 0 = none
  cv: number;      // AR(1) gauge jitter (rho 0.9), 0 = none
  sig: number;     // significant figures in cfs, 0 = as recorded
  seed: number;
  pmul: number;    // target-decade precipitation multiplier
  dT: number;      // target-decade warming (C); PET x (1 + 0.025 dT)
}
export const RECORDED: WhatIf = { years: 9, res_k: 0, cv: 0, sig: 0, seed: 1, pmul: 1, dT: 0 };
const CFS = 0.028316846592;

/** The basin's real record, as stored (int32 hundredths -> the same doubles Python parsed from the CSV). */
export function basinRecord(pack: Pack<Examples>, i: number) {
  const info = pack.meta.basins[i];
  const c = pack.t['cal' + i].data, t = pack.t['tgt' + i].data;
  const nc = info.n_cal, nt = info.n_tgt;
  const col = (d: Float32Array, k: number, w: number, n: number) => Float64Array.from({ length: n }, (_, j) => d[j * w + k] / 100);
  const ccal = calendar(info.cal_start, nc), ctgt = calendar(info.tgt_start, nt);
  return {
    info,
    cal: { P: col(c, 0, 4, nc), T: col(c, 1, 4, nc), E: col(c, 2, 4, nc), month: ccal.month, wy: ccal.wy } as Days,
    q: col(c, 3, 4, nc),
    tgt: { P: col(t, 0, 3, nt), T: col(t, 1, 3, nt), E: col(t, 2, 3, nt), month: ctgt.month, wy: ctgt.wy } as Days,
  };
}
export type BasinRecord = ReturnType<typeof basinRecord>;

/** mulberry32 + Box-Muller, as in kit.rng and the exporter's Python port. */
function normals(seed: number) {
  let a = seed >>> 0;
  const u = () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  return () => { let x = 0; while (x === 0) x = u(); const y = u(); return Math.sqrt(-2 * Math.log(x)) * Math.cos(2 * Math.PI * y); };
}
/** np.round: halves to even. */
function roundHalfEven(x: number) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

const slice = (d: Days, i0: number): Days => ({ P: d.P.subarray(i0), T: d.T.subarray(i0), E: d.E.subarray(i0), month: d.month.subarray(i0), wy: d.wy.subarray(i0) });

/** Apply a what-if: returns the calibration forcing, the gauged flow and the target forcing the network will read. */
export function apply(rec: BasinRecord, w: WhatIf): { cal: Days; q: Float64Array; tgt: Days } {
  let wyMax = -Infinity;
  for (const y of rec.cal.wy) wyMax = Math.max(wyMax, y);
  let i0 = 0;
  while (i0 < rec.cal.wy.length && !(rec.cal.wy[i0] > wyMax - w.years)) i0++;
  const cal = slice(rec.cal, i0);
  let q = Float64Array.from(rec.q.subarray(i0));
  if (w.res_k > 0) {
    const a = Math.exp(-1 / w.res_k);
    let s = 0;
    for (let t = 0; t < Math.min(365, q.length); t++) s += q[t];
    s /= Math.min(365, q.length);
    const out = new Float64Array(q.length);
    for (let t = 0; t < q.length; t++) { if (t > 0) s = a * s + (1 - a) * q[t]; out[t] = s; }
    q = out;
  }
  if (w.cv > 0) {
    const n = normals(w.seed), rho = 0.9, cv = w.cv;
    let e = n();
    for (let t = 0; t < q.length; t++) {
      if (t > 0) e = rho * e + Math.sqrt(1 - rho * rho) * n();
      q[t] = q[t] * Math.exp(cv * e - 0.5 * cv * cv);
    }
  }
  if (w.sig > 0) {
    const k = rec.info.area_km2 / 86.4 / CFS;   // mm/day -> cfs
    for (let t = 0; t < q.length; t++) {
      const c = q[t] * k;
      if (!(c > 0)) { q[t] = 0; continue; }     // as the Python port (np.zeros where not positive)
      // a decimal literal gives the same double as Python's 10.0 ** e (Math.pow can differ in the last bit)
      const step = Math.max(Number('1e' + (Math.floor(Math.log10(c)) - w.sig + 1)), 0.01);
      q[t] = (roundHalfEven(c / step) * step) / k;
    }
  }
  const nt = rec.tgt.P.length;
  const tgt: Days = {
    P: Float64Array.from({ length: nt }, (_, i) => rec.tgt.P[i] * w.pmul),
    T: Float64Array.from({ length: nt }, (_, i) => rec.tgt.T[i] + w.dT),
    E: Float64Array.from({ length: nt }, (_, i) => rec.tgt.E[i] * (1 + 0.025 * w.dT)),
    month: rec.tgt.month, wy: rec.tgt.wy,
  };
  return { cal, q, tgt };
}

export const isRecorded = (w: WhatIf) => w.years === 9 && w.res_k === 0 && w.cv === 0 && w.sig === 0 && w.pmul === 1 && w.dT === 0;
export const gaugeTouched = (w: WhatIf) => w.years !== 9 || w.res_k !== 0 || w.cv !== 0 || w.sig !== 0;
export const climateTouched = (w: WhatIf) => w.pmul !== 1 || w.dT !== 0;
