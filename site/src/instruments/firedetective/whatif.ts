// What-ifs on FireDetective's INPUT, mirrored exactly in models/firedetective/web/export_instrument.py (apply):
// lose VIIRS pixels (cloud, omission), scale their fire radiative power, drier or wetter dead fuels, more or less
// wind, a lower or higher canopy base, a shorter record. The FEDS perimeters always stay as observed. No DOM.
import type { Pack } from '../../lib/nn/pack.ts';
import { rng } from '../kit.ts';
import type { Dets, FireInput, Layers, Wx } from './features.ts';

export interface FireInfo {
  fid: string; name: string; short: string; year: number; period: string; region: string; stratum: string;
  steps: string[]; day: boolean[]; dets: Dets[]; wx: Wx[]; doy: number; gamma: number; shape: [number, number, number];
  truth: { hs: number | null; wind: number | null; lfmc: number | null };
  classical: { cfp: number | null; erc_max: number | null; wind_bc: number | null };
  release: Record<string, number>;
}
export interface Examples { fires: FireInfo[]; layers: string[] }

export interface WhatIf {
  drop: number;   // probability of losing each VIIRS detection (cloud, omission)
  seed: number;
  frp: number;    // fire radiative power x
  dry: number;    // 100-h and 1000-h dead fuel moisture minus this many percentage points (floor 1 %)
  wind: number;   // gridMET wind speed x
  cbh: number;    // canopy base height x
  steps: number;  // half-day steps kept (first N)
}
export const RECORDED: WhatIf = { drop: 0, seed: 1, frp: 1, dry: 0, wind: 1, cbh: 1, steps: 12 };
export const isRecorded = (w: WhatIf, n: number) =>
  w.drop === 0 && w.frp === 1 && w.dry === 0 && w.wind === 1 && w.cbh === 1 && Math.min(w.steps, n) === n;

export interface FireRecord { info: FireInfo; input: FireInput }

/** One exam fire's raw observations as stored (masks u8, fuel u8, other layers float32). */
export function fireRecord(pack: Pack<Examples>, i: number): FireRecord {
  const info = pack.meta.fires[i];
  const [n, ny, nx] = info.shape;
  const L = Object.fromEntries(pack.meta.layers.map((k) => [k, pack.t[`${k}${i}`].data])) as unknown as Layers;
  return { info, input: { masks: { n, ny, nx, data: pack.t[`masks${i}`].data }, dets: info.dets, day: info.day, wx: info.wx,
    layers: L, doy: info.doy, gamma: info.gamma } };
}

/** Which VIIRS detections survive the drop what-if: u < drop loses one, u from mulberry32(seed), all steps in order. */
export function keepMask(rec: FireRecord, w: WhatIf): boolean[][] {
  const u = rng(w.seed).u;
  return rec.input.dets.map((d) => d.frp.map(() => (w.drop > 0 ? u() >= w.drop : true)));
}

/** The changed input (export_instrument.apply). */
export function apply(rec: FireRecord, w: WhatIf): FireInput {
  const f = rec.input;
  const km = keepMask(rec, w);
  const dets: Dets[] = f.dets.map((d, k) => {
    const keep = km[k];
    const pick = (a: number[]) => a.filter((_, j) => keep[j]);
    return { row: pick(d.row), col: pick(d.col), frp: pick(d.frp).map((v) => v * w.frp) };
  });
  const wx: Wx[] = f.wx.map((x) => ({ ...x, fm100: Math.max(x.fm100 - w.dry, 1.0), fm1000: Math.max(x.fm1000 - w.dry, 1.0), vs: x.vs * w.wind }));
  let layers = f.layers;
  if (w.cbh !== 1) {
    const k = Math.fround(w.cbh);
    layers = { ...layers, cbh: Float32Array.from(f.layers.cbh as ArrayLike<number>, (v) => Math.fround(v * k)) };
  }
  const n = Math.min(Math.round(w.steps), f.masks.n);
  return { masks: { ...f.masks, n }, dets: dets.slice(0, n), day: f.day.slice(0, n), wx: wx.slice(0, n), layers, doy: f.doy, gamma: f.gamma };
}
