// Parity hook for site/tests/garage.test.ts: one reference case of shared/vehicle/garage/export_garage.py, rebuilt from
// the stored inputs exactly as the garage page does it, through the instruments' own network ports.
import { loadNet, type Net, type WMeta } from '../instruments/hybridpilot/net.ts';
import type { EMeta } from '../instruments/energynet/net.ts';
import { flat, type TMeta } from '../instruments/tripdetective/net.ts';
import type { Pack } from '../lib/nn/pack.ts';
import { cellTrip, enOutputs, enRun, hpInputs, hpRun, tag, tdRaw, tdRead, type GMeta, type VMeta } from './data.ts';

export interface Case { kind: 'en' | 'td' | 'hp'; v: string; r: number; t: number; l: number; pool?: number[]; steps?: number[]; outputs: number[] }
/** load(dir, name): dir 'garage' or an instrument id */
export type Loader = (dir: string, name: string) => Pack<any>;

const cache = new Map<string, Pack<any>>();
const get = <M>(load: Loader, dir: string, name: string) => {
  const k = `${dir}/${name}`;
  if (!cache.has(k)) cache.set(k, load(dir, name));
  return cache.get(k) as Pack<M>;
};
let hpNet: Net | null = null;

export async function run(c: Case, load: Loader): Promise<number[]> {
  const R = get<GMeta>(load, 'garage', 'routes'), G = R.meta;
  const v = G.vehicles.find((x) => x.id === c.v)!;
  const P = get<VMeta>(load, 'garage', `v_${v.id}`);
  if (c.kind === 'en') return enOutputs(enRun(get<EMeta>(load, 'energynet', 'weights'), R, v, c.r, c.t, c.l));
  if (c.kind === 'td') {
    const bus = v.cls === 'bus';
    const W = bus ? get<TMeta>(load, 'garage', 'tdbus') : get<TMeta>(load, 'tripdetective', 'weights');
    const trips = (c.pool ?? [c.r]).map((ri) => tdRaw(G, P, v, ri, c.t, c.l, cellTrip(R, ri, c.t)));
    return flat(tdRead(W, trips, bus).h);
  }
  hpNet ??= loadNet(get<WMeta>(load, 'hybridpilot', 'weights'));
  const tg = tag(c.r, c.t, c.l), cell = P.meta.cells[tg];
  const o = hpRun(hpNet, G, cell, hpInputs(G, P, cell, tg));
  const mean = (a: Float64Array) => a.reduce((s, x) => s + x, 0) / a.length;
  return [...c.steps!.flatMap((i) => [o.y0[i], o.y1[i]]), mean(o.y0), mean(o.y1)];
}
