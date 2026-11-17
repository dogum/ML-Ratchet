// Parity hook for site/tests/parity.test.ts: verifies the browser 3-D wake physics + EdgeWakeGNN
// against Python across all 6 real exam snapshots and 4 deterministic input perturbations.
import { read, parityVector, type WMeta, type WExampleMeta } from './net.ts';
import type { Pack } from '../../lib/nn/pack.ts';

let W: Pack<WMeta> | null = null;
let X: Pack<WExampleMeta[]> | null = null;

export async function run(c: { example: number; variant: string }, load: (name: string) => Pack<any>) {
  W ??= load('weights');
  X ??= load('examples');

  const i = c.example;
  const m = X.meta[i];
  const east = X.t[`east_${i}`].data;
  const north = X.t[`north_${i}`].data;
  const elev = X.t[`elev_${i}`].data;
  const hubH = X.t[`hub_h_${i}`].data;
  const yaw = Float64Array.from(X.t[`yaw_${i}`].data);
  const online = Array.from(X.t[`online_${i}`].data, (v) => v > 0.5);
  const free = Array.from(X.t[`free_${i}`].data, (v) => v > 0.5);

  let wd = m.wind_dir_deg;
  const swd = m.sigma_wd_deg;

  if (c.variant === 'steer15') {
    for (let k = 0; k < yaw.length; k++) {
      if (free[k] && online[k]) yaw[k] = Math.max(-30, Math.min(30, yaw[k] + 15.0));
    }
  } else if (c.variant === 'rot12') {
    wd = (wd + 12.0) % 360.0;
  } else if (c.variant === 'park_front') {
    const idx = online.findIndex((on, k) => on && free[k]);
    if (idx >= 0) online[idx] = false;
  }

  const r = read(W, east, north, elev, hubH, m.turbine_model, m.u_inf_ms, wd, m.ti_inf, swd, yaw, online, m.air_density, m.p_front_kw);
  return parityVector(r);
}
