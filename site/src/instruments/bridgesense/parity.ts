// Parity hook for site/tests/parity.test.ts: run one reference case and return raw outputs.
import { predict, type BSMeta, type BSExample } from './net.ts';
import type { Pack } from '../../lib/nn/pack.ts';

let W: Pack<BSMeta> | null = null;
let X: Pack<BSExample[]> | null = null;

export async function run(c: { example: number; variant: string }, load: (name: string) => Pack<any>) {
  W ??= load('weights');
  X ??= load('examples');
  const i = c.example;
  const T_ref = X.t[`e${i}_T_ref`].data;
  const T_q = new Float32Array(X.t[`e${i}_T_q`].data);
  const df_ref = X.t[`e${i}_df_ref`].data;
  const df_q = new Float32Array(X.t[`e${i}_df_q`].data);
  const mask_ref = X.t[`e${i}_mask_ref`].data;
  const mask_q = X.t[`e${i}_mask_q`].data;

  if (c.variant === 'cold_snap') {
    for (let k = 0; k < T_q.length; k++) T_q[k] -= 6.0;
  } else if (c.variant === 'stiff_drop') {
    const s = Math.sqrt(0.94);
    for (let k = 0; k < df_q.length; k++) {
      df_q[k] = mask_q[k] > 0.5 ? (1 + df_q[k]) * s - 1 : 0;
    }
  }
  const r = predict(W, T_ref, T_q, df_ref, df_q, mask_ref, mask_q);
  return [r.logit_raw, r.delta_k_pct, r.delta_k_sigma, r.thermal_shift_pct, r.thermal_shift_sigma];
}
