// One reading shape for both detectors (black-hole cascade and neutron-star network), so the UI and the scan worker share it.
import type { Reading } from './net.ts';
import type { BReading } from './bns.ts';

export interface Rd {
  stat: number; finder: number; check: number | null; vetoed: boolean;
  mc: number; mc_lo: number; mc_hi: number; q: number; t_m: number; t_sig: number;
}

export function toRd(r: Reading | BReading): Rd {
  if ('finder' in r) return { stat: r.stat, finder: r.finder[0], check: r.check, vetoed: r.vetoed, mc: r.mc, mc_lo: r.mc_lo, mc_hi: r.mc_hi, q: r.q, t_m: r.t_m, t_sig: r.t_sig };
  return { stat: r.stat, finder: r.stat, check: null, vetoed: false, mc: r.mc, mc_lo: r.mc_lo, mc_hi: r.mc_hi, q: r.q, t_m: r.t_m, t_sig: r.t_sig };
}
