// Parity hook for site/tests/parity.test.ts: apply the case's controls (history length, temperature shift, operating
// hours, a closed week) to a real exam meter, recompute the features and both networks for one week, and return
// [kW (168 h), sigma (168 h)] as the exporter wrote them.
import { frame, predict, type WMeta, type XMeta, type Mode, type Variant, type Sched } from './net.ts';
import type { Pack } from '../../lib/nn/pack.ts';

let W: Pack<WMeta> | null = null;
let X: Pack<XMeta> | null = null;

interface P { mode: Mode; var: Variant; wk: number; dT?: number; sched?: Sched; closed?: boolean }

export async function run(c: { example: number; variant: string; p: P }, load: (name: string) => Pack<any>) {
  W ??= load('weights');
  X ??= load('examples');
  const p = c.p, u = c.example;
  const F = frame(X, X.meta.units[u].site, p.dT ?? 0, p.closed ? p.wk : null);
  const r = predict(W, X, F, u, 168 * p.wk, 168 * (p.wk + 1), p.mode, p.var, p.sched ?? 'unknown');
  return [...r.kW, ...r.sd];
}
