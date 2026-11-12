// Parity hook for site/tests/parity.test.ts: run one reference case, return the raw network outputs.
import { forward, features, lifeS, L, type CMeta, type CExamples } from './net.ts';
import type { Pack } from '../../lib/nn/pack.ts';
import { vcat } from '../../lib/nn/ops.ts';

let W: Pack<CMeta> | null = null;
let X: Pack<CExamples> | null = null;

const row = (t: { data: Float32Array }, i: number) => t.data.slice(i * L, (i + 1) * L);

export async function run(c: { example: number; variant: string; stop?: number }, load: (name: string) => Pack<any>) {
  W ??= load('weights');
  X ??= load('examples');
  const i = c.example;
  const Q = row(X.t.Q, i), T = row(X.t.T, i), Q0 = row(X.t.Q, 0), T0 = row(X.t.T, 0);
  const S = lifeS(X.meta.cycles[i], X.meta.charge_time);
  const sl = W.meta.life.smooth ?? 0, sp = W.meta.partial.smooth ?? 0;
  if (c.variant === 'full') return vcat(forward(W, 'health', features(Q, T, Q0, T0)), forward(W, 'life', features(Q, T, Q0, T0, sl), S));
  if (c.variant === 'noref') return vcat(forward(W, 'health', features(Q, T, null, null)), forward(W, 'life', features(Q, T, null, null, sl), S));
  // cutNNN: a partial discharge stopped at c.stop Ah, everything after it thrown away
  for (let k = 0; k < L; k++) if (Q[k] > c.stop!) { Q[k] = NaN; T[k] = NaN; }
  return forward(W, 'partial', features(Q, T, Q0, T0, sp));
}
