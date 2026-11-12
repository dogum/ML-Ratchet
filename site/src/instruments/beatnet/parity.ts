// Parity hook for site/tests/parity.test.ts: one reference case = one excerpt (raw or perturbed exactly as the
// instrument perturbs it), the whole front end redone here, the raw network outputs of every beat in the shown minute.
import { ecgOf, perturb, filt, features, forward, forwardFast, shownBeats, type BMeta, type BExample } from './net.ts';
import type { Pack } from '../../lib/nn/pack.ts';

let W: Pack<BMeta> | null = null;
let X: Pack<BExample[]> | null = null;

export async function run(c: { example: number; variant: string; noise: number; wander: number; seed: number }, load: (name: string) => Pack<any>) {
  const w: Pack<BMeta> = (W ??= load('weights'));
  X ??= load('examples');
  const e = X.meta[c.example], pos = X.t[`beats${c.example}`].data;
  const x = perturb(ecgOf(X, c.example), { noise: c.noise, wander: c.wander, seed: c.seed });
  const F = features(filt(x, w.meta), pos, shownBeats(pos, e.show), w.meta);
  // the fast network the instrument runs; for the first excerpt also the plain ops.ts port, which must agree
  const B = F.map((f) => forwardFast(w, f.w, f.rr));
  if (c.example === 0 && c.variant === 'raw') {
    for (let i = 0; i < F.length; i += 9) {
      const o = forward(w, { data: F[i].w, shape: [1, F[i].w.length] }, F[i].rr);
      for (let j = 0; j < o.length; j++) if (Math.abs(o[j] - B[i][j]) > 1e-4) throw new Error(`fast != plain at beat ${i}`);
    }
  }
  const out: number[] = [];
  for (const o of B) out.push(...o);
  return out;
}
