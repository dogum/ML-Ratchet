// Parity hook for site/tests/parity.test.ts: one reference case -> [finder 7, checker 7, posterior samples (8 x 6)].
import { finder, checker, peContext, peSample, WIN, type CMeta, type CExamples } from './net.ts';
import type { Pack } from '../../lib/nn/pack.ts';

let W: Pack<CMeta> | null = null;
let X: Pack<CExamples> | null = null;

export async function run(c: { example: number; variant: string; hs: number; ls: number }, load: (name: string) => Pack<any>) {
  W ??= load('weights');
  X ??= load('examples');
  const H = X.t[`h${c.example}`].data, L = X.t[`l${c.example}`].data;
  const data = new Float32Array(2 * WIN);
  data.set(H.subarray(c.hs, c.hs + WIN), 0);
  data.set(L.subarray(c.ls, c.ls + WIN), WIN);
  const x = { data, shape: [2, WIN] };
  const o = finder(W, x), k = checker(W, x);
  const s = peSample(W, peContext(W, x, X.meta.events[c.example].asd), X.t.z.data);
  const out = new Float32Array(14 + s.length);
  out.set(o, 0); out.set(k, 7); out.set(s, 14);
  return out;
}
