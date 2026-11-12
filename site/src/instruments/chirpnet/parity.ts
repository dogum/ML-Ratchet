// Parity hook for site/tests/parity.test.ts.
// Black-hole cases -> [finder 7, checker 7, posterior samples (8 x 6)]; GW170817 cases ('bns*') -> ChirpNetBNS's 7 outputs,
// on the 46 s span gated in the browser exactly as chirpnet.gate does it.
import { finder, checker, peContext, peSample, WIN, type CMeta, type CExamples } from './net.ts';
import { bnsForward, gate, WINB, type BMeta } from './bns.ts';
import type { Pack } from '../../lib/nn/pack.ts';

let W: Pack<CMeta & { bns: BMeta }> | null = null;
let X: Pack<CExamples> | null = null;

interface Case { example: number; variant: string; hs: number; ls: number; l?: string; gate?: boolean }

export async function run(c: Case, load: (name: string) => Pack<any>) {
  W ??= load('weights');
  X ??= load('examples');
  if (c.variant.startsWith('bns')) {
    const g = (a: Float32Array) => (c.gate ? gate(a).x : a);
    const H = g(X.t.n_H.data), L = g(X.t[`n_${c.l}`].data);
    const data = new Float32Array(2 * WINB);
    data.set(H.subarray(c.hs, c.hs + WINB), 0);
    data.set(L.subarray(c.ls, c.ls + WINB), WINB);
    return bnsForward(W, { data, shape: [2, WINB] });
  }
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
