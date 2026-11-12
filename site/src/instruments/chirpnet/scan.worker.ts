// The live scan runs here, off the page's main thread: the same networks (net.ts, bns.ts) on every window of the span.
// Messages in:  {type:'init', base}  ·  {type:'scan', gen, mode:'bbh'|'bns', H, L, starts, win}
// Messages out: {type:'ready'}  ·  {type:'pt', gen, k, r}  ·  {type:'done', gen, ms}  ·  {type:'error', message}
import { fetchPack, type Pack } from '../../lib/nn/pack.ts';
import { read, type CMeta } from './net.ts';
import { bnsRead, type BMeta } from './bns.ts';
import { toRd } from './reading.ts';

let W: Pack<CMeta & { bns: BMeta }> | null = null;
let latest = 0;

const post = (m: unknown) => (self as unknown as { postMessage: (m: unknown) => void }).postMessage(m);

self.onmessage = async (ev: MessageEvent) => {
  const m = ev.data;
  try {
    if (m.type === 'init') {
      W = await fetchPack(`${m.base}weights`);
      post({ type: 'ready' });
      return;
    }
    if (m.type !== 'scan' || !W) return;
    latest = m.gen;
    const { H, L, starts, win } = m as { H: Float32Array; L: Float32Array; starts: number[]; win: number };
    let busy = 0;
    for (let k = 0; k < starts.length; k++) {
      await new Promise((r) => setTimeout(r, 0)); // let a newer scan request in
      if (m.gen !== latest) return;
      const data = new Float32Array(2 * win);
      data.set(H.subarray(starts[k], starts[k] + win), 0);
      data.set(L.subarray(starts[k], starts[k] + win), win);
      const t = performance.now();
      const x = { data, shape: [2, win] };
      const r = m.mode === 'bns' ? toRd(bnsRead(W, x)) : toRd(read(W, x));
      busy += performance.now() - t;
      post({ type: 'pt', gen: m.gen, k, r });
    }
    post({ type: 'done', gen: m.gen, ms: busy });
  } catch (e) {
    post({ type: 'error', message: String(e) });
  }
};
