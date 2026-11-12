// Read an mlratchet-pack (written by mlratchet/webexport.py): <name>.json manifest + <name>.bin data.
import type { Tensor } from './tensor.ts';

export interface PackManifest {
  format: string;
  tensors: Record<string, { dtype: 'f32' | 'f16' | 'i32' | 'u8'; shape: number[]; offset: number; bytes: number }>;
  meta: unknown;
}
export interface Pack<M = unknown> {
  t: Record<string, Tensor>;
  meta: M;
}

const f16tab = (() => {
  // exponent/mantissa decode table for speed
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    t[h] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
  }
  return t;
})();

export function parsePack<M = unknown>(manifest: PackManifest, buf: ArrayBuffer): Pack<M> {
  const out: Record<string, Tensor> = {};
  for (const [name, d] of Object.entries(manifest.tensors)) {
    let data: Float32Array;
    if (d.dtype === 'f32') data = new Float32Array(buf.slice(d.offset, d.offset + d.bytes));
    else if (d.dtype === 'f16') {
      const h = new Uint16Array(buf, d.offset, d.bytes / 2);
      data = new Float32Array(h.length);
      for (let i = 0; i < h.length; i++) data[i] = f16tab[h[i]];
    } else if (d.dtype === 'i32') data = Float32Array.from(new Int32Array(buf, d.offset, d.bytes / 4));
    else data = Float32Array.from(new Uint8Array(buf, d.offset, d.bytes));
    out[name] = { data, shape: d.shape };
  }
  return { t: out, meta: manifest.meta as M };
}

/** Browser loader: fetch `${url}.json` and `${url}.bin`. */
export async function fetchPack<M = unknown>(url: string, onProgress?: (frac: number) => void): Promise<Pack<M>> {
  const [mr, br] = await Promise.all([fetch(`${url}.json`), fetch(`${url}.bin`)]);
  if (!mr.ok || !br.ok) throw new Error(`could not load ${url}`);
  const manifest = (await mr.json()) as PackManifest;
  let buf: ArrayBuffer;
  const total = Number(br.headers.get('content-length')) || 0;
  if (onProgress && br.body) {
    // content-length may be the compressed size, so collect chunks and report a capped fraction
    const reader = br.body.getReader();
    const chunks: Uint8Array[] = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      onProgress(total ? Math.min(got / total, 0.99) : 0.5);
    }
    const bytes = new Uint8Array(got);
    let o = 0;
    for (const c of chunks) { bytes.set(c, o); o += c.length; }
    buf = bytes.buffer;
    onProgress(1);
  } else buf = await br.arrayBuffer();
  return parsePack<M>(manifest, buf);
}
