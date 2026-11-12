// numpy-equivalent layers for the collection's exported networks (PyTorch layouts, BatchNorm already folded).
// Shapes: 1-D activations [C, L] (batch of one), 2-D [C, H, W]; conv weights [O, C, K] / [O, C, KH, KW].
import { type Tensor, size, tensor } from './tensor.ts';

/** erf, Abramowitz & Stegun 7.1.26 — the same formula the Python modules use, so results match bit-for-bit closely. */
export function erf(x: number): number {
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return x < 0 ? -y : y;
}

export function gelu(t: Tensor): Tensor {
  const o = new Float32Array(t.data.length);
  for (let i = 0; i < o.length; i++) { const x = t.data[i]; o[i] = 0.5 * x * (1 + erf(x * 0.7071067811865476)); }
  return { data: o, shape: [...t.shape] };
}

export function gelu_(t: Tensor): Tensor {
  const d = t.data;
  for (let i = 0; i < d.length; i++) { const x = d[i]; d[i] = 0.5 * x * (1 + erf(x * 0.7071067811865476)); }
  return t;
}

export function relu_(t: Tensor): Tensor {
  const d = t.data;
  for (let i = 0; i < d.length; i++) if (d[i] < 0) d[i] = 0;
  return t;
}

export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export function softmax(v: ArrayLike<number>): number[] {
  let m = -Infinity;
  for (let i = 0; i < v.length; i++) m = Math.max(m, v[i]);
  const e = Array.from(v, (x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

/**
 * 1-D cross-correlation. x [C, L], w [O, C, K], b [O].
 * pad = 'same' pads K//2 on both sides (as the collection's numpy `_conv`), or give a number.
 */
export function conv1d(x: Tensor, w: Tensor, b: Tensor | null, stride = 1, pad: number | 'same' = 'same', dilation = 1): Tensor {
  const [C, L] = x.shape;
  const [O, Cw, K] = w.shape;
  if (C !== Cw) throw new Error(`conv1d: input ${C} channels, weight ${Cw}`);
  const p = pad === 'same' ? Math.floor(K / 2) * dilation : pad;
  const span = dilation * (K - 1) + 1;
  const Lout = Math.floor((L + 2 * p - span) / stride) + 1;
  const out = new Float32Array(O * Lout);
  const xd = x.data, wd = w.data;
  for (let o = 0; o < O; o++) {
    const orow = o * Lout;
    const bias = b ? b.data[o] : 0;
    for (let j = 0; j < Lout; j++) out[orow + j] = bias;
    for (let c = 0; c < C; c++) {
      const xrow = c * L;
      const wbase = (o * C + c) * K;
      for (let k = 0; k < K; k++) {
        const wv = wd[wbase + k];
        if (wv === 0) continue;
        const shift = k * dilation - p;
        // j range so that 0 <= j*stride + shift < L
        let j0 = shift >= 0 ? 0 : Math.ceil(-shift / stride);
        let j1 = Math.min(Lout, Math.floor((L - 1 - shift) / stride) + 1);
        if (stride === 1) {
          for (let j = j0, xi = xrow + j0 + shift; j < j1; j++, xi++) out[orow + j] += wv * xd[xi];
        } else {
          for (let j = j0, xi = xrow + j0 * stride + shift; j < j1; j++, xi += stride) out[orow + j] += wv * xd[xi];
        }
      }
    }
  }
  return { data: out, shape: [O, Lout] };
}

/** 2-D cross-correlation. x [C, H, W], w [O, C, KH, KW]. pad 'same' = floor(K/2) per axis. */
export function conv2d(x: Tensor, w: Tensor, b: Tensor | null, stride: number | [number, number] = 1, pad: 'same' | number | [number, number] = 'same'): Tensor {
  const [C, H, W] = x.shape;
  const [O, Cw, KH, KW] = w.shape;
  if (C !== Cw) throw new Error(`conv2d: input ${C} channels, weight ${Cw}`);
  const [sh, sw] = typeof stride === 'number' ? [stride, stride] : stride;
  const [ph, pw] = pad === 'same' ? [Math.floor(KH / 2), Math.floor(KW / 2)] : typeof pad === 'number' ? [pad, pad] : pad;
  const Ho = Math.floor((H + 2 * ph - KH) / sh) + 1;
  const Wo = Math.floor((W + 2 * pw - KW) / sw) + 1;
  const out = new Float32Array(O * Ho * Wo);
  const xd = x.data, wd = w.data;
  for (let o = 0; o < O; o++) {
    const ob = o * Ho * Wo;
    const bias = b ? b.data[o] : 0;
    for (let i = 0; i < Ho * Wo; i++) out[ob + i] = bias;
    for (let c = 0; c < C; c++) {
      const xb = c * H * W;
      for (let kh = 0; kh < KH; kh++) {
        for (let kw = 0; kw < KW; kw++) {
          const wv = wd[((o * C + c) * KH + kh) * KW + kw];
          if (wv === 0) continue;
          for (let i = 0; i < Ho; i++) {
            const y = i * sh + kh - ph;
            if (y < 0 || y >= H) continue;
            const xrow = xb + y * W;
            const orow = ob + i * Wo;
            for (let j = 0; j < Wo; j++) {
              const xx = j * sw + kw - pw;
              if (xx >= 0 && xx < W) out[orow + j] += wv * xd[xrow + xx];
            }
          }
        }
      }
    }
  }
  return { data: out, shape: [O, Ho, Wo] };
}

/** y = W x + b for a vector x, PyTorch layout W [out, in]. */
export function linear(x: ArrayLike<number>, w: Tensor, b: Tensor | null): Float32Array {
  const [O, I] = w.shape;
  if (x.length !== I) throw new Error(`linear: input ${x.length}, weight expects ${I}`);
  const out = new Float32Array(O);
  const wd = w.data;
  for (let o = 0; o < O; o++) {
    let s = b ? b.data[o] : 0;
    const base = o * I;
    for (let i = 0; i < I; i++) s += wd[base + i] * x[i];
    out[o] = s;
  }
  return out;
}

export function add_(a: Tensor, b: Tensor): Tensor {
  if (a.data.length !== b.data.length) throw new Error(`add: [${a.shape}] + [${b.shape}]`);
  for (let i = 0; i < a.data.length; i++) a.data[i] += b.data[i];
  return a;
}

/** Concatenate along axis 0 (channels). */
export function concat0(ts: Tensor[]): Tensor {
  const inner = ts[0].shape.slice(1);
  const n = ts.reduce((s, t) => s + t.shape[0], 0);
  const out = new Float32Array(n * size(inner));
  let off = 0;
  for (const t of ts) { out.set(t.data, off); off += t.data.length; }
  return { data: out, shape: [n, ...inner] };
}

/** Every `step`-th sample along the last axis (numpy x[..., ::step]). */
export function stride1d(x: Tensor, step: number): Tensor {
  const [C, L] = x.shape;
  const Lo = Math.ceil(L / step);
  const out = new Float32Array(C * Lo);
  for (let c = 0; c < C; c++) for (let j = 0; j < Lo; j++) out[c * Lo + j] = x.data[c * L + j * step];
  return { data: out, shape: [C, Lo] };
}

/** Mean and max over everything after the channel axis -> two vectors of length C. */
export function globalPool(x: Tensor): { mean: Float32Array; max: Float32Array } {
  const C = x.shape[0];
  const n = x.data.length / C;
  const mean = new Float32Array(C), max = new Float32Array(C);
  for (let c = 0; c < C; c++) {
    let s = 0, m = -Infinity;
    for (let i = c * n; i < (c + 1) * n; i++) { const v = x.data[i]; s += v; if (v > m) m = v; }
    mean[c] = s / n; max[c] = m;
  }
  return { mean, max };
}

export function vcat(...parts: ArrayLike<number>[]): Float32Array {
  const out = new Float32Array(parts.reduce((s, p) => s + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p as ArrayLike<number>, o); o += p.length; }
  return out;
}

export const vgelu = (v: Float32Array) => { for (let i = 0; i < v.length; i++) v[i] = 0.5 * v[i] * (1 + erf(v[i] * 0.7071067811865476)); return v; };

export { tensor };
