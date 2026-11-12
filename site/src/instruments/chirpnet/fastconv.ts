// Faster conv1d / conv2d for ChirpNet's networks (same numbers as ops.conv1d / conv2d up to summation order):
// im2col + a 4x4 register-tiled matrix multiply, with the weights repacked once per tensor.
import type { Tensor } from '../../lib/nn/tensor.ts';

interface Packed { O: number; CK: number; ob: number; w: Float32Array }
const cache = new WeakMap<Float32Array, Packed>();

/** Weights [O, C, K] (or [O, C, KH, KW], flattened per output) -> blocks of 4 output channels, laid out [block][p = c*K + k][4] (zero-padded). */
function pack(w: Tensor): Packed {
  let p = cache.get(w.data);
  if (p) return p;
  const O = w.shape[0], CK = w.data.length / O, ob = Math.ceil(O / 4);
  const out = new Float32Array(ob * CK * 4);
  for (let o = 0; o < O; o++) {
    const b = o >> 2, l = o & 3;
    for (let q = 0; q < CK; q++) out[(b * CK + q) * 4 + l] = w.data[o * CK + q];
  }
  p = { O, CK, ob, w: out };
  cache.set(w.data, p);
  return p;
}

let colBuf = new Float32Array(1 << 16);

/** out[o][j] = bias[o] + sum_q W[o][q] col[q][j] for the packed weights; col rows are Lp long (Lp % 4 == 0). */
function gemm(P: Packed, col: Float32Array, Lp: number, Lout: number, b: Tensor | null): Float32Array {
  const CK = P.CK, O = P.O;
  const out = new Float32Array(O * Lout);
  const wp = P.w, bd = b ? b.data : null;
  for (let blk = 0; blk < P.ob; blk++) {
    const wb = blk * CK * 4;
    const o0 = blk * 4;
    const b0 = bd && o0 < O ? bd[o0] : 0, b1 = bd && o0 + 1 < O ? bd[o0 + 1] : 0, b2 = bd && o0 + 2 < O ? bd[o0 + 2] : 0, b3 = bd && o0 + 3 < O ? bd[o0 + 3] : 0;
    for (let j = 0; j < Lp; j += 4) {
      let a00 = b0, a01 = b0, a02 = b0, a03 = b0;
      let a10 = b1, a11 = b1, a12 = b1, a13 = b1;
      let a20 = b2, a21 = b2, a22 = b2, a23 = b2;
      let a30 = b3, a31 = b3, a32 = b3, a33 = b3;
      for (let q = 0, wi = wb, ci = j; q < CK; q++, wi += 4, ci += Lp) {
        const x0 = col[ci], x1 = col[ci + 1], x2 = col[ci + 2], x3 = col[ci + 3];
        const w0 = wp[wi], w1 = wp[wi + 1], w2 = wp[wi + 2], w3 = wp[wi + 3];
        a00 += w0 * x0; a01 += w0 * x1; a02 += w0 * x2; a03 += w0 * x3;
        a10 += w1 * x0; a11 += w1 * x1; a12 += w1 * x2; a13 += w1 * x3;
        a20 += w2 * x0; a21 += w2 * x1; a22 += w2 * x2; a23 += w2 * x3;
        a30 += w3 * x0; a31 += w3 * x1; a32 += w3 * x2; a33 += w3 * x3;
      }
      const put = (o: number, v0: number, v1: number, v2: number, v3: number) => {
        if (o >= O) return;
        const r = o * Lout + j;
        out[r] = v0;
        if (j + 1 < Lout) out[r + 1] = v1;
        if (j + 2 < Lout) out[r + 2] = v2;
        if (j + 3 < Lout) out[r + 3] = v3;
      };
      put(o0, a00, a01, a02, a03); put(o0 + 1, a10, a11, a12, a13); put(o0 + 2, a20, a21, a22, a23); put(o0 + 3, a30, a31, a32, a33);
    }
  }
  return out;
}


/** y = conv1d(x [C, L], w [O, C, K], b [O]) with zero padding `pad` and `stride`, as numpy's _conv. */
export function conv1dFast(x: Tensor, w: Tensor, b: Tensor | null, stride: number, pad: number): Tensor {
  const [C, L] = x.shape;
  const K = w.shape[2];
  const Lout = Math.floor((L + 2 * pad - K) / stride) + 1;
  const P = pack(w), CK = P.CK, O = P.O;
  const Lp = (Lout + 3) & ~3; // pad columns to a multiple of 4
  const need = CK * Lp;
  if (colBuf.length < need) colBuf = new Float32Array(need);
  const col = colBuf, xd = x.data;
  // im2col: col[c*K + k][j] = x[c][j*stride + k - pad] (0 outside)
  for (let c = 0; c < C; c++) {
    const xr = c * L;
    for (let k = 0; k < K; k++) {
      const row = (c * K + k) * Lp;
      const sh = k - pad;
      for (let j = 0; j < Lp; j++) {
        const i = j * stride + sh;
        col[row + j] = j < Lout && i >= 0 && i < L ? xd[xr + i] : 0;
      }
    }
  }
  return { data: gemm(P, col, Lp, Lout, b), shape: [O, Lout] };
}

/** Batched dense layer: X [n, I] (row-major), W [O, I] (PyTorch layout), bias [O] -> Y [n, O] = X W^T + bias. 4x4 tiles. */
export function linearBatch(X: Float32Array, n: number, I: number, W: Float32Array, O: number, bias: ArrayLike<number> | null): Float32Array {
  const Y = new Float32Array(n * O);
  const n4 = n & ~3, O4 = O & ~3;
  for (let r = 0; r < n4; r += 4) {
    const x0 = r * I, x1 = x0 + I, x2 = x1 + I, x3 = x2 + I;
    for (let o = 0; o < O4; o += 4) {
      const w0 = o * I, w1 = w0 + I, w2 = w1 + I, w3 = w2 + I;
      let a00 = 0, a01 = 0, a02 = 0, a03 = 0, a10 = 0, a11 = 0, a12 = 0, a13 = 0;
      let a20 = 0, a21 = 0, a22 = 0, a23 = 0, a30 = 0, a31 = 0, a32 = 0, a33 = 0;
      for (let i = 0; i < I; i++) {
        const p0 = X[x0 + i], p1 = X[x1 + i], p2 = X[x2 + i], p3 = X[x3 + i];
        const q0 = W[w0 + i], q1 = W[w1 + i], q2 = W[w2 + i], q3 = W[w3 + i];
        a00 += p0 * q0; a01 += p0 * q1; a02 += p0 * q2; a03 += p0 * q3;
        a10 += p1 * q0; a11 += p1 * q1; a12 += p1 * q2; a13 += p1 * q3;
        a20 += p2 * q0; a21 += p2 * q1; a22 += p2 * q2; a23 += p2 * q3;
        a30 += p3 * q0; a31 += p3 * q1; a32 += p3 * q2; a33 += p3 * q3;
      }
      const b0 = bias ? bias[o] : 0, b1 = bias ? bias[o + 1] : 0, b2 = bias ? bias[o + 2] : 0, b3 = bias ? bias[o + 3] : 0;
      let y = r * O + o;
      Y[y] = a00 + b0; Y[y + 1] = a01 + b1; Y[y + 2] = a02 + b2; Y[y + 3] = a03 + b3; y += O;
      Y[y] = a10 + b0; Y[y + 1] = a11 + b1; Y[y + 2] = a12 + b2; Y[y + 3] = a13 + b3; y += O;
      Y[y] = a20 + b0; Y[y + 1] = a21 + b1; Y[y + 2] = a22 + b2; Y[y + 3] = a23 + b3; y += O;
      Y[y] = a30 + b0; Y[y + 1] = a31 + b1; Y[y + 2] = a32 + b2; Y[y + 3] = a33 + b3;
    }
  }
  // remainders (rows beyond n4, or outputs beyond O4)
  for (let r = 0; r < n; r++) {
    for (let o = r < n4 ? O4 : 0; o < O; o++) {
      let s = bias ? bias[o] : 0;
      const xr = r * I, wr = o * I;
      for (let i = 0; i < I; i++) s += X[xr + i] * W[wr + i];
      Y[r * O + o] = s;
    }
  }
  return Y;
}

/** y = conv2d(x [C, H, W], w [O, C, KH, KW], b [O]) with zero padding [ph, pw] and stride [sh, sw], as numpy's _conv2d. */
export function conv2dFast(x: Tensor, w: Tensor, b: Tensor | null, stride: [number, number], pad: [number, number]): Tensor {
  const [C, H, W] = x.shape;
  const [, , KH, KW] = w.shape;
  const [sh, sw] = stride, [ph, pw] = pad;
  const Ho = Math.floor((H + 2 * ph - KH) / sh) + 1, Wo = Math.floor((W + 2 * pw - KW) / sw) + 1;
  const N = Ho * Wo, Lp = (N + 3) & ~3;
  const P = pack(w);
  const need = P.CK * Lp;
  if (colBuf.length < need) colBuf = new Float32Array(need);
  const col = colBuf, xd = x.data;
  for (let c = 0; c < C; c++) {
    for (let kh = 0; kh < KH; kh++) {
      for (let kw = 0; kw < KW; kw++) {
        const row = ((c * KH + kh) * KW + kw) * Lp;
        for (let i = 0; i < Ho; i++) {
          const y = i * sh + kh - ph, base = row + i * Wo;
          if (y < 0 || y >= H) { col.fill(0, base, base + Wo); continue; }
          const xr = (c * H + y) * W;
          for (let j = 0; j < Wo; j++) {
            const xx = j * sw + kw - pw;
            col[base + j] = xx >= 0 && xx < W ? xd[xr + xx] : 0;
          }
        }
        col.fill(0, row + N, row + Lp);
      }
    }
  }
  return { data: gemm(P, col, Lp, N, b), shape: [P.O, Ho, Wo] };
}
