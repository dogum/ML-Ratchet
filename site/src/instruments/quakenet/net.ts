// QuakeNet in the browser: a line-for-line port of models/quakenet/quakenet.py forward() and predict_windows().
import { conv1d, gelu_, add_, stride1d, globalPool, linear, vcat, vgelu, concat0 } from '../../lib/nn/ops.ts';
import type { Pack } from '../../lib/nn/pack.ts';
import type { Tensor } from '../../lib/nn/tensor.ts';

export interface QMeta {
  n_blocks: number; n_head: number; mu_y: number[]; sd_y: number[];
  sigma_factor: { mw: number; dist: number; logz: number; baz: number };
}
export interface QExample {
  id: string; place: string; time: string; station: string; ls: number; peak: number; pre: number;
  truth: { mw: number; magType: string; depth: number; dist: number; baz: number };
}
export const L = 2460;

/** log10 of the 9-sample moving average of |x| (zero-padded), per channel — as in forward(). */
export function envelope(x: Tensor): Tensor {
  const [C, n] = x.shape;
  const out = new Float32Array(C * n);
  for (let c = 0; c < C; c++) {
    const row = c * n;
    let s = 0;
    for (let i = -4; i <= 4; i++) if (i >= 0 && i < n) s += Math.abs(x.data[row + i]);
    for (let j = 0; j < n; j++) {
      out[row + j] = Math.log10(s / 9 + 1e-3);
      const add = j + 5, sub = j - 4;
      if (add < n) s += Math.abs(x.data[row + add]);
      if (sub >= 0) s -= Math.abs(x.data[row + sub]);
    }
  }
  return { data: out, shape: [C, n] };
}

/** Raw network outputs (8 numbers) for x [3, 2460] and the two scalars s. */
export function forward(W: Pack<QMeta>, x: Tensor, s: [number, number]): Float32Array {
  const t = W.t, m = W.meta;
  let h = concat0([x, envelope(x)]);
  for (let i = 0; i < m.n_blocks; i++) {
    let y = gelu_(conv1d(h, t[`b${i}_c1_w`], t[`b${i}_c1_b`], 2));
    y = conv1d(y, t[`b${i}_c2_w`], t[`b${i}_c2_b`]);
    h = gelu_(add_(y, conv1d(stride1d(h, 2), t[`b${i}_sk_w`], t[`b${i}_sk_b`])));
  }
  const { mean, max } = globalPool(h);
  let z = vcat(mean, max, s);
  for (let j = 0; j < m.n_head; j++) {
    z = linear(z, t[`h${j}_w`], t[`h${j}_b`]);
    if (j < m.n_head - 1) vgelu(z);
  }
  return z;
}

export interface Reading { mw: number; mw_sig: number; dist: number; dist_sig: number; depth: number; logz: number; logz_sig: number; baz: number; kappa: number; half: number }

/** predict_windows() for one window, with the dev-fitted widening factors (calibrated=True). */
export function read(W: Pack<QMeta>, x: Tensor, ls: number, peak: number): { r: Reading; raw: Float32Array } {
  const s: [number, number] = [(ls + 6) / 2, Math.log10(Math.max(peak, 1))];
  const o = forward(W, x, s);
  const { mu_y: mu, sd_y: sd, sigma_factor: f } = W.meta;
  const F = [f.mw ?? 1, f.dist ?? 1, f.logz ?? 1];
  const val = (k: number) => o[k] * sd[k] + mu[k];
  const sig = (k: number) => Math.exp(Math.min(3, Math.max(-6, o[3 + k]))) * sd[k] * F[k];
  const kappa = Math.hypot(o[6], o[7]) / (f.baz ?? 1) ** 2;
  const logz = val(2);
  const r: Reading = {
    mw: val(0), mw_sig: sig(0), dist: val(1), dist_sig: sig(1),
    depth: 10 ** logz, logz, logz_sig: sig(2),
    baz: ((Math.atan2(o[6], o[7]) * 180) / Math.PI + 360) % 360,
    kappa, half: 57.29577951308232 / Math.sqrt(Math.max(kappa, 1e-3)),
  };
  return { r, raw: o };
}
