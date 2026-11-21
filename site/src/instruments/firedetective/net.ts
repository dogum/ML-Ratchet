// FireDetective in the browser: the 3-member 1-D CNN ensemble (port of models/firedetective/firedetective.py,
// FireDetective.raw and .predict) on the features of features.ts. Weights from
// site/public/instruments/firedetective/weights (float16 member tensors, float32 normalisation). No DOM.
import type { Pack } from '../../lib/nn/pack.ts';
import { featurize, NF, T_STEPS, type FireInput } from './features.ts';

export interface FDMeta {
  members: string[];
  calib: { hs_map: { a: number; b: number }; hs_widen: number; wind_widen: number; lfmc_widen: number };
  seq_names: string[]; stat_names: string[]; wind_cols: number[]; params: number; version: string;
}

const relu = (v: number) => (v > 0 ? v : 0);

function dense(x: Float64Array, w: Float32Array, b: Float32Array, act: boolean) {
  const nout = b.length, nin = x.length, o = new Float64Array(nout);
  for (let i = 0; i < nout; i++) {
    let s = 0;
    const off = i * nin;
    for (let j = 0; j < nin; j++) s += w[off + j] * x[j];
    s += b[i];
    o[i] = act ? relu(s) : s;
  }
  return o;
}

/** _conv1d (same padding) + ReLU + time mask: h [C][T] -> [O][T]. */
function conv(h: Float64Array[], w: { data: Float32Array; shape: number[] }, b: Float32Array, mask: number[]) {
  const [O, C, K] = w.shape, p = K >> 1, T = mask.length;
  const out: Float64Array[] = [];
  for (let o = 0; o < O; o++) {
    const row = new Float64Array(T);
    for (let t = 0; t < T; t++) {
      let s = 0;
      for (let k = 0; k < K; k++) {
        const j = t + k - p;
        if (j < 0 || j >= T) continue;
        const wb = o * C * K + k;
        for (let c = 0; c < C; c++) s += w.data[wb + c * K] * h[c][j];
      }
      row[t] = relu(s + b[o]) * mask[t];
    }
    out.push(row);
  }
  return out;
}

export interface Raw { mu: number[]; sd: number[]; p: number[] }

/** One member's forward pass (FireDetective._raw1) on a [T_STEPS * NF] sequence and 17 statics. */
function member(pack: Pack<FDMeta>, id: string, seq: Float64Array, st: Float64Array): Raw {
  const t = pack.t, W = (k: string) => t[`${id}.${k}`];
  const mu0 = t.seq_mu.data, sd0 = t.seq_sd.data, smu = t.st_mu.data, ssd = t.st_sd.data;
  const mask: number[] = Array.from({ length: T_STEPS }, (_, k) => (seq[k * NF] > 0 ? 1 : 0));
  const emb = W('emb.weight'), H = emb.shape[0];
  const h: Float64Array[] = Array.from({ length: H }, () => new Float64Array(T_STEPS));
  for (let k = 0; k < T_STEPS; k++) {
    const x = new Float64Array(NF);
    for (let f = 0; f < NF; f++) x[f] = ((seq[k * NF + f] - mu0[f]) / sd0[f]) * mask[k];
    const e = dense(x, emb.data, W('emb.bias').data, true);
    for (let i = 0; i < H; i++) h[i][k] = e[i] * mask[k];
  }
  const c1 = conv(h, W('c1.weight'), W('c1.bias').data, mask);
  const c2 = conv(c1, W('c2.weight'), W('c2.bias').data, mask);
  const C = c2.length, n = Math.max(mask.reduce((a, b) => a + b, 0), 1);
  const cat = new Float64Array(2 * C + W('st.bias').data.length);
  for (let c = 0; c < C; c++) {
    let s = 0, mx = -Infinity;
    for (let k = 0; k < T_STEPS; k++) { s += c2[c][k]; mx = Math.max(mx, c2[c][k] - 1e4 * (1 - mask[k])); }
    cat[c] = s / n;
    cat[C + c] = mx;
  }
  const xs = new Float64Array(st.length);
  for (let i = 0; i < st.length; i++) xs[i] = (st[i] - smu[i]) / ssd[i];
  cat.set(dense(xs, W('st.weight').data, W('st.bias').data, true), 2 * C);
  const z1 = dense(cat, W('f1.weight').data, W('f1.bias').data, true);
  const z2 = dense(z1, W('f2.weight').data, W('f2.bias').data, true);
  const mu = dense(z2, W('mu.weight').data, W('mu.bias').data, false);
  const ls = dense(z2, W('ls.weight').data, W('ls.bias').data, false);
  const bl = dense(z2, W('bn.weight').data, W('bn.bias').data, false);
  return { mu: Array.from(mu), sd: Array.from(ls, (v) => Math.exp(Math.min(Math.max(v, -5), 3))), p: Array.from(bl, (v) => 1 / (1 + Math.exp(-v))) };
}

/** FireDetective.raw: mean of member means; variance = mean member variance + spread of member means. */
export function raw(pack: Pack<FDMeta>, seq: Float64Array, st: Float64Array): Raw {
  const outs = pack.meta.members.map((id) => member(pack, id, seq, st));
  const K = outs.length, nt = outs[0].mu.length, np_ = outs[0].p.length;
  const mu: number[] = [], sd: number[] = [], p: number[] = [];
  for (let j = 0; j < nt; j++) {
    let m = 0, v = 0, sp = 0;
    for (const o of outs) { m += o.mu[j]; v += o.sd[j] ** 2; }
    m /= K; v /= K;
    for (const o of outs) sp += (o.mu[j] - m) ** 2;
    mu.push(m);
    sd.push(Math.sqrt(v + sp / K));
  }
  for (let j = 0; j < np_; j++) p.push(outs.reduce((a, o) => a + o.p[j], 0) / K);
  return { mu, sd, p };
}

export const OUTS = ['hs', 'hs_sd', 'crown', 'wind', 'wind_lo', 'wind_hi', 'lfmc', 'lfmc_sd', 'ros_adj', 'm1', 'wind_regime',
  'p_spot', 'p_ia'] as const;
export type Out = (typeof OUTS)[number];
export type Prediction = Record<Out, number>;

/** FireDetective.predict: readable outputs, dev-calibrated severity and widened error bars. */
export function predict(pack: Pack<FDMeta>, seq: Float64Array, seqH: Float64Array, st: Float64Array): Prediction {
  const c = pack.meta.calib;
  const a = raw(pack, seq, st), b = raw(pack, seqH, st);
  const e = Math.exp(a.mu[0]);
  const crown = (1.02 * e - 0.02) / (1 + e), dcdy = (1.04 * e) / (1 + e) ** 2;
  const hs = c.hs_map.a + c.hs_map.b * crown;
  const wsd = b.sd[1] * c.wind_widen;
  return {
    hs: Math.min(Math.max(hs, 0), 1), hs_sd: Math.abs(c.hs_map.b) * dcdy * a.sd[0] * c.hs_widen, crown,
    wind: Math.exp(b.mu[1]), wind_lo: Math.exp(b.mu[1] - wsd), wind_hi: Math.exp(b.mu[1] + wsd),
    lfmc: a.mu[2] * 100, lfmc_sd: a.sd[2] * 100 * c.lfmc_widen, ros_adj: Math.exp(a.mu[3]), m1: Math.exp(a.mu[4]),
    wind_regime: Math.min(Math.max(a.mu[6], 0), 1), p_spot: a.p[0], p_ia: a.p[1],
  };
}

export interface Reading { out: Prediction; seq: Float64Array; seqH: Float64Array; st: Float64Array; ms: number }

/** The whole pipeline the instrument runs on every change: raw input -> features -> network. */
export function read(pack: Pack<FDMeta>, f: FireInput): Reading {
  const t0 = performance.now();
  const { seq, seqH, st } = featurize(f);
  const out = predict(pack, seq, seqH, st);
  return { out, seq, seqH, st, ms: performance.now() - t0 };
}
