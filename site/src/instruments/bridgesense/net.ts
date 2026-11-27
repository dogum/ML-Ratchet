// Pure TypeScript port of models/bridgesense/bridgesense.py (Dual-Axis Transformer).
// No DOM imports so site/tests/parity.test.ts can run this directly in Node.
import { erf, sigmoid } from '../../lib/nn/ops.ts';
import type { Pack } from '../../lib/nn/pack.ts';

export const N_STEPS = 32;
export const MAX_MODES = 14;

export interface BSConfig {
  d_model: number;
  n_heads: number;
  d_ff: number;
  n_t_layers: number;
  n_m_layers: number;
  use_cross_attn: boolean;
  use_anchor: boolean;
}

export interface BSMeta {
  model: string;
  version: string;
  n_members: number;
  total_params: number;
  config: BSConfig;
  configs?: BSConfig[];
  widen_dk: number;
  widen_env: number;
  prob_temp: number;
}

export interface BSExample {
  idx: number;
  id: string;
  bridge: string;
  bridge_label: string;
  stratum: string;
  is_changed: number;
  true_dk_pct: number | null;
  n_modes: number;
  T_min: number;
  T_mean: number;
  truth_label: string;
  licence: string;
  python: {
    p_change: number;
    delta_k_pct: number;
    delta_k_sigma: number;
    thermal_shift_pct: number;
    thermal_shift_sigma: number;
    anchor_dk_pct: number;
  };
}

export interface BSReading {
  logit_raw: number;
  p_change: number;
  delta_k_pct: number;
  delta_k_sigma: number;
  thermal_shift_pct: number;
  thermal_shift_sigma: number;
  anchor_dk_pct: number;
  anchor_env_pct: number;
  mode_dk_pct: Float32Array;
  active_modes: Uint8Array;
}

const SQRT1_2 = 0.7071067811865476;
const clip = (x: number, lo: number, hi: number) => (x < lo ? lo : x > hi ? hi : x);

function geluInplace(a: Float32Array): Float32Array {
  for (let i = 0; i < a.length; i++) {
    const x = a[i];
    a[i] = 0.5 * x * (1 + erf(x * SQRT1_2));
  }
  return a;
}

function layerNormRows(x: Float32Array, rows: number, D: number, w: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(rows * D);
  for (let r = 0; r < rows; r++) {
    const off = r * D;
    let mu = 0;
    for (let j = 0; j < D; j++) mu += x[off + j];
    mu /= D;
    let v = 0;
    for (let j = 0; j < D; j++) { const d = x[off + j] - mu; v += d * d; }
    const inv = 1 / Math.sqrt(v / D + 1e-5);
    for (let j = 0; j < D; j++) out[off + j] = (x[off + j] - mu) * inv * w[j] + b[j];
  }
  return out;
}

function linearRows(x: Float32Array, rows: number, inD: number, w: Float32Array, b: Float32Array, outD: number): Float32Array {
  const out = new Float32Array(rows * outD);
  for (let r = 0; r < rows; r++) {
    const xoff = r * inD;
    const ooff = r * outD;
    for (let o = 0; o < outD; o++) {
      const woff = o * inD;
      let s = b[o];
      for (let i = 0; i < inD; i++) s += x[xoff + i] * w[woff + i];
      out[ooff + o] = s;
    }
  }
  return out;
}

function mha(
  qIn: Float32Array, Lq: number,
  kvIn: Float32Array, Lk: number,
  D: number, nHeads: number,
  W: Pack<BSMeta>['t'], prefix: string,
  kvMask?: Float32Array,
): Float32Array {
  const dh = D / nHeads;
  const scale = 1 / Math.sqrt(dh);
  const Q = linearRows(qIn, Lq, D, W[`${prefix}.q_proj.weight`].data, W[`${prefix}.q_proj.bias`].data, D);
  const K = linearRows(kvIn, Lk, D, W[`${prefix}.k_proj.weight`].data, W[`${prefix}.k_proj.bias`].data, D);
  const V = linearRows(kvIn, Lk, D, W[`${prefix}.v_proj.weight`].data, W[`${prefix}.v_proj.bias`].data, D);

  const ctx = new Float32Array(Lq * D);
  const scores = new Float64Array(Lk);
  for (let h = 0; h < nHeads; h++) {
    const hOff = h * dh;
    for (let i = 0; i < Lq; i++) {
      let mx = -Infinity;
      const qiBase = i * D + hOff;
      for (let j = 0; j < Lk; j++) {
        if (kvMask && kvMask[j] < 0.5) { scores[j] = -1e9; continue; }
        const kjBase = j * D + hOff;
        let dot = 0;
        for (let k = 0; k < dh; k++) dot += Q[qiBase + k] * K[kjBase + k];
        const s = dot * scale;
        scores[j] = s;
        if (s > mx) mx = s;
      }
      let denom = 0;
      for (let j = 0; j < Lk; j++) { const e = Math.exp(scores[j] - mx); scores[j] = e; denom += e; }
      const oiBase = i * D + hOff;
      for (let j = 0; j < Lk; j++) {
        const w = scores[j] / denom;
        const vjBase = j * D + hOff;
        for (let k = 0; k < dh; k++) ctx[oiBase + k] += w * V[vjBase + k];
      }
    }
  }
  return linearRows(ctx, Lq, D, W[`${prefix}.o_proj.weight`].data, W[`${prefix}.o_proj.bias`].data, D);
}

function safeCorr(x: Float64Array, y: Float64Array, mask: Float64Array): number {
  let cnt = 0, mx = 0, my = 0;
  for (let i = 0; i < x.length; i++) {
    if (mask[i] > 0.5 && Number.isFinite(x[i]) && Number.isFinite(y[i])) { cnt++; mx += x[i]; my += y[i]; }
  }
  if (cnt < 4) return 0;
  mx /= cnt; my /= cnt;
  let vx = 0, vy = 0, cov = 0;
  for (let i = 0; i < x.length; i++) {
    if (mask[i] > 0.5 && Number.isFinite(x[i]) && Number.isFinite(y[i])) {
      const dx = x[i] - mx, dy = y[i] - my;
      vx += dx * dx; vy += dy * dy; cov += dx * dy;
    }
  }
  const den = Math.sqrt(vx * vy);
  return den < 1e-8 ? 0 : clip(cov / den, -1, 1);
}

export function extractFeatures(
  T_ref: Float32Array, T_q: Float32Array,
  df_ref: Float32Array, df_q: Float32Array,
  mask_ref: Float32Array, mask_q: Float32Array,
  N_ref = N_STEPS, N_q = N_STEPS, M = MAX_MODES,
) {
  const Tr0 = new Float64Array(N_ref);
  const Tq0 = new Float64Array(N_q);
  for (let i = 0; i < N_ref; i++) Tr0[i] = T_ref[i * 3];
  for (let i = 0; i < N_q; i++) Tq0[i] = T_q[i * 3];

  let T_ref_mean = 0, T_ref_min = Infinity, T_ref_max = -Infinity;
  for (let i = 0; i < N_ref; i++) {
    T_ref_mean += Tr0[i];
    if (Tr0[i] < T_ref_min) T_ref_min = Tr0[i];
    if (Tr0[i] > T_ref_max) T_ref_max = Tr0[i];
  }
  T_ref_mean /= N_ref;

  let T_q_mean = 0, T_q_min = Infinity, T_q_max = -Infinity, frzQ = 0, frzR = 0;
  for (let i = 0; i < N_q; i++) {
    T_q_mean += Tq0[i];
    if (Tq0[i] < T_q_min) T_q_min = Tq0[i];
    if (Tq0[i] > T_q_max) T_q_max = Tq0[i];
    if (Tq0[i] < 1.0) frzQ++;
  }
  for (let i = 0; i < N_ref; i++) if (Tr0[i] < 1.0) frzR++;
  T_q_mean /= N_q;
  const freeze_frac_q = frzQ / N_q;
  const freeze_frac_ref = frzR / N_ref;

  const slopes = new Float64Array(M);
  const intercepts = new Float64Array(M);
  const ref_std = new Float64Array(M).fill(0.005);
  const refCounts = new Int32Array(M);
  const qCounts = new Int32Array(M);

  for (let m = 0; m < M; m++) {
    let cnt = 0;
    for (let i = 0; i < N_ref; i++) if (mask_ref[i * M + m] > 0.5) cnt++;
    refCounts[m] = cnt;
    if (cnt >= 4) {
      // OLS with intercept, centred on this mode's valid readings (two passes, as in bridgesense.py)
      let sx = 0, sy = 0;
      for (let i = 0; i < N_ref; i++) {
        if (mask_ref[i * M + m] > 0.5) { sx += Tr0[i] - T_ref_mean; sy += df_ref[i * M + m]; }
      }
      const xm = sx / cnt, ym = sy / cnt;
      let vx = 0, xy = 0;
      for (let i = 0; i < N_ref; i++) {
        if (mask_ref[i * M + m] > 0.5) {
          const xc = Tr0[i] - T_ref_mean - xm;
          const yc = df_ref[i * M + m] - ym;
          vx += xc * xc; xy += xc * yc;
        }
      }
      const a = vx > 1e-4 ? clip(xy / (vx + 1e-2), -0.015, 0.005) : 0;
      const b = ym - a * xm;
      slopes[m] = a; intercepts[m] = b;
      let mr = 0;
      for (let i = 0; i < N_ref; i++) if (mask_ref[i * M + m] > 0.5) mr += df_ref[i * M + m] - (a * (Tr0[i] - T_ref_mean) + b);
      mr /= cnt;
      let vr = 0;
      for (let i = 0; i < N_ref; i++) if (mask_ref[i * M + m] > 0.5) {
        const d = (df_ref[i * M + m] - (a * (Tr0[i] - T_ref_mean) + b)) - mr;
        vr += d * d;
      }
      ref_std[m] = Math.max(Math.sqrt(vr / cnt), 5e-4);
    } else if (cnt >= 1) {
      let sy = 0;
      for (let i = 0; i < N_ref; i++) if (mask_ref[i * M + m] > 0.5) sy += df_ref[i * M + m];
      intercepts[m] = sy / cnt;
    }
  }

  const resid_ref = new Float64Array(N_ref * M);
  const resid_q = new Float64Array(N_q * M);
  for (let i = 0; i < N_ref; i++) {
    for (let m = 0; m < M; m++) {
      if (mask_ref[i * M + m] > 0.5) resid_ref[i * M + m] = df_ref[i * M + m] - (intercepts[m] + slopes[m] * (Tr0[i] - T_ref_mean));
    }
  }
  const mode_mean_resid = new Float64Array(M);
  const mode_mean_raw = new Float64Array(M);
  for (let m = 0; m < M; m++) {
    let cnt = 0, sr = 0, sraw = 0;
    for (let i = 0; i < N_q; i++) {
      if (mask_q[i * M + m] > 0.5) {
        const rv = df_q[i * M + m] - (intercepts[m] + slopes[m] * (Tq0[i] - T_ref_mean));
        resid_q[i * M + m] = rv;
        sr += rv; sraw += df_q[i * M + m]; cnt++;
      }
    }
    qCounts[m] = cnt;
    if (cnt > 0) { mode_mean_resid[m] = sr / cnt; mode_mean_raw[m] = sraw / cnt; }
  }

  // Global summaries use the same eligibility as the mode mask (>= 3 ref and >= 2 query readings)
  const active = new Uint8Array(M);
  const m_mask = new Float32Array(M);
  let nAct = 0;
  for (let m = 0; m < M; m++) {
    if (refCounts[m] >= 3 && qCounts[m] >= 2) { active[m] = 1; m_mask[m] = 1; nAct++; }
  }
  let actQCnt = 0, sumResQ = 0, sumRawQ = 0, maxAbsZ = 0, sumAbsZ = 0;
  for (let m = 0; m < M; m++) {
    if (active[m]) {
      actQCnt++; sumResQ += mode_mean_resid[m]; sumRawQ += mode_mean_raw[m];
      const az = Math.abs(mode_mean_resid[m] / ref_std[m]);
      if (az > maxAbsZ) maxAbsZ = az;
      sumAbsZ += az;
    }
  }
  let anchor_dk_pct = 0, anchor_env_pct = 0, mean_abs_z = 0, resid_std_modes = 0;
  if (actQCnt > 0) {
    const mr = sumResQ / actQCnt;
    anchor_dk_pct = 200 * mr;
    anchor_env_pct = 100 * (sumRawQ / actQCnt - mr);
    mean_abs_z = sumAbsZ / actQCnt;
    if (actQCnt >= 2) {
      let vr = 0;
      for (let m = 0; m < M; m++) if (active[m]) { const d = mode_mean_resid[m] - mr; vr += d * d; }
      resid_std_modes = Math.sqrt(vr / actQCnt);
    }
  }

  const nActSafe = Math.max(nAct, 1);

  function buildTime(T3: Float32Array, df: Float32Array, resid: Float64Array, msk: Float32Array, N: number): Float32Array {
    const out = new Float32Array(N * 10);
    for (let i = 0; i < N; i++) {
      let cnt = 0, sdf = 0, sres = 0;
      for (let m = 0; m < M; m++) {
        if (active[m] && msk[i * M + m] > 0.5) { cnt++; sdf += df[i * M + m]; sres += resid[i * M + m]; }
      }
      const mdf = cnt > 0 ? sdf / cnt : 0;
      const mres = cnt > 0 ? sres / cnt : 0;
      let sq = 0;
      if (cnt > 0) {
        for (let m = 0; m < M; m++) if (active[m] && msk[i * M + m] > 0.5) { const d = df[i * M + m] - mdf; sq += d * d; }
      }
      const stdf = cnt > 0 ? Math.sqrt(sq / cnt) : 0;
      const t0 = T3[i * 3], t24 = T3[i * 3 + 1], t72 = T3[i * 3 + 2];
      const o = i * 10;
      out[o] = (t0 - 10) / 15;
      out[o + 1] = (t24 - 10) / 15;
      out[o + 2] = (t72 - 10) / 15;
      out[o + 3] = clip((t0 - t24) / 10, -3, 3);
      out[o + 4] = Math.max(0, 2 - t0) / 10;
      out[o + 5] = Math.max(0, 2 - t24) / 10;
      out[o + 6] = cnt > 0 ? clip(mdf * 20, -5, 5) : 0;
      out[o + 7] = cnt > 0 ? clip(stdf * 20, 0, 5) : 0;
      out[o + 8] = cnt > 0 ? clip(mres * 20, -5, 5) : 0;
      out[o + 9] = cnt / nActSafe;
    }
    return out;
  }

  const x_ref = buildTime(T_ref, df_ref, resid_ref, mask_ref, N_ref);
  const x_q = buildTime(T_q, df_q, resid_q, mask_q, N_q);

  let bMeanRes = 0, bMeanRaw = 0;
  if (nAct > 0) {
    for (let m = 0; m < M; m++) if (active[m]) { bMeanRes += mode_mean_resid[m]; bMeanRaw += mode_mean_raw[m]; }
    bMeanRes /= nAct; bMeanRaw /= nAct;
  }

  const x_mode = new Float32Array(M * 14);
  const mode_anchor = new Float32Array(M);
  let sumQCorr = 0, cntQCorr = 0;
  const colDfR = new Float64Array(N_ref), colMkR = new Float64Array(N_ref);
  const colResQ = new Float64Array(N_q), colMkQ = new Float64Array(N_q);

  for (let m = 0; m < M; m++) {
    mode_anchor[m] = 200 * mode_mean_resid[m];
    if (!active[m]) continue;
    let cQ = 0, cQc = 0, sQc = 0, cQw = 0, sQw = 0;
    for (let i = 0; i < N_ref; i++) { colDfR[i] = df_ref[i * M + m]; colMkR[i] = mask_ref[i * M + m]; }
    for (let i = 0; i < N_q; i++) {
      const mk = mask_q[i * M + m];
      colResQ[i] = resid_q[i * M + m]; colMkQ[i] = mk;
      if (mk > 0.5) {
        cQ++;
        if (Tq0[i] < 1.0) { cQc++; sQc += resid_q[i * M + m]; } else { cQw++; sQw += resid_q[i * M + m]; }
      }
    }
    let vq = 0;
    if (cQ >= 2) {
      for (let i = 0; i < N_q; i++) if (mask_q[i * M + m] > 0.5) { const d = resid_q[i * M + m] - mode_mean_resid[m]; vq += d * d; }
      vq = Math.sqrt(vq / cQ);
    }
    const coldRes = cQc >= 1 ? sQc / cQc : 0;
    const warmRes = cQw >= 1 ? sQw / cQw : mode_mean_resid[m];
    const cRef = safeCorr(Tr0, colDfR, colMkR);
    const cQres = safeCorr(Tq0, colResQ, colMkQ);
    sumQCorr += cQres; cntQCorr++;
    const zM = clip(mode_mean_resid[m] / ref_std[m], -30, 30);
    const o = m * 14;
    x_mode[o] = clip(slopes[m] * 200, -4, 4);
    x_mode[o + 1] = clip(ref_std[m] * 100, 0, 5);
    x_mode[o + 2] = clip(mode_mean_raw[m] * 20, -5, 5);
    x_mode[o + 3] = clip(mode_mean_resid[m] * 20, -5, 5);
    x_mode[o + 4] = zM / 5;
    x_mode[o + 5] = clip(vq * 100, 0, 5);
    x_mode[o + 6] = clip(coldRes * 20, -5, 5);
    x_mode[o + 7] = clip(warmRes * 20, -5, 5);
    x_mode[o + 8] = refCounts[m] / N_ref;
    x_mode[o + 9] = qCounts[m] / N_q;
    x_mode[o + 10] = cRef;
    x_mode[o + 11] = cQres;
    x_mode[o + 12] = clip((mode_mean_resid[m] - bMeanRes) * 20, -5, 5);
    x_mode[o + 13] = clip((mode_mean_raw[m] - bMeanRaw) * 20, -5, 5);
  }

  const g_feat = new Float32Array([
    clip(anchor_dk_pct / 10, -5, 5),
    clip(anchor_env_pct / 5, -5, 5),
    clip(maxAbsZ / 5, 0, 6),
    clip(mean_abs_z / 5, 0, 6),
    clip(resid_std_modes * 50, 0, 5),
    (T_ref_mean - 10) / 15,
    (T_ref_min - 10) / 15,
    (T_ref_max - 10) / 15,
    (T_q_mean - 10) / 15,
    (T_q_min - 10) / 15,
    clip(Math.max(0, T_ref_min - T_q_min) / 10, 0, 4),
    clip(Math.max(0, T_q_max - T_ref_max) / 10, 0, 4),
    freeze_frac_ref,
    freeze_frac_q,
    Math.log(nActSafe) / 3,
    cntQCorr > 0 ? sumQCorr / cntQCorr : 0,
  ]);

  return { x_ref, x_q, x_mode, m_mask, active, g_feat, anchor_dk_pct, anchor_env_pct, mode_anchor };
}

type Features = ReturnType<typeof extractFeatures>;
interface MemberOut { logit: number; dk_mu: number; dk_sig: number; env_mu: number; env_sig: number; mode_dk: Float32Array }

// Ensemble members are stored as "m{k}.<name>"; a single-member pack may use bare names.
const memberCache = new WeakMap<object, Pack<BSMeta>['t'][]>();
function members(pack: Pack<BSMeta>): Pack<BSMeta>['t'][] {
  let ms = memberCache.get(pack);
  if (ms) return ms;
  const n = pack.meta.n_members || 1;
  if (!('m0.t_in.weight' in pack.t)) {
    if (n !== 1) throw new Error(`bridgesense: ${n} members but no m0.* tensors`);
    ms = [pack.t];
  } else {
    ms = [];
    for (let k = 0; k < n; k++) {
      const pre = `m${k}.`;
      const W: Pack<BSMeta>['t'] = {};
      for (const name in pack.t) if (name.startsWith(pre)) W[name.slice(pre.length)] = pack.t[name];
      ms.push(W);
    }
  }
  memberCache.set(pack, ms);
  return ms;
}

export function predict(
  pack: Pack<BSMeta>,
  T_ref: Float32Array, T_q: Float32Array,
  df_ref: Float32Array, df_q: Float32Array,
  mask_ref: Float32Array, mask_q: Float32Array,
): BSReading {
  const meta = pack.meta;
  const f = extractFeatures(T_ref, T_q, df_ref, df_q, mask_ref, mask_q);
  const Ws = members(pack);
  // Each member runs with its own config (bridgesense.py does the same)
  const outs = Ws.map((W, k) => forwardOne(W, meta.configs?.[k] ?? meta.config, f));
  const n = outs.length;
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const pvar = (xs: number[]) => { const m = mean(xs); return mean(xs.map((x) => (x - m) * (x - m))); };
  const logit_raw = mean(outs.map((o) => o.logit));
  const dks = outs.map((o) => o.dk_mu);
  const envs = outs.map((o) => o.env_mu);
  const dk_sig_raw = Math.sqrt(mean(outs.map((o) => o.dk_sig * o.dk_sig)) + pvar(dks));
  const env_sig_raw = Math.sqrt(mean(outs.map((o) => o.env_sig * o.env_sig)) + pvar(envs));
  const mode_dk_pct = new Float32Array(MAX_MODES);
  for (const o of outs) for (let m = 0; m < MAX_MODES; m++) mode_dk_pct[m] += o.mode_dk[m] / n;

  return {
    logit_raw,
    p_change: sigmoid(clip(logit_raw / (meta.prob_temp || 1), -30, 30)),
    delta_k_pct: mean(dks),
    delta_k_sigma: dk_sig_raw * (meta.widen_dk || 1),
    thermal_shift_pct: mean(envs),
    thermal_shift_sigma: env_sig_raw * (meta.widen_env || 1),
    anchor_dk_pct: f.anchor_dk_pct,
    anchor_env_pct: f.anchor_env_pct,
    mode_dk_pct,
    active_modes: f.active,
  };
}

function forwardOne(W: Pack<BSMeta>['t'], cfg: BSConfig, f: Features): MemberOut {
  const { d_model: D, n_heads: nH, d_ff: dFF, n_t_layers: nT, n_m_layers: nM, use_cross_attn: useCross, use_anchor: useAnc } = cfg;

  // 1. Temporal cross-attention
  const hRef = geluInplace(linearRows(f.x_ref, N_STEPS, 10, W['t_in.weight'].data, W['t_in.bias'].data, D));
  let hQ = geluInplace(linearRows(f.x_q, N_STEPS, 10, W['t_in.weight'].data, W['t_in.bias'].data, D));
  const segQ = W['t_seg_q'].data;
  for (let i = 0; i < N_STEPS; i++) for (let d = 0; d < D; d++) hQ[i * D + d] += segQ[d];

  for (let l = 0; l < nT; l++) {
    const p = `t_blocks.${l}`;
    const qn = layerNormRows(hQ, N_STEPS, D, W[`${p}.ln_q.weight`].data, W[`${p}.ln_q.bias`].data);
    const kvn = layerNormRows(useCross ? hRef : hQ, N_STEPS, D, W[`${p}.ln_kv.weight`].data, W[`${p}.ln_kv.bias`].data);
    const h = mha(qn, N_STEPS, kvn, N_STEPS, D, nH, W, `${p}.attn`);
    for (let k = 0; k < hQ.length; k++) hQ[k] += h[k];
    const fn = layerNormRows(hQ, N_STEPS, D, W[`${p}.ln_ff.weight`].data, W[`${p}.ln_ff.bias`].data);
    const ff1 = geluInplace(linearRows(fn, N_STEPS, D, W[`${p}.ff1.weight`].data, W[`${p}.ff1.bias`].data, dFF));
    const ff2 = linearRows(ff1, N_STEPS, dFF, W[`${p}.ff2.weight`].data, W[`${p}.ff2.bias`].data, D);
    for (let k = 0; k < hQ.length; k++) hQ[k] += ff2[k];
  }
  hQ = layerNormRows(hQ, N_STEPS, D, W['t_ln.weight'].data, W['t_ln.bias'].data);
  const tLogits = linearRows(hQ, N_STEPS, D, W['t_pool.weight'].data, W['t_pool.bias'].data, 1);
  let mxT = -Infinity;
  for (let i = 0; i < N_STEPS; i++) if (tLogits[i] > mxT) mxT = tLogits[i];
  let sumT = 0;
  const wT = new Float64Array(N_STEPS);
  for (let i = 0; i < N_STEPS; i++) { wT[i] = Math.exp(tLogits[i] - mxT); sumT += wT[i]; }
  const zT = new Float32Array(D);
  for (let i = 0; i < N_STEPS; i++) {
    const wi = wT[i] / sumT;
    for (let d = 0; d < D; d++) zT[d] += wi * hQ[i * D + d];
  }

  // 2. Mode-set masked self-attention
  let hM = geluInplace(linearRows(f.x_mode, MAX_MODES, 14, W['m_in.weight'].data, W['m_in.bias'].data, D));
  for (let m = 0; m < MAX_MODES; m++) if (f.m_mask[m] < 0.5) hM.fill(0, m * D, (m + 1) * D);

  for (let l = 0; l < nM; l++) {
    const p = `m_blocks.${l}`;
    const xn = layerNormRows(hM, MAX_MODES, D, W[`${p}.ln1.weight`].data, W[`${p}.ln1.bias`].data);
    const h = mha(xn, MAX_MODES, xn, MAX_MODES, D, nH, W, `${p}.attn`, f.m_mask);
    for (let m = 0; m < MAX_MODES; m++) if (f.m_mask[m] > 0.5) for (let d = 0; d < D; d++) hM[m * D + d] += h[m * D + d];
    const fn = layerNormRows(hM, MAX_MODES, D, W[`${p}.ln_ff.weight`].data, W[`${p}.ln_ff.bias`].data);
    const ff1 = geluInplace(linearRows(fn, MAX_MODES, D, W[`${p}.ff1.weight`].data, W[`${p}.ff1.bias`].data, dFF));
    const ff2 = linearRows(ff1, MAX_MODES, dFF, W[`${p}.ff2.weight`].data, W[`${p}.ff2.bias`].data, D);
    for (let m = 0; m < MAX_MODES; m++) if (f.m_mask[m] > 0.5) for (let d = 0; d < D; d++) hM[m * D + d] += ff2[m * D + d];
  }
  hM = layerNormRows(hM, MAX_MODES, D, W['m_ln.weight'].data, W['m_ln.bias'].data);
  for (let m = 0; m < MAX_MODES; m++) if (f.m_mask[m] < 0.5) hM.fill(0, m * D, (m + 1) * D);

  const mLogits = linearRows(hM, MAX_MODES, D, W['m_pool.weight'].data, W['m_pool.bias'].data, 1);
  let mxM = -Infinity;
  for (let m = 0; m < MAX_MODES; m++) {
    const v = f.m_mask[m] < 0.5 ? -1e9 : mLogits[m];
    if (v > mxM) mxM = v;
  }
  let sumM = 0, cntM = 0;
  const wM = new Float64Array(MAX_MODES);
  for (let m = 0; m < MAX_MODES; m++) {
    const v = f.m_mask[m] < 0.5 ? -1e9 : mLogits[m];
    wM[m] = Math.exp(v - mxM); sumM += wM[m];
    if (f.m_mask[m] > 0.5) cntM++;
  }
  const zMAttn = new Float32Array(D);
  const zMMean = new Float32Array(D);
  const mDen = Math.max(cntM, 1);
  for (let m = 0; m < MAX_MODES; m++) {
    const wm = wM[m] / sumM;
    for (let d = 0; d < D; d++) {
      zMAttn[d] += wm * hM[m * D + d];
      zMMean[d] += hM[m * D + d] / mDen;
    }
  }

  const modeCorr = linearRows(hM, MAX_MODES, D, W['mode_head.weight'].data, W['mode_head.bias'].data, 1);
  const mode_dk_pct = new Float32Array(MAX_MODES);
  for (let m = 0; m < MAX_MODES; m++) {
    mode_dk_pct[m] = f.m_mask[m] > 0.5 ? (useAnc ? f.mode_anchor[m] + modeCorr[m] : modeCorr[m]) : 0;
  }

  // 3. Fusion MLP
  const zG = geluInplace(linearRows(f.g_feat, 1, 16, W['g_in.weight'].data, W['g_in.bias'].data, D));
  const zCat = new Float32Array(D * 4);
  zCat.set(zT, 0); zCat.set(zMAttn, D); zCat.set(zMMean, 2 * D); zCat.set(zG, 3 * D);
  const f1 = geluInplace(layerNormRows(linearRows(zCat, 1, D * 4, W['fuse1.weight'].data, W['fuse1.bias'].data, D * 2), 1, D * 2, W['fuse_ln.weight'].data, W['fuse_ln.bias'].data));
  const f2 = geluInplace(linearRows(f1, 1, D * 2, W['fuse2.weight'].data, W['fuse2.bias'].data, D));
  const out = linearRows(f2, 1, D, W['head.weight'].data, W['head.bias'].data, 5);

  return {
    logit: out[0],
    dk_mu: useAnc ? f.anchor_dk_pct + out[1] : out[1],
    dk_sig: Math.exp(clip(out[2], -2.5, 3.5)),
    env_mu: useAnc ? f.anchor_env_pct + out[3] : out[3],
    env_sig: Math.exp(clip(out[4], -3.0, 3.0)),
    mode_dk: mode_dk_pct,
  };
}
