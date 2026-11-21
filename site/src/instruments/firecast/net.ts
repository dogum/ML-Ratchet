// FireCast in the browser: a pure port of models/firecast/firecast.py (make_inputs with a crop, FireCastNP.forward).
// The 22 input maps are rebuilt here from the fire-day's raw inputs (burned area, VIIRS detections, fire line,
// gridMET features); only the nearest-fire-line offsets come precomputed from scipy (their ties decide off_e/off_n).
// The network then runs on the map window, a U-Net (374k weights, BatchNorm folded) with FiLM biases from the
// 22 scalars. No DOM, so the parity test runs it in Node.
import { linear } from '../../lib/nn/ops.ts';
import type { Pack } from '../../lib/nn/pack.ts';
import type { Tensor } from '../../lib/nn/tensor.ts';

export const RES = 90;
export const N_CH = 22;
const f32 = Math.fround;

export interface WeightsMeta { cal_T: number; cal_b: number; thr: number; rung: string; channels: string[] }
export interface Net { t: Record<string, Tensor>; T: number; b: number; thr: number }

export interface DayInfo {
  fire_id: string; name: string; date: string; k: number; period: string; stratum: string;
  shape: [number, number]; window: [number, number, number, number]; gamma: number; wx: number[];
  has_prev: boolean; first: boolean; area_km2: number; area_prev_km2: number; fl_frac: number;
  d1: { vs: number; th: number; rmin: number; fm100: number };
  exam: { ap: number; f1: number; fl_prev_ap: number; fl_prev_f1: number; new_obs_km2: number };
  window_scores: { ap: number; f1: number; new_obs_km2: number; new_pred_km2: number };
  f16_max_dprob: number;
}
export interface DaysMeta { days: DayInfo[]; clip: number; step: number; start: number; res_m: number }

/** What the controls change. d1: tomorrow's gridMET values (only the keys given replace the recorded features). */
export interface Ops {
  d1?: { vs?: number; th?: number; rmin?: number; fm100?: number };
  viirs_off?: boolean;
  prev_off?: boolean;
}

export interface Day {
  info: DayInfo;
  H: number; W: number;              // full grid
  now: Uint8Array;                   // full grid, 1 = burned at issue
  pm: Float32Array[]; am: Float32Array[];   // detections: row, col, ps, pt, frp
  grew: Uint8Array; fl: Uint8Array; nxt: Uint8Array;   // window
  dr: Int16Array; dc: Int16Array;    // window: offset to the nearest fire-line cell (cells, clipped)
}

export function loadNet(p: Pack<WeightsMeta>): Net {
  return { t: p.t, T: p.meta.cal_T, b: p.meta.cal_b, thr: p.meta.thr };
}

function bits(t: Tensor, n: number): Uint8Array {
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = (t.data[i >> 3] >> (7 - (i & 7))) & 1;   // numpy packbits: MSB first
  return out;
}

function dets(t: Tensor): Float32Array[] {
  const n = t.shape[1] ?? 0;
  return [0, 1, 2, 3, 4].map((k) => t.data.subarray(k * n, (k + 1) * n));
}

export function day(pack: Pack<DaysMeta>, i: number): Day {
  const info = pack.meta.days[i];
  const [H, W] = info.shape, [, , hh, ww] = info.window, n = hh * ww, c = pack.meta.clip;
  const off = pack.t[`off${i}`].data;
  const dr = new Int16Array(n), dc = new Int16Array(n);
  for (let j = 0; j < n; j++) { dr[j] = off[j] - c; dc[j] = off[n + j] - c; }
  return {
    info, H, W, now: bits(pack.t[`now${i}`], H * W), pm: dets(pack.t[`pm${i}`]), am: dets(pack.t[`am${i}`]),
    grew: bits(pack.t[`grew${i}`], n), fl: bits(pack.t[`fl${i}`], n), nxt: bits(pack.t[`nxt${i}`], n), dr, dc,
  };
}

// ---- features (firecast.py full_fields / stack / fire_scalars / feature_vector) ----

/** Max log1p(FRP) over each detection's footprint (ps km along columns, pt km along rows). */
export function pixelRaster(d: Float32Array[] | null, H: number, W: number): Float32Array {
  const out = new Float32Array(H * W);
  if (!d || d[0].length === 0) return out;
  const [row, col, ps, pt, frp] = d;
  for (let i = 0; i < row.length; i++) {
    const r = row[i], c = col[i];
    const hx = 0.5 * ps[i] * 1000.0 / RES, hy = 0.5 * pt[i] * 1000.0 / RES;
    let r0 = Math.floor(r - hy + 0.5), r1 = Math.floor(r + hy + 0.5);
    let c0 = Math.floor(c - hx + 0.5), c1 = Math.floor(c + hx + 0.5);
    r0 = Math.max(r0, 0); c0 = Math.max(c0, 0);
    r1 = Math.min(Math.max(r1, r0 + 1), H); c1 = Math.min(Math.max(c1, c0 + 1), W);
    if (r0 >= H || c0 >= W || r1 <= 0 || c1 <= 0) continue;
    const v = f32(Math.log1p(Math.max(frp[i], 0)));
    for (let y = r0; y < r1; y++) for (let x = c0; x < c1; x++) if (out[y * W + x] < v) out[y * W + x] = v;
  }
  return out;
}

/** Exact squared Euclidean distance (in cells) to the nearest feature cell (Felzenszwalb & Huttenlocher 2012). */
export function edt2(feat: Uint8Array, H: number, W: number): Float64Array {
  const INF = 1e20, n = Math.max(H, W);
  const g = new Float64Array(H * W);
  for (let i = 0; i < H * W; i++) g[i] = feat[i] ? 0 : INF;
  const f = new Float64Array(n), d = new Float64Array(n), v = new Int32Array(n), z = new Float64Array(n + 1);
  const pass = (len: number) => {
    let k = 0;
    v[0] = 0; z[0] = -Infinity; z[1] = Infinity;
    for (let q = 1; q < len; q++) {
      let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
      while (s <= z[k]) { k--; s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]); }
      k++; v[k] = q; z[k] = s; z[k + 1] = Infinity;
    }
    k = 0;
    for (let q = 0; q < len; q++) { while (z[k + 1] < q) k++; d[q] = (q - v[k]) * (q - v[k]) + f[v[k]]; }
  };
  for (let x = 0; x < W; x++) {
    for (let y = 0; y < H; y++) f[y] = g[y * W + x];
    pass(H);
    for (let y = 0; y < H; y++) g[y * W + x] = d[y];
  }
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) f[x] = g[y * W + x];
    pass(W);
    for (let x = 0; x < W; x++) g[y * W + x] = d[x];
  }
  return g;
}

/** scipy's distance_transform_edt(~feat, sampling=90) in metres (1e5 everywhere when there is no feature). */
function distM(feat: Uint8Array, H: number, W: number, any: boolean): Float64Array {
  if (!any) return new Float64Array(H * W).fill(1e5);
  const D = edt2(feat, H, W);
  for (let i = 0; i < D.length; i++) D[i] = Math.sqrt(8100 * D[i]);   // sqrt((dr*90)^2 + (dc*90)^2), exactly
  return D;
}

const scaled = (dm: number) => f32(Math.min(Math.max(dm / 1000.0, 0), 10) / 5.0);
const DEG = Math.PI / 180;

/** The 16 gridMET features with tomorrow's values (indices 8-15) replaced where the controls changed them. */
export function weather(info: DayInfo, d1?: Ops['d1']): Float32Array {
  const wx = Float32Array.from(info.wx);
  if (!d1) return wx;
  if (d1.vs != null || d1.th != null) {
    const vs = d1.vs ?? 3.0, th = d1.th ?? 0.0;
    const tow = (th + 180.0 + info.gamma) * DEG;
    wx[8] = f32(vs * Math.sin(tow) / 5.0); wx[9] = f32(vs * Math.cos(tow) / 5.0); wx[10] = f32(vs / 5.0);
  }
  if (d1.rmin != null) wx[11] = f32((d1.rmin - 20.0) / 15.0);
  if (d1.fm100 != null) wx[15] = f32((d1.fm100 - 9.0) / 4.0);
  return wx;
}

const reqM = (a: number) => Math.sqrt(Math.max(a, 0) * 1e6 / Math.PI);

export interface Inputs {
  X: Tensor; s: Float32Array; hh: number; ww: number;
  now: Uint8Array; inwin: Uint8Array;     // window, for drawing and scoring
  P: Float32Array; A: Float32Array;       // window FRP rasters (log1p MW), for drawing
}

/** make_inputs(now, prev, fl, pix_pm, pix_am, land=None, wx, has_prev, first, crop=window). */
export function prepare(dy: Day, ops: Ops = {}): Inputs {
  const { info, H, W } = dy;
  const [r0, c0, hh, ww] = info.window, n = hh * ww;
  const pm = ops.viirs_off ? null : dy.pm, am = ops.viirs_off ? null : dy.am;
  const hasPrev = info.has_prev && !ops.prev_off;
  const Pf = pixelRaster(pm, H, W), Af = pixelRaster(am, H, W);
  let anyNow = false, anyPix = false;
  const pix = new Uint8Array(H * W);
  for (let i = 0; i < H * W; i++) { if (dy.now[i]) anyNow = true; if (Pf[i] > 0 || Af[i] > 0) { pix[i] = 1; anyPix = true; } }
  const dBurn = distM(dy.now, H, W, anyNow), dPix = distM(pix, H, W, anyPix);
  const wx = weather(info, ops.d1);
  let ue = wx[8], un = wx[9];
  const sp = Math.hypot(ue, un);
  if (sp > 1e-6) { ue = ue / sp; un = un / sp; } else { ue = 0; un = 0; }
  const ue32 = f32(ue), un32 = f32(un);
  const X = new Float32Array(N_CH * n);
  const ch = (k: number) => X.subarray(k * n, (k + 1) * n);
  const [cNow, cGrew, cFl, cPm, cAm, cDb, cDfl, cDp, cAl, cAc, cWin] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(ch);
  const now = new Uint8Array(n), inwin = new Uint8Array(n), P = new Float32Array(n), A = new Float32Array(n);
  for (let y = 0; y < hh; y++) for (let x = 0; x < ww; x++) {
    const j = y * ww + x, g = (r0 + y) * W + (c0 + x);
    now[j] = dy.now[g];
    cNow[j] = dy.now[g];
    cGrew[j] = hasPrev ? dy.grew[j] : 0;
    cFl[j] = dy.fl[j];
    P[j] = Pf[g]; A[j] = Af[g];
    cPm[j] = f32(Pf[g] / 4.0); cAm[j] = f32(Af[g] / 4.0);
    cDb[j] = scaled(dBurn[g]);
    cDp[j] = scaled(dPix[g]);
    const dr = dy.dr[j], dc = dy.dc[j];
    cDfl[j] = scaled(Math.sqrt((dr * 90) * (dr * 90) + (dc * 90) * (dc * 90)));
    const offE = f32(Math.min(Math.max(dc * 90 / 1000.0, -10), 10));
    const offN = f32(Math.min(Math.max(-dr * 90 / 1000.0, -10), 10));
    const along = f32(f32(offE * ue32) + f32(offN * un32));
    const across = Math.abs(f32(f32(-offE * un32) + f32(offN * ue32)));
    cAl[j] = f32(Math.min(Math.max(along, -10), 10) / 5.0);
    cAc[j] = f32(Math.min(across, 10) / 5.0);
    inwin[j] = dBurn[g] <= 10000.0 ? 1 : 0;
    cWin[j] = inwin[j];
  }
  // channels 11-21 (LANDFIRE: nonburn, GR, GS, SH, TU, TL, slope, up_e, up_n, cc, land_ok) stay 0: exam mode F
  let npix = 0;
  if (pm) npix = pm[0].length;
  const dPrev = hasPrev ? reqM(info.area_km2) - reqM(info.area_prev_km2) : 0;
  const s = new Float32Array(N_CH);
  s.set(wx, 0);
  s[16] = f32(Math.log10(info.area_km2 + 0.01) / 2.0);
  s[17] = hasPrev ? f32(Math.log1p(Math.max(dPrev, 0.0) / 100.0) / 3.0) : 0;
  s[18] = hasPrev ? 1 : 0;
  s[19] = info.first ? 1 : 0;
  s[20] = f32(Math.log1p(npix) / 5.0);
  s[21] = f32(info.fl_frac);
  return { X: { data: X, shape: [N_CH, hh, ww] }, s, hh, ww, now, inwin, P, A };
}

// ---- network (FireCastNP.forward) ----

/** 3x3 convolution, padding 1 (numpy conv3x3): four output channels per pass over the input rows. Same sums as
 * ops.conv2d, in the same order per output cell (input channel, then kernel row, then kernel column). */
export function conv3x3(x: Tensor, w: Tensor, bias: Tensor, stride: number): Tensor {
  const [C, H, W] = x.shape, O = w.shape[0];
  const Ho = Math.floor((H - 1) / stride) + 1, Wo = Math.floor((W - 1) / stride) + 1, n = Ho * Wo;
  const out = new Float32Array(O * n), xd = x.data, wd = w.data, bd = bias.data;
  for (let o = 0; o < O; o++) out.fill(bd[o], o * n, (o + 1) * n);
  // output columns j whose input column j*stride + dx - 1 lies inside [0, W)
  const jlo = [1, 0, 0], jhi = [0, 0, 0];
  for (let dx = 0; dx < 3; dx++) {
    jlo[dx] = dx === 0 ? Math.ceil(1 / stride) : 0;
    jhi[dx] = Math.min(Wo, Math.floor((W - dx) / stride) + 1);
  }
  for (let o0 = 0; o0 < O; o0 += 4) {
    const no = Math.min(4, O - o0);
    const a0 = out.subarray(o0 * n, (o0 + 1) * n);
    const a1 = no > 1 ? out.subarray((o0 + 1) * n, (o0 + 2) * n) : a0;
    const a2 = no > 2 ? out.subarray((o0 + 2) * n, (o0 + 3) * n) : a0;
    const a3 = no > 3 ? out.subarray((o0 + 3) * n, (o0 + 4) * n) : a0;
    for (let c = 0; c < C; c++) {
      const xb = c * H * W;
      for (let dy = 0; dy < 3; dy++) {
        for (let dx = 0; dx < 3; dx++) {
          const k = (c * 3 + dy) * 3 + dx, step = C * 9;
          const w0 = wd[o0 * step + k], w1 = no > 1 ? wd[(o0 + 1) * step + k] : 0;
          const w2 = no > 2 ? wd[(o0 + 2) * step + k] : 0, w3 = no > 3 ? wd[(o0 + 3) * step + k] : 0;
          const j0 = jlo[dx], j1 = jhi[dx];
          for (let i = 0; i < Ho; i++) {
            const yi = i * stride + dy - 1;
            if (yi < 0 || yi >= H) continue;
            const xr = xb + yi * W + dx - 1, orow = i * Wo;
            if (stride === 1) {
              if (no === 4) {
                for (let j = j0; j < j1; j++) {
                  const v = xd[xr + j], q = orow + j;
                  a0[q] += w0 * v; a1[q] += w1 * v; a2[q] += w2 * v; a3[q] += w3 * v;
                }
              } else {
                for (let j = j0; j < j1; j++) {
                  const v = xd[xr + j], q = orow + j;
                  a0[q] += w0 * v; if (no > 1) a1[q] += w1 * v; if (no > 2) a2[q] += w2 * v;
                }
              }
            } else {
              for (let j = j0; j < j1; j++) {
                const v = xd[xr + j * stride], q = orow + j;
                a0[q] += w0 * v; if (no > 1) a1[q] += w1 * v; if (no > 2) a2[q] += w2 * v; if (no > 3) a3[q] += w3 * v;
              }
            }
          }
        }
      }
    }
  }
  return { data: out, shape: [O, Ho, Wo] };
}

function cbr(net: Net, name: string, x: Tensor, stride: number, film: Float32Array): Tensor {
  const y = conv3x3(x, net.t[`${name}.w`], net.t[`${name}.b`], stride);
  const [C, H, W] = y.shape, n = H * W, d = y.data;
  for (let c = 0; c < C; c++) {
    const fb = film[c], o = c * n;
    for (let i = 0; i < n; i++) { const v = d[o + i]; d[o + i] = (v > 0 ? v : 0) + fb; }
  }
  return y;
}

function pool2(x: Tensor): Tensor {
  const [C, H, W] = x.shape, Ho = H >> 1, Wo = W >> 1, out = new Float32Array(C * Ho * Wo), d = x.data;
  for (let c = 0; c < C; c++) for (let y = 0; y < Ho; y++) for (let q = 0; q < Wo; q++) {
    const i = (c * H + 2 * y) * W + 2 * q;
    out[(c * Ho + y) * Wo + q] = Math.max(d[i], d[i + 1], d[i + W], d[i + W + 1]);
  }
  return { data: out, shape: [C, Ho, Wo] };
}

/** concatenate(up2(a), s) along channels. */
function upcat(a: Tensor, s: Tensor): Tensor {
  const [Ca, Ha, Wa] = a.shape, [Cs, H, W] = s.shape, out = new Float32Array((Ca + Cs) * H * W);
  for (let c = 0; c < Ca; c++) for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
    out[(c * H + y) * W + x] = a.data[(c * Ha + (y >> 1)) * Wa + (x >> 1)];
  out.set(s.data, Ca * H * W);
  return { data: out, shape: [Ca + Cs, H, W] };
}

const relu = (v: Float32Array) => { for (let i = 0; i < v.length; i++) if (v[i] < 0) v[i] = 0; return v; };

/** Raw logits for every cell of X (C, H, W); H and W are padded to multiples of 16 as in numpy. */
export function forward(net: Net, X: Tensor, s: Float32Array): Float32Array {
  const t = net.t, [C, H, W] = X.shape;
  const Hp = Math.ceil(H / 16) * 16, Wp = Math.ceil(W / 16) * 16;
  let Xp = X;
  if (Hp !== H || Wp !== W) {
    const d = new Float32Array(C * Hp * Wp);
    for (let c = 0; c < C; c++) for (let y = 0; y < H; y++) d.set(X.data.subarray((c * H + y) * W, (c * H + y + 1) * W), (c * Hp + y) * Wp);
    Xp = { data: d, shape: [C, Hp, Wp] };
  }
  const h = relu(linear(relu(linear(s, t['mlp0.w'], t['mlp0.b'])), t['mlp1.w'], t['mlp1.b']));
  const fb = Array.from({ length: 9 }, (_, i) => linear(h, t[`film${i}.w`], t[`film${i}.b`]));
  const x0 = cbr(net, 'stem', Xp, 1, fb[0]);
  const x1 = cbr(net, 'e1b', cbr(net, 'e1a', x0, 2, new Float32Array(32)), 1, fb[1]);
  const x2 = cbr(net, 'e2b', cbr(net, 'e2a', pool2(x1), 1, new Float32Array(48)), 1, fb[2]);
  const x3 = cbr(net, 'e3b', cbr(net, 'e3a', pool2(x2), 1, new Float32Array(64)), 1, fb[3]);
  const x4 = cbr(net, 'e4b', cbr(net, 'e4a', pool2(x3), 1, new Float32Array(64)), 1, fb[4]);
  let y = cbr(net, 'd3', upcat(x4, x3), 1, fb[5]);
  y = cbr(net, 'd2', upcat(y, x2), 1, fb[6]);
  y = cbr(net, 'd1', upcat(y, x1), 1, fb[7]);
  y = cbr(net, 'd0', upcat(y, x0), 1, fb[8]);
  const hw = t['head.w'].data, hs0 = t['head_s.w'].data;
  let hs = 0;
  for (let i = 0; i < h.length; i++) hs += hs0[i] * h[i];
  const hsF = f32(f32(hs) + t['head_s.b'].data[0]), hb = t['head.b'].data[0];
  const Cy = y.shape[0], np_ = Hp * Wp, out = new Float32Array(H * W);
  for (let r = 0; r < H; r++) for (let q = 0; q < W; q++) {
    const i = r * Wp + q;
    let v = 0;
    for (let c = 0; c < Cy; c++) v += hw[c] * y.data[c * np_ + i];
    out[r * W + q] = f32(f32(f32(v) + hb) + hsF);     // (tensordot + head.b) + head_s, in float32 as numpy
  }
  return out;
}

/** The dev calibration: p = sigmoid(logit / T + b). */
export function calibrate(net: Net, lg: Float32Array): Float32Array {
  const p = new Float32Array(lg.length);
  for (let i = 0; i < lg.length; i++) p[i] = 1 / (1 + Math.exp(-(lg[i] / net.T + net.b)));
  return p;
}

// ---- scoring (firedata.metrics, as in firecast.py nextday) on the window's scored cells ----

export interface Scores { ap: number; f1: number; newObsKm2: number; newPredKm2: number; nCand: number }

export function score(now: Uint8Array, nxt: Uint8Array, inwin: Uint8Array, p: Float32Array, thr: number): Scores {
  const ys: number[] = [], ss: number[] = [];
  let tp = 0, fp = 0, fn = 0;
  for (let i = 0; i < now.length; i++) {
    if (!inwin[i] || now[i]) continue;
    const y = nxt[i] ? 1 : 0, yes = p[i] >= thr;
    ys.push(y); ss.push(p[i]);
    if (y && yes) tp++; else if (!y && yes) fp++; else if (y && !yes) fn++;
  }
  const idx = ss.map((_, i) => i).sort((a, b) => ss[b] - ss[a] || a - b);
  const P = tp + fn;
  let ap = NaN;
  if (P > 0) {
    ap = 0;
    let ctp = 0, prevRec = 0;
    for (let k = 0; k < idx.length; k++) {
      ctp += ys[idx[k]];
      if (k === idx.length - 1 || ss[idx[k + 1]] !== ss[idx[k]]) {
        const rec = ctp / P, prec = ctp / (k + 1);
        ap += (rec - prevRec) * prec;
        prevRec = rec;
      }
    }
  }
  const cell = (RES * RES) / 1e6;
  return { ap, f1: 2 * tp + fp + fn ? (2 * tp) / (2 * tp + fp + fn) : NaN, newObsKm2: P * cell, newPredKm2: (tp + fp) * cell, nCand: ys.length };
}
