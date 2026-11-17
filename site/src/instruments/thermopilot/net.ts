// ThermoPilot in the browser: a port of models/thermopilot/thermopilot_core.py (features, MLP, 36-pair action menu,
// 6 h comfort shield), started from the release's training/tp.js, plus the lab page's browser house (BuildingSim's
// hourly 5R1C maps of the nominal twin with the heat-pump / furnace / AC tables). No DOM.
// The Python reference (models/thermopilot/web/export_instrument.py) runs the same closed loop.
import type { Pack } from '../../lib/nn/pack.ts';

export const H = 24, NS = 34, NB = 11, NF = NS + NB * H;
const KINDS = ['hp', 'furnace', 'elec', 'baseboard', 'none'];
const DHm: (null | 'edge' | number)[] = [null, 'edge', 0, 1, 2, 3];
const DCm: (null | 'edge' | number)[] = [null, 'edge', 0, -1, -2, -3];
export const ACTIONS: [number, number][] = [];
for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) ACTIONS.push([i, j]);
const SHIELD_RATE = 1.0;

export interface WMeta { n_params: number; shield: number; layers: number; arch: string }
export interface Card {
  UA: number; C: number; area: number; kind: string; gas_heat: boolean; fan_cont: boolean; res: boolean; P_fan: number; backup: number;
  Tg: Float32Array; Qmax: Float32Array; Qh: Float32Array; Ph: Float32Array; eff: Float32Array; Qs: Float32Array; Pc: Float32Array;
}
export interface LiveMeta { Aq: number; Cq: number; Tm0: number; lam: number; gas_price: number; card: Omit<Card, 'Tg' | 'Qmax' | 'Qh' | 'Ph' | 'eff' | 'Qs' | 'Pc'> }
export interface CtrlResult { bill: number; DH: number; J: number; kWh: number; demand: number }
export interface XMeta {
  homes: { label: string; id: string; system: string; area: number; year: number; state: string; pref_lo: number; pref_hi: number }[];
  climates: [string, string][]; tariffs: [string, string][]; weeks: string[];
  live: LiveMeta[]; runs: Record<string, CtrlResult>[]; nz: number; week_days: [string, string][]; year: number;
}

export interface Net { mu: Float32Array; sd: Float32Array; layers: { W: Float32Array; b: Float32Array; nin: number; nout: number }[]; shield: number; nParams: number }
export function loadNet(P: Pack<WMeta>): Net {
  const layers = [];
  for (let i = 0; i < P.meta.layers; i++) {
    const W = P.t[`W${i}`];
    layers.push({ W: W.data, b: P.t[`b${i}`].data, nin: W.shape[0], nout: W.shape[1] });
  }
  return { mu: P.t.mu.data, sd: P.t.sd.data, layers, shield: P.meta.shield, nParams: P.meta.n_params };
}

export function interp(x: number, xs: ArrayLike<number>, ys: ArrayLike<number>) {
  const n = xs.length;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
  const t = (x - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] + t * (ys[hi] - ys[lo]);
}

export function setpoints(loK: number, hiK: number, lo: number, hi: number, ih: number, ic: number): [number, number] {
  let d = DHm[ih];
  const h = d === null ? -100 : d === 'edge' ? loK : Math.max(lo + d, loK);
  d = DCm[ic];
  let c = d === null ? 100 : d === 'edge' ? hiK : Math.min(hi + d, hiK);
  if (c < h + 0.5) c = h + 0.5;
  return [h, c];
}

/** The per-hour context the controller is told: prices, demand charges, comfort band, occupancy, calendar. */
export interface Ctx {
  price: Float64Array; lo: Float64Array; hi: Float64Array; state: Int32Array; pref_lo: number; pref_hi: number;
  dperiod: Int32Array; drate: Float64Array; frate: Float64Array; drem: Float64Array; frem: Float64Array; hour: Int32Array; wd: Uint8Array;
}

function prefKW(card: Card, base: ArrayLike<number>) {
  let ph = 0;
  if (card.kind === 'hp') ph = interp(-5, card.Tg, card.Ph);
  else if ((card.kind === 'elec' || card.kind === 'baseboard') && !card.gas_heat) ph = interp(-5, card.Tg, card.Qmax);
  const pc = interp(35, card.Tg, card.Pc);
  let mb = 0; for (let i = 0; i < base.length; i++) mb += base[i]; mb /= base.length;
  return Math.max(0.5, (Math.max(ph, pc) + card.P_fan + mb) / 1000);
}

export class Featurizer {
  card: Card; ctx: Ctx; lam: number; base: ArrayLike<number>;
  P: number; Th: number[] = []; qh: number[] = []; peak = new Map<string, number>(); ua: number; desc: Float64Array;
  constructor(card: Card, ctx: Ctx, lam: number, base: ArrayLike<number>) {
    this.card = card; this.ctx = ctx; this.lam = lam; this.base = base;
    this.P = prefKW(card, base);
    this.ua = Math.max(card.UA, 1);
    const d = new Float64Array(13);
    d[0] = Math.log(this.ua / 150); d[1] = Math.log(Math.max(card.C / this.ua / 3600, 1) / 30);
    d[2] = Math.log(Math.max(card.area, 10) / 150); d[3 + KINDS.indexOf(card.kind)] = 1;
    d[8] = card.fan_cont ? 1 : 0; d[9] = card.res ? 1 : 0; d[10] = card.backup > 0 ? 1 : 0;
    d[11] = card.gas_heat ? 1 : 0; d[12] = Math.log(this.P);
    this.desc = d;
  }
  observe(k: number, e: number) {
    const c = this.ctx; if (k <= 0) return; const j = k - 1;
    if (c.dperiod[j] >= 0) { const key = 'd' + c.dperiod[j]; this.peak.set(key, Math.max(this.peak.get(key) || 0, e)); }
    if (c.frate[j] > 0) this.peak.set('f', Math.max(this.peak.get('f') || 0, e));
  }
  step(k: number, Tz: number, q: number, e: number, fcT: ArrayLike<number>, fcG: ArrayLike<number>) {
    const c = this.ctx, card = this.card; this.observe(k, e);
    this.Th.push(Tz); this.qh.push(q);
    const n = c.price.length, idx: number[] = [];
    for (let j = 0; j < H; j++) idx.push(Math.min(k + j, n - 1));
    const price = idx.map((i) => c.price[i]);
    let pm = 0; for (const p of price) pm += Math.abs(p); pm = Math.max(pm / H, 0.02);
    const lo = idx.map((i) => c.lo[i]), hi = idx.map((i) => c.hi[i]);
    const x = new Float64Array(NF);
    x[0] = (Tz - lo[0]) / 3; x[1] = (hi[0] - Tz) / 3; x[2] = (Tz - c.pref_lo) / 3; x[3] = (c.pref_hi - Tz) / 3;
    for (let i = 1; i < 4; i++) x[3 + i] = this.Th.length > i ? Tz - this.Th[this.Th.length - 1 - i] : 0;
    for (let i = 1; i < 4; i++) x[6 + i] = this.qh.length >= i ? this.qh[this.qh.length - i] / this.ua / 10 : 0;
    x[10] = e / this.P;
    x[11] = c.dperiod[k] >= 0 ? (this.peak.get('d' + c.dperiod[k]) || 0) / this.P : 0;
    x[12] = c.frate[k] > 0 ? (this.peak.get('f') || 0) / this.P : 0;
    const h = c.hour[k];
    x[13] = Math.sin(2 * Math.PI * h / 24); x[14] = Math.cos(2 * Math.PI * h / 24); x[15] = c.wd[k] ? 1 : 0;
    x[16] = Math.log(Math.max(price[0], 0.005) / 0.15);
    x[17] = Math.log(this.lam / (pm * this.P));
    x[18] = c.drate[k] / Math.max(c.drem[k], 1) / pm / 10;
    x[19] = c.frate[k] / Math.max(c.frem[k], 1) / pm / 10;
    x[20] = c.state[k] === 0 ? 1 : 0;
    for (let i = 0; i < 13; i++) x[21 + i] = this.desc[i];
    const b = NS;
    for (let j = 0; j < H; j++) {
      const To = fcT[j], G = fcG[j], i = idx[j];
      const pc = interp(To, card.Tg, card.Pc), qs = interp(To, card.Tg, card.Qs);
      x[b + j] = (To - Tz) / 10;
      x[b + H + j] = G / 800;
      x[b + 2 * H + j] = price[j] / pm - 1;
      x[b + 3 * H + j] = (lo[j] - c.pref_lo) / 3;
      x[b + 4 * H + j] = (hi[j] - c.pref_hi) / 3;
      x[b + 5 * H + j] = interp(To, card.Tg, card.eff) / 4;
      x[b + 6 * H + j] = pc > 0 ? qs / Math.max(pc, 1e-9) / 4 : 0;
      x[b + 7 * H + j] = this.base[i] / 1000 / this.P;
      x[b + 8 * H + j] = (c.drate[i] > 0 ? 1 : 0) + 0.5 * (c.frate[i] > 0 ? 1 : 0);
      x[b + 9 * H + j] = interp(To, card.Tg, card.Qmax) / this.ua / 20;
      x[b + 10 * H + j] = qs / this.ua / 20;
    }
    return x;
  }
}

/** MLP 298 -> 384 -> 384 -> 36 on float32-rounded features (as numpy does), float64 activations. */
export function mlp(net: Net, x: Float64Array) {
  let h = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) h[i] = (Math.fround(x[i]) - net.mu[i]) / net.sd[i];
  for (let l = 0; l < net.layers.length; l++) {
    const L = net.layers[l], out = new Float64Array(L.nout);
    for (let i = 0; i < L.nin; i++) {
      const hi = h[i]; if (hi === 0) continue;
      const row = i * L.nout;
      for (let o = 0; o < L.nout; o++) out[o] += hi * L.W[row + o];
    }
    for (let o = 0; o < L.nout; o++) out[o] += L.b[o];
    if (l < net.layers.length - 1) for (let o = 0; o < L.nout; o++) if (out[o] < 0) out[o] = 0;
    h = out;
  }
  return h;
}

export class Pilot {
  net: Net; ctx: Ctx; feat: Featurizer; last = -1; logits: Float64Array | null = null;
  constructor(net: Net, card: Card, ctx: Ctx, lam: number, base: ArrayLike<number>) {
    this.net = net; this.ctx = ctx; this.feat = new Featurizer(card, ctx, lam, base);
  }
  step(k: number, Tz: number, q: number, e: number, fcT: ArrayLike<number>, fcG: ArrayLike<number>): [number, number] {
    const x = this.feat.step(k, Tz, q, e, fcT, fcG);
    const lg = mlp(this.net, x);
    let a = 0; for (let i = 1; i < lg.length; i++) if (lg[i] > lg[a]) a = i;
    this.last = a; this.logits = lg;
    const c = this.ctx;
    let [h, cs] = setpoints(c.lo[k], c.hi[k], c.pref_lo, c.pref_hi, ACTIONS[a][0], ACTIONS[a][1]);
    const sh = this.net.shield;
    if (sh) {
      let lo = c.lo[k], hi = c.hi[k]; const n = c.lo.length;
      if (sh === 2) { const k1 = Math.min(k + 1, n - 1); lo = Math.max(lo, c.lo[k1]); hi = Math.min(hi, c.hi[k1]); }
      else if (sh >= 4) for (let j = 1; j < 7; j++) {
        const kj = Math.min(k + j, n - 1);
        lo = Math.max(lo, c.lo[kj] - SHIELD_RATE * (j - 1)); hi = Math.min(hi, c.hi[kj] + 0.5 * SHIELD_RATE * (j - 1));
      }
      else if (sh >= 3) for (let j = 1; j < 5; j++) {
        const kj = Math.min(k + j, n - 1);
        lo = Math.max(lo, c.lo[kj] - SHIELD_RATE * (j - 1)); hi = Math.min(hi, c.hi[kj] + SHIELD_RATE * (j - 1));
      }
      h = Math.max(h, lo); cs = Math.min(cs, hi); if (cs < h + 0.5) cs = h + 0.5;
    }
    return [h, cs];
  }
}

// ---------------------------------------------------------------------------------------------- examples
export interface House {
  A: Float32Array; At: Float32Array; C: Float32Array; Ct: Float32Array; To: Float32Array; G: Float32Array; base: Float32Array;
  state: Float32Array; hour: Float32Array; wd: Float32Array; Aq: number; Cq: number; Tm0: number; lam: number; gas: number; card: Card;
}
export const N_LIVE = 192, N_RUN = 168;
const row = (P: Pack, name: string, i: number, n: number) => P.t[name].data.subarray(i * n, (i + 1) * n);

export function houseOf(X: Pack<XMeta>, li: number): House {
  const m = X.meta.live[li];
  const L = (k: string) => row(X, `L_${k}`, li, N_LIVE), K = (k: string) => row(X, `K_${k}`, li, 91);
  return {
    A: L('A'), At: L('At'), C: L('C'), Ct: L('Ct'), To: L('To'), G: L('G'), base: L('base'), state: L('state'), hour: L('hour'), wd: L('wd'),
    Aq: Math.fround(m.Aq), Cq: Math.fround(m.Cq), Tm0: Math.fround(m.Tm0), lam: m.lam, gas: m.gas_price,
    card: { ...m.card, Tg: K('Tg'), Qmax: K('Qmax'), Qh: K('Qh'), Ph: K('Ph'), eff: K('eff'), Qs: K('Qs'), Pc: K('Pc') },
  };
}

export function ctxOf(X: Pack<XMeta>, house: House, ri: number, home: { pref_lo: number; pref_hi: number }): Ctx {
  const N = N_LIVE;
  const R = (k: string) => row(X, `R_${k}`, ri, N_RUN);
  const cl = (a: Float32Array) => { const o = new Float64Array(N); for (let i = 0; i < N; i++) o[i] = a[Math.min(i, a.length - 1)]; return o; };
  const ci = (a: Float32Array) => Int32Array.from(cl(a));
  return {
    price: cl(R('price')), lo: cl(R('lo')), hi: cl(R('hi')), state: Int32Array.from(house.state), pref_lo: home.pref_lo, pref_hi: home.pref_hi,
    dperiod: ci(R('dperiod')), drate: cl(R('drate')), frate: cl(R('frate')), drem: cl(R('drem')), frem: cl(R('frem')),
    hour: Int32Array.from(house.hour), wd: Uint8Array.from(house.wd, (v) => (v > 0.5 ? 1 : 0)),
  };
}

/** 24 h forecast: real weather + bias + an error growing with lead time (the lab page's model, fixed noise draw). */
export function forecast(house: House, z: ArrayLike<number>, k: number, fe: number, bias: number): [Float64Array, Float64Array] {
  const N = house.To.length, nz = z.length, T = new Float64Array(24), G = new Float64Array(24);
  for (let j = 0; j < 24; j++) {
    const i = Math.min(k + j, N - 1);
    let e = 0;
    if (j > 0 && fe > 0) e = fe * (0.6 + 1.6 * (1 - Math.exp(-j / 24))) * (0.7 * z[(k + j) % nz] + 0.3 * z[(k + 2 * j + 5) % nz]);
    T[j] = house.To[i] + bias + e; G[j] = house.G[i];
  }
  return [T, G];
}

export interface Sim { T: Float64Array; h: Float64Array; c: Float64Array; P: Float64Array; bill: number; demand: number; DH: number; kWh: number; gas: number }
export type Ctrl = (k: number, Tz: number, q: number, e: number) => [number, number];

/** The browser house: one step per hour with BuildingSim's maps; the setpoints are met within the equipment limits. */
export function simulate(house: House, ctx: Ctx, n: number, ctrl: Ctrl): Sim {
  const card = house.card, Tg = card.Tg;
  let hasCool = false; for (let i = 0; i < card.Qs.length; i++) if (card.Qs[i] > 0) hasCool = true;
  let Tm = house.Tm0, Tz = ctx.pref_lo + 0.5, q = 0, e = 0, bill = 0, gas = 0, DH = 0, kWh = 0;
  const peaks = new Map<number, number>(), rate = new Map<number, number>();
  const T = new Float64Array(n), hh = new Float64Array(n), cc = new Float64Array(n), PP = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const [h, c] = ctrl(k, Tz, q, e);
    const Tb = house.A[k] + house.At[k] * Tm, be = house.Aq, To = house.To[k];
    const Qmax = interp(To, Tg, card.Qmax), Qs = interp(To, Tg, card.Qs);
    const Qhd = Tb < h ? Math.min((h - Tb) / be, Qmax) : 0;
    const Qcd = Tb > c && Qs > 0 ? Math.min((Tb - c) / be, Qs) : 0;
    const Tk = Tb + be * (Qhd - Qcd);
    Tm = house.C[k] + house.Ct[k] * Tm + house.Cq * (Qhd - Qcd);
    let P = 0, g = 0, runH = 0;
    if (card.kind === 'hp') {
      const Qh = interp(To, Tg, card.Qh), Ph = interp(To, Tg, card.Ph);
      const f = Qh > 0 ? Math.min(Qhd, Qh) / Qh : 0;
      P = f > 0 ? Ph * f / (0.85 + 0.15 * f) : 0;
      P += Math.max(Qhd - Qh, 0);
      runH = Qhd > Qh ? 1 : f;
    } else if (card.gas_heat) {
      g = Qhd / Math.max(interp(To, Tg, card.eff), 0.5);
      runH = Qmax > 0 ? Qhd / Qmax : 0;
    } else {
      P = Qhd;
      runH = card.kind === 'baseboard' ? 0 : Qmax > 0 ? Qhd / Qmax : 0;
    }
    const Pc = interp(To, Tg, card.Pc), fcl = Qs > 0 ? Qcd / Qs : 0;
    if (fcl > 0) P += Pc * fcl / (0.85 + 0.15 * fcl);
    P += card.P_fan * Math.min(runH + fcl, 1);
    const el = (house.base[k] + P) / 1000;
    bill += ctx.price[k] * el + house.gas * g / 1000;
    gas += g / 1000; kWh += P / 1000;
    if (ctx.drate[k] > 0) {
      const key = ctx.dperiod[k];
      peaks.set(key, Math.max(peaks.get(key) ?? 0, el));
      if (!rate.has(key)) rate.set(key, ctx.drate[k]);
    }
    DH += Math.max(ctx.lo[k] - Tk, 0) + (hasCool ? Math.max(Tk - ctx.hi[k], 0) : 0);
    Tz = Tk; q = Qhd - Qcd; e = el;
    T[k] = Tk; hh[k] = h; cc[k] = c; PP[k] = P / 1000 + g / 1000;
  }
  let demand = 0; for (const [key, p] of peaks) demand += p * rate.get(key)!;
  return { T, h: hh, c: cc, P: PP, bill: bill + demand, demand, DH, kWh, gas };
}

export interface Setting { lamMul: number; fe: number; bias: number }

/** ThermoPilot closed-loop on the browser house; also returns the logits at the requested hours. */
export function runPilot(net: Net, house: House, ctx: Ctx, z: ArrayLike<number>, s: Setting, n = N_RUN, logHours: number[] = []) {
  const pilot = new Pilot(net, house.card, ctx, house.lam * s.lamMul, house.base);
  const acts = new Int32Array(n), logits: Float64Array[] = [];
  const sim = simulate(house, ctx, n, (k, Tz, q, e) => {
    const [T, G] = forecast(house, z, k, s.fe, s.bias);
    const r = pilot.step(k, Tz, q, e, T, G);
    acts[k] = pilot.last;
    if (logHours.includes(k)) logits.push(pilot.logits!);
    return r;
  });
  return { ...sim, acts, logits };
}

export const fixedCtrl = (ctx: Ctx): Ctrl => () => [ctx.pref_lo, ctx.pref_hi];
export const setbackCtrl = (ctx: Ctx): Ctrl => (k) => [ctx.lo[k], ctx.hi[k]];
