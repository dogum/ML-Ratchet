// BasinDetective's two scopes: the basin's 19 years (target decade forcing with the levels the network predicts,
// gauged decade forcing and flow), and the six signature changes with their bars, the truth and GR4J.
import { SCOPE, label, type Canvas2D } from '../kit.ts';
import type { Days } from './features.ts';
import { CHANGES, LEVEL, level, type Change, type Reading } from './net.ts';
import type { BasinInfo } from './whatif.ts';

export const AMBER = '#e0a84c', SKY = '#6aa7e8', ROSE = '#e3736b';

export interface RecordState {
  info: BasinInfo;
  tgtRec: Days; tgtNow: Days;          // target-decade forcing, recorded and as read
  calRec: Days; qRec: Float64Array;    // gauged decade, recorded
  qNow: Float64Array; i0: number;      // flow as read, and the first day still read (fewer gauged years)
  now: Reading; recd: Reading;         // reading now, and on the recorded input
  climateTouched: boolean; gaugeTouched: boolean;
  accent: string;
}

const YEAR0 = 1989.75;   // 1 Oct 1989
const dayYear = (start: string, i: number) => { const t = Date.parse(start + 'T00:00:00Z') + i * 86400000; const d = new Date(t); return d.getUTCFullYear() + (d.getUTCMonth() + (d.getUTCDate() - 1) / 31) / 12; };

/** Monthly sums (P) or means (T) of a daily series, with the decimal year of each month's start. */
function monthly(start: string, v: Float64Array, months: Int32Array, how: 'sum' | 'mean') {
  const out: { y: number; v: number }[] = [];
  let acc = 0, n = 0, m0 = months[0], i0 = 0;
  for (let i = 0; i <= v.length; i++) {
    if (i === v.length || months[i] !== m0) {
      out.push({ y: dayYear(start, i0), v: how === 'sum' ? acc : acc / n });
      if (i === v.length) break;
      acc = 0; n = 0; m0 = months[i]; i0 = i;
    }
    acc += v[i]; n++;
  }
  return out;
}

export function drawRecord(c: Canvas2D, s: RecordState) {
  const { ctx } = c;
  const L = 46, R = 40, T = 18;
  const W = c.w - L - R;
  const hP = Math.round(c.h * 0.17), hT = Math.round(c.h * 0.09), gap = 8;
  const yP = T, yT = yP + hP + 4, yQ = yT + hT + gap, hQ = c.h - yQ - 22;
  const span = 19;   // water years 1990-2008
  const X = (y: number) => L + ((y - YEAR0) / span) * W;
  const xMid = X(1999.75);
  ctx.fillStyle = SCOPE.bg;
  ctx.fillRect(0, 0, c.w, c.h);

  // decade shading and headers
  ctx.fillStyle = 'rgba(235,230,216,0.035)';
  ctx.fillRect(L, T, xMid - L, c.h - T - 22);
  const wide = xMid - L > 300, mid = xMid - L > 170;
  label(c, wide ? 'TARGET DECADE 1989–99 · weather only' : mid ? '1989–99 · weather only' : '1989–99', L + 4, 8, SCOPE.text, 'left', 9);
  label(c, wide ? 'GAUGED DECADE 1999–2008 · weather + flow' : mid ? '1999–2008 · weather + flow' : '1999–2008 · read', xMid + 4, 8, SCOPE.text, 'left', 9);
  ctx.strokeStyle = 'rgba(235,230,216,0.25)';
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(xMid + 0.5, T); ctx.lineTo(xMid + 0.5, c.h - 22); ctx.stroke();
  ctx.setLineDash([]);

  // year ticks
  for (let y = 1990; y <= 2008; y += 2) {
    const x = X(y);
    ctx.strokeStyle = SCOPE.grid;
    ctx.beginPath(); ctx.moveTo(x + 0.5, yQ); ctx.lineTo(x + 0.5, yQ + hQ); ctx.stroke();
    label(c, String(y), x, c.h - 10, SCOPE.faint, 'center', 9);
  }

  // ---- precipitation (monthly totals), recorded outline and as read
  const pt = monthly(s.info.tgt_start, s.tgtNow.P, s.tgtNow.month, 'sum');
  const ptr = monthly(s.info.tgt_start, s.tgtRec.P, s.tgtRec.month, 'sum');
  const pc = monthly(s.info.cal_start, s.calRec.P, s.calRec.month, 'sum');
  const pmax = Math.max(...pt.map((m) => m.v), ...ptr.map((m) => m.v), ...pc.map((m) => m.v), 1);
  const bw = Math.max(1, W / (span * 12) - 0.6);
  const bar = (ms: { y: number; v: number }[], color: string, outline = false) => {
    for (const m of ms) {
      const hgt = (m.v / pmax) * hP;
      if (outline) { ctx.strokeStyle = color; ctx.strokeRect(X(m.y) + 0.5, yP + 0.5, bw, hgt); }
      else { ctx.fillStyle = color; ctx.fillRect(X(m.y), yP, bw, Math.max(0.5, hgt)); }
    }
  };
  bar(pc, 'rgba(106,167,232,0.55)');
  bar(pt, s.climateTouched ? 'rgba(106,167,232,0.85)' : 'rgba(106,167,232,0.55)');
  if (s.climateTouched) bar(ptr, 'rgba(235,230,216,0.35)', true);
  label(c, 'P', L - 6, yP + 6, SKY, 'right', 9);
  label(c, String(Math.round(pmax)), L - 6, yP + hP - 4, SCOPE.faint, 'right', 8);
  label(c, 'mm/mo', L - 6, yP + hP + 6, SCOPE.faint, 'right', 7);

  // ---- temperature (monthly means)
  const tt = monthly(s.info.tgt_start, s.tgtNow.T, s.tgtNow.month, 'mean');
  const ttr = monthly(s.info.tgt_start, s.tgtRec.T, s.tgtRec.month, 'mean');
  const tc = monthly(s.info.cal_start, s.calRec.T, s.calRec.month, 'mean');
  const all = [...tt, ...tc, ...ttr].map((m) => m.v);
  const tlo = Math.min(...all), thi = Math.max(...all);
  const YT = (v: number) => yT + hT - ((v - tlo) / (thi - tlo || 1)) * hT;
  const line = (ms: { y: number; v: number }[], color: string, width = 1) => {
    ctx.strokeStyle = color; ctx.lineWidth = width;
    ctx.beginPath();
    ms.forEach((m, j) => (j ? ctx.lineTo : ctx.moveTo).call(ctx, X(m.y), YT(m.v)));
    ctx.stroke();
  };
  if (s.climateTouched) line(ttr, 'rgba(235,230,216,0.3)');
  line(tt, s.climateTouched ? ROSE : 'rgba(224,168,76,0.7)', s.climateTouched ? 1.3 : 1);
  line(tc, 'rgba(224,168,76,0.7)');
  if (tlo < 0 && thi > 0) {
    ctx.strokeStyle = 'rgba(235,230,216,0.12)';
    ctx.beginPath(); ctx.moveTo(L, YT(0) + 0.5); ctx.lineTo(L + W, YT(0) + 0.5); ctx.stroke();
  }
  label(c, 'T', L - 6, yT + hT / 2, AMBER, 'right', 9);

  // ---- flow panel (log axis)
  const finiteQ = (a: Float64Array) => Array.from(a).filter((v) => v > 0);
  const lv = (r: Reading, t: Change) => level(t, r.sig[LEVEL[t]], r.get(t)[0]);
  const truthLvl = (t: Change) => level(t, s.info.cal_signatures[LEVEL[t]], s.info.observed_change[t]);
  const gr4jLvl = (t: Change) => level(t, s.info.cal_signatures[LEVEL[t]], s.info.gr4j_change[t]);
  const qs = [...finiteQ(s.qRec), ...finiteQ(s.qNow)];
  const qmax = Math.max(...qs, truthLvl('d_logq95') * 1.5);
  const qmin = Math.max(0.005, Math.min(Math.min(...qs), truthLvl('d_logq5') * 0.7, lv(s.now, 'd_logq5') * 0.7));
  const lo = Math.log10(qmin), hi = Math.log10(qmax);
  const YQ = (v: number) => yQ + hQ - ((Math.log10(Math.max(v, qmin)) - lo) / (hi - lo)) * hQ;
  for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) {
    const y = YQ(10 ** e);
    ctx.strokeStyle = SCOPE.grid;
    ctx.beginPath(); ctx.moveTo(L, y + 0.5); ctx.lineTo(L + W, y + 0.5); ctx.stroke();
    label(c, e >= 0 ? String(10 ** e) : (10 ** e).toFixed(-e), L - 6, y, SCOPE.faint, 'right', 9);
  }
  label(c, 'Q', L - 6, yQ + hQ / 2, s.accent, 'right', 9);
  label(c, 'mm/d', L - 6, yQ + hQ / 2 + 11, SCOPE.faint, 'right', 8);

  // daily flow, min/max per pixel column
  const flow = (q: Float64Array, start: string, from: number, color: string, alpha: number) => {
    const n = q.length;
    const x0 = X(dayYear(start, from)), x1 = X(dayYear(start, n - 1));
    const cols = Math.max(1, Math.floor(x1 - x0));
    ctx.strokeStyle = color; ctx.globalAlpha = alpha; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let px = 0; px < cols; px++) {
      const a = from + Math.floor((px / cols) * (n - from)), b = Math.max(a + 1, from + Math.floor(((px + 1) / cols) * (n - from)));
      let mn = Infinity, mx = -Infinity;
      for (let i = a; i < b && i < n; i++) { const v = q[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
      if (!Number.isFinite(mn)) continue;
      ctx.moveTo(x0 + px + 0.5, YQ(mx)); ctx.lineTo(x0 + px + 0.5, Math.max(YQ(mn), YQ(mx) + 0.8));
    }
    ctx.stroke(); ctx.globalAlpha = 1;
  };
  const calStart = s.info.cal_start;
  if (s.gaugeTouched) flow(s.qRec, calStart, 0, SCOPE.ink, 0.22);
  if (s.i0 > 0) {
    ctx.fillStyle = 'rgba(13,15,12,0.55)';
    ctx.fillRect(xMid, yQ, X(dayYear(calStart, s.i0)) - xMid, hQ);
    label(c, 'not read', (xMid + X(dayYear(calStart, s.i0))) / 2, yQ + 10, SCOPE.faint, 'center', 9);
  }
  // the flow as read lives on the record's own calendar from day i0
  const qFull = new Float64Array(s.qRec.length).fill(NaN);
  qFull.set(s.qNow, s.i0);
  flow(qFull, calStart, s.i0, s.accent, s.gaugeTouched ? 0.95 : 0.85);

  // target decade: hidden flow, and the predicted / observed / GR4J levels
  label(c, wide ? 'daily flow hidden from the network (and not in the release)' : 'daily flow hidden', (L + xMid) / 2, yQ + hQ - 8, SCOPE.faint, 'center', 9);
  const xa = L + 6, xb = xMid - 6;
  let lastY = -Infinity;
  for (const [t, name] of [['d_logq95', 'Q95'], ['d_logq', 'mean'], ['d_logq5', 'Q5']] as [Change, string][]) {
    const [mu, sd] = s.now.get(t);
    const c0 = s.now.sig[LEVEL[t]];
    const yl = YQ(level(t, c0, mu - sd)), yh = YQ(level(t, c0, mu + sd));
    ctx.fillStyle = s.accent + '38';
    ctx.fillRect(xa, Math.min(yl, yh), xb - xa, Math.abs(yl - yh));
    ctx.strokeStyle = s.accent; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(xa, YQ(lv(s.now, t))); ctx.lineTo(xb, YQ(lv(s.now, t))); ctx.stroke();
    // calibration level (no-change answer), across the gauged decade and carried back
    ctx.strokeStyle = 'rgba(235,230,216,0.35)'; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(xa, YQ(c0)); ctx.lineTo(L + W, YQ(c0)); ctx.stroke();
    // GR4J (calibrated on the gauged decade)
    ctx.strokeStyle = AMBER; ctx.setLineDash([6, 3]); ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.moveTo(xa, YQ(gr4jLvl(t))); ctx.lineTo(xb, YQ(gr4jLvl(t))); ctx.stroke();
    // observed 1989-99 level (truth); dashed when the what-if climate is not the recorded one
    ctx.strokeStyle = SCOPE.ink; ctx.lineWidth = 1.6; ctx.setLineDash(s.climateTouched ? [3, 3] : []);
    ctx.beginPath(); ctx.moveTo(xa, YQ(truthLvl(t))); ctx.lineTo(xb, YQ(truthLvl(t))); ctx.stroke();
    ctx.setLineDash([]);
    const yLab = Math.max(YQ(c0), lastY + 11);
    lastY = yLab;
    label(c, name, L + W + 5, yLab, SCOPE.text, 'left', 9);
  }
}

// ---- the six changes ----------------------------------------------------------------------------
export const NICE: Record<Change, [string, 'pct' | 'abs' | 'days']> = {
  d_logq: ['mean flow', 'pct'], d_rr: ['runoff ratio', 'abs'], d_bfi: ['baseflow index', 'abs'],
  d_logq5: ['low flow Q5', 'pct'], d_logq95: ['high flow Q95', 'pct'], d_hfd: ['half-flow date', 'days'],
};
export function fmtChange(t: Change, v: number, digits?: number) {
  if (!Number.isFinite(v)) return '—';
  const kind = NICE[t][1];
  const sgn = (x: number) => (x > 0 ? '+' : x < 0 ? '−' : '±');
  if (kind === 'pct') { const p = (Math.exp(v) - 1) * 100; return sgn(Math.round(p)) + Math.abs(p).toFixed(digits ?? 0) + '%'; }
  if (kind === 'days') return sgn(Math.round(v)) + Math.abs(v).toFixed(digits ?? 0) + ' d';
  return sgn(v) + Math.abs(v).toFixed(digits ?? 3);
}

const UNIT: Record<Change, string> = { d_logq: 'ratio, as %', d_rr: 'Q / P, absolute', d_bfi: 'Lyne-Hollick', d_logq5: 'ratio, as %', d_logq95: 'ratio, as %', d_hfd: 'days' };
/** One sigma as the demo prints it: log ratios x 100 %, days, or absolute. */
export function fmtSd(t: Change, sd: number) {
  const kind = NICE[t][1];
  return kind === 'pct' ? `${Math.round(sd * 100)}%` : kind === 'days' ? `${sd.toFixed(0)} d` : sd.toFixed(3);
}

export interface ChangeState {
  info: BasinInfo; now: Reading; recd: Reading; touched: boolean; climateTouched: boolean; accent: string;
}

export function drawChanges(c: Canvas2D, s: ChangeState) {
  const { ctx } = c;
  ctx.fillStyle = SCOPE.bg;
  ctx.fillRect(0, 0, c.w, c.h);
  const L = Math.min(118, c.w * 0.24), R = Math.min(178, c.w * 0.34), T = 16;
  const W = c.w - L - R;
  const rh = (c.h - T - 6) / CHANGES.length;
  label(c, 'change, gauged decade → target decade', L, 7, SCOPE.text, 'left', 9);
  label(c, 'network · observed · GR4J', c.w - 6, 7, SCOPE.text, 'right', 9);
  CHANGES.forEach((t, k) => {
    const y = T + k * rh + rh / 2;
    const [mu, sd] = s.now.get(t), [mu0, sd0] = s.recd.get(t);
    const obs = s.info.observed_change[t], gr = s.info.gr4j_change[t];
    const spanV = Math.max(2.2 * sd0, Math.abs(obs), Math.abs(gr), Math.abs(mu0) + sd0, Math.abs(mu) + 1.2 * sd) * 1.08;
    const X = (v: number) => L + W / 2 + (Math.max(-spanV, Math.min(spanV, v)) / spanV) * (W / 2);
    if (k % 2 === 0) { ctx.fillStyle = 'rgba(235,230,216,0.025)'; ctx.fillRect(0, y - rh / 2, c.w, rh); }
    ctx.strokeStyle = 'rgba(235,230,216,0.22)';
    ctx.beginPath(); ctx.moveTo(X(0) + 0.5, y - rh / 2 + 3); ctx.lineTo(X(0) + 0.5, y + rh / 2 - 3); ctx.stroke();
    label(c, NICE[t][0], 8, y - 5, SCOPE.ink, 'left', 10);
    label(c, UNIT[t], 8, y + 8, SCOPE.faint, 'left', 8);
    const bh = Math.min(10, rh * 0.32);
    ctx.fillStyle = s.accent + '22';
    ctx.fillRect(X(mu - 2 * sd), y - bh / 2, X(mu + 2 * sd) - X(mu - 2 * sd), bh);
    ctx.fillStyle = s.accent + '66';
    ctx.fillRect(X(mu - sd), y - bh / 2, X(mu + sd) - X(mu - sd), bh);
    if (s.touched) {   // where the recorded input put it
      ctx.strokeStyle = 'rgba(235,230,216,0.45)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(X(mu0), y, 3.5, 0, 2 * Math.PI); ctx.stroke();
    }
    ctx.fillStyle = s.accent;
    ctx.beginPath(); ctx.arc(X(mu), y, 4, 0, 2 * Math.PI); ctx.fill();
    // GR4J: amber diamond
    ctx.fillStyle = AMBER;
    const gx = X(gr);
    ctx.beginPath(); ctx.moveTo(gx, y - 5); ctx.lineTo(gx + 4, y); ctx.lineTo(gx, y + 5); ctx.lineTo(gx - 4, y); ctx.closePath(); ctx.fill();
    // observed: white tick (hollow when the what-if climate is not the recorded one)
    ctx.strokeStyle = SCOPE.ink; ctx.lineWidth = 2;
    if (s.climateTouched) ctx.setLineDash([2, 2]);
    ctx.beginPath(); ctx.moveTo(X(obs), y - bh); ctx.lineTo(X(obs), y + bh); ctx.stroke();
    ctx.setLineDash([]);
    const inside = Math.abs(obs - mu) <= sd;
    label(c, `${fmtChange(t, mu)} ±${fmtSd(t, sd)}`, c.w - R + 8, y - 6, s.accent, 'left', 10);
    label(c, `obs ${fmtChange(t, obs)} · GR4J ${fmtChange(t, gr)}`, c.w - R + 8, y + 7, s.climateTouched ? SCOPE.faint : (inside ? SCOPE.text : ROSE), 'left', 8.5);
  });
}
