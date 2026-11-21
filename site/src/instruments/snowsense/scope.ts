// The snow scope: one water year at a snow pillow. Top: SWE, SnowSense's median and its 68/90 % bands, the observed
// pillow and the two uncalibrated physics routines. Below: the daily temperature and precipitation the network read,
// each wet day split into snow and rain by WaterSim's default partition. A second, small canvas shows the decade.
import { SCOPE, hex, label, type Canvas2D } from '../kit.ts';
import { snowFraction, type Reading, type Window } from './net.ts';

export const AMBER = '#e0a84c', LILAC = 'rgba(190,170,235,0.9)', SKY = '#6aa7e8', ROSE = '#e3736b';
/** Mix a #rrggbb colour toward white (0..1); the result stays #rrggbb so alpha suffixes still work. */
export function lighten(c: string, t: number) {
  return '#' + hex(c).map(v => Math.round(v + (255 - v) * t).toString(16).padStart(2, '0')).join('');
}
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const WARM = 61, L = 427, DAY = 864e5;

export interface ScopeState {
  w: Window;
  r: Reading;
  obs: Float64Array;            // observed end-of-day SWE over the water year
  eb: Float64Array; dd: Float64Array;
  obsMelt: number; obsOk: boolean;
  modified: boolean;
  show: { eb: boolean; dd: boolean; band90: boolean };
  hover: number;                // window day under the pointer, or -1
  accent: string;
}

export function layout(c: Canvas2D) {
  const left = 46, right = 10, top = 10;
  const w = c.w - left - right;
  const hSwe = Math.round(c.h * 0.56);
  const gapY = 10, axis = 16;
  const strip = Math.max(26, Math.round((c.h - top - hSwe - axis - 3 * gapY) / 2));
  const yT = top + hSwe + gapY + axis, yP = yT + strip + gapY;
  return { left, w, top, hSwe, strip, yT, yP, bottom: yP + strip };
}

/** Window day (0 .. 426) under x. */
export function dayAt(c: Canvas2D, x: number) {
  const L0 = layout(c);
  return Math.max(0, Math.min(L - 1, Math.round(((x - L0.left) / L0.w) * (L - 1))));
}

export function drawScope(c: Canvas2D, s: ScopeState) {
  const { ctx } = c, { w, r } = s;
  const Lz = layout(c);
  const X = (d: number) => Lz.left + (d / (L - 1)) * Lz.w;
  ctx.fillStyle = SCOPE.bg;
  ctx.fillRect(0, 0, c.w, c.h);

  // ---- SWE scale
  let top = Math.max(r.peakHi68, r.peak, s.obsOk ? Math.max(...Array.from(s.obs).filter(Number.isFinite)) : 0, 50);
  if (s.show.eb) top = Math.max(top, ...s.eb);
  if (s.show.dd) top = Math.max(top, ...s.dd);
  top *= 1.1;
  const step = top > 1200 ? 400 : top > 600 ? 200 : top > 250 ? 100 : 50;
  const Y = (v: number) => Lz.top + Lz.hSwe - (Math.min(v, top) / top) * Lz.hSwe;

  // lead-in shading: 1 Aug - 30 Sep only warms the network up
  ctx.fillStyle = 'rgba(235,230,216,0.035)';
  ctx.fillRect(Lz.left, Lz.top, X(WARM) - Lz.left, Lz.bottom - Lz.top);
  label(c, 'lead-in', (Lz.left + X(WARM)) / 2, Lz.top + 8, SCOPE.faint, 'center', 9);

  // grid and month ticks
  ctx.lineWidth = 1;
  for (let v = 0; v <= top; v += step) {
    ctx.strokeStyle = v === 0 ? 'rgba(235,230,216,0.16)' : SCOPE.grid;
    ctx.beginPath(); ctx.moveTo(Lz.left, Math.round(Y(v)) + 0.5); ctx.lineTo(Lz.left + Lz.w, Math.round(Y(v)) + 0.5); ctx.stroke();
    label(c, String(v), Lz.left - 5, Y(v), SCOPE.faint, 'right', 9);
  }
  for (let d = 0; d < L; d++) {
    const dt = new Date(w.t0 + d * DAY);
    if (dt.getUTCDate() !== 1) continue;
    ctx.strokeStyle = SCOPE.grid;
    ctx.beginPath(); ctx.moveTo(Math.round(X(d)) + 0.5, Lz.top); ctx.lineTo(Math.round(X(d)) + 0.5, Lz.bottom); ctx.stroke();
    const m = dt.getUTCMonth();
    if (d < L - 15 && (Lz.w > 480 || m % 2 === 1)) label(c, MONTHS[m], X(d) + 3, Lz.top + Lz.hSwe + 9, SCOPE.text, 'left', 9);
  }
  label(c, 'SWE mm', Lz.left + 4, Lz.top + Lz.hSwe - 8, SCOPE.text, 'left', 9);

  // gauge outage and the removed storm
  if (w.gap) {
    ctx.fillStyle = 'rgba(227,115,107,0.10)';
    ctx.fillRect(X(w.gap[0]), Lz.top, X(w.gap[1]) - X(w.gap[0]), Lz.bottom - Lz.top);
    label(c, 'gauge out', (X(w.gap[0]) + X(w.gap[1])) / 2, Lz.top + 8, ROSE, 'center', 9);
  }

  // bands: 90 % then 68 %
  const xs = (i: number) => X(WARM + i);
  const area = (lo: Float64Array, hi: Float64Array, fill: string) => {
    ctx.fillStyle = fill;
    ctx.beginPath();
    for (let i = 0; i < lo.length; i++) (i ? ctx.lineTo : ctx.moveTo).call(ctx, xs(i), Y(hi[i]));
    for (let i = lo.length - 1; i >= 0; i--) ctx.lineTo(xs(i), Y(lo[i]));
    ctx.closePath(); ctx.fill();
  };
  if (s.show.band90) area(r.lo90, r.hi90, s.accent + '22');
  area(r.lo68, r.hi68, s.accent + '48');

  const line = (v: ArrayLike<number>, color: string, width: number, dash: number[] = []) => {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < v.length; i++) {
      if (!Number.isFinite(v[i])) { pen = false; continue; }
      (pen ? ctx.lineTo : ctx.moveTo).call(ctx, xs(i), Y(v[i])); pen = true;
    }
    ctx.stroke(); ctx.setLineDash([]);
  };
  if (s.show.dd) line(s.dd, LILAC, 1.2, [5, 3]);
  if (s.show.eb) line(s.eb, AMBER, 1.3, [6, 3]);
  line(s.obs, s.modified ? 'rgba(235,230,216,0.35)' : SCOPE.ink, s.modified ? 1.1 : 1.6, s.modified ? [2, 3] : []);
  line(r.swe, s.accent, 2);

  // annual readings: peak with its 68 % interval, melt-out with its interval on the base line
  const base = Lz.top + Lz.hSwe;
  ctx.strokeStyle = s.accent; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(xs(r.peakDay), Y(r.peakLo68)); ctx.lineTo(xs(r.peakDay), Y(r.peakHi68)); ctx.stroke();
  ctx.fillStyle = s.accent;
  ctx.beginPath(); ctx.arc(xs(r.peakDay), Y(r.peak), 4, 0, 2 * Math.PI); ctx.fill();
  if (r.melt >= 0) {
    const a = r.meltLo68 >= 0 ? r.meltLo68 : r.melt, b = r.meltHi68 >= 0 ? r.meltHi68 : w.nwy - 1;
    ctx.fillStyle = s.accent + '88';
    ctx.fillRect(xs(a), base - 5, Math.max(2, xs(b) - xs(a)), 4);
    ctx.fillStyle = s.accent;
    ctx.fillRect(xs(r.melt) - 1, base - 9, 2, 9);
  }
  if (!s.modified && s.obsOk) {
    let ip = 0;
    for (let i = 1; i < s.obs.length; i++) if ((s.obs[i] || 0) > (s.obs[ip] || 0)) ip = i;
    ctx.strokeStyle = SCOPE.ink; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(xs(ip), Y(s.obs[ip]), 5, 0, 2 * Math.PI); ctx.stroke();
    if (s.obsMelt >= 0) {
      ctx.fillStyle = SCOPE.ink;
      ctx.beginPath(); ctx.moveTo(xs(s.obsMelt), base - 2); ctx.lineTo(xs(s.obsMelt) - 4, base - 10); ctx.lineTo(xs(s.obsMelt) + 4, base - 10); ctx.closePath(); ctx.fill();
    }
  }

  // ---- temperature strip (T_min..T_max band, T_mean line, 0 C)
  let tlo = Infinity, thi = -Infinity;
  for (let d = 0; d < L; d++) { tlo = Math.min(tlo, w.Tn[d]); thi = Math.max(thi, w.Tx[d]); }
  tlo = Math.min(-15, Math.floor(tlo / 5) * 5); thi = Math.max(20, Math.ceil(thi / 5) * 5);
  const YT = (v: number) => Lz.yT + ((thi - v) / (thi - tlo)) * Lz.strip;
  ctx.fillStyle = 'rgba(235,230,216,0.025)'; ctx.fillRect(Lz.left, Lz.yT, Lz.w, Lz.strip);
  ctx.fillStyle = 'rgba(235,230,216,0.12)';
  ctx.beginPath();
  for (let d = 0; d < L; d++) (d ? ctx.lineTo : ctx.moveTo).call(ctx, X(d), YT(w.Tx[d]));
  for (let d = L - 1; d >= 0; d--) ctx.lineTo(X(d), YT(w.Tn[d]));
  ctx.closePath(); ctx.fill();
  ctx.strokeStyle = 'rgba(235,230,216,0.3)'; ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(Lz.left, Math.round(YT(0)) + 0.5); ctx.lineTo(Lz.left + Lz.w, Math.round(YT(0)) + 0.5); ctx.stroke();
  ctx.setLineDash([]);
  ctx.lineWidth = 1.1;
  for (let d = 1; d < L; d++) {
    ctx.strokeStyle = w.Tm[d] > 0 ? 'rgba(227,115,107,0.9)' : 'rgba(106,167,232,0.95)';
    ctx.beginPath(); ctx.moveTo(X(d - 1), YT(w.Tm[d - 1])); ctx.lineTo(X(d), YT(w.Tm[d])); ctx.stroke();
  }
  label(c, '°C', Lz.left - 5, Lz.yT + 5, SCOPE.text, 'right', 9);
  label(c, '0', Lz.left - 5, YT(0), SCOPE.faint, 'right', 8);

  // ---- precipitation strip: snow (blue) and rain (red) by the default partition
  let pmax = 20;
  for (let d = 0; d < L; d++) pmax = Math.max(pmax, w.P[d]);
  pmax = Math.min(pmax, 120);
  const YP = (v: number) => Lz.yP + Lz.strip - (Math.min(v, pmax) / pmax) * Lz.strip;
  ctx.fillStyle = 'rgba(235,230,216,0.025)'; ctx.fillRect(Lz.left, Lz.yP, Lz.w, Lz.strip);
  const bw = Math.max(1, Lz.w / L - 0.3);
  for (let d = 0; d < L; d++) {
    const p = w.P[d];
    if (w.storm >= 0 && d >= w.storm && d < w.storm + 3) {
      ctx.strokeStyle = 'rgba(235,230,216,0.6)'; ctx.setLineDash([2, 2]);
      ctx.strokeRect(X(d) - bw / 2, Lz.yP + 2, bw, Lz.strip - 3); ctx.setLineDash([]);
    }
    if (!(p > 0)) continue;
    const fs = snowFraction(w.Tm[d], 0, 2);
    const ySnow = YP(p * fs);
    ctx.fillStyle = SKY; ctx.fillRect(X(d) - bw / 2, ySnow, bw, YP(0) - ySnow);
    ctx.fillStyle = ROSE; ctx.fillRect(X(d) - bw / 2, YP(p), bw, ySnow - YP(p));
  }
  label(c, 'P mm', Lz.left - 5, Lz.yP + Lz.strip / 2, SCOPE.text, 'right', 9);
  label(c, String(Math.round(pmax)), Lz.left - 5, Lz.yP + 4, SCOPE.faint, 'right', 8);
  if (w.storm >= 0) label(c, 'storm removed', X(w.storm) + 6, Lz.yP + 8, SCOPE.ink, 'left', 9);

  // ---- hover cursor
  if (s.hover >= 0) {
    ctx.strokeStyle = 'rgba(235,230,216,0.45)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(Math.round(X(s.hover)) + 0.5, Lz.top); ctx.lineTo(Math.round(X(s.hover)) + 0.5, Lz.bottom); ctx.stroke();
  }
}

// ---- the decade: observed peak (ink) against SnowSense's peak and 68 % interval, recorded weather
export interface DecadeYear { wy: number; peak: number; lo: number; hi: number; obs: number; ok: boolean; nse: number }

export function decadeLayout(c: Canvas2D, n: number) {
  const left = 46, right = 10, top = 14, bottom = 16;
  return { left, top, w: c.w - left - right, h: c.h - top - bottom, col: (c.w - left - right) / n };
}

export function drawDecade(c: Canvas2D, years: DecadeYear[], selected: number, accent: string) {
  const { ctx } = c;
  const Lz = decadeLayout(c, years.length);
  ctx.fillStyle = SCOPE.bg; ctx.fillRect(0, 0, c.w, c.h);
  const top = Math.max(50, ...years.map(y => Math.max(y.hi, y.ok ? y.obs : 0))) * 1.08;
  const Y = (v: number) => Lz.top + Lz.h - (Math.min(v, top) / top) * Lz.h;
  label(c, 'peak SWE, every water year · recorded weather', Lz.left, 6, SCOPE.text, 'left', 9);
  years.forEach((y, i) => {
    const x0 = Lz.left + i * Lz.col, cx = x0 + Lz.col / 2;
    if (y.wy === selected) { ctx.fillStyle = 'rgba(235,230,216,0.07)'; ctx.fillRect(x0 + 1, Lz.top - 2, Lz.col - 2, Lz.h + 4); }
    const bw = Math.min(18, Lz.col * 0.32);
    ctx.fillStyle = accent + 'cc';
    ctx.fillRect(cx - bw - 1, Y(y.peak), bw, Y(0) - Y(y.peak));
    ctx.strokeStyle = accent; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(cx - bw / 2 - 1, Y(y.lo)); ctx.lineTo(cx - bw / 2 - 1, Y(y.hi)); ctx.stroke();
    if (y.ok) {
      ctx.strokeStyle = SCOPE.ink; ctx.lineWidth = 1.2;
      ctx.strokeRect(cx + 1, Y(y.obs), bw, Y(0) - Y(y.obs));
    }
    label(c, String(y.wy), cx, Lz.top + Lz.h + 9, y.wy === selected ? SCOPE.ink : SCOPE.text, 'center', 9);
  });
  label(c, String(Math.round(top / 1.08)), Lz.left - 5, Y(top / 1.08), SCOPE.faint, 'right', 8);
  label(c, '0', Lz.left - 5, Y(0), SCOPE.faint, 'right', 8);
}
