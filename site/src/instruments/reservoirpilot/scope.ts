// The gauge board: a decade (or one year) of a real exam dam, operated by every policy on the same inflow.
// Storage on top (the water J does not charge for), release against demand and the safe channel capacity Qmax,
// the inflow the dam received (and what the policies read), and the pilot's two daily decisions.
import { SCOPE, label, type Canvas2D } from '../kit.ts';

export const AMBER = '#e0a84c', SLATE = '#8fa9b8', LILAC = 'rgba(190,170,235,0.9)', ROSE = '#ff6b6b', SKY = '#6aa7e8';

export interface Trace { R: Float64Array; S: Float64Array }
export interface ScopeState {
  start: string;                 // ISO date of day 0
  n: number;
  view: [number, number];        // day range shown
  cap: number; Qmax: number;     // of the (possibly changed) task
  capRecord: number;             // the real dam's capacity (record storage is drawn on its own scale)
  I: Float64Array; seen: Float64Array | null; Iclim: Float64Array; D: Float64Array;
  pilot: Trace & { H: Float64Array; E: Float64Array };
  rule: Trace; sop: Trace; dp: { S: Float64Array } | null; record: Trace | null;
  show: { sop: boolean; rule: boolean; dp: boolean; record: boolean };
  means: { name: string; v: number; color: string }[];
  pulse: [number, number] | null; // pulse day range
  hover: number | null;
  accent: string; compact: boolean;
}

export function layout(c: Canvas2D, compact: boolean) {
  const left = 44, right = 56, top = 8, gap = 9, axis = 16;
  const w = c.w - left - right;
  const avail = c.h - top - axis - gap * (compact ? 2 : 3);
  const fr = compact ? [0.5, 0.28, 0.22, 0] : [0.44, 0.26, 0.2, 0.1];
  const hs = fr.map(f => Math.round(avail * f));
  const ys: number[] = []; let y = top;
  for (const hh of hs) { ys.push(y); y += hh + (hh ? gap : 0); }
  return { left, w, top, hs, ys, bottom: y - gap };
}

export function dayAt(c: Canvas2D, s: ScopeState, x: number) {
  const L = layout(c, s.compact);
  const [a, b] = s.view;
  return Math.max(a, Math.min(b - 1, Math.round(a + ((x - L.left) / L.w) * (b - 1 - a))));
}

/** Min/max per pixel column, joined, so a decade at 5 days per pixel and a year at 2 pixels per day both draw faithfully. */
function line(ctx: CanvasRenderingContext2D, v: ArrayLike<number>, a: number, b: number, X: (i: number) => number, Y: (v: number) => number) {
  const n = b - a, cols = Math.max(1, Math.round(X(b - 1) - X(a)));
  ctx.beginPath();
  if (n <= cols * 1.5) {
    for (let i = a; i < b; i++) (i === a ? ctx.moveTo : ctx.lineTo).call(ctx, X(i), Y(v[i]));
  } else {
    for (let px = 0; px < cols; px++) {
      const i0 = a + Math.floor((px / cols) * n), i1 = Math.max(i0 + 1, a + Math.floor(((px + 1) / cols) * n));
      let lo = Infinity, hi = -Infinity;
      for (let i = i0; i < i1; i++) { if (v[i] < lo) lo = v[i]; if (v[i] > hi) hi = v[i]; }
      const x = X(i0);
      if (px === 0) ctx.moveTo(x, Y(v[i0])); else ctx.lineTo(x, Y(v[i0]));
      ctx.lineTo(x, Y(lo)); ctx.lineTo(x, Y(hi)); ctx.lineTo(x, Y(v[i1 - 1]));
    }
  }
  ctx.stroke();
}

function bars(ctx: CanvasRenderingContext2D, v: ArrayLike<number>, a: number, b: number, X: (i: number) => number, Y: (v: number) => number, color: string) {
  const n = b - a, cols = Math.max(1, Math.round(X(b - 1) - X(a)));
  ctx.fillStyle = color;
  const y0 = Y(0);
  if (n <= cols) {
    const bw = Math.max(1, cols / n - 0.3);
    for (let i = a; i < b; i++) { const y = Y(v[i]); ctx.fillRect(X(i) - bw / 2, Math.min(y, y0), bw, Math.max(1, Math.abs(y - y0))); }
    return;
  }
  for (let px = 0; px < cols; px++) {
    const i0 = a + Math.floor((px / cols) * n), i1 = Math.max(i0 + 1, a + Math.floor(((px + 1) / cols) * n));
    let lo = 0, hi = 0;
    for (let i = i0; i < i1; i++) { if (v[i] < lo) lo = v[i]; if (v[i] > hi) hi = v[i]; }
    const yh = Y(hi), yl = Y(lo);
    ctx.fillRect(X(i0), yh, 1, Math.max(1, yl - yh));
  }
}

const pct = (v: number) => `${Math.round(v * 100)}%`;

/** A panel title on a dark backing, so traces never run through the words. */
function tag(c: Canvas2D, text: string, x: number, y: number) {
  c.ctx.font = '9px "Martian Mono Variable", ui-monospace, monospace';
  const w = c.ctx.measureText(text).width;
  c.ctx.fillStyle = 'rgba(13,15,12,0.78)';
  c.ctx.fillRect(x - 3, y - 6, w + 6, 12);
  label(c, text, x, y, SCOPE.text, 'left', 9);
}

export function drawScope(c: Canvas2D, s: ScopeState) {
  const { ctx } = c;
  const L = layout(c, s.compact);
  const [a, b] = s.view;
  const X = (i: number) => L.left + ((i - a) / Math.max(1, b - 1 - a)) * L.w;
  ctx.fillStyle = SCOPE.bg;
  ctx.fillRect(0, 0, c.w, c.h);
  ctx.save();
  ctx.beginPath(); ctx.rect(L.left - 1, 0, L.w + 2, c.h); ctx.clip();

  // ---- time grid: calendar years (or months when one year is shown)
  const t0 = Date.parse(s.start + 'T00:00:00Z');
  const dayOf = (ms: number) => Math.round((ms - t0) / 86400000);
  const ticks: [number, string][] = [];
  const first = new Date(t0 + a * 86400000);
  if (b - a > 800) {
    for (let y = first.getUTCFullYear() + 1; ; y++) { const d = dayOf(Date.UTC(y, 0, 1)); if (d >= b) break; ticks.push([d, String(y)]); }
  } else {
    for (let m = 0; ; m++) {
      const ms = Date.UTC(first.getUTCFullYear(), first.getUTCMonth() + 1 + m, 1), d = dayOf(ms);
      if (d >= b) break;
      const dt = new Date(ms);
      ticks.push([d, dt.getUTCMonth() === 0 ? String(dt.getUTCFullYear()) : 'JFMAMJJASOND'[dt.getUTCMonth()]]);
    }
  }
  ctx.strokeStyle = SCOPE.grid; ctx.lineWidth = 1;
  ctx.beginPath();
  for (const [d] of ticks) { const x = Math.round(X(d)) + 0.5; ctx.moveTo(x, L.top); ctx.lineTo(x, L.bottom); }
  ctx.stroke();
  ctx.restore();
  for (const [d, t] of ticks) if (X(d) > L.left + 4 && X(d) < L.left + L.w - 10) label(c, t, X(d) + 3, L.bottom + 10, SCOPE.text, 'left', 9);

  if (s.pulse) {
    const [p0, p1] = s.pulse;
    if (p1 >= a && p0 < b) {
      ctx.fillStyle = 'rgba(106,167,232,0.08)';
      ctx.fillRect(Math.max(L.left, X(p0)), L.top, Math.min(L.left + L.w, X(p1)) - Math.max(L.left, X(p0)), L.bottom - L.top);
    }
  }

  const panel = (k: number, lo: number, hi: number) => {
    const y = L.ys[k], hh = L.hs[k];
    const Y = (v: number) => y + ((hi - Math.max(lo, Math.min(hi, v))) / (hi - lo)) * hh;
    ctx.fillStyle = 'rgba(235,230,216,0.022)'; ctx.fillRect(L.left, y, L.w, hh);
    return Y;
  };
  const clipTo = (k: number) => { ctx.save(); ctx.beginPath(); ctx.rect(L.left, L.ys[k] - 1, L.w, L.hs[k] + 2); ctx.clip(); };
  const stroke = (color: string, width: number, dash: number[] = []) => { ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash); };

  // ---- 1. storage, fraction of capacity
  {
    const Y = panel(0, 0, 1.04);
    ctx.lineWidth = 1;
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      ctx.strokeStyle = v === 1 || v === 0 ? 'rgba(235,230,216,0.16)' : SCOPE.grid; ctx.setLineDash(v === 1 ? [] : [3, 4]);
      ctx.beginPath(); ctx.moveTo(L.left, Math.round(Y(v)) + 0.5); ctx.lineTo(L.left + L.w, Math.round(Y(v)) + 0.5); ctx.stroke();
      if (v !== 0.25 && v !== 0.75) label(c, pct(v), L.left - 5, Y(v), SCOPE.faint, 'right', 9);
    }
    ctx.setLineDash([]);
    clipTo(0);
    const frac = (S: Float64Array, cap: number) => { const o = new Float64Array(S.length); for (let i = 0; i < S.length; i++) o[i] = S[i] / cap; return o; };
    if (s.record && s.show.record) { stroke('rgba(235,230,216,0.8)', 1); line(ctx, frac(s.record.S, s.capRecord), a, b, X, Y); }
    if (s.dp && s.show.dp) { stroke(LILAC, 1.2, [5, 3]); line(ctx, frac(s.dp.S, s.cap), a, b, X, Y); }
    if (s.show.sop) { stroke(SLATE, 1.1); line(ctx, frac(s.sop.S, s.cap), a, b, X, Y); }
    if (s.show.rule) { stroke(AMBER, 1.1, [6, 3]); line(ctx, frac(s.rule.S, s.cap), a, b, X, Y); }
    stroke(s.accent, 1.9); line(ctx, frac(s.pilot.S, s.cap), a, b, X, Y);
    ctx.setLineDash([]);
    ctx.restore();
    // decade means at the right edge: the storage J does not charge for
    const used: number[] = [];
    label(c, 'mean', L.left + L.w + 6, L.ys[0] + 4, SCOPE.faint, 'left', 8);
    for (const m of [...s.means].sort((p, q) => q.v - p.v)) {
      let y = Y(m.v);
      for (const u of used) if (Math.abs(u - y) < 10) y = u + 10;
      used.push(y);
      ctx.fillStyle = m.color; ctx.fillRect(L.left + L.w + 2, Y(m.v) - 1, 5, 2);
      label(c, pct(m.v), L.left + L.w + 10, y, m.color, 'left', 9);
    }
    tag(c, 'storage · share of capacity', L.left + 5, L.ys[0] + 8);
  }

  // ---- 2. release / Qmax, with demand and the safe channel capacity
  {
    let mx = 1.25;
    for (let i = a; i < b; i++) { const r = s.pilot.R[i] / s.Qmax; if (r > mx) mx = r; if (s.show.sop) { const q = s.sop.R[i] / s.Qmax; if (q > mx) mx = q; } }
    mx = Math.min(mx * 1.05, 3);
    const Y = panel(1, 0, mx);
    clipTo(1);
    // demand as a filled area
    ctx.fillStyle = 'rgba(235,230,216,0.13)';
    ctx.beginPath(); ctx.moveTo(X(a), Y(0));
    const step = Math.max(1, Math.floor((b - a) / L.w));
    for (let i = a; i < b; i += step) ctx.lineTo(X(i), Y(s.D[i] / s.Qmax));
    ctx.lineTo(X(b - 1), Y(s.D[b - 1] / s.Qmax)); ctx.lineTo(X(b - 1), Y(0)); ctx.closePath(); ctx.fill();
    const rel = (R: Float64Array) => { const o = new Float64Array(R.length); for (let i = 0; i < R.length; i++) o[i] = R[i] / s.Qmax; return o; };
    if (s.show.sop) { stroke(SLATE, 1); line(ctx, rel(s.sop.R), a, b, X, Y); }
    if (s.show.rule) { stroke(AMBER, 1, [6, 3]); line(ctx, rel(s.rule.R), a, b, X, Y); }
    stroke(s.accent, 1.5); line(ctx, rel(s.pilot.R), a, b, X, Y);
    ctx.setLineDash([]);
    // Qmax
    ctx.strokeStyle = 'rgba(255,107,107,0.6)'; ctx.setLineDash([4, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L.left, Math.round(Y(1)) + 0.5); ctx.lineTo(L.left + L.w, Math.round(Y(1)) + 0.5); ctx.stroke(); ctx.setLineDash([]);
    // days above Qmax
    const mark = (R: Float64Array, color: string, dy: number) => {
      ctx.fillStyle = color;
      for (let i = a; i < b; i++) if (R[i] > s.Qmax) ctx.fillRect(X(i) - 1, L.ys[1] + dy, 2, 4);
    };
    if (s.show.sop) mark(s.sop.R, SLATE, 1);
    mark(s.pilot.R, ROSE, 6);
    ctx.restore();
    label(c, 'Qmax', L.left + L.w + 6, Y(1), 'rgba(255,107,107,0.85)', 'left', 9);
    label(c, '0', L.left - 5, Y(0), SCOPE.faint, 'right', 9);
    label(c, '1', L.left - 5, Y(1), SCOPE.faint, 'right', 9);
    tag(c, 'release ÷ Qmax · shaded: demand', L.left + 5, L.ys[1] + 15);
  }

  // ---- 3. inflow / Qmax: what arrived, what the policies read, and the calibration climatology they expect
  {
    let lo = 0, hi = 0.2;
    for (let i = a; i < b; i++) { const v = s.I[i] / s.Qmax; if (v > hi) hi = v; if (v < lo) lo = v; if (s.seen) { const u = s.seen[i] / s.Qmax; if (u > hi) hi = u; if (u < lo) lo = u; } }
    const Y = panel(2, lo * 1.05, hi * 1.05);
    clipTo(2);
    const rel = (v: Float64Array) => { const o = new Float64Array(v.length); for (let i = 0; i < v.length; i++) o[i] = v[i] / s.Qmax; return o; };
    if (s.seen) bars(ctx, rel(s.seen), a, b, X, Y, s.accent + '70');
    bars(ctx, rel(s.I), a, b, X, Y, 'rgba(106,167,232,0.75)');
    stroke('rgba(235,230,216,0.6)', 1, [3, 3]); line(ctx, rel(s.Iclim), a, b, X, Y); ctx.setLineDash([]);
    ctx.restore();
    tag(c, 'inflow ÷ Qmax · dashed: calibration-half normal', L.left + 5, L.ys[2] + 8);
    label(c, '0', L.left - 5, Y(0), SCOPE.faint, 'right', 9);
  }

  // ---- 4. the pilot's decisions: h (share of demand released), e (extra release for flood space)
  if (!s.compact && L.hs[3] > 0) {
    const Y = panel(3, 0, 1);
    clipTo(3);
    stroke(s.accent, 1); line(ctx, s.pilot.H, a, b, X, Y);
    stroke(AMBER, 1); line(ctx, s.pilot.E, a, b, X, Y);
    ctx.restore();
    label(c, 'h', L.left - 14, L.ys[3] + L.hs[3] * 0.3, s.accent, 'right', 9);
    label(c, 'e', L.left - 5, L.ys[3] + L.hs[3] * 0.3, AMBER, 'right', 9);
    tag(c, 'pilot decisions · h share of demand · e flood-space release', L.left + 5, L.ys[3] + 7);
  }

  // ---- hover cursor
  if (s.hover != null && s.hover >= a && s.hover < b) {
    const x = Math.round(X(s.hover)) + 0.5;
    ctx.strokeStyle = 'rgba(235,230,216,0.35)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, L.top); ctx.lineTo(x, L.bottom); ctx.stroke();
  }
}
