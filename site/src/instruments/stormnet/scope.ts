// The storm scope: a whole storm's Dst with every hourly forecast, and the solar wind that drove it.
import { SCOPE, label, type Canvas2D } from '../kit.ts';
import type { Reading, WindRow } from './net.ts';

export interface ScopeState {
  rows: WindRow[];
  dst: number[];
  replay: (Reading & { k: number; truth: number })[];
  k: number;                       // issue hour (index into rows)
  live: Reading;                   // the reading at k, possibly with what-if wind
  modified: boolean;
  show: { physics: boolean; published: boolean; persistence: boolean };
  startMs: number;                 // UTC ms of rows[0]
  accent: string;
}

const AMBER = '#e0a84c', ROSE = '#e3736b', SKY = '#6aa7e8', LILAC = 'rgba(190,170,235,0.75)';

/** Region boxes, shared with the pointer handler so a click maps to an hour. */
export function layout(c: Canvas2D) {
  const left = 54, right = 10, top = 12;
  const w = c.w - left - right;
  const hDst = Math.round(c.h * 0.56);
  const strip = Math.round((c.h - hDst - top - 46) / 3);
  return { left, w, top, hDst, strip, y1: top + hDst + 14, y2: top + hDst + 14 + strip + 8, y3: top + hDst + 14 + 2 * (strip + 8) };
}

export function hourAt(c: Canvas2D, n: number, x: number) {
  const L = layout(c);
  return Math.round(((x - L.left) / L.w) * (n - 1));
}

export function drawScope(c: Canvas2D, s: ScopeState) {
  const { ctx } = c;
  const n = s.dst.length;
  const L = layout(c);
  const X = (i: number) => L.left + (i / (n - 1)) * L.w;
  ctx.fillStyle = SCOPE.bg;
  ctx.fillRect(0, 0, c.w, c.h);

  // ---- Dst panel: range from the data, always showing 0 and the storm thresholds reached
  const finite = s.dst.filter(Number.isFinite);
  const lowAll = Math.min(...finite, ...s.replay.map(r => r.minimum - r.sigma), s.live.minimum - s.live.sigma);
  const lo = Math.min(-60, Math.floor((lowAll - 15) / 50) * 50), hi = 40;
  const Y = (v: number) => L.top + ((hi - v) / (hi - lo)) * L.hDst;

  // lead-in shading: the first 35 hours only feed the first forecast
  ctx.fillStyle = 'rgba(235,230,216,0.035)';
  ctx.fillRect(L.left, L.top, X(35) - L.left, L.hDst);
  label(c, 'lead-in', X(17), L.top + L.hDst - 8, SCOPE.faint, 'center', 9);

  // thresholds
  ctx.lineWidth = 1;
  for (const [v, name] of [[0, ''], [-50, 'storm'], [-100, 'intense'], [-150, ''], [-200, 'severe'], [-300, ''], [-400, '']] as const) {
    if (v < lo) continue;
    ctx.strokeStyle = v === 0 ? 'rgba(235,230,216,0.16)' : SCOPE.grid;
    ctx.setLineDash(v === 0 ? [] : [3, 4]);
    ctx.beginPath(); ctx.moveTo(L.left, Y(v) + 0.5); ctx.lineTo(L.left + L.w, Y(v) + 0.5); ctx.stroke();
    label(c, v === 0 ? '0' : '−' + Math.abs(v), L.left - 5, Y(v), SCOPE.faint, 'right', 9);
    if (name) label(c, name, L.left + L.w - 4, Y(v) - 7, SCOPE.faint, 'right', 9);
  }
  ctx.setLineDash([]);

  // day ticks
  const DAY = 86400000;
  const firstDay = Math.ceil(s.startMs / DAY) * DAY;
  for (let t = firstDay; t <= s.startMs + (n - 1) * 3600000; t += DAY) {
    const i = (t - s.startMs) / 3600000;
    ctx.strokeStyle = SCOPE.grid;
    ctx.beginPath(); ctx.moveTo(X(i) + 0.5, L.top); ctx.lineTo(X(i) + 0.5, L.y3 + L.strip); ctx.stroke();
    const d = new Date(t);
    label(c, d.toISOString().slice(5, 10).replace('-', '/'), X(i) + 3, L.y3 + L.strip + 12, SCOPE.text, 'left', 9);
  }

  // forecast trail: each issue's three-hour minimum drawn at the middle of its window
  const at = (r: { k: number }) => X(r.k + 2);
  ctx.fillStyle = s.accent + '30';
  ctx.beginPath();
  s.replay.forEach((r, j) => (j ? ctx.lineTo : ctx.moveTo).call(ctx, at(r), Y(r.minimum + r.sigma)));
  for (let j = s.replay.length - 1; j >= 0; j--) ctx.lineTo(at(s.replay[j]), Y(s.replay[j].minimum - s.replay[j].sigma));
  ctx.closePath(); ctx.fill();

  const line = (key: 'minimum' | 'classical' | 'published' | 'persistence', color: string, dash: number[] = [], width = 1.4) => {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
    ctx.beginPath();
    s.replay.forEach((r, j) => (j ? ctx.lineTo : ctx.moveTo).call(ctx, at(r), Y(r[key])));
    ctx.stroke(); ctx.setLineDash([]);
  };
  if (s.show.persistence) line('persistence', 'rgba(235,230,216,0.35)', [2, 3], 1);
  if (s.show.published) line('published', LILAC, [5, 3], 1.2);
  if (s.show.physics) line('classical', AMBER, [6, 3], 1.3);
  line('minimum', s.accent, [], 1.7);

  // observed Dst
  ctx.strokeStyle = SCOPE.ink; ctx.lineWidth = 1.6;
  ctx.beginPath();
  let pen = false;
  s.dst.forEach((v, i) => {
    if (!Number.isFinite(v)) { pen = false; return; }
    (pen ? ctx.lineTo : ctx.moveTo).call(ctx, X(i), Y(v)); pen = true;
  });
  ctx.stroke();

  // scrubber, its verification window and the live reading
  const k = s.k;
  ctx.fillStyle = 'rgba(235,230,216,0.06)';
  ctx.fillRect(X(k), L.top, X(k + 3) - X(k), L.y3 + L.strip - L.top);
  ctx.strokeStyle = s.accent; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(X(k) + 0.5, L.top); ctx.lineTo(X(k) + 0.5, L.y3 + L.strip); ctx.stroke();
  const m = s.live, xm = X(k + 2);
  ctx.fillStyle = s.accent + '55';
  ctx.fillRect(X(k + 1), Y(m.minimum + m.sigma), X(k + 3) - X(k + 1), Y(m.minimum - m.sigma) - Y(m.minimum + m.sigma));
  ctx.strokeStyle = s.accent; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(X(k + 1), Y(m.minimum)); ctx.lineTo(X(k + 3), Y(m.minimum)); ctx.stroke();
  if (!s.modified) {
    const fut = s.dst.slice(k + 1, k + 4).filter(Number.isFinite);
    if (fut.length) {
      ctx.fillStyle = SCOPE.ink;
      ctx.beginPath(); ctx.arc(xm, Y(Math.min(...fut)), 3.2, 0, 2 * Math.PI); ctx.fill();
    }
  }
  label(c, 'Dst · nT', L.left + 4, L.top + 9, SCOPE.text, 'left', 9);

  // ---- wind strips
  const strip = (y: number, title: string, values: number[], color: (v: number) => string, lo2: number, hi2: number, bars: boolean, zero?: number) => {
    const Ys = (v: number) => y + ((hi2 - Math.max(lo2, Math.min(hi2, v))) / (hi2 - lo2)) * L.strip;
    ctx.fillStyle = 'rgba(235,230,216,0.025)'; ctx.fillRect(L.left, y, L.w, L.strip);
    if (zero !== undefined) {
      ctx.strokeStyle = 'rgba(235,230,216,0.14)';
      ctx.beginPath(); ctx.moveTo(L.left, Ys(zero) + 0.5); ctx.lineTo(L.left + L.w, Ys(zero) + 0.5); ctx.stroke();
    }
    const bw = Math.max(1, L.w / n - 0.4);
    values.forEach((v, i) => {
      if (!Number.isFinite(v)) return;
      ctx.fillStyle = color(v);
      if (bars) { const y0 = Ys(zero ?? lo2), y1 = Ys(v); ctx.fillRect(X(i) - bw / 2, Math.min(y0, y1), bw, Math.max(1, Math.abs(y1 - y0))); }
      else ctx.fillRect(X(i) - bw / 2, Ys(v) - 1, bw, 2);
    });
    label(c, title, L.left - 5, y + L.strip / 2, SCOPE.text, 'right', 9);
  };
  const bz = s.rows.map(r => r[0]), v = s.rows.map(r => r[1]), p = s.rows.map(r => r[2]);
  const bzMax = Math.max(10, ...bz.filter(Number.isFinite).map(Math.abs));
  strip(L.y1, 'Bz', bz, x => (x < 0 ? ROSE : SKY), -bzMax, bzMax, true, 0);
  strip(L.y2, 'speed', v, () => 'rgba(235,230,216,0.6)', 250, Math.max(700, ...v.filter(Number.isFinite)), false);
  strip(L.y3, 'pressure', p, () => AMBER, 0, Math.max(8, ...p.filter(Number.isFinite)), true, 0);
}
