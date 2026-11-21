// The FlowCast scope: a river's daily hydrograph (observed USGS flow against FlowCast's mean and its 25-75 % and
// 5-95 % bands), with the weather that drove it underneath: precipitation (rain or snow) and air temperature.
import { SCOPE, label, type Canvas2D } from '../kit.ts';

export interface Fan { day: number; mean: number[]; lo: number[]; hi: number[]; persist: number }

export interface ScopeState {
  n: number;                 // predicted days
  startMs: number;           // UTC ms of predicted day 0
  i0: number; i1: number;    // visible window [i0, i1)
  obs: ArrayLike<number>;
  mean: ArrayLike<number>;   // the reading drawn in the accent (ungauged, or the nowcast at the chosen lead)
  q: ArrayLike<number>;      // its quantiles, 5 per day
  ref: ArrayLike<number> | null;      // ungauged mean drawn dashed in nowcast mode
  fan: Fan | null;           // nowcast: the forecast issued on the cursor day for 1-7 days ahead
  P: ArrayLike<number>; P0: ArrayLike<number>;   // precipitation on the predicted days, what-if and recorded
  T: ArrayLike<number>; T0: ArrayLike<number>;   // daily mean air temperature, what-if and recorded
  cursor: number;
  outage: [number, number] | null;
  log: boolean;
  accent: string;
  cursorLabel: string;
}

const AMBER = '#e0a84c', SKY = '#6aa7e8', SNOW = 'rgba(214,205,255,0.9)', ROSE = '#e3736b';
const MONTH = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'];

export function layout(c: Canvas2D) {
  const left = 46, right = 10, top = 10;
  const w = c.w - left - right;
  const hq = Math.round(c.h * 0.58);
  const hp = Math.round(c.h * 0.15), ht = Math.round(c.h * 0.12);
  const yp = top + hq + 10, yt = yp + hp + 8;
  return { left, w, top, hq, hp, ht, yp, yt, bottom: yt + ht };
}

export function dayAt(c: Canvas2D, s: { i0: number; i1: number }, x: number) {
  const L = layout(c);
  return Math.round(s.i0 + ((x - L.left) / L.w) * (s.i1 - 1 - s.i0));
}

export function drawScope(c: Canvas2D, s: ScopeState) {
  const { ctx } = c;
  const L = layout(c);
  const span = Math.max(1, s.i1 - 1 - s.i0);
  const X = (i: number) => L.left + ((i - s.i0) / span) * L.w;
  ctx.fillStyle = SCOPE.bg;
  ctx.fillRect(0, 0, c.w, c.h);

  // ---- flow axis: from the observed flow and the mean in view (the 95 % bound may run off the top)
  let top = 0.5;
  for (let i = s.i0; i < s.i1; i++) {
    if (Number.isFinite(s.obs[i])) top = Math.max(top, s.obs[i]);
    top = Math.max(top, s.mean[i]);
    if (s.ref) top = Math.max(top, s.ref[i]);
  }
  if (s.fan) for (const v of s.fan.mean) top = Math.max(top, v);
  top *= 1.12;
  const lo = 0.01;
  const Y = s.log
    ? (v: number) => L.top + L.hq * (1 - (Math.log10(Math.max(v, lo)) - Math.log10(lo)) / (Math.log10(top) - Math.log10(lo)))
    : (v: number) => L.top + L.hq * (1 - Math.max(0, Math.min(v, top)) / top);
  const clampY = (y: number) => Math.max(L.top, Math.min(L.top + L.hq, y));

  // grid + y labels
  ctx.lineWidth = 1;
  const ticks = s.log ? [0.01, 0.1, 1, 10, 100].filter(v => v <= top) : niceTicks(top);
  for (const v of ticks) {
    ctx.strokeStyle = SCOPE.grid;
    ctx.beginPath(); ctx.moveTo(L.left, Math.round(Y(v)) + 0.5); ctx.lineTo(L.left + L.w, Math.round(Y(v)) + 0.5); ctx.stroke();
    label(c, v < 1 ? String(v) : v.toFixed(0), L.left - 5, Y(v), SCOPE.faint, 'right', 9);
  }
  // month ticks over every strip
  const DAY = 86400000;
  const every = span > 500 ? 2 : 1;
  for (let i = s.i0; i < s.i1; i++) {
    const d = new Date(s.startMs + i * DAY);
    if (d.getUTCDate() !== 1 || d.getUTCMonth() % every) continue;
    ctx.strokeStyle = d.getUTCMonth() === 9 ? 'rgba(235,230,216,0.16)' : SCOPE.grid;
    ctx.beginPath(); ctx.moveTo(Math.round(X(i)) + 0.5, L.top); ctx.lineTo(Math.round(X(i)) + 0.5, L.bottom); ctx.stroke();
    const txt = d.getUTCMonth() === 0 || d.getUTCMonth() === 9 ? MONTH[d.getUTCMonth()] + " '" + String(d.getUTCFullYear()).slice(2) : MONTH[d.getUTCMonth()];
    label(c, txt, X(i) + 3, L.bottom + 9, SCOPE.text, 'left', 9);
  }

  // gauge outage shading
  if (s.outage) {
    const [a, b] = s.outage;
    ctx.fillStyle = 'rgba(227,115,107,0.10)';
    ctx.fillRect(X(Math.max(a, s.i0)), L.top, X(Math.min(b, s.i1 - 1)) - X(Math.max(a, s.i0)), L.hq);
    label(c, 'gauge out', X(Math.max(a, s.i0)) + 3, L.top + 8, ROSE, 'left', 9);
  }

  // bands: 5-95 % and 25-75 %
  const band = (j0: number, j1: number, alpha: string) => {
    ctx.fillStyle = s.accent + alpha;
    ctx.beginPath();
    for (let i = s.i0; i < s.i1; i++) (i === s.i0 ? ctx.moveTo : ctx.lineTo).call(ctx, X(i), clampY(Y(s.q[i * 5 + j1])));
    for (let i = s.i1 - 1; i >= s.i0; i--) ctx.lineTo(X(i), clampY(Y(s.q[i * 5 + j0])));
    ctx.closePath(); ctx.fill();
  };
  band(0, 4, '26');
  band(1, 3, '40');

  const line = (v: ArrayLike<number>, color: string, width: number, dash: number[] = []) => {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
    ctx.beginPath();
    let pen = false;
    for (let i = s.i0; i < s.i1; i++) {
      if (!Number.isFinite(v[i])) { pen = false; continue; }
      (pen ? ctx.lineTo : ctx.moveTo).call(ctx, X(i), clampY(Y(v[i]))); pen = true;
    }
    ctx.stroke(); ctx.setLineDash([]);
  };
  if (s.ref) line(s.ref, 'rgba(235,230,216,0.35)', 1, [4, 3]);
  line(s.mean, s.accent, 1.5);
  line(s.obs, SCOPE.ink, 1.1);

  // cursor (issue day in nowcast mode)
  const xc = X(s.cursor);
  ctx.strokeStyle = 'rgba(235,230,216,0.45)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(Math.round(xc) + 0.5, L.top); ctx.lineTo(Math.round(xc) + 0.5, L.bottom); ctx.stroke();
  label(c, s.cursorLabel, Math.min(xc + 4, L.left + L.w - 4), L.top + 20, SCOPE.text, xc + 4 > L.left + L.w - 120 ? 'right' : 'left', 9);

  // nowcast fan: the forecast issued on the cursor day, 1-7 days ahead, and persistence
  if (s.fan) {
    const f = s.fan, xs = f.mean.map((_, k) => X(f.day + k + 1));
    ctx.fillStyle = AMBER + '30';
    ctx.beginPath();
    xs.forEach((x, k) => (k ? ctx.lineTo : ctx.moveTo).call(ctx, x, clampY(Y(f.hi[k]))));
    for (let k = xs.length - 1; k >= 0; k--) ctx.lineTo(xs[k], clampY(Y(f.lo[k])));
    ctx.closePath(); ctx.fill();
    ctx.strokeStyle = 'rgba(235,230,216,0.55)'; ctx.setLineDash([2, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(xc, clampY(Y(f.persist))); ctx.lineTo(xs[xs.length - 1], clampY(Y(f.persist))); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = AMBER; ctx.lineWidth = 2;
    ctx.beginPath();
    xs.forEach((x, k) => (k ? ctx.lineTo : ctx.moveTo).call(ctx, x, clampY(Y(f.mean[k])))); ctx.stroke();
    ctx.fillStyle = AMBER;
    xs.forEach((x, k) => { ctx.beginPath(); ctx.arc(x, clampY(Y(f.mean[k])), 2.2, 0, 2 * Math.PI); ctx.fill(); });
  }
  label(c, 'flow · mm/day' + (s.log ? ' (log)' : ''), L.left + 4, L.top + 7, SCOPE.text, 'left', 9);

  // ---- precipitation strip (snow when the day's mean temperature is below 0 °C, as the snow-share static counts it)
  let pmax = 10;
  for (let i = s.i0; i < s.i1; i++) pmax = Math.max(pmax, s.P[i], s.P0[i]);
  ctx.fillStyle = 'rgba(235,230,216,0.025)'; ctx.fillRect(L.left, L.yp, L.w, L.hp);
  const bw = Math.max(1, L.w / (span + 1) - 0.3);
  for (let i = s.i0; i < s.i1; i++) {
    const h0 = (s.P0[i] / pmax) * L.hp, h1 = (s.P[i] / pmax) * L.hp;
    if (s.P0[i] !== s.P[i]) { ctx.fillStyle = 'rgba(235,230,216,0.18)'; ctx.fillRect(X(i) - bw / 2, L.yp + L.hp - h0, bw, h0); }
    ctx.fillStyle = s.T[i] < 0 ? SNOW : SKY;
    ctx.fillRect(X(i) - bw / 2, L.yp + L.hp - h1, bw, Math.max(h1, s.P[i] > 0 ? 0.6 : 0));
  }
  label(c, 'P', L.left - 5, L.yp + L.hp / 2 - 5, SCOPE.text, 'right', 9);
  label(c, pmax.toFixed(0), L.left - 5, L.yp + 5, SCOPE.faint, 'right', 8);

  // ---- temperature strip
  let tlo = -5, thi = 25;
  for (let i = s.i0; i < s.i1; i++) { tlo = Math.min(tlo, s.T[i], s.T0[i]); thi = Math.max(thi, s.T[i], s.T0[i]); }
  const YT = (v: number) => L.yt + L.ht * (1 - (v - tlo) / (thi - tlo));
  ctx.fillStyle = 'rgba(235,230,216,0.025)'; ctx.fillRect(L.left, L.yt, L.w, L.ht);
  ctx.strokeStyle = 'rgba(214,205,255,0.35)'; ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(L.left, Math.round(YT(0)) + 0.5); ctx.lineTo(L.left + L.w, Math.round(YT(0)) + 0.5); ctx.stroke(); ctx.setLineDash([]);
  const tline = (v: ArrayLike<number>, color: string) => {
    ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.beginPath();
    for (let i = s.i0; i < s.i1; i++) (i === s.i0 ? ctx.moveTo : ctx.lineTo).call(ctx, X(i), YT(v[i]));
    ctx.stroke();
  };
  if (s.T0[s.i0] !== s.T[s.i0]) tline(s.T0, 'rgba(235,230,216,0.22)');
  tline(s.T, AMBER);
  label(c, 'T', L.left - 5, L.yt + L.ht / 2 - 5, SCOPE.text, 'right', 9);
  label(c, '0°', L.left - 5, YT(0), SCOPE.faint, 'right', 8);
}

function niceTicks(top: number) {
  const raw = top / 4, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map(k => k * mag).find(v => v >= raw) ?? raw;
  const out: number[] = [];
  for (let v = 0; v <= top; v += step) out.push(+v.toFixed(6));
  return out;
}
