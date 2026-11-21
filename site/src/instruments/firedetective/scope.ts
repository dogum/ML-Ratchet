// FireDetective's two scopes: the fire seen from above (the half-day pass in which each 180 m cell was first mapped as
// burned, the fuels around it, the footprints of the VIIRS pixels the network reads), and the record step by step
// (new burned area per half-day pass, day or night, and the total fire radiative power).
import { SCOPE, hex, label, type Canvas2D } from '../kit.ts';
import { RES, fuelGroup, type FireInput } from './features.ts';

export const DAY = '#e0a84c', NIGHT = '#7f95c9';
/** Muted background: fuels sit just above the scope colour so the fire reads first. */
export const FOREST = 'rgb(27,37,28)', BURNABLE = 'rgb(21,23,19)', NONBURN = 'rgb(15,16,14)', LATER = 'rgb(44,43,40)';
const BG_FOREST: RGB = [27, 37, 28], BG_BURNABLE: RGB = [21, 23, 19], BG_NONBURN: RGB = [15, 16, 14], BG_LATER: RGB = [44, 43, 40];
/** VIIRS I-band pixel at nadir. The export has no per-pixel footprint, so every pixel is drawn at this size. */
export const PIXEL_M = 375;
/** Margin around the fire in the fitted view. */
const FIT_MARGIN_M = 1000;

type RGB = [number, number, number];
const INK: RGB = [235, 230, 216];
const mix = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
export const rgb = (c: RGB) => `rgb(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])})`;

/** Burn-time colours for passes 0..n-1, a sequential ramp in the accent's family: early pale, then the accent, late deep. */
export function burnRamp(accent: string, n: number): RGB[] {
  const a = hex(accent);
  const early = mix(a, INK, 0.62), late: RGB = [a[0] * 0.5, a[1] * 0.33, a[2] * 0.25];
  return Array.from({ length: n }, (_, k) => {
    const t = n > 1 ? k / (n - 1) : 0.5;
    return t < 0.5 ? mix(early, a, t * 2) : mix(a, late, (t - 0.5) * 2);
  });
}

export interface MapState { rec: FireInput; now: FireInput; keep: boolean[][]; frp: number; accent: string; full: boolean; pixels: boolean;
  /** One pass picked in the key: show only its pixels and the perimeter burned by then (null = all passes). */
  pass: number | null }

/** First step at which each cell is mapped burned (255 = never), from the cumulative masks. */
export function firstStep(f: FireInput): Uint8Array {
  const { n, ny, nx, data } = f.masks, sz = ny * nx, out = new Uint8Array(sz).fill(255);
  for (let k = n - 1; k >= 0; k--) for (let i = 0; i < sz; i++) if (data[k * sz + i] > 0) out[i] = k;
  return out;
}

interface View { c0: number; c1: number; r0: number; r1: number }

/** The whole recorded fire (every pass, every VIIRS footprint) plus 1 km, widened to the canvas shape, in whole cells. */
function fitView(f: FireInput, first: Uint8Array, aspect: number): View {
  const { ny, nx } = f.masks;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < ny * nx; i++) {
    if (first[i] === 255) continue;
    const r = Math.floor(i / nx), c = i % nx;
    x0 = Math.min(x0, c); x1 = Math.max(x1, c + 1); y0 = Math.min(y0, r); y1 = Math.max(y1, r + 1);
  }
  const half = PIXEL_M / RES / 2;
  for (const d of f.dets) d.row.forEach((r, j) => {
    const x = d.col[j] + 0.5, y = r + 0.5;
    x0 = Math.min(x0, x - half); x1 = Math.max(x1, x + half); y0 = Math.min(y0, y - half); y1 = Math.max(y1, y + half);
  });
  if (!Number.isFinite(x0)) return { c0: 0, c1: nx, r0: 0, r1: ny };
  const m = FIT_MARGIN_M / RES;
  x0 -= m; x1 += m; y0 -= m; y1 += m;
  let w = x1 - x0, h = y1 - y0;
  if (w / h < aspect) w = h * aspect; else h = w / aspect;
  const axis = (mid: number, len: number, N: number): [number, number] => {
    let lo = mid - len / 2, hi = mid + len / 2;
    if (lo < 0) { hi -= lo; lo = 0; }
    if (hi > N) { lo -= hi - N; hi = N; }
    return [Math.max(0, Math.floor(lo)), Math.min(N, Math.ceil(hi))];
  };
  const [c0, c1] = axis((x0 + x1) / 2, w, nx), [r0, r1] = axis((y0 + y1) / 2, h, ny);
  return { c0, c1, r0, r1 };
}

// One offscreen layer, reused, for the merged VIIRS footprints (drawing is synchronous, so sharing it is safe).
let layer: HTMLCanvasElement | null = null;
function layerFor(c: Canvas2D) {
  layer ??= document.createElement('canvas');
  if (layer.width !== c.cv.width || layer.height !== c.cv.height) { layer.width = c.cv.width; layer.height = c.cv.height; }
  const o = layer.getContext('2d')!;
  o.setTransform(1, 0, 0, 1, 0, 0);
  o.globalCompositeOperation = 'source-over';
  o.clearRect(0, 0, layer.width, layer.height);
  const dpr = c.cv.width / c.w;
  o.setTransform(dpr, 0, 0, dpr, 0, 0);
  return o;
}
function blit(c: Canvas2D, alpha: number) {
  c.ctx.globalAlpha = alpha;
  c.ctx.drawImage(layer!, 0, 0, c.w, c.h);
  c.ctx.globalAlpha = 1;
}

/** Squares of side s centred on (x, y), merged: a faint fill and one outline around the union (with a dark casing
 *  when `casing`), so hundreds of overlapping pixels read as the area VIIRS saw burning, not as noise. */
function footprints(c: Canvas2D, xy: number[], s: number, col: string, fillA: number, lineA: number, lw: number, casing: boolean) {
  if (!xy.length) return;
  const path = (o: CanvasRenderingContext2D) => {
    o.beginPath();
    for (let i = 0; i < xy.length; i += 2) o.rect(xy[i] - s / 2, xy[i + 1] - s / 2, s, s);
  };
  let o: CanvasRenderingContext2D;
  if (fillA > 0) {
    o = layerFor(c);
    path(o); o.fillStyle = col; o.fill();
    blit(c, fillA);
  }
  o = layerFor(c);
  path(o);
  o.lineJoin = 'miter';
  if (casing) { o.strokeStyle = 'rgba(8,9,7,0.85)'; o.lineWidth = 2 * lw + 2 * Math.max(0.8, lw * 0.9); o.stroke(); }
  o.strokeStyle = col; o.lineWidth = 2 * lw; o.stroke();
  o.globalCompositeOperation = 'destination-out';
  o.fillStyle = '#000'; o.fill();   // keep only what lies outside every square: the outline of the union
  blit(c, lineA);
}

/** A nice scale-bar length (km) that is at most a quarter of the map wide. */
function scaleKm(pxPerKm: number, width: number): number {
  const opts = [0.2, 0.5, 1, 2, 5, 10, 20];
  let best = opts[0];
  for (const v of opts) if (v * pxPerKm <= width * 0.25) best = v;
  return best;
}

export function drawMap(c: Canvas2D, s: MapState) {
  const { ctx } = c;
  const { ny, nx } = s.rec.masks;
  ctx.fillStyle = SCOPE.bg;
  ctx.fillRect(0, 0, c.w, c.h);
  const pad = 12, top = 20, bot = 20;
  const W = c.w - 2 * pad, H = c.h - top - bot;
  const nRec = s.rec.masks.n, nNow = s.now.masks.n;
  const pass = s.pass != null && s.pass < nNow ? s.pass : null;
  const first = firstStep(s.rec);
  const v = s.full ? { c0: 0, c1: nx, r0: 0, r1: ny } : fitView(s.rec, first, W / H);
  const vw = v.c1 - v.c0, vh = v.r1 - v.r0;
  const cell = Math.min(W / vw, H / vh);
  const gx = pad + (W - cell * vw) / 2, gy = top + (H - cell * vh) / 2;
  const X = (col: number) => gx + (col - v.c0) * cell, Y = (row: number) => gy + (row - v.r0) * cell;

  // cells: burn-time colour by the pass that first mapped them; muted fuels elsewhere
  const ramp = burnRamp(s.accent, nRec);
  const img = ctx.createImageData(nx, ny);
  const L = s.now.layers;
  for (let i = 0; i < ny * nx; i++) {
    const k = first[i], q = i * 4;
    let col: RGB;
    if (k < 255) col = k < nNow ? ramp[k] : BG_LATER;   // LATER: burned after the passes the network reads
    else {
      const grp = fuelGroup(L.fuel[i]);
      col = grp === 'TU' || grp === 'TL' || grp === 'SB' ? BG_FOREST : grp === 'NB' ? BG_NONBURN : BG_BURNABLE;
    }
    img.data[q] = col[0]; img.data[q + 1] = col[1]; img.data[q + 2] = col[2]; img.data[q + 3] = 255;
  }
  const off = document.createElement('canvas');
  off.width = nx; off.height = ny;
  off.getContext('2d')!.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, v.c0, v.r0, vw, vh, gx, gy, cell * vw, cell * vh);
  ctx.imageSmoothingEnabled = true;

  ctx.save();
  ctx.beginPath();
  ctx.rect(gx, gy, cell * vw, cell * vh);
  ctx.clip();

  // growth: cell-edge lines between passes (where two burned cells first burned in different passes), the
  // perimeter of the burned area as read, and with a pass picked the perimeter burned by the end of that pass
  const kAt = (r: number, q: number) => (r >= 0 && r < ny && q >= 0 && q < nx ? first[r * nx + q] : 255);
  const iso = new Path2D(), per = new Path2D(), perP = new Path2D();
  const edge = (ka: number, kb: number, x1: number, y1: number, x2: number, y2: number) => {
    const a = ka < nNow, b = kb < nNow;
    if (a !== b) { per.moveTo(x1, y1); per.lineTo(x2, y2); }
    else if (a && ka !== kb) { iso.moveTo(x1, y1); iso.lineTo(x2, y2); }
    if (pass != null && (ka <= pass) !== (kb <= pass)) { perP.moveTo(x1, y1); perP.lineTo(x2, y2); }
  };
  for (let r = v.r0; r < v.r1; r++) for (let q = v.c0; q <= v.c1; q++) edge(kAt(r, q - 1), kAt(r, q), X(q), Y(r), X(q), Y(r + 1));
  for (let r = v.r0; r <= v.r1; r++) for (let q = v.c0; q < v.c1; q++) edge(kAt(r - 1, q), kAt(r, q), X(q), Y(r), X(q + 1), Y(r));
  if (cell >= 3.5) {
    ctx.strokeStyle = 'rgba(13,15,12,0.55)';
    ctx.lineWidth = Math.min(1, cell / 10);
    ctx.stroke(iso);
  }
  ctx.strokeStyle = pass == null ? 'rgba(235,230,216,0.5)' : 'rgba(235,230,216,0.22)';
  ctx.lineWidth = cell >= 3.5 ? 1 : 0.7;
  ctx.stroke(per);

  // VIIRS pixels the network reads, at their nominal 375 m footprint (none drawn past the record's end)
  if (s.pixels) {
    const fp = (PIXEL_M / RES) * cell;
    const lw = Math.max(0.6, Math.min(1.3, fp / 14));
    const dayXY: number[] = [], nightXY: number[] = [], lostXY: number[] = [];
    s.rec.dets.forEach((d, k) => {
      if (k >= nNow || (pass != null && k !== pass)) return;
      d.row.forEach((rr, j) => {
        const to = !s.keep[k][j] ? lostXY : s.rec.day[k] ? dayXY : nightXY;
        to.push(X(d.col[j] + 0.5), Y(rr + 0.5));
      });
    });
    // all passes over a large fire: a calm tint and a thin line; one pass, or big squares: crisp cased outlines
    const crisp = pass != null || fp >= 12;
    footprints(c, lostXY, fp, 'rgba(235,230,216,1)', 0, 0.4, lw, false);
    footprints(c, nightXY, fp, NIGHT, crisp ? 0.14 : 0.13, crisp ? 0.95 : 0.45, crisp ? lw : 0.6, crisp);
    footprints(c, dayXY, fp, DAY, crisp ? 0.12 : 0.1, crisp ? 0.95 : 0.45, crisp ? lw : 0.6, crisp);
  }
  if (pass != null) {
    ctx.strokeStyle = 'rgba(8,9,7,0.8)'; ctx.lineWidth = 3.2; ctx.stroke(perP);
    ctx.strokeStyle = SCOPE.ink; ctx.lineWidth = 1.4; ctx.stroke(perP);
  }
  ctx.restore();
  ctx.strokeStyle = 'rgba(235,230,216,0.15)';
  ctx.lineWidth = 1;
  ctx.strokeRect(gx + 0.5, gy + 0.5, cell * vw - 1, cell * vh - 1);

  const km = (n: number) => ((n * RES) / 1000).toFixed(1);
  const tight = c.w < 480;
  label(c, `${km(vw)} × ${km(vh)} km · ${RES} m cells${tight ? '' : ' · north up'}`, pad, 9, SCOPE.text, 'left', 9);
  const view = s.full ? 'full frame' : 'fire + 1 km';
  if (pass != null) label(c, `burned by pass ${pass + 1}${tight ? '' : ' · ' + view}`, c.w - pad, 9, SCOPE.ink, 'right', 9);
  else label(c, view, c.w - pad, 9, SCOPE.faint, 'right', 9);
  // scale bar
  const pxKm = (cell * 1000) / RES, bar = scaleKm(pxKm, W);
  const yb = c.h - 9;
  ctx.strokeStyle = SCOPE.ink;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(pad, yb - 3); ctx.lineTo(pad, yb); ctx.lineTo(pad + bar * pxKm, yb); ctx.lineTo(pad + bar * pxKm, yb - 3);
  ctx.stroke();
  label(c, bar < 1 ? `${bar * 1000} m` : `${bar} km`, pad + bar * pxKm + 5, yb, SCOPE.text, 'left', 9);
}

export interface StepState { rec: FireInput; now: FireInput; steps: string[]; accent: string; pass?: number | null }

/** New burned area per half-day pass (bars) and log10 total FRP (line), recorded vs as read. */
export function drawSteps(c: Canvas2D, s: StepState) {
  const { ctx } = c;
  ctx.fillStyle = SCOPE.bg;
  ctx.fillRect(0, 0, c.w, c.h);
  const L = 44, R = 54, T = 20, B = 30;
  const W = c.w - L - R, H = c.h - T - B;
  const n = s.rec.masks.n, nNow = s.now.masks.n, sz = s.rec.masks.ny * s.rec.masks.nx;
  const area: number[] = [];
  for (let k = 0; k < n; k++) { let a = 0; for (let i = 0; i < sz; i++) a += s.rec.masks.data[k * sz + i] > 0 ? 1 : 0; area.push(a * 0.0324); }
  const grow = area.map((a, k) => Math.max(a - (k ? area[k - 1] : 0), 0));
  const gmax = Math.max(0.05, ...grow);
  const bw = W / 12;
  const ramp = burnRamp(s.accent, n);
  for (let k = 0; k < 12; k++) {
    const x = L + k * bw;
    if (k >= n) continue;
    if (s.pass === k) { ctx.fillStyle = 'rgba(235,230,216,0.07)'; ctx.fillRect(x, T, bw, H + 6); }
    const hgt = (grow[k] / gmax) * H * 0.9;
    ctx.fillStyle = s.rec.day[k] ? DAY : NIGHT;
    ctx.globalAlpha = k < nNow ? 0.85 : 0.2;
    ctx.fillRect(x + bw * 0.18, T + H - hgt, bw * 0.64, hgt);
    ctx.globalAlpha = 1;
    // the pass's burn-time colour on the map
    ctx.fillStyle = k < nNow ? rgb(ramp[k]) : LATER;
    ctx.fillRect(x + bw * 0.18, T + H + 2, bw * 0.64, 3);
    label(c, s.rec.day[k] ? 'D' : 'N', x + bw / 2, T + H + 12, k < nNow ? SCOPE.text : SCOPE.faint, 'center', 9);
  }
  label(c, `${gmax.toFixed(gmax < 1 ? 2 : 1)} km²`, L - 4, T + H * 0.1, SCOPE.text, 'right', 9);
  label(c, 'new area', L - 4, T + H * 0.1 + 12, SCOPE.faint, 'right', 9);
  label(c, 'half-day passes (D = 13:30 day, N = 01:30 night)', L, c.h - 6, SCOPE.faint, 'left', 9);
  // FRP line
  const lf = (d: { frp: number[] }) => Math.log10(1 + d.frp.reduce((a, b) => a + b, 0));
  const fmax = Math.max(1, ...s.rec.dets.map(lf), ...s.now.dets.map(lf));
  const yF = (v: number) => T + H - (v / fmax) * H * 0.9;
  const line = (dets: { frp: number[] }[], col: string, dash: number[]) => {
    ctx.strokeStyle = col; ctx.lineWidth = 1.6; ctx.setLineDash(dash);
    ctx.beginPath();
    dets.forEach((d, k) => { const x = L + (k + 0.5) * bw, y = yF(lf(d)); if (k) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
    ctx.stroke(); ctx.setLineDash([]);
  };
  line(s.rec.dets.slice(0, n), 'rgba(235,230,216,0.35)', [3, 3]);
  line(s.now.dets, s.accent, []);
  label(c, `10^${fmax.toFixed(1)} MW`, c.w - R + 4, yF(fmax), s.accent, 'left', 9);
  label(c, 'FRP', c.w - R + 4, yF(fmax) + 12, SCOPE.faint, 'left', 9);
  if (nNow < n) {
    const x = L + nNow * bw;
    ctx.strokeStyle = 'rgba(235,230,216,0.5)'; ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(x + 0.5, T); ctx.lineTo(x + 0.5, T + H); ctx.stroke(); ctx.setLineDash([]);
    label(c, 'record ends', x + 4, T + 6, SCOPE.text, 'left', 9);
  }
  label(c, s.steps.length ? `from ${s.steps[0].slice(0, 16).replace('T', ' ')} UTC` : '', L, 9, SCOPE.text, 'left', 9);
}
