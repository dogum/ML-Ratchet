// FireDetective's two scopes: the fire seen from above (when each 180 m cell was first mapped as burned, the fuels
// around it, the VIIRS pixels the network reads), and the record step by step (new burned area per half-day pass,
// day or night, and the total fire radiative power).
import { SCOPE, hex, label, type Canvas2D } from '../kit.ts';
import { fuelGroup, type FireInput } from './features.ts';

export const DAY = '#e0a84c', NIGHT = '#7f95c9', FOREST = 'rgba(95,160,110,0.20)', NONBURN = 'rgba(235,230,216,0.06)';

export interface MapState { rec: FireInput; now: FireInput; keep: boolean[][]; frp: number; accent: string }

/** First step at which each cell is mapped burned (255 = never), from the cumulative masks. */
export function firstStep(f: FireInput): Uint8Array {
  const { n, ny, nx, data } = f.masks, sz = ny * nx, out = new Uint8Array(sz).fill(255);
  for (let k = n - 1; k >= 0; k--) for (let i = 0; i < sz; i++) if (data[k * sz + i] > 0) out[i] = k;
  return out;
}

export function drawMap(c: Canvas2D, s: MapState) {
  const { ctx } = c;
  const { ny, nx } = s.rec.masks;
  ctx.fillStyle = SCOPE.bg;
  ctx.fillRect(0, 0, c.w, c.h);
  const pad = 14, top = 20;
  const cell = Math.min((c.w - 2 * pad) / nx, (c.h - top - pad - 14) / ny);
  const x0 = (c.w - cell * nx) / 2, y0 = top + (c.h - top - pad - 14 - cell * ny) / 2;
  const nRec = s.rec.masks.n, nNow = s.now.masks.n;
  const first = firstStep(s.rec);
  const [ar, ag, ab] = hex(s.accent);
  const img = ctx.createImageData(nx, ny);
  const L = s.now.layers;
  for (let i = 0; i < ny * nx; i++) {
    const k = first[i], q = i * 4;
    let r = 13, g = 15, b = 12, a = 255;
    const grp = fuelGroup(L.fuel[i]);
    if (grp === 'TU' || grp === 'TL' || grp === 'SB') { r = 26; g = 40; b = 30; }
    else if (grp === 'NB') { r = 22; g = 23; b = 22; }
    if (k < 255) {
      const t = nRec > 1 ? k / (nRec - 1) : 1;
      const shade = 0.35 + 0.65 * (1 - t);
      if (k < nNow) { r = ar * shade + 20 * (1 - shade); g = ag * shade + 15 * (1 - shade); b = ab * shade + 12 * (1 - shade); }
      else { r = 70; g = 66; b = 60; a = 255; }   // burned later than the record the network reads
    }
    img.data[q] = r; img.data[q + 1] = g; img.data[q + 2] = b; img.data[q + 3] = a;
  }
  const off = document.createElement('canvas');
  off.width = nx; off.height = ny;
  off.getContext('2d')!.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, x0, y0, cell * nx, cell * ny);
  ctx.strokeStyle = 'rgba(235,230,216,0.15)';
  ctx.strokeRect(x0 + 0.5, y0 + 0.5, cell * nx - 1, cell * ny - 1);
  // VIIRS detections the network reads (open rings: lost to the what-if; none drawn past the record's end)
  s.rec.dets.forEach((d, k) => {
    if (k >= nNow) return;
    d.row.forEach((rr, j) => {
      const on = s.keep[k][j];
      const rad = Math.max(1.2, Math.min(6, 0.9 + 1.1 * Math.log10(1 + d.frp[j] * s.frp))) * Math.max(0.6, Math.min(1.4, cell / 5));
      const x = x0 + (d.col[j] + 0.5) * cell, y = y0 + (rr + 0.5) * cell;
      ctx.beginPath();
      ctx.arc(x, y, rad, 0, 2 * Math.PI);
      if (on) { ctx.fillStyle = s.rec.day[k] ? 'rgba(224,168,76,0.75)' : 'rgba(127,149,201,0.8)'; ctx.fill(); }
      else { ctx.strokeStyle = 'rgba(235,230,216,0.35)'; ctx.lineWidth = 0.8; ctx.stroke(); }
    });
  });
  label(c, `${(nx * 0.18).toFixed(1)} × ${(ny * 0.18).toFixed(1)} km · 180 m cells · north up`, pad, 9, SCOPE.text, 'left', 9);
  label(c, 'bright = burned early · dull = late', c.w - pad, c.h - 8, SCOPE.faint, 'right', 9);
  // 1 km scale bar
  const km = cell / 0.18;
  ctx.strokeStyle = SCOPE.ink;
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(pad, c.h - 8); ctx.lineTo(pad + km, c.h - 8); ctx.stroke();
  label(c, '1 km', pad + km + 4, c.h - 8, SCOPE.text, 'left', 9);
}

export interface StepState { rec: FireInput; now: FireInput; steps: string[]; accent: string }

/** New burned area per half-day pass (bars) and log10 total FRP (line), recorded vs as read. */
export function drawSteps(c: Canvas2D, s: StepState) {
  const { ctx } = c;
  ctx.fillStyle = SCOPE.bg;
  ctx.fillRect(0, 0, c.w, c.h);
  const L = 44, R = 40, T = 20, B = 26;
  const W = c.w - L - R, H = c.h - T - B;
  const n = s.rec.masks.n, nNow = s.now.masks.n, sz = s.rec.masks.ny * s.rec.masks.nx;
  const area: number[] = [];
  for (let k = 0; k < n; k++) { let a = 0; for (let i = 0; i < sz; i++) a += s.rec.masks.data[k * sz + i] > 0 ? 1 : 0; area.push(a * 0.0324); }
  const grow = area.map((a, k) => Math.max(a - (k ? area[k - 1] : 0), 0));
  const gmax = Math.max(0.05, ...grow);
  const bw = W / 12;
  for (let k = 0; k < 12; k++) {
    const x = L + k * bw;
    if (k >= n) continue;
    const hgt = (grow[k] / gmax) * H * 0.9;
    ctx.fillStyle = s.rec.day[k] ? DAY : NIGHT;
    ctx.globalAlpha = k < nNow ? 0.85 : 0.2;
    ctx.fillRect(x + bw * 0.18, T + H - hgt, bw * 0.64, hgt);
    ctx.globalAlpha = 1;
    label(c, s.rec.day[k] ? 'D' : 'N', x + bw / 2, T + H + 9, k < nNow ? SCOPE.text : SCOPE.faint, 'center', 9);
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
