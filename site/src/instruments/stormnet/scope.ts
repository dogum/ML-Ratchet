import { SCOPE, type Canvas2D } from '../kit.ts';
import type { Frame, Reading } from './net.ts';

export function drawScope(c: Canvas2D, frame: Frame, r: Reading, accent: string, recorded: boolean) {
  const { ctx, w, h } = c;
  const left = 37, right = w - 12, top = 35, bottom = h * .57;
  const fieldTop = h * .72, fieldBottom = h - 24;
  const x = (hour: number) => left + (hour + 12) / 15 * (right - left);
  const past = frame.past.slice(-13);
  const values = [...past.filter(Number.isFinite), r.minimum - r.sigma, r.minimum + r.sigma,
    ...(recorded ? frame.future : [])];
  const span = Math.max(...values) - Math.min(...values);
  const tick = span > 90 ? 40 : span > 45 ? 20 : span > 20 ? 10 : 5;
  const min = Math.floor((Math.min(...values) - 1) / tick) * tick;
  const max = Math.ceil((Math.max(...values) + 1) / tick) * tick;
  const y = (v: number) => bottom - (v - min) / (max - min) * (bottom - top);
  const txt = (s: string, xx: number, yy: number, color = SCOPE.text, align: CanvasTextAlign = 'left') => {
    ctx.fillStyle = color; ctx.textAlign = align; ctx.fillText(s, xx, yy); ctx.textAlign = 'left';
  };
  const line = (ax: number, ay: number, bx: number, by: number, color: string, width = 1) => {
    ctx.strokeStyle = color; ctx.lineWidth = width;
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
  };
  ctx.fillStyle = SCOPE.bg; ctx.fillRect(0, 0, w, h);
  ctx.font = '10px ui-monospace, monospace';
  txt('Dst · nT', left, 17, SCOPE.ink);
  txt('next 3 h', right, 17, accent, 'right');
  ctx.fillStyle = 'rgba(115,224,182,.045)'; ctx.fillRect(x(0), top, right - x(0), bottom - top);
  for (let v = min; v <= max; v += tick) {
    line(left, y(v), right, y(v), SCOPE.grid);
    txt(String(v), left - 5, y(v) + 3, SCOPE.text, 'right');
  }
  ctx.globalAlpha = .18; ctx.fillStyle = accent;
  ctx.fillRect(x(0), y(r.minimum + r.sigma), right - x(0), y(r.minimum - r.sigma) - y(r.minimum + r.sigma));
  ctx.globalAlpha = 1;
  ctx.setLineDash([4, 3]); line(x(0), y(r.minimum), right, y(r.minimum), accent, 1.8); ctx.setLineDash([]);
  for (let j = 1; j < past.length; j++) {
    if (Number.isFinite(past[j - 1]) && Number.isFinite(past[j])) {
      line(x(j - 13), y(past[j - 1]), x(j - 12), y(past[j]), SCOPE.ink, 1.8);
    }
  }
  if (recorded) {
    const path = [frame.dst, ...frame.future];
    for (let j = 1; j < path.length; j++) line(x(j - 1), y(path[j - 1]), x(j), y(path[j]), SCOPE.ink, 1.5);
    const lowest = Math.min(...frame.future);
    ctx.fillStyle = SCOPE.ink; ctx.beginPath(); ctx.arc(x(1 + frame.future.indexOf(lowest)), y(lowest), 3, 0, Math.PI * 2); ctx.fill();
  }
  ctx.setLineDash([2, 4]); line(x(0), top, x(0), fieldBottom, SCOPE.faint); ctx.setLineDash([]);
  const bz = frame.wind.slice(-13).map(row => row[0]);
  const extent = Math.max(5, Math.ceil(Math.max(...bz.map(Math.abs)) / 5) * 5);
  const by = (v: number) => fieldTop + (extent - v) / (2 * extent) * (fieldBottom - fieldTop);
  txt('Solar wind Bz · nT', left, fieldTop - 8, SCOPE.ink);
  line(left, by(0), x(0), by(0), SCOPE.faint);
  txt('0', left - 5, by(0) + 3, SCOPE.text, 'right');
  txt('−' + extent, left - 5, fieldBottom + 1, SCOPE.text, 'right');
  const bw = (right - left) / 15;
  for (let j = 1; j < bz.length; j++) {
    const v = bz[j]; ctx.fillStyle = v < 0 ? accent : SCOPE.faint;
    ctx.fillRect(x(j - 13) + 1, Math.min(by(0), by(v)), Math.max(1, bw - 2), Math.max(1, Math.abs(by(v) - by(0))));
  }
  for (const [t, text] of [[-12,'−12 h'],[-6,'−6 h'],[0,'issue'],[3,'+3 h']] as const) {
    txt(text, x(t), h - 6, t === 0 ? SCOPE.ink : SCOPE.text, t === -12 ? 'left' : t === 3 ? 'right' : 'center');
  }
}
