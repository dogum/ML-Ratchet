// Geometry for drawing a ratchet ladder: the incumbent's score as a staircase that only steps up,
// rejected rungs as ghosts hanging off it, the released rung as a diamond, and the margin each new
// rung must clear as a band above the incumbent.
import type { Rung } from './data';

export interface Box { w: number; h: number; top: number; right: number; bottom: number; left: number }

export interface RungMark {
  rung: Rung;
  i: number;
  x: number;
  y: number | null; // null when the rung has no frozen-set score
  incY: number; // incumbent level before this rung was judged
  kind: 'first' | 'kept' | 'rejected' | 'released';
}

export interface LadderGeom {
  stairs: string; // SVG path of the incumbent staircase
  marks: RungMark[];
  bands: { x0: number; x1: number; y0: number; y1: number }[]; // margin bands
  y: (v: number) => number;
  ticks: number[];
  colW: number;
  min: number;
  max: number;
}

export function ladderGeometry(rungs: Rung[], box: Box, margin = 0.01, opts: { bands?: boolean } = {}): LadderGeom {
  const scored = rungs.filter((r) => typeof r.sim === 'number').map((r) => r.sim as number);
  let min = scored.length ? Math.min(...scored) : 0;
  let max = scored.length ? Math.max(...scored) : 1;
  if (max - min < 1e-9) { min -= 0.05; max += 0.05; }
  const span = max - min;
  min -= span * 0.12;
  max += span * 0.12;
  const iw = box.w - box.left - box.right;
  const ih = box.h - box.top - box.bottom;
  const n = Math.max(rungs.length, 1);
  const colW = iw / n;
  const y = (v: number) => box.top + ih * (1 - (v - min) / (max - min));
  const xAt = (i: number) => box.left + colW * (i + 0.5);

  const marks: RungMark[] = [];
  const bands: LadderGeom['bands'] = [];
  let inc: number | null = null;
  let d = '';
  rungs.forEach((r, i) => {
    const x = xAt(i);
    const v = typeof r.sim === 'number' ? r.sim : null;
    const kept = r.decision === 'kept' || r.decision === 'released';
    const incY = inc === null ? (v === null ? y(min) : y(v)) : y(inc);
    let kind: RungMark['kind'] = r.decision === 'released' ? 'released' : kept ? 'kept' : 'rejected';
    if (inc === null && kept) {
      kind = r.decision === 'released' ? 'released' : 'first';
      if (v !== null) {
        inc = v;
        d = `M${(x - colW / 2).toFixed(1)} ${y(v).toFixed(1)}H${x.toFixed(1)}`;
      }
    } else if (kept && v !== null && inc !== null) {
      d += `H${x.toFixed(1)}V${y(v).toFixed(1)}`;
      inc = v;
    }
    if (opts.bands && inc !== null && i < rungs.length - 1) {
      bands.push({ x0: x, x1: x + colW, y0: y(inc + margin), y1: y(inc) });
    }
    marks.push({ rung: r, i, x, y: v === null ? null : y(v), incY, kind });
  });
  if (d) d += `H${(box.left + iw).toFixed(1)}`;

  // ~4 readable ticks
  const raw = (max - min) / 4;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((s) => s * mag).find((s) => s >= raw) ?? raw;
  const ticks: number[] = [];
  for (let t = Math.ceil(min / step) * step; t <= max; t += step) ticks.push(+t.toFixed(6));

  return { stairs: d, marks, bands, y, ticks, colW, min, max };
}

/** The ladder to feature for a model: the one with the most rungs that has scores. */
export function mainLadder<T extends { rungs: Rung[] }>(ladders: T[]): T | undefined {
  const withRelease = ladders.find((l) => l.rungs.some((r) => r.decision === 'released') && l.rungs.some((r) => r.sim !== undefined));
  if (withRelease) return withRelease;
  return [...ladders].sort((a, b) => b.rungs.filter((r) => r.sim !== undefined).length - a.rungs.filter((r) => r.sim !== undefined).length)[0];
}
