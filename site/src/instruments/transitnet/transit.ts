// For drawing only: the quadratic-limb-darkened transit of models/transitnet/transitsim.py (as the lab page draws it),
// so the instrument can lay the network's planet (k, b, a/R*) and the published one over the folded data.

const GL = (() => { // 48-point Gauss-Legendre nodes and weights on [-1, 1]
  const n = 48, x = new Float64Array(n), w = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let z = Math.cos((Math.PI * (i + 0.75)) / (n + 0.5)), pp = 0;
    for (let it = 0; it < 100; it++) {
      let p1 = 1, p2 = 0;
      for (let j = 1; j <= n; j++) { const p3 = p2; p2 = p1; p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j; }
      pp = (n * (z * p1 - p2)) / (z * z - 1);
      const z1 = z; z = z1 - p1 / pp;
      if (Math.abs(z - z1) < 1e-15) break;
    }
    x[i] = z; w[i] = 2 / ((1 - z * z) * pp * pp);
  }
  return { x, w };
})();

/** Fraction of a quadratic-limb-darkened star blocked by a disk of radius p at separation z (stellar radii). */
export function occult(z: number, p: number, u1: number, u2: number): number {
  z = Math.abs(z);
  if (z >= 1 + p) return 0;
  const lo = Math.max(0, z - p), hi = Math.min(1, z + p);
  let blocked = 0;
  for (let i = 0; i < GL.x.length; i++) {
    const phi = ((GL.x[i] + 1) * Math.PI) / 2, wt = (GL.w[i] * Math.PI) / 2;
    const r = lo + ((hi - lo) * (1 - Math.cos(phi))) / 2, dr = ((hi - lo) * Math.sin(phi)) / 2;
    let th: number;
    if (z < 1e-9) th = r < p ? Math.PI : 0;
    else if (r <= p - z) th = Math.PI;
    else th = Math.acos(Math.max(-1, Math.min(1, (r * r + z * z - p * p) / (2 * Math.max(r, 1e-12) * z))));
    const mu = Math.sqrt(Math.max(0, 1 - r * r)), I = 1 - u1 * (1 - mu) - u2 * (1 - mu) * (1 - mu);
    blocked += I * 2 * th * r * dr * wt;
  }
  return blocked / (Math.PI * (1 - u1 / 3 - u2 / 6));
}

/** Relative flux - 1 at time dt (days) from mid-transit, circular orbit. */
export function dip(dt: number, P: number, k: number, b: number, aR: number, ld: [number, number]): number {
  const inc = Math.acos(Math.min(1, b / aR)), M = (2 * Math.PI * dt) / P;
  const X = aR * Math.sin(M), Y = -aR * Math.cos(M) * Math.cos(inc);
  return Math.cos(M) > 0 ? -occult(Math.hypot(X, Y), k, ld[0], ld[1]) : 0;
}

export const LD: [number, number] = [0.38, 0.21];

/** The limb darkening (Kipping q1, q2 = 0.33) whose central depth matches `depth` — the lab page's choice for drawing. */
export function ldFor(P: number, k: number, b: number, aR: number, depth: number): [number, number] {
  if (!Number.isFinite(depth) || depth <= 0) return LD;
  let best = LD, be = Infinity;
  for (let q1 = 0.02; q1 <= 0.9; q1 += 0.02) {
    const u: [number, number] = [2 * Math.sqrt(q1) * 0.33, Math.sqrt(q1) * 0.34];
    const e = Math.abs(-dip(0, P, k, b, aR, u) - depth);
    if (e < be) { be = e; best = u; }
  }
  return best;
}
