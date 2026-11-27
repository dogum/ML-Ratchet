// Pure TypeScript gradient-boosted tree runtime & exact path-dependent TreeSHAP.
// Matches mlratchet/gbm.py (flat-array layout, Lundberg et al. 2020 Algorithm 2) with zero
// external dependencies so any tree ensemble can run and explain itself live in a browser tab.

export interface TreePack {
  tree_offsets: Int32Array;
  feat: Int32Array;
  thresh: Float32Array | Float64Array;
  default_left: Uint8Array;
  left: Int32Array;
  right: Int32Array;
  value: Float32Array | Float64Array;
  cover: Float32Array | Float64Array;
  base_score: Float32Array | Float64Array;
  tree_output: Int32Array;
  n_features: number;
  n_outputs: number;
  objective?: 'regression' | 'binary' | 'multiclass';
}

/** Evaluate raw margins for a single sample ``x`` (length ``n_features``). */
export function evalTreesRaw(pack: TreePack, x: ArrayLike<number>): Float64Array {
  const nTrees = pack.tree_offsets.length - 1;
  const out = new Float64Array(pack.n_outputs);
  for (let k = 0; k < pack.n_outputs; k++) out[k] = pack.base_score[k];

  const { tree_offsets, feat, thresh, default_left, left, right, value, tree_output } = pack;
  for (let t = 0; t < nTrees; t++) {
    const s = tree_offsets[t];
    let u = 0;
    while (true) {
      const f = feat[s + u];
      if (f < 0) {
        out[tree_output[t]] += value[s + u];
        break;
      }
      const xv = x[f];
      const goLeft = Number.isNaN(xv) ? default_left[s + u] !== 0 : xv <= thresh[s + u];
      u = goLeft ? left[s + u] : right[s + u];
    }
  }
  return out;
}

/** Evaluate predictions in target space (identity / sigmoid / softmax). */
export function evalTrees(pack: TreePack, x: ArrayLike<number>): Float64Array {
  const raw = evalTreesRaw(pack, x);
  if (pack.objective === 'binary') {
    for (let k = 0; k < raw.length; k++) {
      const z = Math.max(-40, Math.min(40, raw[k]));
      raw[k] = 1 / (1 + Math.exp(-z));
    }
    return raw;
  }
  if (pack.objective === 'multiclass') {
    let m = -Infinity;
    for (let k = 0; k < raw.length; k++) if (raw[k] > m) m = raw[k];
    let s = 0;
    for (let k = 0; k < raw.length; k++) { raw[k] = Math.exp(raw[k] - m); s += raw[k]; }
    for (let k = 0; k < raw.length; k++) raw[k] /= s;
    return raw;
  }
  return raw;
}

function treeExpectedValue(
  s: number,
  feat: Int32Array,
  left: Int32Array,
  right: Int32Array,
  value: Float32Array | Float64Array,
  cover: Float32Array | Float64Array,
): number {
  const rec = (u: number): number => {
    if (feat[s + u] < 0) return value[s + u];
    const lc = left[s + u];
    const rc = right[s + u];
    const cl = cover[s + lc];
    const cr = cover[s + rc];
    const ct = cl + cr;
    if (ct <= 0) return 0.5 * (rec(lc) + rec(rc));
    return (cl * rec(lc) + cr * rec(rc)) / ct;
  };
  return rec(0);
}

interface Path {
  d: number[];
  z: number[];
  o: number[];
  w: number[];
}

function extendPath(m: Path, pz: number, po: number, pi: number): Path {
  const L = m.d.length;
  const d = m.d.concat(pi);
  const z = m.z.concat(pz);
  const o = m.o.concat(po);
  const w = m.w.concat(L === 0 ? 1 : 0);
  for (let i = L - 1; i >= 0; i--) {
    w[i + 1] += (po * w[i] * (i + 1)) / (L + 1);
    w[i] = (pz * w[i] * (L - i)) / (L + 1);
  }
  return { d, z, o, w };
}

function unwindPath(m: Path, idx: number): Path {
  const L = m.d.length - 1;
  const d = m.d.slice(0, idx).concat(m.d.slice(idx + 1));
  const z = m.z.slice(0, idx).concat(m.z.slice(idx + 1));
  const o = m.o.slice(0, idx).concat(m.o.slice(idx + 1));
  const w = new Array<number>(L).fill(0);
  let n = m.w[L];
  const oi = m.o[idx];
  const zi = m.z[idx];
  if (oi !== 0) {
    for (let j = L - 1; j >= 0; j--) {
      const t = m.w[j];
      w[j] = (n * (L + 1)) / ((j + 1) * oi);
      n = t - (w[j] * zi * (L - j)) / (L + 1);
    }
  } else {
    for (let j = L - 1; j >= 0; j--) {
      w[j] = (m.w[j] * (L + 1)) / (zi * (L - j));
    }
  }
  return { d, z, o, w };
}

function unwoundPathSum(m: Path, idx: number): number {
  const L = m.w.length - 1;
  let total = 0;
  const oi = m.o[idx];
  const zi = m.z[idx];
  if (oi !== 0) {
    let n = m.w[L];
    for (let j = L - 1; j >= 0; j--) {
      const tmp = n / ((j + 1) * oi);
      total += tmp;
      n = m.w[j] - tmp * zi * (L - j);
    }
    total *= L + 1;
  } else {
    for (let j = L - 1; j >= 0; j--) {
      total += (m.w[j] * (L + 1)) / (zi * (L - j));
    }
  }
  return total;
}

/** Exact path-dependent TreeSHAP for one sample ``x`` (Lundberg et al. 2020, Algorithm 2). */
export function treeShap(
  pack: TreePack,
  x: ArrayLike<number>,
  outputIdx = 0,
): { phi: Float64Array; expectedValue: number } {
  const nTrees = pack.tree_offsets.length - 1;
  const phi = new Float64Array(pack.n_features);
  let ev = pack.base_score[outputIdx];
  const { tree_offsets, feat, thresh, default_left, left, right, value, cover, tree_output } = pack;

  for (let t = 0; t < nTrees; t++) {
    if (tree_output[t] !== outputIdx) continue;
    const s = tree_offsets[t];
    ev += treeExpectedValue(s, feat, left, right, value, cover);

    const recurse = (u: number, m: Path, pz: number, po: number, pi: number): void => {
      let cur = extendPath(m, pz, po, pi);
      const f = feat[s + u];
      if (f < 0) {
        const v = value[s + u];
        for (let i = 1; i < cur.d.length; i++) {
          const wsum = unwoundPathSum(cur, i);
          phi[cur.d[i]] += wsum * (cur.o[i] - cur.z[i]) * v;
        }
        return;
      }

      const xv = x[f];
      const goLeft = Number.isNaN(xv) ? default_left[s + u] !== 0 : xv <= thresh[s + u];
      const hot = goLeft ? left[s + u] : right[s + u];
      const cold = goLeft ? right[s + u] : left[s + u];

      let iz = 1;
      let io = 1;
      let k = -1;
      for (let i = 1; i < cur.d.length; i++) {
        if (cur.d[i] === f) { k = i; break; }
      }
      if (k >= 0) {
        iz = cur.z[k];
        io = cur.o[k];
        cur = unwindPath(cur, k);
      }

      const ch = cover[s + hot];
      const cc = cover[s + cold];
      const cu = ch + cc;
      const fHot = cu > 0 ? ch / cu : 0.5;
      const fCold = cu > 0 ? cc / cu : 0.5;

      recurse(hot, cur, iz * fHot, io, f);
      recurse(cold, cur, iz * fCold, 0, f);
    };

    recurse(0, { d: [], z: [], o: [], w: [] }, 1, 1, -1);
  }

  return { phi, expectedValue: ev };
}
