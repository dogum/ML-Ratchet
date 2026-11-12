// CellSense network in the browser: 1-D residual CNN with BatchNorm fused, same arithmetic as cellsense.py
const CellNet = (() => {
  function load(pack) {
    const s = atob(pack.b64), buf = new ArrayBuffer(s.length), u = new Uint8Array(buf);
    for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
    const all = new Float32Array(buf), T = {};
    for (const t of pack.man.tensors) { const n = t.shape.reduce((a, b) => a * b, 1); T[t.k] = { d: all.subarray(t.off, t.off + n), s: t.shape }; }
    return { T, meta: pack.man.meta };
  }
  const gelu = (x) => { const a = Math.abs(x) * 0.7071067811865476, t = 1 / (1 + 0.3275911 * a);
    const erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a); return 0.5 * x * (1 + (x >= 0 ? erf : -erf)); };
  function conv(x, C, L, w, b, stride, pad) {   // x: Float32Array C*L -> out O*Lo
    const [O, Ci, K] = w.s, Lo = Math.floor((L + 2 * pad - K) / stride) + 1, out = new Float32Array(O * Lo), W = w.d, Bv = b.d;
    for (let o = 0; o < O; o++) { const ob = o * Lo; for (let l = 0; l < Lo; l++) out[ob + l] = Bv[o];
      for (let c = 0; c < Ci; c++) { const xb = c * L, wb = (o * Ci + c) * K;
        for (let k = 0; k < K; k++) { const wv = W[wb + k]; if (wv === 0) continue; const off = k - pad;
          for (let l = 0; l < Lo; l++) { const p = l * stride + off; if (p >= 0 && p < L) out[ob + l] += wv * x[xb + p]; } } } }
    return [out, O, Lo];
  }
  function forward(net, x, L, s) {   // x: 8*L input channels; s: [log10(cycle)/3.5, charge_min/15] or null
    const T = net.T, M = net.meta; let [h, C, Lh] = conv(x, 8, L, T.stem_w, T.stem_b, 1, 3); for (let i = 0; i < h.length; i++) h[i] = gelu(h[i]);
    M.blocks.forEach(([st, p1, p2], i) => {
      let [y, Cy, Ly] = conv(h, C, Lh, T[`b${i}c1_w`], T[`b${i}c1_b`], st, p1); for (let j = 0; j < y.length; j++) y[j] = gelu(y[j]);
      [y, Cy, Ly] = conv(y, Cy, Ly, T[`b${i}c2_w`], T[`b${i}c2_b`], 1, p2); const [sk] = conv(h, C, Lh, T[`b${i}sk_w`], T[`b${i}sk_b`], st, 0);
      for (let j = 0; j < y.length; j++) y[j] = gelu(y[j] + sk[j]); h = y; C = Cy; Lh = Ly; });
    const z = []; for (let c = 0; c < C; c++) { let m = 0; for (let l = 0; l < Lh; l++) m += h[c * Lh + l]; z.push(m / Lh); }
    for (let c = 0; c < C; c++) { let m = -Infinity; for (let l = 0; l < Lh; l++) m = Math.max(m, h[c * Lh + l]); z.push(m); }
    if (M.extra) z.push(...s);
    const dense = (v, w, b, act) => { const [O, I] = w.s, o = new Float32Array(O); for (let i = 0; i < O; i++) { let a = b.d[i]; for (let j = 0; j < I; j++) a += w.d[i * I + j] * v[j]; o[i] = act ? gelu(a) : a; } return o; };
    const o = dense(dense(z, T.h0_w, T.h0_b, true), T.h1_w, T.h1_b, false), n = M.labels.length, res = {};
    M.labels.forEach((k, j) => { res[k] = o[j] * M.sd_y[j] + M.mu_y[j]; res[k + '_sig'] = Math.exp(Math.max(-7, Math.min(3, o[n + j]))) * M.sd_y[j]; });
    return res;
  }
  return { load, forward };
})();
if (typeof module !== 'undefined') module.exports = CellNet;
