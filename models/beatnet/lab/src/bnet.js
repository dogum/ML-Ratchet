// BeatNet in the browser: the same arithmetic as release/beatnet.py + pipeline.py (BatchNorm folded into each convolution)
const BNet = (() => {
  function half(h) { const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    if (e === 0) return s * Math.pow(2, -14) * (f / 1024); if (e === 31) return f ? NaN : s * Infinity; return s * Math.pow(2, e - 15) * (1 + f / 1024); }
  function load(pack) {
    const bin = atob(pack.b64), n = bin.length >> 1, all = new Float32Array(n);
    if (pack.f32) { const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); const f = new Float32Array(u.buffer); return mk(pack, f); }
    const tab = new Float32Array(65536); for (let i = 0; i < 65536; i++) tab[i] = half(i);
    for (let i = 0; i < n; i++) all[i] = tab[bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8)];
    return mk(pack, all);
  }
  function mk(pack, all) { const T = {}; for (const t of pack.man) { const sz = t.shape.reduce((a, b) => a * b, 1); T[t.k] = { d: all.subarray(t.off, t.off + sz), s: t.shape }; } return { T, meta: pack.meta }; }
  const gelu = (x) => { const a = Math.abs(x) * 0.7071067811865476, t = 1 / (1 + 0.3275911 * a);
    const erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a); return 0.5 * x * (1 + (x >= 0 ? erf : -erf)); };
  function conv(x, Ci, L, w, b, stride) {   // x (Ci, L) flat; kernel K, padding K>>1
    const [O, , K] = w.s, p = K >> 1, Lo = Math.floor((L + 2 * p - K) / stride) + 1, out = new Float32Array(O * Lo), W = w.d;
    for (let o = 0; o < O; o++) { const ob = o * Lo; out.fill(b.d[o], ob, ob + Lo);
      for (let c = 0; c < Ci; c++) { const xb = c * L, wb = (o * Ci + c) * K;
        for (let k = 0; k < K; k++) { const wv = W[wb + k], off = k - p; for (let l = 0; l < Lo; l++) { const j = l * stride + off; if (j >= 0 && j < L) out[ob + l] += wv * x[xb + j]; } } } }
    return [out, O, Lo];
  }
  function forward(net, x, r) {   // x: Float32Array(cin*450), r: 6 rhythm features -> 6 raw outputs
    const T = net.T, M = net.meta; let h = x, C = M.cin, L = x.length / M.cin;
    for (let i = 0; i < M.n_blocks; i++) { const s = M.strides[i];
      let [y, O, Lo] = conv(h, C, L, T[`b${i}_c1_w`], T[`b${i}_c1_b`], s); for (let j = 0; j < y.length; j++) y[j] = gelu(y[j]);
      [y] = conv(y, O, Lo, T[`b${i}_c2_w`], T[`b${i}_c2_b`], 1); const [sk] = conv(h, C, L, T[`b${i}_sk_w`], T[`b${i}_sk_b`], s);
      for (let j = 0; j < y.length; j++) y[j] = gelu(y[j] + sk[j]); h = y; C = O; L = Lo; }
    const m = L >> 1, k = M.ctr >> 1, nc = M.ctr; let z = new Float32Array(C * nc + 2 * C + r.length), q = 0;
    for (let c = 0; c < C; c++) for (let l = m - k; l <= m + k; l++) z[q++] = h[c * L + l];
    for (let c = 0; c < C; c++) { let s = 0, mx = -Infinity; for (let l = 0; l < L; l++) { const v = h[c * L + l]; s += v; if (v > mx) mx = v; } z[C * nc + c] = s / L; z[C * nc + C + c] = mx; }
    for (let j = 0; j < r.length; j++) z[C * nc + 2 * C + j] = r[j];
    for (let j = 0; j < M.n_head; j++) { const w = T[`h${j}_w`], b = T[`h${j}_b`], [O, I] = w.s, o = new Float32Array(O);
      for (let a = 0; a < O; a++) { let s = b.d[a]; const wb = a * I; for (let c = 0; c < I; c++) s += w.d[wb + c] * z[c]; o[a] = j < M.n_head - 1 ? gelu(s) : s; } z = o; }
    return z;
  }
  // ---- front end (pipeline.py): 0.5-40 Hz zero-phase band-pass at 360 Hz, decimate to 180 Hz
  const SOS = [[0.07876235, 0.15752471, 0.07876235, 1, -1.06709243, 0.38423425], [1, -2, 1, 1, -1.98766422, 0.98774172]];
  const ZI = [[0.91463995, -0.30293683], [-0.9934023, 0.9934023]], PAD = 15;
  function sosfiltAll(x) {   // cascade with per-section initial state scaled by the section's own input start (scipy: zi * x_0 for all sections)
    const y = Float64Array.from(x), x0 = x[0];
    for (let s = 0; s < 2; s++) { const [b0, b1, b2, , a1, a2] = SOS[s]; let z0 = ZI[s][0] * x0, z1 = ZI[s][1] * x0;
      for (let i = 0; i < y.length; i++) { const xi = y[i], yi = b0 * xi + z0; z0 = b1 * xi - a1 * yi + z1; z1 = b2 * xi - a2 * yi; y[i] = yi; } }
    return y; }
  function filtfilt(x) { const n = x.length, e = new Float64Array(n + 2 * PAD);
    for (let i = 0; i < PAD; i++) { e[i] = 2 * x[0] - x[PAD - i]; e[n + PAD + i] = 2 * x[n - 1] - x[n - 2 - i]; } e.set(x, PAD);
    let y = sosfiltAll(e); y.reverse(); y = sosfiltAll(y); y.reverse(); return y.subarray(PAD, PAD + n); }
  function filt360(x) { const y = filtfilt(x), o = new Float32Array(Math.ceil(y.length / 2)); for (let i = 0; i < o.length; i++) o[i] = y[2 * i]; return o; }
  const HALF = 225, NB = 20;
  const rhe = (x) => { const r = Math.round(x); return Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : r; };   // numpy's round-half-even
  const median = (a) => { const s = Float64Array.from(a).sort(), n = s.length; return n % 2 ? s[(n - 1) >> 1] : 0.5 * (s[n / 2 - 1] + s[n / 2]); };
  // features for beats `keep` (indices into pos, positions in samples at 360 Hz) of filtered 180 Hz signal y
  function features(y, pos, keep) {
    const n = pos.length, W = [], ptp = new Float64Array(n), q = Math.round(0.08 * 180);
    for (let i = 0; i < n; i++) { const p = rhe(pos[i] / 2), w = new Float32Array(2 * HALF); for (let j = 0; j < 2 * HALF; j++) { const k = p - HALF + j; w[j] = (k >= 0 && k < y.length) ? y[k] : 0; }
      let mx = -Infinity, mn = Infinity; for (let j = HALF - q; j < HALF + q; j++) { if (w[j] > mx) mx = w[j]; if (w[j] < mn) mn = w[j]; } ptp[i] = mx - mn; W.push(w); }
    const t = Array.from(pos, (p) => p / 360), rr = []; for (let i = 1; i < n; i++) rr.push(t[i] - t[i - 1]);
    const pre = [n > 1 ? rr[0] : 1, ...rr], post = [...rr, n > 1 ? rr[rr.length - 1] : 1], out = [];
    for (const i of keep) {
      const lo = Math.max(0, i - NB), hi = Math.min(n, i + NB + 1); let nb = []; for (let j = lo; j < hi; j++) if (j !== i) nb.push(j); if (!nb.length) nb = [i];
      const s = Math.max(median(nb.map((j) => ptp[j])), 0.05), tpl = new Float32Array(2 * HALF), col = new Float64Array(nb.length);
      for (let l = 0; l < 2 * HALF; l++) { for (let a = 0; a < nb.length; a++) col[a] = W[nb[a]][l]; tpl[l] = median(col) / s; }
      const k0 = Math.max(0, i - 10), k1 = Math.min(n - 1, i + 10); let loc = 1, irr = 0;
      if (k1 > k0) { let sm = 0; for (let j = k0; j < k1; j++) sm += rr[j]; loc = sm / (k1 - k0); }
      if (k1 - k0 > 2) { const d = []; for (let j = k0 + 1; j < k1; j++) d.push(Math.abs(Math.log(rr[j]) - Math.log(rr[j - 1]))); irr = median(d); }
      const w = Float32Array.from(W[i], (v) => v / s);
      out.push({ w, tpl, rr: [Math.log(pre[i] / loc), Math.log(post[i] / loc), Math.log(loc / 0.8), Math.log(pre[i] / post[i]), irr * 5, Math.log(ptp[i] / s)], scale: s });
    }
    return out;
  }
  function read(net, f) {   // one beat's features -> {p: [N,S,V,F], qrs, qrsSig}
    const M = net.meta, x = new Float32Array(M.cin * 2 * HALF); x.set(f.w, 0); if (M.cin === 2) x.set(f.tpl, 2 * HALF);
    const o = forward(net, x, f.rr.map((v) => (isFinite(v) ? v : 0))), lp = M.logprior || [0, 0, 0, 0], z = [0, 1, 2, 3].map((i) => o[i] + lp[i]), mx = Math.max(...z);
    const e = z.map((v) => Math.exp(v - mx)), s = e.reduce((a, b) => a + b, 0), c = M.qrs_cal || { a: 0, b: 1, sig_factor: 1 };
    const q = Math.exp(o[4] * M.qsd + M.qmu), qs = Math.exp(Math.min(2, Math.max(-4, o[5]))) * M.qsd * q;
    return { p: e.map((v) => v / s), raw: Array.from(o), qrs: c.a + c.b * q, qrsSig: qs * c.b * c.sig_factor };
  }
  return { load, forward, filt360, features, read, median, _filtfilt: filtfilt };
})();
if (typeof module !== 'undefined') module.exports = BNet;
