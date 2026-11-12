// ChirpNet forward pass in plain JavaScript (fused conv + GELU, identical to the numpy release).
function makeChirpJS() {
  function erf(x) {
    const s = x < 0 ? -1 : 1; x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x);
    return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
  }
  function gelu(a) { for (let i = 0; i < a.length; i++) { const v = a[i]; a[i] = 0.5 * v * (1 + erf(v * 0.7071067811865476)); } return a; }
  function conv(x, Cin, L, w, b, Cout, K, stride, pad) {
    const Lout = Math.floor((L + 2 * pad - K) / stride) + 1, y = new Float32Array(Cout * Lout);
    for (let o = 0; o < Cout; o++) {
      const yo = o * Lout, bo = b[o];
      for (let t = 0; t < Lout; t++) y[yo + t] = bo;
      for (let c = 0; c < Cin; c++) {
        const xc = c * L, wb = (o * Cin + c) * K;
        for (let k = 0; k < K; k++) {
          const wk = w[wb + k];
          const t0 = Math.max(0, Math.ceil((pad - k) / stride)), t1 = Math.min(Lout, Math.floor((L - 1 + pad - k) / stride) + 1);
          let i = t0 * stride - pad + k;
          for (let t = t0; t < t1; t++, i += stride) y[yo + t] += wk * x[xc + i];
        }
      }
    }
    return { y, L: Lout, C: Cout };
  }
  function load(pack) {
    const bin = atob(pack.b64), buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    const all = new Float32Array(buf), T = {};
    for (const t of pack.man.tensors) { const n = t.shape.reduce((a, b) => a * b, 1); T[t.k] = { a: all.subarray(t.off, t.off + n), shape: t.shape }; }
    return { T, conv: pack.man.conv };
  }
  // x: Float32Array(2*4096) [H1..., L1...]  ->  Float32Array(7)
  function forward(net, x) {
    const T = net.T, cv = net.conv;
    const L0 = (i, h) => { const w = T['w' + i]; return conv(h.y, h.C, h.L, w.a, T['b' + i].a, w.shape[0], w.shape[2], cv[i][0], cv[i][1]); };
    let h = { y: x, C: 2, L: 4096 };
    h = L0(0, h); gelu(h.y);
    for (let i = 1; i < cv.length; i += 3) {
      const a = L0(i, h); gelu(a.y);
      const b2 = L0(i + 1, a);
      const s = L0(i + 2, h);
      for (let j = 0; j < b2.y.length; j++) b2.y[j] += s.y[j];
      gelu(b2.y); h = b2;
    }
    const C = h.C, L = h.L, feats = [];
    for (let c = 0; c < C; c++) { let m = 0; for (let t = 0; t < L; t++) m += h.y[c * L + t]; feats.push(m / L); }
    for (let c = 0; c < C; c++) { let m = -1e30; for (let t = 0; t < L; t++) m = Math.max(m, h.y[c * L + t]); feats.push(m); }
    if (T.rw) { const r = conv(h.y, C, L, T.rw.a, T.rb.a, T.rw.shape[0], 1, 1, 0); gelu(r.y); for (let j = 0; j < r.y.length; j++) feats.push(r.y[j]); }
    const f = Float32Array.from(feats);
    const dense = (v, W, B) => { const [o, n] = W.shape, out = new Float32Array(o); for (let i = 0; i < o; i++) { let s = B.a[i]; for (let j = 0; j < n; j++) s += W.a[i * n + j] * v[j]; out[i] = s; } return out; };
    const z = gelu(dense(f, T.lw0, T.lb0));
    return dense(z, T.lw1, T.lb1);
  }
  // two-stream network: the same trunk reads each detector, then an interaction stage compares them position by position
  function forward2s(net, x) {
    const T = net.T, cv = net.conv, n = x.length / 2;
    const L0 = (i, h) => { const w = T['w' + i]; return conv(h.y, h.C, h.L, w.a, T['b' + i].a, w.shape[0], w.shape[2], cv[i][0], cv[i][1]); };
    const trunk = (sig) => {
      let h = L0(0, { y: sig, C: 1, L: n }); gelu(h.y);
      for (let i = 1; i < cv.length; i += 3) { const a = L0(i, h); gelu(a.y); const b2 = L0(i + 1, a), s = L0(i + 2, h); for (let j = 0; j < b2.y.length; j++) b2.y[j] += s.y[j]; gelu(b2.y); h = b2; }
      return h;
    };
    const h = trunk(x.subarray(0, n)), l = trunk(x.subarray(n)), C = h.C, L = h.L, N = C * L;
    const cat = new Float32Array(4 * N);
    for (let j = 0; j < N; j++) { const a = h.y[j], b = l.y[j]; cat[j] = a; cat[N + j] = b; cat[2 * N + j] = a * b; cat[3 * N + j] = Math.abs(a - b); }
    const cw = (k, v, Cin, Ln, pad) => { const w = T[k + 'w']; return conv(v, Cin, Ln, w.a, T[k + 'b'].a, w.shape[0], w.shape[2], 1, pad); };
    let z = cw('mix', cat, 4 * C, L, 0); gelu(z.y);
    const y1 = cw('p1', z.y, C, L, T.p1w.shape[2] >> 1); gelu(y1.y);
    const y2 = cw('p2', y1.y, C, L, T.p2w.shape[2] >> 1), sk = cw('ps', z.y, C, L, 0);
    for (let j = 0; j < y2.y.length; j++) y2.y[j] += sk.y[j]; gelu(y2.y); z = y2;
    const f = new Float32Array(4 * C);
    for (let c = 0; c < C; c++) {
      let m = 0, mx = -1e30, mh = -1e30, ml = -1e30;
      for (let t = 0; t < L; t++) { const v = z.y[c * L + t]; m += v; if (v > mx) mx = v; if (h.y[c * L + t] > mh) mh = h.y[c * L + t]; if (l.y[c * L + t] > ml) ml = l.y[c * L + t]; }
      f[c] = m / L; f[C + c] = mx; f[2 * C + c] = mh; f[3 * C + c] = ml;
    }
    const dense = (v, W, B) => { const [o, nn] = W.shape, out = new Float32Array(o); for (let i = 0; i < o; i++) { let s = B.a[i]; for (let j = 0; j < nn; j++) s += W.a[i * nn + j] * v[j]; out[i] = s; } return out; };
    return dense(gelu(dense(f, T.lw0, T.lb0)), T.lw1, T.lb1);
  }
  return { load, forward, forward2s };
}
const ChirpJS = makeChirpJS();
if (typeof module !== 'undefined') module.exports = ChirpJS;
