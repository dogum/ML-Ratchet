// TsunamiNet in the browser: per-station CNN + set pooling + RealNVP flow (port of release/tsunaminet.py).
(function (g) {
  function b64f32(s) { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new Float32Array(u.buffer); }
  function erf(x) { const t = 1 / (1 + 0.3275911 * Math.abs(x)); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; }
  const gelu = x => 0.5 * x * (1 + erf(x / Math.SQRT2));
  function Net(W) {
    this.m = W.meta; const T = {};
    for (const k in W.t) { const [shape, b64] = W.t[k]; T[k] = { s: shape, d: b64f32(b64) }; }
    this.T = T;
  }
  Net.prototype.conv = function (x, C, L, w, b, stride, pad) { // x: Float32Array C*L
    const O = w.s[0], k = w.s[2], Lo = Math.floor((L + 2 * pad - k) / stride) + 1, y = new Float32Array(O * Lo);
    for (let o = 0; o < O; o++) for (let l = 0; l < Lo; l++) {
      let s = b.d[o]; const st = l * stride - pad;
      for (let c = 0; c < C; c++) { const wb = (o * C + c) * k, xb = c * L; for (let j = 0; j < k; j++) { const p = st + j; if (p >= 0 && p < L) s += w.d[wb + j] * x[xb + p]; } }
      y[o * Lo + l] = gelu(s);
    }
    return [y, O, Lo];
  };
  Net.prototype.lin = function (x, w, b, act) { const O = w.s[0], I = w.s[1], y = new Float32Array(O); for (let o = 0; o < O; o++) { let s = b.d[o]; for (let i = 0; i < I; i++) s += w.d[o * I + i] * x[i]; y[o] = act ? gelu(s) : s; } return y; };
  // SIFT-style NNLS on the input windows (projected coordinate descent on the normal equations, 18 unknowns)
  function nnlsFeatures(X) {
    const n = 18, AtA = new Float64Array(n * n), Atb = new Float64Array(n); let bb = 0;
    for (const x of X) { if (!x) continue;
      for (let t = 0; t < 90; t++) { if (!(x[90 + t] > 0)) continue; const b = x[t] * 10; bb += b * b;
        for (let i = 0; i < n; i++) { const ai = x[(2 + i) * 90 + t]; if (!ai) continue; Atb[i] += ai * b; for (let j = 0; j < n; j++) AtA[i * n + j] += ai * x[(2 + j) * 90 + t]; } } }
    const s = new Float64Array(n);
    for (let it = 0; it < 3000; it++) { let mv = 0;
      for (let i = 0; i < n; i++) { const d = AtA[i * n + i]; if (d <= 1e-12) continue; let g = Atb[i]; for (let j = 0; j < n; j++) if (j !== i) g -= AtA[i * n + j] * s[j];
        const v = Math.max(0, g / d); mv = Math.max(mv, Math.abs(v - s[i])); s[i] = v; }
      if (mv < 1e-10) break; }
    let r2 = bb; for (let i = 0; i < n; i++) { r2 -= 2 * s[i] * Atb[i]; for (let j = 0; j < n; j++) r2 += s[i] * AtA[i * n + j] * s[j]; }
    const out = new Float32Array(23), tot = s.reduce((a, b) => a + b, 0);
    if (tot <= 0) { out[0] = -1.5; out[3] = 1; out[4] = -1; return out; }
    out[0] = (Math.log10(4e10 * 100e3 * 50e3 * tot) - 9.1) / 1.5 - 8.3;
    let c = 0, deep = 0; for (let i = 0; i < n; i++) { c += ((i % 9) - 4) * s[i]; if (i >= 9) deep += s[i]; }
    out[1] = c / tot; out[2] = deep / tot; out[3] = Math.sqrt(Math.max(r2, 0)) / (Math.sqrt(bb) + 1e-9); out[4] = Math.log10(tot + 0.1);
    for (let i = 0; i < n; i++) out[5 + i] = s[i] / tot;
    return out;
  }
  // X: array of K stations, each Float32Array(20*90) or null; S: arrays of 3
  Net.prototype.context = function (X, S) {
    const T = this.T, E = [];
    for (let k = 0; k < X.length; k++) {
      if (!X[k]) continue;
      let h = X[k], C = 20, L = 90;
      this.m.conv.forEach((sp, i) => { [h, C, L] = this.conv(h, C, L, T['c' + i + 'w'], T['c' + i + 'b'], sp[0], sp[1]); });
      const f = new Float32Array(2 * C + 3);
      for (let c = 0; c < C; c++) { let s = 0, m = -1e9; for (let l = 0; l < L; l++) { const v = h[c * L + l]; s += v; if (v > m) m = v; } f[c] = s / L; f[C + c] = m; }
      f.set(S[k], 2 * C);
      E.push(this.lin(this.lin(f, T.st0w, T.st0b, true), T.st1w, T.st1b, false));
    }
    const D = E[0].length, c = new Float32Array(2 * D + 1);
    for (let d = 0; d < D; d++) { let s = 0, m = -1e9; for (const e of E) { s += e[d]; if (e[d] > m) m = e[d]; } c[d] = s / E.length; c[D + d] = m; }
    c[2 * D] = E.length / 3;
    if (T.ctx0w.s[1] > 2 * D + 1) { const f = nnlsFeatures(X), cc = new Float32Array(2 * D + 1 + 23); cc.set(c); cc.set(f, 2 * D + 1); return this.lin(this.lin(cc, T.ctx0w, T.ctx0b, true), T.ctx1w, T.ctx1b, false); }
    return this.lin(this.lin(c, T.ctx0w, T.ctx0b, true), T.ctx1w, T.ctx1b, false);
  };
  Net.prototype.sample = function (ctx, n, rng) {
    const T = this.T, out = [], mu = this.m.mu, sd = this.m.sd, nf = this.m.n_flow;
    for (let i = 0; i < n; i++) {
      let z = [rng(), rng(), rng(), rng()];
      for (let f = nf - 1; f >= 0; f--) {
        const m = T['f' + f + 'm'].d, inp = new Float32Array(4 + ctx.length);
        for (let j = 0; j < 4; j++) inp[j] = z[j] * m[j]; inp.set(ctx, 4);
        const o = this.lin(this.lin(this.lin(inp, T['f' + f + 'w0'], T['f' + f + 'b0'], true), T['f' + f + 'w1'], T['f' + f + 'b1'], true), T['f' + f + 'w2'], T['f' + f + 'b2'], false);
        for (let j = 0; j < 4; j++) { if (m[j]) continue; const s = 2.5 * Math.tanh(o[j] / 2.5), t = o[4 + j]; z[j] = (z[j] - t) * Math.exp(-s); }
      }
      out.push(z.map((v, j) => v * sd[j] + mu[j]));
    }
    return out;
  };
  function gauss(seed) { let s = seed >>> 0 || 1; const u = () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return (s + 0.5) / 4294967296; }; return () => Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u()); }
  g.TNet = { Net, gauss, nnlsFeatures };
})(window);
