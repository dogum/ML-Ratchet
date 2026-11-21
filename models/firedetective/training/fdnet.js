// FireDetective in the browser: the same arithmetic as release/firedetective.py
const FDNet = (() => {
  function load(pack) {   // pack: {k: {s: shape, d: [numbers]}}
    const T = {};
    for (const k in pack) T[k] = { s: pack[k].s, d: Float64Array.from(pack[k].d) };
    return T;
  }
  const relu = (v) => (v > 0 ? v : 0);
  function lin(x, W, b) {   // x Float64Array(I), W [O, I]
    const [O, I] = W.s, out = new Float64Array(O);
    for (let o = 0; o < O; o++) { let a = b.d[o]; const ob = o * I; for (let i = 0; i < I; i++) a += W.d[ob + i] * x[i]; out[o] = a; }
    return out;
  }
  // raw forward for one fire: seq = array of T rows of F numbers, st = array of S numbers
  function raw(W, seq, st) {
    const Tn = seq.length, F = seq[0].length;
    const mask = seq.map((r) => (r[0] > 0 ? 1 : 0));
    const H = W['emb.weight'].s[0];
    const h = [];   // [T][H]
    for (let t = 0; t < Tn; t++) {
      const x = new Float64Array(F);
      for (let f = 0; f < F; f++) x[f] = ((seq[t][f] - W.seq_mu.d[f]) / W.seq_sd.d[f]) * mask[t];
      const e = lin(x, W['emb.weight'], W['emb.bias']);
      for (let i = 0; i < H; i++) e[i] = relu(e[i]) * mask[t];
      h.push(e);
    }
    function conv(inp, w, b) {   // inp [T][C] -> [T][O]
      const [O, C, K] = w.s, p = K >> 1, out = [];
      for (let t = 0; t < Tn; t++) {
        const o_ = new Float64Array(O);
        for (let o = 0; o < O; o++) {
          let a = b.d[o];
          for (let k = 0; k < K; k++) {
            const j = t + k - p; if (j < 0 || j >= Tn) continue;
            const row = inp[j], wb = o * C * K + k;
            for (let c = 0; c < C; c++) a += w.d[wb + c * K] * row[c];
          }
          o_[o] = relu(a) * mask[t];
        }
        out.push(o_);
      }
      return out;
    }
    const c1 = conv(h, W['c1.weight'], W['c1.bias']);
    const c2 = conv(c1, W['c2.weight'], W['c2.bias']);
    const C = W['c2.weight'].s[0];
    const n = Math.max(mask.reduce((a, b) => a + b, 0), 1);
    const mean = new Float64Array(C), mx = new Float64Array(C).fill(-Infinity);
    for (let t = 0; t < Tn; t++) for (let c = 0; c < C; c++) {
      mean[c] += c2[t][c] / n;
      const v = c2[t][c] - 1e4 * (1 - mask[t]); if (v > mx[c]) mx[c] = v;
    }
    const xs = new Float64Array(st.length);
    for (let i = 0; i < st.length; i++) xs[i] = (st[i] - W.st_mu.d[i]) / W.st_sd.d[i];
    const s = lin(xs, W['st.weight'], W['st.bias']).map(relu);
    const cat = new Float64Array(2 * C + s.length);
    cat.set(mean, 0); cat.set(mx, C); cat.set(s, 2 * C);
    const z1 = lin(cat, W['f1.weight'], W['f1.bias']).map(relu);
    const z2 = lin(z1, W['f2.weight'], W['f2.bias']).map(relu);
    const mu = lin(z2, W['mu.weight'], W['mu.bias']);
    const ls = lin(z2, W['ls.weight'], W['ls.bias']).map((v) => Math.min(Math.max(v, -5), 3));
    const bl = lin(z2, W['bn.weight'], W['bn.bias']);
    return { mu: Array.from(mu), sd: ls.map(Math.exp), p: Array.from(bl).map((v) => 1 / (1 + Math.exp(-v))) };
  }
  function rawEns(Ws, seq, st) {   // Ws: list of member weight sets (shared normalisation inside each)
    const outs = Ws.map((W) => raw(W, seq, st)), K = outs.length, n = outs[0].mu.length, m = outs[0].p.length;
    const mu = [], sd = [], p = [];
    for (let i = 0; i < n; i++) {
      let a = 0, v = 0; for (const o of outs) a += o.mu[i] / K;
      for (const o of outs) v += (o.sd[i] * o.sd[i] + (o.mu[i] - a) * (o.mu[i] - a)) / K;
      mu.push(a); sd.push(Math.sqrt(v));
    }
    for (let i = 0; i < m; i++) { let a = 0; for (const o of outs) a += o.p[i] / K; p.push(a); }
    return { mu, sd, p };
  }
  function predict(Ws, cal, seq, seqH, st) {
    const a = rawEns(Ws, seq, st), b = rawEns(Ws, seqH, st);
    const e = Math.exp(a.mu[0]), crown = (1.02 * e - 0.02) / (1 + e), dcdy = 1.04 * e / Math.pow(1 + e, 2);
    const hs = cal.hs_map.a + cal.hs_map.b * crown;
    const hsSd = Math.abs(cal.hs_map.b) * dcdy * a.sd[0] * cal.hs_widen;
    const wsd = b.sd[1] * cal.wind_widen;
    return {
      hs: Math.min(Math.max(hs, 0), 1), hs_sd: hsSd, crown,
      wind: Math.exp(b.mu[1]), wind_lo: Math.exp(b.mu[1] - wsd), wind_hi: Math.exp(b.mu[1] + wsd),
      lfmc: a.mu[2] * 100, lfmc_sd: a.sd[2] * 100 * cal.lfmc_widen, ros_adj: Math.exp(a.mu[3]),
      m1: Math.exp(a.mu[4]), wind_regime: Math.min(Math.max(a.mu[6], 0), 1), p_spot: a.p[0], p_ia: a.p[1],
    };
  }
  return { load, raw, rawEns, predict };
})();
if (typeof module !== 'undefined') module.exports = FDNet;
