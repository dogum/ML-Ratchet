// FaultFinder in the browser: the same arithmetic as release/faultfinder.py (features.py + ffnumpy.py).
// raw: 13 channel arrays of 96 values (null = missing), order RAWC below.
const FFNet = (() => {
  const RAWC = ['T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_sa_sp', 'T_zone', 'damper_oa', 'valve_cc', 'valve_hc', 'fan_speed',
    'fan_power', 'fan_on', 'occ'];
  const R = {}; RAWC.forEach((c, i) => { R[c] = i; });
  const NT = 96, NF = 35;

  function load(pack) {           // pack: {b64: float32 little-endian, man: [{k, shape, off}], meta}
    const bin = atob(pack.b64), buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    const all = new Float32Array(buf), T = {};
    for (const t of pack.man) { const sz = t.shape.reduce((a, b) => a * b, 1); T[t.k] = { d: all.subarray(t.off, t.off + sz), s: t.shape }; }
    return { T, meta: pack.meta };
  }

  const fin = (v) => v !== null && v !== undefined && Number.isFinite(v);
  function fill(v) {               // linear interpolation over missing samples (np.interp semantics, flat ends)
    const idx = []; for (let i = 0; i < NT; i++) if (fin(v[i])) idx.push(i);
    const o = new Float64Array(NT);
    if (idx.length === 0) return o;
    for (let i = 0; i < NT; i++) {
      if (fin(v[i])) { o[i] = v[i]; continue; }
      if (i <= idx[0]) { o[i] = v[idx[0]]; continue; }
      if (i >= idx[idx.length - 1]) { o[i] = v[idx[idx.length - 1]]; continue; }
      let a = 0; while (idx[a + 1] < i) a++;
      const i0 = idx[a], i1 = idx[a + 1];
      o[i] = v[i0] + (v[i1] - v[i0]) * (i - i0) / (i1 - i0);
    }
    return o;
  }
  function pct(vals, q) {          // np.percentile, linear
    const s = vals.slice().sort((a, b) => a - b), n = s.length;
    const r = (q / 100) * (n - 1), lo = Math.floor(r), hi = Math.ceil(r);
    return s[lo] + (s[hi] - s[lo]) * (r - lo);
  }
  const clip = (x, a, b) => Math.min(b, Math.max(a, x));

  function featurize(raw) {
    const f = new Float32Array(NF * NT), avail = [], M = [];
    for (let c = 0; c < 13; c++) {
      const m = new Float32Array(NT); let any = false;
      for (let i = 0; i < NT; i++) if (fin(raw[c][i])) { m[i] = 1; any = true; }
      M.push(m); avail.push(any);
    }
    const X = raw.map(fill);
    const fpv = raw[R.fan_power].filter(fin);
    if (fpv.length) {
      const p = pct(fpv, 95);
      for (let i = 0; i < NT; i++) X[R.fan_power][i] = p > 1e-9 ? X[R.fan_power][i] / p : 0;
    }
    let k = 0;
    const set = (row, fn) => { for (let i = 0; i < NT; i++) f[row * NT + i] = fn(i); };
    for (const c of ['T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_sa_sp', 'T_zone']) {
      if (avail[R[c]]) set(k, (i) => (X[R[c]][i] - 20) / 10); k++;
    }
    for (const [a, b, s] of [['T_ma', 'T_ra', 5], ['T_sa', 'T_ma', 5], ['T_sa', 'T_sa_sp', 3], ['T_oa', 'T_ra', 10], ['T_zone', 'T_ra', 3]]) {
      if (avail[R[a]] && avail[R[b]]) set(k, (i) => clip((X[R[a]][i] - X[R[b]][i]) / s, -5, 5)); k++;
    }
    const frm = new Float32Array(NT);
    if (avail[R.T_oa] && avail[R.T_ma] && avail[R.T_ra]) {
      set(k, (i) => {
        const den = X[R.T_oa][i] - X[R.T_ra][i];
        if (Math.abs(den) > 3) { frm[i] = 1; return clip((X[R.T_ma][i] - X[R.T_ra][i]) / den, -0.5, 1.5); }
        return 0;
      });
    }
    k++;
    for (const c of ['damper_oa', 'valve_cc', 'valve_hc', 'fan_speed', 'fan_power', 'fan_on', 'occ']) {
      if (avail[R[c]]) set(k, (i) => clip(X[R[c]][i], -0.2, 2.0)); k++;
    }
    set(k, (i) => Math.sin(i / NT * 2 * Math.PI)); set(k + 1, (i) => Math.cos(i / NT * 2 * Math.PI)); k += 2;
    for (let c = 0; c < 13; c++) set(k + c, (i) => M[c][i]);
    set(k + 13, (i) => frm[i]);
    return f;
  }

  function conv(x, Ci, L, w, b) {   // 'same' padding, stride 1
    const [O, , K] = w.s, p = K >> 1, out = new Float32Array(O * L), W = w.d;
    for (let o = 0; o < O; o++) {
      const ob = o * L; out.fill(b.d[o], ob, ob + L);
      for (let c = 0; c < Ci; c++) {
        const xb = c * L, wb = (o * Ci + c) * K;
        for (let kk = 0; kk < K; kk++) {
          const wv = W[wb + kk], off = kk - p;
          const l0 = Math.max(0, -off), l1 = Math.min(L, L - off);
          for (let l = l0; l < l1; l++) out[ob + l] += wv * x[xb + l + off];
        }
      }
    }
    for (let i = 0; i < out.length; i++) if (out[i] < 0) out[i] = 0;
    return [out, O];
  }
  function pool(x, C, L) {
    const L2 = L >> 1, o = new Float32Array(C * L2);
    for (let c = 0; c < C; c++) for (let l = 0; l < L2; l++) o[c * L2 + l] = Math.max(x[c * L + 2 * l], x[c * L + 2 * l + 1]);
    return [o, L2];
  }
  function dense(z, W, b, relu) {
    const [O, I] = W.s, o = new Float32Array(O);
    for (let i = 0; i < O; i++) { let a = b.d[i]; const r = i * I; for (let j = 0; j < I; j++) a += W.d[r + j] * z[j]; o[i] = relu && a < 0 ? 0 : a; }
    return o;
  }
  function forward(net, f) {
    const T = net.T;
    let [h, C] = conv(f, NF, NT, T.c1_W, T.c1_b), L = NT;
    [h, C] = conv(h, C, L, T.c2_W, T.c2_b); [h, L] = pool(h, C, L);
    [h, C] = conv(h, C, L, T.c3_W, T.c3_b); [h, L] = pool(h, C, L);
    [h, C] = conv(h, C, L, T.c4_W, T.c4_b);
    const z = new Float32Array(2 * C);
    for (let c = 0; c < C; c++) { let m = 0, M = -Infinity; for (let l = 0; l < L; l++) { const v = h[c * L + l]; m += v; if (v > M) M = v; } z[c] = m / L; z[C + c] = M; }
    const g = dense(z, T.fc_W, T.fc_b, true);
    return { logits: dense(g, T.cls_W, T.cls_b, false), par: dense(g, T.par_W, T.par_b, false), waste: dense(g, T.waste_W, T.waste_b, false)[0] };
  }
  // calibrated outputs: p(fault) from detection log-odds / T; diagnosis = softmax(fault logits / T)
  function predict(net, raw) {
    const out = forward(net, featurize(raw)), lo = out.logits, T = net.meta.temperature;
    let m = -Infinity; for (let i = 1; i < lo.length; i++) m = Math.max(m, lo[i]);
    let s = 0; for (let i = 1; i < lo.length; i++) s += Math.exp(lo[i] - m);
    const d = m + Math.log(s) - lo[0], pFault = 1 / (1 + Math.exp(-d / T));
    let m2 = -Infinity; for (let i = 1; i < lo.length; i++) m2 = Math.max(m2, lo[i] / T);
    const e = []; let s2 = 0; for (let i = 1; i < lo.length; i++) { const v = Math.exp(lo[i] / T - m2); e.push(v); s2 += v; }
    const probs = e.map((v) => v / s2);
    return { d, pFault, probs, classes: net.meta.classes.slice(1), par: out.par, waste: out.waste };
  }
  return { load, featurize, forward, predict, RAWC };
})();
if (typeof module !== 'undefined') module.exports = FFNet;
