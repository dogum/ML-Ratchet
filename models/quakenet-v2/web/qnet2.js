// QuakeNet v2 in the browser: the same arithmetic as release/quakenet2.py (anchor least squares, features, graph network).
const QNet2 = (() => {
  const R2D = 180 / Math.PI, D2R = Math.PI / 180, KM = Math.PI * 6371 / 180;
  let TT = null, W = null, META = null;
  function init(tt, weights) {
    TT = tt; META = weights.meta; W = {};
    for (const k in weights.w) { const a = weights.w[k];
      const bin = (typeof atob !== 'undefined') ? atob(a.b64) : Buffer.from(a.b64, 'base64').toString('binary'); const u = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
      W[k] = { d: new Float32Array(u.buffer), s: a.s }; }
  }
  // ---- ak135 P travel time, bilinear in distance (0.25 deg) and depth
  function ttP(d, z) {
    const D = TT.depths, T = TT.P, nd = T[0].length;
    d = Math.min(Math.max(d, 0), 100); z = Math.min(Math.max(z, 2), 700);
    const fj = d / 0.25; let j = Math.min(Math.max(Math.floor(fj), 0), nd - 2); const wj = fj - j;
    let i = 0; while (i < D.length - 2 && D[i + 1] < z) i++;
    const wi = (z - D[i]) / (D[i + 1] - D[i]);
    return (T[i][j] * (1 - wj) + T[i][j + 1] * wj) * (1 - wi) + (T[i + 1][j] * (1 - wj) + T[i + 1][j + 1] * wj) * wi;
  }
  const slow = (d, z) => (ttP(d + 0.25, z) - ttP(d - 0.25, z)) / 0.5;
  function dist(a1, o1, a2, o2) {
    const a = a1 * D2R, b = a2 * D2R, dl = (o2 - o1) * D2R;
    const c = Math.sin(a) * Math.sin(b) + Math.cos(a) * Math.cos(b) * Math.cos(dl);
    const s = Math.hypot(Math.cos(b) * Math.sin(dl), Math.cos(a) * Math.sin(b) - Math.sin(a) * Math.cos(b) * Math.cos(dl));
    return Math.atan2(s, c) * R2D;
  }
  function azim(a1, o1, a2, o2) {
    const a = a1 * D2R, b = a2 * D2R, dl = (o2 - o1) * D2R;
    let z = Math.atan2(Math.sin(dl) * Math.cos(b), Math.cos(a) * Math.sin(b) - Math.sin(a) * Math.cos(b) * Math.cos(dl)) * R2D;
    return ((z % 360) + 360) % 360;
  }
  function dest(lat, lon, az, d) {
    const a = lat * D2R, l = lon * D2R, t = az * D2R, dd = d * D2R;
    const b = Math.asin(Math.sin(a) * Math.cos(dd) + Math.cos(a) * Math.sin(dd) * Math.cos(t));
    const l2 = l + Math.atan2(Math.sin(t) * Math.sin(dd) * Math.cos(a), Math.cos(dd) - Math.sin(a) * Math.sin(b));
    return [b * R2D, ((l2 * R2D + 180) % 360 + 360) % 360 - 180];
  }
  const adiff = (a, b) => ((((a - b + 180) % 360) + 360) % 360) - 180;
  const gelu = (x) => { const a = Math.abs(x) * 0.7071067811865476, t = 1 / (1 + 0.3275911 * a);
    const erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a); return 0.5 * x * (1 + (x >= 0 ? erf : -erf)); };

  // ---- the classical combination of v1.2's readings (Levenberg-Marquardt on lat, lon, numeric Jacobian)
  const F = { mw: 1.0, dist: 1.6404, logz: 2.4165, baz: 1.1398 };
  function combine(st) {
    const sd = st.map(s => s.dist_sig * F.dist), sb = st.map(s => Math.max(R2D / Math.sqrt(Math.max(s.kappa, 1e-3)) * F.baz, 0.5));
    const res = (p) => { const r = []; st.forEach((s, i) => r.push((dist(p[0], p[1], s.lat, s.lon) - s.dist) / sd[i]));
      st.forEach((s, i) => r.push(adiff(azim(s.lat, s.lon, p[0], p[1]), s.baz) / sb[i])); return r; };
    const cost = (p) => res(p).reduce((a, v) => a + v * v, 0);
    const starts = st.map(s => dest(s.lat, s.lon, s.baz, s.dist));
    let ux = 0, uy = 0, uz = 0;
    starts.forEach((c, i) => { const w = 1 / (sd[i] * sb[i]), la = c[0] * D2R, lo = c[1] * D2R; ux += w * Math.cos(la) * Math.cos(lo); uy += w * Math.cos(la) * Math.sin(lo); uz += w * Math.sin(la); });
    const n = Math.hypot(ux, uy, uz); starts.push([Math.asin(uz / n) * R2D, Math.atan2(uy, ux) * R2D]);
    const order = starts.map((s, i) => [cost(s), i]).sort((a, b) => a[0] - b[0]).slice(0, 2);
    let best = null;
    for (const [, i] of order) {
      let p = starts[i].slice(), lam = 1e-3, c = cost(p);
      for (let it = 0; it < 100; it++) {
        const r = res(p), J = r.map(() => [0, 0]);
        for (let k = 0; k < 2; k++) { const q = p.slice(); const h = 1e-5 * Math.max(1, Math.abs(p[k])); q[k] += h; const rq = res(q); rq.forEach((v, m) => J[m][k] = (v - r[m]) / h); }
        let a = 0, b = 0, d = 0, g0 = 0, g1 = 0; J.forEach((j, m) => { a += j[0] * j[0]; b += j[0] * j[1]; d += j[1] * j[1]; g0 += j[0] * r[m]; g1 += j[1] * r[m]; });
        let moved = false;
        for (let t = 0; t < 10; t++) {
          const A = a * (1 + lam), Dd = d * (1 + lam), det = A * Dd - b * b; if (!isFinite(det) || det === 0) { lam *= 10; continue; }
          const s0 = -(Dd * g0 - b * g1) / det, s1 = -(A * g1 - b * g0) / det, q = [p[0] + s0, p[1] + s1], cq = cost(q);
          if (cq < c) { p = q; c = cq; lam = Math.max(lam / 3, 1e-7); moved = true; break; } lam *= 4;
        }
        if (!moved) break;
      }
      if (!best || c < best.c) best = { p, c };
    }
    const lat = Math.min(Math.max(best.p[0], -90), 90), lon = ((best.p[1] + 180) % 360 + 360) % 360 - 180;
    let wm = 0, sm = 0, wz = 0, sz = 0;
    st.forEach(s => { const a = 1 / (s.mw_sig * F.mw) ** 2, b = 1 / (s.logz_sig * F.logz) ** 2; wm += a; sm += a * s.mw; wz += b; sz += b * s.logz; });
    return { lat, lon, mw: sm / wm, logz: sz / wz };
  }

  function lin(name, x, n) {   // x: Float32Array (rows * in), returns rows * out
    let cur = x, rows = null;
    for (let j = 0; j < n; j++) {
      const w = W[`${name}_${j}_w`], b = W[`${name}_${j}_b`], [O, I] = w.s; rows = cur.length / I; const out = new Float32Array(rows * O);
      for (let r = 0; r < rows; r++) for (let o = 0; o < O; o++) { let s = b.d[o]; const wo = o * I, xr = r * I; for (let i = 0; i < I; i++) s += w.d[wo + i] * cur[xr + i]; out[r * O + o] = j < n - 1 ? gelu(s) : s; }
      cur = out;
    }
    return cur;
  }
  function ccFeats(pz) {
    const K = pz.length, Wn = pz.map(p => { const w = p.slice(40, 160); const m = w.reduce((a, v) => a + v, 0) / w.length; const c = w.map(v => v - m); const nn = Math.hypot(...c) + 1e-9; return c.map(v => v / nn); });
    const out = []; for (let i = 0; i < K; i++) { out.push([]); for (let j = 0; j < K; j++) out[i].push([0, 0]); }
    for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) {
      let bk = 0, bv = 0; for (let k = 0; k <= 40; k++) { let s = 0; for (let m = 0; m < 120; m++) { const ii = k + m - 20; if (ii >= 0 && ii < 120) s += Wn[i][ii] * Wn[j][m]; } if (Math.abs(s) > Math.abs(bv) || k === 0) { if (k === 0 || Math.abs(s) > Math.abs(bv)) { bv = s; bk = k; } } }
      const lag = bk - 20; out[i][j] = [lag / 10, Math.abs(bv)]; out[j][i] = [-lag / 10, Math.abs(bv)];
    }
    return out;
  }
  // st: [{lat, lon, mw, mw_sig, dist, dist_sig, logz, logz_sig, baz, kappa, ls, peak, tpick, pz}]
  function locate(st, calibrated = true) {
    const K = st.length, A = combine(st), zA = Math.pow(10, A.logz);
    const dA = st.map(s => dist(A.lat, A.lon, s.lat, s.lon)), azA = st.map(s => azim(A.lat, A.lon, s.lat, s.lon) * D2R);
    const base = st.map((s, i) => s.tpick - ttP(dA[i], zA)), t0 = base.reduce((a, v) => a + v, 0) / K;
    const useT = META.use_time, r = base.map(b => useT ? Math.min(Math.max((b - t0) / 10, -10), 10) : 0);
    const FN = 16, H = META.h, xn = new Float32Array(K * FN);
    st.forEach((s, i) => {
      const sbv = R2D / Math.sqrt(Math.max(s.kappa, 1e-3)), toA = azim(s.lat, s.lon, A.lat, A.lon);
      const f = [(s.dist - dA[i]) / 10, Math.log(s.dist_sig), adiff(s.baz, toA) / 30, Math.log(sbv), s.mw - A.mw, Math.log(s.mw_sig), s.logz - A.logz, Math.log(s.logz_sig),
        (s.ls + 6) / 2, Math.log10(Math.max(s.peak, 1)), dA[i] / 90, Math.sin(azA[i]), Math.cos(azA[i]), r[i], useT ? slow(dA[i], zA) / 10 : 0, K / 10];
      f.forEach((v, k) => xn[i * FN + k] = v);
    });
    const FE = META.use_cc ? 8 : 6, cc = META.use_cc ? ccFeats(st.map(s => s.pz)) : null;
    let h = lin('enc', xn, 2);
    for (let rd = 0; rd < META.rounds; rd++) {
      const inp = new Float32Array(K * K * (2 * H + FE));
      for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
        const o = (i * K + j) * (2 * H + FE);
        for (let k = 0; k < H; k++) { inp[o + k] = h[i * H + k]; inp[o + H + k] = h[j * H + k]; }
        const sep = dist(st[i].lat, st[i].lon, st[j].lat, st[j].lon) / 90, daz = azA[j] - azA[i];
        const e = [r[j] - r[i], sep, Math.sin(daz), Math.cos(daz), (dA[j] - dA[i]) / 30, 0]; if (cc) e.push(cc[i][j][0], cc[i][j][1]);
        e.forEach((v, k) => inp[o + 2 * H + k] = v);
      }
      const msg = lin(`msg${rd}`, inp, 2), agg = new Float32Array(K * H);
      for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) if (i !== j) for (let k = 0; k < H; k++) agg[i * H + k] += msg[(i * K + j) * H + k] / Math.max(K - 1, 1);
      const cat = new Float32Array(K * 2 * H); for (let i = 0; i < K; i++) for (let k = 0; k < H; k++) { cat[i * 2 * H + k] = h[i * H + k]; cat[i * 2 * H + H + k] = agg[i * H + k]; }
      const up = lin(`upd${rd}`, cat, 2); const nh = new Float32Array(K * H); for (let q = 0; q < K * H; q++) nh[q] = h[q] + up[q]; h = nh;
    }
    const pool = new Float32Array(2 * H); for (let k = 0; k < H; k++) { let s = 0, m = -1e30; for (let i = 0; i < K; i++) { s += h[i * H + k]; m = Math.max(m, h[i * H + k]); } pool[k] = s / K; pool[H + k] = m; }
    const o = lin('out', pool, 3), fct = calibrated ? META.sigma_factor : { epi: 1, z: 1, mw: 1, t0: 1 };
    const cl = (v, a, b) => Math.min(Math.max(v, a), b), a = Math.exp(cl(o[5], -5, 4)), b = Math.exp(cl(o[6], -5, 4)), off = o[7];
    const L = [[a * 100 * fct.epi, 0], [off * 100 * fct.epi, b * 100 * fct.epi]], C = [[L[0][0] ** 2, L[0][0] * L[1][0]], [L[0][0] * L[1][0], L[1][0] ** 2 + L[1][1] ** 2]];
    const n = o[0] * 100, e = o[1] * 100, p = dest(A.lat, A.lon, ((Math.atan2(e, n) * R2D) + 360) % 360, Math.hypot(n, e) / KM);
    return { lat: p[0], lon: p[1], cov: C, depth: Math.pow(10, A.logz + o[2]), logz_sig: Math.exp(cl(o[8], -6, 3)) * fct.z, mw: A.mw + o[3], mw_sig: Math.exp(cl(o[9], -6, 3)) * fct.mw,
      t0: t0 + 10 * o[4], t0_sig: 10 * Math.exp(cl(o[10], -6, 3)) * fct.t0, anchor: { lat: A.lat, lon: A.lon, depth: zA, mw: A.mw, t0 }, raw: Array.from(o) };
  }
  return { init, locate, combine, dist, azim, dest, ttP };
})();
if (typeof module !== 'undefined') module.exports = QNet2;
