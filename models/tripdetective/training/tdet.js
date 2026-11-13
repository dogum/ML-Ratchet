// TripDetective in the browser: the same arithmetic as release/tripdetective.py (BatchNorm folded into each convolution).
const TDet = (() => {
  function half(h) {
    const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  function load(pack) {
    const bin = atob(pack.b64), n = bin.length >> 1, all = new Float32Array(n), tab = new Float32Array(65536);
    for (let i = 0; i < 65536; i++) tab[i] = half(i);
    for (let i = 0; i < n; i++) all[i] = tab[bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8)];
    const T = {};
    for (const t of pack.man) { const sz = t.shape.reduce((a, b) => a * b, 1); T[t.k] = { d: all.subarray(t.off, t.off + sz), s: t.shape }; }
    return { T, meta: pack.meta };
  }
  function erf(x) { // Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7)
    const s = x < 0 ? -1 : 1, a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
    return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a));
  }
  const gelu = (x) => 0.5 * x * (1 + erf(x * 0.7071067811865476));
  function conv(x, Ci, L, w, b, stride) {
    const [O, , K] = w.s, p = K >> 1, Lo = Math.floor((L + 2 * p - K) / stride) + 1, out = new Float32Array(O * Lo), W = w.d;
    for (let o = 0; o < O; o++) {
      const ob = o * Lo; out.fill(b.d[o], ob, ob + Lo);
      for (let c = 0; c < Ci; c++) {
        const xb = c * L, wb = (o * Ci + c) * K;
        for (let k = 0; k < K; k++) {
          const wv = W[wb + k], off = k - p;
          for (let l = 0; l < Lo; l++) { const j = l * stride + off; if (j >= 0 && j < L) out[ob + l] += wv * x[xb + j]; }
        }
      }
    }
    return [out, O, Lo];
  }
  function dense(w, b, x, act) {
    const [O, I] = w.s, y = new Float32Array(O);
    for (let o = 0; o < O; o++) { let s = b.d[o]; const r = o * I; for (let i = 0; i < I; i++) s += w.d[r + i] * x[i]; y[o] = act ? gelu(s) : s; }
    return y;
  }
  // raw: {v, grade(%), T, hw, rho, fuel, rpmk, battkW, soc, auxkW, valid} arrays of length n; fl: [has_fuel, fuel_est, has_rpm, has_batt, has_aux]
  function netInput(raw, fl, feat) {
    const n = Math.min(raw.v.length, 1024), T = Math.min(1024, Math.max(32, Math.ceil(n / 32) * 32)), C = feat >= 2 ? 22 : 19, x = new Float32Array(C * T);
    for (let i = 0; i < n; i++) {
      const v = raw.v[i], a = (i > 0 && i < n - 1) ? (raw.v[i + 1] - raw.v[i - 1]) / 2 : 0, va = raw.valid ? raw.valid[i] : 1;
      const f = fl[0] ? raw.fuel[i] : 0;
      x[0 * T + i] = v / 15; x[1 * T + i] = a / 1.5; x[2 * T + i] = raw.grade[i] / 3; x[3 * T + i] = (raw.T[i] - 10) / 12;
      x[4 * T + i] = raw.hw[i] / 5; x[5 * T + i] = (raw.rho[i] - 1.22) * 10; x[6 * T + i] = f * va; x[7 * T + i] = Math.log1p(Math.max(f, 0) * 4) * va;
      x[8 * T + i] = (fl[2] ? raw.rpmk[i] : 0) * va; x[9 * T + i] = (fl[3] ? raw.battkW[i] : 0) / 20 * va;
      x[10 * T + i] = ((fl[3] ? raw.soc[i] : 0) - 0.5) * 2 * fl[3]; x[11 * T + i] = (fl[4] ? raw.auxkW[i] : 0) / 2;
      x[12 * T + i] = va; x[13 * T + i] = 1;
      for (let k = 0; k < 5; k++) x[(14 + k) * T + i] = fl[k];
      if (feat >= 2) { const va2 = v + raw.hw[i]; x[19 * T + i] = v * (1.03 * a + 9.81 * raw.grade[i] / 100) / 10;
        x[20 * T + i] = 0.5 * raw.rho[i] * va2 * Math.abs(va2) * v / 2000; x[21 * T + i] = 9.81 * v / 200; }
    }
    return [x, T, n, C];
  }

  // ---- trip-level classical estimates (feature version 3), same arithmetic as features.scalars ----
  function smooth5(x) { const n = x.length, y = new Float64Array(n); for (let i = 0; i < n; i++) { let s = 0; for (let k = -2; k <= 2; k++) { const j = i + k; if (j >= 0 && j < n) s += x[j]; } y[i] = s / 5; } return y; }
  function pctl(arr, p) { const s = Float64Array.from(arr).sort(), N = s.length, pos = p / 100 * (N - 1), lo = Math.floor(pos), hi = Math.min(lo + 1, N - 1); return s[lo] + (s[hi] - s[lo]) * (pos - lo); }
  function solve(A, b) { const n = b.length, M = A.map((r, i) => [...r, b[i]]);
    for (let c = 0; c < n; c++) { let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r; [M[c], M[p]] = [M[p], M[c]];
      for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; } }
    return M.map((r, i) => r[n] / r[i]); }
  function scalars(raw, fl, n) {
    const out = new Float32Array(16), v = Float64Array.from(raw.v.slice(0, n)), ok = (i) => (raw.valid ? raw.valid[i] : 1) > 0.5;
    const a = new Float64Array(n); for (let i = 1; i < n - 1; i++) a[i] = (v[i + 1] - v[i - 1]) / 2;
    const P = new Float64Array(n); let have = false;
    if (fl[0] > 0.5) { for (let i = 0; i < n; i++) P[i] += 0.25 * raw.fuel[i] * 42600; have = true; }
    if (fl[3] > 0.5) { for (let i = 0; i < n; i++) P[i] += raw.battkW[i] * 1000; have = true; }
    if (have) {
      const g = Float64Array.from({ length: n }, (_, i) => raw.grade[i] / 100);
      const Ps = smooth5(P), vs = smooth5(v), as = smooth5(a), gs = smooth5(g), rows = [], y = [];
      for (let i = 0; i < n; i++) if (ok(i) && vs[i] > 3 && as[i] > -0.3 && Ps[i] > 0) {
        const vw = v[i] + raw.hw[i]; rows.push([vs[i] * (1.03 * as[i] + 9.81 * gs[i]), 9.81 * vs[i], 0.5 * raw.rho[i] * vw * Math.abs(vw) * vs[i], 1]); y.push(Ps[i]); }
      const N = rows.length;
      if (N >= 30) {
        const sc = [0, 1, 2, 3].map((k) => Math.sqrt(rows.reduce((s, r) => s + r[k] * r[k], 0) / N) + 1e-9);
        const A = [0, 1, 2, 3].map((p) => [0, 1, 2, 3].map((q) => rows.reduce((s, r) => s + r[p] * r[q], 0) / (sc[p] * sc[q]) + (p === q ? 1e-3 * N : 0)));
        const bb = [0, 1, 2, 3].map((p) => rows.reduce((s, r, i) => s + r[p] * y[i], 0) / sc[p]);
        const b = solve(A, bb).map((x, k) => x / sc[k]);
        let my = 0; y.forEach((x) => my += x / N); let vy = 0, mr = 0; const res = y.map((x, i) => x - rows[i].reduce((s, c, k) => s + c * b[k], 0));
        res.forEach((x) => mr += x / N); res.forEach((x) => vy += (x - mr) * (x - mr) / N); let vt = 0; y.forEach((x) => vt += (x - my) * (x - my) / N);
        out[0] = Math.asinh(b[0] / 500); out[1] = Math.asinh(b[1] / 10); out[2] = Math.asinh(b[2] / 0.3); out[3] = Math.asinh(b[3] / 2000);
        out[4] = Math.max(-1, Math.min(1, 1 - vy / Math.max(vt, 1e-9))); out[5] = Math.log1p(N) / 7;
      }
    }
    let km = 0; for (let i = 0; i < n; i++) if (ok(i)) km += v[i]; km = Math.max(km / 1000, 0.05);
    const mv = [], idleF = [], idleR = [], cr = []; for (let i = 0; i < n; i++) { if (ok(i) && v[i] > 2) mv.push(i); }
    if (fl[0] > 0.5) {
      for (let i = 0; i < n; i++) if (ok(i) && v[i] < 0.3 && (fl[2] > 0.5 ? raw.rpmk[i] > 0.3 : raw.fuel[i] > 0)) idleF.push(raw.fuel[i]);
      if (idleF.length >= 5) out[6] = pctl(idleF, 50) * 3;
      if (mv.length >= 10) out[8] = pctl(mv.map((i) => raw.fuel[i]), 90) / 2;
      let s = 0; for (let i = 0; i < n; i++) if (ok(i)) s += raw.fuel[i]; out[12] = s / km / 100;
    }
    if (fl[2] > 0.5) {
      for (let i = 0; i < n; i++) { if (ok(i) && v[i] < 0.3 && raw.rpmk[i] > 0.3) idleR.push(raw.rpmk[i]); if (ok(i) && v[i] > 11 && v[i] < 20 && raw.rpmk[i] > 0.3) cr.push(raw.rpmk[i] / v[i]); }
      if (idleR.length >= 5) out[7] = pctl(idleR, 50);
      if (cr.length >= 5) out[9] = pctl(cr, 50) * 10;
      if (mv.length >= 10) out[10] = mv.filter((i) => raw.rpmk[i] < 0.1).length / mv.length;
      out[11] = 1;
    }
    if (fl[3] > 0.5) {
      let s = 0; for (let i = 0; i < n; i++) if (ok(i)) s += raw.battkW[i]; out[13] = s / 3600 / km * 5;
      if (mv.length >= 10) { const pb = mv.map((i) => raw.battkW[i]); out[14] = pctl(pb, 90) / 50; out[15] = pctl(pb, 10) / 50; }
    }
    return out;
  }
  function encode(net, raw, fl) {
    const Tn = net.T, M = net.meta;
    let [h, T, n, C] = netInput(raw, fl, M.feat || 1), L = T;
    let mk = new Float32Array(L); for (let i = 0; i < n; i++) mk[i] = 1;
    for (let b = 0; b < M.n_blocks; b++) {
      let [y, O, Lo] = conv(h, C, L, Tn[`c${b}_w`], Tn[`c${b}_b`], M.strides[b]);
      if (b > 0) { const m2 = new Float32Array(Math.ceil(mk.length / 2)); for (let i = 0; i < m2.length; i++) m2[i] = Math.max(mk[2 * i], 2 * i + 1 < mk.length ? mk[2 * i + 1] : 0); mk = m2; }
      for (let o = 0; o < O; o++) for (let l = 0; l < Lo; l++) { const j = o * Lo + l; y[j] = gelu(y[j]) * (l < mk.length ? mk[l] : 0); }
      h = y; C = O; L = Lo;
    }
    let cnt = 0; for (let l = 0; l < L; l++) cnt += mk[l];
    const f = new Float32Array(2 * C + 2);
    for (let c = 0; c < C; c++) {
      let s = 0, mx = -1e30;
      for (let l = 0; l < L; l++) { const v = h[c * L + l] * mk[l]; s += v; const u = v + (mk[l] - 1) * 1e4; if (u > mx) mx = u; }
      f[c] = s / Math.max(cnt, 1); f[C + c] = mx;
    }
    const dur = n / 1024; f[2 * C] = dur; f[2 * C + 1] = Math.log(dur + 1e-3);
    let fin = f;
    if ((M.feat || 1) >= 3) { fin = new Float32Array(f.length + 16); fin.set(f, 0); fin.set(scalars(raw, fl, n), f.length); }
    return dense(Tn.trip1_w, Tn.trip1_b, dense(Tn.trip0_w, Tn.trip0_b, fin, true), false);
  }
  const LED = [200, 150, 150, 150, 1500, 150];
  function heads(net, E) {
    const Tn = net.T, D = E[0].length, z0 = new Float32Array(D + 1);
    for (const e of E) for (let i = 0; i < D; i++) z0[i] += e[i] / E.length;
    z0[D] = Math.log(E.length);
    const z = dense(Tn.car1_w, Tn.car1_b, dense(Tn.car0_w, Tn.car0_b, z0, true), true);
    const sm = (u) => { const m = Math.max(...u), e = Array.from(u, (x) => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); };
    const arch = sm(dense(Tn.arch_w, Tn.arch_b, z, false)), strat = sm(dense(Tn.strat_w, Tn.strat_b, z, false));
    const r = dense(Tn.reg_w, Tn.reg_b, z, false), nr = r.length / 2;
    const mu = Array.from(r.slice(0, nr)), sig = Array.from(r.slice(nr), (x) => Math.exp(Math.min(3, Math.max(-6, x))));
    const led = E.map((e) => {
      const u = new Float32Array(D * 2); u.set(e, 0); u.set(z, D);
      const o = dense(Tn.led1_w, Tn.led1_b, dense(Tn.led0_w, Tn.led0_b, u, true), false);
      return { mu: LED.map((s, i) => o[i] * s), sig: LED.map((s, i) => Math.exp(Math.min(3, Math.max(-6, o[6 + i]))) * s) };
    });
    return { arch, strat, mu, sig, led };
  }
  function phys(net, h, nTrips) {
    const M = net.meta, k = M.sigk || {}, km = nTrips > 1 ? (k.mass_car || 1) : (k.mass_trip || 1), kd = nTrips > 1 ? (k.disp_car || 1) : (k.disp_trip || 1);
    const m = 1800 * Math.exp(h.mu[0]), d = 2.5 * Math.exp(h.mu[1]);
    return {
      arch: M.archs.map((a, i) => [a, h.arch[i]]).sort((a, b) => b[1] - a[1]), strat: M.strats.map((a, i) => [a, h.strat[i]]).sort((a, b) => b[1] - a[1]),
      mass: m, massLo: m * Math.exp(-h.sig[0] * km), massHi: m * Math.exp(h.sig[0] * km),
      disp: d, dispLo: d * Math.exp(-h.sig[1] * kd), dispHi: d * Math.exp(h.sig[1] * kd),
      CdA: 0.75 * Math.exp(h.mu[2]), CdAsig: h.sig[2], Crr: 0.008 * Math.exp(h.mu[3]), Crrsig: h.sig[3],
      evShare: h.mu[4], Pon: Math.expm1(h.mu[5] + 2), vEv: (h.mu[6] + 1.5) * 10, led: h.led,
    };
  }
  return { load, encode, heads, phys };
})();
