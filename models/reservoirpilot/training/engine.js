// ReservoirPilot in the browser: features, network, B's mass balance, SOP and the perfect-foresight DP.
// Mirrors release/reservoirpilot.py and WaterData's run_baselines_reservoir.py (checked by web/check_js.py).
const RP = (() => {
  const K = 0.0864, TAUS = [5, 20, 60];
  function fwdMean(curve, w) {            // mean of a 366-day curve over doy..doy+w-1, for every doy (wrapping)
    const c = curve.concat(curve.slice(0, w)), cs = [0];
    for (let i = 0; i < c.length; i++) cs.push(cs[i] + c[i]);
    const out = new Float64Array(366);
    for (let i = 0; i < 366; i++) out[i] = (cs[i + w] - cs[i]) / w;
    return out;
  }
  function calendar(task) {
    const n = task.I.length, Dc = task.Dcurve, Ic = task.Icurve;
    const d30 = fwdMean(Dc, 30), d90 = fwdMean(Dc, 90), i30 = fwdMean(Ic, 30), i90 = fwdMean(Ic, 90);
    const cal = { D: new Float64Array(n), Dn30: new Float64Array(n), Dn90: new Float64Array(n), Ic30: new Float64Array(n), Ic90: new Float64Array(n) };
    for (let t = 0; t < n; t++) {
      const j = task.doy[t] - 1;
      cal.D[t] = Dc[j]; cal.Dn30[t] = d30[j]; cal.Dn90[t] = d90[j]; cal.Ic30[t] = i30[j]; cal.Ic90[t] = i90[j];
    }
    return cal;
  }
  function features(S, I, e, D, Dn30, Dn90, Ic30, Ic90, q, cap, fset) {
    const qK = q * K, as = Math.asinh;
    const f = [S / cap, as(I / q), as(e[0] / q), as(e[1] / q), as(e[2] / q), D / q, Dn30 / q, Math.log(cap / qK) / 3,
      as((cap - S) / qK) / 3, as((S + I * K - D * K - cap) / qK), as(S / (Dn30 * K + 1e-3 * qK) / 10)];
    if (fset === 'clim') f.push(as(Ic30 / q), as(Ic90 / q), as((S + 90 * K * (Ic90 - Dn90) - cap) / qK) / 3,
      as((S + 30 * K * (Ic30 - Dn30)) / qK) / 3);
    return f;
  }
  function mlp(net, x) {
    let h = x.map((v, i) => (v - net.mu[i]) / net.sd[i]);
    for (let l = 0; l < net.W.length; l++) {
      const W = net.W[l], b = net.b[l], out = new Array(b.length);
      for (let j = 0; j < b.length; j++) {
        let s = b[j];
        for (let i = 0; i < h.length; i++) s += h[i] * W[i][j];
        out[j] = l < net.W.length - 1 ? Math.tanh(s) : s;
      }
      h = out;
    }
    return h;
  }
  const sig = z => 1 / (1 + Math.exp(-z));
  // run a policy ('pilot' | 'sop') with B's mass balance; returns release, storage, and pilot's h, e per day
  function run(task, policy, net) {
    const n = task.I.length, I = task.I, q = task.Qmax, cap = task.cap, cal = calendar(task);
    const R = new Float64Array(n), S = new Float64Array(n), H = new Float64Array(n), E = new Float64Array(n);
    let s = task.S0;
    const e0 = net && net.fset === 'clim' ? cal.Ic30[0] : I[0];
    let ema = [e0, e0, e0];
    const qf = Math.max(q, 1e-9);
    for (let t = 0; t < n; t++) {
      const D = cal.D[t];
      let r;
      if (policy === 'sop') r = D;
      else {
        ema = ema.map((v, k) => v + (I[t] - v) / TAUS[k]);
        const o = mlp(net, features(s, I[t], ema, D, cal.Dn30[t], cal.Dn90[t], cal.Ic30[t], cal.Ic90[t], qf, cap, net.fset));
        const h = sig(o[0]), e = sig(o[1]);
        r = h * D + 1.25 * e * Math.max(qf - D, 0); H[t] = h; E[t] = e;
      }
      const avail = s + I[t] * K;
      r = Math.min(Math.max(r, 0), Math.max(avail, 0) / K);
      let s1 = avail - r * K;
      if (s1 > cap) { r += (s1 - cap) / K; s1 = cap; }
      s = Math.max(s1, 0); R[t] = r; S[t] = s;
    }
    return { R, S, H, E, D: cal.D };
  }
  function interp(x, xs, ys) {            // xs uniform grid
    const n = xs.length, dx = xs[1] - xs[0];
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    const k = Math.min(Math.floor((x - xs[0]) / dx), n - 2), w = (x - xs[k]) / dx;
    return ys[k] * (1 - w) + ys[k + 1] * w;
  }
  const dayCost = (r, D, q) => { const a = Math.max(0, D - r) / Math.max(D, 1e-6), b = Math.max(0, r - q) / q; return a * a + 10 * b * b; };
  // B's perfect-foresight DP (201 storage levels, 21 + 21 release candidates, end at least as full as the start)
  function dp(task, nlev = 201, nr = 21) {
    const n = task.I.length, I = task.I, q = task.Qmax, cap = task.cap, S0 = task.S0, D = calendar(task).D;
    const lev = new Float64Array(nlev); for (let i = 0; i < nlev; i++) lev[i] = cap * i / (nlev - 1);
    let V = new Float64Array(nlev); for (let i = 0; i < nlev; i++) V[i] = 1000 * Math.max(0, S0 - lev[i]) / cap;
    const pol = new Array(n);
    for (let t = n - 1; t >= 0; t--) {
      const set = new Set();
      const hi = Math.max(q, D[t]), st1 = D[t] / (nr - 1), st2 = (hi - D[t]) / (nr - 1);   // numpy linspace arithmetic
      for (let k = 0; k < nr; k++) { set.add(k === nr - 1 ? D[t] : k * st1); set.add(k === nr - 1 ? hi : k * st2 + D[t]); }
      const Rc = Array.from(set).sort((a, b) => a - b);
      const V2 = new Float64Array(nlev), P = new Float32Array(nlev);
      for (let i = 0; i < nlev; i++) {
        const avail = lev[i] + I[t] * K; let best = Infinity, bj = 0;
        for (let j = 0; j < Rc.length; j++) {
          let r = Math.min(Rc[j], Math.max(avail, 0) / K), s1 = avail - r * K;
          const sp = Math.max(0, s1 - cap); r += sp / K; s1 = Math.min(Math.max(s1 - sp, 0), cap);
          const c = dayCost(r, D[t], q) + interp(s1, lev, V);
          if (c < best) { best = c; bj = j; }
        }
        V2[i] = best; P[i] = Rc[bj];
      }
      V = V2; pol[t] = P;
    }
    const R = new Float64Array(n), S = new Float64Array(n); let s = S0;
    for (let t = 0; t < n; t++) {
      let r = interp(s, lev, pol[t]); const avail = s + I[t] * K;
      r = Math.min(r, Math.max(avail, 0) / K); let s1 = avail - r * K;
      if (s1 > cap) { r += (s1 - cap) / K; s1 = cap; }
      R[t] = r; s = Math.max(s1, 0); S[t] = s;
    }
    return { R, S, D };
  }
  function components(R, S, D, q, cap, S0) {
    const n = R.length; let js = 0, jf = 0, sh = 0, sd = 0, fd = 0, sm = 0;
    for (let t = 0; t < n; t++) {
      const a = Math.max(0, D[t] - R[t]) / Math.max(D[t], 1e-6), b = Math.max(0, R[t] - q) / q;
      js += a * a; jf += 10 * b * b; sh += Math.max(0, D[t] - R[t]); sd += D[t]; if (R[t] > q) fd++; sm += S[t];
    }
    return { J: (js + jf) / n, J_short: js / n, J_flood: jf / n, shortage_ratio: sd > 0 ? sh / sd : 0,
      flood_days_per_yr: fd / n * 365.25, S_mean_frac: sm / n / cap, dS_end: (S[n - 1] - S0) / cap };
  }
  return { run, dp, components, calendar, K };
})();
if (typeof module !== 'undefined') module.exports = RP;
