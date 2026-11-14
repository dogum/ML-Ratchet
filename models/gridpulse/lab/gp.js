// GridPulse in the browser: (1) a port of gridsim.frequency_event for one area (governors with deadband, lag, rate
// limit, headroom clip and two lead-lags; FFR steps; load damping; UFLS; secondary), (2) the GridPulse MLP.
// Checked against Python (check_js.py -> checks/js_parity.json).
(function (root) {
  const TECH = { // [H default unused, R, T1, LL_A(n,d), LL_B(n,d), rate, resp, head] from gridsim TECH_DYN
    nuclear: [0.05, 0.3, 2.1, 7.0, 0, 0, 0.05, false, 0.0], coal_steam: [0.05, 0.3, 2.1, 7.0, 0, 0, 0.02, true, 0.06],
    gas_steam: [0.05, 0.3, 2.1, 7.0, 0, 0, 0.03, true, 0.06], biomass: [0.05, 0.3, 2.1, 7.0, 0, 0, 0.02, true, 0.05],
    gas_cc: [0.05, 0.5, 26.8, 40.0, 0, 0, 0.10, true, 0.08], gas_ct: [0.05, 0.5, 0, 0, 0, 0, 0.15, true, 0.10],
    hydro_res: [0.05, 0.5, 5.0, 38.0, -1.5, 0.75, 0.10, true, 0.15], battery: [0.01, 0.2, 0, 0, 0, 0, 5.0, true, 1.0],
    wind: [0.05, 0.3, 0, 0, 0, 0, 1.0, false, 0.0]
  };
  // compile a state built like lab.py (units: {tech, MW, pmax_MW, pmin_MW, gov:{droop, deadband_Hz, headroom_frac,
  // enabled, up_MW, dn_MW, T_s}})
  function compile(units, f0) {
    const n = units.length, P = { K: [], db: [], T1: [], rate: [], up: [], dn: [], An: [], Ad: [], Bn: [], Bd: [] };
    for (const u of units) {
      const d = TECH[u.tech], g = u.gov || {};
      const en = g.enabled !== undefined ? g.enabled : d[7];
      const R = g.droop || d[0], pmax = u.pmax_MW, p = u.MW, pmin = u.pmin_MW || 0;
      const head = g.headroom_frac !== undefined ? g.headroom_frac : d[8];
      P.K.push(en ? pmax / (R * f0) : 0); P.db.push(g.deadband_Hz); P.T1.push(g.T_s || d[1]);
      P.rate.push(d[6] * pmax);
      let up = en ? Math.max(0, Math.min(pmax - p, head * pmax)) : 0;
      let dn = en ? Math.max(0, Math.min(p - pmin, Math.max(head, 0.1) * pmax)) : 0;
      if (en && g.up_MW !== undefined) up = g.up_MW; if (en && g.dn_MW !== undefined) dn = g.dn_MW;
      P.up.push(up); P.dn.push(dn); P.An.push(d[2]); P.Ad.push(d[3]); P.Bn.push(d[4]); P.Bd.push(d[5]);
    }
    return P;
  }
  // st: {f0, E, D, load, P (compiled), ffr:[{MW,trigger_Hz,delay_s,ramp_s}], ufls:[{f_Hz,frac,delay_s}],
  //      secondary:{MW,T_s,deadband_Hz}}; losses: [[t, MW], ...]
  function simulate(st, losses, tEnd = 68, dt = 0.1, h = 0.02) {
    const P = st.P, n = P.K.length, f0 = st.f0, E = Math.max(st.E, 1);
    const lag = T => T > 0 ? 1 - Math.exp(-h / T) : 1;
    const a1 = P.T1.map(lag), aA = P.Ad.map(lag), aB = P.Bd.map(lag);
    const rA = P.Ad.map((d, i) => d > 0 ? P.An[i] / d : 1), rB = P.Bd.map((d, i) => d > 0 ? P.Bn[i] / d : 1);
    const x1 = new Float64Array(n), xa = new Float64Array(n), xb = new Float64Array(n);
    const F = st.ffr || [], fT = F.map(() => Infinity), fOn = F.map(() => Infinity);
    const U = st.ufls || [], uT = U.map(() => Infinity), uTrip = U.map(() => Infinity);
    const sec = st.secondary || null, aS = sec ? 1 - Math.exp(-h / Math.max(sec.T_s, h)) : 0;
    let df = 0, shed = 0, secx = 0;
    const nsteps = Math.round(tEnd / h), every = Math.max(1, Math.round(dt / h));
    const t = [], f = [];
    for (let k = 0; k <= nsteps; k++) {
      const tt = k * h;
      let lost = 0; for (const L of losses) if (L[0] <= tt + 1e-12) lost += L[1];
      let gen = 0;
      for (let i = 0; i < n; i++) {
        const dfd = Math.sign(df) * Math.max(Math.abs(df) - P.db[i], 0);
        const rh = P.rate[i] * h;
        const dx = Math.min(Math.max((-P.K[i] * dfd - x1[i]) * a1[i], -rh), rh);
        x1[i] = Math.min(Math.max(x1[i] + dx, -P.dn[i]), P.up[i]);
        let g = x1[i];
        if (P.Ad[i] > 0) { const ya = rA[i] * g + (1 - rA[i]) * xa[i]; xa[i] += (g - xa[i]) * aA[i]; g = ya; }
        if (P.Bd[i] > 0) { const yb = rB[i] * g + (1 - rB[i]) * xb[i]; xb[i] += (g - xb[i]) * aB[i]; g = yb; }
        gen += g;
      }
      let pffr = 0;
      F.forEach((x, j) => {
        const below = df <= x.trigger_Hz - f0;
        if (below && !isFinite(fT[j])) fT[j] = tt; else if (!(below || isFinite(fOn[j]))) fT[j] = Infinity;
        if (isFinite(fT[j]) && !isFinite(fOn[j]) && tt - fT[j] >= x.delay_s - 1e-12) fOn[j] = tt;
        if (isFinite(fOn[j])) pffr += x.MW * Math.min(Math.max((tt - fOn[j]) / Math.max(x.ramp_s, 1e-3), 0), 1);
      });
      U.forEach((x, j) => {
        if (isFinite(uTrip[j])) return;
        const c = df <= x.f_Hz - f0;
        if (c && !isFinite(uT[j])) uT[j] = tt; else if (!c) uT[j] = Infinity;
        if (isFinite(uT[j]) && tt - uT[j] >= x.delay_s - 1e-12) { uTrip[j] = tt; shed += (x.MW !== undefined ? x.MW : x.frac * st.load); }
      });
      if (sec && Math.abs(df) > (sec.deadband_Hz || 0)) secx += aS * (-Math.sign(df) * sec.MW - secx);
      const damp = -st.D / f0 * Math.max(st.load - shed, 0) * df;
      const bal = gen + pffr + shed + damp - lost + secx;
      const dfdt = f0 / 2 / E * bal;
      if (k % every === 0) { t.push(tt); f.push(f0 + df); }
      df += h * dfdt;
    }
    return { t, f };
  }
  // ---- GridPulse MLP (weights exported by export.py)
  function erf(x) { // W. J. Cody-quality rational approximation via erfc continued (|err| < 1.2e-7)
    const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
      t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? 1 - r : r - 1;
  }
  function dense(W, b, x, nin, nout, act) {
    const y = new Float64Array(nout);
    for (let o = 0; o < nout; o++) { let s = b[o]; const off = o * nin; for (let i = 0; i < nin; i++) s += W[off + i] * x[i];
      y[o] = act ? 0.5 * s * (1 + erf(s / Math.SQRT2)) : s; }
    return y;
  }
  function features(w, N, isGB, f0, demand, ctx) { // w: 81 deviations [Hz], t = -20..60
    const x = []; for (let i = 0; i < 81; i++) x.push((i - 20) <= N ? w[i] * (50 / f0) * 10 : 0);
    x.push(isGB ? 1 : 0, N / 60, Math.log(N) / 4);
    if (ctx) { const k = demand !== null && demand !== undefined && isFinite(demand); x.push(k ? Math.log(demand / 30000) : 0, k ? 1 : 0); }
    return x;
  }
  function f32(s) { if (typeof s !== 'string') return s; const b = atob(s), u = new Uint8Array(b.length);
    for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new Float32Array(u.buffer); }
  function prep(M) { if (M._ok) return M; for (const L of M.layers) L.w = f32(L.w); M.mu_w = f32(M.mu_w); M.ls_w = f32(M.ls_w); M._ok = true; return M; }
  function predict(M, w, N, isGB, f0, demand) {
    prep(M);
    const x0 = features(w, N, isGB, f0, demand, M.ctx); let h = x0.map((v, i) => (v - M.xm[i]) / M.xs[i]);
    let nin = h.length;
    for (const L of M.layers) { h = dense(L.w, L.b, h, nin, L.n, true); nin = L.n; }
    const mu = dense(M.mu_w, M.mu_b, h, nin, M.nout, false), ls = dense(M.ls_w, M.ls_b, h, nin, M.nout, false);
    const out = { mu: [], sg: [] };
    for (let i = 0; i < M.nout; i++) { out.mu.push(mu[i] * M.ys[i] + M.ym[i]); out.sg.push(Math.exp(Math.min(Math.max(ls[i], -6), 3)) * M.ys[i] * M.widen[i]); }
    return out;
  }
  function readout(M, w, N, isGB, f0, demand) {
    const p = predict(M, w, N, isGB, f0, demand), e = Math.exp, k = f0 / 50 / 10;
    const r = {};
    [['loss_MW', 0, 1], ['E_GWs', 1, 1e-3], ['beta_MW_Hz', 2, 1], ['head_MW', 3, 1]].forEach(([nm, i, sc]) => {
      r[nm] = e(p.mu[i]) * sc; r[nm + '_lo'] = e(p.mu[i] - p.sg[i]) * sc; r[nm + '_hi'] = e(p.mu[i] + p.sg[i]) * sc; });
    r.ffr_MW = e(p.mu[4]) - 100; r.ffr_MW_lo = e(p.mu[4] - p.sg[4]) - 100; r.ffr_MW_hi = e(p.mu[4] + p.sg[4]) - 100;
    r.nadir_Hz = p.mu[5] * k; r.nadir_sig = p.sg[5] * k; r.settle_Hz = p.mu[6] * k; r.settle_sig = p.sg[6] * k;
    return r;
  }
  // lab system from sliders (mirror of src/lab.py)
  const MIX = { GB: [['gas_cc', 0.55], ['nuclear', 0.25], ['biomass', 0.12], ['hydro_res', 0.03], ['gas_ct', 0.05]],
    ERCO: [['gas_cc', 0.45], ['coal_steam', 0.18], ['nuclear', 0.12], ['gas_ct', 0.15], ['gas_steam', 0.10]] };
  const UFLS = { GB: [[48.8, .05], [48.75, .05], [48.7, .10], [48.6, .075], [48.5, .075], [48.4, .075], [48.2, .075], [48.0, .05]],
    ERCO: [[59.3, .05], [58.9, .10], [58.5, .10]] };
  function labState(system, loss, inertiaGWs, ibr, fastMW, demand) {
    const f0 = system === 'GB' ? 50 : 60, dem = demand || (system === 'GB' ? 30000 : 50000), sync = dem * (1 - ibr);
    const units = [];
    for (const [tech, sh] of MIX[system]) {
      const P = sync * sh;
      units.push({ tech, MW: 0.85 * P, pmax_MW: P, pmin_MW: 0.4 * P, gov: { droop: 0.05, headroom_frac: tech !== 'nuclear' ? 0.06 : 0,
        enabled: tech !== 'nuclear', deadband_Hz: system === 'GB' ? 0.015 : ((tech === 'coal_steam' || tech === 'gas_steam') ? 0.034 : 0.0166) } });
    }
    const ffr = [];
    if (system === 'GB' && fastMW > 0) for (const [MW, db, full] of [[0.05 * fastMW, 0.015, 0.2], [0.95 * fastMW, 0.2, 0.5]])
      units.push({ tech: 'battery', MW: 0, pmax_MW: MW, pmin_MW: -MW, gov: { enabled: true, droop: (full - db) / f0, deadband_Hz: db, up_MW: MW, dn_MW: MW, T_s: 0.5 } });
    if (system === 'ERCO') { if (fastMW > 0) ffr.push({ MW: fastMW, trigger_Hz: 59.85, delay_s: 0.25, ramp_s: 0.1 });
      ffr.push({ MW: 1000, trigger_Hz: 59.7, delay_s: 0.4, ramp_s: 0.05 }); }
    const ufls = UFLS[system].map(([f, fr]) => ({ f_Hz: f, frac: fr, delay_s: system === 'GB' ? 0.2 : 0.3 }));
    return { f0, E: inertiaGWs * 1e3, D: system === 'GB' ? 0.75 : 1.0, load: dem, P: compile(units, f0), ffr, ufls,
      secondary: { MW: 0.15 * loss, T_s: 60, deadband_Hz: 0.015 } };
  }
  root.GP = { TECH, compile, simulate, predict, readout, features, labState };
  if (typeof module !== 'undefined') module.exports = root.GP;
})(typeof window !== 'undefined' ? window : globalThis);
