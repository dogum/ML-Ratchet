// FlowCast in the browser: the same arithmetic as release/flowcast.py (LSTM, 365-day warm-up blocks).
const FCNet = (() => {
  const flat = (a) => Float64Array.from(a.flat(Infinity));
  function load(W) {
    const H = W.w_hh[0].length, nin = W.w_ih[0].length;
    return { H, nin, wih: flat(W.w_ih), whh: flat(W.w_hh), b: flat(W.b_ih).map((v, i) => v + W.b_hh[i]),
      hw: flat(W.head_w), hb: flat(W.head_b), mu: flat(W.st_mu), sd: flat(W.st_sd), k: W.k_widen };
  }
  const sig = (x) => 1 / (1 + Math.exp(-x));
  const sp = (x) => (x > 30 ? x : Math.log1p(Math.exp(x)));
  function climate(F) {   // F: array of [P, PET, T_mean, T_min, T_max, SRAD, vp, wind]
    let sp_ = 0, se = 0, ss = 0, st = 0, hp = 0, lp = 0; const n = F.length;
    for (const r of F) { sp_ += r[0]; se += r[1]; st += r[2]; if (r[2] < 0) ss += r[0]; }
    const pm = Math.max(sp_ / n, 0.05), em = Math.max(se / n, 0.05);
    for (const r of F) { if (r[0] > 5 * pm) hp++; if (r[0] < 1) lp++; }
    return { p_mean: pm, pet_mean: em, aridity: em / pm, frac_snow: ss / Math.max(sp_, 1e-6), hp_freq: hp / n, lp_freq: lp / n, t_mean: st / n };
  }
  function staticVec(a, c) {
    return [Math.log10(Math.max(a.area_km2, 1)), a.elev_mean_m / 1000, Math.max(a.elev_max_m - a.elev_min_m, 0) / 1000,
      Math.log10(Math.max(a.slope_mkm, 0.5)), a.forest_frac, a.soil_porosity, Math.log10(Math.max(a.ksat_cmh, 0.01)),
      a.carbonate_frac, Math.min(a.reservoir_storage_frac, 2), a.irrigation_frac, a.urban_frac,
      Math.log(c.p_mean), c.pet_mean, Math.log(c.aridity), c.frac_snow, c.hp_freq, c.lp_freq, c.t_mean / 10];
  }
  function dyn(r, doy) {
    const P = Math.max(r[0], 0), w = 2 * Math.PI * doy / 365.25;
    return [P / 10, Math.log1p(P), r[1] / 5, r[2] / 10, r[3] / 10, r[4] / 10, r[5] / 150, r[6] / 1000, r[7] / 3, Math.sin(w), Math.cos(w)];
  }
  // run one block: rows (L x 8), doys; returns outputs for the last nOut steps
  function block(N, rows, doys, s, nOut) {
    const { H, nin, wih, whh, b, hw, hb } = N, L = rows.length, G = 4 * H;
    let h = new Float64Array(H), c = new Float64Array(H); const g = new Float64Array(G), z = new Float64Array(nin);
    for (let j = 0; j < s.length; j++) z[nin - s.length + j] = s[j];
    const mean = [], q = [];
    for (let t = 0; t < L; t++) {
      const d = dyn(rows[t], doys[t]); for (let j = 0; j < d.length; j++) z[j] = d[j];
      for (let r = 0; r < G; r++) {
        let a = b[r]; const o1 = r * nin, o2 = r * H;
        for (let j = 0; j < nin; j++) a += wih[o1 + j] * z[j];
        for (let j = 0; j < H; j++) a += whh[o2 + j] * h[j];
        g[r] = a;
      }
      const hn = new Float64Array(H);
      for (let j = 0; j < H; j++) {
        const i = sig(g[j]), f = sig(g[H + j]), gg = Math.tanh(g[2 * H + j]), o = sig(g[3 * H + j]);
        c[j] = f * c[j] + i * gg; hn[j] = o * Math.tanh(c[j]);
      }
      h = hn;
      if (t >= L - nOut) {
        const out = new Array(6);
        for (let k = 0; k < 6; k++) { let a = hb[k]; for (let j = 0; j < H; j++) a += hw[k * H + j] * h[j]; out[k] = a; }
        mean.push(sp(out[0]));
        let acc = sp(out[1]) * 0.2; const qq = [acc];
        for (let k = 2; k < 6; k++) { acc += sp(out[k]); qq.push(acc); }
        q.push(qq);
      }
    }
    return { mean, q };
  }
  // F: rows from 365 days before the first target day; doy: day-of-year of each row; attrs in sim units
  function predict(N, F, doy, attrs, nTarget) {
    const clim = climate(F.slice(0, 365 + nTarget + 1));
    const s0 = staticVec(attrs, clim), s = s0.map((v, j) => (v - N.mu[j]) / N.sd[j]);
    const mean = [], q = [];
    for (let a = 0; a < nTarget; a += 365) {
      const b = Math.min(a + 365, nTarget), L = 365 + 365;
      const rows = [], doys = [];
      for (let t = 0; t < L; t++) { const i = Math.min(a + t, a + 365 + (b - a) - 1); rows.push(F[i]); doys.push(((doy[a] - 1 + t) % 365) + 1); }
      const r = block(N, rows, doys, s, 365);
      for (let t = 0; t < b - a; t++) { mean.push(r.mean[t] * clim.p_mean); q.push(r.q[t].map((v) => v * clim.p_mean)); }
    }
    // widen-only calibration about the median
    const [k50, k90] = N.k;
    for (const r of q) { const m = r[2];
      r[1] = Math.max(m - k50 * (m - r[1]), 0); r[3] = m + k50 * (r[3] - m); r[0] = Math.max(m - k90 * (m - r[0]), 0); r[4] = m + k90 * (r[4] - m); }
    return { mean, q, clim };
  }
  return { load, predict, climate };
})();
if (typeof module !== 'undefined') module.exports = FCNet;
