// BatteryPilot in the browser: causal features, the value-curve network and the offer-curve battery physics.
// Mirrors src/features.py, src/model.py and src/battery.py (checked JS == Python, see web/jscheck.json).
(function (root) {
  'use strict';
  const H_DA = 36, H_RT = 24, H_DF = 24, PUBLISH = 14;
  const KNOTS = [0, 0.25, 0.5, 0.75, 1.0];

  function battery(duration, opt) {
    opt = opt || {};
    const P = 1.0, E = P * duration, eta = opt.eta_rt || 0.85;
    const lo = 0.05, hi = 0.95, Enom = E / (hi - lo);
    const cost = opt.cost_per_kWh === undefined ? 130 : opt.cost_per_kWh;
    const N100 = 6000, k = 1.3, Ea = 31700, b = 370.3, kappa = 0.25, R = 8.314;
    const crate = P / Enom;
    const tf = (T) => {
      const Tc = 25 + kappa * (T - 25) + 3 * crate;
      const f = Math.exp(-(Ea - b * crate) / (R * (Tc + 273.15)));
      const f0 = Math.exp(-(Ea - b * 0.5) / (R * (25 + 1.5 + 273.15)));
      return f / f0;
    };
    const phi = (d) => Math.pow(d, k) / N100;
    const etac = Math.sqrt(eta), etad = Math.sqrt(eta);
    const perCycle = 1000 * cost * Enom * phi(E / Enom);
    return { P, E, d: duration, eta, etac, etad, Enom, cost, phi, tf, lo, hi,
      clin: (T) => cost > 0 ? perCycle * tf(T) / (E * etad) : 0 };
  }

  function rollingScale(da) {
    const n = da.length, out = new Float64Array(n);
    let last = 25.0;
    for (let t = 0; t < n; t++) {
      if (t % 24 === 0) {
        const w = [];
        for (let i = Math.max(0, t - 168); i < t; i++) if (isFinite(da[i])) w.push(da[i]);
        if (t - Math.max(0, t - 168) >= 24 && w.length) last = median(w);
        else { const w2 = []; for (let i = t; i < Math.min(n, t + 24); i++) if (isFinite(da[i])) w2.push(da[i]); if (w2.length) last = median(w2); }
      }
      out[t] = Math.max(last, 5.0);
    }
    return out;
  }
  function median(a) { const s = Float64Array.from(a).sort(); const m = s.length >> 1; return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]); }
  function rollMax(x, w) { const n = x.length, out = new Float64Array(n); for (let t = 0; t < n; t++) { let m = -Infinity; for (let i = Math.max(0, t - w); i <= t; i++) if (x[i] > m) m = x[i]; out[t] = m; } return out; }

  // features for hour t (same order as features.build)
  function featuresAt(p, t, dur, cl, ps, dmax, out) {
    const n = p.rt.length; let j = 0;
    const cl_ = (i) => Math.min(Math.max(i, 0), n - 1);
    const known = new Array(H_DA);
    for (let o = 0; o < H_DA; o++) {
      const i = cl_(t + o), dd = p.day[i] - p.day[t];
      known[o] = (dd <= 0) || (dd === 1 && p.hour[t] >= PUBLISH);
      out[j++] = known[o] ? Math.asinh(p.da[i] / ps) : 0;
    }
    for (let o = 0; o < H_DA; o++) out[j++] = known[o] ? 1 : 0;
    for (let o = -H_RT; o < 0; o++) out[j++] = Math.asinh(p.rt[cl_(t + o)] / ps);
    for (let o = -6; o < 0; o++) { const i = cl_(t + o); out[j++] = Math.asinh(p.rt[i] / ps) - Math.asinh(p.da[i] / ps); }
    for (let o = 0; o < H_DF; o++) out[j++] = p.df[cl_(t + o)] / Math.max(dmax, 1) - 0.8;
    out[j++] = (p.T[cl_(t - 1)] - 20) / 10;
    const hh = 2 * Math.PI * p.hour[t] / 24, dy = 2 * Math.PI * p.doy[t] / 365.25;
    out[j++] = Math.sin(hh); out[j++] = Math.cos(hh); out[j++] = Math.sin(dy); out[j++] = Math.cos(dy);
    out[j++] = p.dow[t] >= 5 ? 1 : 0;
    out[j++] = Math.log2(dur); out[j++] = cl / ps; out[j++] = Math.log(ps / 30);
    for (let i = 0; i < j; i++) if (!isFinite(out[i])) out[i] = 0;
    return j;
  }

  function net(W) {
    // W: {x_mu, x_sd, layers: [{W: Float32Array(out*in), b, nin, nout}]}
    const sizes = W.layers.map(l => l.nout);
    const bufs = sizes.map(s => new Float64Array(s));
    return function (x) {
      let h = Float64Array.from(x, (v, i) => (v - W.x_mu[i]) / W.x_sd[i]);
      for (let L = 0; L < W.layers.length; L++) {
        const l = W.layers[L], o = bufs[L];
        for (let r = 0; r < l.nout; r++) {
          let s = l.b[r]; const off = r * l.nin;
          for (let c = 0; c < l.nin; c++) s += l.W[off + c] * h[c];
          o[r] = L < W.layers.length - 1 ? s / (1 + Math.exp(-s)) : s;
        }
        h = o;
      }
      const z = new Float64Array(KNOTS.length); z[0] = h[0];
      for (let k = 1; k < KNOTS.length; k++) { const sp = h[k] > 30 ? h[k] : Math.log1p(Math.exp(h[k])); z[k] = z[k - 1] - sp - 0.01; }
      for (let k = 0; k < z.length; k++) z[k] = Math.min(Math.max(z[k], -12), 12);
      return z;
    };
  }

  function sStar(v, y) {
    let s = 0;
    for (let k = 0; k < v.length - 1; k++) {
      const a = v[k], b = v[k + 1];
      const f = Math.min(Math.max((a - y) / Math.max(a - b, 1e-9), 0), 1);
      s += f * (KNOTS[k + 1] - KNOTS[k]);
    }
    return s;
  }

  function stepOffer(bat, soc, v, price, cl) {
    const yd = bat.etad * (price - cl), yc = price / bat.etac;
    const shi = sStar(v, yd), slo = sStar(v, yc);
    let tgt = Math.min(Math.max(soc, slo), Math.max(shi, slo));
    tgt = Math.min(Math.max(tgt, soc - bat.P / (bat.etad * bat.E)), soc + bat.P * bat.etac / bat.E);
    tgt = Math.min(Math.max(tgt, 0), 1);
    const d = tgt - soc;
    return [tgt, d > 0 ? d * bat.E / bat.etac : 0, d < 0 ? -d * bat.E * bat.etad : 0];
  }

  // Run BatteryPilot over hours [t0, t1) of a path (features use the whole path's history).
  function runPolicy(W, p, bat, t0, t1, soc0) {
    const f = net(W), ps = p._ps || (p._ps = rollingScale(p.da)), dm = p._dmax || (p._dmax = rollMax(p.df, 720));
    const x = new Float64Array(W.x_mu.length);
    const n = t1 - t0, soc = new Float64Array(n + 1), ch = new Float64Array(n), dis = new Float64Array(n);
    const V = [];
    soc[0] = soc0 || 0;
    for (let i = 0; i < n; i++) {
      const t = t0 + i, cl = bat.clin(p.T[t]);
      featuresAt(p, t, bat.d, cl, ps[t], dm[t], x);
      const z = f(x), v = Array.from(z, q => Math.sinh(q) * ps[t]);
      V.push(v);
      const r = stepOffer(bat, soc[i], v, p.rt[t], cl);
      soc[i + 1] = r[0]; ch[i] = r[1]; dis[i] = r[2];
    }
    return { soc, ch, dis, V };
  }

  function dailySpread(p, bat, t0, t1) {
    const n = t1 - t0, ch = new Float64Array(n), dis = new Float64Array(n), k = Math.round(bat.d);
    let i = 0;
    while (i < n) {
      let j = i; while (j < n && p.day[t0 + j] === p.day[t0 + i]) j++;
      const idx = []; for (let q = i; q < j; q++) idx.push(q);
      if (idx.length >= 2 * k) {
        const da = idx.map(q => p.da[t0 + q]);
        const ord = idx.map((_, q) => q).sort((a, b) => da[a] - da[b] || a - b);
        const cheap = ord.slice(0, k).sort((a, b) => a - b), mx = cheap[cheap.length - 1];
        const later = idx.map((_, q) => q).sort((a, b) => da[b] - da[a] || a - b).filter(q => q > mx).slice(0, k);
        if (later.length === k) {
          const out = k * bat.P * bat.etac * bat.etad, cl = bat.clin(p.T[t0 + i]);
          const mL = later.reduce((s, q) => s + da[q], 0) / k, mC = cheap.reduce((s, q) => s + da[q], 0) / k;
          if (mL * out - cl * out > mC * k * bat.P) {
            cheap.forEach(q => ch[i + q] = bat.P); later.forEach(q => dis[i + q] = Math.min(out / k, bat.P));
          }
        }
      }
      i = j;
    }
    return socFrom(bat, ch, dis);
  }

  function threshold(p, bat, t0, t1, lo, hi) {
    const n = t1 - t0, ch = new Float64Array(n), dis = new Float64Array(n);
    let s = 0;
    for (let i = 0; i < n; i++) {
      const pr = p.rt[t0 + i];
      if (pr < lo && s < 1) { const q = Math.min(bat.P, (1 - s) * bat.E / bat.etac); ch[i] = q; s += q * bat.etac / bat.E; }
      else if (pr > hi && s > 0) { const q = Math.min(bat.P, s * bat.E * bat.etad); dis[i] = q; s -= q / (bat.etad * bat.E); }
      s = Math.min(Math.max(s, 0), 1);
    }
    return socFrom(bat, ch, dis);
  }

  function socFrom(bat, ch, dis) {
    const n = ch.length, soc = new Float64Array(n + 1);
    for (let i = 0; i < n; i++) soc[i + 1] = Math.min(Math.max(soc[i] + (bat.etac * ch[i] - dis[i] / bat.etad) / bat.E, 0), 1);
    return { soc, ch, dis };
  }

  // rainflow degradation cost of a SOC trace (fraction of usable)
  function rainflowCost(bat, soc, T) {
    const x = Array.from(soc, s => bat.lo + s * (bat.hi - bat.lo));
    const tp = [], ti = [];
    let last = 0;
    for (let i = 0; i < x.length; i++) {
      if (i === 0 || i === x.length - 1) { tp.push(x[i]); ti.push(i); continue; }
      const d0 = x[i] - x[i - 1], d1 = x[i + 1] - x[i];
      if (d0 !== 0) last = d0;
      if (last * d1 < 0) { tp.push(x[i]); ti.push(i); }
    }
    const st = []; let cost = 0;
    const fT = (a, b) => { // throughput-weighted temperature factor between indices
      let w = 0, s = 0; for (let i = Math.min(a, b); i < Math.max(a, b) && i < T.length; i++) { const d = Math.abs(x[i + 1] - x[i]); w += d; s += d * bat.tf(T[i]); }
      return w > 1e-12 ? s / w : 1;
    };
    const add = (r, c, a, b) => { if (r > 0) cost += c * bat.phi(r) * fT(a, b) * 1000 * bat.cost * bat.Enom; };
    for (let q = 0; q < tp.length; q++) {
      st.push([tp[q], ti[q]]);
      while (st.length >= 3) {
        const X = Math.abs(st[st.length - 1][0] - st[st.length - 2][0]);
        const Y = Math.abs(st[st.length - 2][0] - st[st.length - 3][0]);
        if (X < Y) break;
        if (st.length === 3) { add(Y, 0.5, st[0][1], st[1][1]); st.shift(); }
        else { add(Y, 1.0, st[st.length - 3][1], st[st.length - 2][1]); const l = st.pop(); st.pop(); st.pop(); st.push(l); }
      }
    }
    for (let q = 0; q < st.length - 1; q++) add(Math.abs(st[q + 1][0] - st[q][0]), 0.5, st[q][1], st[q + 1][1]);
    return cost;
  }

  root.BP = { battery, runPolicy, dailySpread, threshold, rainflowCost, rollingScale, featuresAt, net, stepOffer, socFrom };
})(typeof window !== 'undefined' ? window : globalThis);
