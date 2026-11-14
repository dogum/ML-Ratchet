// PlantDetective in the browser: feature vector (port of pdcore.features), the MLP, decoding, pooling,
// plus a small unit simulator for the lab (self-commitment DP + CAMPD-style emulator).  Checked against Python.
(function (root) {
  'use strict';
  const TECHS = ['coal_steam', 'gas_cc', 'gas_ct', 'gas_steam'];
  const REG = ['pmin_frac', 'ramp_up', 'ramp_dn', 'starts_yr', 'min_up', 'med_up', 'min_dn', 'hot', 'warm', 'cold',
    'hr_full', 'hr_min', 'cf'];
  const RUN_EDGES = [1, 2, 3, 4, 7, 13, 25, 49, 97, 193, 385, 1e9];
  const ZBINS = [-1e9, 4, 6, 7, 8, 9, 10, 11, 12, 14, 17, 25, 1e9];
  const LOADBINS = Array.from({ length: 23 }, (_, i) => 1.1 * i / 22);
  const HOT_H = 8, WARM_H = 48, PART_MIN = 0.4;
  const CTX_KEYS = ['pm_ct', 'pm_st', 'pm_ca', 'fuel_coal', 'chp', 'lnp_own', 'lnp_st', 'r_own', 'r_blk', 'minload_frac',
    'minload_has', 'cold_lh', 'cold_has', 'hr923', 'hr923_has', 'age'];

  function pct(a, q) {           // numpy percentile, linear
    if (!a.length) return NaN;
    const s = Float64Array.from(a).sort();
    const pos = q / 100 * (s.length - 1), lo = Math.floor(pos), hi = Math.ceil(pos);
    return s[lo] + (s[hi] - s[lo]) * (pos - lo);
  }
  const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN;
  function hist(a, edges) {
    const h = new Array(edges.length - 1).fill(0);
    for (const v of a) {
      if (v < edges[0] || v > edges[edges.length - 1]) continue;
      let k = edges.length - 2;
      for (let i = 0; i < edges.length - 1; i++) { if (v < edges[i + 1]) { k = i; break; } }
      h[k]++;
    }
    return h;
  }
  function runs(on) {
    const s = [], L = [], v = [];
    for (let i = 0; i < on.length; i++) {
      if (i === 0 || on[i] !== on[i - 1]) { s.push(i); L.push(1); v.push(on[i]); } else L[L.length - 1]++;
    }
    return { s, L, v };
  }
  function qs(a, q, fill) { return q.map(x => { const r = pct(a, x); return isFinite(r) ? r : fill; }); }

  function features(x, hourLocal, dow, price, gas, ctx) {
    const n = x.length;
    const miss = x.map(v => !isFinite(v));
    const xv = x.map((v, i) => miss[i] ? 0 : v);
    const on = xv.map(v => v > 0.5);
    const known = miss.map(m => !m);
    const f = [];
    const inter = on.map((o, i) => o && i > 0 && on[i - 1] && i < n - 1 && on[i + 1]);
    const nInter = inter.filter(Boolean).length;
    let base = nInter >= 12 ? xv.filter((_, i) => inter[i]) : xv.filter((_, i) => on[i]);
    let pcap = base.length ? pct(base, 99.5) : 1.0; pcap = Math.max(pcap, 1.0);
    const y = xv.map(v => v / pcap);
    const nk0 = known.filter(Boolean).length;
    f.push(Math.log(n / 720), miss.filter(Boolean).length / n,
      nk0 ? on.filter((o, i) => known[i] && o).length / nk0 : 0, Math.log(pcap / 100), nInter < 12 ? 1 : 0);
    let yi = y.filter((_, i) => inter[i]); if (!yi.length) yi = [0];
    const h = hist(yi.map(v => Math.min(Math.max(v, 0), 1.0999)), LOADBINS);
    const hs = Math.max(h.reduce((a, b) => a + b, 0), 1); h.forEach(c => f.push(c / hs));
    qs(yi, [0.5, 1, 2, 5, 10, 25, 50, 75, 90, 99], 0).forEach(v => f.push(v));
    const my = mean(yi);
    f.push(yi.length > 1 ? Math.sqrt(mean(yi.map(v => (v - my) ** 2))) : 0, mean(yi.map(v => Math.abs(v - 1) < 0.03 ? 1 : 0)));
    const R = runs(on);
    const st = R.s.filter((s, i) => R.v[i] && s > 0);
    const first = st.map(s => y[s]);
    const second = st.filter(s => s + 1 < n).map(s => y[s + 1]);
    const ends = R.s.map((s, i) => [s + R.L[i] - 1, R.v[i] && s + R.L[i] < n]).filter(e => e[1]).map(e => e[0]);
    const last = ends.map(e => y[e]);
    for (const a of [first, second, last]) qs(a, [10, 50, 90], -1).forEach(v => f.push(v));
    let dy = [];
    for (let i = 0; i < n - 1; i++) if (inter[i] && inter[i + 1]) dy.push(y[i + 1] - y[i]);
    if (!dy.length) dy = [0];
    qs(dy, [0.1, 1, 5, 25, 75, 95, 99, 99.9], 0).forEach(v => f.push(v));
    f.push(mean(dy.map(d => Math.abs(d) < 0.01 ? 1 : 0)), mean(dy.map(d => Math.abs(d) > 0.1 ? 1 : 0)),
      Math.log1p(mean(dy.map(Math.abs)) * 100));
    const mm = [0]; for (let i = 0; i < n; i++) mm.push(mm[i] + (miss[i] ? 1 : 0));
    const good = R.s.map((s, i) => i > 0 && i < R.s.length - 1 && (mm[s + R.L[i]] - mm[s]) === 0);
    const up = R.L.filter((_, i) => good[i] && R.v[i]), dn = R.L.filter((_, i) => good[i] && !R.v[i]);
    const nk = Math.max(nk0, 1);
    for (const a of [up, dn]) {
      hist(a, RUN_EDGES).forEach(c => f.push(c * 1000 / nk));
      f.push(a.length ? Math.log(pct(a, 50)) : -1, a.length >= 3 ? Math.log(pct(a, 5)) : -1, a.length >= 3 ? 1 : 0);
    }
    const nst = st.length;
    f.push(Math.log1p(nst * 8760 / nk), Math.log1p(nst));
    const prev = [];
    R.s.forEach((s, i) => { if (R.v[i] && s > 0 && i - 1 > 0) prev.push(R.L[i - 1]); });
    if (prev.length) f.push(mean(prev.map(p => p < HOT_H ? 1 : 0)), mean(prev.map(p => p >= HOT_H && p < WARM_H ? 1 : 0)),
      mean(prev.map(p => p >= WARM_H ? 1 : 0)), 1);
    else f.push(0, 0, 0, 0);
    f.push(xv.reduce((a, b) => a + b, 0) / (pcap * nk));
    if (hourLocal) {
      for (let b = 0; b < 8; b++) {
        let c = 0, k = 0, c2 = 0, k2 = 0;
        for (let i = 0; i < n; i++) if (known[i] && Math.floor(hourLocal[i] / 3) === b) {
          k++; if (on[i]) c++; if (inter[i]) { k2++; c2 += y[i]; }
        }
        f.push(k ? c / k : 0, k2 ? c2 / k2 : 0);
      }
      let a1 = 0, b1 = 0, a2 = 0, b2 = 0;
      for (let i = 0; i < n; i++) if (known[i]) {
        if (dow[i] === 5 || dow[i] === 6) { b1++; if (on[i]) a1++; } else { b2++; if (on[i]) a2++; }
      }
      f.push(b1 ? a1 / b1 : 0, b2 ? a2 / b2 : 0);
    } else for (let i = 0; i < 18; i++) f.push(0);
    if (price && gas) {
      const z = price.map((p, i) => p / Math.max(gas[i], 0.5));
      const ok = z.map((v, i) => known[i] && isFinite(v));
      const zb = z.map(v => { let k = 0; for (let j = 1; j < ZBINS.length - 1; j++) if (v >= ZBINS[j]) k = j; return k; });
      for (let b = 0; b < ZBINS.length - 1; b++) {
        let cnt = 0, con = 0, ci = 0, sy = 0;
        for (let i = 0; i < n; i++) if (ok[i] && zb[i] === b) { cnt++; if (on[i]) con++; if (inter[i]) { ci++; sy += y[i]; } }
        f.push(cnt / n, cnt >= 5 ? con / cnt : -1, ci >= 5 ? sy / ci : -1);
      }
      const zo = z.filter((v, i) => ok[i] && on[i]), zf = z.filter((v, i) => ok[i] && !on[i]);
      const cl = v => Math.log(Math.min(Math.max(v, 1), 100));
      f.push(zo.length ? cl(pct(zo, 50)) : 0, zf.length ? cl(pct(zf, 50)) : 0, zo.length >= 10 ? cl(pct(zo, 10)) : 0, 1);
      const zz = [], yy = [];
      for (let i = 0; i < n; i++) if (ok[i] && inter[i]) { zz.push(z[i]); yy.push(y[i]); }
      let r = 0;
      if (zz.length > 30) {
        const mz = mean(zz), my2 = mean(yy);
        const sz = Math.sqrt(mean(zz.map(v => (v - mz) ** 2))), sy = Math.sqrt(mean(yy.map(v => (v - my2) ** 2)));
        if (sz > 1e-6 && sy > 1e-6) r = mean(zz.map((v, i) => (v - mz) * (yy[i] - my2))) / (sz * sy);
      }
      f.push(r);
    } else for (let i = 0; i < 3 * (ZBINS.length - 1) + 5; i++) f.push(0);
    if (ctx) { CTX_KEYS.forEach(k => f.push(isFinite(ctx[k]) ? ctx[k] : 0)); f.push(1); }
    else for (let i = 0; i <= CTX_KEYS.length; i++) f.push(0);
    return { v: f.map(v => isFinite(v) ? v : 0), pcap };
  }

  function contextVector(c, pcap) {
    const npo = c.np_own || 0, nps = c.np_st_share || 0, ml = c.minload, cold = c.cold_h, hr = c.hr923_window;
    return {
      pm_ct: c.pm_ct || 0, pm_st: c.pm_st || 0, pm_ca: c.pm_ca || 0, fuel_coal: c.fuel_coal || 0, chp: c.chp || 0,
      lnp_own: Math.log(Math.max(npo, 1) / 100), lnp_st: Math.log1p(nps / 100),
      r_own: npo > 0 ? pcap / npo : 0, r_blk: npo > 0 ? pcap / (npo + nps) : 0,
      minload_frac: (ml && npo > 0) ? ml / (npo + nps) : 0, minload_has: ml ? 1 : 0,
      cold_lh: cold != null ? Math.log1p(cold) : 0, cold_has: cold != null ? 1 : 0,
      hr923: hr ? Math.log(hr / 10) : 0, hr923_has: hr ? 1 : 0,
      age: ((c.ref_year || 2020) - (c.online_year || 2000)) / 30
    };
  }

  // ---------------------------------------------------------------- network
  function forward(W, X) {      // X: array of feature vectors
    const out = [];
    for (const x of X) {
      let h = x.map((v, i) => (v - W.mu[i]) / W.sd[i]);
      for (let l = 0; l < W.layers.length; l++) {
        const { w, b, nin, nout } = W.layers[l];
        const o = new Array(nout);
        for (let j = 0; j < nout; j++) {
          let s = b[j]; const off = j * nin;
          for (let i = 0; i < nin; i++) s += w[off + i] * h[i];
          o[j] = (l < W.layers.length - 1) ? (s > 0 ? s : 0.01 * s) : s;
        }
        h = o;
      }
      const lg = h.slice(0, 4), mx = Math.max(...lg), e = lg.map(v => Math.exp(v - mx)), se = e.reduce((a, b) => a + b, 0);
      const nr = REG.length;
      out.push({ p: e.map(v => v / se), m: h.slice(4, 4 + nr), s: h.slice(4 + nr, 4 + 2 * nr).map(v => Math.exp(Math.min(Math.max(v, -6), 4))) });
    }
    return out;
  }
  const sg = a => 1 / (1 + Math.exp(-a));
  function natural(v) {
    const o = {
      pmin_frac: sg(v[0]), ramp_up: Math.exp(v[1]), ramp_dn: Math.exp(v[2]), starts_yr: Math.max(Math.expm1(v[3]), 0),
      min_up: Math.exp(v[4]), med_up: Math.exp(v[5]), min_dn: Math.exp(v[6]), hot: sg(v[7]), warm: sg(v[8]), cold: sg(v[9]),
      hr_full: Math.exp(v[10]), hr_min: Math.exp(v[10] + v[11]), cf: sg(v[12])
    };
    return o;
  }
  function pool(outs) {
    const k = outs.length, nr = REG.length;
    const lp = [0, 0, 0, 0];
    outs.forEach(o => o.p.forEach((p, i) => { lp[i] += Math.log(Math.max(p, 1e-6)); }));
    const mx = Math.max(...lp), e = lp.map(v => Math.exp(v - mx)), se = e.reduce((a, b) => a + b, 0);
    const m = new Array(nr).fill(0), s = new Array(nr).fill(0);
    for (let j = 0; j < nr; j++) {
      let sw = 0, sm = 0, ss = 0;
      outs.forEach(o => { const w = 1 / o.s[j] ** 2; sw += w; sm += w * o.m[j]; ss += o.s[j]; });
      m[j] = sm / sw; s[j] = ss / k;
    }
    return { p: e.map(v => v / se), m, s };
  }
  function interval(o, j, k, factors) {
    const f = factors ? (factors[REG[j]] || 1) : 1;
    const lo = o.m.slice(), hi = o.m.slice();
    lo[j] -= k * o.s[j] * f; hi[j] += k * o.s[j] * f;
    const a = natural(lo)[REG[j]], b = natural(hi)[REG[j]];
    return [Math.min(a, b), Math.max(a, b)];
  }
  function flexClass(starts, cf, medUp) {
    if (!isFinite(starts) || !isFinite(cf)) return 'unknown';
    if (cf < 0.15 && (!isFinite(medUp) || medUp < 12)) return 'peaking';
    if (starts >= 50) return 'cycling';
    if (starts >= 12) return 'load-following';
    return 'baseload';
  }

  // ---------------------------------------------------------------- lab simulator
  function mulberry(seed) {
    let a = seed >>> 0;
    return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

  // Self-commitment DP of one unit against a price series (same model as selfcommit.py, L levels).
  function schedule(price, fuelP, u) {
    const T = price.length, L = 7;
    const lev = Array.from({ length: L }, (_, i) => u.pmin + (u.pmax - u.pmin) * i / (L - 1));
    const F = p => { const f = Math.max(p / u.pmax, 1e-3); return p * u.hr_full * (1 + u.beta * (1 / f - 1)); };
    const fuel = lev.map(F);
    const Um = Math.max(1, Math.round(u.min_up)), Dm = Math.max(Math.round(u.min_dn), WARM_H) + 1;
    const cap = Math.max(u.pmin, u.ramp);
    const dl = (u.pmax - u.pmin) / (L - 1);
    const rb = Math.min(dl > 0 ? Math.floor(u.ramp / dl + 1e-9) : L, L - 1);
    const sc = []; for (let d = 0; d < Dm; d++) { const h = d + 1; sc.push(u.start_cost * (h < HOT_H ? 0.6 : h < WARM_H ? 1 : 1.6)); }
    let Woff = new Float64Array(Dm), Won = new Float64Array(Um * L);
    const decOff = new Int8Array(T * Dm), decOn = new Int8Array(T * Um * L);
    const NEG = -1e18, kap = u.kappa || 0;
    const prof = new Float64Array(L);
    for (let t = T - 1; t >= 0; t--) {
      for (let l = 0; l < L; l++) prof[l] = (price[t] - u.vom) * lev[l] - fuelP[t] * fuel[l] - kap * (lev[l] - u.pmin) ** 2 / Math.max(u.pmax - u.pmin, 1e-6);
      const nOff = new Float64Array(Dm), nOn = new Float64Array(Um * L);
      let bl = 0, bg = NEG;
      for (let l = 0; l < L; l++) if (lev[l] <= cap + 1e-9) { const g = prof[l] + Won[l]; if (g > bg) { bg = g; bl = l; } }
      for (let d = 0; d < Dm; d++) {
        const stay = Woff[Math.min(d + 1, Dm - 1)];
        const can = (d + 1) >= u.min_dn;
        const sv = can ? bg - sc[d] : NEG;
        if (sv > stay) { nOff[d] = sv; decOff[t * Dm + d] = bl; } else { nOff[d] = stay; decOff[t * Dm + d] = -1; }
      }
      const stopV = Woff[0];
      for (let k = 0; k < Um; k++) {
        const kn = Math.min(k + 1, Um - 1);
        for (let l = 0; l < L; l++) {
          let best = NEG, arg = l;
          for (let l2 = Math.max(0, l - rb); l2 <= Math.min(L - 1, l + rb); l2++) {
            const v = prof[l2] + Won[kn * L + l2]; if (v > best) { best = v; arg = l2; }
          }
          const canStop = (k + 1) >= u.min_up && lev[l] <= cap + 1e-9;
          if (canStop && stopV > best) { nOn[k * L + l] = stopV; decOn[(t * Um + k) * L + l] = -1; }
          else { nOn[k * L + l] = best; decOn[(t * Um + k) * L + l] = arg; }
        }
      }
      Woff = nOff; Won = nOn;
    }
    const on = new Array(T).fill(false), P = new Array(T).fill(0);
    let st = 0, d = Dm - 1, k = 0, l = 0;
    for (let t = 0; t < T; t++) {
      if (st === 0) {
        const a = decOff[t * Dm + d];
        if (a < 0) d = Math.min(d + 1, Dm - 1); else { st = 1; k = 0; l = a; on[t] = true; P[t] = lev[l]; }
      } else {
        const a = decOn[(t * Um + k) * L + l];
        if (a < 0) { st = 0; d = 0; } else { k = Math.min(k + 1, Um - 1); l = a; on[t] = true; P[t] = lev[l]; }
      }
    }
    return { on, P };
  }

  const START = { gas_ct: [0.3, 0.8, 0.5, 1.0], gas_cc: [0.08, 0.3, 0.25, 0.6], coal_steam: [0, 0.05, 0.02, 0.08], gas_steam: [0, 0.05, 0.06, 0.2] };
  // CAMPD-style record from a schedule: load noise, start-up / shut-down hours, heat input with CEMS noise.
  function emulate(u, sch, seed) {
    const r = mulberry(seed), T = sch.on.length;
    const op = new Array(T).fill(0), P = new Array(T).fill(0), HI = new Array(T).fill(NaN);
    let a = 0; const sig = 0.004 + 0.02 * r(), rho = 0.8;
    for (let t = 0; t < T; t++) {
      a = rho * a + gauss(r) * sig * Math.sqrt(1 - rho * rho);
      if (sch.on[t]) { op[t] = 1; P[t] = Math.min(Math.max(sch.P[t] * (1 + a), 0.9 * u.pmin), u.pmax * 1.01); }
    }
    const S = START[u.tech]; const a1 = S[0] + (S[1] - S[0]) * r(), rw = S[2] + (S[3] - S[2]) * r();
    const R = runs(sch.on);
    R.s.forEach((t0, i) => {
      if (!R.v[i] || t0 === 0) return;
      op[t0] = 0.3 + 0.7 * r();
      let lvl = u.pmax * a1 * (0.5 + r()), j = 0;
      while (t0 + j < T && j < R.L[i] && lvl < P[t0 + j]) { P[t0 + j] = lvl; lvl += rw * u.pmax; j++; }
      const te = t0 + R.L[i] - 1;
      if (R.L[i] >= 2 && te + 1 < T && r() < 0.8) { op[te] = 0.2 + 0.8 * r(); P[te] = Math.min(P[te], u.pmax * (0.1 + 0.4 * r())); }
    });
    for (let t = 0; t < T; t++) {
      if (op[t] > 0) {
        P[t] = Math.round(P[t]);
        const f = Math.max(P[t] / u.pmax, 1e-3);
        HI[t] = op[t] * Math.max(P[t], 1e-3) * u.hr_full * (1 + u.beta * (1 / f - 1)) * Math.exp(gauss(r) * 0.01);
      }
    }
    return { op, P, HI };
  }

  // truth statistics (port of pdcore.truth, without the heat-rate fit's robust loop details)
  function truth(rec) {
    const n = rec.op.length, op = rec.op, P = rec.P;
    const full = op.map((o, i) => o >= 0.999 && P[i] > 0);
    const steady = full.map((f, i) => f && i > 0 && full[i - 1] && i < n - 1 && full[i + 1]);
    const r = {};
    const pf = P.filter((_, i) => full[i]);
    if (pf.length < 24) return r;
    const pmax = pct(pf, 99.5); r.pmax = pmax;
    const ps = P.filter((_, i) => steady[i]);
    r.pmin_frac = pct(ps, 2) / pmax;
    const d = []; for (let i = 0; i < n - 1; i++) if (steady[i] && steady[i + 1]) d.push((P[i + 1] - P[i]) / pmax);
    r.ramp_up = Math.max(pct(d, 99), 0); r.ramp_dn = Math.max(-pct(d, 1), 0);
    const on = op.map((o, i) => o > 0 && P[i] > 0), R = runs(on);
    const st = R.s.filter((s, i) => R.v[i] && s > 0);
    r.starts_yr = st.length * 8760 / n;
    let e = 0; for (let i = 0; i < n; i++) e += P[i] * op[i]; r.cf = e / (pmax * n);
    // heat-rate: linear fuel fit on steady hours
    const xs = [], ys = [];
    for (let i = 0; i < n; i++) if (steady[i] && isFinite(rec.HI[i])) { xs.push(P[i]); ys.push(rec.HI[i]); }
    if (xs.length > 48) {
      const mx = mean(xs), my = mean(ys);
      let sxx = 0, sxy = 0; xs.forEach((x, i) => { sxx += (x - mx) ** 2; sxy += (x - mx) * (ys[i] - my); });
      const b = sxx > 0 ? sxy / sxx : my / mx, a0 = my - b * mx;
      const lo = Math.max(r.pmin_frac, PART_MIN) * pmax;
      r.hr_full = (a0 + b * pmax) / pmax; r.hr_min = (a0 + b * lo) / lo;
    }
    return r;
  }

  root.PD = { TECHS, REG, features, contextVector, forward, natural, pool, interval, flexClass, schedule, emulate, truth, runs, pct, mulberry };
})(typeof window !== 'undefined' ? window : globalThis);
