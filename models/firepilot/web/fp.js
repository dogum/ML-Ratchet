// FirePilot planner in JavaScript: a line-by-line port of fp/core.py + fp/planner.py (features, value net).
// Works in the browser (window.FP) and in node (module.exports) for the JS == Python check.
(function (root) {
  'use strict';
  const HORIZON = 72.0, FL_HOLD = 2.4;
  const mod = (a, n) => ((a % n) + n) % n;
  const wrap180 = a => mod(a + 180.0, 360.0) - 180.0;

  // exact Euclidean distance transform (Felzenszwalb & Huttenlocher) of "distance to nearest true cell"
  function edt(mask, ny, nx) {
    const INF = 1e20, f = new Float64Array(Math.max(ny, nx)), d = new Float64Array(Math.max(ny, nx));
    const v = new Int32Array(Math.max(ny, nx)), z = new Float64Array(Math.max(ny, nx) + 1);
    const g = new Float64Array(ny * nx);
    function dt1(n) {
      let k = 0; v[0] = 0; z[0] = -INF; z[1] = INF;
      for (let q = 1; q < n; q++) {
        let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
        while (s <= z[k]) { k--; s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]); }
        k++; v[k] = q; z[k] = s; z[k + 1] = INF;
      }
      k = 0;
      for (let q = 0; q < n; q++) { while (z[k + 1] < q) k++; d[q] = (q - v[k]) * (q - v[k]) + f[v[k]]; }
    }
    for (let c = 0; c < nx; c++) {
      for (let r = 0; r < ny; r++) f[r] = mask[r * nx + c] ? 0 : INF;
      dt1(ny);
      for (let r = 0; r < ny; r++) g[r * nx + c] = d[r];
    }
    for (let r = 0; r < ny; r++) {
      for (let c = 0; c < nx; c++) f[c] = g[r * nx + c];
      dt1(nx);
      for (let c = 0; c < nx; c++) g[r * nx + c] = Math.sqrt(d[c]);
    }
    return g;
  }

  function searchLeft(W, w) { let lo = 0, hi = W.length; while (lo < hi) { const m = (lo + hi) >> 1; if (W[m] < w) lo = m + 1; else hi = m; } return lo; }

  // ctx input: {ny, nx, res, burned0 (Uint8), arr (Float64, NaN = never), fl (Float64), cost (Float64), rate,
  //             clock_t, clock_w (Float64Array), head}
  function Context(inp) {
    const n = inp.ny * inp.nx, ny = inp.ny, nx = inp.nx;
    Object.assign(this, inp);
    this.n = n;
    this.D = edt(inp.burned0, ny, nx);
    for (let i = 0; i < n; i++) this.D[i] *= inp.res;
    let sr = 0, sc = 0, nb = 0;
    for (let i = 0; i < n; i++) if (inp.burned0[i]) { sr += Math.floor(i / nx); sc += i % nx; nb++; }
    const r0 = sr / nb, c0 = sc / nb;
    this.Phi = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const r = Math.floor(i / nx), c = i % nx;
      const az = Math.atan2(c - c0, -(r - r0)) * 180 / Math.PI;
      this.Phi[i] = mod(az - inp.head + 180.0, 360.0) - 180.0;
    }
    this.fburn = new Uint8Array(n);
    let af = 0;
    for (let i = 0; i < n; i++) { const a = inp.arr[i]; if (a === a && a <= HORIZON) { this.fburn[i] = 1; af++; } }
    this.a0 = nb; this.af = af; this.gf = Math.max(af - nb, 1);
    this.scn = this.scenarioFeatures();
  }

  Context.prototype.scenarioFeatures = function () {
    const { arr, burned0, fl, res, n, ny, nx, cost } = this;
    const cell = res * res / 1e4;
    const g = [12, 24, 48].map(h => { let k = 0; for (let i = 0; i < n; i++) if (!burned0[i] && arr[i] === arr[i] && arr[i] <= h) k++; return k / this.gf; });
    const fln = []; let costsum = 0, nnew = 0, edge = 0;
    for (let i = 0; i < n; i++) {
      if (this.fburn[i] && !burned0[i]) { fln.push(fl[i]); costsum += cost[i]; nnew++; }
      const r = Math.floor(i / nx), c = i % nx;
      if (this.fburn[i] && (r === 0 || r === ny - 1 || c === 0 || c === nx - 1)) edge = 1;
    }
    let over = 0; for (const v of fln) if (v > FL_HOLD) over++;
    const srt = fln.slice().sort((a, b) => a - b);
    const med = srt.length ? (srt.length % 2 ? srt[(srt.length - 1) / 2] : (srt[srt.length / 2 - 1] + srt[srt.length / 2]) / 2) : 0;
    const req0 = Math.sqrt(this.a0 * res * res / Math.PI), req1 = Math.sqrt(this.af * res * res / Math.PI);
    const W = this.clock_w, Wend = W[W.length - 1];
    return [Math.log10(this.a0 * cell + 1), Math.log10((this.af - this.a0) * cell + 1), g[0], g[1], g[2],
      Math.log10(this.rate), Wend / 36.0, (req1 - req0) / HORIZON / 100.0,
      fln.length ? over / fln.length : 0, fln.length ? Math.log10(med + 0.1) : -1, edge,
      nnew ? costsum / nnew : 1.0, Math.log10(this.rate * Wend / Math.max(2 * Math.PI * req1, 1.0) + 1e-3)];
  };

  // plan -> {idx, tb, km, dl}. Only cells with D <= max offset + 2 cells can be on the line (sorted-by-D scan).
  Context.prototype.prep = function () {
    if (this.byD) return;
    const n = this.n, D = this.D;
    this.byD = Uint32Array.from({ length: n }, (_, i) => i).sort((a, b) => D[a] - D[b]);
    this.Ds = Float64Array.from(this.byD, i => D[i]);
    const fb = []; for (let i = 0; i < n; i++) if (this.fburn[i]) fb.push(i);
    this.fbList = Uint32Array.from(fb);
  };
  Context.prototype.lines = function (p) {
    this.prep();
    const { ny, nx, res, D, Phi, burned0 } = this;
    const dOf = i => { const dl = wrap180(Phi[i] - p.phi_c); const w = Math.pow((1.0 + Math.cos(dl * Math.PI / 180)) / 2.0, 2); return Math.max(p.d_f + (p.d_h - p.d_f) * w, res); };
    const memo = new Int8Array(this.n);    // 0 unknown, 1 out, 2 inside
    const isOut = i => { let v = memo[i]; if (!v) { v = D[i] >= dOf(i) ? 1 : 2; memo[i] = v; } return v === 1; };
    const dmax = Math.max(p.d_h, p.d_f, res) + 2.0 * res;
    let lo = 0, hi = this.Ds.length; while (lo < hi) { const m = (lo + hi) >> 1; if (this.Ds[m] <= dmax) lo = m + 1; else hi = m; }
    const dmin = Math.max(Math.min(p.d_h, p.d_f), res) - 1e-9;    // line cells are outside: D >= d >= dmin
    let a = 0, b = lo; while (a < b) { const m = (a + b) >> 1; if (this.Ds[m] < dmin) a = m + 1; else b = m; }
    const idx = []; let n4 = 0;
    for (let q = a; q < lo; q++) {
      const i = this.byD[q], r = Math.floor(i / nx), c = i % nx;
      if (burned0[i] || !isOut(i)) continue;
      if (Math.abs(wrap180(Phi[i] - p.phi_c)) > p.omega) continue;
      let any8 = false, any4 = false;
      for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
        if (!dr && !dc) continue;
        const rr = r + dr, cc = c + dc;
        if (rr < 0 || rr >= ny || cc < 0 || cc >= nx) continue;
        if (!isOut(rr * nx + cc)) { any8 = true; if (!dr || !dc) any4 = true; }
      }
      if (any4) n4++;
      if (any8 && r > 0 && r < ny - 1 && c > 0 && c < nx - 1) idx.push(i);
    }
    idx.sort((a, b) => a - b);
    const tb = new Float64Array(idx.length).fill(Infinity);
    if (!idx.length) return { idx, tb, km: 0, dl: [] };
    const cellLen = res * (Math.PI / 4.0) * n4 / Math.max(idx.length, 1);
    const dl = idx.map(i => wrap180(Phi[i] - p.phi_c)), cst = idx.map(i => this.cost[i] * cellLen);
    const T = this.clock_t, W = this.clock_w, Wend = W[W.length - 1];
    for (const side of [0, 1]) {
      const s = []; for (let k = 0; k < idx.length; k++) if ((side === 0) === (dl[k] >= 0)) s.push(k);
      const key = k => (p.mode === 0 ? Math.abs(dl[k]) : -Math.abs(dl[k]));
      s.sort((a, b) => (key(a) - key(b)) || (a - b));
      let cum = 0;
      for (const k of s) {
        cum += cst[k];
        const w = cum / (this.rate / 2.0);
        let t = Infinity;
        if (w <= Wend) t = T[Math.min(searchLeft(W, w), T.length - 1)];
        tb[k] = t > HORIZON ? Infinity : t;
      }
    }
    let km = 0; for (let k = 0; k < idx.length; k++) if (isFinite(tb[k])) km += cst[k];
    return { idx, tb, km: km / 1000.0, dl };
  };

  Context.prototype.planEval = function (p) {
    const L = this.lines(p), { idx, tb } = L, nl = idx.length;
    if (!nl) return { s_proxy: 0, f: new Array(18).fill(0), km: 0, L };
    const { n, nx, ny, arr, fl, burned0, res } = this;
    let built = 0, wonR = 0, reachedN = 0, reachNotHold = 0, effN = 0, tfin = -Infinity;
    const eff = new Uint8Array(nl), eff3 = new Uint8Array(nl);
    for (let k = 0; k < nl; k++) {
      let a = arr[idx[k]]; if (!(a === a)) a = 1e9;
      const b = isFinite(tb[k]), hold = fl[idx[k]] <= FL_HOLD, reached = a <= HORIZON;
      const won = b && tb[k] < a, won3 = b && tb[k] < a - 3.0;
      if (b) { built++; tfin = Math.max(tfin, tb[k]); }
      if (won && reached) wonR++;
      if (reached) reachedN++;
      if (reached && !hold) reachNotHold++;
      eff[k] = won && hold ? 1 : 0; eff3[k] = won3 && hold ? 1 : 0;
      if (eff[k]) effN++;
    }
    if (!built) tfin = HORIZON;
    const sp = [eff, eff3].map(e => {
      const blk = new Uint8Array(n); for (let k = 0; k < nl; k++) if (e[k]) blk[idx[k]] = 1;
      const seen = new Uint8Array(n), st = []; let cnt = 0;
      for (const i of this.fbList) if (burned0[i] && !blk[i]) { seen[i] = 1; st.push(i); }
      while (st.length) {
        const i = st.pop(); cnt++;
        const r = Math.floor(i / nx), c = i % nx;
        for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
          const rr = r + dr, cc = c + dc;
          if (rr < 0 || rr >= ny || cc < 0 || cc >= nx) continue;
          const j = rr * nx + cc;
          if (!seen[j] && this.fburn[j] && !blk[j]) { seen[j] = 1; st.push(j); }
        }
      }
      const ap = cnt || this.a0;
      return Math.min(Math.max((this.af - ap) / this.gf, 0), 1);
    });
    // shadow
    let sh = 0, shT = 0;
    for (const i of this.fbList) {
      const dl = wrap180(this.Phi[i] - p.phi_c);
      const w = Math.pow((1.0 + Math.cos(dl * Math.PI / 180)) / 2.0, 2);
      const d = Math.max(p.d_f + (p.d_h - p.d_f) * w, res);
      if (this.D[i] >= d && Math.abs(dl) <= p.omega) { sh++; shT += arr[i]; }
    }
    const sSh = sh / this.gf, tSh = sh ? shT / sh / HORIZON : 1.0, eFrac = effN / Math.max(nl, 1);
    const f = [p.mode, Math.cos(p.phi_c * Math.PI / 180), Math.sin(p.phi_c * Math.PI / 180), p.omega / 180.0,
      Math.log10(p.d_h) - 3, Math.log10(p.d_f) - 3, Math.log10(L.km + 0.1), built / nl,
      wonR / Math.max(reachedN, 1), reachNotHold / nl, reachedN / nl, tfin / HORIZON, sSh, tSh, sSh * eFrac, eFrac,
      sp[0], sp[1]];
    return { s_proxy: sp[0], f, km: L.km, L };
  };

  Context.prototype.features = function (p) { const pe = this.planEval(p); return { x: this.scn.concat(pe.f), pe }; };

  // value net: {mu, sd, members: [{W: [W0, W1, W2] (row-major in x hidden), b: [...]}]}
  function predict(net, x) {
    const h0 = x.map((v, i) => (v - net.mu[i]) / net.sd[i]);
    const ms = [], ss = [];
    for (const m of net.members) {
      let h = h0;
      for (let l = 0; l < 3; l++) {
        const W = m.W[l], b = m.b[l], nin = h.length, nout = b.length, o = new Array(nout);
        for (let j = 0; j < nout; j++) { let s = b[j]; for (let i = 0; i < nin; i++) s += h[i] * W[i * nout + j]; o[j] = l < 2 ? Math.tanh(s) : s; }
        h = o;
      }
      ms.push(1 / (1 + Math.exp(-h[0]))); ss.push(Math.exp(Math.min(Math.max(h[1], -6), 2)));
    }
    const M = ms.reduce((a, b) => a + b, 0) / ms.length;
    const v = ms.reduce((a, b) => a + (b - M) * (b - M), 0) / ms.length;
    const s2 = ss.reduce((a, b) => a + b * b, 0) / ss.length;
    return { mean: M, sigma: Math.sqrt(s2 + v) * (net.sig_scale || 1) };
  }

  // small seeded PRNG for the browser planner's candidates
  function rng(seed) { let s = seed >>> 0; return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function normal(r) { return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r()); }
  const BASELINES = {
    direct_head: { mode: 0, phi_c: 0, omega: 90, d_h: 90, d_f: 90 },
    indirect_1km: { mode: 0, phi_c: 0, omega: 90, d_h: 1000, d_f: 1000 },
    indirect_2km: { mode: 0, phi_c: 0, omega: 90, d_h: 2000, d_f: 2000 },
    indirect_4km: { mode: 0, phi_c: 0, omega: 90, d_h: 4000, d_f: 4000 },
    anchor_flank: { mode: 1, phi_c: 0, omega: 180, d_h: 90, d_f: 90 },
  };
  function samplePlan(r) {
    const mode = r() < 0.5 ? 1 : 0;
    let phi = r() < 0.75 ? normal(r) * 50 : r() * 360 - 180; phi = wrap180(phi);
    const omega = 15 + r() * 165, lo = Math.log(90), hi = Math.log(8000);
    const d_h = Math.exp(lo + r() * (hi - lo)), d_f = r() < 0.6 ? Math.exp(lo + r() * (hi - lo)) : d_h;
    return { mode, phi_c: phi, omega, d_h, d_f };
  }
  function candidates(r, n) {
    const c = Object.values(BASELINES).map(p => Object.assign({}, p));
    for (const d of [500, 1000, 2000, 3000, 5000]) for (const om of [60, 90, 120, 180]) for (const mode of [0, 1]) c.push({ mode, phi_c: 0, omega: om, d_h: d, d_f: d });
    for (let k = 0; k < n; k++) c.push(samplePlan(r));
    return c;
  }
  function plan(ctx, net, seed, n = 100, refine = 1, nRef = 36, top = 6) {
    const r = rng(seed);
    let cands = candidates(r, n);
    let scored = cands.map(p => ({ p, ...predict(net, ctx.features(p).x) }));
    for (let it = 0; it < refine; it++) {
      const best = scored.slice().sort((a, b) => b.mean - a.mean).slice(0, top);
      for (const b of best) for (let k = 0; k < nRef / top; k++) {
        const v = b.p, mode = r() < 0.85 ? v.mode : 1 - v.mode;
        const q = { mode, phi_c: wrap180(v.phi_c + 25 * normal(r)), omega: Math.min(Math.max(v.omega + 20 * normal(r), 5), 180),
          d_h: Math.exp(Math.min(Math.max(Math.log(v.d_h) + 0.35 * normal(r), Math.log(90)), Math.log(8000))),
          d_f: Math.exp(Math.min(Math.max(Math.log(v.d_f) + 0.35 * normal(r), Math.log(90)), Math.log(8000))) };
        scored.push({ p: q, ...predict(net, ctx.features(q).x) });
      }
    }
    scored.sort((a, b) => b.mean - a.mean);
    return scored[0];
  }
  const api = { Context, predict, plan, candidates, BASELINES, edt, HORIZON, FL_HOLD };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.FP = api;
})(typeof window !== 'undefined' ? window : globalThis);
