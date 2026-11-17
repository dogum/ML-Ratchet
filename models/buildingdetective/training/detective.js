// BuildingDetective in the browser: a line-by-line port of features.py (hourly / monthly / annual) and the numpy MLP
// in detective.py. Verified against Python on real and simulated buildings (web/verify_js.json).
(function (root) {
  'use strict';
  const CH = ['elec', 'fuel', 'dheat', 'dcool'];
  const USES = ['office', 'education', 'lodging', 'retail', 'healthcare', 'assembly', 'warehouse', 'residential_single',
    'residential_multi', 'food', 'lab', 'other'];
  const HEAT = ['heat_pump', 'resistance', 'fuel', 'district', 'none'];
  const COOL = ['none', 'dx', 'chiller', 'district'];
  const REG = ['logA', 'year', 'logUA', 'logUAm2', 'logtau', 'Tb_heat', 'Tb_cool', 'logS_heat', 'logS_cool', 'base_share',
    'op_hours', 'weekend'];
  const TB = []; for (let t = -21; t < 37; t += 3) TB.push(t);
  const NB = TB.length - 1;
  const TAUS = [0, 12, 24, 48, 96, 192];
  const HB = 18.33;
  const N_W_H = 8 + NB, N_C_H = 3 + 2 * NB + 168 + 48 + 6 + 12 + 10 + 8;
  const N_W_M = 12 + 12 + 4, N_C_M = 3 + 12 + 6;

  const isF = Number.isFinite;
  function nanmean(a) { let s = 0, n = 0; for (const v of a) if (isF(v)) { s += v; n++; } return n ? s / n : NaN; }
  function nanstd(a) { const m = nanmean(a); let s = 0, n = 0; for (const v of a) if (isF(v)) { s += (v - m) ** 2; n++; } return n ? Math.sqrt(s / n) : NaN; }
  function percentile(arr, p) {               // numpy linear
    const a = Float64Array.from(arr).sort(); const n = a.length; if (!n) return NaN;
    const r = (p / 100) * (n - 1); const lo = Math.floor(r), hi = Math.ceil(r); return a[lo] + (a[hi] - a[lo]) * (r - lo);
  }
  function corr(a, b) {
    let n = 0, sa = 0, sb = 0;
    for (let i = 0; i < a.length; i++) if (isF(a[i]) && isF(b[i])) { n++; sa += a[i]; sb += b[i]; }
    if (n < 10) return 0;
    const ma = sa / n, mb = sb / n; let sab = 0, saa = 0, sbb = 0;
    for (let i = 0; i < a.length; i++) if (isF(a[i]) && isF(b[i])) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
    const d = Math.sqrt(saa * sbb); return d > 0 ? sab / d : 0;
  }
  function fillBins(v) {
    const x = [], y = []; v.forEach((e, i) => { if (isF(e)) { x.push(i); y.push(e); } });
    if (!x.length) return v.map(() => 0);
    return v.map((e, i) => {
      if (i <= x[0]) return y[0]; if (i >= x[x.length - 1]) return y[y.length - 1];
      let k = 0; while (x[k + 1] < i) k++; return y[k] + (y[k + 1] - y[k]) * (i - x[k]) / (x[k + 1] - x[k]);
    });
  }
  function ewma(x, tau) {
    if (tau === 0) return Float64Array.from(x);
    const a = 1 - Math.exp(-1 / tau); const y = new Float64Array(x.length); let acc = x[0];
    for (let i = 0; i < x.length; i++) { acc = a * x[i] + (1 - a) * acc; y[i] = acc; }
    return y;
  }
  // least squares on selected columns of [1, xh, xc] via normal equations (columns that are all zero get coef 0)
  function lsq(cols, X, E) {
    const k = cols.filter(c => X[c].some(v => v !== 0));
    const m = k.length, A = [], b = [];
    for (let i = 0; i < m; i++) {
      A.push([]); let s = 0; for (let t = 0; t < E.length; t++) s += X[k[i]][t] * E[t]; b.push(s);
      for (let j = 0; j < m; j++) { let q = 0; for (let t = 0; t < E.length; t++) q += X[k[i]][t] * X[k[j]][t]; A[i].push(q); }
    }
    for (let i = 0; i < m; i++) {               // gaussian elimination with partial pivoting
      let p = i; for (let r = i + 1; r < m; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
      [A[i], A[p]] = [A[p], A[i]]; [b[i], b[p]] = [b[p], b[i]];
      for (let r = i + 1; r < m; r++) { const f = A[r][i] / A[i][i]; for (let c = i; c < m; c++) A[r][c] -= f * A[i][c]; b[r] -= f * b[i]; }
    }
    const x = new Array(m).fill(0);
    for (let i = m - 1; i >= 0; i--) { let s = b[i]; for (let c = i + 1; c < m; c++) s -= A[i][c] * x[c]; x[i] = s / A[i][i]; }
    const coef = [0, 0, 0]; k.forEach((c, i) => { coef[c] = x[i]; }); return coef;
  }
  function prism(Td0, Ed0) {
    const Td = [], Ed = [];
    for (let i = 0; i < Td0.length; i++) if (isF(Td0[i]) && isF(Ed0[i])) { Td.push(Td0[i]); Ed.push(Ed0[i]); }
    if (Td.length < 8 || nanstd(Ed) === 0) return [Ed.length ? nanmean(Ed) : 0, 0, 0, 15, 20, 0];
    const me = nanmean(Ed); let sst = 1e-12; for (const e of Ed) sst += (e - me) ** 2;
    let best = null; const one = Td.map(() => 1);
    for (let i = 0; 4 + 1.5 * i < 23; i++) {
      const tbh = 4 + 1.5 * i; const xh = Td.map(t => Math.max(tbh - t, 0));
      const c0 = Math.max(tbh, 10);
      for (let j = 0; c0 + 1.5 * j < 29; j++) {
        const tbc = c0 + 1.5 * j; const xc = Td.map(t => Math.max(t - tbc, 0)); const X = [one, xh, xc];
        let coef = lsq([0, 1, 2], X, Ed);
        if (coef[1] < 0 || coef[2] < 0) {
          const cols = [0].concat([1, 2].filter(k => coef[k] >= 0));
          coef = lsq(cols, X, Ed); coef = [coef[0], Math.max(coef[1], 0), Math.max(coef[2], 0)];
        }
        let sse = 0; for (let t = 0; t < Ed.length; t++) { const r = coef[0] + coef[1] * xh[t] + coef[2] * xc[t] - Ed[t]; sse += r * r; }
        if (best === null || sse < best[0]) best = [sse, coef, tbh, tbc];
      }
    }
    const [sse, coef, tbh, tbc] = best;
    return [coef[0], coef[1], coef[2], coef[1] > 0 ? tbh : 15, coef[2] > 0 ? tbc : 20, 1 - sse / sst];
  }
  // time: {hr: Uint8Array, dow: Uint8Array (0=Mon), day: Int32Array (local day index), month: Uint8Array (1..12)}
  function featHourly(meters, T, GHI, tm, lat) {
    const n = T.length, hr = tm.hr, dow = tm.dow;
    const dmap = new Map(); const dinv = new Int32Array(n);
    for (let i = 0; i < n; i++) { if (!dmap.has(tm.day[i])) dmap.set(tm.day[i], dmap.size); dinv[i] = dmap.get(tm.day[i]); }
    // numpy unique sorts day keys; local days are monotone so insertion order == sorted order
    const nd = dmap.size, cnt = new Float64Array(nd), Tds = new Float64Array(nd), wds = new Float64Array(nd);
    for (let i = 0; i < n; i++) { cnt[dinv[i]]++; Tds[dinv[i]] += isF(T[i]) ? T[i] : 0; wds[dinv[i]] += dow[i] < 5 ? 1 : 0; }
    const Td = Array.from(Tds, (s, d) => s / Math.max(cnt[d], 1));
    const ddwd = Array.from(wds, (s, d) => s / Math.max(cnt[d], 1) > 0.5);
    let hdd = 0, cdd = 0; for (const t of Td) { hdd += Math.max(HB - t, 0); cdd += Math.max(t - HB, 0); }
    hdd /= nd; cdd /= nd;
    const occ = new Array(NB).fill(0);
    for (const t0 of Td) { const t = Math.min(Math.max(t0, TB[0] + 1e-6), TB[NB] - 1e-6); const b = Math.floor((t - TB[0]) / 3); occ[Math.min(Math.max(b, 0), NB - 1)] += 1 / nd; }
    const out = [nanmean(T) / 10, nanstd(T) / 10, hdd / 10, cdd / 10, nanmean(GHI) / 200, lat / 90, nd / 365, Math.log10(Math.max(nd, 1)) / 3].concat(occ);
    const mT = nanmean(T); const Tf = Float64Array.from(T, v => isF(v) ? v : mT);
    const Tew = TAUS.map(tau => { const e = ewma(Tf, tau); const s = new Float64Array(nd); for (let i = 0; i < n; i++) s[dinv[i]] += e[i]; return Array.from(s, (v, d) => v / Math.max(cnt[d], 1)); });
    // centred 24-h rolling mean (pandas: window [i-12, i+11])
    const Tdev = new Float64Array(n); { const cs = new Float64Array(n + 1); for (let i = 0; i < n; i++) cs[i + 1] = cs[i] + Tf[i];
      for (let i = 0; i < n; i++) { const a = Math.max(0, i - 12), b = Math.min(n, i + 12); Tdev[i] = Tf[i] - (cs[b] - cs[a]) / (b - a); } }
    const Th = new Float64Array(n); for (let i = 0; i < n; i++) Th[i] = Td[dinv[i]];
    for (const ch of CH) {
      const x = meters[ch];
      let tot = 0, any = false; if (x) for (const v of x) if (isF(v)) { tot += Math.abs(v); any = true; }
      if (!x || !any || tot <= 0) { for (let k = 0; k < N_C_H; k++) out.push(0); continue; }
      const mu = nanmean(x);
      if (!isF(mu) || mu <= 0) { for (let k = 0; k < N_C_H; k++) out.push(0); continue; }
      const v = Array.from(x, e => isF(e)); const xn = Float64Array.from(x, e => e / mu);
      let vm = 0; for (const b of v) vm += b; vm /= n;
      const f = [1, vm, Math.log10(mu)];
      const dv = new Float64Array(nd), es = new Float64Array(nd);
      for (let i = 0; i < n; i++) if (v[i]) { dv[dinv[i]]++; es[dinv[i]] += xn[i]; }
      const Ed = Array.from(es, (s, d) => dv[d] < 12 ? NaN : s / Math.max(dv[d], 1));
      for (const want of [true, false]) {
        const s = new Float64Array(NB), c = new Float64Array(NB);
        for (let d = 0; d < nd; d++) {
          if (ddwd[d] !== want || !isF(Ed[d])) continue;
          let b = -1; for (let k = 0; k < TB.length; k++) if (Td[d] >= TB[k]) b = k; // np.digitize - 1
          if (b >= 0 && b < NB) { s[b] += Ed[d]; c[b]++; }
        }
        fillBins(Array.from(s, (q, k) => c[k] > 0 ? q / Math.max(c[k], 1) : NaN)).forEach(q => f.push(q));
      }
      const how = (i) => dow[i] * 24 + hr[i];
      const ps = new Float64Array(168), pc = new Float64Array(168);
      for (let i = 0; i < n; i++) if (v[i]) { ps[how(i)] += xn[i]; pc[how(i)]++; }
      const pw = fillBins(Array.from(ps, (q, k) => pc[k] > 0 ? q / Math.max(pc[k], 1) : NaN));
      pw.forEach(q => f.push(q));
      for (const cold of [true, false]) {
        const s = new Float64Array(24), c = new Float64Array(24);
        for (let i = 0; i < n; i++) { const sel = (cold ? Th[i] < 5 : Th[i] > 22) && dow[i] < 5 && v[i]; if (sel) { s[hr[i]] += xn[i]; c[hr[i]]++; } }
        const p = Array.from(s, (q, k) => c[k] > 0 ? q / Math.max(c[k], 1) : NaN);
        if (p.every(isF)) p.forEach(q => f.push(q));
        else for (let h = 0; h < 24; h++) { let q = 0; for (let d = 0; d < 5; d++) q += pw[d * 24 + h]; f.push(q / 5); }
      }
      const pr = prism(Td, Ed);
      f.push(pr[0], pr[1] * 10, pr[2] * 10, pr[3] / 10, pr[4] / 10, pr[5]);
      for (const hot of [false, true]) {
        const idx = []; for (let d = 0; d < nd; d++) if (hot ? Td[d] > 20 : Td[d] < 10) idx.push(d);
        for (const tw of Tew) f.push(idx.length > 10 ? corr(idx.map(d => Ed[d]), idx.map(d => tw[d])) : 0);
      }
      const dev = Float64Array.from(xn, (q, i) => q - pw[how(i)]);
      for (const cold of [true, false]) {
        for (const lag of [0, 2, 4, 8, 16]) {
          const a = [], b = [];
          for (let i = lag; i < n; i++) { const sel = (cold ? Th[i] < 5 : Th[i] > 22) && v[i]; if (sel) { a.push(dev[i]); b.push(Tdev[i - lag]); } }
          f.push(a.length > 48 ? corr(a, b) : 0);
        }
      }
      const xv = []; for (let i = 0; i < n; i++) if (v[i]) xv.push(xn[i]);
      const nightA = [], dayA = []; for (let i = 0; i < n; i++) if (v[i] && dow[i] < 5) { if (hr[i] <= 4) nightA.push(xn[i]); if (hr[i] >= 10 && hr[i] <= 15) dayA.push(xn[i]); }
      const night = nightA.length ? nanmean(nightA) : NaN, day_ = dayA.length ? nanmean(dayA) : NaN;
      const Ewd = [], Ewe = []; for (let d = 0; d < nd; d++) if (isF(Ed[d])) (ddwd[d] ? Ewd : Ewe).push(Ed[d]);
      const wkd = Ewd.length ? nanmean(Ewd) : 1, wke = Ewe.length ? nanmean(Ewe) : wkd;
      let z = 0; for (const q of xv) if (q <= 1e-6) z++;
      let ad = 0, an = 0; if (xv.length > 2) for (let i = 1; i < n; i++) { const q = Math.abs(xn[i] - xn[i - 1]); if (isF(q)) { ad += q; an++; } }
      f.push(percentile(xv, 1), percentile(xv, 50), percentile(xv, 99) / 3,
        isF(night) && isF(day_) ? night / Math.max(day_, 1e-3) : 1, wke / Math.max(wkd, 1e-3), nanstd(Ed), z / xv.length,
        xv.length > 2 ? (an ? ad / an : 0) : 0);
      for (const q of f) out.push(Math.min(Math.max(isF(q) ? q : 0, -20), 20));
    }
    return Float64Array.from(out);
  }
  function monthlyFromHourly(meters, T, tm) {
    const n = T.length; const mT = nanmean(T);
    const dmap = new Map(); const dinv = new Int32Array(n);
    for (let i = 0; i < n; i++) { if (!dmap.has(tm.day[i])) dmap.set(tm.day[i], dmap.size); dinv[i] = dmap.get(tm.day[i]); }
    const nd0 = dmap.size, cnt = new Float64Array(nd0), ts = new Float64Array(nd0), ms = new Float64Array(nd0);
    for (let i = 0; i < n; i++) { cnt[dinv[i]]++; ts[dinv[i]] += isF(T[i]) ? T[i] : mT; ms[dinv[i]] += tm.month[i] - 1; }
    const nd = new Float64Array(12), Tm = new Float64Array(12), Hm = new Float64Array(12), Cm = new Float64Array(12);
    for (let d = 0; d < nd0; d++) {
      const t = ts[d] / Math.max(cnt[d], 1); const m = Math.round(ms[d] / Math.max(cnt[d], 1));
      nd[m]++; Tm[m] += t; Hm[m] += Math.max(HB - t, 0); Cm[m] += Math.max(t - HB, 0);
    }
    for (let m = 0; m < 12; m++) { const q = Math.max(nd[m], 1); Tm[m] /= q; Hm[m] /= q; Cm[m] /= q; }
    const mm = {};
    for (const ch of Object.keys(meters)) {
      const x = meters[ch]; const s = new Float64Array(12), c = new Float64Array(12);
      for (let i = 0; i < n; i++) if (isF(x[i])) { s[tm.month[i] - 1] += x[i]; c[tm.month[i] - 1]++; }
      mm[ch] = Array.from(s, (q, m) => c[m] > 24 * 5 ? q / Math.max(c[m], 1) : NaN);
    }
    return { mm, Tm: Array.from(Tm), Hm: Array.from(Hm), Cm: Array.from(Cm), nd: Array.from(nd) };
  }
  function featMonthly(M) {
    const { mm, Tm, Hm, Cm, nd } = M; const ok = nd.map(q => q > 0);
    const tok = Tm.filter((q, m) => ok[m]); let sn = 0, sh = 0, sc = 0;
    for (let m = 0; m < 12; m++) { sn += nd[m]; sh += Hm[m] * nd[m]; sc += Cm[m] * nd[m]; }
    const out = Tm.map((q, m) => ok[m] ? q / 10 : 0).concat(Hm.map((q, m) => ok[m] ? q / 10 : 0),
      [nanmean(tok) / 10, nanstd(tok) / 10, sh / Math.max(sn, 1) / 10, sc / Math.max(sn, 1) / 10]);
    for (const ch of CH) {
      const x = mm[ch]; let s = 0, any = false; if (x) for (const q of x) if (isF(q)) { s += q; any = true; }
      if (!x || !any || s <= 0) { for (let k = 0; k < N_C_M; k++) out.push(0); continue; }
      const mu = nanmean(x); if (mu <= 0) { for (let k = 0; k < N_C_M; k++) out.push(0); continue; }
      const xn = x.map(q => q / mu); const f = [1, x.filter(isF).length / 12, Math.log10(mu)];
      xn.forEach(q => f.push(isF(q) ? q : 0));
      const pr = prism(Tm, xn); f.push(pr[0], pr[1] * 10, pr[2] * 10, pr[3] / 10, pr[4] / 10, pr[5]);
      for (const q of f) out.push(Math.min(Math.max(isF(q) ? q : 0, -20), 20));
    }
    return Float64Array.from(out);
  }
  function featAnnual(kwh, hdd, cdd) {
    const f = [hdd / 3000, cdd / 1500, Math.log10(hdd + 50), Math.log10(cdd + 50)];
    for (const ch of CH) { const e = kwh[ch]; if (!isF(e) || e <= 0) f.push(0, 0); else f.push(1, Math.log10(e) - 5); }
    return Float64Array.from(f);
  }
  // ---------------------------------------------------------------- MLP
  function erf(x) {      // W. J. Cody-style rational approximation via erfc continued fraction, |err| < 1.2e-7
    const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
      t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? 1 - r : r - 1;
  }
  function runNet(net, X) {
    let h = Array.from(X, (v, i) => (v - net.xm[i]) / net.xs[i]);
    for (let l = 0; l < net.W.length; l++) {
      const W = net.W[l], b = net.b[l], nin = h.length, nout = b.length, o = new Array(nout);
      for (let j = 0; j < nout; j++) { let s = b[j]; for (let i = 0; i < nin; i++) s += W[j * nin + i] * h[i]; o[j] = 0.5 * s * (1 + erf(s / Math.SQRT2)); }
      h = o;
    }
    const nr = net.ym.length, d = h.length; const r = [];
    for (let j = 0; j < 2 * nr; j++) { let s = net.br[j]; for (let i = 0; i < d; i++) s += net.Wr[j * d + i] * h[i]; r.push(s); }
    const nc = net.bc.length, lg = [];
    for (let j = 0; j < nc; j++) { let s = net.bc[j]; for (let i = 0; i < d; i++) s += net.Wc[j * d + i] * h[i]; lg.push(s); }
    const mu = r.slice(0, nr).map((q, j) => q * net.ys[j] + net.ym[j]);
    const sg = r.slice(nr).map((q, j) => Math.exp(Math.min(Math.max(q, -6), 3)) * net.ys[j] * net.widen[j]);
    const sm = (a, T) => { const m = Math.max(...a); const e = a.map(q => Math.exp((q - m) / T)); const s = e.reduce((p, q) => p + q, 0); return e.map(q => q / s); };
    const nu = USES.length, nh = HEAT.length;
    return { mu, sg, use: sm(lg.slice(0, nu), net.temp[0]), heat: sm(lg.slice(nu, nu + nh), net.temp[1]), cool: sm(lg.slice(nu + nh), net.temp[2]) };
  }
  root.BD = { featHourly, monthlyFromHourly, featMonthly, featAnnual, runNet, prism, CH, USES, HEAT, COOL, REG, N_W_H, N_C_H };
  if (typeof module !== 'undefined') module.exports = root.BD;
})(typeof window !== 'undefined' ? window : globalThis);
