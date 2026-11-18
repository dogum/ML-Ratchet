// SnowSense network in plain JS (port of release/snowsense.py; verified against numpy by web/check_js.cjs).
// W: {weights: {name: nested arrays}, k: {daily, peak, melt}}
const SS = (() => {
  const L = 427, NCH = 10, Z68 = 0.9945, Z90 = 1.6449;
  const sig = x => 1 / (1 + Math.exp(-x));
  const softplus = x => Math.log1p(Math.exp(-Math.abs(x))) + Math.max(x, 0);
  function dayOfYear(d) { // d: Date (UTC)
    const s = Date.UTC(d.getUTCFullYear(), 0, 1);
    return Math.floor((d - s) / 864e5) + 1;
  }
  // window features; P,Tm,Tn,Tx arrays of length n; startISO = 'YYYY-08-01'; mP,mT gap flags (0/1)
  function features(P, Tm, Tn, Tx, mP, mT, startISO, lat, elev) {
    const n = P.length, X = new Array(n);
    const t0 = Date.parse(startISO + 'T00:00:00Z');
    for (let i = 0; i < n; i++) {
      const doy = dayOfYear(new Date(t0 + i * 864e5));
      const a = 2 * Math.PI * (doy - 1) / 365.25;
      X[i] = [Math.log1p(Math.max(P[i], 0)) / 2, Tm[i] / 10, Tn[i] / 10, Tx[i] / 10, Math.sin(a), Math.cos(a),
              mP ? mP[i] : 0, mT ? mT[i] : 0, (elev - 2200) / 800, (lat - 43) / 8];
    }
    return X;
  }
  function matvec(M, v, b) { // M: rows x cols
    const r = new Float64Array(M.length);
    for (let i = 0; i < M.length; i++) {
      const row = M[i]; let s = b ? b[i] : 0;
      for (let j = 0; j < v.length; j++) s += row[j] * v[j];
      r[i] = s;
    }
    return r;
  }
  function forward(W, X) {
    const w = W.weights;
    let h = X.map(x => Array.from(matvec(w['inp.weight'], x, w['inp.bias']), Math.tanh));
    for (let l = 0; ('gru.weight_ih_l' + l) in w; l++) {
      const Wi = w['gru.weight_ih_l' + l], Wh = w['gru.weight_hh_l' + l];
      const bi = w['gru.bias_ih_l' + l], bh = w['gru.bias_hh_l' + l];
      const H = Wh[0].length;
      let hs = new Float64Array(H);
      const out = new Array(h.length);
      for (let t = 0; t < h.length; t++) {
        const gi = matvec(Wi, h[t], bi), gh = matvec(Wh, hs, bh);
        const nh = new Float64Array(H);
        for (let j = 0; j < H; j++) {
          const r = sig(gi[j] + gh[j]);
          const z = sig(gi[H + j] + gh[H + j]);
          const nn = Math.tanh(gi[2 * H + j] + r * gh[2 * H + j]);
          nh[j] = (1 - z) * nn + z * hs[j];
        }
        hs = nh; out[t] = nh;
      }
      h = out;
    }
    const Q = new Array(h.length);
    for (let t = 0; t < h.length; t++) {
      const o = matvec(w['head.weight'], h[t], w['head.bias']);
      if (o.length === 2) {
        const m = o[0], sg = softplus(o[1]) + 0.01;
        Q[t] = [m - Z90 * sg, m - Z68 * sg, m, m + Z68 * sg, m + Z90 * sg];
      } else {
        const d = [1, 2, 3, 4].map(i => softplus(o[i]) + 1e-3);
        Q[t] = [o[0] - d[0] - d[1], o[0] - d[0], o[0], o[0] + d[2], o[0] + d[2] + d[3]];
      }
    }
    return Q;
  }
  const band = (q, j, k = 1) => 100 * Math.max(q[2] + k * (q[j] - q[2]), 0) ** 2;
  function meltOut(arr, thresh = 5) {
    let im = 0; for (let i = 1; i < arr.length; i++) if (arr[i] > arr[im]) im = i;
    for (let i = im; i < arr.length; i++) if (arr[i] < thresh) return i;
    return null;
  }
  // run one water-year window; returns per-day bands for the water-year days (index 61 .. 61+nwy-1)
  function run(W, win) {
    const X = features(win.P, win.Tm, win.Tn, win.Tx, win.mP, win.mT, win.start, win.lat, win.elev);
    const Qa = forward(W, X).slice(61, 61 + win.nwy);
    const kd = W.k.daily, kp = W.k.peak, km = W.k.melt;
    const tr = (j, k) => Qa.map(q => band(q, j, k));
    const out = { swe: tr(2, 1), lo68: tr(1, kd), hi68: tr(3, kd), lo90: tr(0, kd), hi90: tr(4, kd) };
    const med = out.swe;
    let ip = 0; for (let i = 1; i < med.length; i++) if (med[i] > med[ip]) ip = i;
    out.peak = med[ip]; out.peakDay = ip; out.melt = meltOut(med);
    out.peakLo = Math.max(...tr(1, kp)); out.peakHi = Math.max(...tr(3, kp));
    out.meltLo = meltOut(tr(1, km)); out.meltHi = meltOut(tr(3, km));
    return out;
  }
  return { features, forward, run, L, Z68, Z90 };
})();
if (typeof module !== 'undefined') module.exports = SS;
