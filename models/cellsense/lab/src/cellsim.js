// CellSense physics in the browser: the same single-particle model as cellsim.py (one cell at a time).
// LFP (Afshar 2017) and graphite (Chen 2020) open-circuit curves, smeared over local states of charge; exact modal
// diffusion in each particle; Butler-Volmer kinetics; slow electrolyte/transport polarization; a slow, poorly connected
// graphite population; lumped thermal model; 4C discharge to 2.0 V, then a hold at 2.0 V down to C/50.
const CellSim = (() => {
  const F = 96485.33212, Rg = 8.314462618;
  const Up = (x) => 3.4077 - 0.020269 * x + 0.5 * Math.exp(-150 * x) - 0.9 * Math.exp(-30 * (1 - x));
  const Un = (y) => 1.9793 * Math.exp(-39.3631 * y) + 0.2482 - 0.0909 * Math.tanh(29.8538 * (y - 0.1234)) - 0.04478 * Math.tanh(14.9159 * (y - 0.2769)) - 0.0205 * Math.tanh(30.4444 * (y - 0.6103));
  let GH = null, SPH = null;
  const clip = (v, a, b) => Math.min(b, Math.max(a, v));
  function table(U, sig, n = 2001) {
    const T = new Float64Array(n);
    for (let i = 0; i < n; i++) { const s = i / (n - 1); let v = 0; for (let k = 0; k < GH.z.length; k++) v += GH.w[k] * U(clip(s + sig * GH.z[k], 1e-6, 1 - 1e-6)); T[i] = v; }
    return (s) => { const f = clip(s, 0, 1) * (n - 1), i = Math.min(n - 2, Math.floor(f)), w = f - i; return T[i] * (1 - w) + T[i + 1] * w; };
  }
  function fullState(P, TP, TN, V) {
    let lo = 1e-6, hi = Math.min(1 - 1e-6, P.Q_Li / P.Q_p);
    for (let it = 0; it < 60; it++) { const x = 0.5 * (lo + hi), y = clip((P.Q_Li - x * P.Q_p) / P.Q_n, 1e-6, 1 - 1e-6); if (TP(x) - TN(y) < V) hi = x; else lo = x; }
    const x = 0.5 * (lo + hi); return [x, clip((P.Q_Li - x * P.Q_p) / P.Q_n, 1e-6, 1 - 1e-6)];
  }
  // modal sphere: state a (N); c = Phi a
  const sInit = (c0) => SPH.init.map((v) => v * c0);
  function sStep(a, h, rate) { const o = new Float64Array(a.length); for (let j = 0; j < a.length; j++) { const l = SPH.lam[j], E = Math.exp(l * h), G = l < 0 ? (E - 1) / l : h; o[j] = E * a[j] + G * SPH.b[j] * rate; } return o; }
  const sSurf = (a) => { let v = 0; for (let j = 0; j < a.length; j++) v += a[j] * SPH.s[j]; return v; };
  function simulate(P, opt = {}) {
    const I = opt.I || 4.4, dt = opt.dt || 3.0, Vmin = 2.0, Icut = 0.022;
    const TP = table(Up, P.sig_p), TN = table(Un, P.sig_n); const [x0, y0] = fullState(P, TP, TN, P.V_full);
    let cp = sInit(x0), cn = sInit(y0), cs = sInit(y0), T = P.T0, ve = 0, Q = 0, inCV = false, Qcc = NaN, Vprev = 9.9, Iprev = I;
    const fs = P.f_s || 0, Rs = P.R_s || 1, slow = fs > 1e-6, Qnf = P.Q_n * (1 - fs), Qns = Math.max(P.Q_n * fs, 1e-9), I0f = P.I0_n * (1 - fs), I0s = Math.max(P.I0_n * fs, 1e-9);
    const arr = (Ea, TK) => Math.exp(Ea / Rg * (1 / 298.15 - 1 / TK)); const out = { V: [], Q: [], T: [], cc: [] };
    const steps = Math.ceil((3600 * 1.2 * Math.min(P.Q_Li, P.Q_p) / I + 2400) / dt) + 1;
    for (let k = 0; k < steps; k++) {
      const TK = T + 273.15, Re = P.R_e * arr(P.Ea_e, TK), Ro = P.R_ohm, tp = P.tau_p / arr(P.Ea_dp, TK), tn = P.tau_n / arr(P.Ea_dn, TK), akp = arr(P.Ea_kp, TK), akn = arr(P.Ea_kn, TK), c2 = 2 * Rg * TK / F;
      const split = (Ii, ys, yss) => { const i0f = I0f * akn * 2 * Math.sqrt(ys * (1 - ys)), i0s = I0s * akn * 2 * Math.sqrt(yss * (1 - yss));
        const rf = c2 / Math.sqrt(4 * i0f * i0f + (Ii * (1 - fs)) ** 2), rs = Rs + c2 / (2 * i0s), Uf = TN(ys), Us = TN(yss), phi = (Ii + Uf / rf + Us / rs) / (1 / rf + 1 / rs);
        return [slow ? (phi - Us) / rs : 0, i0f]; };
      const volt = (Ii, xs, ys, yss) => { xs = clip(xs, 1e-6, 1 - 1e-6); ys = clip(ys, 1e-6, 1 - 1e-6); yss = clip(yss, 1e-6, 1 - 1e-6);
        const [Is, i0f] = split(Ii, ys, yss), If = Ii - Is, i0p = P.I0_p * akp * 2 * Math.sqrt(xs * (1 - xs));
        const eta = c2 * (Math.asinh(Ii / (2 * i0p)) + Math.asinh(If / (2 * i0f))), o = TP(xs) - TN(ys); return [o - eta - Ii * Ro - ve, o, Is]; };
      let Ik = I; let [Vcc, ocv, Isk] = volt(Ik, sSurf(cp), sSurf(cn), sSurf(cs));
      if (!inCV && Vcc <= Vmin) { const f = k ? clip((Vprev - Vmin) / Math.max(Vprev - Vcc, 1e-9), 0, 1) : 1; Qcc = Q - (1 - f) * I * dt / 3600; inCV = true; }
      Vprev = Vcc;
      if (inCV) {
        const x0s = sSurf(sStep(cp, dt / tp, 0)), xd = sSurf(sStep(cp, dt / tp, 1 / 3600 / P.Q_p * tp)) - x0s, y0s = sSurf(sStep(cn, dt / tn, 0)), yd = sSurf(sStep(cn, dt / tn, -1 / 3600 / Qnf * tn)) - y0s, yss = sSurf(cs);
        let lo = 0, hi = Math.min(I, 1.2 * Iprev + 0.01);
        for (let it = 0; it < 17; it++) { const mid = 0.5 * (lo + hi), Ism = split(mid, clip(y0s + mid * yd, 1e-6, 1 - 1e-6), clip(yss, 1e-6, 1 - 1e-6))[0];
          if (volt(mid, x0s + mid * xd, y0s + (mid - Ism) * yd, yss)[0] - Vmin > 0) lo = mid; else hi = mid; }
        Ik = 0.5 * (lo + hi); if (Ik < Icut) break; Iprev = Ik;
        const Iscv = split(Ik, clip(y0s + Ik * yd, 1e-6, 1 - 1e-6), clip(yss, 1e-6, 1 - 1e-6))[0], o = volt(Ik, x0s + Ik * xd, y0s + (Ik - Iscv) * yd, yss); ocv = o[1]; Isk = o[2];
      }
      const V = inCV ? Vmin : Vcc; out.V.push(V); out.Q.push(Q); out.T.push(T); out.cc.push(!inCV);
      Q += Ik * dt / 3600;
      cp = sStep(cp, dt / tp, Ik / 3600 / P.Q_p * tp); cn = sStep(cn, dt / tn, -(Ik - Isk) / 3600 / Qnf * tn); cs = sStep(cs, dt / tn, -Isk / 3600 / Qns * tn);
      ve += (Ik * Re - ve) * (1 - Math.exp(-dt / P.tau_e)); T += dt * (Ik * (ocv - V) - P.hA * (T - P.T_amb)) / P.C_th;
    }
    out.Q_cc = isFinite(Qcc) ? Qcc : Q; out.Q_total = Q; return out;
  }
  function qOfV(out, vgrid) {   // constant-current part, monotone voltage, as cellsim.q_of_v
    const v = [], q = [], t = []; out.V.forEach((x, i) => { if (out.cc[i]) { v.push(x); q.push(out.Q[i]); t.push(out.T[i]); } });
    for (let i = 1; i < v.length; i++) v[i] = Math.min(v[i], v[i - 1]);
    const Qg = new Float32Array(vgrid.length).fill(NaN), Tg = new Float32Array(vgrid.length).fill(NaN);
    vgrid.forEach((g, j) => { if (g > v[0]) { Qg[j] = 0; Tg[j] = t[0]; return; } if (g < v[v.length - 1]) return;
      let i = 1; while (i < v.length - 1 && v[i] > g) i++; const a = v[i - 1], b = v[i], w = a === b ? 0 : (a - g) / (a - b); Qg[j] = q[i - 1] + w * (q[i] - q[i - 1]); Tg[j] = t[i - 1] + w * (t[i] - t[i - 1]); });
    return [Qg, Tg];
  }
  function pulseR(P) {
    const TP = table(Up, P.sig_p), TN = table(Un, P.sig_n), [x0, y0] = fullState(P, TP, TN, P.V_full);
    const usable = Math.min(P.Q_Li - 1e-3 * P.Q_n - x0 * P.Q_p, (1 - x0) * P.Q_p), dq = 0.2 * usable, x = clip(x0 + dq / P.Q_p, 1e-4, 1 - 1e-4), y = clip(y0 - dq / P.Q_n, 1e-4, 1 - 1e-4), TK = 303.15;
    const arr = (Ea) => Math.exp(Ea / Rg * (1 / 298.15 - 1 / TK)), c2 = 2 * Rg * TK / F, Ip = 3.96;
    return P.R_ohm + c2 * (Math.asinh(Ip / (2 * P.I0_p * arr(P.Ea_kp) * 2 * Math.sqrt(x * (1 - x)))) + Math.asinh(Ip / (2 * P.I0_n * arr(P.Ea_kn) * 2 * Math.sqrt(y * (1 - y))))) / Ip;
  }
  return { setup: (gh, sph) => { GH = gh; SPH = sph; }, simulate, qOfV, pulseR, fullState, table, Up, Un };
})();
if (typeof module !== 'undefined') module.exports = CellSim;
