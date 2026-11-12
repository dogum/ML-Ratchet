// BeatNet's heart simulator in the browser: the same physics as src/heart.py + src/sim.py (one reference heart).
// Eikonal activation by Dijkstra on the 26-neighbour voxel graph, transmembrane-potential dipole
// H(t) = sum[f'(t - tau) grad tau - g'(t - r) grad r] dV, Dower lead II, real noise from the MIT-BIH NSTDB.
const BeatSim = (() => {
  let G = null;   // reference heart (from webdata.py)
  const HV = 2.0, DT = 1.0, MYO = 0, LVF = 1, RVF = 2, ATR = 3;
  function b64i16(s) { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new Int16Array(u.buffer); }
  function b64f32(s) { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new Float32Array(u.buffer); }
  function init(pack) {
    const ijk = b64i16(pack.ijk), n = ijk.length / 3, [S0, S1, S2] = pack.shape, idx = new Int32Array(S0 * S1 * S2).fill(-1);
    for (let v = 0; v < n; v++) idx[(ijk[3 * v] * S1 + ijk[3 * v + 1]) * S2 + ijk[3 * v + 2]] = v;
    const cls = Uint8Array.from(atob(pack.cls), (c) => c.charCodeAt(0)), atr = Uint8Array.from(cls, (c) => (c === ATR ? 1 : 0));
    const e = b64f32(pack.e), b = b64f32(pack.b);
    // 26-neighbour graph (no atrial-ventricular edges); edge class = shared class, else myocardium of that chamber
    const off = []; for (let a = -1; a <= 1; a++) for (let c = -1; c <= 1; c++) for (let d = -1; d <= 1; d++) if (a || c || d) off.push([a, c, d, HV * Math.hypot(a, c, d)]);
    const start = new Int32Array(n + 1), J = [], Lg = [], Ty = [];
    for (let v = 0; v < n; v++) { start[v] = J.length; const i = ijk[3 * v], j = ijk[3 * v + 1], k = ijk[3 * v + 2];
      for (const [a, c, d, l] of off) { const x = i + a, y = j + c, z = k + d; if (x < 0 || y < 0 || z < 0 || x >= S0 || y >= S1 || z >= S2) continue;
        const u = idx[(x * S1 + y) * S2 + z]; if (u < 0 || atr[u] !== atr[v]) continue; J.push(u); Lg.push(l); Ty.push(cls[u] === cls[v] ? cls[v] : (atr[v] ? ATR : MYO)); } }
    start[n] = J.length;
    const nbr = []; for (let a = 0; a < 3; a++) for (const s of [1, -1]) { const t = new Int32Array(n).fill(-1);
      for (let v = 0; v < n; v++) { const p = [ijk[3 * v], ijk[3 * v + 1], ijk[3 * v + 2]]; p[a] += s; if (p[a] < 0 || p[a] >= pack.shape[a]) continue;
        const u = idx[(p[0] * S1 + p[1]) * S2 + p[2]]; if (u >= 0 && atr[u] === atr[v]) t[v] = u; } nbr.push(t); }
    const pos = new Float32Array(3 * n); for (let v = 0; v < n; v++) for (let a = 0; a < 3; a++) pos[3 * v + a] = pack.lo[a] + ijk[3 * v + a] * HV;
    G = { ...pack, n, ijk, cls, atr, e, b, start, J: Int32Array.from(J), Lg: Float32Array.from(Lg), Ty: Uint8Array.from(Ty), nbr, pos, cache: {} };
    return G;
  }
  // ---------- eikonal: Dijkstra with a binary heap
  function activate(sites, t0, cv) {
    const n = G.n, dist = new Float64Array(n).fill(Infinity), done = new Uint8Array(n), hk = [], hv = [];
    const push = (k, v) => { hk.push(k); hv.push(v); let i = hk.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (hk[p] <= hk[i]) break; [hk[p], hk[i]] = [hk[i], hk[p]]; [hv[p], hv[i]] = [hv[i], hv[p]]; i = p; } };
    const pop = () => { const k = hk[0], v = hv[0], lk = hk.pop(), lv = hv.pop(); if (hk.length) { hk[0] = lk; hv[0] = lv; let i = 0; for (;;) { const l = 2 * i + 1, r = l + 1; let m = i;
      if (l < hk.length && hk[l] < hk[m]) m = l; if (r < hk.length && hk[r] < hk[m]) m = r; if (m === i) break; [hk[m], hk[i]] = [hk[i], hk[m]]; [hv[m], hv[i]] = [hv[i], hv[m]]; i = m; } } return [k, v]; };
    sites.forEach((s, i) => { if (t0[i] < dist[s]) { dist[s] = t0[i]; push(t0[i], s); } });
    while (hk.length) { const [d, v] = pop(); if (done[v]) continue; done[v] = 1;
      for (let q = G.start[v]; q < G.start[v + 1]; q++) { const u = G.J[q], nd = d + G.Lg[q] / cv[G.Ty[q]]; if (nd < dist[u]) { dist[u] = nd; push(nd, u); } } }
    return dist;
  }
  function grad(f) {   // f: Float64Array with NaN outside the mask
    const n = G.n, g = new Float64Array(3 * n);
    for (let a = 0; a < 3; a++) { const jp = G.nbr[2 * a], jm = G.nbr[2 * a + 1];
      for (let v = 0; v < n; v++) { const fp = jp[v] >= 0 ? f[jp[v]] : NaN, fm = jm[v] >= 0 ? f[jm[v]] : NaN, c = (fp - fm) / (2 * HV), fw = (fp - f[v]) / HV, bw = (f[v] - fm) / HV;
        let val = isFinite(c) ? c : isFinite(fw) ? fw : isFinite(bw) ? bw : 0; g[3 * v + a] = isFinite(val) ? val : 0; } }
    return g;
  }
  function kernel(width) { const n = Math.ceil(8 * width / DT), k = []; let s = 0; for (let i = -n; i <= n; i++) { const sg = 1 / (1 + Math.exp(-i * DT / width)), v = sg * (1 - sg) / width; k.push(v); s += v; } return k.map((v) => v / s); }
  function sweep(times, vec, mask, L, width) {   // sum_i vec_i k(t - times_i)
    const out = [new Float64Array(L), new Float64Array(L), new Float64Array(L)];
    for (let v = 0; v < G.n; v++) { if (!mask[v]) continue; const t = times[v]; if (!(isFinite(t) && t >= 0 && t < L - 2)) continue; const x = t / DT, i0 = Math.floor(x), fr = x - i0;
      for (let c = 0; c < 3; c++) { out[c][i0] += vec[3 * v + c] * (1 - fr); out[c][i0 + 1] += vec[3 * v + c] * fr; } }
    const k = kernel(width), h = k.length >> 1, res = new Float64Array(3 * L);
    for (let c = 0; c < 3; c++) for (let t = 0; t < L; t++) { let s = 0; for (let j = 0; j < k.length; j++) { const u = t + h - j; if (u >= 0 && u < L) s += out[c][u] * k[j]; } res[3 * t + c] = s; }
    return res;
  }
  const nanmax = (a, m) => { let x = -Infinity; for (let i = 0; i < a.length; i++) if ((!m || m[i]) && a[i] > x) x = a[i]; return x; };
  const nanmin = (a, m) => { let x = Infinity; for (let i = 0; i < a.length; i++) if ((!m || m[i]) && a[i] < x) x = a[i]; return x; };
  // ---------- activation maps (ventricular): kind normal | pvc ; bbb none | LBBB | RBBB
  function vmap(o) {
    const key = JSON.stringify(o); if (G.cache[key]) return G.cache[key];
    const cv = [o.myo, o.fast, o.fast, G.cv.atr]; let sites = [], t0 = [];
    if (o.kind === 'pvc') { cv[LVF] = cv[RVF] = Math.min(o.myo * o.engage, o.fast); sites = [o.focus]; t0 = [0]; }
    else { for (const k of ['lvs', 'lva', 'lvp', 'rv']) { if (o.bbb === 'LBBB' && k !== 'rv') continue; if (o.bbb === 'RBBB' && k === 'rv') continue; sites.push(G.roots[k]); t0.push(G.root_t[k]); }
      if (o.bbb === 'LBBB') cv[LVF] = o.myo * 2.2; if (o.bbb === 'RBBB') cv[RVF] = o.myo * 1.8; }
    const d = activate(sites, t0, cv), m = Uint8Array.from(G.atr, (a) => 1 - a), tau = new Float64Array(G.n).fill(NaN);
    for (let v = 0; v < G.n; v++) if (m[v] && isFinite(d[v])) tau[v] = d[v];
    const mn = nanmin(tau.map((x) => (isFinite(x) ? x : Infinity))); for (let v = 0; v < G.n; v++) tau[v] -= mn;
    G.cache[key] = tau; return tau;
  }
  function amap(focus) { const k = 'a' + focus; if (G.cache[k]) return G.cache[k]; const d = activate([focus], [0], [1, 1, 1, G.cv.atr]), t = new Float64Array(G.n).fill(NaN);
    for (let v = 0; v < G.n; v++) if (G.atr[v] && isFinite(d[v])) t[v] = d[v]; G.cache[k] = t; return t; }
  const fin = (a) => Uint8Array.from(a, (x) => (isFinite(x) ? 1 : 0));
  function dipoleParts(tau, rt, wr) {   // heart-frame (Hd, Hr) for maps tau and rt
    const m = fin(tau), L = Math.floor(nanmax(rt, m) + 6 * wr) + 10, vol = HV ** 3;
    const gt = grad(Float64Array.from(tau, (x, i) => (m[i] ? x : NaN))), gr = grad(Float64Array.from(rt, (x, i) => (m[i] ? x : NaN)));
    const Hd = sweep(tau, gt, m, L, 1.5), Hr = sweep(rt, gr, m, L, wr); for (let i = 0; i < Hd.length; i++) { Hd[i] *= vol; Hr[i] *= -vol; } return [Hd, Hr, L];
  }
  function medianOf(a) { const s = Array.from(a).filter((x) => isFinite(x)).sort((x, y) => x - y), n = s.length; return n % 2 ? s[(n - 1) >> 1] : 0.5 * (s[n / 2 - 1] + s[n / 2]); }
  function scale() { if (G.cache.scale) return G.cache.scale; const tn = vmap({ kind: 'normal', bbb: 'none', myo: G.cv.myo, fast: G.cv.fast }), m = fin(tn), g = grad(Float64Array.from(tn, (x, i) => (m[i] ? x : NaN)));
    const H = sweep(tn, g, m, 250, 1.5); let mx = 0; for (let t = 0; t < 250; t++) mx = Math.max(mx, Math.hypot(H[3 * t], H[3 * t + 1], H[3 * t + 2]) * HV ** 3); G.cache.scale = G.scale_target / mx; return G.cache.scale; }
  function vparts(tau) {   // cached ventricular dipole parts at RR = 1 s (heart frame, scaled to mV)
    if (tau._parts) return tau._parts; const r = G.rep, med = medianOf(tau), rt = new Float64Array(G.n);
    for (let v = 0; v < G.n; v++) rt[v] = tau[v] + r.apd1 + r.tm * (1 - G.e[v]) + r.ab * (G.b[v] - 0.5) - r.k * (tau[v] - med);
    const [Hd, Hr, L] = dipoleParts(tau, rt, r.wr), s = scale(); for (let i = 0; i < Hd.length; i++) { Hd[i] *= s; Hr[i] *= s; } tau._parts = [Hd, Hr, L]; return tau._parts;
  }
  function vbeat(tau, rr, gain) {   // heart-frame dipole (3*len) for a beat after an RR of rr s
    const [Hd, Hr, L] = vparts(tau), sh = Math.round(G.rep.apd1 * (Math.pow(Math.min(2, Math.max(0.25, rr)), 0.35) - 1)), n = Math.max(L, L + sh), out = new Float64Array(3 * n);
    for (let i = 0; i < 3 * L; i++) out[i] = Hd[i] * gain; for (let t = 0; t < L; t++) { const u = t + sh; if (u >= 0 && u < n) for (let c = 0; c < 3; c++) out[3 * u + c] += Hr[3 * t + c] * gain; } return out;
  }
  function abeat(focus) { const k = 'ab' + focus; if (G.cache[k]) return G.cache[k]; const tau = amap(focus), rt = Float64Array.from(tau, (x) => x + G.rep.apda);
    const [Hd, Hr] = dipoleParts(tau, rt, G.rep.wra), s = scale() * G.atr_gain; for (let i = 0; i < Hd.length; i++) Hd[i] = (Hd[i] + (Hr[i] || 0)) * s; G.cache[k] = Hd; return Hd; }
  function qrsWidth(tau) { if (tau._qw) return tau._qw; const m = fin(tau), L = Math.floor(nanmax(tau, m)) + 40, g = grad(Float64Array.from(tau, (x, i) => (m[i] ? x : NaN))), H = sweep(tau, g, m, L, 1.5);
    const mag = []; for (let t = 0; t < L; t++) mag.push(Math.hypot(H[3 * t], H[3 * t + 1], H[3 * t + 2])); const mx = Math.max(...mag); let a = -1, b = -1;
    mag.forEach((v, t) => { if (v > 0.05 * mx) { if (a < 0) a = t; b = t; } }); tau._qw = (b - a) * DT; return tau._qw; }
  // ---------- rotations and lead II
  function rot(ax, ang) { const n = Math.hypot(...ax), [x, y, z] = ax.map((v) => v / n), c = Math.cos(ang), s = Math.sin(ang), C = 1 - c;
    return [[c + x * x * C, x * y * C - z * s, x * z * C + y * s], [y * x * C + z * s, c + y * y * C, y * z * C - x * s], [z * x * C - y * s, z * y * C + x * s, c + z * z * C]]; }
  const mm = (A, B) => A.map((r) => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
  function leadVector(frontalDeg) {   // heart frame -> body (patient orientation, then extra frontal rotation) -> Dower lead II
    const M = mm(rot([0, 0, 1], frontalDeg * Math.PI / 180), G.M), L2 = [0.235, 1.066, -0.132]; return [0, 1, 2].map((j) => L2.reduce((s, l, i) => s + l * M[i][j], 0));
  }
  // ---------- one strip: 16 beats of sinus rhythm, beat 9 (index 8) replaced by the chosen ectopic beat (and every other beat after it in bigeminy)
  function strip(o) {   // o: {type: 'N'|'S'|'V'|'F', focus, afocus, myo, engage, bbb, frontal, coupling, hr, noise: 'none'|'bw'|'em'|'ma'|'all', snr, seed}
    const rr0 = 60 / o.hr * 1000, PR = G.PR, nb = 16, Vev = [], Aev = []; let tp = 400, k = 0;
    const base = { kind: 'normal', bbb: o.bbb, myo: o.myo, fast: G.cv.fast }, tn = vmap(base);
    while (Vev.length < nb) {
      const ect = (k === 8 || (o.bigeminy && k > 8 && (k - 8) % 2 === 0)) && o.type !== 'N';
      if (ect && o.type === 'V') { const tv = Vev[Vev.length - 1].t + o.coupling * rr0; Vev.push({ t: tv, tau: vmap({ kind: 'pvc', focus: o.focus, myo: o.myo, fast: G.cv.fast, engage: o.engage }), label: 2, gain: G.vgain });
        Aev.push({ t: tp, f: G.sinus }); tp += rr0; Vev.push({ t: tp + PR, tau: tn, label: 0, gain: 1 }); Aev.push({ t: tp, f: G.sinus }); tp += rr0; k += 2; continue; }
      if (ect && o.type === 'S') { const ta = tp - rr0 + o.coupling * rr0; Aev.push({ t: ta, f: o.afocus }); Vev.push({ t: ta + PR + 30 * (1 - o.coupling) / 0.5, tau: tn, label: 1, gain: 1 }); tp = ta + rr0; k++; continue; }
      if (ect && o.type === 'F') { const te = vmap({ kind: 'pvc', focus: o.focus, myo: o.myo, fast: G.cv.fast, engage: o.engage }), d = (o.coupling - 1) * 200;
        const fu = Float64Array.from(tn, (x, i) => Math.min(x, te[i] + d)); Vev.push({ t: tp + PR, tau: fu, label: 3, gain: 1 }); Aev.push({ t: tp, f: G.sinus }); tp += rr0; k++; continue; }
      Vev.push({ t: tp + PR, tau: tn, label: 0, gain: 1 }); Aev.push({ t: tp, f: G.sinus }); tp += rr0; k++;
    }
    const T = Math.floor(Vev[Vev.length - 1].t + 1500), H = new Float64Array(3 * T), widths = [];
    Vev.forEach((e, i) => { const rr = i ? (e.t - Vev[i - 1].t) / 1000 : 0.8, d = vbeat(e.tau, rr, e.gain), t0 = Math.floor(e.t); e.rr = rr; widths.push(qrsWidth(e.tau));
      for (let j = 0; j < d.length && 3 * t0 + j < H.length; j++) H[3 * t0 + j] += d[j]; });
    for (const a of Aev) { const d = abeat(a.f), t0 = Math.floor(a.t); for (let j = 0; j < d.length && 3 * t0 + j < H.length; j++) if (t0 >= 0) H[3 * t0 + j] += d[j]; }
    const lv = leadVector(o.frontal), v1k = new Float64Array(T); for (let t = 0; t < T; t++) v1k[t] = lv[0] * H[3 * t] + lv[1] * H[3 * t + 1] + lv[2] * H[3 * t + 2];
    // 1 kHz -> 360 Hz: windowed-sinc low-pass at 150 Hz, then linear interpolation
    const fk = []; for (let j = -40; j <= 40; j++) { const x = j * 0.3; fk.push((j ? Math.sin(Math.PI * x) / (Math.PI * x) : 1) * 0.3 * (0.54 + 0.46 * Math.cos(Math.PI * j / 40))); }
    const lp = new Float64Array(T); for (let t = 0; t < T; t++) { let s = 0; for (let j = 0; j < fk.length; j++) { const u = t + j - 40; if (u >= 0 && u < T) s += v1k[u] * fk[j]; } lp[t] = s; }
    const n = Math.floor(T * 0.36), clean = new Float64Array(n); for (let i = 0; i < n; i++) { const x = i / 0.36, a = Math.floor(x), f = x - a; clean[i] = lp[a] * (1 - f) + (lp[Math.min(a + 1, T - 1)] || 0) * f; }
    // annotations at the clean QRS peak (as in strip.record)
    const cb = BNet.filt360 ? filtFull(clean) : clean, ann = Vev.map((e, i) => { const on = Math.round(e.t * 0.36), w = Math.max(Math.round(widths[i] * 0.36), 8); let b = on, mx = -1;
      for (let j = on; j < Math.min(on + w, n); j++) if (Math.abs(cb[j]) > mx) { mx = Math.abs(cb[j]); b = j; } return b; });
    let qamp = []; for (const a of ann) { let mx = -Infinity, mn = Infinity; for (let j = Math.max(a - 25, 0); j < Math.min(a + 25, n); j++) { mx = Math.max(mx, cb[j]); mn = Math.min(mn, cb[j]); } qamp.push(mx - mn); }
    qamp = BNet.median(qamp);
    const x = Float64Array.from(clean), rng = mulberry(o.seed || 1);
    if (o.noise !== 'none') { const tot = qamp * Math.pow(10, -o.snr / 20), types = o.noise === 'all' ? ['bw', 'em', 'ma'] : [o.noise];
      for (const k of types) { const N = G.noise[k], s0 = Math.floor(rng() * (N.d.length - n - 1)), w = 1 / Math.sqrt(types.length);
        let mean = 0; for (let i = 0; i < n; i++) mean += N.d[s0 + i]; mean /= n; const lvl = tot * w / N.r * (k === 'bw' ? 3 : 1);
        for (let i = 0; i < n; i++) x[i] += (N.d[s0 + i] - mean) * lvl; } }
    for (let i = 0; i < n; i++) x[i] += 0.002 * gauss(rng);
    const y = recorder(x);
    return { x: y, clean, ann, labels: Vev.map((e) => e.label), widths, tauEct: Vev.find((e) => e.label > 0)?.tau || tn, tauN: tn, ev: Vev, Aev };
  }
  function filtFull(x) { // zero-phase 0.5-40 Hz at 360 Hz (no decimation), same filter as the front end
    const y = BNet._filtfilt ? BNet._filtfilt(x) : x; return y; }
  function recorder(x) {   // single-pole high-pass 0.3 Hz, 2-pole Butterworth low-pass 80 Hz, 5 uV ADC
    const [hb, ha] = G.rec.hp, [lb, la] = G.rec.lp, y = new Float64Array(x.length); let z0 = 0;
    for (let i = 0; i < x.length; i++) { const v = hb[0] * x[i] + z0; z0 = hb[1] * x[i] - ha[1] * v; y[i] = v; }
    let s0 = 0, s1 = 0; for (let i = 0; i < y.length; i++) { const xi = y[i], v = lb[0] * xi + s0; s0 = lb[1] * xi - la[1] * v + s1; s1 = lb[2] * xi - la[2] * v; y[i] = Math.round(v * 200) / 200; }
    return y;
  }
  function mulberry(a) { return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function gauss(r) { return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r()); }
  // focus picker: ventricular voxel nearest to a point given by height (0 apex .. 1 base), angle around the long axis and depth (0 endo .. 1 epi)
  function pickFocus(height, angDeg, depth) { let best = -1, bd = Infinity; const a = angDeg * Math.PI / 180, w = G.apex * (1 - height);
    for (let v = 0; v < G.n; v++) { if (G.atr[v]) continue; const p = G.pos[3 * v], q = G.pos[3 * v + 1], ww = G.pos[3 * v + 2], ang = Math.atan2(q, p);
      const da = Math.atan2(Math.sin(ang - a), Math.cos(ang - a)), d = (ww - w) ** 2 / 25 + (da * 30) ** 2 + ((G.e[v] - depth) * 20) ** 2; if (d < bd) { bd = d; best = v; } } return best; }
  function pickAtrial(angDeg) { let best = -1, bd = Infinity; const a = angDeg * Math.PI / 180, c = [G.pos[3 * G.avn], G.pos[3 * G.avn + 1]];
    for (let v = 0; v < G.n; v++) { if (!G.atr[v]) continue; const ang = Math.atan2(G.pos[3 * v + 2] - 20, G.pos[3 * v] - 5), d = Math.abs(Math.atan2(Math.sin(ang - a), Math.cos(ang - a))) - 0.002 * G.pos[3 * v + 2]; if (d < bd) { bd = d; best = v; } } return best; }
  function bodyM(frontalDeg) { return mm(rot([0, 0, 1], frontalDeg * Math.PI / 180), G.M); }
  return { init, activate, vmap, amap, vparts, vbeat, abeat, qrsWidth, strip, pickFocus, pickAtrial, leadVector, bodyM, get G() { return G; } };
})();
if (typeof module !== 'undefined') module.exports = BeatSim;
