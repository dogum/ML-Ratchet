// StarQuake in the browser: processing (sq.py), the network (starquake.py), and the simulator for the lab (starsim.py).
// A line-by-line port; JS == Python is checked in web/check.js.
(function (G) {
'use strict';
const DT = 1765.4637, NYQ = 1e6 / (2 * DT), NB = 1024, NC = 48, STEP = 100, HALF = 3.5, NW = 700, L0 = 56, NL = 110;
const EDGES = new Float64Array(NB + 1); for (let i = 0; i <= NB; i++) EDGES[i] = Math.exp(Math.log(3) + (Math.log(283) - Math.log(3)) * i / NB);
const FC = new Float64Array(NB); for (let i = 0; i < NB; i++) FC[i] = Math.sqrt(EDGES[i] * EDGES[i + 1]);
const CEN = new Float64Array(NC); for (let j = 0; j < NC; j++) CEN[j] = Math.exp(Math.log(6) + (Math.log(260) - Math.log(6)) * j / (NC - 1));
const dnuS = nm => 0.276 * Math.pow(nm, 0.751);

// ---------- FFT: radix-2 in place, Bluestein for any length ----------
function fft2(re, im, inv) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) { let b = n >> 1; for (; j & b; b >>= 1) j ^= b; j ^= b; if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (inv ? 2 : -2) * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang), h = len >> 1;
    for (let i = 0; i < n; i += len) { let cr = 1, ci = 0; for (let k = 0; k < h; k++) { const a = i + k, b = a + h, xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr; re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi; const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t; } }
  }
}
function fftAny(xr, xi) {   // forward DFT of any length (Bluestein); returns [re, im]
  const n = xr.length; if ((n & (n - 1)) === 0) { const r = Float64Array.from(xr), i = Float64Array.from(xi); fft2(r, i, false); return [r, i]; }
  let m = 1; while (m < 2 * n - 1) m <<= 1;
  const cr = new Float64Array(n), ci = new Float64Array(n);
  for (let k = 0; k < n; k++) { const a = Math.PI * ((k * k) % (2 * n)) / n; cr[k] = Math.cos(a); ci[k] = -Math.sin(a); }
  const ar = new Float64Array(m), ai = new Float64Array(m), br = new Float64Array(m), bi = new Float64Array(m);
  for (let k = 0; k < n; k++) { ar[k] = xr[k] * cr[k] - xi[k] * ci[k]; ai[k] = xr[k] * ci[k] + xi[k] * cr[k]; }
  br[0] = cr[0]; bi[0] = -ci[0]; for (let k = 1; k < n; k++) { br[k] = br[m - k] = cr[k]; bi[k] = bi[m - k] = -ci[k]; }
  fft2(ar, ai, false); fft2(br, bi, false);
  for (let k = 0; k < m; k++) { const r = ar[k] * br[k] - ai[k] * bi[k], i = ar[k] * bi[k] + ai[k] * br[k]; ar[k] = r; ai[k] = i; }
  fft2(ar, ai, true); const or = new Float64Array(n), oi = new Float64Array(n);
  for (let k = 0; k < n; k++) { const r = ar[k] / m, i = ai[k] / m; or[k] = r * cr[k] - i * ci[k]; oi[k] = r * ci[k] + i * cr[k]; }
  return [or, oi];
}

// ---------- processing (sq.py) ----------
function median(a) { const s = Float64Array.from(a).sort(); const n = s.length; return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]); }
function fillShort(y, maxgap = 3) {
  const n = y.length, out = Float64Array.from(y); let first = -1, last = -1; for (let i = 0; i < n; i++) if (isFinite(y[i])) { if (first < 0) first = i; last = i; }
  if (first < 0) return out;
  let i = 0; while (i < n) { if (isFinite(y[i])) { i++; continue; } let j = i; while (j < n && !isFinite(y[j])) j++;
    if (i > 0 && j < n && j - i <= maxgap) { const a = y[i - 1], b = y[j]; for (let k = i; k < j; k++) out[k] = a + (b - a) * (k - (i - 1)) / (j - (i - 1)); }
    i = j; }
  return out;
}
function highpass(y, days = 10, clip = 5) {
  const n = y.length; let w = Math.round(days * 86400 / DT) | 1; const h = (w - 1) >> 1;
  const cx = new Float64Array(n + 1), cm = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) { const ok = isFinite(y[i]); cx[i + 1] = cx[i] + (ok ? y[i] : 0); cm[i + 1] = cm[i] + (ok ? 1 : 0); }
  const out = new Float64Array(n); const vals = [];
  for (let i = 0; i < n; i++) { if (!isFinite(y[i])) { out[i] = NaN; continue; } const a = Math.max(0, i - h), b = Math.min(n, i + h + 1); out[i] = y[i] - (cx[b] - cx[a]) / Math.max(cm[b] - cm[a], 1); vals.push(out[i]); }
  const med = median(vals); const mad = 1.4826 * median(vals.map(v => Math.abs(v - med)));
  for (let i = 0; i < n; i++) if (isFinite(out[i]) && Math.abs(out[i] - med) > clip * mad) out[i] = NaN;
  return out;
}
function psd(y) {
  const n = y.length; let s = 0, nv = 0; for (let i = 0; i < n; i++) if (isFinite(y[i])) { s += y[i]; nv++; }
  const mu = s / Math.max(nv, 1), xr = new Float64Array(n), xi = new Float64Array(n); for (let i = 0; i < n; i++) xr[i] = isFinite(y[i]) ? y[i] - mu : 0;
  const [R, I] = fftAny(xr, xi); const nf = Math.floor(n / 2); const f = new Float64Array(nf), P = new Float64Array(nf); const c = 2 * DT / Math.max(nv, 1) * 1e-6;
  for (let k = 1; k <= nf; k++) { f[k - 1] = k * 1e6 / (n * DT); P[k - 1] = c * (R[k] * R[k] + I[k] * I[k]); }
  return [f, P];
}
function makeRebin(f, P) {   // area-preserving rebin, like np.interp on the cumulative sum
  const n = f.length, df = f[1] - f[0], e0 = f[0] - df / 2, C = new Float64Array(n + 1); for (let i = 0; i < n; i++) C[i + 1] = C[i] + P[i] * df;
  const cum = x => { const t = (x - e0) / df; if (t <= 0) return 0; if (t >= n) return C[n]; const i = Math.floor(t); return C[i] + (C[i + 1] - C[i]) * (t - i); };
  return edges => { const out = new Float64Array(edges.length - 1); let prev = cum(edges[0]); for (let i = 1; i < edges.length; i++) { const c = cum(edges[i]); out[i - 1] = (c - prev) / (edges[i] - edges[i - 1]); prev = c; } return out; };
}
function features(f, P) {
  const rb = makeRebin(f, P), F1 = new Float32Array(NB), b = rb(EDGES); for (let i = 0; i < NB; i++) F1[i] = Math.log10(Math.max(b[i], 1e-6));
  const F2 = new Float32Array(NC * NL), fmax = f[f.length - 1], t = new Float64Array(NW); for (let i = 0; i < NW; i++) t[i] = -1 + 2 * i / (NW - 1);
  const e = new Float64Array(NW + 1), ec = new Float64Array(NW + 1), ok = new Uint8Array(NW), z = new Float64Array(NW);
  for (let j = 0; j < NC; j++) {
    const d = dnuS(CEN[j]); let nok = 0;
    for (let i = 0; i <= NW; i++) { e[i] = CEN[j] + (i - NW / 2) * d / STEP; ec[i] = Math.min(Math.max(e[i], f[0]), fmax); }
    for (let i = 0; i < NW; i++) { ok[i] = (e[i] > f[0] && e[i + 1] < fmax) ? 1 : 0; nok += ok[i]; }
    if (nok < NW / 2) continue;
    const y = rb(ec); const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], r = [0, 0, 0];
    for (let i = 0; i < NW; i++) { y[i] = Math.log(Math.max(y[i], 1e-12)); if (!ok[i]) continue; const v = [t[i] * t[i], t[i], 1]; for (let a = 0; a < 3; a++) { r[a] += v[a] * y[i]; for (let c = 0; c < 3; c++) A[a][c] += v[a] * v[c]; } }
    const co = solve3(A, r); let mz = 0;
    for (let i = 0; i < NW; i++) { z[i] = ok[i] ? Math.exp(y[i] - (co[0] * t[i] * t[i] + co[1] * t[i] + co[2])) - 1 : 0; if (ok[i]) mz += z[i]; }
    mz /= nok; for (let i = 0; i < NW; i++) if (ok[i]) z[i] -= mz;
    let r0 = 0; for (let i = 0; i < NW; i++) r0 += z[i] * z[i]; r0 = Math.max(r0, 1e-12);
    for (let l = 0; l < NL; l++) { const lag = L0 + l; let s = 0; for (let i = 0; i + lag < NW; i++) s += z[i] * z[i + lag]; F2[j * NL + l] = s / r0; }
  }
  return [F1, F2];
}
function solve3(A, b) {
  const M = A.map((r, i) => r.concat([b[i]]));
  for (let c = 0; c < 3; c++) { let p = c; for (let r = c + 1; r < 3; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r; [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < 3; r++) if (r !== c) { const k = M[r][c] / M[c][c]; for (let q = c; q < 4; q++) M[r][q] -= k * M[c][q]; } }
  return [M[0][3] / M[0][0], M[1][3] / M[1][1], M[2][3] / M[2][2]];
}
function scalars(y) {
  let first = -1, last = -1, n = 0; for (let i = 0; i < y.length; i++) if (isFinite(y[i])) { if (first < 0) first = i; last = i; n++; }
  const T = first < 0 ? 1 : (last - first + 1) * DT / 86400; return [new Float32Array([Math.log10(Math.max(T, 1)) - 2.5, n / y.length]), T];
}
function prepare(y) { const yh = highpass(fillShort(y)); const [f, P] = psd(yh); const [F1, F2] = features(f, P); const [s, T] = scalars(yh); return { F1, F2, s, f, P, T, yh }; }

// ---------- network (starquake.py) ----------
function erf(x) { const s = Math.sign(x); x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return s * y; }
const gelu = x => 0.5 * x * (1 + erf(x / Math.SQRT2));
function conv1d(x, C, L, w, b, O, K, stride, pad) {
  const Lo = Math.floor((L + 2 * pad - K) / stride) + 1, out = new Float64Array(O * Lo);
  for (let o = 0; o < O; o++) for (let l = 0; l < Lo; l++) { let s = b[o]; const s0 = l * stride - pad;
    for (let c = 0; c < C; c++) { const wo = (o * C + c) * K, xo = c * L; for (let k = 0; k < K; k++) { const p = s0 + k; if (p >= 0 && p < L) s += w[wo + k] * x[xo + p]; } }
    out[o * Lo + l] = s; }
  return [out, Lo];
}
function conv2d(x, C, H, W, w, b, O, K, sh, sw, pad) {
  const Ho = Math.floor((H + 2 * pad - K) / sh) + 1, Wo = Math.floor((W + 2 * pad - K) / sw) + 1, out = new Float64Array(O * Ho * Wo);
  for (let o = 0; o < O; o++) for (let i = 0; i < Ho; i++) for (let j = 0; j < Wo; j++) { let s = b[o];
    for (let c = 0; c < C; c++) for (let a = 0; a < K; a++) { const r = i * sh - pad + a; if (r < 0 || r >= H) continue; const xo = (c * H + r) * W, wo = ((o * C + c) * K + a) * K;
      for (let q = 0; q < K; q++) { const cc = j * sw - pad + q; if (cc >= 0 && cc < W) s += w[wo + q] * x[xo + cc]; } }
    out[(o * Ho + i) * Wo + j] = s; }
  return [out, Ho, Wo];
}
const mapG = a => { for (let i = 0; i < a.length; i++) a[i] = gelu(a[i]); return a; };
function forward(net, F1, F2, s) {
  const W = net.W, m = net.meta, sh = k => net.shape[k];
  let x = new Float64Array(NB); for (let i = 0; i < NB; i++) x[i] = (F1[i] - 2.5) / 2;
  let [h, L] = conv1d(x, 1, NB, W.s1_w, W.s1_b, sh('s1_w')[0], 7, 2, 3); mapG(h); let C = sh('s1_w')[0];
  for (let i = 0; i < m.n1; i++) { const O = sh(`r1_${i}_c1_w`)[0];
    let [a, La] = conv1d(h, C, L, W[`r1_${i}_c1_w`], W[`r1_${i}_c1_b`], O, 7, 2, 3); mapG(a); [a] = conv1d(a, O, La, W[`r1_${i}_c2_w`], W[`r1_${i}_c2_b`], O, 7, 1, 3);
    const [k] = conv1d(h, C, L, W[`r1_${i}_sk_w`], W[`r1_${i}_sk_b`], O, 1, 2, 0); for (let q = 0; q < a.length; q++) a[q] = gelu(a[q] + k[q]); h = a; L = La; C = O; }
  const h1 = h;
  x = new Float64Array(NC * NL); for (let i = 0; i < x.length; i++) x[i] = F2[i] * 4;
  let [g, H, Wd] = conv2d(x, 1, NC, NL, W.s2_w, W.s2_b, sh('s2_w')[0], 3, 1, 2, 1); mapG(g); C = sh('s2_w')[0];
  for (let i = 0; i < m.n2; i++) { const O = sh(`r2_${i}_c1_w`)[0];
    let [a, Ha, Wa] = conv2d(g, C, H, Wd, W[`r2_${i}_c1_w`], W[`r2_${i}_c1_b`], O, 3, 2, 2, 1); mapG(a); [a] = conv2d(a, O, Ha, Wa, W[`r2_${i}_c2_w`], W[`r2_${i}_c2_b`], O, 3, 1, 1, 1);
    const [k] = conv2d(g, C, H, Wd, W[`r2_${i}_sk_w`], W[`r2_${i}_sk_b`], O, 1, 2, 2, 0); for (let q = 0; q < a.length; q++) a[q] = gelu(a[q] + k[q]); g = a; H = Ha; Wd = Wa; C = O; }
  let v = new Float64Array(h1.length + g.length + 2); v.set(h1, 0); v.set(g, h1.length); v[v.length - 2] = s[0]; v[v.length - 1] = s[1];
  for (let j = 0; j < m.n_head; j++) { const w = W[`h${j}_w`], b = W[`h${j}_b`], O = b.length, I = v.length, o = new Float64Array(O);
    for (let a = 0; a < O; a++) { let t = b[a]; for (let q = 0; q < I; q++) t += w[a * I + q] * v[q]; o[a] = j < m.n_head - 1 ? gelu(t) : t; } v = o; }
  const mu = [v[0] * m.sd_y[0] + m.mu_y[0], v[1] * m.sd_y[1] + m.mu_y[1]], sg = [Math.exp(Math.min(Math.max(v[2], -7), 3)) * m.sd_y[0], Math.exp(Math.min(Math.max(v[3], -7), 3)) * m.sd_y[1]];
  const wd = m.widen || [1, 1], pdet = v.length > 4 ? 1 / (1 + Math.exp(-Math.min(Math.max(v[4], -30), 30))) : 1;
  return { p_det: pdet, ln_numax: mu[0], ln_dnu: mu[1], sig_numax: sg[0], sig_dnu: sg[1], numax: Math.exp(mu[0]), dnu: Math.exp(mu[1]), numax_err: Math.exp(mu[0]) * sg[0] * wd[0], dnu_err: Math.exp(mu[1]) * sg[1] * wd[1] };
}
function b64f32(s) { const bin = atob(s), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return new Float32Array(u.buffer); }
function loadNet(J) { const W = {}, shape = {}; for (const k in J.W) { W[k] = Float64Array.from(b64f32(J.W[k].d)); shape[k] = J.W[k].s; } return { W, shape, meta: J.meta }; }

// ---------- simulator for the lab (starsim.py, same physics; random numbers differ from Python) ----------
function rngMake(seed) { let s = seed >>> 0 || 1; const u = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return (s + 0.5) / 4294967296; };
  return { u, n: () => Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u()) }; }
const XI = 2 * Math.SQRT2 / Math.PI, VIS = [1, 1.54, 0.63, 0.071];
const harvey = (nu, a, b) => XI * a * a / b / (1 + Math.pow(nu / b, 4));
function yuAmp(nm, dn, teff, clump) { const [al, be, ga, de] = clump ? [0.36, -1.99, 1.36, -1.23] : [2.02, -2.85, 2.95, -1.82]; return 3.6 * al * Math.pow(nm / 3090, be) * Math.pow(dn / 135.1, ga) * Math.pow(teff / 5777, de); }
function starFromSliders(o) {   // o: numax, dnu, ampScale, teff, clump, mixed
  const nm = o.numax, teff = o.teff || 4800, M = o.M || 1.3, cl = !!o.clump;
  return { numax: nm, dnu: o.dnu, amp: yuAmp(nm, o.dnu, teff, cl) / Math.SQRT2 * (o.ampScale ?? 1), fwhm: 0.66 * Math.pow(nm, 0.88) * Math.exp(0.073 - 0.058 * Math.log(nm)),
    a_gran: 3710 * Math.pow(nm, -0.613) * Math.pow(M, -0.26), b1: 0.317 * Math.pow(nm, 0.970), b2: 0.948 * Math.pow(nm, 0.992), gamma0: 0.12 * Math.pow(teff / 4750, 3),
    eps_off: cl ? -0.04 : 0, d02_off: 0, mixed: o.mixed !== false, dpi: cl ? 290 : 75, q: cl ? 0.3 : 0.14, v1: 1.54 };
}
function mixedModes(nup, pw, g0, dnu, dpi, q) {
  const dP = dpi * 1e-6, N = 4000, out = []; let prev = null, prevNu = 0;
  const h = nu => { const tp = Math.PI * (nu - nup) / dnu, tg = Math.PI / (nu * dP); return Math.sin(tp) * Math.cos(tg) - q * Math.cos(tp) * Math.sin(tg); };
  const roots = []; for (let i = 0; i < N; i++) { const nu = nup - dnu / 2 + dnu * i / (N - 1), v = h(nu); if (prev !== null && Math.sign(prev) * Math.sign(v) < 0) roots.push(prevNu - prev * (nu - prevNu) / (v - prev)); prev = v; prevNu = nu; }
  let ws = 0; const rr = [];
  for (const r of roots) { const tp = Math.PI * (r - nup) / dnu, tg = Math.PI / (r * dP), Nn = dnu / (r * r * dP);
    if (Math.abs(Math.tan(tp) - q * Math.tan(tg)) >= 0.2 * (1 + Math.abs(Math.tan(tp)))) continue;
    const zeta = 1 / (1 + Math.cos(tg) ** 2 / Math.max(Math.cos(tp) ** 2, 1e-9) / (q * Nn)); rr.push([r, 1 - zeta]); ws += 1 - zeta; }
  if (!rr.length || ws <= 0) return [[nup, pw, g0]];
  return rr.map(([r, w]) => [r, pw * w / ws, Math.max(g0 * w, 1e-4)]);
}
function modeList(s) {
  const nm = s.numax, dn = s.dnu, lg = Math.log10(dn), eps = 0.634 + 0.546 * lg + s.eps_off, d0 = [0, -0.056 - 0.002 * lg, 0.131 - 0.033 * lg + s.d02_off, 0.280], alpha = 0.008;
  const sig = s.fwhm / (2 * Math.sqrt(2 * Math.log(2))), nmax = nm / dn - eps, out = [];
  for (let n = Math.max(1, Math.floor(nmax) - 8); n < Math.floor(nmax) + 10; n++) for (let l = 0; l < 4; l++) {
    const nu = (n + l / 2 + eps - d0[l] + alpha / 2 * (n - nmax) ** 2) * dn; if (nu <= 0.5 || nu > NYQ * 1.02 || Math.abs(nu - nm) > 4.5 * sig) continue;
    const pw = s.amp * s.amp * Math.exp(-((nu - nm) ** 2) / (2 * sig * sig)) * (l === 1 ? s.v1 : VIS[l]);
    if (l === 1 && s.mixed) out.push(...mixedModes(nu, pw, s.gamma0, dn, s.dpi, s.q)); else out.push([nu, pw, s.gamma0]); }
  return out;
}
function synth(s, n, seed) {   // signal on a power-of-two grid >= 2n, first n samples (ppm)
  let ng = 1; while (ng < 2 * n) ng <<= 1; const df = 1e6 / (ng * DT), nf = ng / 2 + 1, pw = new Float64Array(nf);
  for (let k = 1; k < nf; k++) { const f = k * df; pw[k] = (harvey(f, s.a_gran, s.b1) + harvey(f, s.a_gran, s.b2)) * df; }
  for (const [nu0, p, g] of modeList(s)) { const w = Math.max(40 * g, 30 * df), j0 = Math.max(Math.floor((nu0 - w) / df), 1), j1 = Math.min(Math.floor((nu0 + w) / df) + 2, nf);
    for (let k = j0; k < j1; k++) { const f = k * df; pw[k] += p / Math.PI * (Math.atan(2 * (f + df / 2 - nu0) / g) - Math.atan(2 * (f - df / 2 - nu0) / g)); } }
  const R = rngMake(seed), re = new Float64Array(ng), im = new Float64Array(ng);
  for (let k = 1; k < nf; k++) { const x = k * df / (2 * NYQ), eta = x === 0 ? 1 : Math.pow(Math.sin(Math.PI * x) / (Math.PI * x), 2), a = (ng / 2) * Math.sqrt(pw[k] * eta);
    const gr = a * R.n(), gi = -a * R.n(); re[k] = gr; im[k] = gi; if (k < ng - k) { re[ng - k] = gr; im[ng - k] = -gi; } }
  fft2(re, im, true); const out = new Float64Array(n); for (let i = 0; i < n; i++) out[i] = re[i] / ng; return out;
}
function limitSpectrum(s, f) { const out = new Float64Array(f.length), ml = modeList(s);
  for (let i = 0; i < f.length; i++) { let p = harvey(f[i], s.a_gran, s.b1) + harvey(f[i], s.a_gran, s.b2); for (const [nu0, pw, g] of ml) { const d = 2 * (f[i] - nu0) / g; if (Math.abs(d) < 400) p += pw * 2 / (Math.PI * g) / (1 + d * d); }
    const x = f[i] / (2 * NYQ); out[i] = p * Math.pow(Math.sin(Math.PI * x) / (Math.PI * x), 2); } return out; }
function scaling(nm, dn, teff, enm, edn, ete) { const a = nm / 3090, b = dn / 135.1, t = teff / 5777, M = a ** 3 * b ** -4 * t ** 1.5, R = a * b ** -2 * t ** 0.5, fn = enm / nm, fd = edn / dn, ft = ete / teff;
  return { M, eM: M * Math.hypot(3 * fn, 4 * fd, 1.5 * ft), R, eR: R * Math.hypot(fn, 2 * fd, 0.5 * ft) }; }

G.SQ = { DT, NYQ, NB, NC, NL, L0, STEP, FC, CEN, dnuS, fftAny, fft2, fillShort, highpass, psd, features, scalars, prepare, forward, loadNet, b64f32, synth, starFromSliders, modeList, limitSpectrum, scaling, yuAmp, rngMake };
})(typeof window !== 'undefined' ? window : globalThis);
