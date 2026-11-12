// BearingNet in the browser: the same front end as release/pipeline.py and the same network as release/bearingnet.py
const BNet = (() => {
  const FS = 12000, NWIN = 32768, DF = FS / NWIN, NV = 295, V = new Float64Array(NV);
  for (let i = 0; i < NV; i++) V[i] = -2 + i / 64;
  const FIXED = [[500, 5700], [500, 2000], [2000, 3800], [3800, 5700]];
  const HANN = new Float64Array(NWIN); for (let i = 0; i < NWIN; i++) HANN[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (NWIN - 1));

  // ---- radix-2 complex FFT, in place (re, im Float64Array), inverse if inv ----
  const twCache = {};
  function fft(re, im, inv) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) { let b = n >> 1; for (; j & b; b >>= 1) j ^= b; j ^= b; if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    let tw = twCache[n]; if (!tw) { tw = [new Float64Array(n / 2), new Float64Array(n / 2)]; for (let k = 0; k < n / 2; k++) { tw[0][k] = Math.cos(2 * Math.PI * k / n); tw[1][k] = -Math.sin(2 * Math.PI * k / n); } twCache[n] = tw; }
    const s = inv ? -1 : 1;
    for (let len = 2; len <= n; len <<= 1) {
      const h = len >> 1, st = n / len;
      for (let i = 0; i < n; i += len) for (let k = 0; k < h; k++) {
        const wr = tw[0][k * st], wi = s * tw[1][k * st], a = i + k, b = a + h;
        const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
      }
    }
    if (inv) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  }
  function rfftAbs(x) {   // |rfft(x)|, length n/2+1
    const n = x.length, re = Float64Array.from(x), im = new Float64Array(n); fft(re, im, false);
    const o = new Float64Array(n / 2 + 1); for (let i = 0; i <= n / 2; i++) o[i] = Math.hypot(re[i], im[i]); return o;
  }
  function orders(n, d, D, phi) {
    const r = d / D * Math.cos((phi || 0) * Math.PI / 180);
    return { FTF: 0.5 * (1 - r), BPFO: n / 2 * (1 - r), BPFI: n / 2 * (1 + r), BSF: D / (2 * d) * (1 - r * r) };
  }
  function skBand(x, lo = 500, hi = 5700) {
    const N = 256, hop = 64, nf = Math.floor((x.length - N) / hop) + 1, win = new Float64Array(N);
    for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N);   // scipy 'hann' (periodic)
    const m1 = new Float64Array(N / 2 + 1), m2 = new Float64Array(N / 2 + 1), re = new Float64Array(N), im = new Float64Array(N);
    for (let f = 0; f < nf; f++) {
      for (let i = 0; i < N; i++) { re[i] = x[f * hop + i] * win[i]; im[i] = 0; }
      fft(re, im, false);
      for (let k = 0; k <= N / 2; k++) { const p = re[k] * re[k] + im[k] * im[k]; m1[k] += p; m2[k] += p * p; }
    }
    const sk = new Float64Array(N / 2 + 1); for (let k = 0; k <= N / 2; k++) { const a = m1[k] / nf, b = m2[k] / nf; sk[k] = b / Math.max(a * a, 1e-30) - 2; }
    let best = -1e9, band = [lo, hi];
    for (const bw of [600, 1200, 2400]) {
      for (let c = lo + bw / 2; c < hi - bw / 2 + 1; c += 150) {
        let s = 0, cnt = 0;
        for (let k = 0; k <= N / 2; k++) { const f = k * FS / N; if (f >= c - bw / 2 && f <= c + bw / 2) { s += sk[k]; cnt++; } }
        s /= cnt; if (s > best) { best = s; band = [c - bw / 2, c + bw / 2]; }
      }
    }
    return band;
  }
  function envSpec(Xre, Xim, lo, hi) {
    const re = new Float64Array(NWIN), im = new Float64Array(NWIN), T = 60;
    for (let i = 0; i <= NWIN / 2; i++) {
      const f = i * DF, g = Math.min(1, Math.max(0, (f - lo + T) / T)) * Math.min(1, Math.max(0, (hi + T - f) / T));
      re[i] = 2 * Xre[i] * g; im[i] = 2 * Xim[i] * g;
    }
    fft(re, im, true); const e = new Float64Array(NWIN); let m = 0, m2 = 0;
    for (let i = 0; i < NWIN; i++) { e[i] = re[i] * re[i] + im[i] * im[i]; m += e[i]; m2 += e[i] * e[i]; }
    m /= NWIN; m2 /= NWIN; const k = m2 / Math.max(m * m, 1e-30);
    for (let i = 0; i < NWIN; i++) e[i] = (e[i] - m) * HANN[i];
    return { ses: rfftAbs(e), k };
  }
  function strip(spec, f0) {
    const out = new Float64Array(NV), L = spec.length;
    for (let i = 0; i < NV; i++) {
      const fc = f0 * Math.pow(2, V[i]), u = fc / DF, j = Math.floor(u);
      let v = j >= L - 1 ? spec[L - 1] : spec[j] + (spec[j + 1] - spec[j]) * (u - j);
      const lo = Math.min(Math.max(Math.ceil(fc * Math.pow(2, -1 / 128) / DF), 1), L - 1), hi = Math.min(Math.max(Math.floor(fc * Math.pow(2, 1 / 128) / DF), 1), L - 1);
      for (let q = lo; q <= hi; q++) if (spec[q] > v) v = spec[q];
      out[i] = v;
    }
    return out;
  }
  function median(a) { const s = Float64Array.from(a).sort(); const n = s.length; return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]); }
  // one window (NWIN samples at 12 kHz) -> {F: Float32Array(21*295), sc: [8], band, ses (wide band), sesSK, raw}
  function features(xin, geom, fNom) {
    let mu = 0; for (let i = 0; i < NWIN; i++) mu += xin[i]; mu /= NWIN;
    const x = new Float64Array(NWIN); for (let i = 0; i < NWIN; i++) x[i] = xin[i] - mu;
    const o = orders(geom.n, geom.d, geom.D, geom.phi), re = Float64Array.from(x), im = new Float64Array(NWIN); fft(re, im, false);
    const band = skBand(x), bands = FIXED.concat([band]), F = new Float32Array(21 * NV), ks = [], fh = [fNom, o.BPFO * fNom, o.BPFI * fNom, o.BSF * fNom], keep = {};
    bands.forEach(([lo, hi], b) => {
      const { ses, k } = envSpec(re, im, lo, hi); ks.push(k); if (b === 0) keep.ses = ses; if (b === 4) keep.sesSK = ses;
      for (let h = 0; h < 4; h++) {
        const s = strip(ses, fh[h]); for (let i = 0; i < NV; i++) s[i] = Math.log10(s[i] + 1e-12);
        const md = median(s); for (let i = 0; i < NV; i++) F[(b * 4 + h) * NV + i] = Math.min(5, Math.max(-2, s[i] - md));
      }
    });
    const xr = new Float64Array(NWIN); for (let i = 0; i < NWIN; i++) xr[i] = x[i] * HANN[i];
    const raw = rfftAbs(xr), s = strip(raw, fNom); for (let i = 0; i < NV; i++) s[i] = Math.log10(s[i] + 1e-12);
    const md = median(s); for (let i = 0; i < NV; i++) F[20 * NV + i] = Math.min(5, Math.max(-2, s[i] - md));
    const r = geom.d / geom.D * Math.cos((geom.phi || 0) * Math.PI / 180);
    const sc = ks.map((k) => Math.log10(Math.max(k, 1e-3) / 2)).concat([Math.log2(fNom / 25), geom.n / 10, 4 * r]);
    return { F, sc, band, ses: keep.ses, sesSK: keep.sesSK, raw, orders: o };
  }

  // ---- network ----
  function load(pack) {   // float32 little-endian, base64
    const bin = atob(pack.b64), buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    const all = new Float32Array(buf), T = {};
    for (const t of pack.man) { const sz = t.shape.reduce((a, b) => a * b, 1); T[t.k] = { d: all.subarray(t.off, t.off + sz), s: t.shape }; }
    return { T, meta: pack.meta };
  }
  const gelu = (x) => { const a = Math.abs(x) * 0.7071067811865476, t = 1 / (1 + 0.3275911 * a);
    const erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a); return 0.5 * x * (1 + (x >= 0 ? erf : -erf)); };
  function conv(x, Ci, L, w, b, stride) {
    const [O, , K] = w.s, p = K >> 1, Lo = Math.floor((L + 2 * p - K) / stride) + 1, out = new Float32Array(O * Lo), W = w.d;
    for (let o = 0; o < O; o++) {
      const ob = o * Lo; out.fill(b.d[o], ob, ob + Lo);
      for (let c = 0; c < Ci; c++) {
        const xb = c * L, wb = (o * Ci + c) * K;
        for (let k = 0; k < K; k++) {
          const wv = W[wb + k], off = k - p;
          for (let l = 0; l < Lo; l++) { const j = l * stride + off; if (j >= 0 && j < L) out[ob + l] += wv * x[xb + j]; }
        }
      }
    }
    return [out, O, Lo];
  }
  function trunk(T, p, x, Ci, nBlocks) {   // conv stem + residual blocks, then [mean, max] over length
    let [h, C, L] = conv(x, Ci, NV, T[p + 'stem_w'], T[p + 'stem_b'], 2); for (let j = 0; j < h.length; j++) h[j] = gelu(h[j]);
    for (let i = 0; i < nBlocks; i++) {
      let [y, O, Lo] = conv(h, C, L, T[`${p}b${i}_c1_w`], T[`${p}b${i}_c1_b`], 2); for (let j = 0; j < y.length; j++) y[j] = gelu(y[j]);
      [y] = conv(y, O, Lo, T[`${p}b${i}_c2_w`], T[`${p}b${i}_c2_b`], 1);
      const [sk] = conv(h, C, L, T[`${p}b${i}_sk_w`], T[`${p}b${i}_sk_b`], 2);
      for (let j = 0; j < y.length; j++) y[j] = gelu(y[j] + sk[j]);
      h = y; C = O; L = Lo;
    }
    const z = new Float32Array(2 * C);
    for (let c = 0; c < C; c++) { let m = 0, M = -Infinity; for (let l = 0; l < L; l++) { const v = h[c * L + l]; m += v; if (v > M) M = v; } z[c] = m / L; z[C + c] = M; }
    return z;
  }
  function mlp(T, p, z, n) {
    for (let j = 0; j < n; j++) {
      const w = T[`${p}${j}_w`], b = T[`${p}${j}_b`], [O, I] = w.s, o = new Float32Array(O);
      for (let r = 0; r < O; r++) { let a = b.d[r]; const wb = r * I; for (let c = 0; c < I; c++) a += w.d[wb + c] * z[c]; o[r] = j < n - 1 ? gelu(a) : a; }
      z = o;
    }
    return z;
  }
  const cat = (...a) => { const n = a.reduce((s, v) => s + v.length, 0), o = new Float32Array(n); let k = 0; for (const v of a) { o.set(v, k); k += v.length; } return o; };
  function forward(net, F, sc) {
    const T = net.T, m = net.meta;
    if ((m.arch || 'base') === 'base') return mlp(T, 'h', cat(trunk(T, '', F, 21, m.n_blocks), sc), m.n_head);
    const nb = m.nb, nh = m.nh, g = [];
    for (let k = 0; k < nh; k++) {   // the same trunk reads the 5 band strips of each hypothesis axis (shaft, BPFO, BPFI, BSF)
      const e = new Float32Array(nb * NV); for (let b = 0; b < nb; b++) e.set(F.subarray((b * nh + k) * NV, (b * nh + k + 1) * NV), b * NV);
      g.push(trunk(T, '', e, nb, m.n_blocks));
    }
    const r = trunk(T, 'r', F.subarray((nb * nh) * NV, (nb * nh + 1) * NV), 1, m.n_rblocks), E = T.emb, ne = E.s[1], lf = [];
    for (let k = 1; k <= 3; k++) lf.push(mlp(T, 'f', cat(g[k], g[0], E.d.subarray((k - 1) * ne, k * ne), sc), m.n_fault)[0]);
    const o = mlp(T, 'h', cat(...g, r, sc), m.n_rest);
    return Float32Array.from([o[0], ...lf, o[1], o[2], o[3], o[4]]);
  }
  // decode one window's raw outputs with the dev-fitted calibration
  function decode(net, o, fNom) {
    const m = net.meta, cal = m.calib || {}, T = cal.temp || 1, l = [0, 1, 2, 3].map((i) => o[i] / T), mx = Math.max(...l);
    const e = l.map((v) => Math.exp(v - mx)), s = e.reduce((a, b) => a + b, 0), p = e.map((v) => v / s);
    const sp = o[4] * m.sp_scale, sps = Math.exp(Math.min(3, Math.max(-5, o[5]))) * m.sp_scale * (cal.speed || 1);
    const lz = o[6] * m.sz_sd + m.sz_mu, lzs = Math.exp(Math.min(3, Math.max(-5, o[7]))) * m.sz_sd * (cal.size || 1);
    return { p, cls: m.classes[p.indexOf(Math.max(...p))], f: fNom * Math.pow(2, sp), fSigRel: sps * Math.LN2, sizeMm: Math.pow(10, lz), lsizeSig: lzs };
  }
  return { FS, NWIN, NV, V, fft, rfftAbs, orders, features, load, forward, decode, skBand };
})();
if (typeof module !== 'undefined') module.exports = BNet;
