// BearingNet lab: a compact JavaScript version of the training simulator (src/bearsim.py).
// A localized defect makes an impact each time a ball rolls over it; each impact rings the machine's structure.
const BLab = (() => {
  const FS = 12000;
  function gauss(rng) { let u = 0, v = 0; while (u === 0) u = rng(); v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function mulberry(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function stribeck(th, eps) { const a = 1 - (1 - Math.cos(th)) / (2 * eps); return a > 0 ? Math.pow(a, 1.5) : 0; }
  // band power 500-5700 Hz via the FFT from BNet
  function bandPower(x) {
    const n = x.length, re = Float64Array.from(x), im = new Float64Array(n); BNet.fft(re, im, false); let s = 0;
    for (let i = 0; i <= n / 2; i++) { const f = i * FS / n; if (f >= 500 && f <= 5700) s += re[i] * re[i] + im[i] * im[i]; }
    return s / (n * n);
  }
  // background time-scaled by rho (speed change): y(t) = bg(rho t), linear interpolation
  function timeScale(bg, rho, n) { const y = new Float64Array(n); for (let i = 0; i < n; i++) { const u = i * rho, j = Math.floor(u), f = u - j; y[i] = bg[j] + (bg[j + 1] - bg[j]) * f; } let m = 0; for (let i = 0; i < n; i++) m += y[i]; m /= n; for (let i = 0; i < n; i++) y[i] -= m; return y; }
  /* opt: {cls: 'N'|'OR'|'IR'|'B', fr (Hz), geom {n,d,D}, sizeMm, snrDb, noiseDb, seed} ; bg: Float32Array at 12 kHz, fBg its shaft speed */
  function make(bg, fBg, opt) {
    const N = BNet.NWIN, rng = mulberry(opt.seed || 1), rho = opt.fr / fBg, y = timeScale(bg, rho, N), pb = bandPower(y), x = Float64Array.from(y);
    const out = { x, fDef: null, impacts: [] };
    if (opt.cls !== 'N') {
      const o = BNet.orders(opt.geom.n, opt.geom.d, opt.geom.D, 0), slip = 0.01, ftf = o.FTF * (1 - slip), fr = opt.fr, g = opt.geom;
      const fDef = { OR: g.n * ftf, IR: g.n * (1 - ftf), B: 2 * o.BSF * (1 - slip) }[opt.cls] * fr; out.fDef = fDef;
      const v = { OR: Math.PI * (g.D + g.d) * ftf * fr, IR: Math.PI * (g.D - g.d) * (1 - ftf) * fr, B: Math.PI * g.d * o.BSF * fr }[opt.cls];   // mm/s over the defect
      const delta = opt.sizeMm / v;                      // entry -> exit time, s
      const s = new Float64Array(N + 4000); let t = rng() / fDef, k = 0; const ph = rng() * 2 * Math.PI;
      while (t < N / FS) {
        let A = Math.exp(0.15 * gauss(rng));
        if (opt.cls === 'IR') A *= 0.1 + 0.9 * stribeck(2 * Math.PI * fr * t + ph, 0.6);
        if (opt.cls === 'B') { A *= 0.1 + 0.9 * stribeck(2 * Math.PI * ftf * fr * t + ph, 0.6); if (k % 2) A *= 0.5; }
        if (opt.cls === 'OR') A *= 1 + 0.3 * Math.cos(2 * Math.PI * fr * t + ph);
        const i0 = Math.round(t * FS), i1 = Math.round((t + delta) * FS);
        s[i0] -= 0.3 * A; s[i1] += A; out.impacts.push(t); k++;
        t += (1 + 0.01 * gauss(rng)) / fDef;
      }
      // structural resonances: two damped oscillators
      const res = [{ f: 2900, z: 0.012, a: 1 }, { f: 4300, z: 0.02, a: 0.5 }], L = 900, h = new Float64Array(L);
      for (const r of res) for (let i = 0; i < L; i++) { const tt = i / FS; h[i] += r.a * Math.exp(-2 * Math.PI * r.f * r.z * tt) * Math.sin(2 * Math.PI * r.f * tt); }
      const sig = new Float64Array(N);
      for (let i = 0; i < N; i++) { const a = s[i]; if (a === 0) continue; const m = Math.min(L, N - i); for (let j = 0; j < m; j++) sig[i + j] += a * h[j]; }
      const sc = Math.sqrt(pb * Math.pow(10, opt.snrDb / 10) / Math.max(bandPower(sig), 1e-30));
      for (let i = 0; i < N; i++) x[i] += sig[i] * sc;
    }
    if (opt.noiseDb > -60) {   // extra broadband sensor noise, relative to the background's 0.5-5.7 kHz power
      const sd = Math.sqrt(pb * FS / 5200) * Math.pow(10, opt.noiseDb / 20);   // white noise of variance sd^2 has band power sd^2 * 5200 / FS
      for (let i = 0; i < N; i++) x[i] += sd * gauss(rng);
    }
    return out;
  }
  return { make, bandPower };
})();
