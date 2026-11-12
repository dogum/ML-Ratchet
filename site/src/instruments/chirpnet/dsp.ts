// Display-only signal processing for the ChirpNet instrument (the network itself reads the raw whitened strain):
// a 30-350 Hz band-pass for the eye, like LIGO's own strain figures, and a constant-Q time-frequency map.

/** In-place radix-2 complex FFT (inverse scales by 1/n). */
export function fft(re: Float64Array, im: Float64Array, inv = false) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inv ? 2 : -2) * Math.PI) / len, wr = Math.cos(ang), wi = Math.sin(ang), half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k, b = a + half;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
  if (inv) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

export const TF_ROWS = 44, TF_FLO = 20, TF_FHI = 512;
export const tfFreq = (r: number) => TF_FLO * (TF_FHI / TF_FLO) ** (r / (TF_ROWS - 1));

/** Band-passed copies of two equal-length streams and their combined time-frequency energy map
 *  (rows: TF_ROWS log-spaced frequencies 20-512 Hz, Q ~ 6; cols: `cols` time bins; each row divided by its median). */
export function display(streams: Float32Array[], fs: number, cols: number, withTF = true) {
  const n = streams[0].length;
  let N = 1;
  while (N < n + fs) N <<= 1; // zero-pad by at least 1 s
  const df = fs / N, taper = Math.round(0.1 * fs);
  const M = N >> 2; // complex envelope sampled at fs/4: positive bins 0..M-1 cover 0-512 Hz
  const per = Math.floor((n / 4) / cols);
  const tf = new Float32Array(TF_ROWS * cols);
  const bands: Float32Array[] = [];
  for (const s of streams) {
    const re = new Float64Array(N), im = new Float64Array(N);
    for (let i = 0; i < n; i++) {
      const w = i < taper ? 0.5 - 0.5 * Math.cos((Math.PI * i) / taper) : i >= n - taper ? 0.5 - 0.5 * Math.cos((Math.PI * (n - 1 - i)) / taper) : 1;
      re[i] = s[i] * w;
    }
    fft(re, im);
    // 30-350 Hz band-pass, soft edges
    const br = new Float64Array(N), bi = new Float64Array(N);
    for (let k = 0; k < N; k++) {
      const f = Math.min(k, N - k) * df;
      const g = f < 25 || f > 400 ? 0 : f < 30 ? 0.5 - 0.5 * Math.cos((Math.PI * (f - 25)) / 5) : f > 350 ? 0.5 + 0.5 * Math.cos((Math.PI * (f - 350)) / 50) : 1;
      br[k] = re[k] * g; bi[k] = im[k] * g;
    }
    fft(br, bi, true);
    bands.push(Float32Array.from(br.subarray(0, n)));
    if (!withTF) continue;
    // constant-Q energy
    const yr = new Float64Array(M), yi = new Float64Array(M), row = new Float64Array(cols);
    for (let r = 0; r < TF_ROWS; r++) {
      const f0 = tfFreq(r), sg = f0 / 6;
      yr.fill(0); yi.fill(0);
      const k0 = Math.max(1, Math.floor((f0 - 4 * sg) / df)), k1 = Math.min(M - 1, Math.ceil((f0 + 4 * sg) / df));
      for (let k = k0; k <= k1; k++) { const g = Math.exp(-0.5 * ((k * df - f0) / sg) ** 2); yr[k] = re[k] * g; yi[k] = im[k] * g; }
      fft(yr, yi, true);
      for (let c = 0; c < cols; c++) {
        let e = 0;
        for (let j = c * per; j < (c + 1) * per; j++) e += yr[j] * yr[j] + yi[j] * yi[j];
        row[c] = e;
      }
      const med = Float64Array.from(row).sort()[cols >> 1] || 1;
      for (let c = 0; c < cols; c++) tf[r * cols + c] += row[c] / med / streams.length;
    }
  }
  return { bands, tf, cols };
}
