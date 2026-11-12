(() => {
'use strict';
const $ = (id) => document.getElementById(id);
const FS = DATA.fs, WIN = 4096;
const BNS = DATA.bns || null;
const C = { h1: '#ffb35c', l1: '#58d6ea', net: '#b9a2ff', hit: '#ff6fa0', ink: '#e8eaf2', dim: '#9aa0b8', faint: '#646a82', grid: 'rgba(205,214,255,0.08)', ok: '#7be3a8' };

// ---------------------------------------------------------------- decoding
function u8(b64) { const s = atob(b64), a = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i); return a; }
function f16(b64) {
  const b = u8(b64), n = b.length >> 1, out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const h = b[2 * i] | (b[2 * i + 1] << 8), s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    out[i] = e === 0 ? s * m * 5.960464477539063e-8 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * Math.pow(2, e - 15);
  }
  return out;
}

// ---------------------------------------------------------------- FFT (radix 2, in place)
function fft(re, im, inv) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (inv ? 2 : -2) * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2, tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
  if (inv) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}
function bandpass(x, lo, hi) {
  const n = x.length, re = Float64Array.from(x), im = new Float64Array(n); fft(re, im, false);
  for (let k = 0; k < n; k++) { const f = Math.min(k, n - k) * FS / n; if (f < lo || f > hi) { re[k] = 0; im[k] = 0; } }
  fft(re, im, true); return Float32Array.from(re);
}
function bandMask(f) { const lo = 20, hi = 900, t = 8; if (f >= lo && f <= hi) return 1; if (f > lo - t && f < lo) return 0.5 - 0.5 * Math.cos(Math.PI * (f - (lo - t)) / t); if (f > hi && f < hi + t) return 0.5 + 0.5 * Math.cos(Math.PI * (f - hi) / t); return 0; }

// ---------------------------------------------------------------- canvas helpers
function setup(cv) {
  const r = cv.getBoundingClientRect(), d = Math.min(devicePixelRatio || 1, 2);
  const w = Math.max(10, Math.round(r.width)), h = Math.max(10, Math.round(r.height));
  if (cv.width !== w * d || cv.height !== h * d) { cv.width = w * d; cv.height = h * d; }
  const ctx = cv.getContext('2d'); ctx.setTransform(d, 0, 0, d, 0, 0); ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}
const LUT = (() => {
  const stops = [[0, [5, 6, 10]], [0.18, [30, 16, 66]], [0.42, [105, 36, 140]], [0.62, [226, 78, 130]], [0.8, [255, 170, 90]], [1, [255, 246, 214]]];
  const out = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255; let k = 0; while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
    const [t0, c0] = stops[k], [t1, c1] = stops[k + 1], u = (t - t0) / (t1 - t0);
    for (let j = 0; j < 3; j++) out[i * 3 + j] = c0[j] + (c1[j] - c0[j]) * u;
  }
  return out;
})();
function heat(ctx, w, h, grid, nf, nt, c0, c1, vmax) {
  const iw = c1 - c0, img = new ImageData(iw, nf);
  for (let fi = 0; fi < nf; fi++) for (let c = 0; c < iw; c++) {
    const v = grid[fi * nt + c0 + c], t = Math.max(0, Math.min(255, Math.round(255 * Math.min(1, v / vmax)))), p = ((nf - 1 - fi) * iw + c) * 4;
    img.data[p] = LUT[t * 3]; img.data[p + 1] = LUT[t * 3 + 1]; img.data[p + 2] = LUT[t * 3 + 2]; img.data[p + 3] = 255;
  }
  const off = document.createElement('canvas'); off.width = iw; off.height = nf; off.getContext('2d').putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true; ctx.drawImage(off, 0, 0, w, h);
}
function axisText(ctx, txt, x, y, align, color) { ctx.font = '500 10px "JetBrains Mono", ui-monospace, monospace'; ctx.fillStyle = color || 'rgba(232,234,242,0.72)'; ctx.textAlign = align || 'left'; ctx.fillText(txt, x, y); }

// ---------------------------------------------------------------- physics helpers
const GM = 4.925490947e-6;
const chirpF = (mc, tau) => (1 / Math.PI) * Math.pow(5 / (256 * tau), 0.375) * Math.pow(mc * GM, -0.625);
const fy = (f, h) => h - Math.log(f / 20) / Math.log(512 / 20) * h;
function masses(mc, q) { const eta = q / Math.pow(1 + q, 2), M = mc / Math.pow(eta, 0.6); return [M / (1 + q), M * q / (1 + q)]; }

// ---------------------------------------------------------------- inference (worker with main-thread fallback)
// cascade: the finder scores every window; where it scores above 0 the coincidence checker (two-stream) must agree, or the score drops
function cascade(CJ, net, chk, V, w) { const o = CJ.forward(net, w); if (chk && o[0] > 0) { const c = CJ.forward2s(chk, w)[0]; if (c < V.veto) o[0] -= V.drop; } return o; }
let worker = null, wid = 0; const pending = new Map();
try {
  const src = `const CJ=(${makeChirpJS.toString()})();let net=null,chk=null,V=null;const fwd=${cascade.toString()};onmessage=(e)=>{const d=e.data;if(d.type==='init'){net=CJ.load(d.pack);if(d.pack.checker){chk=CJ.load(d.pack.checker);V=d.pack.checker.cfg;}return;}const out=[];for(let i=0;i<d.wins.length;i++){out.push(fwd(CJ,net,chk,V,d.wins[i]));if(d.progress)postMessage({id:d.id,i,o:out[i]});}postMessage({id:d.id,done:true,out});};`;
  worker = new Worker(URL.createObjectURL(new Blob([src], { type: 'text/javascript' })));
  worker.postMessage({ type: 'init', pack: WEIGHTS });
  worker.onmessage = (e) => { const p = pending.get(e.data.id); if (!p) return; if (e.data.done) { pending.delete(e.data.id); p.resolve(e.data.out); } else if (p.onp) p.onp(e.data.i, e.data.o); };
  worker.onerror = () => { worker = null; };
} catch (_) { worker = null; }
let mainNet = null, mainChk = null;
function infer(wins, onp) {
  if (worker) return new Promise((resolve) => { const id = ++wid; pending.set(id, { resolve, onp }); worker.postMessage({ type: 'run', id, wins, progress: !!onp }); });
  if (!mainNet) { mainNet = ChirpJS.load(WEIGHTS); mainChk = WEIGHTS.checker ? ChirpJS.load(WEIGHTS.checker) : null; }
  return new Promise((resolve) => {
    const out = []; let i = 0;
    const step = () => { const t = performance.now(); while (i < wins.length && performance.now() - t < 30) { out.push(cascade(ChirpJS, mainNet, mainChk, WEIGHTS.checker && WEIGHTS.checker.cfg, wins[i])); if (onp) onp(i, out[i]); i++; } if (i < wins.length) setTimeout(step, 0); else resolve(out); };
    step();
  });
}

// ---------------------------------------------------------------- events
const EV = DATA.events.map((e) => ({ ...e, _H: null }));
const THR = DATA.threshold, BGMAX = DATA.bg_max;
function evData(e) {
  if (!e._H) {
    e._H = f16(e.H); e._L = f16(e.L); e._sH = u8(e.specH); e._sL = u8(e.specL); e._aud = f16(e.audio);
    e._bH = bandpass(e._H, 30, 350); e._bL = bandpass(e._L, 30, 350);
    // align L1 to H1 (light-travel delay + antenna sign), as in LIGO's own figures
    const i0 = Math.round((6 - 0.25) * FS), i1 = Math.round((6 + 0.03) * FS); let best = { c: 0, lag: 0, s: 1 };
    for (let lag = -22; lag <= 22; lag++) { let c = 0; for (let i = i0; i < i1; i++) c += e._bH[i] * e._bL[i - lag]; if (Math.abs(c) > Math.abs(best.c)) best = { c, lag, s: Math.sign(c) || 1 }; }
    e._lag = best.lag; e._sign = best.s;
  }
  return e;
}
let cur = 0, liveTrace = null;
const T0 = -2.0, T1 = 0.5;      // spectrogram window around the merger
function colOf(t) { return Math.round((t - DATA.seg_t0) / 8 * 512); }
function drawSpec(cv, grid, e, withChirp) {
  const { ctx, w, h } = setup(cv);
  heat(ctx, w, h, grid, 96, 512, colOf(T0), colOf(T1), 190);
  const x = (t) => (t - T0) / (T1 - T0) * w;
  ctx.strokeStyle = 'rgba(255,255,255,0.10)'; ctx.lineWidth = 1;
  for (const f of [32, 64, 128, 256]) { const y = fy(f, h); ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); axisText(ctx, f + ' Hz', 6, y - 3); }
  for (const t of [-1.5, -1, -0.5, 0]) { axisText(ctx, (t > 0 ? '+' : '') + t.toFixed(1) + ' s', x(t), h - 5, 'center', 'rgba(232,234,242,0.55)'); }
  if (withChirp && e.hit) {
    const tc = e.hit.gps - e.gps, mc = e.hit.chirp_mass, [m1, m2] = [e.hit.m1, e.hit.m2], fmax = 4400 / (m1 + m2) * 1.1;
    ctx.save(); ctx.strokeStyle = C.net; ctx.lineWidth = 2; ctx.setLineDash([6, 4]); ctx.shadowColor = 'rgba(185,162,255,0.9)'; ctx.shadowBlur = 6;
    ctx.beginPath(); let started = false;
    for (let t = T0; t < tc - 0.002; t += 0.002) {
      const f = chirpF(mc, tc - t); if (f < 20) continue; if (f > Math.min(fmax, 512)) break;
      const X = x(t), Y = fy(f, h); if (!started) { ctx.moveTo(X, Y); started = true; } else ctx.lineTo(X, Y);
    }
    ctx.stroke(); ctx.restore();
  }
}
function drawStrain(e) {
  const { ctx, w, h } = setup($('strain'));
  const a = -0.35, b = 0.05, i0 = Math.round((a + 6) * FS), i1 = Math.round((b + 6) * FS);
  let m = 0; for (let i = i0; i < i1; i++) m = Math.max(m, Math.abs(e._bH[i]), Math.abs(e._bL[i - e._lag]));
  const sy = (h / 2 - 8) / (m || 1), X = (i) => (i - i0) / (i1 - i0) * w;
  ctx.strokeStyle = C.grid; ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
  const tc = e.hit ? e.hit.gps - e.gps : 0;
  ctx.strokeStyle = 'rgba(185,162,255,0.5)'; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(((tc - a) / (b - a)) * w, 0); ctx.lineTo(((tc - a) / (b - a)) * w, h); ctx.stroke(); ctx.setLineDash([]);
  for (const [arr, col, lag, sg] of [[e._bL, C.l1, e._lag, e._sign], [e._bH, C.h1, 0, 1]]) {
    ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.95; ctx.beginPath();
    for (let i = i0; i < i1; i++) { const v = sg * arr[i - lag]; const xx = X(i), yy = h / 2 - v * sy; i === i0 ? ctx.moveTo(xx, yy) : ctx.lineTo(xx, yy); }
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  axisText(ctx, `L1 shifted ${(e._lag / FS * 1000).toFixed(1)} ms${e._sign < 0 ? ', inverted' : ''}`, w - 8, 14, 'right', 'rgba(88,214,234,0.8)');
}
function drawScore(e) {
  const { ctx, w, h } = setup($('scoreCv'));
  const tr = liveTrace || e.trace, tt = e.trace_t, ymin = -12, ymax = 16;
  const X = (t) => (t - tt[0]) / (tt[tt.length - 1] - tt[0]) * (w - 40) + 34, Y = (v) => h - 12 - (Math.max(ymin, Math.min(ymax, v)) - ymin) / (ymax - ymin) * (h - 22);
  ctx.strokeStyle = C.grid; for (const v of [-8, 0, 8]) { ctx.beginPath(); ctx.moveTo(34, Y(v)); ctx.lineTo(w, Y(v)); ctx.stroke(); axisText(ctx, String(v), 28, Y(v) + 3, 'right', 'rgba(232,234,242,0.45)'); }
  ctx.fillStyle = 'rgba(100,106,130,0.18)'; ctx.fillRect(34, Y(BGMAX), w - 34, Y(ymin) - Y(BGMAX));
  ctx.strokeStyle = C.faint; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(34, Y(BGMAX)); ctx.lineTo(w, Y(BGMAX)); ctx.stroke(); ctx.setLineDash([]);
  axisText(ctx, 'loudest real noise', 40, Y(BGMAX) - 4, 'left', 'rgba(154,160,184,0.85)');
  ctx.strokeStyle = C.net; ctx.lineWidth = 2; ctx.beginPath(); let st = false;
  for (let i = 0; i < tr.length; i++) { if (tr[i] === null || tr[i] === undefined) continue; const xx = X(tt[i]), yy = Y(tr[i]); if (!st) { ctx.moveTo(xx, yy); st = true; } else ctx.lineTo(xx, yy); }
  ctx.stroke();
  let k = 0; for (let i = 1; i < tr.length; i++) if (tr[i] != null && tr[i] > (tr[k] ?? -1e9)) k = i;
  if (tr[k] != null) { ctx.fillStyle = C.hit; ctx.beginPath(); ctx.arc(X(tt[k]), Y(tr[k]), 4, 0, 6.283); ctx.fill(); axisText(ctx, tr[k].toFixed(1), X(tt[k]) + 8, Y(tr[k]) + 4, 'left', C.hit); }
  for (const t of [-5, -4, -3, -2, -1, 0, 1]) if (t >= tt[0] && t <= tt[tt.length - 1]) axisText(ctx, (t > 0 ? '+' : '') + t + ' s', X(t), h - 1, 'center', 'rgba(232,234,242,0.45)');
}
function fmt(x, d = 1) { return Number(x).toFixed(d); }
function drawCmp(e) {
  const svg = $('cmpSvg'), hit = e.hit; if (!hit) { svg.innerHTML = ''; return; }
  const lo = Math.min(e.mc_det_lo, hit.chirp_mass_68[0]) * 0.85, hi = Math.max(e.mc_det_hi, hit.chirp_mass_68[1]) * 1.12;
  const X = (v) => 8 + (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) * 214;
  svg.innerHTML = `
    <rect x="${X(e.mc_det_lo)}" y="10" width="${X(e.mc_det_hi) - X(e.mc_det_lo)}" height="10" rx="5" fill="url(#lg)" opacity=".85"/>
    <defs><linearGradient id="lg" x1="0" x2="1"><stop offset="0" stop-color="${C.h1}"/><stop offset="1" stop-color="${C.l1}"/></linearGradient></defs>
    <text x="${X(e.mc_det_hi) + 6}" y="19" fill="${C.dim}" font-family="JetBrains Mono" font-size="11">LIGO</text>
    <line x1="${X(hit.chirp_mass_68[0])}" x2="${X(hit.chirp_mass_68[1])}" y1="38" y2="38" stroke="${C.net}" stroke-width="3" stroke-linecap="round"/>
    <circle cx="${X(hit.chirp_mass)}" cy="38" r="5" fill="${C.net}"/>
    <text x="${X(hit.chirp_mass_68[1]) + 8}" y="42" fill="${C.net}" font-family="JetBrains Mono" font-size="11">ChirpNet</text>
    <text x="8" y="62" fill="${C.faint}" font-family="JetBrains Mono" font-size="10">LIGO 90% ${fmt(e.mc_det_lo)}–${fmt(e.mc_det_hi)} · ChirpNet 68% ${fmt(hit.chirp_mass_68[0])}–${fmt(hit.chirp_mass_68[1])}</text>`;
}
function renderEvent() {
  const e = evData(EV[cur]);
  document.querySelectorAll('.ev').forEach((b, i) => b.setAttribute('aria-pressed', String(i === cur)));
  drawSpec($('specH'), e._sH, e, true); drawSpec($('specL'), e._sL, e, true);
  drawStrain(e); drawScore(e); drawCmp(e);
  const hit = e.hit, det = hit && hit.score > THR;
  $('verdict').classList.toggle('no', !det); $('verdictTxt').textContent = det ? 'Detected' : 'Not detected';
  const bgN = DATA.bg_n.toLocaleString();
  $('verdictSub').innerHTML = hit ? `Detection statistic <b>${fmt(hit.score)}</b>. Louder than all <b>${bgN}</b> windows of time-slid real LIGO noise, whose loudest reached ${fmt(BGMAX)}. False-alarm rate below <b>${fmt(hit.far_per_hour, 2)}</b> per hour.` : '';
  $('mcBig').innerHTML = hit ? `${fmt(hit.chirp_mass)}<small>M☉</small>` : '—';
  const z = e.z, md1 = e.m1 * (1 + z), md2 = e.m2 * (1 + z);
  $('kv').innerHTML = hit ? `
    <dt>Component masses (rough)</dt><dd>${fmt(hit.m1, 0)} + ${fmt(hit.m2, 0)} M☉</dd>
    <dt>LIGO (detector frame)</dt><dd>${fmt(md1, 0)} + ${fmt(md2, 0)} M☉</dd>
    <dt>Source-frame chirp mass</dt><dd>${fmt(hit.chirp_mass / (1 + z))} M☉</dd>
    <dt>LIGO source frame</dt><dd>${fmt(e.mc)} <span style="color:var(--faint)">+${fmt(e.mc_p)}/−${fmt(e.mc_m)}</span></dd>
    <dt>Merger time vs LIGO</dt><dd>${((hit.gps - e.gps) * 1000 >= 0 ? '+' : '')}${fmt((hit.gps - e.gps) * 1000, 0)} ms</dd>
    ${hit.coincidence != null ? `<dt>Both detectors agree?</dt><dd style="color:var(--ok)">${hit.coincidence >= (WEIGHTS.checker ? WEIGHTS.checker.cfg.veto : -1) ? 'yes' : 'no'} · ${fmt(hit.coincidence, 1)}</dd>` : ''}
    <dt>LIGO network SNR</dt><dd>${fmt(e.snr)}</dd>` : '';
  $('chirpLbl').innerHTML = hit ? `<span style="color:var(--net)">- -</span> chirp for ℳ = ${fmt(hit.chirp_mass)} M☉` : '';
}
const evlist = $('evlist');
EV.forEach((e, i) => {
  const b = document.createElement('button'); b.className = 'ev'; b.type = 'button';
  b.innerHTML = `<span class="n">${e.label}<i>SNR ${fmt(e.snr)}</i></span><span class="d">${e.date} · ${e.note}</span>`;
  b.addEventListener('click', () => { cur = i; liveTrace = null; $('liveNote').textContent = 'The traces above were computed with the released weights. “Run it” repeats the whole scan here, on your device.'; renderEvent(); });
  evlist.appendChild(b);
});

// live scan
$('liveBtn').addEventListener('click', async () => {
  const e = evData(EV[cur]), btn = $('liveBtn'); btn.disabled = true;
  const wins = []; for (let s = 0; s + WIN <= e._H.length; s += 128) { const w = new Float32Array(2 * WIN); w.set(e._H.subarray(s, s + WIN), 0); w.set(e._L.subarray(s, s + WIN), WIN); wins.push(w); }
  liveTrace = new Array(wins.length).fill(null);
  const t0 = performance.now();
  const out = await infer(wins, (i, o) => { liveTrace[i] = o[0]; if (i % 4 === 0) drawScore(e); $('liveNote').textContent = `Running on your device… window ${i + 1} of ${wins.length}`; });
  out.forEach((o, i) => { liveTrace[i] = o[0]; });
  drawScore(e);
  let d = 0; out.forEach((o, i) => { d = Math.max(d, Math.abs(o[0] - e.trace[i])); });
  $('liveNote').textContent = `Scanned ${wins.length} windows in ${((performance.now() - t0) / 1000).toFixed(1)} s on your device. Largest difference from the released Python run: ${d.toExponential(1)}.`;
  btn.disabled = false;
});

// audio
let actx = null, playing = null;
$('speedBtn').addEventListener('click', (ev) => { const b = ev.currentTarget; b.setAttribute('aria-pressed', String(b.getAttribute('aria-pressed') !== 'true')); });
$('listenBtn').addEventListener('click', () => {
  try {
    actx = actx || new (window.AudioContext || window.webkitAudioContext)();
    if (playing) { try { playing.stop(); } catch (_) { } playing = null; }
    const e = evData(EV[cur]), a = e._aud, fast = $('speedBtn').getAttribute('aria-pressed') === 'true', sr = 8192;
    const n = fast ? a.length : a.length * 4, buf = actx.createBuffer(1, n, sr), ch = buf.getChannelData(0);
    for (let i = 0; i < n; i++) { const p = fast ? i : i / 4, j = Math.floor(p), u = p - j; ch[i] = 0.8 * ((a[j] || 0) * (1 - u) + (a[j + 1] || 0) * u); }
    const fade = Math.min(800, n / 10); for (let i = 0; i < fade; i++) { ch[i] *= i / fade; ch[n - 1 - i] *= i / fade; }
    const src = actx.createBufferSource(); src.buffer = buf; src.connect(actx.destination); src.start(); playing = src;
  } catch (_) { $('liveNote').textContent = 'Audio is not available in this browser.'; }
});

// ---------------------------------------------------------------- the lab
const TPL = f16(DATA.tpl), TI = DATA.tpl_info, MCS = [...new Set(TI.map((t) => t.mc))], QS = [...new Set(TI.map((t) => t.q))];
const lab = { mc: 10, q: 0, snr: 10, gl: false, seed: 1 };
function gaussRng(seed) { let s = seed >>> 0 || 1; const r = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; return () => { let u = r() || 1e-9, v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.283185307 * v); }; }
const MASK = (() => { const m = new Float64Array(WIN); let acc = 0; for (let k = 0; k < WIN; k++) { const kk = Math.min(k, WIN - k); m[k] = bandMask(kk * FS / WIN); } for (let k = 1; k <= WIN / 2; k++) acc += m[k] * m[k]; return { m, norm: Math.sqrt(acc / (WIN / 2)) }; })();
function bandNoise(g) { const re = new Float64Array(WIN), im = new Float64Array(WIN); for (let i = 0; i < WIN; i++) re[i] = g(); fft(re, im, false); for (let k = 0; k < WIN; k++) { re[k] *= MASK.m[k]; im[k] *= MASK.m[k]; } fft(re, im, true); const o = new Float32Array(WIN); for (let i = 0; i < WIN; i++) o[i] = re[i] / MASK.norm; return o; }
function glitch() { const o = new Float32Array(WIN), f0 = 140, Q = 9, t0 = 1.05, tau = Q / (2 * Math.PI * f0); let s2 = 0; for (let i = 0; i < WIN; i++) { const t = i / FS - t0; o[i] = Math.exp(-((t / tau) ** 2)) * Math.cos(2 * Math.PI * f0 * t); s2 += o[i] * o[i]; } const k = 28 / Math.sqrt(DATA.phi * s2); for (let i = 0; i < WIN; i++) o[i] *= k; return o; }
function cwtCombined(chs) {
  const nf = 64, nt = 256, freqs = Array.from({ length: nf }, (_, i) => 20 * Math.pow(512 / 20, i / (nf - 1))), grid = new Float32Array(nf * nt);
  for (const x of chs) {
    const re = Float64Array.from(x), im = new Float64Array(WIN); fft(re, im, false);
    for (let fi = 0; fi < nf; fi++) {
      const f0 = freqs[fi], sg = f0 / 6, wr = new Float64Array(WIN), wi = new Float64Array(WIN);
      for (let k = 1; k < WIN / 2; k++) { const f = k * FS / WIN, g = Math.exp(-0.5 * ((f - f0) / sg) ** 2); if (g < 1e-4) continue; wr[k] = re[k] * g; wi[k] = im[k] * g; }
      fft(wr, wi, true); let m = 0; const e = new Float64Array(WIN); for (let i = 0; i < WIN; i++) { e[i] = wr[i] * wr[i] + wi[i] * wi[i]; m += e[i]; } m /= WIN;
      for (let c = 0; c < nt; c++) { let s = 0; for (let j = 0; j < 16; j++) s += e[c * 16 + j]; grid[fi * nt + c] += s / 16 / (m || 1) / 2; }
    }
  }
  return { grid, nf, nt };
}
let labBusy = false, labAgain = false;
async function runLab() {
  if (labBusy) { labAgain = true; return; }
  labBusy = true;
  const ti = lab.q * MCS.length + lab.mc, info = TI[ti], g = gaussRng(lab.seed * 7919 + 17);
  const H = bandNoise(g), L = bandNoise(g), s = TPL.subarray(ti * 2 * WIN, (ti + 1) * 2 * WIN);
  for (let i = 0; i < WIN; i++) { H[i] += lab.snr * s[i]; L[i] += lab.snr * s[WIN + i]; }
  if (lab.gl) { const gl = glitch(); for (let i = 0; i < WIN; i++) H[i] += gl[i]; }
  const x = new Float32Array(2 * WIN); x.set(H, 0); x.set(L, WIN);
  const [o] = await infer([x]);
  // draw strain
  { const { ctx, w, h } = setup($('labStrain')); const sy = (h / 4 - 4) / 4.5;
    for (const [arr, col, y0] of [[H, C.h1, h * 0.27], [L, C.l1, h * 0.73]]) { ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.globalAlpha = 0.85; ctx.beginPath(); for (let i = 0; i < WIN; i += 2) { const xx = i / WIN * w, yy = y0 - arr[i] * sy; i ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy); } ctx.stroke(); }
    ctx.globalAlpha = 1; const xm = DATA.tpl_t_merge / 2 * w; ctx.strokeStyle = 'rgba(232,234,242,0.5)'; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(xm, 0); ctx.lineTo(xm, h); ctx.stroke(); ctx.setLineDash([]);
    for (const t of [0.5, 1, 1.5]) axisText(ctx, t + ' s', t / 2 * w, h - 3, 'center', 'rgba(232,234,242,0.45)'); }
  { const cw = cwtCombined([H, L]), { ctx, w, h } = setup($('labSpec')); heat(ctx, w, h, cw.grid, cw.nf, cw.nt, 0, cw.nt, 16);
    ctx.strokeStyle = 'rgba(255,255,255,0.10)'; for (const f of [32, 64, 128, 256]) { const y = fy(f, h); ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); axisText(ctx, f + ' Hz', 6, y - 3); }
    const det = o[0] > THR;
    if (det) { const mc = Math.exp(o[1]), tc = o[5]; ctx.save(); ctx.strokeStyle = C.net; ctx.lineWidth = 2; ctx.setLineDash([6, 4]); ctx.beginPath(); let st = false; const [m1, m2] = masses(mc, Math.min(1, Math.max(0.1, o[3]))); const fmax = 4400 / (m1 + m2) * 1.1;
      for (let t = 0; t < tc - 0.002; t += 0.002) { const f = chirpF(mc, tc - t); if (f < 20) continue; if (f > Math.min(512, fmax)) break; const X = t / 2 * w, Y = fy(f, h); st ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); st = true; } ctx.stroke(); ctx.restore(); } }
  const det = o[0] > THR, mc = Math.exp(o[1]), sl = Math.exp(Math.max(-7, Math.min(3, o[2])));
  $('labRes').innerHTML = `
    <div><span>Network says</span><b class="${det ? 'yes' : 'nope'}">${det ? 'Merger' : 'Just noise'}</b></div>
    <div><span>Detection statistic</span><b>${fmt(o[0])} <small style="color:var(--faint);font-size:12px">/ ${fmt(THR)}</small></b></div>
    <div><span>Chirp mass (true ${info.mc})</span><b>${det ? fmt(mc) + ' <small style="color:var(--faint);font-size:12px">±' + fmt(mc * sl) + '</small>' : '—'}</b></div>
    <div><span>Merger at (true 1.60 s)</span><b>${det ? fmt(o[5], 2) + ' s' : '—'}</b></div>`;
  $('labTruth').textContent = `Injected: ${info.m1} + ${info.m2} M☉ (detector frame), IMRPhenomD, at network SNR ${fmt(lab.snr)}${lab.gl ? ', plus a glitch in Hanford with SNR 28' : ''}.`;
  labBusy = false;
  if (labAgain) { labAgain = false; runLab(); }
}
$('mcR').addEventListener('input', (e) => { lab.mc = +e.target.value; $('mcV').textContent = MCS[lab.mc] + ' M☉'; runLab(); });
$('snrR').addEventListener('input', (e) => { lab.snr = +e.target.value; $('snrV').textContent = fmt(lab.snr); runLab(); });
document.querySelectorAll('#qSeg button').forEach((b) => b.addEventListener('click', () => { lab.q = +b.dataset.q; document.querySelectorAll('#qSeg button').forEach((x) => x.setAttribute('aria-pressed', String(x === b))); $('qV').textContent = QS[lab.q].toFixed(1); runLab(); }));
$('glBtn').addEventListener('click', (e) => { lab.gl = !lab.gl; e.currentTarget.setAttribute('aria-pressed', String(lab.gl)); runLab(); });
$('noiseBtn').addEventListener('click', () => { lab.seed++; runLab(); });
$('mcV').textContent = MCS[lab.mc] + ' M☉';

// ---------------------------------------------------------------- ratchet
const R = DATA.rungs;
const titleOf = (n) => n.split(':')[0];
{ let k = 0; R.forEach((r) => { r._num = r.rejudged ? k : ++k; }); }
const sfLoud = (r) => r.single_fa ? (Object.entries(r.single_fa).filter(([k]) => +k >= 12).reduce((a, [, v]) => a + v, 0) / Object.keys(r.single_fa).filter((k) => +k >= 12).length) : null;
(() => {
  const el = $('rungs');
  R.forEach((r0, i) => {
    let r = r0; const re = r.rejudged, num = `Rung ${r._num}`;
    if (re) r = Object.assign({}, r, { note: 'Glitch training, re-judged: same candidate, scored by the new single objective (Gaussian sensitivity + glitchy sensitivity − 2 × chirp-mass error)' });
    const c = document.createElement('div'); c.className = 'rung ' + (r.accepted ? 'acc' : 'rej');
    const desc = r.note.includes(':') ? r.note.split(':').slice(1).join(':').trim() : r.note;
    const rel = r.accepted && r.ckpt === DATA.release.checkpoint;
    if (rel) c.style.borderColor = 'rgba(185,162,255,0.6)';
    c.innerHTML = `<div class="tag"><span>${num}${re ? ' · re-judged' : ''}</span><b style="${rel ? 'color:var(--net)' : ''}">${rel ? 'Kept · released' : r.accepted ? 'Kept' : 'Rejected'}</b></div>
      <h3>${titleOf(r.note)}</h3><p>${desc}</p>${r.accepted ? '' : `<p style="color:var(--faint)">${r.why}</p>`}
      <div class="m"><div><span>Found at SNR 8</span><b>${Math.round(100 * r.eff['8'])}%</b></div><div><span>Chirp-mass error</span><b>${fmt(100 * r.mc_err)}%</b></div>
      <div><span>Found with glitches</span><b>${Math.round(100 * r.sens_glitch)}%</b></div><div><span>Merger timing</span><b>${Math.round(r.t_err_ms)} ms</b></div>
      ${sfLoud(r) != null ? `<div style="grid-column:1/-1"><span>Fooled by a loud chirp in one detector only</span><b style="color:${sfLoud(r) < 0.3 ? 'var(--ok)' : sfLoud(r) > 0.8 ? 'var(--hit)' : 'var(--ink)'}">${Math.round(100 * sfLoud(r))}%</b></div>` : ''}</div>`;
    el.appendChild(c);
  });
})();
function svgEff() {
  const svg = $('effSvg'), W = 520, H = 300, L = 46, B = 34, T = 14, Rr = 12;
  const X = (s) => L + (s - 5) / 15 * (W - L - Rr), Y = (v) => H - B - v * (H - B - T);
  let g = '';
  for (const v of [0, 0.25, 0.5, 0.75, 1]) g += `<line x1="${L}" x2="${W - Rr}" y1="${Y(v)}" y2="${Y(v)}" stroke="${C.grid}"/><text x="${L - 8}" y="${Y(v) + 4}" fill="${C.dim}" font-size="11" font-family="JetBrains Mono" text-anchor="end">${Math.round(v * 100)}%</text>`;
  for (const s of [5, 8, 10, 12, 15, 20]) g += `<text x="${X(s)}" y="${H - 12}" fill="${C.dim}" font-size="11" font-family="JetBrains Mono" text-anchor="middle">${s}</text>`;
  g += `<text x="${(W + L) / 2}" y="${H}" fill="${C.faint}" font-size="11" font-family="Figtree" text-anchor="middle">network signal-to-noise ratio</text>`;
  const acc = R.filter((r) => r.accepted), n = acc.length;
  acc.forEach((r, i) => {
    const pts = Object.entries(r.eff).map(([s, v]) => [+s, v]).sort((a, b) => a[0] - b[0]);
    const col = i === n - 1 ? C.net : `rgba(185,162,255,${0.18 + 0.5 * i / Math.max(1, n - 1)})`;
    g += `<polyline fill="none" stroke="${col}" stroke-width="${i === n - 1 ? 2.6 : 1.6}" ${r.accepted ? '' : 'stroke-dasharray="4 4"'} points="${pts.map(([s, v]) => `${X(s)},${Y(v)}`).join(' ')}"/>`;
    const last = pts.find((p) => p[0] === 8); g += `<circle cx="${X(8)}" cy="${Y(last[1])}" r="3" fill="${col}"/>`;
  });
  const lg = acc.map((r, i) => `<tspan x="${X(12.6)}" dy="${i ? 16 : 0}" fill="${i === n - 1 ? C.net : C.dim}">Rung ${r._num}: ${r.rejudged ? 'Glitch training' : titleOf(r.note)} · ${Math.round(100 * r.eff['8'])}% at SNR 8</tspan>`).join('');
  g += `<text x="${X(12.6)}" y="${Y(0.42)}" font-size="11.5" font-family="Figtree">${lg}</text>`;
  svg.innerHTML = g;
}
function svgConv() {
  const svg = $('convSvg'), W = 520, H = 300, rows = EV.length, rh = (H - 24) / rows, acc = R.filter((r) => r.accepted);
  let g = '';
  EV.forEach((e, ri) => {
    const y0 = 6 + ri * rh, lo = e.mc_det_ref * 0.72, hi = e.mc_det_ref * 1.38, L = 96, Rr = 22;
    const X = (v) => L + (Math.log(Math.max(lo, Math.min(hi, v))) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) * (W - L - Rr);
    g += `<text x="0" y="${y0 + 16}" fill="${C.ink}" font-size="12" font-family="JetBrains Mono">${e.label}</text><text x="0" y="${y0 + 31}" fill="${C.faint}" font-size="10" font-family="JetBrains Mono">LIGO ${fmt(e.mc_det_ref)}</text>`;
    g += `<rect x="${X(e.mc_det_lo)}" y="${y0 + 2}" width="${X(e.mc_det_hi) - X(e.mc_det_lo)}" height="${rh - 14}" rx="4" fill="rgba(255,179,92,0.12)" stroke="rgba(255,179,92,0.35)"/>`;
    acc.forEach((r, i) => {
      const m = r.real[e.id]; if (!m) return; const yy = y0 + 8 + i * ((rh - 22) / Math.max(1, acc.length - 1));
      const col = i === acc.length - 1 ? C.net : `rgba(185,162,255,${0.25 + 0.5 * i / Math.max(1, acc.length - 1)})`;
      g += `<line x1="${X(m.lo)}" x2="${X(m.hi)}" y1="${yy}" y2="${yy}" stroke="${col}" stroke-width="2.2" stroke-linecap="round"/><circle cx="${X(m.mc)}" cy="${yy}" r="3.4" fill="${col}"/><text x="${W - 2}" y="${yy + 3.5}" fill="${C.faint}" font-size="9" font-family="JetBrains Mono" text-anchor="end">${r._num}</text>`;
    });
  });
  g += `<text x="96" y="${H - 4}" fill="${C.faint}" font-size="10" font-family="Figtree">Shaded: LIGO 90%, detector frame. Lines: ChirpNet 68%, kept rungs ${acc.map((r) => r._num).join(', ')} from top.</text>`;
  svg.innerHTML = g;
}
function svgArch() {
  const s = $('archSvg'), boxes = [['Input', '2 × 4096', 'H1 + L1, 2 s'], ['Stem', '24 × 1024', 'conv ÷4'], ['Block 1', '32 × 512', 'k15'], ['Block 2', '48 × 256', 'k9'], ['Block 3', '64 × 128', 'k7'], ['Block 4', '96 × 64', 'k5'], ['Block 5', '128 × 32', 'k3'], ['Pool', 'mean + max', '256'], ['Head', '128', 'MLP'], ['Out', '7', 'score, ℳ±σ, q±σ, t±σ']];
  const n = boxes.length, bw = 92, gap = (1100 - n * bw) / (n - 1); let g = '';
  boxes.forEach((b, i) => {
    const x = i * (bw + gap), hgt = i === 0 ? 120 : i < 7 ? 120 - i * 12 : 60, y = 85 - hgt / 2, col = i === 0 ? C.h1 : i === n - 1 ? C.net : 'rgba(205,214,255,0.35)';
    g += `<rect x="${x}" y="${y}" width="${bw}" height="${hgt}" rx="10" fill="rgba(205,214,255,0.04)" stroke="${col}"/>`;
    g += `<text x="${x + bw / 2}" y="${y + 18}" fill="${C.ink}" font-size="12" font-family="Figtree" font-weight="600" text-anchor="middle">${b[0]}</text>`;
    g += `<text x="${x + bw / 2}" y="${y + hgt / 2 + 8}" fill="${C.dim}" font-size="11" font-family="JetBrains Mono" text-anchor="middle">${b[1]}</text>`;
    if (i < n - 1) g += `<path d="M${x + bw + 4} 85 h${gap - 8}" stroke="${C.faint}" marker-end="url(#ar)"/>`;
  });
  g = `<defs><marker id="ar" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L6 3L0 6z" fill="${C.faint}"/></marker></defs>` + g;
  g += `<text x="550" y="166" fill="${C.faint}" font-size="11" font-family="Figtree" text-anchor="middle">${(DATA.release.params || 0).toLocaleString()} parameters · residual 1-D convolutions with GELU · about ${DATA.release.mmacs || '40'} million multiply-adds per window</text>`;
  s.innerHTML = g;
}

// ---------------------------------------------------------------- static text
(() => {
  const rel = DATA.release, found = EV.filter((e) => e.hit && e.hit.score > THR).length;
  const worst = Math.max(...EV.map((e) => e.hit ? Math.abs(e.hit.chirp_mass / e.mc_det_ref - 1) : 1));
  const within = Math.ceil(worst * 100);
  const faint = Math.min(...EV.map((e) => e.hit ? e.hit.score : 99));
  $('stats').innerHTML = `
    <div class="stat"><b>${found} of 4</b><span>real mergers found in raw LIGO recordings it had never seen</span></div>
    <div class="stat"><b>0 of ${DATA.bg_n.toLocaleString()}</b><span>windows of real detector noise as loud as even the faintest event (statistic ${fmt(faint)} vs ${fmt(BGMAX)})</span></div>
    <div class="stat"><b>±${within}%</b><span>the worst of its four chirp-mass estimates against LIGO’s published values, from 2 s of data and one forward pass</span></div>
    ${BNS ? `<div class="stat"><b>GW170817</b><span>the neutron-star merger, found at statistic ${fmt(BNS.hit.score)} against a loudest background of ${fmt(BNS.bg_max)}, chirp mass ${BNS.hit.chirp_mass.toFixed(3)} M☉ (LIGO ${BNS.mc_ref.toFixed(4)})</span></div>` : ''}`;
  $('verTag').textContent = `v2.0 · ${rel.version || ''}`;
  $('liveH').textContent = fmt(DATA.bg_live_s / 3600, 1);
  if (rel.samples) $('howData').innerHTML += ` Over the whole ratchet it saw about <b>${(rel.samples / 1e6).toFixed(1)} million</b> training windows drawn from ${rel.bank.toLocaleString()} distinct mergers, never the same window twice.`;
  $('codeBlock').innerHTML = `<span class="c"># pip install numpy scipy h5py</span>
<span class="k">import</span> chirpnet

net = chirpnet.ChirpNet(<span class="s">'chirpnet_weights.npz'</span>)
h1, t0 = chirpnet.load_gwosc(<span class="s">'H-H1_LOSC_4_V2-1126259446-32.hdf5'</span>)
l1, _  = chirpnet.load_gwosc(<span class="s">'L-L1_LOSC_4_V2-1126259446-32.hdf5'</span>)

<span class="k">for</span> hit <span class="k">in</span> net.scan(h1, l1, gps_start=t0):
    <span class="k">print</span>(hit)

<span class="c"># {'gps': ${EV[0].hit ? EV[0].hit.gps.toFixed(3) : '…'}, 'score': ${EV[0].hit ? fmt(EV[0].hit.score, 2) : '…'},</span>
<span class="c">#  'h1_alone': ${EV[0].hit && EV[0].hit.h1_alone != null ? fmt(EV[0].hit.h1_alone, 2) : '…'}, 'l1_alone': ${EV[0].hit && EV[0].hit.l1_alone != null ? fmt(EV[0].hit.l1_alone, 2) : '…'},</span>
<span class="c">#  'far_per_hour': ${EV[0].hit ? fmt(EV[0].hit.far_per_hour, 2) : '…'}, 'far_is_upper_bound': True,</span>
<span class="c">#  'chirp_mass': ${EV[0].hit ? fmt(EV[0].hit.chirp_mass, 2) : '…'}, 'chirp_mass_68': [${EV[0].hit ? EV[0].hit.chirp_mass_68.map((v) => fmt(v, 2)).join(', ') : '…'}],</span>
<span class="c">#  'mass_ratio': ${EV[0].hit ? fmt(EV[0].hit.mass_ratio, 2) : '…'}, 'm1': ${EV[0].hit ? fmt(EV[0].hit.m1, 1) : '…'}, 'm2': ${EV[0].hit ? fmt(EV[0].hit.m2, 1) : '…'}, ...}</span>

<span class="c"># full posterior for a hit: masses, spin, distance, inclination</span>
pe = chirpnet.ChirpPE(<span class="s">'chirpnet_pe_weights.npz'</span>)
post = pe.weigh(h1, l1, t0, hit[<span class="s">'gps'</span>])
post[<span class="s">'summary_90'</span>][<span class="s">'m1_source'</span>]    <span class="c"># [5%, median, 95%]</span>

<span class="c"># neutron stars: give it minutes of data (the noise spectrum needs it)</span>
bns = chirpnet.ChirpNetBNS(<span class="s">'chirpnet_bns_weights.npz'</span>)
bns.scan(h1_long, l1_long, gps_start=t0_long)`;
  $('files').innerHTML = `
    <div class="file"><code>chirpnet.py</code><span>the whole model: whitening, gating, forward pass, scanning, false-alarm rate, single-detector diagnostics. Numpy only.</span></div>
    <div class="file"><code>chirpnet_weights.npz</code><span>the black-hole finder (rung 4) plus the real-noise background used for false-alarm rates.</span></div>
    <div class="file"><code>chirpnet_coinc_weights.npz</code><span>the two-stream coincidence checker (rung 8). Loaded automatically; together they are rung 9, ${(rel.params || 0).toLocaleString()} weights.</span></div>
    <div class="file"><code>chirpnet_pe_weights.npz</code><span>the posterior network: CNN embedding plus an 8-layer normalizing flow.</span></div>
    <div class="file"><code>chirpnet_bns_weights.npz</code><span>the neutron-star model, with its own background from 34 minutes of O2 noise.</span></div>
    <div class="file"><code>demo.py</code><span>runs the four black-hole events (their LIGO data is included), their posteriors, and GW170817 if you add its two files.</span></div>
    <div class="file"><code>MODEL_CARD.md</code><span>training data, every rung of the ratchet, validation numbers and limits.</span></div>
    <div class="file"><code>training/</code><span>the simulator, sampler, training and evaluation scripts that produced it.</span></div>
    <p class="note" style="margin:6px 0 0">The files are attached in the chat alongside this page.</p>`;
})();

// ---------------------------------------------------------------- neutron stars
const bnsState = { view: 0, grids: null };
function bGrid(k) { if (!bnsState.grids) bnsState.grids = [u8(BNS.spec), u8(BNS.specRaw), u8(BNS.specGated)]; return bnsState.grids[k]; }
function drawBSpec() {
  const { ctx, w, h } = setup($('bSpec')), v = bnsState.view, nf = BNS.nf, nt = BNS.nt, fr = BNS.freqs;
  heat(ctx, w, h, bGrid(v), nf, nt, 0, nt, v === 0 ? 150 : 110);
  const x = (t) => (t - BNS.t_lo) / (BNS.t_hi - BNS.t_lo) * w, yf = (f) => h - Math.log(f / fr[0]) / Math.log(fr[nf - 1] / fr[0]) * h;
  ctx.strokeStyle = 'rgba(255,255,255,0.10)'; ctx.lineWidth = 1;
  for (const f of [32, 64, 128, 256]) { const y = yf(f); ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); axisText(ctx, f + ' Hz', 6, y - 3); }
  for (let t = -30; t <= 0; t += (w < 520 ? 10 : 5)) axisText(ctx, (t === 0 ? "0" : t) + " s", x(t), h - 5, "center", "rgba(232,234,242,0.55)");
  if (v === 2 && BNS.gate.length) { const [a, b] = BNS.gate[0]; ctx.strokeStyle = 'rgba(123,227,168,0.75)'; ctx.setLineDash([3, 3]); ctx.strokeRect(x(a), 4, x(b) - x(a), h - 20); ctx.setLineDash([]); axisText(ctx, 'gated', x(b) + 4, 16, 'left', C.ok); }
  if (v === 1) axisText(ctx, '← the glitch, ~1.1 s before merger', x(-1.1) - 8, 16, 'right', C.hit);
  const mc = BNS.hit.chirp_mass, tc = BNS.hit.dt_ms / 1000;
  const track = (m, t1, col, off) => { ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 1.6; ctx.setLineDash([6, 5]); ctx.globalAlpha = 0.9; ctx.beginPath(); let st = false;
    for (let t = BNS.t_lo; t < t1 - 0.003; t += 0.01) { const f = chirpF(m, t1 - t); if (f < fr[0]) continue; if (f > 400) break; const X = x(t), Y = yf(f) - off; if (!st) { ctx.moveTo(X, Y); st = true; } else ctx.lineTo(X, Y); }
    ctx.stroke(); ctx.restore(); };
  track(BNS.mc_ref, 0, 'rgba(255,179,92,0.8)', -9); track(mc, tc, C.net, 9);
  $('bChirpLbl').innerHTML = `<span style="color:var(--net)">- -</span> ChirpNet ℳ ${mc.toFixed(3)} · <span style="color:var(--h1)">- -</span> LIGO ${BNS.mc_ref.toFixed(4)} (drawn either side of the track)`;
  $('bSpecLbl').textContent = ['H1 + L1 (cleaned) · time–frequency energy, 24–420 Hz', 'Livingston as recorded · same scale', 'Livingston after automatic gating'][v];
}
function drawBScore() {
  const { ctx, w, h } = setup($('bScore')), tt = BNS.scan_t, a = BNS.scan_s, r = BNS.scan_raw_nogate, ymin = -20, ymax = Math.max(16, Math.ceil(Math.max(...a) + 2));
  const L = 34, X = (t) => L + (t - tt[0]) / (tt[tt.length - 1] - tt[0]) * (w - L - 6), Y = (v) => h - 14 - (Math.max(ymin, Math.min(ymax, v)) - ymin) / (ymax - ymin) * (h - 22);
  ctx.strokeStyle = C.grid; for (const v of [-10, 0, 10]) { ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(w, Y(v)); ctx.stroke(); axisText(ctx, String(v), L - 6, Y(v) + 3, 'right', 'rgba(232,234,242,0.45)'); }
  ctx.fillStyle = 'rgba(100,106,130,0.18)'; ctx.fillRect(L, Y(BNS.bg_max), w - L, Y(ymin) - Y(BNS.bg_max));
  ctx.strokeStyle = C.faint; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(L, Y(BNS.bg_max)); ctx.lineTo(w, Y(BNS.bg_max)); ctx.stroke(); ctx.setLineDash([]);
  const line = (arr, col, wd, dash) => { ctx.strokeStyle = col; ctx.lineWidth = wd; ctx.setLineDash(dash || []); ctx.beginPath(); arr.forEach((v, i) => { const xx = X(tt[i]), yy = Y(v); i ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy); }); ctx.stroke(); ctx.setLineDash([]); };
  line(r, 'rgba(255,111,160,0.75)', 1.4, [4, 3]); line(a, C.net, 2);
  let k = 0; a.forEach((v, i) => { if (v > a[k]) k = i; });
  ctx.fillStyle = C.hit; ctx.beginPath(); ctx.arc(X(tt[k]), Y(a[k]), 4, 0, 6.283); ctx.fill(); axisText(ctx, a[k].toFixed(1), X(tt[k]) + 8, Y(a[k]) + 4, 'left', C.hit);
  for (const t of [-20, -10, 0, 10, 20]) if (t >= tt[0] && t <= tt[tt.length - 1]) axisText(ctx, (t > 0 ? '+' : '') + t + ' s', X(t), h - 2, 'center', 'rgba(232,234,242,0.45)');
  axisText(ctx, 'window end, relative to merger', w - 6, 12, 'right', 'rgba(154,160,184,0.7)');
}
function renderBNS() {
  if (!BNS) return;
  drawBSpec(); drawBScore();
  const p = BNS.hit, det = p.score > p.threshold, hrs = BNS.bg_live_s / 3600;
  $('bVerdict').classList.toggle('no', !det); $('bVerdictTxt').textContent = det ? 'Detected' : 'Not detected';
  $('bVerdictSub').innerHTML = `Detection statistic <b>${fmt(p.score)}</b>. The loudest of <b>${BNS.bg_n.toLocaleString()}</b> time-slid windows of the same day’s noise (${fmt(hrs, 1)} hours) reached ${fmt(BNS.bg_max)}. False-alarm rate below <b>${fmt(p.far_per_hour, 2)}</b> per hour.`;
  const mc = p.chirp_mass, err = (mc / BNS.mc_ref - 1) * 100, [lo, hi] = p.chirp_mass_68, inside = BNS.mc_ref >= lo && BNS.mc_ref <= hi;
  $('bMc').innerHTML = `${mc.toFixed(3)}<small>M☉</small>`;
  $('bMcSub').innerHTML = `LIGO: <b>${BNS.mc_ref.toFixed(4)}</b> M☉ (±0.0001). Off by <b>${(err >= 0 ? '+' : '') + err.toFixed(1)}%</b>. ${inside ? 'LIGO’s value is inside its 68% range.' : `Its 68% range, ${lo.toFixed(3)}–${hi.toFixed(3)}, misses LIGO’s value: on this event it is overconfident.`}`;
  const s = BNS.single, rg = BNS.raw_glitch;
  $('bKv').innerHTML = `
    <dt>Source-frame chirp mass</dt><dd>${(mc / (1 + BNS.z_ref)).toFixed(3)} M☉</dd>
    <dt>LIGO source frame</dt><dd>${BNS.mc_src_ref.toFixed(3)} M☉</dd>
    <dt>Merger time vs LIGO</dt><dd>${(p.dt_ms >= 0 ? '+' : '')}${fmt(p.dt_ms, 0)} ms</dd>
    <dt>Livingston as recorded, gated</dt><dd>${fmt(rg.score)} · ℳ ${rg.mc.toFixed(3)}</dd>
    <dt>…with the glitch left in</dt><dd style="color:var(--hit)">${fmt(BNS.raw_nogate)}</dd>
    <dt>Hanford alone / Livingston alone</dt><dd>${fmt(s.h1_alone)} / ${fmt(s.l1_alone)}</dd>`;
}
let bPlay = null;
function playArr(b64, btn) {
  try {
    actx = actx || new (window.AudioContext || window.webkitAudioContext)();
    if (bPlay) { try { bPlay.stop(); } catch (_) { } bPlay = null; }
    const a = f16(b64), up = 8, sr = BNS.audio_fs * up, n = a.length * up, buf = actx.createBuffer(1, n, sr), ch = buf.getChannelData(0);
    for (let i = 0; i < n; i++) { const p = i / up, j = Math.floor(p), u = p - j; ch[i] = 0.7 * ((a[j] || 0) * (1 - u) + (a[j + 1] || 0) * u); }
    const fade = 2000; for (let i = 0; i < fade; i++) { ch[i] *= i / fade; ch[n - 1 - i] *= i / fade; }
    const src = actx.createBufferSource(); src.buffer = buf; src.connect(actx.destination); src.start(); bPlay = src;
  } catch (_) { $('bNote').textContent = 'Audio is not available in this browser.'; }
}
if (BNS) {
  const el = $('bRungs');
  BNS.rungs.forEach((r, i) => {
    const c = document.createElement('div'), rel = r.name === BNS.release; c.className = 'rung ' + (r.accepted ? 'acc' : 'rej'); if (rel) c.style.borderColor = 'rgba(185,162,255,0.6)';
    const v = r.val, e = v.eff;
    c.innerHTML = `<div class="tag"><span>Neutron-star rung ${i + 1}</span><b style="${rel ? 'color:var(--net)' : ''}">${rel ? 'Kept · released' : r.accepted ? 'Kept' : 'Rejected'}</b></div>
      <h3>${titleOf(r.note)}</h3><p>${(() => { const d = r.note.includes(':') ? r.note.split(':').slice(1).join(':').trim() : ''; return d.charAt(0).toUpperCase() + d.slice(1); })()}</p>
      <div class="m"><div><span>Found at SNR 12 / 15</span><b>${Math.round(100 * e['12'])}% / ${Math.round(100 * e['15'])}%</b></div><div><span>Chirp-mass error</span><b>${fmt(100 * v.mc_relerr_med)}%</b></div>
      <div><span>GW170817 vs loudest noise</span><b style="color:${r.real.score > r.real.bg_max ? 'var(--ok)' : 'var(--hit)'}">${fmt(r.real.score)} vs ${fmt(r.real.bg_max)}</b></div><div><span>GW170817 ℳ (LIGO 1.1975)</span><b>${r.real.mc.toFixed(3)}</b></div></div>`;
    el.appendChild(c);
  });
  document.querySelectorAll('#bView button').forEach((b) => b.addEventListener('click', () => { bnsState.view = +b.dataset.v; document.querySelectorAll('#bView button').forEach((x) => x.setAttribute('aria-pressed', String(x === b))); drawBSpec(); }));
  $('bListen').addEventListener('click', () => playArr(BNS.audio));
  $('bListenT').addEventListener('click', () => playArr(BNS.tpl));
} else { const sec = $('bns'); if (sec) sec.hidden = true; }

// ---------------------------------------------------------------- posteriors
const PE = DATA.pe || null;
let peCur = 0;
const PEV = PE ? EV.filter((e) => PE.events[e.id]).map((e) => ({ e, p: PE.events[e.id], s: null })) : [];
function peS(k) { const o = PEV[k]; if (!o.s) { o.s = {}; for (const key in o.p.samples) o.s[key] = f16(o.p.samples[key]); } return o.s; }
function qtl(a, q) { const b = Float32Array.from(a).sort(); return b[Math.min(b.length - 1, Math.max(0, Math.round(q * (b.length - 1))))]; }
function peAxes(ctx, w, h, P, xt, yt, xl, yl) {
  ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
  xt.forEach((v) => { const x = P.X(v); ctx.beginPath(); ctx.moveTo(x, P.t); ctx.lineTo(x, h - P.b); ctx.stroke(); axisText(ctx, String(v), x, h - P.b + 14, 'center', C.dim); });
  yt.forEach((v) => { const y = P.Y(v); ctx.beginPath(); ctx.moveTo(P.l, y); ctx.lineTo(w - P.r, y); ctx.stroke(); axisText(ctx, String(v), P.l - 6, y + 3.5, 'right', C.dim); });
  if (xl) axisText(ctx, xl, (P.l + w - P.r) / 2, h - 3, 'center', C.faint);
  if (yl) { ctx.save(); ctx.translate(11, (P.t + h - P.b) / 2); ctx.rotate(-Math.PI / 2); axisText(ctx, yl, 0, 0, 'center', C.faint); ctx.restore(); }
}
function frame(w, h, x0, x1, y0, y1, logx) {
  const P = { l: 44, r: 10, t: 8, b: 34 };
  const fx = logx ? (v) => Math.log(v) : (v) => v;
  P.X = (v) => P.l + (fx(v) - fx(x0)) / (fx(x1) - fx(x0)) * (w - P.l - P.r);
  P.Y = (v) => h - P.b - (v - y0) / (y1 - y0) * (h - P.t - P.b);
  return P;
}
function nice(lo, hi, n) { const span = hi - lo, st = Math.pow(10, Math.floor(Math.log10(span / n))), m = [1, 2, 5, 10].find((k) => span / (k * st) <= n) * st, out = []; for (let v = Math.ceil(lo / m) * m; v <= hi + 1e-9; v += m) out.push(+v.toFixed(6)); return out; }
function cross(ctx, x, y) { ctx.strokeStyle = C.h1; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x - 7, y); ctx.lineTo(x + 7, y); ctx.moveTo(x, y - 7); ctx.lineTo(x, y + 7); ctx.stroke(); ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fillStyle = C.h1; ctx.fill(); }
function dots(ctx, P, xs, ys, col) { ctx.fillStyle = col; for (let i = 0; i < xs.length; i++) { ctx.beginPath(); ctx.arc(P.X(xs[i]), P.Y(ys[i]), 1.5, 0, 6.3); ctx.fill(); } }
function renderPE() {
  if (!PE) return;
  const o = PEV[peCur], s = peS(peCur), ref = o.p.ref, sm = o.p.summary;
  document.querySelectorAll('#pePills button').forEach((b, i) => b.setAttribute('aria-pressed', String(i === peCur)));
  $('peNote').textContent = `${o.p.n.toLocaleString()} of 20,000 posterior draws shown · window placed at ChirpNet’s own merger time (${((o.p.t_hit - o.p.t_ligo) * 1000).toFixed(0)} ms from LIGO’s)`;
  // masses
  { const { ctx, w, h } = setup($('peM')); const m1 = s.m1_source, m2 = s.m2_source;
    const x0 = Math.min(qtl(m1, 0.003), ref.m1) * 0.9, x1 = Math.max(qtl(m1, 0.99), ref.m1) * 1.05, y0 = Math.min(qtl(m2, 0.003), ref.m2) * 0.9, y1 = Math.max(qtl(m2, 0.997), ref.m2) * 1.08;
    const P = frame(w, h, x0, x1, y0, y1); peAxes(ctx, w, h, P, nice(x0, x1, 5), nice(y0, y1, 4), 'm₁ (M☉)', 'm₂ (M☉)');
    ctx.setLineDash([4, 4]); ctx.strokeStyle = 'rgba(205,214,255,0.25)'; ctx.beginPath(); ctx.moveTo(P.X(Math.max(x0, y0)), P.Y(Math.max(x0, y0))); ctx.lineTo(P.X(Math.min(x1, y1)), P.Y(Math.min(x1, y1))); ctx.stroke(); ctx.setLineDash([]);
    dots(ctx, P, m1, m2, 'rgba(185,162,255,0.32)'); cross(ctx, P.X(ref.m1), P.Y(ref.m2));
    $('peMl').innerHTML = `<span style="color:var(--h1)">+</span> LIGO median`; }
  // distance vs inclination
  { const { ctx, w, h } = setup($('peD')); const dl = s.luminosity_distance_mpc, ci = s.cos_inclination, inc = Float32Array.from(ci, (c) => Math.acos(Math.max(-1, Math.min(1, c))) * 180 / Math.PI);
    const x0 = Math.min(qtl(dl, 0.003), ref.dl) * 0.85, x1 = Math.max(qtl(dl, 0.997), ref.dl) * 1.1;
    const P = frame(w, h, x0, x1, 0, 180, true); const xt = [100, 200, 300, 500, 1000, 2000, 3000].filter((v) => v >= x0 && v <= x1);
    peAxes(ctx, w, h, P, xt, [0, 45, 90, 135, 180], 'luminosity distance (Mpc)', 'inclination (°)');
    dots(ctx, P, dl, inc, 'rgba(185,162,255,0.32)');
    ctx.strokeStyle = C.h1; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(P.X(ref.dl), P.t); ctx.lineTo(P.X(ref.dl), h - P.b); ctx.stroke(); ctx.setLineDash([]);
    axisText(ctx, `LIGO ${ref.dl} Mpc`, P.X(ref.dl) + 5, P.t + 12, 'left', C.h1); }
  // chi_eff histogram
  { const { ctx, w, h } = setup($('peX')); const x = s.chi_eff, nb = 40, hist = new Float32Array(nb);
    for (const v of x) { const k = Math.floor((v + 1) / 2 * nb); if (k >= 0 && k < nb) hist[k]++; }
    const mx = Math.max(...hist), P = frame(w, h, -1, 1, 0, mx * 1.12);
    peAxes(ctx, w, h, P, [-1, -0.5, 0, 0.5, 1], [], 'χ_eff', '');
    const [c, up, dn] = ref.chi; ctx.fillStyle = 'rgba(255,179,92,0.13)'; ctx.fillRect(P.X(c - dn), P.t, P.X(c + up) - P.X(c - dn), h - P.b - P.t);
    ctx.strokeStyle = C.h1; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(P.X(c), P.t); ctx.lineTo(P.X(c), h - P.b); ctx.stroke();
    ctx.fillStyle = 'rgba(185,162,255,0.55)'; const bw = (w - P.l - P.r) / nb;
    for (let k = 0; k < nb; k++) { const y = P.Y(hist[k]); ctx.fillRect(P.l + k * bw + 1, y, bw - 2, h - P.b - y); } }
  const f = (a, d) => `${fmt(a[1], d)} <span style="color:var(--faint)">[${fmt(a[0], d)}, ${fmt(a[2], d)}]</span>`;
  const inside = (a, v) => v >= a[0] && v <= a[2];
  const cell = (lab, a, d, ligo, lv) => `<div><span>${lab}</span><b>${f(a, d)}</b><em>LIGO ${ligo}${inside(a, lv) ? ' <i class="okk">✓ inside</i>' : ''}</em></div>`;
  $('peTab').innerHTML =
    cell('m₁ (M☉)', sm.m1_source, 1, fmt(ref.m1), ref.m1) + cell('m₂ (M☉)', sm.m2_source, 1, fmt(ref.m2), ref.m2) +
    cell('Chirp mass ℳ (M☉)', sm.chirp_mass_source, 1, `${fmt(ref.mc[0])} +${fmt(ref.mc[1])}/−${fmt(ref.mc[2])}`, ref.mc[0]) +
    cell('χ<sub>eff</sub>', sm.chi_eff, 2, `${ref.chi[0] >= 0 ? '+' : ''}${fmt(ref.chi[0], 2)}`, ref.chi[0]) +
    cell('Distance (Mpc)', sm.luminosity_distance_mpc, 0, `${ref.dl}`, ref.dl);
}
function svgPP() {
  if (!PE) return;
  const svg = $('ppSvg'), W = 520, H = 330, L = 44, B = 36, T = 10, Rr = 150, pp = PE.pp, n = pp.n;
  const X = (v) => L + v * (W - L - Rr), Y = (v) => H - B - v * (H - B - T);
  let g = '', up = '', dn = '';
  pp.grid.forEach((p) => { const e = 1.96 * Math.sqrt(p * (1 - p) / n); up += `${X(p)},${Y(Math.min(1, p + e))} `; dn = `${X(p)},${Y(Math.max(0, p - e))} ` + dn; });
  g += `<polygon points="${up}${dn}" fill="rgba(205,214,255,0.07)"/>`;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) g += `<text x="${L - 8}" y="${Y(v) + 4}" fill="${C.dim}" font-size="11" font-family="JetBrains Mono" text-anchor="end">${v}</text><text x="${X(v)}" y="${H - 18}" fill="${C.dim}" font-size="11" font-family="JetBrains Mono" text-anchor="middle">${v}</text>`;
  g += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="rgba(205,214,255,0.35)" stroke-dasharray="4 4"/>`;
  g += `<text x="${X(0.5)}" y="${H - 2}" fill="${C.faint}" font-size="11" font-family="Figtree" text-anchor="middle">posterior percentile</text>`;
  const names = ['chirp mass', 'mass ratio', 'χ_eff', 'distance', 'inclination', 'merger time'], cols = ['#b9a2ff', '#ff6fa0', '#7be3a8', '#ffb35c', '#58d6ea', '#646a82'];
  const keys = Object.keys(pp.maxdev);
  pp.cdf.forEach((c, j) => {
    g += `<polyline fill="none" stroke="${cols[j]}" stroke-width="${j === 5 ? 1.2 : 1.8}" ${j === 5 ? 'stroke-dasharray="3 3"' : ''} points="${c.map((v, i) => `${X(pp.grid[i])},${Y(v)}`).join(' ')}"/>`;
    g += `<rect x="${W - Rr + 16}" y="${T + 8 + j * 22}" width="12" height="3" rx="1.5" fill="${cols[j]}"/><text x="${W - Rr + 34}" y="${T + 13 + j * 22}" fill="${j === 5 ? C.faint : C.ink}" font-size="12" font-family="Figtree">${names[j]}</text><text x="${W - 4}" y="${T + 13 + j * 22}" fill="${C.dim}" font-size="10.5" font-family="JetBrains Mono" text-anchor="end">${pp.maxdev[keys[j]].toFixed(3)}</text>`;
  });
  g += `<text x="${W - Rr + 16}" y="${T + 8 + 6 * 22 + 10}" fill="${C.faint}" font-size="10.5" font-family="Figtree">max gap from diagonal;</text><text x="${W - Rr + 16}" y="${T + 8 + 6 * 22 + 24}" fill="${C.faint}" font-size="10.5" font-family="Figtree">chance allows ~0.07</text>`;
  svg.innerHTML = g;
}
if (PE) {
  const pills = $('pePills');
  PEV.forEach((o, i) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = o.e.label; b.addEventListener('click', () => { peCur = i; renderPE(); }); pills.appendChild(b); });
} else { const sec = $('pe'); if (sec) sec.hidden = true; }

function renderAll() { renderEvent(); svgEff(); svgConv(); svgArch(); renderPE(); svgPP(); renderBNS(); }
renderAll(); runLab();
let rz = null; addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => { renderEvent(); runLab(); renderPE(); renderBNS(); }, 150); });
})();
