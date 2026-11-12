(() => {
'use strict';
const $ = (id) => document.getElementById(id);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let C = {};
function colors() { C = {}; ['ink', 'dim', 'faint', 'grid', 'line2', 'panel', 'panel2', 'copper', 'cufill', 'net', 'li', 'gr', 'lfp', 'bad', 'ok'].forEach((k) => { C[k] = css('--' + k); }); }
colors();
const fmt = (x, d = 1) => (x == null || !isFinite(x)) ? '—' : Number(x).toFixed(d);
const pct = (x, d = 1) => (x == null || !isFinite(x)) ? '—' : (100 * x).toFixed(d) + '%';
const mAh = (x, d = 0) => (x == null || !isFinite(x)) ? '—' : (1000 * x).toFixed(d);
function u8(b64) { const s = atob(b64), a = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i); return a; }
function f16(b64) {
  const b = u8(b64), n = b.length >> 1, out = new Float32Array(n);
  for (let i = 0; i < n; i++) { const h = b[2 * i] | (b[2 * i + 1] << 8), s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023; out[i] = e === 0 ? s * m * 5.960464477539063e-8 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * Math.pow(2, e - 15); }
  return out;
}
function setup(cv) {
  const r = cv.getBoundingClientRect(), d = Math.min(devicePixelRatio || 1, 2), w = Math.max(10, Math.round(r.width)), h = Math.max(10, Math.round(r.height));
  if (cv.width !== w * d || cv.height !== h * d) { cv.width = w * d; cv.height = h * d; }
  const ctx = cv.getContext('2d'); ctx.setTransform(d, 0, 0, d, 0, 0); ctx.clearRect(0, 0, w, h); return { ctx, w, h };
}
function txt(ctx, s, x, y, align, color, size, weight) { ctx.font = `${weight || 400} ${size || 10}px "Chivo Mono", ui-monospace, monospace`; ctx.fillStyle = color || C.dim; ctx.textAlign = align || 'left'; ctx.fillText(s, x, y); }
function mix(a, b, t) {   // two #rrggbb colours
  const p = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16)), A = p(a), B = p(b);
  return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(',')})`;
}
const VG = DATA.vgrid;

// ------------------------------------------------------------ axes helper
function frame(cv, xr, yr, opt = {}) {
  const { ctx, w, h } = setup(cv), L = opt.L ?? 44, R = opt.R ?? 10, T = opt.T ?? 10, B = opt.B ?? 22;
  const X = (v) => L + (v - xr[0]) / (xr[1] - xr[0]) * (w - L - R), Y = (v) => T + (1 - (v - yr[0]) / (yr[1] - yr[0])) * (h - T - B);
  ctx.lineWidth = 1;
  (opt.yt || []).forEach((v) => { ctx.strokeStyle = C.grid; ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(w - R, Y(v)); ctx.stroke(); txt(ctx, opt.yf ? opt.yf(v) : String(v), L - 6, Y(v) + 3.5, 'right', C.faint); });
  (opt.xt || []).forEach((v) => { ctx.strokeStyle = C.grid; ctx.beginPath(); ctx.moveTo(X(v), T); ctx.lineTo(X(v), h - B); ctx.stroke(); txt(ctx, opt.xf ? opt.xf(v) : String(v), X(v), h - 6, 'center', C.faint); });
  if (opt.xl) txt(ctx, opt.xl, w - R, h - B - 5, 'right', C.faint);
  if (opt.yl) txt(ctx, opt.yl, L + 4, T + 10, 'left', C.faint);
  return { ctx, w, h, X, Y, L, R, T, B };
}
function curve(ctx, X, Y, q, color, width, dash) {   // V against Q, from Q(V) on the grid
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []); ctx.beginPath(); let s = false;
  for (let i = 0; i < VG.length; i++) { const v = q[i]; if (!isFinite(v)) continue; const x = X(v), y = Y(VG[i]); if (!s) { ctx.moveTo(x, y); s = true; } else ctx.lineTo(x, y); }
  ctx.stroke(); ctx.setLineDash([]);
}
const lastQ = (q) => { for (let i = q.length - 1; i >= 0; i--) if (isFinite(q[i])) return q[i]; return NaN; };

// ------------------------------------------------------------ hero
function renderHero() {
  const H = DATA.hero, cs = H.curves.map((c) => ({ ...c, _q: f16(c.q) }));
  const f = frame($('heroCv'), [0, 1.12], [1.86, 3.5], { yt: [2.0, 2.5, 3.0, 3.5], xt: [0, 0.2, 0.4, 0.6, 0.8, 1.0], xf: (v) => v.toFixed(1), xl: 'charge out (Ah) →', yl: 'volts', B: 24 });
  const { ctx, X, Y } = f, n = cs.length;
  cs.forEach((c, i) => curve(ctx, X, Y, c._q, mix(C.faint, C.copper, i / Math.max(1, n - 1)), i === n - 1 ? 2.2 : 1.3));
  // the hold: what each cycle got back at 2.0 V, stacked below the curves
  const y0 = Y(2.0), dy = (Y(1.86) - y0) / (n + 1);
  cs.forEach((c, i) => { const col = mix(C.faint, C.copper, i / Math.max(1, n - 1)), y = y0 + (i + 1) * dy;
    ctx.strokeStyle = col; ctx.lineWidth = 2.4; ctx.beginPath(); ctx.moveTo(X(c.qcc), y); ctx.lineTo(X(c.qd), y); ctx.stroke(); ctx.fillStyle = col; ctx.beginPath(); ctx.arc(X(c.qcc), y, 2, 0, 7); ctx.fill(); });
  txt(ctx, 'held at 2.0 V →', X(0.62), Y(1.93), 'right', C.dim, 10);
  $('heroTitle').textContent = `Cell ${H.label} · ${H.policy} charging · cycles ${cs[0].c}–${cs[n - 1].c}`;
  $('heroLeg').innerHTML = `<span><i style="background:var(--faint)"></i>cycle ${cs[0].c}</span><span><i style="background:var(--copper)"></i>cycle ${cs[n - 1].c}</span>`;
  $('heroCap').textContent = `Each line is one 4C discharge of the same real cell, early cycles grey, late ones copper. The short bars underneath are the charge the cell gave back while held at 2.0 V: ${mAh(cs[0].qd - cs[0].qcc)} mAh when new, ${mAh(cs[n - 1].qd - cs[n - 1].qcc)} mAh at the end. Its life ended at cycle ${H.life}, when total capacity fell below 0.88 Ah.`;
}

// ------------------------------------------------------------ hidden capacity scatter
function renderHidden() {
  const S = DATA.scatter, qd = f16(S.qd), pr = f16(S.pred), qc = f16(S.qcc), n = qd.length;
  const f = frame($('hidCv'), [0.8, 1.12], [0.68, 1.15], { yt: [0.7, 0.8, 0.9, 1.0, 1.1], xt: [0.8, 0.9, 1.0, 1.1], xf: (v) => v.toFixed(1), yf: (v) => v.toFixed(1), xl: 'measured, with the hold (Ah) →', yl: 'estimate (Ah)', L: 40 });
  const { ctx, X, Y } = f;
  ctx.strokeStyle = C.line2; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(0.8), Y(0.8)); ctx.lineTo(X(1.12), Y(1.12)); ctx.stroke();
  ctx.strokeStyle = C.bad; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(0.88), Y(0.68)); ctx.lineTo(X(0.88), Y(1.15)); ctx.stroke(); ctx.setLineDash([]); txt(ctx, 'end of life', X(0.88) + 4, Y(0.7), 'left', C.bad, 10);
  ctx.globalAlpha = 0.28; ctx.fillStyle = C.faint; for (let i = 0; i < n; i++) ctx.fillRect(X(qd[i]) - 1, Y(qc[i]) - 1, 2, 2);
  ctx.globalAlpha = 0.4; ctx.fillStyle = C.net; for (let i = 0; i < n; i++) ctx.fillRect(X(qd[i]) - 1.2, Y(pr[i]) - 1.2, 2.4, 2.4); ctx.globalAlpha = 1;
  const R = DATA.real.with_ref;
  $('hidSub').textContent = `${R.n.toLocaleString()} readings of ${R.cells} cells, every 25 cycles. Teal: CellSense's estimate. Grey: the 4C curve's own end, which falls further below the truth as the cell ages.`;
  const rows = DATA.hidTable;
  $('hidTable').innerHTML = `<tr><th>Stage of life</th><th class="num">Readings</th><th class="num">CellSense error</th><th class="num">Curve's end error</th></tr>` +
    rows.map((r) => `<tr${r.stage === 'all' ? ' class="hl"' : ''}><td>${r.stage === 'all' ? 'All readings' : r.stage}</td><td class="num">${r.n.toLocaleString()}</td><td class="num">${mAh(r.mae_net, 1)} mAh</td><td class="num">${mAh(r.mae_base, 1)} mAh</td></tr>`).join('');
  $('hidNote').textContent = `Mean absolute error. Without the cell's early curve as a reference, CellSense's error is ${mAh(DATA.real.no_ref.q_total_mae_mAh / 1000, 1)} mAh. Pulse resistance, measured separately by the cycler: correlation ${fmt(R.ir_corr, 2)} with CellSense's estimate, typical error ${fmt(R.ir_mae_mohm, 2)} mΩ. Capacity fade since new: correlation ${fmt(R.fade_corr, 3)}. Near end of life the estimate still runs low: the slowest charge is the hardest to see.`;
}

// ------------------------------------------------------------ cell explorer
const CELLS = DATA.cells.map((c) => ({ ...c, _q: null }));
let cur = 0, ri = 0;
function cdata(c) { if (!c._q) { c._q = f16(c.curves); c._q0 = f16(c.q0); } return c; }
function rowQ(c, r) { return c._q.subarray(r * VG.length, (r + 1) * VG.length); }
function timeline(cv, c, yr, yt, yf, draw) {
  const f = frame(cv, [0, c.life * 1.08], yr, { yt, yf, xt: niceTicks(c.life * 1.08), L: 44, B: 20 });
  draw(f); const { ctx, X, T, h, B } = f; ctx.strokeStyle = C.ink; ctx.globalAlpha = 0.35; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(c.cyc[ri]), T); ctx.lineTo(X(c.cyc[ri]), h - B); ctx.stroke(); ctx.globalAlpha = 1;
  return f;
}
function niceTicks(mx) { const st = [100, 200, 250, 500, 1000].find((s) => mx / s <= 6) || 1000; const o = []; for (let v = 0; v <= mx; v += st) o.push(v); return o; }
function band(ctx, X, Y, xs, lo, hi, color) {
  ctx.fillStyle = color; ctx.globalAlpha = 0.16; ctx.beginPath(); xs.forEach((x, i) => { i ? ctx.lineTo(X(x), Y(hi[i])) : ctx.moveTo(X(x), Y(hi[i])); });
  for (let i = xs.length - 1; i >= 0; i--) ctx.lineTo(X(xs[i]), Y(lo[i])); ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
}
function line(ctx, X, Y, xs, ys, color, wd, dash) { ctx.strokeStyle = color; ctx.lineWidth = wd; ctx.setLineDash(dash || []); ctx.beginPath(); let s = false; xs.forEach((x, i) => { const y = ys[i]; if (y == null || !isFinite(y)) { s = false; return; } s ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)); s = true; }); ctx.stroke(); ctx.setLineDash([]); }
function renderCell() {
  const c = cdata(CELLS[cur]), P = c.P, n = c.cyc.length; ri = Math.min(ri, n - 1);
  document.querySelectorAll('#clist .cbtn').forEach((b, i) => b.setAttribute('aria-pressed', String(i === cur)));
  const sc = $('cScrub'); sc.max = n - 1; sc.value = ri; $('cCyc').textContent = c.cyc[ri];
  $('cTitle').textContent = `Cell ${c.label} · ${c.policy} · cycle ${c.cyc[ri]} of ${c.life}`;
  // discharge curve at this reading
  const f = frame($('cCurve'), [0, 1.12], [1.9, 3.5], { yt: [2.0, 2.5, 3.0, 3.5], xt: [0, 0.2, 0.4, 0.6, 0.8, 1.0], xf: (v) => v.toFixed(1), xl: 'Ah →', L: 40, B: 20 });
  curve(f.ctx, f.X, f.Y, c._q0, C.faint, 1.4); const q = rowQ(c, ri); curve(f.ctx, f.X, f.Y, q, C.copper, 2.2);
  const y2 = f.Y(2.0) - 2; f.ctx.strokeStyle = C.copper; f.ctx.lineWidth = 2; f.ctx.setLineDash([3, 3]); f.ctx.beginPath(); f.ctx.moveTo(f.X(c.qcc[ri]), y2); f.ctx.lineTo(f.X(c.qd[ri]), y2); f.ctx.stroke(); f.ctx.setLineDash([]);
  f.ctx.fillStyle = C.ink; f.ctx.beginPath(); f.ctx.arc(f.X(c.qd[ri]), y2, 3.2, 0, 7); f.ctx.fill();
  const pq = P.q_total[ri], ps = P.q_total_sig[ri]; f.ctx.fillStyle = C.net; f.ctx.globalAlpha = 0.25; f.ctx.fillRect(f.X(pq - ps), f.Y(2.12), f.X(pq + ps) - f.X(pq - ps), f.Y(1.9) - f.Y(2.12)); f.ctx.globalAlpha = 1;
  f.ctx.strokeStyle = C.net; f.ctx.lineWidth = 2; f.ctx.beginPath(); f.ctx.moveTo(f.X(pq), f.Y(2.14)); f.ctx.lineTo(f.X(pq), f.Y(1.9)); f.ctx.stroke();
  // capacity through life
  const capMin = Math.min(0.68, ...c.qcc) - 0.01;
  timeline($('cCap'), c, [capMin, 1.12], [0.7, 0.8, 0.9, 1.0, 1.1].filter((v) => v > capMin), (v) => v.toFixed(1), ({ ctx, X, Y }) => {
    ctx.strokeStyle = C.bad; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(0), Y(0.88)); ctx.lineTo(X(c.life * 1.08), Y(0.88)); ctx.stroke(); ctx.setLineDash([]);
    band(ctx, X, Y, c.cyc, P.q_total.map((v, i) => v - P.q_total_sig[i]), P.q_total.map((v, i) => v + P.q_total_sig[i]), C.net);
    line(ctx, X, Y, c.cyc, c.qcc, C.faint, 1.4, [4, 3]); line(ctx, X, Y, c.cyc, P.q_total, C.net, 2);
    ctx.fillStyle = C.ink; c.cyc.forEach((x, i) => { ctx.beginPath(); ctx.arc(X(x), Y(c.qd[i]), 1.8, 0, 7); ctx.fill(); }); });
  // modes, stacked
  const mx = Math.max(0.1, ...P.lli.map((v, i) => Math.max(0, v) + Math.max(0, P.lam_ne[i]) + Math.max(0, P.lam_pe[i]))) * 1.1;
  timeline($('cModes'), c, [0, mx], niceY(mx), (v) => Math.round(v * 100) + '%', ({ ctx, X, Y }) => {
    let acc = c.cyc.map(() => 0);
    [['lli', C.li], ['lam_ne', C.gr], ['lam_pe', C.lfp]].forEach(([k, col]) => { const top = acc.map((a, i) => a + Math.max(0, P[k][i]));
      ctx.fillStyle = col; ctx.globalAlpha = 0.75; ctx.beginPath(); c.cyc.forEach((x, i) => { i ? ctx.lineTo(X(x), Y(top[i])) : ctx.moveTo(X(x), Y(top[i])); });
      for (let i = c.cyc.length - 1; i >= 0; i--) ctx.lineTo(X(c.cyc[i]), Y(acc[i])); ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1; acc = top; }); });
  // lifetime forecast
  const lmax = Math.max(c.life * 1.6, ...P.life_hi.slice(0, n)) ;
  timeline($('cLife'), c, [0, Math.min(lmax, c.life * 2.6)], niceTicks(Math.min(lmax, c.life * 2.6)), (v) => String(v), ({ ctx, X, Y, w, R }) => {
    band(ctx, X, Y, c.cyc, P.life_lo, P.life_hi, C.copper); line(ctx, X, Y, c.cyc, P.life, C.copper, 2);
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(X(0), Y(c.life)); ctx.lineTo(X(c.life * 1.08), Y(c.life)); ctx.stroke(); });
  // readout
  const e = P.q_total[ri] - c.qd[ri];
  $('rHead').textContent = `${fmt(P.q_total[ri], 3)} Ah left`;
  $('rSub').textContent = `CellSense, ±${mAh(P.q_total_sig[ri])} mAh. The cycler measured ${fmt(c.qd[ri], 3)} Ah (${e >= 0 ? '+' : ''}${mAh(e)} mAh). The 4C curve alone shows ${fmt(c.qcc[ri], 3)} Ah.`;
  $('rKv').innerHTML = `
    <dt>Pulse resistance</dt><dd>${fmt(P.ir[ri], 1)} mΩ<small>measured ${fmt(c.ir[ri], 1)}</small></dd>
    <dt>Lithium lost</dt><dd>${pct(Math.max(0, P.lli[ri]), 1)}</dd>
    <dt>Graphite lost</dt><dd>${pct(Math.max(0, P.lam_ne[ri]), 1)}</dd>
    <dt>LFP lost</dt><dd>${pct(Math.max(0, P.lam_pe[ri]), 1)}</dd>
    <dt>End of life forecast</dt><dd>cycle ${Math.round(P.life[ri])}<small>${Math.round(P.life_lo[ri])}–${Math.round(P.life_hi[ri])}</small></dd>
    <dt>Actual end of life</dt><dd>cycle ${c.life}</dd>
    <dt>Charging 0–80%</dt><dd>${fmt(c.ct, 1)} min</dd>`;
  $('rNote').textContent = `${c.desc} Lost lithium and lost electrode material have no answer key: the cells were never taken apart. They are CellSense's physical explanation of the curve.`;
}
function niceY(mx) { const st = [0.02, 0.05, 0.1, 0.2].find((s) => mx / s <= 5) || 0.2; const o = []; for (let v = 0; v <= mx + 1e-9; v += st) o.push(+v.toFixed(3)); return o; }
(() => {
  const el = $('clist');
  CELLS.forEach((c, i) => { const b = document.createElement('button'); b.className = 'cbtn'; b.type = 'button';
    b.innerHTML = `<span class="n">${c.name}<i>${c.label}</i></span><span class="d">${c.policy} · ${c.life} cycles</span>`;
    b.addEventListener('click', () => { cur = i; ri = Math.floor(CELLS[i].cyc.length * 0.6); renderCell(); }); el.appendChild(b); });
  $('cScrub').addEventListener('input', (e) => { ri = +e.target.value; renderCell(); });
  ri = Math.floor(CELLS[0].cyc.length * 0.6);
})();

// ------------------------------------------------------------ lab: physics and network in the browser
CellSim.setup(DATA.lab.gh, DATA.lab.sph);
const NET = CellNet.load(WEIGHTS), NETP = CellNet.load(WEIGHTS_P);
function gsmooth(a, s) {   // as model.gsmooth: Gaussian along the voltage axis, edges held
  if (!s) return a; const r = Math.floor(4 * s + 0.5), k = []; let z = 0; for (let i = -r; i <= r; i++) { const w = Math.exp(-0.5 * (i / s) ** 2); k.push(w); z += w; }
  const L = a.length, o = new Float32Array(L); for (let i = 0; i < L; i++) { let v = 0; for (let j = -r; j <= r; j++) v += k[j + r] * a[Math.min(L - 1, Math.max(0, i + j))]; o[i] = v / z; } return o;
}
function features(Q1, T1, Q0, T0, sm) {   // as model.features: 8 channels x 250
  const L = VG.length, x = new Float32Array(8 * L), has = Q0 ? 1 : 0;
  const fill = (a, def) => { const o = new Float32Array(L), m = new Float32Array(L); let last = NaN, first = -1;
    for (let i = 0; i < L; i++) { if (a && isFinite(a[i])) { last = a[i]; m[i] = 1; if (first < 0) first = i; } o[i] = isFinite(last) ? last : NaN; }
    for (let i = 0; i < L; i++) if (!isFinite(o[i])) o[i] = first >= 0 ? a[first] : def; return [o, m]; };
  let [q1, m1] = fill(Q1, 0), [t1] = fill(T1, 30), [q0, m0] = has ? fill(Q0, 0) : [new Float32Array(L), new Float32Array(L)], [t0] = has ? fill(T0, 30) : [new Float32Array(L).fill(30)];
  if (sm) { q1 = gsmooth(q1, sm); t1 = gsmooth(t1, sm); q0 = gsmooth(q0, sm); t0 = gsmooth(t0, sm); }
  for (let i = 0; i < L; i++) { x[i] = q1[i]; x[L + i] = q0[i] * has; x[2 * L + i] = (q1[i] - q0[i]) * 10 * has; x[3 * L + i] = (t1[i] - 30) / 10; x[4 * L + i] = (t0[i] - 30) / 10 * has; x[5 * L + i] = m1[i]; x[6 * L + i] = m0[i] * has; x[7 * L + i] = has; }
  return x;
}
const lab = { li: 8, gr: 4, fp: 0, sl: 6, tr: 4, oh: 1, st: 1.1 };
let NEW = null;
function aged(P0) {
  const P = { ...P0 }, [x0, y0] = CellSim.fullState(P0, CellSim.table(CellSim.Up, P0.sig_p), CellSim.table(CellSim.Un, P0.sig_n), P0.V_full);
  const lli = lab.li / 100, lamn = lab.gr / 100, lamp = lab.fp / 100;
  P.Q_n = P0.Q_n * (1 - lamn); P.Q_p = P0.Q_p * (1 - lamp); P.Q_Li = Math.min(P0.Q_Li * (1 - lli) - lamn * P0.Q_n * y0 * 0.5, 0.995 * P.Q_n + 0.005 * P.Q_p);
  P.R_e = P0.R_e * (1 + lab.tr / 2); P.R_ohm = P0.R_ohm * (1 + lab.oh / 10); P.I0_n = P0.I0_n / (1 + lab.oh / 20); P.f_s = lab.sl / 100; P.R_s = DATA.lab.R_s;
  return P;
}
let labBusy = false, labAgain = false;
function runLab() {
  if (labBusy) { labAgain = true; return; } labBusy = true;
  requestAnimationFrame(() => setTimeout(() => {
    const t0 = performance.now(), P0 = DATA.lab.P;
    if (!NEW) { const o = CellSim.simulate(P0); const [q, t] = CellSim.qOfV(o, VG); NEW = { o, q, t, ir: CellSim.pulseR(P0) }; }
    const P = aged(P0), o = CellSim.simulate(P), [q, t] = CellSim.qOfV(o, VG), ir = CellSim.pulseR(P), ts = performance.now();
    const part = lab.st < 1.1 && lab.st < o.Q_cc, cut = (a) => Float32Array.from(a, (v, i) => (q[i] > lab.st ? NaN : v));   // a partial discharge goes to the third network
    const r = part ? CellNet.forward(NETP, features(cut(q), cut(t), NEW.q, NEW.t, NETP.meta.smooth || 0), VG.length, null)
                   : CellNet.forward(NET, features(q, t, NEW.q, NEW.t), VG.length, NET.meta.extra ? [Math.log10(300) / 3.5, 10 / 15] : null);
    $('lTime').textContent = `simulated in ${Math.round(ts - t0)} ms, diagnosed in ${Math.round(performance.now() - ts)} ms`;
    const f = frame($('lCv'), [0, 1.15], [1.9, 3.5], { yt: [2.0, 2.5, 3.0, 3.5], xt: [0, 0.2, 0.4, 0.6, 0.8, 1.0], xf: (v) => v.toFixed(1), xl: 'charge out (Ah) →', yl: 'volts', B: 24 });
    const path = (oo, col, wd) => { f.ctx.strokeStyle = col; f.ctx.lineWidth = wd; f.ctx.beginPath(); oo.V.forEach((v, i) => { i ? f.ctx.lineTo(f.X(oo.Q[i]), f.Y(v)) : f.ctx.moveTo(f.X(oo.Q[i]), f.Y(v)); }); f.ctx.stroke(); };
    path(NEW.o, C.faint, 1.5);
    if (part) { path(o, C.line2, 1.3); const oc = { V: o.V.filter((v, i) => o.Q[i] <= lab.st), Q: o.Q.filter((v) => v <= lab.st) }; path(oc, C.copper, 2.4);
      const vs = oc.V[oc.V.length - 1]; f.ctx.fillStyle = C.copper; f.ctx.beginPath(); f.ctx.arc(f.X(lab.st), f.Y(vs), 4, 0, 7); f.ctx.fill(); txt(f.ctx, `you stopped at ${fmt(lab.st, 2)} Ah`, f.X(lab.st) + 8, f.Y(vs) - 8, 'left', C.copper, 10.5, 500); }
    else path(o, C.copper, 2.4);
    const hy = f.Y(2.0); if (!part) { f.ctx.fillStyle = C.copper; f.ctx.beginPath(); f.ctx.arc(f.X(o.Q_cc), hy, 3.5, 0, 7); f.ctx.fill(); txt(f.ctx, `4C stops at ${fmt(o.Q_cc, 3)} Ah`, f.X(o.Q_cc) - 6, hy - 10, 'right', C.copper, 10.5, 500); }
    f.ctx.fillStyle = C.ink; f.ctx.beginPath(); f.ctx.arc(f.X(o.Q_total), hy, 3.5, 0, 7); f.ctx.fill(); txt(f.ctx, `hold ends ${fmt(o.Q_total, 3)}`, f.X(o.Q_total), hy + 16, 'right', C.ink, 10.5, 500);
    const pq = r.q_total, ps = r.q_total_sig; f.ctx.fillStyle = C.net; f.ctx.globalAlpha = 0.22; f.ctx.fillRect(f.X(pq - ps), f.Y(2.25), Math.max(2, f.X(pq + ps) - f.X(pq - ps)), f.Y(2.0) - f.Y(2.25)); f.ctx.globalAlpha = 1;
    f.ctx.strokeStyle = C.net; f.ctx.lineWidth = 2; f.ctx.beginPath(); f.ctx.moveTo(f.X(pq), f.Y(2.28)); f.ctx.lineTo(f.X(pq), f.Y(2.0)); f.ctx.stroke(); txt(f.ctx, 'CellSense', f.X(pq), f.Y(2.3) - 3, 'center', C.net, 10.5, 500);
    const tr = { q_total: o.Q_total, q_cc: o.Q_cc, ir: ir * 1000, fade: 1 - o.Q_total / NEW.o.Q_total, lli: 1 - P.Q_Li / P0.Q_Li, lam_ne: lab.gr / 100, lam_pe: lab.fp / 100, hold: o.Q_total - o.Q_cc };
    const cell = (name, pred, truth, good) => `<div><span>${name}</span><b style="color:${good ? 'var(--ok)' : 'var(--ink)'}">${pred}</b><em>you: ${truth}</em></div>`;
    $('lDiag').innerHTML =
      cell('Total capacity', `${fmt(r.q_total, 3)} Ah`, `${fmt(tr.q_total, 3)} Ah`, Math.abs(r.q_total - tr.q_total) < 0.015) +
      (part ? cell('Not yet read', `${mAh(r.q_total - lab.st)} mAh`, `${mAh(tr.q_total - lab.st)} mAh`, Math.abs(r.q_total - tr.q_total) < 0.015)
            : cell('Hidden in the hold', `${mAh(r.q_total - lastQ(q))} mAh`, `${mAh(tr.hold)} mAh`, Math.abs((r.q_total - lastQ(q)) - tr.hold) < 0.015)) +
      cell('Capacity fade', pct(r.fade, 1), pct(tr.fade, 1), Math.abs(r.fade - tr.fade) < 0.015) +
      cell('Pulse resistance', `${fmt(r.ir, 1)} mΩ`, `${fmt(tr.ir, 1)} mΩ`, Math.abs(r.ir - tr.ir) < 1) +
      cell('Lithium lost', pct(Math.max(0, r.lli), 1), pct(tr.lli, 1), Math.abs(r.lli - tr.lli) < 0.03) +
      cell('Graphite lost', pct(Math.max(0, r.lam_ne), 1), pct(tr.lam_ne, 0), Math.abs(r.lam_ne - tr.lam_ne) < 0.04) +
      cell('LFP lost', pct(Math.max(0, r.lam_pe), 1), pct(tr.lam_pe, 0), Math.abs(r.lam_pe - tr.lam_pe) < 0.04) +
      cell('4C capacity', `${fmt(r.q_cc, 3)} Ah`, `${fmt(tr.q_cc, 3)} Ah`, Math.abs(r.q_cc - tr.q_cc) < 0.01);
    $('lNote').textContent = (part ? `A partial discharge: the curve stops at ${fmt(lab.st, 2)} Ah, and a third network, trained on curves cut short, reads what is there. Below about 0.6 Ah it has little to go on and its error bars widen. ` : '') + 'Green: within 15 mAh, 1.5 points of fade, 1 mΩ, or 3 to 4 points on a degradation mode. Lithium lost counts the lithium that left with lithiated graphite too, so it can exceed the slider. Lost LFP usually changes nothing you can see: in this cell the graphite and the lithium run out first, so the network cannot tell, and says so with a wide error bar.';
    labBusy = false; if (labAgain) { labAgain = false; runLab(); }
  }, 0));
}
(() => {
  const bind = (id, key, show) => { const e = $(id); e.value = lab[key]; const upd = () => { lab[key] = +e.value; $(id + 'V').textContent = show(lab[key]); }; e.addEventListener('input', () => { upd(); runLab(); }); upd(); };
  bind('lSt', 'st', (v) => (v >= 1.1 ? 'full' : fmt(v, 2) + ' Ah')); bind('lLi', 'li', (v) => v + '%'); bind('lGr', 'gr', (v) => v + '%'); bind('lFp', 'fp', (v) => v + '%'); bind('lSl', 'sl', (v) => v + '%'); bind('lTr', 'tr', (v) => '×' + fmt(1 + v / 2, 1)); bind('lOh', 'oh', (v) => '×' + fmt(1 + v / 10, 1));
  $('lReset').addEventListener('click', () => { Object.assign(lab, { li: 0, gr: 0, fp: 0, sl: 0, tr: 0, oh: 0 }); ['lLi', 'lGr', 'lFp', 'lSl', 'lTr', 'lOh'].forEach((id) => { $(id).value = 0; $(id).dispatchEvent(new Event('input')); }); $('lSt').value = 1.1; $('lSt').dispatchEvent(new Event('input')); });
})();

// ------------------------------------------------------------ lifetime
function renderLife() {
  const L = DATA.life, pts = L.c100, lo = 100, hi = 3000, lg = Math.log10;
  const f = frame($('lifeCv'), [lg(lo), lg(hi)], [lg(lo), lg(hi)], { xt: [100, 300, 1000, 3000].map(lg), yt: [100, 300, 1000, 3000].map(lg), xf: (v) => String(Math.round(10 ** v)), yf: (v) => String(Math.round(10 ** v)), xl: 'actual cycle life →', yl: 'forecast at cycle 100', L: 44 });
  const { ctx, X, Y } = f; ctx.strokeStyle = C.line2; ctx.beginPath(); ctx.moveTo(X(lg(lo)), Y(lg(lo))); ctx.lineTo(X(lg(hi)), Y(lg(hi))); ctx.stroke();
  [0.8, 1.25].forEach((k) => { ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(X(lg(lo)), Y(lg(lo * k))); ctx.lineTo(X(lg(hi)), Y(lg(hi * k))); ctx.stroke(); ctx.setLineDash([]); });
  const col = { primary: C.copper, secondary: C.net, train: C.faint };
  pts.forEach((p) => { const x = X(lg(p.life)); ctx.strokeStyle = col[p.split]; ctx.globalAlpha = 0.35; ctx.beginPath(); ctx.moveTo(x, Y(lg(Math.max(lo, p.lo)))); ctx.lineTo(x, Y(lg(Math.min(hi, p.hi)))); ctx.stroke(); ctx.globalAlpha = 1;
    ctx.fillStyle = col[p.split]; ctx.beginPath(); ctx.arc(x, Y(lg(Math.min(hi, Math.max(lo, p.pred)))), 3.2, 0, 7); ctx.fill(); });
  txt(ctx, '±20%', X(lg(1800)), Y(lg(1800 * 1.25)) - 4, 'center', C.faint, 9.5);
  const S = L.ours, T = L.severson;
  $('lifeSub').textContent = L.sub; if (L.h) $('lifeH').textContent = L.h;
  $('lifeSc').textContent = `${pts.length} cells read at cycle 100. Dashed lines: ±20%. Vertical bars: CellSense's own error bar. ` + L.sc;
  $('lifeNote').textContent = L.note;
  const row = (name, d, hl) => `<tr${hl ? ' class="hl"' : ''}><td>${name}</td><td class="num">${d.primary_rmse != null ? Math.round(d.primary_rmse) : '—'}</td><td class="num">${d.primary_mpe != null ? fmt(d.primary_mpe, 1) + '%' : '—'}</td><td class="num">${d.secondary_rmse != null ? Math.round(d.secondary_rmse) : '—'}</td><td class="num">${d.secondary_mpe != null ? fmt(d.secondary_mpe, 1) + '%' : '—'}</td></tr>`;
  $('lifeTable').innerHTML = `<tr><th>Model</th><th class="num">Primary RMSE</th><th class="num">error</th><th class="num">Secondary RMSE</th><th class="num">error</th></tr>` +
    row('Severson: variance (1 feature)', T.variance) + row('Severson: discharge (6 features)', T.discharge) + row('Severson: full (9 features)', T.full) + row('CellSense', S, true) + row('Guess the test cells\' median life', L.median);
  $('stageTable').innerHTML = `<tr><th>Read at</th><th class="num">Readings</th><th class="num">Life error</th><th class="num">Cycles-left error</th></tr>` +
    L.stage.map((s) => `<tr><td>${s.label}</td><td class="num">${s.n}</td><td class="num">${fmt(s.mpe, 1)}%</td><td class="num">${Math.round(s.left_mae)}</td></tr>`).join('');
}

function renderPartial() {
  const P = DATA.partial; if (!P) return; $('partSub').textContent = P.sub; $('partNote').textContent = P.note;
  $('partTable').innerHTML = `<tr><th>Stopped after</th><th class="num">Readings cut short</th><th class="num">CellSense</th><th class="num">reference cut too</th><th class="num">No model: shift at the cut</th><th class="num">No model: no fade</th></tr>` +
    P.rows.map((r) => `<tr${r.hl ? ' class="hl"' : ''}><td>${r.label}</td><td class="num">${Math.round(100 * r.frac_cut)}%</td><td class="num">${fmt(r.net, 0)} mAh</td><td class="num">${fmt(r.net_cut, 0)} mAh</td><td class="num">${fmt(r.base, 0)} mAh</td><td class="num">${fmt(P.nofade, 0)} mAh</td></tr>`).join('') +
    `<tr><td>Full discharge (health network)</td><td class="num">0%</td><td class="num">${fmt(P.full, 0)} mAh</td><td class="num">—</td><td class="num">${fmt(P.full_base, 0)} mAh</td><td class="num">${fmt(P.nofade, 0)} mAh</td></tr>`;
}
function renderEarly() {   // the six calibration cells: real change by cycle 100 against the slow-graphite fit
  const E = DATA.life.early; if (!E) return;
  const f = frame($('earlyCv'), [2.0, 3.5], [-72, 6], { xt: [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5], yt: [-60, -40, -20, 0], xf: (v) => v.toFixed(2), xl: 'volts →', yl: 'mAh, cycle 100 minus cycle 10', L: 40 });
  const cols = [C.copper, C.net, C.lfp, C.gr, C.li, C.ink], ord = E.map((e, i) => i).sort((a, b) => E[a].life - E[b].life);
  ord.forEach((j, n) => { const e = E[j], col = cols[n % cols.length];
    [[e.real, 2.2, []], [e.sim, 1.4, [4, 3]]].forEach(([d, wd, dash]) => { f.ctx.strokeStyle = col; f.ctx.lineWidth = wd; f.ctx.setLineDash(dash); f.ctx.beginPath();
      d.forEach((v, i) => { const x = f.X(VG[i]), y = f.Y(v * 1000); i ? f.ctx.lineTo(x, y) : f.ctx.moveTo(x, y); }); f.ctx.stroke(); }); f.ctx.setLineDash([]); });
  $('earlyLeg').innerHTML = ord.map((j, n) => `<span><i style="background:${cols[n % cols.length]}"></i>${E[j].label}</span>`).join('') + '<span style="color:var(--faint)">solid: real · dashed: simulator</span>';
  $('earlySub').textContent = DATA.life.earlySub;
}

// ------------------------------------------------------------ ratchet
function renderRatchet() {
  const R = DATA.rungs, rel = DATA.release.ckpt;
  $('ratchetSub').textContent = DATA.ratchetSub;
  $('rungs').innerHTML = R.map((r, i) => { const t = r.note.split(':'), d = t.slice(1).join(':').trim(), isRel = r.ckpt === rel && r.life_ckpt === DATA.release.life_ckpt && (r.partial_ckpt || null) === (DATA.release.partial_ckpt || null);
    return `<div class="rung ${r.accepted ? 'acc' : 'rej'} ${isRel ? 'rel' : ''}"><div class="tag"><span>Rung ${i + 1}</span><b>${isRel ? 'Kept · released' : r.accepted ? 'Kept' : 'Rejected'}</b></div>
      <h3>${t[0]}</h3><p>${d.charAt(0).toUpperCase() + d.slice(1)}</p>${r.accepted ? '' : `<p style="color:var(--faint)">${r.why}</p>`}
      <div class="m">${r.metrics.map(([k, v, full]) => `<div${full ? ' style="grid-column:1/-1"' : ''}><span>${k}</span><b>${v}</b></div>`).join('')}</div></div>`; }).join('');
}

// ------------------------------------------------------------ static text
function renderStatic() {
  const R = DATA.real.with_ref, L = DATA.life.ours;
  $('nCells').textContent = R.cells;
  $('stats').innerHTML = `
    <div class="stat"><b>${fmt(R.q_total_mae_mAh, 0)} mAh</b><span>typical error in total capacity over ${R.n.toLocaleString()} real readings, against ${fmt(R.q_total_baseline_mae_mAh, 0)} mAh reading the 4C curve's end</span></div>
    <div class="stat"><b>${fmt(R.fade_corr, 2)}</b><span>correlation with the real capacity fade since each cell was new</span></div>
    <div class="stat"><b>${pct(R.eol_called_right, 0)}</b><span>of readings correctly called before or past end of life (0.88 Ah), from the 4C curve alone</span></div>
    <div class="stat"><b>0</b><span>real cells in the networks' training data: ${(DATA.release.samples / 1000).toFixed(0)}k simulated ones. Six real cells tuned the physics.</span></div>`;
  const ph = DATA.physics;
  $('physChecks').innerHTML = `checked against PyBaMM:<br>4C capacity ±${fmt(ph.spm_cap, 2)}% · voltage ${fmt(ph.spm_med, 1)} mV median<br>electrolyte (SPMe) within ${fmt(ph.spme_max, 1)} mV · 2.0 V hold ±${fmt(ph.cv_cap, 1)}%`;
  $('calFacts').innerHTML = `six new cells fitted:<br>voltage ${fmt(DATA.cal.v_lo, 0)}–${fmt(DATA.cal.v_hi, 0)} mV rms · can temperature ±${fmt(DATA.cal.t, 1)} °C<br>pulse resistance ±${fmt(DATA.cal.ir, 1)} mΩ`;
  $('netFacts').innerHTML = `8 channels × 250 voltages (+2 numbers)<br>${DATA.release.params.toLocaleString()} weights · 5 residual blocks<br>${DATA.release.labels} outputs, each with an error bar`;
  $('verTag').textContent = `v1.1 · rung ${DATA.rungs.findIndex((r) => r.ckpt === DATA.release.ckpt && r.life_ckpt === DATA.release.life_ckpt && (r.partial_ckpt || null) === (DATA.release.partial_ckpt || null)) + 1}`;
  $('codeBlock').innerHTML = `<span class="c"># pip install numpy</span>
<span class="k">import</span> cellsense, numpy <span class="k">as</span> np
cs = cellsense.CellSense()

<span class="c"># a real cell from the exam: curves on the 250-voltage grid</span>
d = np.load(<span class="s">'examples/${DATA.example.file}'</span>)
r = cs.diagnose(d[<span class="s">'Q'</span>][-1], d[<span class="s">'T'</span>][-1], d[<span class="s">'Q'</span>][0], d[<span class="s">'T'</span>][0],
                cycle=<span class="k">int</span>(d[<span class="s">'cycle'</span>][-1]), charge_time_min=<span class="k">float</span>(d[<span class="s">'charge_time'</span>]))
r[<span class="s">'q_total'</span>], r[<span class="s">'q_total_sig'</span>]     <span class="c"># Ah left, with the 2.0 V hold</span>
r[<span class="s">'lli'</span>], r[<span class="s">'lam_ne'</span>], r[<span class="s">'lam_pe'</span>]  <span class="c"># what is wearing out</span>
r[<span class="s">'life'</span>], r[<span class="s">'life_lo'</span>], r[<span class="s">'life_hi'</span>]  <span class="c"># forecast end of life (cycles)</span>

<span class="c"># your own discharge, recorded in time</span>
Q, T = cellsense.q_of_v(volts, amp_hours, can_temp)`;
  $('files').innerHTML = `
    <div class="file"><code>cellsense.py</code><span>the network in numpy, the input features and a converter from raw cycler data.</span></div>
    <div class="file"><code>cellsense_weights.npz</code><span>the health network.</span></div>
    <div class="file"><code>cellsense_life_weights.npz</code><span>the lifetime network (${DATA.release.params.toLocaleString()} weights for the two).</span></div>
    <div class="file"><code>examples/</code><span>one real exam cell's discharges through life, with the cycler's measurements.</span></div>
    <div class="file"><code>MODEL_CARD.md</code><span>the physics and its PyBaMM checks, the calibration, every rung, both exams and the limits.</span></div>
    <div class="file"><code>training/</code><span>simulator, calibration, data, training, ratchet and exam scripts (PyTorch, PyBaMM).</span></div>
    <p class="note" style="margin:4px 0 0">The files are attached in the chat alongside this page.</p>`;
  $('limits').innerHTML = DATA.limits.map((x) => `<li>${x}</li>`).join('');
}

function renderAll() { colors(); renderHero(); renderHidden(); renderCell(); renderLife(); renderEarly(); renderPartial(); renderRatchet(); renderStatic(); runLab(); }
renderAll();
let rz = null; addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => { renderHero(); renderHidden(); renderCell(); renderLife(); renderEarly(); runLab(); }, 150); });
try { matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => renderAll()); } catch (_) { }
})();
