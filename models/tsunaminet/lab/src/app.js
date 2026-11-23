(function () {
  const $ = id => document.getElementById(id);
  const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const NS = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  const fmt = (v, d = 2) => (v == null || isNaN(v)) ? '–' : Number(v).toFixed(d);
  function b64(s, T) { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new T(u.buffer); }
  const net = new TNet.Net(DATA.weights);
  const MU_RIG = 4e10, AREA = 100e3 * 50e3, DELAY = DATA.delay_frac || 0;

  // ---------------- hero map
  const H = DATA.hero, arr = b64(H.arr, Uint16Array), amp = b64(H.amp, Uint8Array);
  const cv = $('heroMap'), ctx = cv.getContext('2d');
  const W = cv.width, Hh = cv.height, sx = W / H.nx, sy = Hh / H.ny;
  let tmin = 0, playing = !window.matchMedia('(prefers-reduced-motion: reduce)').matches, last = 0;
  function rgb(hex) { const h = hex.replace('#', ''); return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]; }
  function drawHero() {
    const land = rgb(css('--land')), sea = rgb(css('--paper')), lo = rgb(css('--sea2')), hi = rgb(css('--buoy')), front = rgb(css('--ink'));
    const img = ctx.createImageData(H.nx, H.ny);
    for (let j = 0; j < H.ny; j++) for (let i = 0; i < H.nx; i++) {
      const k = j * H.nx + i, p = ((H.ny - 1 - j) * H.nx + i) * 4, a = arr[k];
      let c;
      if (a === 65535) c = land;
      else if (a === 65534 || a > tmin) c = sea;
      else { const f = amp[k] / 255; c = [0, 1, 2].map(q => Math.round(lo[q] + (hi[q] - lo[q]) * f)); if (tmin - a < 12) c = c.map((v, q) => Math.round(v * 0.4 + front[q] * 0.6)); }
      img.data[p] = c[0]; img.data[p + 1] = c[1]; img.data[p + 2] = c[2]; img.data[p + 3] = 255;
    }
    const off = document.createElement('canvas'); off.width = H.nx; off.height = H.ny; off.getContext('2d').putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false; ctx.drawImage(off, 0, 0, W, Hh);
    ctx.fillStyle = css('--buoy'); ctx.strokeStyle = css('--ink'); ctx.lineWidth = 1.2;
    for (const [lo_, la] of DATA.darts) { const x = (lo_ - H.lon0) / H.d * sx, y = (H.ny - (la - H.lat0) / H.d) * sy; ctx.beginPath(); ctx.arc(x, y, 4.2, 0, 7); ctx.fill(); ctx.stroke(); }
    const [elon, elat] = H.epi; const ex = (elon - H.lon0) / H.d * sx, ey = (H.ny - (elat - H.lat0) / H.d) * sy;
    ctx.strokeStyle = css('--bad'); ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(ex - 8, ey - 8); ctx.lineTo(ex + 8, ey + 8); ctx.moveTo(ex + 8, ey - 8); ctx.lineTo(ex - 8, ey + 8); ctx.stroke();
    $('heroClock').textContent = `${Math.floor(tmin / 60)} h ${String(Math.floor(tmin % 60)).padStart(2, '0')} min after the earthquake`;
  }
  function tick(t) { if (playing && t - last > 60) { tmin = (tmin + 8) % 1460; last = t; drawHero(); } requestAnimationFrame(tick); }
  tmin = playing ? 0 : 1440; drawHero(); requestAnimationFrame(tick);
  $('heroPlay').onclick = () => { playing = !playing; $('heroPlay').textContent = playing ? 'Pause' : 'Play'; $('heroPlay').setAttribute('aria-pressed', playing); };
  if (!playing) { $('heroPlay').textContent = 'Play'; $('heroPlay').setAttribute('aria-pressed', 'false'); }
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { drawHero(); render(); });
  new MutationObserver(() => { drawHero(); render(); }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  $('stats').innerHTML = DATA.stats.map(s => `<div class="stat"><b>${s.v}</b><span>${s.l}</span></div>`).join('');
  $('heroNote').textContent = DATA.heroNote;

  // ---------------- physics helpers on the 9-column frame
  function thetaSlip(th, colsS) {
    const mw = th[0], sc = th[1] * 100, fd = Math.min(1, Math.max(0, th[2])), L = Math.pow(10, th[3]);
    const w = new Array(colsS.length).fill(0);
    for (let i = 0; i < 48; i++) { const x = sc - L / 2 + L * i / 47, v = Math.pow(Math.max(0, Math.sin(Math.PI * i / 47)), 0.7) + 1e-3;
      colsS.forEach((s, c) => { if (s !== null && x >= s * 100 - 50 && x < s * 100 + 50) w[c] += v; }); }
    let tot = w.reduce((a, b) => a + b, 0);
    if (tot <= 0) return null;
    const slip0 = Math.pow(10, 1.5 * mw + 9.1) / (MU_RIG * AREA), out = new Array(18).fill(0);
    w.forEach((v, c) => { out[c] = slip0 * v / tot * (1 - fd); out[9 + c] = slip0 * v / tot * fd; });
    return out;
  }
  function synth(xin, slip, shift) { // predicted obs channel (features units) from basis channels
    const y = new Float32Array(90);
    for (let u = 0; u < 18; u++) { if (!slip[u]) continue; const b = (2 + u) * 90; for (let t = 0; t < 90; t++) y[t] += slip[u] * xin[b + t] / 10; }
    if (!shift) return y;
    const z = new Float32Array(90); for (let t = 0; t < 90; t++) { const s = t - shift, i = Math.floor(s), f = s - i; const a = i >= 0 && i < 90 ? y[i] : 0, b = i + 1 >= 0 && i + 1 < 90 ? y[i + 1] : 0; z[t] = a * (1 - f) + b * f; } return z;
  }
  const quant = (a, q) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.max(0, Math.floor(q * (s.length - 1))))]; };

  // ---------------- explorer
  const EV = DATA.events; let cur = 0, postCache = {};
  $('evlist').innerHTML = EV.map((e, i) => `<button type="button" data-i="${i}" aria-pressed="${i === 0}">${e.short}<span class="k">M${fmt(e.mw_true, 1)}</span></button>`).join('');
  $('evlist').onclick = ev => { const b = ev.target.closest('button'); if (!b) return; cur = +b.dataset.i; [...$('evlist').children].forEach(c => c.setAttribute('aria-pressed', c === b)); labReset(); render(); };
  function inputsOf(e) { return e.inputs.map(s => ({ ...s, X: s.X || (s.X = b64(s.x, Float32Array)) })); }
  function posteriorOf(e) {
    if (postCache[e.id]) return postCache[e.id];
    const I = inputsOf(e), c = net.context(I.map(s => s.X), I.map(s => s.s));
    return (postCache[e.id] = net.sample(c, 400, TNet.gauss(e.id)));
  }
  function traceSVG(host, items) {
    host.innerHTML = '';
    items.forEach(it => {
      const svg = el('svg', { viewBox: '0 0 520 120', role: 'img', 'aria-label': 'DART ' + it.site + ' record and posterior band' }, host);
      const all = [...it.obs.filter((v, t) => it.mask[t]), ...it.hi, ...it.lo]; const m = Math.max(1, ...all.map(Math.abs)) * 1.1;
      const X = t => 40 + t / 89 * 470, Y = v => 60 - v / m * 50;
      el('line', { x1: 40, x2: 510, y1: 60, y2: 60, stroke: css('--rule') }, svg);
      el('line', { x1: X(20), x2: X(20), y1: 8, y2: 112, stroke: css('--rule'), 'stroke-dasharray': '3 3' }, svg);
      let d = ''; for (let t = 0; t < 90; t++) d += (t ? 'L' : 'M') + X(t) + ' ' + Y(it.hi[t]); for (let t = 89; t >= 0; t--) d += 'L' + X(t) + ' ' + Y(it.lo[t]);
      el('path', { d: d + 'Z', fill: css('--sea'), 'fill-opacity': 0.28, stroke: 'none' }, svg);
      let dm = ''; for (let t = 0; t < 90; t++) dm += (t ? 'L' : 'M') + X(t) + ' ' + Y(it.mid[t]);
      el('path', { d: dm, fill: 'none', stroke: css('--sea'), 'stroke-width': 1.6 }, svg);
      for (let t = 0; t < 90; t++) if (it.mask[t]) el('circle', { cx: X(t), cy: Y(it.obs[t]), r: 1.7, fill: css('--ink'), 'fill-opacity': 0.7 }, svg);
      const lab = el('text', { x: 4, y: 14, fill: css('--ink'), 'font-size': 11, 'font-family': css('--f-mono') }, svg); lab.textContent = it.site.split('_')[0];
      const l2 = el('text', { x: 4, y: 28, fill: css('--muted'), 'font-size': 10, 'font-family': css('--f-mono') }, svg); l2.textContent = '±' + fmt(m, m < 10 ? 1 : 0) + ' cm';
      const l3 = el('text', { x: X(20) + 4, y: 116, fill: css('--muted'), 'font-size': 10, 'font-family': css('--f-mono') }, svg); l3.textContent = 'predicted arrival ' + it.ta + ' min';
    });
  }
  function bands(I, P, shift) {
    return I.map(s => {
      const cm = s.scale / 1000 * 10 * 100; // features units -> cm
      const preds = P.slice(0, 80).map(th => { const sl = thetaSlip(th, EV[cur].cols_s); return sl ? synth(s.X, sl, s.ta * DELAY + (shift || 0)) : new Float32Array(90); });
      const lo = [], hi = [], mid = [];
      for (let t = 0; t < 90; t++) { const v = preds.map(p => p[t] * cm); lo.push(quant(v, 0.16)); hi.push(quant(v, 0.84)); mid.push(quant(v, 0.5)); }
      const obs = [], mask = []; for (let t = 0; t < 90; t++) { obs.push(s.X[t] * cm); mask.push(s.X[90 + t] > 0); }
      return { site: s.site, ta: s.ta, lo, hi, mid, obs, mask };
    });
  }
  function postSVG(svg, P, marks) {
    svg.innerHTML = '';
    const xs = P.map(p => p[1] * 100), ys = P.map(p => p[0]);
    const mk = marks.filter(m => m.x != null && m.y != null);
    let x0 = Math.min(-150, ...xs.map(v => quant(xs, 0.01)), ...mk.map(m => m.x)) - 30, x1 = Math.max(150, quant(xs, 0.99), ...mk.map(m => m.x)) + 30;
    let y0 = Math.min(quant(ys, 0.01), ...mk.map(m => m.y)) - 0.15, y1 = Math.max(quant(ys, 0.99), ...mk.map(m => m.y)) + 0.15;
    const X = v => 48 + (v - x0) / (x1 - x0) * 300, Y = v => 262 - (v - y0) / (y1 - y0) * 248;
    const ax = (t, x, y, a) => { const e = el('text', { x, y, fill: css('--muted'), 'font-size': 10, 'font-family': css('--f-mono'), 'text-anchor': a || 'middle' }, svg); e.textContent = t; };
    el('rect', { x: 48, y: 14, width: 300, height: 248, fill: 'none', stroke: css('--rule') }, svg);
    for (let k = Math.ceil(x0 / 100) * 100; k <= x1; k += 100) { el('line', { x1: X(k), x2: X(k), y1: 14, y2: 262, stroke: css('--rule'), 'stroke-width': 0.5 }, svg); ax(k, X(k), 276); }
    for (let k = Math.ceil(y0 * 5) / 5; k <= y1; k += 0.2) { el('line', { x1: 48, x2: 348, y1: Y(k), y2: Y(k), stroke: css('--rule'), 'stroke-width': 0.5 }, svg); ax(k.toFixed(1), 44, Y(k) + 3, 'end'); }
    ax('centroid along trench from epicentre column (km)', 198, 294); ax('Mw', 14, 18, 'start');
    P.forEach(p => el('circle', { cx: X(p[1] * 100), cy: Y(p[0]), r: 1.6, fill: css('--sea'), 'fill-opacity': 0.35 }, svg));
    mk.forEach(m => {
      if (m.kind === 'truth') { el('line', { x1: X(m.x) - 9, x2: X(m.x) + 9, y1: Y(m.y), y2: Y(m.y), stroke: css('--ink'), 'stroke-width': 2 }, svg); el('line', { x1: X(m.x), x2: X(m.x), y1: Y(m.y) - 9, y2: Y(m.y) + 9, stroke: css('--ink'), 'stroke-width': 2 }, svg); }
      else el('rect', { x: X(m.x) - 4, y: Y(m.y) - 4, width: 8, height: 8, fill: m.color, stroke: css('--paper') }, svg);
    });
    let lx = 56; [...(mk.some(m => m.kind === 'truth') ? [['USGS (truth)', css('--ink')]] : []), ...mk.filter(m => m.kind !== 'truth').map(m => [m.label, m.color])].forEach(([t, c]) => { el('rect', { x: lx, y: 20, width: 8, height: 8, fill: c }, svg); const e = el('text', { x: lx + 11, y: 28, fill: css('--ink'), 'font-size': 10, 'font-family': css('--f-body') }, svg); e.textContent = t; lx += 11 + t.length * 5.4 + 10; });
  }
  function fcastSVG(e) {
    const svg = $('fcast'); svg.innerHTML = ''; const T = e.targets;
    const h = Math.max(70, 34 + T.length * 22); svg.setAttribute('viewBox', `0 0 720 ${h}`);
    if (!T.length) { const t = el('text', { x: 10, y: 30, fill: css('--muted'), 'font-size': 13 }, svg); t.textContent = 'No held-out DART passed the target rule for this event.'; return; }
    const lg = v => Math.log10(Math.max(v, 0.05)); const x0 = -1.3, x1 = 2.5; const X = v => 150 + (lg(v) - x0) / (x1 - x0) * 555;
    [0.1, 1, 10, 100].forEach(v => { el('line', { x1: X(v), x2: X(v), y1: 12, y2: h - 14, stroke: css('--rule'), 'stroke-width': 0.6 }, svg); const t = el('text', { x: X(v), y: h - 2, fill: css('--muted'), 'font-size': 10, 'text-anchor': 'middle', 'font-family': css('--f-mono') }, svg); t.textContent = v + ' cm'; });
    T.forEach((t, i) => {
      const y = 26 + i * 22;
      const lab = el('text', { x: 4, y: y + 4, fill: css('--ink'), 'font-size': 11, 'font-family': css('--f-mono') }, svg); lab.textContent = t.site.split('_')[0] + '  ' + Math.round(t.ta / 6) / 10 + ' h';
      el('line', { x1: X(t.lo * 100), x2: X(t.hi * 100), y1: y, y2: y, stroke: css('--sea'), 'stroke-width': 5, 'stroke-opacity': 0.45 }, svg);
      el('circle', { cx: X(t.net * 100), cy: y, r: 4.5, fill: css('--sea') }, svg);
      el('rect', { x: X(t.nnls * 100) - 3.5, y: y - 3.5, width: 7, height: 7, fill: 'none', stroke: css('--buoy'), 'stroke-width': 1.6 }, svg);
      el('path', { d: `M${X(t.seismic * 100)} ${y - 5} l4 8 h-8z`, fill: 'none', stroke: css('--muted'), 'stroke-width': 1.4 }, svg);
      el('circle', { cx: X(t.ffm * 100), cy: y, r: 3, fill: 'none', stroke: css('--twin'), 'stroke-width': 1.4 }, svg);
      el('line', { x1: X(t.obs * 100), x2: X(t.obs * 100), y1: y - 8, y2: y + 8, stroke: css('--ink'), 'stroke-width': 2.4 }, svg);
    });
    let lx = 150; [['observed', 'tick'], ['TsunamiNet', 'net'], ['SIFT-style NNLS', 'nnls'], ['seismic Mw', 'seis'], ['finite-fault twin', 'ffm']].forEach(([t, k]) => {
      const y = 6;
      if (k === 'tick') el('line', { x1: lx, x2: lx, y1: y - 5, y2: y + 5, stroke: css('--ink'), 'stroke-width': 2.4 }, svg);
      if (k === 'net') el('circle', { cx: lx, cy: y, r: 4, fill: css('--sea') }, svg);
      if (k === 'nnls') el('rect', { x: lx - 3.5, y: y - 3.5, width: 7, height: 7, fill: 'none', stroke: css('--buoy'), 'stroke-width': 1.6 }, svg);
      if (k === 'seis') el('path', { d: `M${lx} ${y - 5} l4 8 h-8z`, fill: 'none', stroke: css('--muted'), 'stroke-width': 1.4 }, svg);
      if (k === 'ffm') el('circle', { cx: lx, cy: y, r: 3, fill: 'none', stroke: css('--twin'), 'stroke-width': 1.4 }, svg);
      const e2 = el('text', { x: lx + 8, y: y + 4, fill: css('--muted'), 'font-size': 10 }, svg); e2.textContent = t; lx += t.length * 5.6 + 30; });
  }
  function render() {
    const e = EV[cur], P = posteriorOf(e), I = inputsOf(e);
    traceSVG($('traces'), bands(I, P));
    $('traceNote').textContent = `Window: 20 min before to 70 min after the predicted arrival at each buoy. Epicentre from USGS; nothing else from seismology is used.`;
    const mw = P.map(p => p[0]), sc = P.map(p => p[1] * 100);
    const WF = (DATA.weights.meta.calibration || {}).widen || { 0: 1, 1: 1 };
    const wr = (a, k, d) => { const m = quant(a, .5), lo = m - WF[k] * (m - quant(a, .16)), hi = m + WF[k] * (quant(a, .84) - m); return `${fmt(m, d)} [${fmt(lo, d)}, ${fmt(hi, d)}]`; };
    postSVG($('post'), P, [{ kind: 'truth', x: e.sc_true != null ? e.sc_true * 100 : null, y: e.mw_true }, { label: 'SIFT-style', x: e.nnls.sc * 100, y: e.nnls.mw, color: css('--buoy') }]);
    $('postkv').innerHTML = `<dt>Mw, median [68 %, dev-widened]</dt><dd>${wr(mw, 0, 2)}</dd><dt>USGS W-phase Mw</dt><dd>${fmt(e.mw_true)}</dd>
      <dt>Centroid (km) [68 %]</dt><dd>${wr(sc, 1, 0)}</dd><dt>USGS finite fault</dt><dd>${e.sc_true != null ? fmt(e.sc_true * 100, 0) : '–'}</dd>
      <dt>SIFT-style NNLS Mw</dt><dd>${fmt(e.nnls.mw)}</dd><dt>Mechanism</dt><dd>${e.interface ? 'interface thrust' : 'not an interface thrust'}</dd>`;
    fcastSVG(e);
    lab();
  }

  // ---------------- lab
  let labOn = null;
  function labReset() { labOn = EV[cur].inputs.map(() => true); $('labSt').innerHTML = EV[cur].inputs.map((s, i) => `<button type="button" data-i="${i}" aria-pressed="true">${s.site.split('_')[0]}</button>`).join(''); }
  $('labSt').onclick = ev => { const b = ev.target.closest('button'); if (!b) return; const i = +b.dataset.i; if (labOn.filter(Boolean).length === 1 && labOn[i]) return; labOn[i] = !labOn[i]; b.setAttribute('aria-pressed', labOn[i]); lab(); };
  ['lMw', 'lSc', 'lFd', 'lNz', 'lSh'].forEach(id => $(id).addEventListener('input', lab));
  function lab() {
    const e = EV[cur], I = inputsOf(e), mw = +$('lMw').value, sc = +$('lSc').value, fd = +$('lFd').value, nz = +$('lNz').value, sh = +$('lSh').value;
    $('oMw').textContent = mw.toFixed(2); $('oSc').textContent = sc; $('oFd').textContent = fd.toFixed(2); $('oNz').textContent = nz.toFixed(1); $('oSh').textContent = sh;
    const th = [mw, sc / 100, fd, -2.477 + 0.585 * mw], sl = thetaSlip(th, e.cols_s); if (!sl) return;
    const rng = TNet.gauss(7 + cur), Xs = [], Ss = [], used = [];
    I.forEach((s, k) => {
      if (!labOn[k]) return;
      const cm = s.scale / 1000 * 10 * 100, y = synth(s.X, sl, s.ta * DELAY + sh), x = new Float32Array(s.X);
      let mx = 0; for (let t = 0; t < 90; t++) { const v = (y[t] + nz / cm * rng() + 0.2 / cm * rng()) * x[90 + t]; x[t] = v; mx = Math.max(mx, Math.abs(v)); }
      Xs.push(x); Ss.push([s.s[0], s.s[1], Math.log10(mx * 10 + 1e-3) / 2]); used.push({ ...s, X: x });
    });
    const P = net.sample(net.context(Xs, Ss), 300, TNet.gauss(11));
    traceSVG($('labTr'), bands(used, P));
    postSVG($('labPost'), P, [{ kind: 'truth', x: null, y: null }, { label: 'your source', x: sc, y: mw, color: css('--bad') }]);
    const mws = P.map(p => p[0]), scs = P.map(p => p[1] * 100);
    $('labkv').innerHTML = `<dt>Network Mw</dt><dd>${fmt(quant(mws, .5))} [${fmt(quant(mws, .16))}, ${fmt(quant(mws, .84))}]</dd><dt>Network centroid</dt><dd>${fmt(quant(scs, .5), 0)} km [${fmt(quant(scs, .16), 0)}, ${fmt(quant(scs, .84), 0)}]</dd>`;
  }

  // ---------------- static sections
  const tbl = (id, head, rows) => { $(id).innerHTML = '<thead><tr>' + head.map(h => `<th>${h}</th>`).join('') + '</tr></thead><tbody>' + rows.map(r => '<tr>' + r.map(c => `<td${c && c.cls ? ` class="${c.cls}"` : ''}>${c && c.v !== undefined ? c.v : c}</td>`).join('') + '</tr>').join('') + '</tbody>'; };
  $('examH').textContent = DATA.exam.title; $('examText').innerHTML = DATA.exam.text;
  tbl('examTable', DATA.exam.head, DATA.exam.rows); tbl('eventTable', DATA.exam.ehead, DATA.exam.erows);
  $('ratchetText').innerHTML = DATA.ratchet.text; tbl('ratchetTable', DATA.ratchet.head, DATA.ratchet.rows); $('twinText').innerHTML = DATA.ratchet.twin;
  $('flowSteps').innerHTML = DATA.flow.map(f => `<div><b>${f[0]}</b>${f[1]}</div>`).join('');
  $('checks').innerHTML = DATA.checks.map(c => `<li>${c}</li>`).join(''); $('limits').innerHTML = DATA.limits.map(c => `<li>${c}</li>`).join('');
  $('useText').innerHTML = DATA.use.text; $('useCode').textContent = DATA.use.code;
  $('credits').innerHTML = DATA.credits.map(c => `<li>${c}</li>`).join(''); $('foot').innerHTML = DATA.foot;
  labReset(); render();
})();
