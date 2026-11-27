// Ratchet Observatory: a BI-style report over the whole collection. Built 27 Nov 2026 in a separate session and
// moved into the site; data from mlratchet/observatory.py (site/src/data/observatory.json).

/* ============================================================ data */
const D = window.__OBS__; // set by site/src/pages/observatory.astro (data + StormNet's instrument files)
const $ = (s, e = document) => e.querySelector(s);
const $$ = (s, e = document) => [...e.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const md = s => esc(s).replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
const fmt = d3.format(',');
const pct = v => (v * 100).toFixed(0) + '%';
const bw = el => Math.max(200, el.clientWidth - 28);
/* Phone-width and touch flags: every phone-only layout branches on these, so the desktop page is untouched. */
const MOBILE = () => matchMedia('(max-width: 760px)').matches;
const TOUCH = matchMedia('(hover: none)').matches;
let tapKey = null;
/* On touch, the first tap on a mark shows its numbers; a second tap on the same mark opens it. */
function tapOpen(ev, key, html, open) {
  if (TOUCH && tapKey !== key) { tapKey = key; showTip(html(), ev); return; }
  tapKey = null; hideTip(); open();
}
const REPO = 'https://github.com/dogum/ML-Ratchet';
const SITE = new URL(import.meta.env.BASE_URL, location.origin).href.replace(/\/$/, ''); // this site, wherever it is served
const GRADED = ['win', 'tie', 'mixed', 'loss'];
const VLABEL = { win: 'Win', loss: 'Loss', tie: 'Tie', mixed: 'Mixed', info: 'Context' };
const VC = v => `var(--${v})`;
const PLAT = { vehicle: 'Vehicle · RoadSim', grid: 'Grid · GridSim', building: 'Building · BuildingSim', water: 'Water · WaterSim', fire: 'Fire · FireSim', standalone: 'Own simulator' };
const ACLASS = { critical: 'Critical defect', defect: 'Defect or guardrail', limit: 'Evaluation limit', clean: 'Controls passed', none: 'Not audited yet' };
const AORDER = ['critical', 'defect', 'limit', 'clean', 'none'];
const ROLE = { backdrop: 'Real noise under the sims', calibration: 'Calibrates the simulator', gate: 'Gates each rung', error_bars: 'Sets error-bar width', exam: 'Grades once (exam)' };
const ROLES = Object.keys(ROLE);
const RC = { frozen: 'Frozen set: not enough gain', gate: 'Real dev gate said no', other: 'Several or other reasons', stopped: 'Stopped before scoring' };
const RCOL = { frozen: 'var(--ink-2)', gate: 'var(--brand)', other: 'var(--tie)', stopped: 'var(--info)' };
const ACT = { fixed: 'Fixed', warned: 'Warned', documented: 'Documented', queued: 'Queued', kept: 'Kept as is', other: 'Other' };
const ACOL = { fixed: 'var(--win)', warned: 'var(--mixed)', documented: 'var(--info)', queued: 'var(--sel)', kept: 'var(--tie)', other: 'var(--ink-3)' };

const MODELS = D.models;
const BYID = new Map(MODELS.map(m => [m.id, m]));
const numericSims = L => L.rungs.filter(r => typeof r.sim === 'number').length;
for (const m of MODELS) {
  m.rows.forEach(r => { r.m = m; r.key = m.id + ':' + r.i; });
  const t = { win: 0, loss: 0, tie: 0, mixed: 0, info: 0 };
  m.rows.forEach(r => t[r.verdict]++);
  m.tally = t;
  m.graded = t.win + t.loss + t.tie + t.mixed;
  m.winShare = m.graded ? t.win / m.graded : 0;
  m.allRungs = m.ladders.flatMap(L => L.rungs);
  m.nRungs = m.allRungs.length;
  m.nRej = m.allRungs.filter(r => r.decision === 'rejected').length;
  m.rejRate = m.nRungs ? m.nRej / m.nRungs : 0;
  m.roles = new Set(m.approach.real_data_roles || []);
  m.auditCls = m.audit ? m.audit.cls : 'none';
  m.main = [...m.ladders].sort((a, b) => numericSims(b) - numericSims(a) || b.rungs.length - a.rungs.length)[0];
}
const ROWS = MODELS.flatMap(m => m.rows);
const SROWS = ROWS.filter(r => r.margin !== null);
const COVS = MODELS.flatMap(m => m.exam.coverage.map((c, i) => ({ m, i, key: m.id + '#c' + i, output: c.output, actual: +c.actual })));
const DOMAINS = [...d3.rollup(MODELS, v => v.length, m => m.domain)].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(d => d[0]);
const PLATS = ['vehicle', 'grid', 'building', 'water', 'fire', 'standalone'].filter(p => MODELS.some(m => m.platform === p));
const SER = D.series.map(s => ({ ...s, d: new Date(s.t), kept: s.rungs - s.rej }));

/* ============================================================ state and filters */
const blankF = () => ({ domain: new Set(), platform: new Set(), verdict: new Set(), audit: new Set(), status: new Set(), model: new Set(), dumb: null, classical: null });
const S = {
  page: 'overview', f: blankF(), swarmGroup: 'none', wallSort: 'win', tree: { dims: ['domain', 'model'], path: [] },
  table: { sort: 'margin', dir: -1, ctx: false }, focusRow: null, prevPage: 'overview', auditSel: null,
  live: { storm: 0, k: null, bz: 0, speed: 1, pressure: 1, show: { truth: 1, net: 1, classical: 1, published: 0, single: 0, persistence: 1 } },
};
function mPass(m, skip) {
  const f = S.f;
  if (skip !== 'domain' && f.domain.size && !f.domain.has(m.domain)) return false;
  if (skip !== 'platform' && f.platform.size && !f.platform.has(m.platform)) return false;
  if (skip !== 'audit' && f.audit.size && !f.audit.has(m.auditCls)) return false;
  if (skip !== 'status' && f.status.size && !f.status.has(m.status)) return false;
  if (skip !== 'model' && f.model.size && !f.model.has(m.id)) return false;
  return true;
}
function rPass(r, skip) {
  if (!mPass(r.m, skip)) return false;
  const f = S.f;
  if (skip !== 'verdict' && f.verdict.size && !f.verdict.has(r.verdict)) return false;
  if (skip !== 'dumb' && f.dumb !== null && r.beat.dumb !== f.dumb) return false;
  if (skip !== 'classical' && f.classical !== null && r.beat.classical !== f.classical) return false;
  return true;
}
const rowFiltersOn = () => S.f.verdict.size > 0 || S.f.dumb !== null || S.f.classical !== null;
const anyFilter = () => Object.values(S.f).some(v => v instanceof Set ? v.size : v !== null);
function modelsIn() { return MODELS.filter(m => mPass(m) && (!rowFiltersOn() || m.rows.some(r => rPass(r)))); }
function stats() {
  const ms = modelsIn(), rs = ROWS.filter(r => rPass(r));
  const v = { win: 0, loss: 0, tie: 0, mixed: 0, info: 0 };
  rs.forEach(r => v[r.verdict]++);
  return { ms, rs, v, graded: v.win + v.loss + v.tie + v.mixed, rungs: d3.sum(ms, m => m.nRungs), rej: d3.sum(ms, m => m.nRej), domains: new Set(ms.map(m => m.domain)).size };
}
function toggle(key, v, add) {
  const s = S.f[key];
  if (add) s.has(v) ? s.delete(v) : s.add(v);
  else if (s.size === 1 && s.has(v)) s.clear();
  else { s.clear(); s.add(v); }
  changed();
}
function setTri(key, v) { S.f[key] = S.f[key] === v ? null : v; changed(); }
function resetFilters() { S.f = blankF(); S.tree.path = []; }

/* ============================================================ chrome: gear, tooltip, toast */
let gearTurn = 0;
(function buildGear() {
  const g = $('#gear .wheel'), n = 12, R = 15, r = 10.5, c = 17;
  let d = '';
  for (let k = 0; k < n; k++) {
    const a0 = (k / n) * 2 * Math.PI, a1 = ((k + 1) / n) * 2 * Math.PI;
    const p = (a, rad) => `${(c + rad * Math.cos(a)).toFixed(2)},${(c + rad * Math.sin(a)).toFixed(2)}`;
    d += (k ? 'L' : 'M') + p(a0, R) + 'L' + p(a1, r) + 'L' + p(a1, R);
  }
  g.innerHTML = `<path d="${d}Z" style="fill:var(--ink)"/><circle cx="17" cy="17" r="4" style="fill:var(--paper)"/>`;
})();
function turnGear() { gearTurn += 30; $('#gear .wheel').style.transform = `rotate(${gearTurn}deg)`; }

const tip = $('#tip');
function showTip(html, ev) { tip.innerHTML = html; tip.classList.add('on'); moveTip(ev); }
function moveTip(ev) {
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
  if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
  tip.style.left = Math.max(6, x) + 'px'; tip.style.top = Math.max(6, y) + 'px';
}
function hideTip() { tip.classList.remove('on'); }
let toastT;
function toast(msg) { const t = $('#toast'); t.textContent = msg; t.classList.add('on'); clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove('on'), 2200); }
async function copyText(text, what) {
  try { await navigator.clipboard.writeText(text); toast(`Copied ${what}`); }
  catch (e) { toast('The browser blocked the clipboard here. Open the page in its own tab to copy.'); }
}

/* ============================================================ shared formatting */
function fmtVal(v, unit) {
  if (typeof v !== 'number') return esc(v);
  const s = Math.abs(v) >= 10000 ? fmt(v) : String(v);
  if (!unit) return s;
  if (unit === '%') return s + '%';
  if (unit === 'deg') return s + '°';
  return s + ' ' + esc(unit);
}
function bestBaseline(r) {
  const bs = r.baselines.filter(b => typeof b.value === 'number');
  if (!bs.length || (r.better !== 'lower' && r.better !== 'higher')) return null;
  return bs.reduce((a, b) => (r.better === 'lower' ? b.value < a.value : b.value > a.value) ? b : a);
}
function ratioText(mg) {
  const x = Math.pow(10, Math.abs(mg));
  if (x < 1.005) return 'level with the best baseline';
  return `${x < 10 ? x.toFixed(2) : x.toFixed(0)}× ${mg > 0 ? 'better than' : 'behind'} the best baseline`;
}
const vpill = v => `<span class="vpill v-${v}">${VLABEL[v]}</span>`;
const apill = c => `<span class="vpill a-${c}">${ACLASS[c]}</span>`;
function rowTip(r) {
  const b = bestBaseline(r), m = r.m;
  let h = `<div class="k">${esc(m.name)} · ${esc(m.domain)}</div><div class="t" style="margin:2px 0 6px">${esc(r.metric)}</div>`;
  h += `<div class="row"><span>${esc(m.name)}</span><b>${fmtVal(r.model, r.unit)}</b></div>`;
  for (const x of r.baselines.slice(0, 5)) h += `<div class="row"><span style="opacity:.8">${esc(x.name.length > 44 ? x.name.slice(0, 42) + '…' : x.name)} <span class="k">${x.kind}</span></span><span>${fmtVal(x.value, r.unit)}</span></div>`;
  if (r.baselines.length > 5) h += `<div class="k">+${r.baselines.length - 5} more baselines</div>`;
  h += '<hr>' + vpill(r.verdict) + (r.margin !== null ? ` <span style="margin-left:6px">${ratioText(r.margin)}</span>` : '');
  if (b && r.margin === null) h += '';
  h += `<div class="k" style="margin-top:6px">${TOUCH ? 'Tap again to open the model' : 'Click to open the model'}</div>`;
  return h;
}
function spark(svg, vals, color, W, H) {
  const x = d3.scaleLinear([0, vals.length - 1], [1, W - 4]);
  const ext = d3.extent(vals), y = d3.scaleLinear([Math.min(0, ext[0]), ext[1] || 1], [H - 2, 3]);
  const line = d3.line((d, i) => x(i), d => y(d)).curve(d3.curveStepAfter);
  const area = d3.area((d, i) => x(i), H - 1, d => y(d)).curve(d3.curveStepAfter);
  svg.innerHTML = `<path d="${area(vals)}" style="fill:${color};opacity:.12"/><path d="${line(vals)}" style="fill:none;stroke:${color};stroke-width:1.4"/><circle cx="${x(vals.length - 1)}" cy="${y(vals[vals.length - 1])}" r="2.6" style="fill:${color}"/>`;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
}

/* ============================================================ odometer */
function odo(el, n) {
  const s = fmt(n);
  if (el.dataset.odo === s) return;
  const fresh = !el.dataset.odo || el.dataset.odo.length !== s.length;
  if (fresh) {
    el.innerHTML = [...s].map(ch => /\d/.test(ch)
      ? `<span class="odo-c"><span class="odo-s">${'0123456789'.split('').map(d => `<span>${d}</span>`).join('')}</span></span>`
      : `<span>${ch}</span>`).join('');
  }
  el.dataset.odo = s;
  const cols = $$('.odo-s', el), digits = [...s].filter(ch => /\d/.test(ch));
  const apply = () => cols.forEach((c, i) => { c.style.transform = `translateY(${-digits[i] * 1.15}em)`; });
  if (fresh) requestAnimationFrame(() => requestAnimationFrame(apply)); else apply();
}

/* ============================================================ overview: KPIs */
function drawKpis(body) {
  const st = stats();
  if (!body._k) {
    const tiles = [
      { id: 'models', lab: 'Models', spark: 'models', col: 'var(--ink-2)', go: () => openPalette() },
      { id: 'graded', lab: 'Exam rows graded', spark: 'rows', col: 'var(--ink-2)', go: () => setPage('exams') },
      { id: 'win', lab: 'Wins', spark: 'w', col: 'var(--win)', go: e => toggle('verdict', 'win', e.shiftKey) },
      { id: 'loss', lab: 'Losses', spark: 'l', col: 'var(--loss)', go: e => toggle('verdict', 'loss', e.shiftKey) },
      { id: 'tie', lab: 'Ties and mixed', spark: 'ties', col: 'var(--tie)', go: e => { S.f.verdict = new Set(['tie', 'mixed']); changed(); } },
      { id: 'rungs', lab: 'Rung decisions', spark: 'rungs', col: 'var(--ink-2)', go: () => setPage('ratchet') },
      { id: 'zero', lab: 'Real labelled training examples', spark: null, col: 'var(--brand)', go: () => setPage('lineage') },
    ];
    body.innerHTML = `<div class="kpis">${tiles.map(t => `<button class="kpi${t.id === 'zero' ? ' zero' : ''}" data-k="${t.id}"><span class="kick">${t.lab}</span><span class="v" data-v></span><span class="s" data-s></span><svg aria-hidden="true"></svg></button>`).join('')}</div>`;
    tiles.forEach(t => { const el = $(`[data-k="${t.id}"]`, body); el.onclick = t.go; el._t = t; });
    body._k = tiles;
  }
  const W = Math.max(80, ($('.kpi', body).clientWidth || 140) - 28);
  const val = { models: st.ms.length, graded: st.graded, win: st.v.win, loss: st.v.loss, tie: st.v.tie + st.v.mixed, rungs: st.rungs, zero: 0 };
  const sub = {
    models: `${st.domains} domains · ${new Set(st.ms.map(m => m.platform)).size} simulator families`,
    graded: `plus ${st.v.info} context rows · ${st.graded ? pct(st.v.win / st.graded) : '0%'} won`,
    win: 'beat every baseline listed on the row',
    loss: 'at least one baseline did better',
    tie: 'inside the noise, or split by stratum',
    rungs: `${fmt(st.rej)} rejected (${st.rungs ? pct(st.rej / st.rungs) : '0%'}) by the ratchet`,
    zero: `the rule, held at all ${SER.length} commits that touched a model`,
  };
  for (const t of body._k) {
    const el = $(`[data-k="${t.id}"]`, body);
    odo($('[data-v]', el), val[t.id]);
    $('[data-s]', el).textContent = sub[t.id];
    const svg = $('svg', el);
    if (t.spark) spark(svg, SER.map(s => s[t.spark]), t.col, W, 26);
    else spark(svg, SER.map(() => 0), t.col, W, 26);
    el.title = t.spark ? `History: every commit that changed a model card or ratchet log, ${SER[0].t.slice(0, 10)} to ${SER[SER.length - 1].t.slice(0, 10)} (whole collection)` : 'No model in the collection trains on a real labelled example';
  }
}

/* ============================================================ overview: margin swarm */
const LAY = new Map();
function swarmLayout(W, group) {
  const key = W + '|' + group;
  if (LAY.has(key)) return LAY.get(key);
  const r = W < 640 ? 2.5 : 3.2;
  const labelW = group === 'none' ? 4 : Math.min(150, W * 0.2);
  const x = d3.scaleLinear([-1.3, 1.3], [labelW + 10, W - 12]).clamp(true);
  const laneOf = { none: () => 'All', platform: d => PLAT[d.m.platform], domain: d => d.m.domain, verdict: d => VLABEL[d.verdict] }[group];
  const order = group === 'none' ? ['All'] : group === 'platform' ? PLATS.map(p => PLAT[p]) : group === 'domain' ? [...DOMAINS] : ['Win', 'Tie', 'Mixed', 'Loss', 'Context'];
  for (const d of SROWS) { const k = laneOf(d); if (!order.includes(k)) order.push(k); } // every row gets a lane
  const lanes = [], pos = new Map();
  let y0 = 26;
  for (const name of order) {
    const nodes = SROWS.filter(d => laneOf(d) === name).map((d, i) => {
      const tx = x(Math.max(-1.3, Math.min(1.3, d.margin)));
      return { d, tx, x: tx, y: ((i * 37) % 11 - 5) * 0.6 };
    });
    if (!nodes.length) continue;
    const sim = d3.forceSimulation(nodes)
      .force('x', d3.forceX(n => n.tx).strength(0.95))
      .force('y', d3.forceY(0).strength(group === 'none' ? 0.03 : 0.1))
      .force('c', d3.forceCollide(r + 0.55).iterations(2)).stop();
    for (let i = 0; i < 220; i++) sim.tick();
    const [a, b] = d3.extent(nodes, n => n.y);
    const h = Math.max(group === 'none' ? 200 : 30, b - a + 2 * r + 12);
    const cy = y0 + h / 2 - (a + b) / 2;
    nodes.forEach(n => pos.set(n.d.key, [n.x, n.y + cy]));
    lanes.push({ name, y0, y1: y0 + h, n: nodes.length });
    y0 += h + (group === 'none' ? 0 : 3);
  }
  const L = { x, r, labelW, lanes, pos, height: y0 + 34 };
  LAY.set(key, L);
  return L;
}
function drawSwarm(body) {
  const W = Math.max(320, bw(body));
  const L = swarmLayout(W, S.swarmGroup);
  if (!body._svg) {
    body.innerHTML = `<div class="swarm-wrap"><svg role="img" aria-label="Every exam row with a ratio, placed by how far the model is from its best baseline"><g class="lanes"></g><g class="axis"></g><g class="dots"></g></svg></div><p class="note" data-n></p>`;
    body._svg = d3.select(body).select('svg');
  }
  const svg = body._svg.attr('viewBox', `0 0 ${W} ${L.height}`).attr('height', L.height);
  const x = L.x, top = 26, bottom = L.height - 30;
  // lanes
  svg.select('.lanes').selectAll('g.lane').data(L.lanes, d => d.name).join(
    en => { const g = en.append('g').attr('class', 'lane'); g.append('rect'); g.append('text').attr('class', 'ln'); g.append('text').attr('class', 'lc'); return g; })
    .call(g => g.select('rect').attr('x', 0).attr('width', W).attr('y', d => d.y0).attr('height', d => d.y1 - d.y0)
      .style('fill', (d, i) => S.swarmGroup === 'none' ? 'none' : i % 2 ? 'var(--grid)' : 'none'))
    .call(g => g.select('.ln').attr('x', 4).attr('y', d => (d.y0 + d.y1) / 2 - 1).text(d => S.swarmGroup === 'none' ? '' : d.name)
      .style('font-family', 'var(--sans)').style('font-size', '11.5px').style('fill', 'var(--ink-2)'))
    .call(g => g.select('.lc').attr('x', 4).attr('y', d => (d.y0 + d.y1) / 2 + 11).text(d => S.swarmGroup === 'none' ? '' : d.n + ' rows'));
  // axis
  const ph = MOBILE();
  const ticks = ph ? [-1, -Math.log10(3), 0, Math.log10(3), 1] : [-1, -Math.log10(3), -Math.log10(1.5), 0, Math.log10(1.5), Math.log10(3), 1];
  const tl = ph ? ['10×', '3×', 'even', '3×', '10×'] : ['10×', '3×', '1.5×', 'even', '1.5×', '3×', '10×'];
  const ax = svg.select('.axis');
  ax.selectAll('line.t').data(ticks).join('line').attr('class', 't').attr('x1', x).attr('x2', x).attr('y1', top - 4).attr('y2', bottom)
    .style('stroke', (d) => d === 0 ? 'var(--ink-2)' : 'var(--rule)').style('stroke-dasharray', d => d === 0 ? null : '2 3');
  ax.selectAll('text.t').data(ticks).join('text').attr('class', 't').attr('x', x).attr('y', bottom + 14).attr('text-anchor', 'middle').text((d, i) => tl[i]);
  ax.selectAll('text.ann').data([0, 1]).join('text').attr('class', 'ann').attr('y', 12)
    .attr('x', d => d ? W - 12 : L.labelW + 10).attr('text-anchor', d => d ? 'end' : 'start')
    .text(d => ph ? (d ? 'model better →' : '← baseline better') : d ? 'the model did better →' : '← the best baseline did better')
    .style('fill', d => d ? 'var(--win)' : 'var(--loss)').style('font-size', '10.5px');
  // dots
  const pass = r => rPass(r);
  const dots = svg.select('.dots').selectAll('circle').data(SROWS, d => d.key).join(
    en => en.append('circle').attr('r', L.r).attr('cx', d => L.pos.get(d.key)[0]).attr('cy', (L.height) / 2).style('opacity', 0)
      .on('mouseenter', function (ev, d) { d3.select(this).attr('r', L.r * 1.9); showTip(rowTip(d), ev); })
      .on('mousemove', ev => moveTip(ev))
      .on('mouseleave', function () { d3.select(this).attr('r', L.r); hideTip(); })
      .on('click', (ev, d) => tapOpen(ev, d.key, () => rowTip(d), () => openModel(d.m.id, d.key))));
  dots.style('fill', d => VC(d.verdict)).attr('r', L.r)
    .transition().duration(body._drawn ? 650 : 1100).delay((d, i) => body._drawn ? 0 : (i % 60) * 9).ease(d3.easeCubicOut)
    .attr('cx', d => L.pos.get(d.key)[0]).attr('cy', d => L.pos.get(d.key)[1])
    .style('opacity', d => pass(d) ? 0.92 : 0.07);
  body._drawn = true;
  const shown = SROWS.filter(pass), wins = shown.filter(d => d.margin > 0).length;
  $('[data-n]', body).textContent = `${shown.length} of ${SROWS.length} rows lit · the model is ahead of its best baseline on ${wins} of them. ` +
    `Ratio = model ÷ best baseline (inverted where lower is better), clipped at 20×. ${ROWS.length - SROWS.length} rows are text, change sign or aim at a target, so they appear in the wall and the table only.`;
}

/* ============================================================ overview: verdicts by domain */
function drawDomains(body) {
  const W = Math.max(280, bw(body)), lw = Math.min(118, W * 0.34), rw = 40, rh = 23;
  const data = DOMAINS.map(d => {
    const all = ROWS.filter(r => r.m.domain === d && r.verdict !== 'info');
    const cur = all.filter(r => rPass(r, 'domain'));
    const c = {}; GRADED.forEach(v => c[v] = cur.filter(r => r.verdict === v).length);
    return { d, tot: all.length, n: cur.length, c };
  }).filter(x => x.tot);
  const H = data.length * rh + 8;
  const x = d3.scaleLinear([0, d3.max(data, d => d.tot)], [lw + 6, W - rw - 6]);
  const sel = S.f.domain;
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Graded exam rows by domain and verdict">`;
  data.forEach((row, i) => {
    const y = i * rh + 4, on = !sel.size || sel.has(row.d);
    s += `<g class="bars-row" data-d="${esc(row.d)}" style="opacity:${on ? 1 : .35}">`;
    s += `<rect x="0" y="${y}" width="${W}" height="${rh - 2}" style="fill:transparent"/>`;
    s += `<text class="lbl" x="${lw}" y="${y + 14}" text-anchor="end" style="font-family:var(--sans);font-size:11.5px;fill:${sel.has(row.d) ? 'var(--ink)' : 'var(--ink-2)'};font-weight:${sel.has(row.d) ? 600 : 400}">${esc(row.d)}</text>`;
    s += `<rect x="${x(0)}" y="${y + 3}" width="${x(row.tot) - x(0)}" height="${rh - 8}" rx="2" style="fill:none;stroke:var(--rule);stroke-dasharray:2 2"/>`;
    let x0 = 0;
    for (const v of GRADED) {
      const n = row.c[v]; if (!n) continue;
      s += `<rect data-v="${v}" x="${x(x0)}" y="${y + 3}" width="${Math.max(.5, x(x0 + n) - x(x0))}" height="${rh - 8}" style="fill:${VC(v)}"><title>${esc(row.d)} · ${VLABEL[v]}: ${n}</title></rect>`;
      x0 += n;
    }
    s += `<text x="${W - 2}" y="${y + 14}" text-anchor="end">${row.n ? pct(row.c.win / row.n) : '–'}</text></g>`;
  });
  s += '</svg>';
  body.innerHTML = s + `<div class="legend" style="margin-top:6px">${GRADED.map(v => `<span><i style="background:${VC(v)}"></i> ${VLABEL[v]}</span>`).join('')}<span class="muted">· dashed = all graded rows · % = wins</span></div>`;
  $$('.bars-row', body).forEach(g => g.addEventListener('click', ev => {
    const v = ev.target.dataset.v;
    if (v) { const d = g.dataset.d; const same = S.f.domain.size === 1 && S.f.domain.has(d) && S.f.verdict.size === 1 && S.f.verdict.has(v); S.f.domain = same ? new Set() : new Set([d]); S.f.verdict = same ? new Set() : new Set([v]); changed(); }
    else toggle('domain', g.dataset.d, ev.shiftKey || ev.metaKey || ev.ctrlKey);
  }));
}

/* ============================================================ overview: growth from git history */
function drawGrowth(body) {
  const W = Math.max(320, bw(body)), ml = 6, mr = 46, ph = 62, gap = 24;
  const panels = [
    { t: 'Models released', keys: [['models', 'var(--ink-2)']] },
    { t: 'Graded exam rows', keys: [['w', 'var(--win)'], ['ties', 'var(--tie)'], ['l', 'var(--loss)']] },
    { t: 'Rung decisions', keys: [['kept', 'var(--ink-2)'], ['rej', 'var(--brand)']] },
  ];
  const H = panels.length * (ph + gap) + 18;
  const x = d3.scaleUtc(d3.extent(SER, d => d.d), [ml, W - mr]);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="The collection over time, from git history">`;
  panels.forEach((p, i) => {
    const y0 = i * (ph + gap) + 14;
    const st = d3.stack().keys(p.keys.map(k => k[0]))(SER);
    const max = d3.max(st[st.length - 1], d => d[1]);
    const y = d3.scaleLinear([0, max], [y0 + ph, y0]);
    const area = d3.area(d => x(d.data.d), d => y(d[0]), d => y(d[1])).curve(d3.curveStepAfter);
    s += `<text x="${ml}" y="${y0 - 4}" style="font-family:var(--sans);font-size:11.5px;fill:var(--ink-2)">${p.t}</text>`;
    st.forEach((layer, j) => { s += `<path d="${area(layer)}" style="fill:${p.keys[j][1]};opacity:.78"/>`; });
    const last = SER[SER.length - 1];
    s += `<text x="${W - mr + 6}" y="${y(max) + 9}" style="font-size:11px;fill:var(--ink)">${fmt(max)}</text>`;
    s += `<line x1="${ml}" x2="${W - mr}" y1="${y0 + ph}" y2="${y0 + ph}" style="stroke:var(--rule)"/>`;
    if (i === 1) s += `<text x="${W - mr + 6}" y="${y0 + ph}" style="font-size:9.5px">${pct(last.w / (last.w + last.l + last.ties))} won</text>`;
    if (i === 2) s += `<text x="${W - mr + 6}" y="${y0 + ph}" style="font-size:9.5px">${pct(last.rej / last.rungs)} no</text>`;
  });
  const days = d3.utcDay.range(d3.utcDay.ceil(x.domain()[0]), x.domain()[1], 3);
  days.filter(d => x(d) > ml + 22).forEach(d => { s += `<text x="${x(d)}" y="${H - 2}" text-anchor="middle">${d3.utcFormat('%b %-d')(d)}</text>`; });
  s += `<line class="rule" y1="8" y2="${H - 14}" style="stroke:var(--ink);stroke-width:1;opacity:0"/><rect class="ov" x="${ml}" y="0" width="${W - ml - mr}" height="${H - 12}" style="fill:transparent;cursor:crosshair"/></svg>`;
  body.innerHTML = s + `<p class="note">Rebuilt from git: the collection's totals after each of the ${SER.length} commits that changed a <code>model.yaml</code> or <code>ratchet.yaml</code>. Whole collection; filters do not apply.</p>`;
  const rule = $('.rule', body), bis = d3.bisector(d => d.d).center;
  $('.ov', body).addEventListener('mousemove', ev => {
    const pt = d3.pointer(ev, $('svg', body)); const i = bis(SER, x.invert(pt[0])); const d = SER[i];
    rule.setAttribute('x1', x(d.d)); rule.setAttribute('x2', x(d.d)); rule.style.opacity = .5;
    showTip(`<div class="k">${d3.utcFormat('%a %-d %b %Y, %H:%M UTC')(d.d)} · ${d.h}</div><div style="margin:3px 0 6px">${esc(d.s)}</div>
      <div class="row"><span>Models</span><b>${d.models}</b></div><div class="row"><span>Wins · losses · ties</span><b>${d.w} · ${d.l} · ${d.ties}</b></div><div class="row"><span>Rungs · rejected</span><b>${d.rungs} · ${d.rej}</b></div>`, ev);
  });
  $('.ov', body).addEventListener('mouseleave', () => { rule.style.opacity = 0; hideTip(); });
}

/* ============================================================ overview: coverage */
const covBand = a => a < 58 ? 'narrow' : a > 78 ? 'wide' : 'near';
const COVC = { narrow: 'var(--loss)', near: 'var(--win)', wide: 'var(--tie)' };
const COVL = { narrow: 'too narrow (below 58%)', near: 'within 10 points of 68%', wide: 'too wide (above 78%)' };
function drawCoverage(body) {
  const W = Math.max(300, bw(body)), r = 3.6;
  const key = 'cov' + W;
  if (!LAY.has(key)) {
    const x = d3.scaleLinear([10, 100], [12, W - 12]);
    const nodes = COVS.map(c => ({ c, tx: x(c.actual), x: x(c.actual), y: 0 }));
    const sim = d3.forceSimulation(nodes).force('x', d3.forceX(n => n.tx).strength(1)).force('y', d3.forceY(0).strength(.06)).force('c', d3.forceCollide(r + .6).iterations(2)).stop();
    for (let i = 0; i < 200; i++) sim.tick();
    const [a, b] = d3.extent(nodes, n => n.y), h = b - a + 2 * r + 30;
    LAY.set(key, { x, nodes, off: 26 - a + r, h: h + 30 });
  }
  const L = LAY.get(key), x = L.x, H = L.h;
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Error-bar coverage on the real exams">`;
  s += `<rect x="${x(58)}" y="14" width="${x(78) - x(58)}" height="${H - 36}" style="fill:var(--win);opacity:.08"/>`;
  s += `<line x1="${x(68)}" x2="${x(68)}" y1="10" y2="${H - 22}" style="stroke:var(--ink-2);stroke-dasharray:3 3"/><text x="${x(68)}" y="9" text-anchor="middle" style="fill:var(--ink)">nominal 68%</text>`;
  [20, 40, 60, 80, 100].forEach(t => { s += `<text x="${x(t)}" y="${H - 6}" text-anchor="middle">${t}%</text>`; });
  for (const n of L.nodes) {
    const on = mPass(n.c.m);
    s += `<circle data-c="${n.c.key}" cx="${n.x}" cy="${n.y + L.off}" r="${r}" style="fill:${COVC[covBand(n.c.actual)]};opacity:${on ? .9 : .08};cursor:pointer"/>`;
  }
  s += '</svg>';
  const shown = COVS.filter(c => mPass(c.m)), cnt = d3.rollup(shown, v => v.length, c => covBand(c.actual));
  body.innerHTML = s + `<div class="legend">${['near', 'narrow', 'wide'].map(b => `<span><i style="background:${COVC[b]}"></i> ${cnt.get(b) || 0} ${COVL[b]}</span>`).join('')}</div>
    <p class="note">Share of real exam truths inside each output's 68% band, after its one widen-only factor was fit on dev. ${shown.length} outputs from ${new Set(shown.map(c => c.m.id)).size} models.</p>`;
  const byKey = new Map(COVS.map(c => [c.key, c]));
  const svg = $('svg', body);
  svg.addEventListener('mousemove', ev => { const k = ev.target.dataset?.c; if (!k) return hideTip(); const c = byKey.get(k); showTip(`<div class="k">${esc(c.m.name)}</div><div class="t">${esc(c.output)}</div><div class="row" style="margin-top:4px"><span>Inside the 68% band</span><b>${c.actual}%</b></div><div class="k" style="margin-top:4px">${COVL[covBand(c.actual)]}</div>`, ev); });
  svg.addEventListener('mouseleave', hideTip);
  svg.addEventListener('click', ev => { const k = ev.target.dataset?.c; if (!k) return; const c = byKey.get(k); tapOpen(ev, k, () => `<div class="k">${esc(c.m.name)}</div><div class="t">${esc(c.output)}</div><div class="row" style="margin-top:4px"><span>Inside the 68% band</span><b>${c.actual}%</b></div><div class="k" style="margin-top:4px">Tap again to open</div>`, () => openModel(c.m.id, null, 'coverage')); });
}

/* ============================================================ overview: audit strip and pipeline */
function auditTiles(ms, compact) {
  return AORDER.map(c => {
    const g = ms.filter(m => m.auditCls === c);
    if (!g.length) return '';
    return `<div class="grp"><h6>${ACLASS[c]} · ${g.length}</h6>${g.map(m => `<button class="tile" data-m="${m.id}" aria-pressed="${S.auditSel === m.id}"><i style="background:var(--${c})"></i>${esc(m.name)}${compact ? '' : ` <span class="mono muted">${esc(m.audit ? m.audit.version.split(' ')[0] : '')}</span>`}</button>`).join('')}</div>`;
  }).join('');
}
function actionBar(ms, W) {
  const all = ms.flatMap(m => m.audit ? m.audit.resolutions : []);
  const c = d3.rollup(all, v => v.length, a => a.action), n = all.length || 1;
  let x = 0, s = `<div style="display:flex;height:14px;border-radius:3px;overflow:hidden;margin:6px 0 6px">`;
  for (const k of Object.keys(ACT)) { const v = c.get(k) || 0; if (!v) continue; s += `<i title="${ACT[k]}: ${v}" style="display:block;width:${v / n * 100}%;background:${ACOL[k]}"></i>`; x += v; }
  s += `</div><div class="legend">${Object.keys(ACT).filter(k => c.get(k)).map(k => `<span><i style="background:${ACOL[k]}"></i>${ACT[k]} ${c.get(k)}</span>`).join('')}</div>`;
  return { html: s, n: all.length, fixed: c.get('fixed') || 0 };
}
function drawAuditStrip(body) {
  const ms = MODELS.filter(m => mPass(m));
  const ab = actionBar(ms);
  body.innerHTML = `<div class="board">${auditTiles(ms, true)}</div><div class="kick" style="margin-top:12px">${ab.n} findings, and what was done</div>${ab.html}`;
  $$('.tile', body).forEach(b => b.onclick = () => { S.auditSel = b.dataset.m; setPage('audits'); });
}
function drawPipeline(body) {
  const st = ['done', 'in-progress', 'checked', 'candidate', 'blocked'];
  const KC = { model: 'var(--ink-2)', platform: 'var(--sel)', method: 'var(--brand)', data: 'var(--tie)' };
  const by = d3.group(D.backlog, b => b.status);
  body.innerHTML = `<div class="pipe" style="display:grid;grid-template-columns:repeat(${st.length},minmax(0,1fr));gap:8px">${st.map(s => `<div><div class="kick">${s.replace('-', ' ')} · ${(by.get(s) || []).length}</div>${(by.get(s) || []).map(b => `<div title="${esc(b.question || '')}" style="font-size:12px;margin-top:5px;display:flex;gap:6px;align-items:baseline"><i style="width:7px;height:7px;border-radius:50%;flex:none;background:${KC[b.kind] || 'var(--ink-3)'}"></i><span>${esc(b.name)}</span></div>`).join('')}</div>`).join('')}</div>
  <div class="legend" style="margin-top:10px">${Object.entries(KC).map(([k, c]) => `<span><i style="background:${c};width:8px;height:8px;border-radius:50%"></i>${k}</span>`).join('')}</div><p class="note">From <code>docs/backlog.yaml</code>. Hover a name for its question.</p>`;
}

/* ============================================================ exams: the wall */
function drawWall(body) {
  const ms = MODELS.filter(m => mPass(m));
  const sorters = { win: (a, b) => b.winShare - a.winShare || b.graded - a.graded, domain: (a, b) => DOMAINS.indexOf(a.domain) - DOMAINS.indexOf(b.domain) || a.order - b.order, release: (a, b) => (a.released || '').localeCompare(b.released || '') || a.order - b.order, rows: (a, b) => b.rows.length - a.rows.length };
  ms.sort(sorters[S.wallSort]);
  const order = { win: 0, tie: 1, mixed: 2, loss: 3, info: 4 };
  body.innerHTML = ms.length ? `<div class="wall">${ms.map(m => {
    const rows = [...m.rows].sort((a, b) => order[a.verdict] - order[b.verdict]);
    const cols = Math.min(8, Math.max(3, Math.ceil(Math.sqrt(rows.length * 1.6))));
    return `<div class="wblock"><div class="wn" data-open="${m.id}"><i style="background:${m.accent}"></i>${esc(m.name)}<span class="c">${m.tally.win}–${m.tally.loss}${m.tally.tie + m.tally.mixed ? '–' + (m.tally.tie + m.tally.mixed) : ''}</span></div>
      <div class="wcells" style="grid-template-columns:repeat(${cols},11px)">${rows.map(r => `<button class="wcell ${r.verdict === 'info' ? 'info' : ''} ${rPass(r) ? '' : 'dim'}" data-k="${r.key}" aria-label="${esc(m.name)}: ${esc(r.metric)}: ${VLABEL[r.verdict]}" style="${r.verdict === 'info' ? '' : `background:${VC(r.verdict)}`}"></button>`).join('')}</div></div>`;
  }).join('')}</div>
  <div class="legend" style="margin-top:12px">${['win', 'tie', 'mixed', 'loss'].map(v => `<span><i style="background:${VC(v)};height:9px;width:9px"></i>${VLABEL[v]}</span>`).join('')}<span><i style="height:9px;width:9px;background:repeating-linear-gradient(135deg,var(--info) 0 2px,transparent 2px 4px)"></i>Context (reported, not graded)</span><span class="muted">· score under each name: wins–losses–ties</span></div>` : '<div class="empty">No model passes the filters.</div>';
  if (!body._wired) {
    body._wired = true;
    const rk = new Map(ROWS.map(r => [r.key, r]));
    body.addEventListener('mouseover', ev => { const k = ev.target.dataset?.k; if (k) showTip(rowTip(rk.get(k)), ev); });
    body.addEventListener('mousemove', ev => { if (ev.target.dataset?.k) moveTip(ev); });
    body.addEventListener('mouseout', ev => { if (ev.target.dataset?.k) hideTip(); });
    body.addEventListener('click', ev => {
      const k = ev.target.dataset?.k; if (k) { const r = rk.get(k); tapOpen(ev, k, () => rowTip(r), () => openModel(r.m.id, k)); return; }
      const o = ev.target.closest('[data-open]'); if (o) openModel(o.dataset.open);
    });
  }
}

/* ============================================================ exams: decomposition tree */
const TDIM = {
  domain: { label: 'Domain', f: r => r.m.domain },
  platform: { label: 'Simulator family', f: r => PLAT[r.m.platform] },
  model: { label: 'Model', f: r => r.m.name },
  dumb: { label: 'Against the dumb baseline', f: r => r.beat.dumb === undefined ? 'No dumb baseline on the row' : r.beat.dumb === 'tie' ? 'Ties the dumb baseline' : r.beat.dumb ? 'Beats the dumb baseline' : 'Behind the dumb baseline' },
  classical: { label: 'Against the classical method', f: r => r.beat.classical === undefined ? 'No classical method on the row' : r.beat.classical === 'tie' ? 'Ties the classical method' : r.beat.classical ? 'Beats the classical method' : 'Behind the classical method' },
  better: { label: 'Metric direction', f: r => ({ lower: 'Lower is better', higher: 'Higher is better', closer: 'Closer is better' })[r.better] },
  audit: { label: 'Audit finding', f: r => ACLASS[r.m.auditCls] },
  bars: { label: 'Error bars fit on real dev', f: r => r.m.roles.has('error_bars') ? 'Error bars widened on dev' : 'No dev-fit error bars' },
  anchor: { label: 'Physics anchor in the output', f: r => r.m.approach.physics_anchor ? 'Predicts a correction to physics' : 'Predicts directly' },
  nb: { label: 'Baselines on the row', f: r => r.baselines.length >= 3 ? '3 or more baselines' : r.baselines.length === 2 ? '2 baselines' : r.baselines.length === 1 ? '1 baseline' : 'No baseline' },
};
function comp(rows) { const c = { win: 0, tie: 0, mixed: 0, loss: 0 }; rows.forEach(r => c[r.verdict]++); return c; }
function tbar(c, n, w) {
  return `<div class="tbar" style="width:${Math.max(4, w * 100)}%">${GRADED.map(v => c[v] ? `<i style="width:${c[v] / n * 100}%;background:${VC(v)}"></i>` : '').join('')}</div>`;
}
function drawTree(body) {
  const base = ROWS.filter(r => r.verdict !== 'info' && rPass(r));
  const T = S.tree;
  let html = '<div class="tree">';
  const rc = comp(base);
  html += `<div class="tcol"><div class="kick" style="margin:6px 0 10px">All graded rows</div><button class="tnode" aria-pressed="true"><div class="l"><span>${fmt(base.length)} rows</span><span class="mono">${base.length ? pct(rc.win / base.length) : '–'} won</span></div>${tbar(rc, base.length || 1, 1)}</button></div>`;
  let cur = base;
  for (let i = 0; i <= T.dims.length; i++) {
    if (i === T.dims.length) {
      if (T.path.length === T.dims.length && T.dims.length < 5) {
        const left = Object.keys(TDIM).filter(k => !T.dims.includes(k));
        html += `<div class="tcol"><select data-add aria-label="Split further by"><option value="">+ split further by…</option>${left.map(k => `<option value="${k}">${TDIM[k].label}</option>`).join('')}</select></div>`;
      }
      break;
    }
    const dim = TDIM[T.dims[i]];
    const g = d3.rollups(cur, v => v, dim.f).sort((a, b) => b[1].length - a[1].length);
    const mx = d3.max(g, x => x[1].length) || 1;
    html += `<div class="tcol"><select data-dim="${i}" aria-label="Split level ${i + 1}">${Object.keys(TDIM).map(k => `<option value="${k}" ${k === T.dims[i] ? 'selected' : ''} ${T.dims.includes(k) && k !== T.dims[i] ? 'disabled' : ''}>${TDIM[k].label}</option>`).join('')}${i > 0 ? '<option value="__x">Remove this level</option>' : ''}</select>`;
    html += g.slice(0, 18).map(([k, rs]) => { const c = comp(rs); return `<button class="tnode" data-lv="${i}" data-v="${esc(k)}" aria-pressed="${T.path[i] === k}"><div class="l"><span title="${esc(k)}">${esc(k)}</span><span class="mono">${rs.length} · ${pct(c.win / rs.length)}</span></div>${tbar(c, rs.length, rs.length / mx)}</button>`; }).join('');
    if (g.length > 18) html += `<div class="note">+${g.length - 18} more</div>`;
    html += '</div>';
    if (T.path[i] === undefined) break;
    cur = cur.filter(r => dim.f(r) === T.path[i]);
  }
  body.innerHTML = html + '</div><p class="note">Click a bar to split it by the next level; choose the level from its menu. Bars show win, tie, mixed and loss shares; the number after the count is the win rate.</p>';
  $$('.tnode[data-lv]', body).forEach(b => b.onclick = () => { const i = +b.dataset.lv; T.path = T.path.slice(0, i); if (b.getAttribute('aria-pressed') !== 'true') T.path[i] = b.dataset.v; runCardId('tree'); });
  $$('select[data-dim]', body).forEach(s => s.onchange = () => { const i = +s.dataset.dim; if (s.value === '__x') T.dims.splice(i, 1); else T.dims[i] = s.value; T.path = T.path.slice(0, i); runCardId('tree'); });
  const add = $('select[data-add]', body); if (add) add.onchange = () => { if (add.value) { T.dims.push(add.value); runCardId('tree'); } };
}

/* ============================================================ exams: what goes with a win */
function drawDrivers(body) {
  const base = ROWS.filter(r => r.verdict !== 'info' && rPass(r));
  const p0 = base.length ? base.filter(r => r.verdict === 'win').length / base.length : 0;
  const keys = ['dumb', 'classical', 'platform', 'better', 'bars', 'anchor', 'nb', 'audit', 'domain'];
  const items = [];
  for (const k of keys) for (const [v, rs] of d3.rollups(base, x => x, TDIM[k].f)) {
    if (rs.length < 15 || rs.length === base.length) continue;
    const p = rs.filter(r => r.verdict === 'win').length / rs.length;
    items.push({ k, v, n: rs.length, p, lift: p - p0 });
  }
  items.sort((a, b) => b.lift - a.lift);
  const pick = items.length > 14 ? [...items.slice(0, 7), ...items.slice(-7)] : items;
  const mx = d3.max(pick, d => Math.abs(d.lift)) || .1;
  body.innerHTML = base.length ? `<div class="kick" style="margin-bottom:8px">Win rate overall: ${pct(p0)} of ${fmt(base.length)} graded rows</div><div class="drv">${pick.map(d => `
    <div class="lab" title="${esc(TDIM[d.k].label)}: ${esc(d.v)}"><span class="muted">${esc(TDIM[d.k].label)}:</span> ${esc(d.v)}</div>
    <div style="position:relative;height:12px;background:linear-gradient(90deg,transparent calc(50% - .5px),var(--ink-3) calc(50% - .5px),var(--ink-3) calc(50% + .5px),transparent calc(50% + .5px))"><i style="position:absolute;top:2px;height:8px;border-radius:2px;${d.lift >= 0 ? `left:50%;width:${d.lift / mx * 50}%;background:var(--win)` : `right:50%;width:${-d.lift / mx * 50}%;background:var(--loss)`}"></i></div>
    <div class="mono" style="text-align:right">${pct(d.p)} <span class="muted">n${d.n}</span></div>`).join('')}</div>
    <p class="note">Win rate in each group against the overall rate, groups of 15 rows or more. Rows inside one model are not independent, so read this as where wins cluster, not as cause.</p>` : '<div class="empty">No graded rows under these filters.</div>';
}

/* ============================================================ exams: the table */
function tableRows() {
  return ROWS.filter(r => rPass(r) && (S.table.ctx || r.verdict !== 'info'));
}
function drawTable(body) {
  const rows = tableRows();
  const vo = { win: 0, tie: 1, mixed: 2, loss: 3, info: 4 };
  const key = { model: r => r.m.name, metric: r => r.metric, verdict: r => vo[r.verdict], margin: r => r.margin ?? -99 }[S.table.sort];
  rows.sort((a, b) => { const x = key(a), y = key(b); return (x < y ? -1 : x > y ? 1 : 0) * S.table.dir; });
  const th = (k, l) => `<th data-s="${k}" aria-sort="${S.table.sort === k ? (S.table.dir > 0 ? 'ascending' : 'descending') : 'none'}">${l}${S.table.sort === k ? (S.table.dir > 0 ? ' ↑' : ' ↓') : ''}</th>`;
  body.innerHTML = `<div class="tscroll"><table class="rows"><thead><tr>${th('model', 'Model')}${th('metric', 'Exam row')}<th style="text-align:right">Model</th><th>Best baseline</th>${th('verdict', 'Verdict')}${th('margin', 'Margin')}</tr></thead><tbody>${rows.map(r => {
    const b = bestBaseline(r);
    const w = r.margin === null ? 0 : Math.min(1, Math.abs(r.margin) / 1.3) * 46;
    return `<tr data-k="${r.key}" style="cursor:pointer"><td><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:${r.m.accent};margin-right:6px"></span>${esc(r.m.name)}</td><td>${esc(r.metric)}</td><td class="n">${fmtVal(r.model, r.unit)}</td><td style="font-size:12px">${b ? `${fmtVal(b.value, r.unit)} <span class="muted">${esc(b.name.length > 40 ? b.name.slice(0, 38) + '…' : b.name)}</span>` : '<span class="muted">–</span>'}</td><td>${vpill(r.verdict)}</td><td class="n">${r.margin === null ? '<span class="muted">–</span>' : `<span class="mbar" style="width:${w}px;background:${r.margin >= 0 ? 'var(--win)' : 'var(--loss)'}"></span> ${(Math.pow(10, Math.abs(r.margin))).toFixed(2)}×`}</td></tr>`;
  }).join('')}</tbody></table></div><p class="note">${rows.length} rows. Click a header to sort, a row to open its model. Margin: how many times better (green) or worse (red) than the best baseline.</p>`;
  $$('th[data-s]', body).forEach(h => h.onclick = () => { const k = h.dataset.s; if (S.table.sort === k) S.table.dir *= -1; else { S.table.sort = k; S.table.dir = k === 'margin' ? -1 : 1; } drawTable(body); });
  $$('tr[data-k]', body).forEach(tr => tr.onclick = () => { const r = ROWS.find(x => x.key === tr.dataset.k); openModel(r.m.id, r.key); });
}
function csvRows() {
  const q = v => '"' + String(v ?? '').replace(/"/g, '""') + '"';
  const rows = tableRows();
  const head = ['model', 'domain', 'metric', 'better', 'unit', 'model_value', 'verdict', 'best_baseline', 'best_baseline_value', 'margin_log10', 'baselines'];
  return [head.join(','), ...rows.map(r => { const b = bestBaseline(r); return [r.m.name, r.m.domain, r.metric, r.better, r.unit, r.model, r.verdict, b?.name, b?.value, r.margin, r.baselines.map(x => `${x.name} (${x.kind}) = ${x.value}`).join('; ')].map(q).join(','); })].join('\n');
}

/* ============================================================ ratchet: anatomy of a no */
function drawAnatomy(body) {
  const ms = modelsIn();
  const rs = ms.flatMap(m => m.allRungs.map(r => ({ ...r, m })));
  const rej = rs.filter(r => r.decision === 'rejected');
  const c = d3.rollup(rej, v => v.length, r => r.rc);
  const n = rej.length || 1;
  const order = ['frozen', 'gate', 'other', 'stopped'];
  let bar = '<div style="display:flex;height:22px;border-radius:3px;overflow:hidden;margin:8px 0 10px">';
  order.forEach(k => { const v = c.get(k) || 0; if (v) bar += `<i title="${RC[k]}: ${v}" style="display:block;width:${v / n * 100}%;background:${RCOL[k]}"></i>`; });
  bar += '</div>';
  const ex = k => { const r = rej.find(x => x.rc === k); return r ? `<div class="note" style="margin:2px 0 0 18px">“${esc(r.why.length > 150 ? r.why.slice(0, 148) + '…' : r.why)}” <span class="mono">${esc(r.m.name)} ${esc(r.id)}</span></div>` : ''; };
  body.innerHTML = `<div style="display:flex;gap:22px;align-items:baseline;flex-wrap:wrap"><div><div class="kick">Decisions</div><div style="font:300 30px var(--mono)">${fmt(rs.length)}</div></div><div><div class="kick">Kept or released</div><div style="font:300 30px var(--mono);color:var(--win)">${fmt(rs.length - rej.length)}</div></div><div><div class="kick">Rejected</div><div style="font:300 30px var(--mono);color:var(--brand)">${fmt(rej.length)}</div></div></div>
  ${bar}${order.filter(k => c.get(k)).map(k => `<div style="display:flex;gap:8px;align-items:center;font-size:13px;margin-top:8px"><i style="width:10px;height:10px;border-radius:2px;background:${RCOL[k]}"></i><b class="mono" style="min-width:34px">${c.get(k)}</b>${RC[k]}</div>${ex(k)}`).join('')}
  <p class="note">A rung is kept only if it beats the incumbent on a frozen simulated set by the margin and passes a gate on a few real dev units. Reasons are read from each rung's logged text by keyword, so treat the split as approximate; every reason is on the model's page.</p>`;
}

/* ============================================================ ratchet: strictness vs exam */
function drawStrict(body) {
  const W = Math.max(300, bw(body)), H = 300, m = { l: 40, r: 14, t: 12, b: 34 };
  const ms = modelsIn().filter(d => d.nRungs && d.graded);
  const x = d3.scaleLinear([0, 1], [m.l, W - m.r]), y = d3.scaleLinear([0, 1], [H - m.b, m.t]);
  const rr = d3.scaleSqrt([0, d3.max(MODELS, d => d.nRungs)], [3, 22]);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Rejection rate against exam win rate per model">`;
  [0, .25, .5, .75, 1].forEach(t => { s += `<line x1="${x(t)}" x2="${x(t)}" y1="${m.t}" y2="${H - m.b}" style="stroke:var(--rule-2)"/><text x="${x(t)}" y="${H - m.b + 14}" text-anchor="middle">${pct(t)}</text><line x1="${m.l}" x2="${W - m.r}" y1="${y(t)}" y2="${y(t)}" style="stroke:var(--rule-2)"/><text x="${m.l - 6}" y="${y(t) + 3}" text-anchor="end">${pct(t)}</text>`; });
  s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle" style="fill:var(--ink-2)">share of rungs the ratchet rejected →</text><text transform="translate(11,${(H - m.b + m.t) / 2}) rotate(-90)" text-anchor="middle" style="fill:var(--ink-2)">exam rows won →</text>`;
  [...ms].sort((a, b) => b.nRungs - a.nRungs).forEach(d => {
    s += `<circle data-m="${d.id}" cx="${x(d.rejRate)}" cy="${y(d.winShare)}" r="${rr(d.nRungs)}" style="fill:${d.accent};fill-opacity:.55;stroke:${d.accent};cursor:pointer"/>`;
  });
  const placed = [];
  ms.filter(d => d.nRungs >= 25 || d.winShare >= .75 || (d.winShare <= .02 && d.graded >= 15)).sort((a, b) => MOBILE() ? b.nRungs - a.nRungs : 0).forEach(d => {
    if (MOBILE()) { const px = x(d.rejRate), py = y(d.winShare); if (placed.some(([a, b]) => Math.abs(a - px) < 70 && Math.abs(b - py) < 13)) return; placed.push([px, py]); }
    const flip = MOBILE() && x(d.rejRate) > W * .68;
    s += `<text x="${flip ? x(d.rejRate) - rr(d.nRungs) - 3 : x(d.rejRate) + rr(d.nRungs) + 3}" y="${y(d.winShare) + 3}" ${flip ? 'text-anchor="end" ' : ''}style="fill:var(--ink-2);font-family:var(--sans);font-size:10.5px">${esc(d.name)}</text>`;
  });
  s += '</svg>';
  const rc = ms.length > 2 ? d3.sum(ms, d => (d.rejRate - d3.mean(ms, e => e.rejRate)) * (d.winShare - d3.mean(ms, e => e.winShare))) / Math.sqrt(d3.sum(ms, d => (d.rejRate - d3.mean(ms, e => e.rejRate)) ** 2) * d3.sum(ms, d => (d.winShare - d3.mean(ms, e => e.winShare)) ** 2)) : NaN;
  body.innerHTML = s + `<p class="note">One circle per model, sized by rung decisions. Correlation across these ${ms.length} models: r = ${isFinite(rc) ? rc.toFixed(2) : '–'}. A model that refuses more rungs does not win more often on its exam; the ratchet guards against fooling yourself, not against a hard problem.</p>`;
  const svg = $('svg', body);
  svg.addEventListener('mousemove', ev => { const id = ev.target.dataset?.m; if (!id) return hideTip(); const d = BYID.get(id); showTip(`<div class="t">${esc(d.name)}</div><div class="row"><span>Rungs · rejected</span><b>${d.nRungs} · ${d.nRej} (${pct(d.rejRate)})</b></div><div class="row"><span>Graded rows won</span><b>${d.tally.win} of ${d.graded} (${pct(d.winShare)})</b></div>`, ev); });
  svg.addEventListener('mouseleave', hideTip);
  svg.addEventListener('click', ev => { const id = ev.target.dataset?.m; if (!id) return; const d = BYID.get(id); tapOpen(ev, 'strict:' + id, () => `<div class="t">${esc(d.name)}</div><div class="row"><span>Rungs · rejected</span><b>${d.nRungs} · ${d.nRej} (${pct(d.rejRate)})</b></div><div class="row"><span>Graded rows won</span><b>${d.tally.win} of ${d.graded} (${pct(d.winShare)})</b></div><div class="k" style="margin-top:4px">Tap again to open</div>`, () => openModel(id, null, 'ratchet')); });
}

/* ============================================================ ratchet: ladders */
function ladderSvg(L, W, H, big) {
  const rs = L.rungs, n = rs.length, pad = big ? { l: 34, r: 10, t: 10, b: 18 } : { l: 2, r: 4, t: 5, b: 5 };
  const sims = rs.map(r => typeof r.sim === 'number' ? r.sim : null);
  const vals = sims.filter(v => v !== null);
  const x = d3.scalePoint(d3.range(n), [pad.l, W - pad.r]).padding(.4);
  let s = '';
  if (!vals.length) {
    rs.forEach((r, i) => { s += `<rect x="${x(i) - 2}" y="${(big ? 20 : H) / 2 - 7}" width="4" height="14" rx="1" style="fill:${r.decision === 'rejected' ? 'var(--rule)' : 'var(--ink-2)'}"><title>${esc(r.id)} · ${r.decision}</title></rect>`; });
    return { s, x, y: null };
  }
  const [a, b] = d3.extent(vals), sp = (b - a) || Math.abs(a) || 1;
  const y = d3.scaleLinear([a - sp * .12, b + sp * .12], [H - pad.b, pad.t]);
  let inc = null, d = '';
  const hi = L.higher !== false;
  rs.forEach((r, i) => {
    if (r.decision !== 'rejected' && sims[i] !== null) { d += (inc === null ? `M${x(i)},${y(sims[i])}` : `H${x(i)}V${y(sims[i])}`); inc = sims[i]; }
  });
  if (inc !== null) d += `H${W - pad.r}`;
  if (big) { y.ticks(3).forEach(t => { s += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${y(t)}" y2="${y(t)}" style="stroke:var(--rule-2)"/><text x="${pad.l - 4}" y="${y(t) + 3}" text-anchor="end">${t}</text>`; }); }
  s += `<path d="${d}" style="fill:none;stroke:var(--win);stroke-width:${big ? 2 : 1.6}"/>`;
  rs.forEach((r, i) => {
    if (sims[i] === null) { s += `<line x1="${x(i)}" x2="${x(i)}" y1="${H - pad.b - 4}" y2="${H - pad.b}" style="stroke:var(--ink-3)"/>`; return; }
    const rej = r.decision === 'rejected';
    s += `<circle data-i="${i}" cx="${x(i)}" cy="${y(sims[i])}" r="${big ? 5 : 3}" style="fill:${rej ? 'var(--panel)' : r.decision === 'released' ? 'var(--brand)' : 'var(--win)'};stroke:${rej ? 'var(--ink-3)' : 'var(--panel)'};stroke-width:1.3;${rej ? 'stroke-dasharray:2 1.4' : ''}"/>`;
    if (big && n <= 24) s += `<text x="${x(i)}" y="${H - 4}" text-anchor="middle">${esc(r.id).slice(0, 6)}</text>`;
  });
  void hi;
  return { s, x, y };
}
function drawLadders(body) {
  const ms = modelsIn().sort((a, b) => b.nRungs - a.nRungs);
  body.innerHTML = `<div class="ladders">${ms.map(m => { const L = m.main; const W = 200, H = 62; const g = L ? ladderSvg(L, W, H) : { s: '' }; return `<div class="lad" data-m="${m.id}" title="${esc(L ? L.name : '')}"><div class="ln"><span>${esc(m.name)}</span><span class="mono muted">${m.nRej}/${m.nRungs} no</span></div><svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${g.s}</svg><div class="mono muted" style="font-size:9.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${m.ladders.length > 1 ? m.ladders.length + ' ladders · ' : ''}${esc(L ? (L.scored_on || '') : '')}</div></div>`; }).join('')}</div>
  <div class="legend" style="margin-top:10px"><span><i style="background:var(--win)"></i>incumbent on the frozen set</span><span><i style="background:var(--win);width:8px;height:8px;border-radius:50%"></i>kept</span><span><i style="background:var(--brand);width:8px;height:8px;border-radius:50%"></i>released</span><span><i style="border:1px dashed var(--ink-3);width:8px;height:8px;border-radius:50%"></i>rejected</span></div><p class="note">Each model's longest ladder of scored rungs. Some ladders were re-scored on newer frozen sets part-way, so a later kept rung can sit lower; the model page has every ladder and every reason.</p>`;
  $$('.lad', body).forEach(el => el.onclick = () => openModel(el.dataset.m, null, 'ratchet'));
}

/* ============================================================ lineage: sankey */
function drawSankey(body) {
  const ms = modelsIn();
  const W = Math.max(360, bw(body)), H = Math.max(420, ms.length * 16 + 40);
  if (!d3.sankey) { body.innerHTML = '<div class="empty">The diagram library did not load.</div>'; return; }
  if (!ms.length) { body.innerHTML = '<div class="empty">No model passes the filters.</div>'; return; }
  const nodes = [], links = [];
  const ph = MOBILE();
  const SHORT = { backdrop: 'Noise backdrop', calibration: 'Calibration', gate: 'Rung gate', error_bars: 'Error bars', exam: 'Exam' };
  const plats = ph ? [] : PLATS.filter(p => ms.some(m => m.platform === p));
  plats.forEach(p => nodes.push({ id: 'p:' + p, name: PLAT[p], kind: 'p' }));
  const sorted = [...ms].sort((a, b) => PLATS.indexOf(a.platform) - PLATS.indexOf(b.platform) || a.order - b.order);
  sorted.forEach(m => nodes.push({ id: 'm:' + m.id, name: m.name, kind: 'm', m }));
  ROLES.forEach(r => nodes.push({ id: 'r:' + r, name: ph ? SHORT[r] : ROLE[r], kind: 'r', role: r }));
  sorted.forEach(m => {
    const rs = ROLES.filter(r => m.roles.has(r));
    if (!ph) links.push({ source: 'p:' + m.platform, target: 'm:' + m.id, value: Math.max(1, rs.length), m });
    rs.forEach(r => links.push({ source: 'm:' + m.id, target: 'r:' + r, value: 1, m, role: r }));
  });
  const used = new Set(links.map(l => l.target).concat(links.map(l => l.source)));
  const g = d3.sankey().nodeId(d => d.id).nodeWidth(9).nodePadding(4).nodeSort(null).linkSort(null).extent(ph ? [[104, 8], [W - 92, H - 70]] : [[150, 8], [W - 170, H - 70]])({ nodes: nodes.filter(n => used.has(n.id)).map(d => ({ ...d })), links: links.map(d => ({ ...d })) });
  const path = d3.sankeyLinkHorizontal();
  const RCOLR = { backdrop: 'var(--ink-2)', calibration: 'var(--sel)', gate: 'var(--brand)', error_bars: 'var(--mixed)', exam: 'var(--win)' };
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Where real data enters each model">`;
  g.links.forEach((l, i) => { s += `<path class="sk-link" data-l="${i}" d="${path(l)}" style="stroke:${l.role ? RCOLR[l.role] : l.m.accent};stroke-width:${Math.max(1, l.width)}"/>`; });
  g.nodes.forEach(n => {
    const col = n.kind === 'm' ? n.m.accent : n.kind === 'r' ? RCOLR[n.role] : 'var(--ink-2)';
    s += `<g class="sk-node"><rect ${n.kind === 'm' ? `data-m="${n.m.id}" style="cursor:pointer;fill:${col}"` : `style="fill:${col}"`} x="${n.x0}" y="${n.y0}" width="${n.x1 - n.x0}" height="${Math.max(1, n.y1 - n.y0)}"/>`;
    const mid = (n.y0 + n.y1) / 2;
    if (n.kind === 'p') s += `<text x="${n.x0 - 6}" y="${mid + 4}" text-anchor="end">${esc(n.name)}</text>`;
    else if (n.kind === 'm') s += ph ? `<text x="${n.x0 - 5}" y="${mid + 3.5}" text-anchor="end" style="font-size:10.5px">${esc(n.name)}</text>` : `<text x="${n.x1 + 5}" y="${mid + 3.5}" style="font-size:10px">${esc(n.name)}</text>`;
    else s += `<text x="${n.x1 + 6}" y="${mid + 4}" style="font-size:${ph ? 11.5 : 12}px;fill:var(--ink)">${esc(n.name)}</text><text x="${n.x1 + 6}" y="${mid + 17}">${n.value} models</text>`;
    s += '</g>';
  });
  const rx = (g.nodes.find(n => n.kind === 'r') || { x0: W - 170 }).x0, mx = (g.nodes.find(n => n.kind === 'm') || { x1: W / 2 }).x1, ty = H - 46;
  s += `<path d="M${mx},${ty - 22} C${(mx + rx) / 2},${ty - 22} ${(mx + rx) / 2},${ty + 13} ${rx},${ty + 13}" style="fill:none;stroke:var(--brand);stroke-dasharray:4 4;stroke-width:1.4"/>`;
  s += `<rect x="${rx}" y="${ty}" width="9" height="26" style="fill:none;stroke:var(--brand);stroke-dasharray:3 2"/><text x="${rx + 15}" y="${ty + 12}" style="font-family:var(--sans);font-size:12px;fill:var(--brand);font-weight:600">Training labels</text><text x="${rx + 15}" y="${ty + 25}" style="fill:var(--brand)">${ph ? '0 links' : '0 models · 0 links'}</text>`;
  s += '</svg>';
  body.innerHTML = s;
  const svg = $('svg', body);
  svg.addEventListener('mousemove', ev => {
    const li = ev.target.dataset?.l, mid = ev.target.dataset?.m;
    if (li !== undefined) { const l = g.links[+li]; const txt = l.role === 'backdrop' && l.m.approach.real_backdrop ? l.m.approach.real_backdrop : l.role ? ROLE[l.role] : 'Built on ' + PLAT[l.m.platform]; showTip(`<div class="k">${esc(l.m.name)}${l.role ? ' → ' + ROLE[l.role] : ''}</div><div>${esc(txt)}</div>`, ev); }
    else if (mid) { const m = BYID.get(mid); showTip(`<div class="t">${esc(m.name)}</div><div class="k">${[...m.roles].map(r => ROLE[r]).join(' · ')}</div>`, ev); }
    else hideTip();
  });
  svg.addEventListener('mouseleave', hideTip);
  svg.addEventListener('click', ev => { const mid = ev.target.dataset?.m; if (!mid) return; const m = BYID.get(mid); tapOpen(ev, 'sk:' + mid, () => `<div class="t">${esc(m.name)}</div><div class="k">${[...m.roles].map(r => ROLE[r]).join(' · ')}</div><div class="k" style="margin-top:4px">Tap again to open</div>`, () => openModel(mid, null, 'approach')); });
}
function drawRoleSummary(body) {
  const ms = modelsIn();
  const srcs = new Map();
  ms.forEach(m => m.data.forEach(d => { const k = d.url || d.name; if (!srcs.has(k)) srcs.set(k, d); }));
  body.innerHTML = `<div style="font:300 64px/1 var(--mono);color:var(--brand)">0</div><p style="margin:6px 0 12px;font-size:15px">real labelled examples trained any of these ${ms.length} models. Real data does five other jobs, and each model's card says which.</p>
    ${ROLES.map(r => { const n = ms.filter(m => m.roles.has(r)).length; return `<div style="margin-top:8px"><div style="display:flex;justify-content:space-between;font-size:12.5px"><span>${ROLE[r]}</span><span class="mono">${n}/${ms.length}</span></div><div class="tbar"><i style="width:${ms.length ? n / ms.length * 100 : 0}%;background:var(--ink-2)"></i></div></div>`; }).join('')}
    <p class="note">${srcs.size} distinct real data sources feed these models, each with its licence on the model's page. The rule comes from <a href="${REPO}/blob/main/docs/method.md" target="_blank" rel="noopener">docs/method.md</a>, step 3.</p>`;
}

/* ============================================================ lineage: grey-box matrix, prior wars */
function drawMatrix(body) {
  const ms = modelsIn().sort((a, b) => PLATS.indexOf(a.platform) - PLATS.indexOf(b.platform) || a.order - b.order);
  const cols = [['real_backdrop', 'Real noise backdrop'], ['physics_inputs', 'Physics in the inputs'], ['physics_anchor', 'Predicts a correction to physics'], ...ROLES.slice(1).map(r => ['role:' + r, ROLE[r]])];
  const val = (m, c) => c.startsWith('role:') ? (m.roles.has(c.slice(5)) ? ROLE[c.slice(5)] : false) : m.approach[c];
  body.innerHTML = `<div class="tscroll" style="max-height:640px"><table class="mx"><thead><tr><th style="text-align:left">Model</th>${cols.map(c => `<th>${c[1]}</th>`).join('')}</tr></thead><tbody>${ms.map(m => `<tr><td class="mname" data-m="${m.id}"><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:${m.accent};margin-right:6px"></span>${esc(m.name)} <span class="mono muted" style="font-size:9.5px">${m.platform === 'standalone' ? '' : esc(m.platform)}</span></td>${cols.map(c => { const v = val(m, c[0]); return `<td><span class="dot ${v ? '' : 'off'}" ${v ? `data-t="${esc(typeof v === 'string' ? v : '')}"` : ''}></span></td>`; }).join('')}</tr>`).join('')}</tbody></table></div><p class="note">From the <code>approach</code> block of each <code>model.yaml</code>. Hover a dot for the card's own words.</p>`;
  if (!body._wired) {
    body._wired = true;
    body.addEventListener('mousemove', ev => { const t = ev.target.dataset?.t; if (t) showTip(`<div>${esc(t)}</div>`, ev); else hideTip(); });
    body.addEventListener('mouseleave', hideTip);
  }
  $$('.mname', body).forEach(td => td.onclick = () => openModel(td.dataset.m, null, 'approach'));
}
function drawPrior(body) {
  const arms = { P: 'Physics-trained model', 'G-dev': 'Generic prior + real dev labels', 'G-sim': 'Generic prior + simulated rows', 'G-sim+dev': 'Generic prior + both' };
  const col = a => a === 'P' ? 'var(--win)' : a === 'G-dev' ? 'var(--loss)' : 'var(--tie)';
  const chip = a => `<span class="vpill" style="background:${col(a)}" title="${esc(arms[a] || a)}">${esc(a)}</span>`;
  const z = D.prior.filter(p => p.zero === 'P').length;
  body.innerHTML = `<div style="overflow-x:auto"><table class="rows"><thead><tr><th>Exam</th><th>Best</th><th>0 real labels</th><th>k real labels</th></tr></thead><tbody>${D.prior.map(p => `<tr><td>${esc(p.model)}</td><td>${chip(p.headline)}</td><td>${chip(p.zero)}</td><td>${chip(p.k)}</td></tr>`).join('')}</tbody></table></div>
  <p class="note">Prior Wars, a pre-registered method experiment: the released physics-trained model (P) against TabPFN, a generic synthetic prior, given real dev labels, simulated rows, or both. With zero real labels on both sides, the physics model is best on ${z} of ${D.prior.length}. Give the generic prior the dev labels and it takes ${D.prior.filter(p => p.headline !== 'P').length} headlines. That is the case for rolling real data in, measured. <a href="${REPO}/blob/main/docs/experiments/prior-wars.md" target="_blank" rel="noopener">Read the experiment</a>.</p>`;
}
function licClass(l) {
  const s = (l || '').toLowerCase();
  if (/public domain|u\.s\. government|us government|cc0|cc-0/.test(s)) return 'Public domain or CC0';
  if (/by-sa|odbl|share/.test(s)) return 'Share-alike';
  if (/cc by|cc-by|creative commons/.test(s)) return 'CC BY';
  if (/mit|bsd|apache|gpl/.test(s)) return 'Open-source licence';
  return 'Provider terms';
}
function drawSources(body) {
  const ms = modelsIn(), map = new Map();
  ms.forEach(m => m.data.forEach(d => { const k = (d.url || d.name).replace(/\/$/, ''); if (!map.has(k)) map.set(k, { ...d, models: [] }); map.get(k).models.push(m); }));
  const list = [...map.values()].sort((a, b) => b.models.length - a.models.length || a.name.localeCompare(b.name));
  const q = (body._q || '').toLowerCase();
  const shown = list.filter(d => !q || (d.name + d.licence + d.models.map(m => m.name).join(' ')).toLowerCase().includes(q));
  const lc = d3.rollups(list, v => v.length, d => licClass(d.licence)).sort((a, b) => b[1] - a[1]);
  body.innerHTML = `<div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:8px"><input data-q value="${esc(body._q || '')}" placeholder="Filter sources, licences or models" aria-label="Filter data sources" style="flex:1;min-width:200px;padding:6px 10px;border:1px solid var(--rule);border-radius:4px;background:var(--panel)"><div class="chips">${lc.map(([k, v]) => `<span class="chip static">${esc(k)} <span class="c">${v}</span></span>`).join('')}</div></div>
  <div class="tscroll" style="max-height:420px"><table class="rows"><thead><tr><th>Source</th><th>Licence</th><th>Used by</th></tr></thead><tbody>${shown.map(d => `<tr><td>${d.url ? `<a href="${esc(d.url)}" target="_blank" rel="noopener">${esc(d.name)}</a>` : esc(d.name)}</td><td style="font-size:12px">${esc(d.licence)}</td><td>${d.models.map(m => `<span class="chip static" style="margin:1px"><span class="sw" style="background:${m.accent}"></span>${esc(m.name)}</span>`).join('')}</td></tr>`).join('')}</tbody></table></div><p class="note">${shown.length} of ${list.length} distinct sources, merged by URL across model cards.</p>`;
  const inp = $('[data-q]', body);
  inp.oninput = () => { body._q = inp.value; const pos = inp.selectionStart; drawSources(body); const n = $('[data-q]', body); n.focus(); n.setSelectionRange(pos, pos); };
}

/* ============================================================ audits */
function drawBoard(body) {
  const ms = MODELS.filter(m => mPass(m));
  if (!S.auditSel || !ms.some(m => m.id === S.auditSel)) S.auditSel = (ms.find(m => m.auditCls === 'critical') || ms[0] || {}).id || null;
  body.innerHTML = `<div class="board">${auditTiles(ms)}</div><p class="note">Two independent reviewers audited each model with a ten-point forensic protocol (weights, normalisation, export parity, look-ahead, shortcuts, seeds, rung arithmetic, calibration). Click a model for its finding and what was done.</p>`;
  $$('.tile', body).forEach(b => b.onclick = () => { S.auditSel = b.dataset.m; $$('.tile', body).forEach(t => t.setAttribute('aria-pressed', t === b)); runCardId('detail'); });
}
function drawDetail(body) {
  const m = BYID.get(S.auditSel);
  if (!m) { body.innerHTML = '<div class="empty">Pick a model on the board.</div>'; return; }
  if (!m.audit) { body.innerHTML = `<div class="detail"><h4>${esc(m.name)}</h4>${apill('none')}<p>No audit report is registered for this model yet.</p></div>`; return; }
  const a = m.audit;
  body.innerHTML = `<div class="detail"><div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap"><h4 style="margin-right:6px">${esc(m.name)}</h4>${apill(m.auditCls)}<span class="mono muted">${esc(a.version)}</span></div>
    <div class="kick" style="margin-top:8px">Verdict</div><p style="margin:2px 0 8px;font-weight:600">${md(a.label)}</p>
    <div class="kick">Key finding</div><p style="margin:2px 0 10px;font-size:13px">${md(a.finding)}</p>
    ${a.examEffect ? `<div class="kick">Effect on the published exam</div><p style="margin:2px 0 10px;font-size:13px">${md(a.examEffect)}</p>` : ''}
    <div class="kick" style="margin-bottom:4px">Resolution, finding by finding</div>${a.resolutions.map(r => `<div class="act"><span class="vpill" style="background:${ACOL[r.action]}">${ACT[r.action]}</span><div><div>${md(r.finding)}</div><div class="muted" style="font-size:12px;margin-top:2px">${md(r.text.length > 260 ? r.text.slice(0, 258) + '…' : r.text)}</div></div></div>`).join('')}
    ${a.queued && a.queued !== '—' ? `<div class="kick" style="margin-top:10px">Queued for approval</div><p style="margin:2px 0;font-size:13px">${md(a.queued)}</p>` : ''}
    <p style="margin-top:12px"><a href="${REPO}/blob/main/docs/audits/${esc(a.report)}" target="_blank" rel="noopener">Read the full report</a> · <button class="btn sm" data-open="${m.id}">Open ${esc(m.name)}</button></p></div>`;
  $('[data-open]', body).onclick = () => openModel(m.id, null, 'audit');
}
function drawResolutions(body) {
  const ms = MODELS.filter(m => mPass(m) && m.audit);
  const ab = actionBar(ms);
  const rows = AORDER.slice(0, 4).map(c => { const g = ms.filter(m => m.auditCls === c); const rs = g.flatMap(m => m.audit.resolutions); const cc = d3.rollup(rs, v => v.length, r => r.action); return { c, g, rs, cc }; }).filter(x => x.g.length);
  body.innerHTML = `<div style="display:flex;gap:22px;flex-wrap:wrap"><div><div class="kick">Findings</div><div style="font:300 30px var(--mono)">${ab.n}</div></div><div><div class="kick">Fixed</div><div style="font:300 30px var(--mono);color:var(--win)">${ab.fixed}</div></div><div><div class="kick">Models audited</div><div style="font:300 30px var(--mono)">${ms.length}</div></div></div>${ab.html}
  <div style="overflow-x:auto"><table class="rows" style="margin-top:10px"><thead><tr><th>Severity</th><th>Models</th><th>Findings</th>${Object.keys(ACT).map(k => `<th style="text-align:right">${ACT[k]}</th>`).join('')}</tr></thead><tbody>${rows.map(x => `<tr><td>${apill(x.c)}</td><td class="n">${x.g.length}</td><td class="n">${x.rs.length}</td>${Object.keys(ACT).map(k => `<td class="n">${x.cc.get(k) || ''}</td>`).join('')}</tr>`).join('')}</tbody></table></div>
  <p class="note">From the Resolution section that ends every report. The rule the resolution pass used: a fix that cannot change a published exam number is on by default; one that could is shipped opt-in with a warning, and its re-score waits for the owner.</p>`;
}
function drawEffects(body) {
  const ms = MODELS.filter(m => mPass(m) && m.audit && m.audit.fixedNow);
  body.innerHTML = `<div class="tscroll" style="max-height:460px"><table class="rows"><thead><tr><th>Model</th><th>Fixed now</th><th>Effect on the published exam</th><th>Queued</th></tr></thead><tbody>${ms.map(m => `<tr data-m="${m.id}" style="cursor:pointer"><td style="white-space:nowrap"><span class="dot" style="background:var(--${m.auditCls});margin-right:6px;border-radius:2px"></span>${esc(m.name)}</td><td style="font-size:12px">${md(m.audit.fixedNow)}</td><td style="font-size:12px">${md(m.audit.examEffect)}</td><td style="font-size:12px" class="muted">${md(m.audit.queued)}</td></tr>`).join('')}</tbody></table></div>`;
  $$('tr[data-m]', body).forEach(tr => tr.onclick = () => { S.auditSel = tr.dataset.m; runCardId('board'); runCardId('detail'); document.getElementById('card-detail')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); });
}

/* ============================================================ StormNet, running in this page */
const STORM = (() => {
  const pack = (man, buf) => { const t = {}; for (const [k, d] of Object.entries(man.tensors)) t[k] = { data: new Float32Array(buf.slice(d.offset, d.offset + d.bytes)), shape: d.shape }; return { t, meta: man.meta }; };
  // the page loads StormNet's instrument files (site/public/instruments/stormnet/) before the app starts
  const W = pack(D.storm.weights, D.storm.weightsBin);
  const P = pack(D.storm.storms, D.storm.stormsBin);
  const REC = { bz: 0, speed: 1, pressure: 1 };
  function series(i) {
    const wind = P.t['wind' + i].data, dst = P.t['dst' + i].data, info = P.meta.storms[i], n = info.length;
    return { info, rows: Array.from({ length: n }, (_, j) => [wind[3 * j], wind[3 * j + 1], wind[3 * j + 2]]), dst: Array.from(dst) };
  }
  function windowAt(rows, k, s = REC) {
    if (k < 35) return null;
    const w = rows.slice(k - 35, k + 1).map((r, j) => j >= 33 ? [r[0] + s.bz, r[1] * s.speed, r[2] * s.pressure] : [r[0], r[1], r[2]]);
    return w.every(r => r.every(Number.isFinite) && r[1] > 0 && r[2] > 0) ? w : null;
  }
  function step(state, field, gain, scale, share) {
    const tau = scale * 2.4 * Math.exp(9.74 / (4.69 + field));
    return Math.exp(-1 / tau) * state - gain * share * Math.max(field - .49, 0) * tau * -Math.expm1(-1 / tau);
  }
  function projected(e, p, dst, params, law) {
    const [gain, tf, ts, share, b, c] = params;
    let fast = 0, slow = 0;
    for (const v of e) { fast = step(fast, v, gain, tf, share); slow = step(slow, v, gain, ts, 1 - share); }
    const total = fast + slow, fraction = Math.abs(total) > 1e-8 ? fast / total : share;
    let field = e[e.length - 1], pressure = p[p.length - 1];
    const corrected = dst - b * pressure + c;
    fast = fraction * corrected; slow = (1 - fraction) * corrected;
    const startFast = fast, startSlow = slow, path = [];
    for (let i = 0; i < 3; i++) {
      field = law[0] + law[1] * (field - law[0]);
      pressure = law[2] + law[3] * (pressure - law[2]);
      fast = step(fast, field, gain, tf, share); slow = step(slow, field, gain, ts, 1 - share);
      path.push(fast + slow + b * pressure - c);
    }
    return { path, total, fraction, minimum: Math.min(...path), fast: startFast, slow: startSlow };
  }
  const meanTail = (a, n) => a.slice(-n).reduce((s, v) => s + v, 0) / n;
  function predict(wind, dst) {
    const { meta, t } = W;
    const e = wind.map(([bz, sp]) => Math.max(-bz, 0) * sp / 1000), p = wind.map(r => Math.sqrt(r[2]));
    const x = [dst, e[35], p[35], wind[35][1] / 1000, wind[35][0]];
    let anchor = dst;
    const phys = projected(e, p, dst, meta.physics, meta.wind_law);
    if (meta.mode === 'history') { anchor = phys.minimum; for (const n of [3, 12, 36]) x.push(meanTail(e, n), meanTail(p, n)); x.push(phys.fraction, phys.total, anchor - dst, phys.path[2] - phys.path[0]); }
    const z = x.map((v, i) => Math.max(-8, Math.min(8, (v - t.mean.data[i]) / t.scale.data[i])));
    const h = [1, ...z], size = t.bias.data.length;
    for (let j = 0; j < size; j++) { let a = t.bias.data[j]; for (let i = 0; i < z.length; i++) a += z[i] * t.kernel.data[i * size + j]; h.push(Math.tanh(a)); }
    const dot = key => h.reduce((s, v, i) => s + v * t[key].data[i], 0);
    let sigma = Math.exp(Math.max(-1, Math.min(5, dot('spread'))));
    const by = meta.widen_by_dst;
    if (by) { let k = 0; while (k < by.edges.length && dst >= by.edges[k]) k++; sigma *= by.factors[k]; } else sigma *= meta.widen_factor;
    return { minimum: anchor + dot('location'), sigma, classical: projected(e, p, dst, meta.classical, meta.wind_law).minimum, single: projected(e, p, dst, meta.single, meta.wind_law).minimum, published: projected(e, p, dst, meta.published, meta.wind_law).minimum, persistence: dst, path: phys.path, fast: phys.fast, slow: phys.slow };
  }
  const cache = new Map();
  function run(i) {
    if (cache.has(i)) return cache.get(i);
    const s = series(i), out = [], t0 = performance.now();
    for (let k = 35; k + 3 < s.dst.length; k++) {
      const w = windowAt(s.rows, k), fut = s.dst.slice(k + 1, k + 4);
      if (!w || !Number.isFinite(s.dst[k]) || !fut.every(Number.isFinite)) continue;
      out.push({ ...predict(w, s.dst[k]), k, truth: Math.min(...fut) });
    }
    const r = { ...s, readings: out, ms: performance.now() - t0 };
    cache.set(i, r); return r;
  }
  function parity() {
    const t0 = performance.now(); let worst = 0; const cases = D.storm.reference.cases, sample = [];
    for (const c of cases) {
      const s = series(c.storm), w = windowAt(s.rows, c.k, c), r = predict(w, s.dst[c.k]);
      const out = [r.minimum, r.sigma, r.classical, r.single, r.published, r.fast, r.slow];
      let cw = 0; out.forEach((v, i) => { cw = Math.max(cw, Math.abs(v - c.outputs[i]) / Math.max(1, Math.abs(c.outputs[i]))); });
      worst = Math.max(worst, cw);
      if (sample.length < 6 && (c.variant.includes('Bz+0') || sample.length < 3)) sample.push({ c, out, cw });
    }
    return { n: cases.length, worst, tol: +D.storm.reference.tolerance, ms: performance.now() - t0, sample, bytes: D.storm.weightsBin.byteLength };
  }
  return { W, P, series, windowAt, predict, run, parity, storms: P.meta.storms };
})();
let PAR = null;
const SKEYS = [['truth', 'What happened (lowest Dst, next 3 h)', 'var(--ink)', ''], ['net', 'StormNet ± 68% band', 'var(--brand)', ''], ['classical', 'Two-state physics, dev-tuned', 'var(--sel)', '5 3'], ['single', 'One-reservoir physics, dev-tuned', 'var(--mixed)', '2 2'], ['published', "O'Brien–McPherron, published", 'var(--tie)', '6 2 1 2'], ['persistence', 'Persistence (Dst now)', 'var(--info)', '1 3']];
const stTime = (R, k) => new Date(Date.parse(R.info.start_utc) + k * 3600e3);
const nT = v => `${v < 0 ? '−' : ''}${Math.abs(v).toFixed(0)} nT`;
function liveK(R) {
  if (S.live.k === null || !R.readings.some(r => r.k === S.live.k)) S.live.k = R.readings.reduce((a, b) => b.truth < a.truth ? b : a).k;
  return R.readings.find(r => r.k === S.live.k);
}
function whatIf(R) {
  const L = S.live;
  if (!L.bz && L.speed === 1 && L.pressure === 1) return null;
  const w = STORM.windowAt(R.rows, L.k, L);
  return w ? STORM.predict(w, R.dst[L.k]) : null;
}
function drawStorm(body) {
  const L = S.live, R = STORM.run(L.storm), rd = R.readings, cur = liveK(R);
  const W = Math.max(340, bw(body)), ml = 46, mr = 12, h1 = 270, h2 = 62, gap = 22, H = h1 + gap + h2 + 22;
  const x = d3.scaleUtc([stTime(R, 0), stTime(R, R.dst.length - 1)], [ml, W - mr]);
  const vis = k => L.show[k];
  const lows = rd.flatMap(r => [r.truth, vis('net') ? r.minimum - r.sigma : 0, vis('classical') ? r.classical : 0, vis('published') ? r.published : 0, vis('single') ? r.single : 0]);
  const y = d3.scaleLinear([Math.min(d3.min(lows), d3.min(R.dst)) - 12, Math.max(25, d3.max(R.dst) + 8)], [h1, 8]).nice();
  const ln = key => d3.line(r => x(stTime(R, r.k)), r => y(r[key])).defined(r => Number.isFinite(r[key]));
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="StormNet forecasts across ${esc(R.info.label)}">`;
  y.ticks(6).forEach(t => { s += `<line x1="${ml}" x2="${W - mr}" y1="${y(t)}" y2="${y(t)}" style="stroke:${t === 0 ? 'var(--ink-3)' : 'var(--rule-2)'}"/><text x="${ml - 6}" y="${y(t) + 3}" text-anchor="end">${t}</text>`; });
  s += `<text transform="translate(9,${h1 / 2}) rotate(-90)" text-anchor="middle" style="fill:var(--ink-2)">Dst, nT</text>`;
  s += `<path d="${d3.line((d, i) => x(stTime(R, i)), d => y(d)).defined(Number.isFinite)(R.dst)}" style="fill:none;stroke:var(--ink-3);stroke-width:1;opacity:.6"/>`;
  if (vis('net')) s += `<path d="${d3.area(r => x(stTime(R, r.k)), r => y(r.minimum - r.sigma), r => y(r.minimum + r.sigma))(rd)}" style="fill:var(--brand);opacity:.14"/>`;
  for (const [k, , col, dash] of SKEYS) {
    if (!vis(k)) continue;
    const key = k === 'net' ? 'minimum' : k;
    s += `<path d="${ln(key)(rd)}" style="fill:none;stroke:${col};stroke-width:${k === 'truth' || k === 'net' ? 2 : 1.3};${dash ? `stroke-dasharray:${dash}` : ''}"/>`;
  }
  s += `<g class="scrub"><line class="sl" y1="6" y2="${h1 + gap + h2}" style="stroke:var(--ink);stroke-width:1;opacity:.55"/><circle class="st" r="4.5" style="fill:var(--ink)"/><circle class="sn" r="4.5" style="fill:var(--brand);stroke:var(--panel);stroke-width:1.5"/><path class="wd" style="fill:none;stroke:var(--brand);stroke-width:2"/><text class="wt" style="fill:var(--brand)"></text></g>`;
  // solar wind panel
  const y2t = h1 + gap, bz = R.rows.map(r => r[0]), bm = Math.max(10, d3.max(bz.map(Math.abs).filter(Number.isFinite)));
  const yb = d3.scaleLinear([-bm, bm], [y2t + h2, y2t]);
  const barW = Math.max(1, (W - ml - mr) / R.dst.length - .4);
  s += `<text x="${ml - 6}" y="${y2t + 8}" text-anchor="end">Bz</text><line x1="${ml}" x2="${W - mr}" y1="${yb(0)}" y2="${yb(0)}" style="stroke:var(--rule)"/>`;
  bz.forEach((v, i) => { if (!Number.isFinite(v)) return; const yy = yb(Math.max(0, v)), hh = Math.abs(yb(v) - yb(0)); s += `<rect x="${x(stTime(R, i)) - barW / 2}" y="${yy}" width="${barW}" height="${Math.max(.5, hh)}" style="fill:${v < 0 ? 'var(--brand)' : 'var(--ink-3)'};opacity:${v < 0 ? .8 : .45}"/>`; });
  s += `<text x="${W - mr}" y="${y2t + h2 + 12}" text-anchor="end">${MOBILE() ? 'Bz, nT (red = south)' : 'solar wind Bz, nT (south, red, drives storms)'}</text>`;
  const dayPx = x(stTime(R, 24)) - x(stTime(R, 0)), dayStep = Math.max(1, Math.ceil(58 / dayPx));
  d3.utcDay.range(d3.utcDay.ceil(x.domain()[0]), x.domain()[1]).filter((d, i) => i % dayStep === 0).forEach(d => { s += `<text x="${x(d)}" y="${h1 + 14}" text-anchor="middle">${d3.utcFormat('%b %-d')(d)}</text>`; });
  s += `<rect class="ov" x="${ml}" y="0" width="${W - ml - mr}" height="${H - 18}" style="fill:transparent;cursor:ew-resize;touch-action:pan-y"/></svg>`;
  const chips = STORM.storms.map((st, i) => `<button class="chip" data-st="${i}" aria-pressed="${i === L.storm}">${esc(st.label)} <span class="c">${st.minimum} nT</span></button>`).join('');
  const leg = SKEYS.map(([k, lab, col, dash]) => `<button data-sk="${k}" aria-pressed="${!!L.show[k]}"><i style="background:${dash ? `repeating-linear-gradient(90deg,${col} 0 4px,transparent 4px 6px)` : col}"></i>${lab}</button>`).join('');
  body.innerHTML = `<div class="storm-chips">${chips}</div><div class="legend">${leg}</div>${s}<p class="note">Each point is a forecast issued at that hour from the past 36 hours of real OMNI solar wind and the current Dst, for the lowest Dst in the next three hours. ${rd.length} forecasts computed in your browser in ${R.ms.toFixed(1)} ms. Click or drag across the chart to pick an issue hour.</p>`;
  body._sc = { x, y, R };
  placeScrub(body);
  $$('[data-st]', body).forEach(b => b.onclick = () => { L.storm = +b.dataset.st; L.k = null; runCardId('storm'); runCardId('side'); runCardId('mae'); });
  $$('[data-sk]', body).forEach(b => b.onclick = () => { L.show[b.dataset.sk] = !L.show[b.dataset.sk]; runCardId('storm'); });
  const ov = $('.ov', body), svg = $('svg', body);
  const pick = ev => { const px = d3.pointer(ev, svg)[0], t = x.invert(px); let best = rd[0]; for (const r of rd) if (Math.abs(stTime(R, r.k) - t) < Math.abs(stTime(R, best.k) - t)) best = r; return best; };
  let drag = false;
  ov.addEventListener('pointerdown', ev => { drag = true; try { ov.setPointerCapture(ev.pointerId); } catch (e) { } const r = pick(ev); if (r.k !== L.k) { L.k = r.k; placeScrub(body); runCardId('side'); } });
  ov.addEventListener('pointermove', ev => {
    const r = pick(ev);
    if (drag && r.k !== L.k) { L.k = r.k; placeScrub(body); runCardId('side'); }
    showTip(`<div class="k">${d3.utcFormat('%a %-d %b %Y, %H:00 UTC')(stTime(R, r.k))}</div><div class="row"><span>StormNet</span><b>${nT(r.minimum)} ± ${r.sigma.toFixed(0)}</b></div><div class="row"><span>What happened</span><b>${nT(r.truth)}</b></div><div class="row"><span>Two-state physics</span><span>${nT(r.classical)}</span></div><div class="row"><span>Persistence</span><span>${nT(r.persistence)}</span></div>`, ev);
  });
  ov.addEventListener('pointerup', () => { drag = false; if (TOUCH) hideTip(); });
  ov.addEventListener('pointerleave', hideTip);
}
function placeScrub(body) {
  const sc = body && body._sc; if (!sc) return;
  const { x, y, R } = sc, cur = liveK(R), cx = x(stTime(R, cur.k)), g = $('.scrub', body);
  $('.sl', g).setAttribute('x1', cx); $('.sl', g).setAttribute('x2', cx);
  $('.st', g).setAttribute('cx', cx); $('.st', g).setAttribute('cy', y(cur.truth));
  $('.sn', g).setAttribute('cx', cx); $('.sn', g).setAttribute('cy', y(cur.minimum));
  const wi = whatIf(R);
  $('.wd', g).setAttribute('d', wi ? `M${cx},${y(wi.minimum) - 7}l7,7l-7,7l-7,-7z` : '');
  const t = $('.wt', g); t.textContent = wi ? `what-if ${nT(wi.minimum)}` : '';
  if (wi) { const right = cx < x.range()[1] - 110; t.setAttribute('x', right ? cx + 11 : cx - 11); t.setAttribute('text-anchor', right ? 'start' : 'end'); t.setAttribute('y', y(wi.minimum) + 4); }
}
const stormBody = () => CARDS.find(c => c.id === 'storm')?._b;
function drawSide(body) {
  const L = S.live, R = STORM.run(L.storm), r = liveK(R);
  const inside = Math.abs(r.truth - r.minimum) <= r.sigma;
  const other = [['classical', 'Two-state physics, dev-tuned'], ['single', 'One-reservoir physics'], ['published', "O'Brien–McPherron, published"], ['persistence', 'Persistence']];
  body.innerHTML = `<div class="live-side"><div class="kick">Issue time · ${esc(R.info.label)}</div>
    <div style="display:flex;align-items:center;gap:8px;margin:3px 0 10px"><b>${d3.utcFormat('%a %-d %b %Y, %H:00 UTC')(stTime(R, r.k))}</b><span style="margin-left:auto" class="seg"><button data-step="-1" aria-label="Previous hour">◀</button><button data-step="1" aria-label="Next hour">▶</button></span></div>
    <div class="kick">StormNet: lowest Dst in the next 3 hours</div><div class="big" style="color:var(--brand)">${nT(r.minimum)}</div><div class="mono muted">± ${r.sigma.toFixed(1)} nT (68% band)</div>
    <div style="margin-top:8px;font-size:13px">What happened: <b>${nT(r.truth)}</b> <span class="${inside ? 'ok' : 'bad'}">${inside ? 'inside the band' : 'outside the band'}</span></div>
    <table class="rows" style="margin-top:8px"><tbody>${other.map(([k, l]) => `<tr><td>${l}</td><td class="n">${nT(r[k])}</td><td class="n muted">${Math.abs(r[k] - r.truth).toFixed(0)} off</td></tr>`).join('')}<tr><td><b>StormNet</b></td><td class="n"><b>${nT(r.minimum)}</b></td><td class="n muted">${Math.abs(r.minimum - r.truth).toFixed(0)} off</td></tr></tbody></table>
    <div class="kick" style="margin-top:14px">What if the last 3 hours of wind had been different?</div>
    <label for="wiBz">Bz shift: <b data-o="bz"></b></label><input id="wiBz" type="range" min="-25" max="25" step="1" value="${L.bz}">
    <label for="wiSp">Speed: <b data-o="speed"></b></label><input id="wiSp" type="range" min="0.5" max="2" step="0.05" value="${L.speed}">
    <label for="wiPr">Dynamic pressure: <b data-o="pressure"></b></label><input id="wiPr" type="range" min="0.5" max="3" step="0.1" value="${L.pressure}">
    <div data-wi style="margin-top:10px;font-size:13px"></div><button class="btn sm" data-reset style="margin-top:8px">Back to the recorded wind</button></div>`;
  const upd = () => {
    $('[data-o="bz"]', body).textContent = `${L.bz > 0 ? '+' : ''}${L.bz} nT${L.bz < 0 ? ' (more southward)' : ''}`;
    $('[data-o="speed"]', body).textContent = `× ${(+L.speed).toFixed(2)}`;
    $('[data-o="pressure"]', body).textContent = `× ${(+L.pressure).toFixed(1)}`;
    const wi = whatIf(R);
    $('[data-wi]', body).innerHTML = wi ? `StormNet with this wind: <b style="color:var(--brand)">${nT(wi.minimum)}</b> ± ${wi.sigma.toFixed(0)}, ${wi.minimum < r.minimum ? 'deeper' : 'shallower'} by ${Math.abs(wi.minimum - r.minimum).toFixed(0)} nT. Two-state physics: ${nT(wi.classical)}.` : '<span class="muted">Recorded wind. Move a slider and the network runs again.</span>';
  };
  [['#wiBz', 'bz'], ['#wiSp', 'speed'], ['#wiPr', 'pressure']].forEach(([sel, k]) => { $(sel, body).addEventListener('input', e => { L[k] = +e.target.value; upd(); placeScrub(stormBody()); }); });
  $('[data-reset]', body).onclick = () => { L.bz = 0; L.speed = 1; L.pressure = 1; runCardId('side'); placeScrub(stormBody()); };
  $$('[data-step]', body).forEach(b => b.onclick = () => { const i = R.readings.findIndex(x => x.k === L.k) + +b.dataset.step; if (R.readings[i]) { L.k = R.readings[i].k; runCardId('side'); placeScrub(stormBody()); } });
  upd();
}
function drawMae(body) {
  const R = STORM.run(S.live.storm), rd = R.readings;
  const f = [['minimum', 'StormNet', 'var(--brand)'], ['classical', 'Two-state physics, dev-tuned', 'var(--sel)'], ['single', 'One-reservoir physics', 'var(--mixed)'], ['published', "O'Brien–McPherron, published", 'var(--tie)'], ['persistence', 'Persistence', 'var(--info)']];
  const mae = f.map(([k, l, c]) => ({ l, c, v: d3.mean(rd, r => Math.abs(r[k] - r.truth)) })).sort((a, b) => a.v - b.v);
  const mx = d3.max(mae, d => d.v), cov = rd.filter(r => Math.abs(r.truth - r.minimum) <= r.sigma).length / rd.length;
  body.innerHTML = `${mae.map(d => `<div style="display:grid;grid-template-columns:200px 1fr 64px;gap:8px;align-items:center;font-size:12.5px;margin:6px 0"><span>${d.l}</span><div class="tbar" style="height:12px"><i style="width:${d.v / mx * 100}%;background:${d.c}"></i></div><span class="mono" style="text-align:right">${d.v.toFixed(1)} nT</span></div>`).join('')}
  <p style="margin:10px 0 0;font-size:13px">StormNet's 68% band held the truth ${pct(cov)} of the time on this storm (${rd.length} forecasts).</p>
  <p class="note">Mean absolute error over every eligible issue hour of ${esc(R.info.label)}, computed live from the forecasts on the left. One storm is an anecdote; the graded exam is on the model page.</p>`;
}
function drawParity(body) {
  const p = PAR, ok = p.worst <= p.tol;
  body.innerHTML = `<div style="display:flex;gap:22px;flex-wrap:wrap;align-items:baseline"><div><div class="kick">Reference cases</div><div style="font:300 30px var(--mono)">${p.n}</div></div><div><div class="kick">Worst relative difference</div><div style="font:300 30px var(--mono)" class="${ok ? 'ok' : 'bad'}">${p.worst.toExponential(1)}</div></div><div><div class="kick">Tolerance</div><div style="font:300 30px var(--mono)">${p.tol.toExponential(0)}</div></div></div>
  <p style="font-size:13px">${ok ? 'Pass.' : 'Fail.'} When this page opened it decoded StormNet's ${fmt(p.bytes)}-byte weight pack, ran the ${p.n} cases the site's test suite uses (recorded and what-if winds across six storms) in ${p.ms.toFixed(0)} ms, and compared all seven outputs with Python's.</p>
  <div class="xscroll" style="overflow-x:auto"><table class="rows"><thead><tr><th>Case</th><th style="text-align:right">Python</th><th style="text-align:right">This page</th><th style="text-align:right">Rel. diff</th></tr></thead><tbody>${p.sample.map(x => `<tr><td class="mono" style="font-size:10.5px">storm ${x.c.storm} · ${esc(x.c.variant)}</td><td class="n">${x.c.outputs[0].toFixed(4)}</td><td class="n">${x.out[0].toFixed(4)}</td><td class="n">${x.cw.toExponential(1)}</td></tr>`).join('')}</tbody></table></div>
  <p class="note">A BI tool reports what a model scored. This page carries the model: a 15 → 48 random-feature layer with two linear heads, a two-state ring-current physics anchor, and its three physics rivals. Source: <a href="${REPO}/tree/main/models/stormnet" target="_blank" rel="noopener">models/stormnet</a>.</p>`;
}

/* ============================================================ the model page (drill-through) */
function rowLine(r, W) {
  const vals = [{ v: r.model, me: true, n: r.m.name }, ...r.baselines.filter(b => typeof b.value === 'number').map(b => ({ v: b.value, n: b.name, k: b.kind }))];
  if (typeof r.model !== 'number' || vals.length < 2) return '';
  const xs = vals.map(d => d.v), lo = d3.min(xs), hi = d3.max(xs);
  const log = lo > 0 && hi / lo > 4;
  const pad = 26;
  const x = log ? d3.scaleLog([lo / 1.15, hi * 1.15], [pad, W - pad]) : d3.scaleLinear([lo - (hi - lo || Math.abs(lo) || 1) * .12, hi + (hi - lo || Math.abs(hi) || 1) * .12], [pad, W - pad]);
  let s = `<svg viewBox="0 0 ${W} 34" preserveAspectRatio="none"><line x1="${pad}" x2="${W - pad}" y1="17" y2="17" style="stroke:var(--rule)"/>`;
  vals.filter(d => !d.me).forEach(d => { s += `<circle cx="${x(d.v)}" cy="17" r="4.5" style="fill:var(--panel);stroke:${d.k === 'dumb' ? 'var(--ink-3)' : 'var(--ink-2)'};stroke-width:1.5;${d.k === 'dumb' ? 'stroke-dasharray:2 1.5' : ''}"><title>${esc(d.n)} (${d.k}): ${d.v}</title></circle>`; });
  const me = vals[0];
  s += `<circle cx="${x(me.v)}" cy="17" r="6" style="fill:${VC(r.verdict === 'info' ? 'info' : r.verdict)};stroke:var(--panel);stroke-width:1.5"><title>${esc(me.n)}: ${me.v}</title></circle>`;
  const dir = r.better === 'lower' ? '← better' : r.better === 'higher' ? 'better →' : 'closer is better';
  s += `<text x="${r.better === 'lower' ? pad : W - pad}" y="32" text-anchor="${r.better === 'lower' ? 'start' : 'end'}" style="font-size:9px">${dir}${log ? ' · log scale' : ''}</text></svg>`;
  return s;
}
function drawExamRows(body) {
  const m = BYID.get(S.page.slice(6)), W = Math.max(260, bw(body));
  body.innerHTML = `<p style="margin:0 0 6px;font-size:13px;color:var(--ink-2)">${md(m.exam.summary)}</p>` + m.rows.map(r => `<div class="erow ${S.focusRow === r.key ? 'focus' : ''}" id="row-${r.i}">
    <div class="em"><span>${esc(r.metric)}</span>${vpill(r.verdict)}</div>${rowLine(r, W)}
    <div class="ev">${esc(m.name)} <b>${fmtVal(r.model, r.unit)}</b>${r.baselines.map(b => ` · ${esc(b.name)} <span class="muted">(${b.kind})</span> ${fmtVal(b.value, r.unit)}`).join('')}</div>
    ${r.note ? `<div class="en">${md(r.note)}</div>` : ''}</div>`).join('');
  if (S.focusRow && !body._scrolled) { body._scrolled = true; const el = $('.erow.focus', body); if (el) setTimeout(() => el.scrollIntoView({ behavior: 'smooth', block: 'center' }), 120); }
}
function drawModelLadders(body) {
  const m = BYID.get(S.page.slice(6)), W = Math.max(260, bw(body));
  body.innerHTML = m.ladders.map((L, li) => {
    const hh = L.rungs.some(r => typeof r.sim === 'number') ? 130 : 24, g = ladderSvg(L, W, hh, true);
    return `<div style="margin-bottom:14px"><div style="font-weight:600;font-size:13px">${esc(L.name)}</div><div class="muted" style="font-size:12px">${esc(L.rule)}</div><svg data-li="${li}" viewBox="0 0 ${W} ${hh}" style="width:100%;height:${hh}px;display:block;margin-top:4px">${g.s}</svg></div>`;
  }).join('') + `<details><summary style="cursor:pointer;font-size:12.5px">Every rung and its reason (${m.nRungs})</summary>${m.ladders.map(L => L.rungs.map(r => `<div class="act"><span class="vpill" style="background:${r.decision === 'rejected' ? 'var(--ink-3)' : r.decision === 'released' ? 'var(--brand)' : 'var(--win)'}">${r.decision}</span><div><b class="mono">${esc(r.id)}</b> ${esc(r.change)}<div class="muted" style="font-size:12px">${typeof r.sim === 'number' ? 'frozen set ' + r.sim + ' · ' : ''}${esc(r.why)}</div></div></div>`).join('')).join('')}</details>`;
  $$('svg[data-li]', body).forEach(svg => {
    const L = m.ladders[+svg.dataset.li];
    svg.addEventListener('mousemove', ev => { const i = ev.target.dataset?.i; if (i === undefined) return hideTip(); const r = L.rungs[+i]; showTip(`<div class="k">${esc(r.id)} · ${r.decision}${typeof r.sim === 'number' ? ' · ' + r.sim : ''}</div><div class="t">${esc(r.change)}</div><div style="margin-top:4px;opacity:.85">${esc(r.why)}</div>`, ev); });
    svg.addEventListener('mouseleave', hideTip);
  });
}
function drawModelCoverage(body) {
  const m = BYID.get(S.page.slice(6)), c = m.exam.coverage;
  body.innerHTML = c.length ? c.map(x => `<div style="margin:8px 0"><div style="display:flex;justify-content:space-between;gap:10px;font-size:12.5px"><span>${esc(x.output)}</span><b class="mono">${x.actual}%</b></div><div style="position:relative;height:10px;background:var(--rule-2);border-radius:2px"><i style="position:absolute;left:0;top:0;bottom:0;width:${x.actual}%;background:${COVC[covBand(x.actual)]};border-radius:2px"></i><i style="position:absolute;left:68%;top:-3px;bottom:-3px;width:2px;background:var(--ink)"></i></div></div>`).join('') + '<p class="note">Share of exam truths inside the 68% band (black tick).</p>' : '<p class="muted" style="margin:0">This card reports no coverage of 68% bars on its exam.</p>';
}
function drawModelAudit(body) {
  S.auditSel = S.page.slice(6); drawDetail(body);
  const b = $('[data-open]', body); if (b) b.remove();
}
function drawModelApproach(body) {
  const m = BYID.get(S.page.slice(6)), a = m.approach;
  const row = (l, v) => `<div class="act"><span class="dot ${v ? '' : 'off'}" style="margin-top:5px;flex:none"></span><div><b>${l}</b><div class="muted" style="font-size:12px">${v ? esc(v) : 'No'}</div></div></div>`;
  body.innerHTML = row('Real noise backdrop', a.real_backdrop) + row('Physics in the inputs', a.physics_inputs) + row('Predicts a correction to physics', a.physics_anchor) +
    `<div class="kick" style="margin-top:10px">Jobs real data does here</div><div class="chips" style="margin-top:4px">${ROLES.map(r => `<span class="chip static" style="${m.roles.has(r) ? '' : 'opacity:.35;text-decoration:line-through'}">${ROLE[r]}</span>`).join('')}<span class="chip static" style="opacity:.35;text-decoration:line-through">Training labels</span></div>
    ${m.noise ? `<div class="kick" style="margin-top:10px">Noise</div><p style="margin:2px 0;font-size:12.5px">${esc(m.noise)}</p>` : ''}<div class="kick" style="margin-top:10px">Network</div><p style="margin:2px 0;font-size:12.5px">${esc(m.network)}</p>`;
}
function drawModelLists(body) {
  const m = BYID.get(S.page.slice(6));
  const list = (t, xs) => xs.length ? `<div class="kick" style="margin:10px 0 4px">${t}</div><ul class="plain">${xs.map(x => `<li>${md(x)}</li>`).join('')}</ul>` : '';
  body.innerHTML = list('Where it wins', m.wins) + list('Where it loses', m.losses) + list('Limits', m.limits) + list('Help wanted', m.help) + list('Next rungs worth trying', m.next);
}
function drawModelData(body) {
  const m = BYID.get(S.page.slice(6));
  body.innerHTML = m.data.map(d => `<div class="src"><a href="${esc(d.url)}" target="_blank" rel="noopener">${esc(d.name)}</a><div class="lic">${esc(d.licence)}</div><div style="font-size:12px;color:var(--ink-2)">${esc(d.role)}</div></div>`).join('') +
    `<div class="kick" style="margin:12px 0 4px">Split</div><p style="margin:0;font-size:12.5px"><b>Dev:</b> ${esc(m.split.dev)}</p><p style="margin:4px 0 0;font-size:12.5px"><b>Exam:</b> ${esc(m.split.exam)}</p>`;
}
function drawModelPhysics(body) {
  const m = BYID.get(S.page.slice(6));
  body.innerHTML = `<p style="margin:0 0 8px;font-size:12.5px">${md(m.physics.summary)}</p><div class="kick">Checked against independent references</div><ul class="plain">${m.physics.checks.map(c => `<li>${md(c)}</li>`).join('')}</ul>`;
}
function modelPage(id) {
  const m = BYID.get(id);
  if (!m) return null;
  const idx = MODELS.indexOf(m), prev = MODELS[(idx - 1 + MODELS.length) % MODELS.length], next = MODELS[(idx + 1) % MODELS.length];
  const t = m.tally;
  return {
    columns: true,
    head: () => `<div class="mhead" style="--acc:${m.accent}"><div class="kick">${esc(m.domain)} · ${esc(PLAT[m.platform])} · v${esc(m.version)} · ${esc(m.status)}${m.released ? ' · released ' + esc(m.released) : ''}</div><h2>${esc(m.name)}</h2><p class="q">${esc(m.question.replace(/\*\(([^)]+)\)\*/, '($1)'))}</p>
      <div class="meta">${GRADED.filter(v => t[v]).map(v => `<span class="vpill v-${v}">${t[v]} ${VLABEL[v]}${t[v] > 1 && v !== 'mixed' ? (v === 'loss' ? 'es' : 's') : ''}</span>`).join('')}${t.info ? `<span class="vpill v-info">${t.info} context</span>` : ''}${apill(m.auditCls)}<span class="mono muted">${m.nRungs} rungs · ${m.nRej} rejected${m.params ? ' · ' + fmt(m.params) + ' parameters' : ''}</span></div>
      <div class="meta"><button class="btn sm" data-back>← ${esc(PAGES.find(p => p.id === S.prevPage)?.label || 'Overview')}</button><button class="btn sm" data-go="${prev.id}">‹ ${esc(prev.name)}</button><button class="btn sm" data-go="${next.id}">${esc(next.name)} ›</button>
      <a class="btn sm" href="${REPO}/tree/main/models/${m.id}" target="_blank" rel="noopener">Folder on GitHub</a><a class="btn sm" href="${REPO}/blob/main/models/${m.id}/MODEL_CARD.md" target="_blank" rel="noopener">Model card</a><a class="btn sm" href="${SITE}/models/${m.id}/" target="_blank" rel="noopener">Site page${m.instrument ? ' (runs live)' : ''}</a>${m.id === 'stormnet' ? '<button class="btn sm primary" data-live>Run it here</button>' : ''}</div>
      ${m.disclaimer ? `<p class="note" style="color:var(--loss)">${esc(m.disclaimer)}</p>` : ''}<details style="margin-top:10px;max-width:90ch"><summary style="cursor:pointer;font-size:13px">Summary from the card</summary><p style="font-size:13px;color:var(--ink-2)">${md(m.summary)}</p></details></div>`,
    after: () => {
      $('[data-back]')?.addEventListener('click', () => setPage(S.prevPage || 'overview'));
      $$('[data-go]').forEach(b => b.onclick = () => openModel(b.dataset.go));
      $('[data-live]')?.addEventListener('click', () => setPage('live'));
      if (S.focusSection) { const el = document.getElementById('card-m-' + S.focusSection); if (el) setTimeout(() => { el.scrollIntoView({ behavior: 'smooth', block: 'start' }); el.classList.add('pulse'); }, 200); }
    },
    cards: [
      { id: 'm-exam', col: 'l', title: 'Exam rows, against every baseline', sub: `${m.rows.length} rows · ${m.exam.short || ''}`, draw: drawExamRows },
      { id: 'm-ratchet', col: 'r', title: 'The ratchet', sub: `${m.ladders.length} ladder${m.ladders.length > 1 ? 's' : ''} · ${m.nRungs} rungs`, draw: drawModelLadders },
      { id: 'm-coverage', col: 'r', title: 'Error bars on the exam', sub: '', draw: drawModelCoverage },
      { id: 'm-audit', col: 'r', title: 'Audit', sub: '', draw: drawModelAudit },
      { id: 'm-approach', col: 'r', title: 'Where physics and real data enter', sub: '', draw: drawModelApproach },
      { id: 'm-lists', col: 'l', title: 'Wins, losses, limits and what is next', sub: 'from the card, in its own words', draw: drawModelLists },
      { id: 'm-physics', col: 'l', title: 'Physics', sub: '', draw: drawModelPhysics },
      { id: 'm-data', col: 'l', title: 'Real data and the split', sub: `${m.data.length} sources`, draw: drawModelData },
    ],
  };
}

/* ============================================================ pages */
const segBtns = (name, opts, cur) => `<span class="seg" data-seg="${name}">${opts.map(([v, l]) => `<button data-v="${v}" aria-pressed="${v === cur}">${l}</button>`).join('')}</span>`;
const PAGES = [
  { id: 'overview', label: 'Overview', kick: 'The collection at a glance', title: 'Trained on physics, graded on reality', lede: 'Every model card, ratchet log and audit in the repository, read at build time. Click any mark to filter every view; the pane on the right shows what is filtered.', count: () => modelsIn().length + ' models',
    cards: [
      { id: 'kpis', cls: 'c12', title: 'The ledger', sub: () => anyFilter() ? 'Filtered. Sparklines show the whole collection, commit by commit.' : 'Sparklines: the whole collection, commit by commit, from git history', draw: drawKpis },
      { id: 'swarm', cls: 'c8', title: 'Every exam row, against its best baseline', sub: () => `${SROWS.filter(r => rPass(r)).length} rows lit. ${TOUCH ? 'Tap a dot for the numbers; tap again to open the model.' : 'Hover for the numbers; click to open the model.'}`, acts: () => segBtns('group', [['none', 'One field'], ['platform', 'By simulator'], ['domain', 'By domain'], ['verdict', 'By verdict']], S.swarmGroup), draw: drawSwarm,
        wire: el => $('[data-seg="group"]', el).addEventListener('click', ev => { const v = ev.target.dataset.v; if (!v) return; S.swarmGroup = v; $$('[data-seg="group"] button', el).forEach(b => b.setAttribute('aria-pressed', b.dataset.v === v)); runCardId('swarm'); }) },
      { id: 'domains', cls: 'c4', title: 'Verdicts by domain', sub: 'Click a domain to filter; click a colour for that verdict in it. Shift adds.', draw: drawDomains },
      { id: 'growth', cls: 'c7', title: 'Fifteen days of the collection', sub: () => `${SER[0].t.slice(0, 10)} to ${SER[SER.length - 1].t.slice(0, 10)} · hover for the commit`, draw: drawGrowth },
      { id: 'coverage', cls: 'c5', title: 'Are the error bars honest?', sub: 'Exam coverage of each output’s 68% band', draw: drawCoverage },
      { id: 'auditstrip', cls: 'c7', title: 'Audited by two reviewers each', sub: 'Click a model for its finding', draw: drawAuditStrip },
      { id: 'pipeline', cls: 'c5', title: 'What comes next', sub: () => `${D.backlog.length} ideas in the backlog`, draw: drawPipeline },
    ] },
  { id: 'exams', label: 'Exams', kick: 'Graded once, on real data that chose nothing', title: 'Every exam row, and why it went that way', lede: 'A row is a win only if the model beats every baseline on it, the dumb one included. Ties sit inside the noise. Context rows report numbers without a verdict.', count: () => fmt(stats().graded) + ' rows',
    cards: [
      { id: 'wall', cls: 'c12', title: 'The wall', sub: 'One square per exam row, grouped by model. Lit squares match the filters.', acts: () => segBtns('sort', [['win', 'Win rate'], ['domain', 'Domain'], ['release', 'Release'], ['rows', 'Size']], S.wallSort), draw: drawWall,
        wire: el => $('[data-seg="sort"]', el).addEventListener('click', ev => { const v = ev.target.dataset.v; if (!v) return; S.wallSort = v; $$('[data-seg="sort"] button', el).forEach(b => b.setAttribute('aria-pressed', b.dataset.v === v)); runCardId('wall'); }) },
      { id: 'tree', cls: 'c7', title: 'Decomposition tree', sub: 'Split the graded rows by any factor, then split again', draw: drawTree },
      { id: 'drivers', cls: 'c5', title: 'Where the wins cluster', sub: 'Win rate by factor, against the overall rate', draw: drawDrivers },
      { id: 'table', cls: 'c12', title: 'The rows', sub: () => `${tableRows().length} rows under the filters`, acts: () => `<label class="mono" style="display:inline-flex;gap:5px;align-items:center;margin-right:6px"><input type="checkbox" data-ctx ${S.table.ctx ? 'checked' : ''}> context rows</label><button class="btn sm" data-csv>Copy CSV</button>`, draw: drawTable,
        wire: el => { $('[data-csv]', el).onclick = () => copyText(csvRows(), `${tableRows().length} rows as CSV`); $('[data-ctx]', el).onchange = e => { S.table.ctx = e.target.checked; runCardId('table'); }; } },
    ] },
  { id: 'ratchet', label: 'Ratchet', kick: 'Kept only when it beats the frozen set and the real dev gate', title: 'The ratchet says no more often than yes', lede: 'Every change to a model is a rung. It survives only if it beats the incumbent on a frozen simulated set by more than a margin and does not get worse on a handful of real dev units. Every decision is logged.', count: () => fmt(stats().rungs) + ' rungs',
    cards: [
      { id: 'anatomy', cls: 'c5', title: 'Anatomy of a no', sub: 'Why rungs were rejected', draw: drawAnatomy },
      { id: 'strict', cls: 'c7', title: 'Does a stricter ratchet win more exams?', sub: 'Rejection rate against exam win rate', draw: drawStrict },
      { id: 'ladders', cls: 'c12', title: 'Every ladder', sub: 'Sorted by rung decisions. Click for every rung and its reason.', draw: drawLadders },
    ] },
  { id: 'lineage', label: 'Lineage', kick: 'Where real data goes, and where it never goes', title: 'Zero real training labels, five other jobs', lede: 'Real data sits under the simulations as noise, calibrates the simulator on dev, gates each rung, widens the error bars, and grades the final exam. It is never a training label.', count: () => 'roles',
    cards: [
      { id: 'sankey', cls: 'c8', title: 'Simulator → model → what real data does', sub: 'Hover a band; click a model', draw: drawSankey },
      { id: 'roles', cls: 'c4', title: 'The rule', sub: '', draw: drawRoleSummary },
      { id: 'matrix', cls: 'c7', title: 'Grey box, model by model', sub: 'Where the physics sits and what real data touches', draw: drawMatrix },
      { id: 'prior', cls: 'c5', title: 'When the physics prior meets a generic one', sub: 'Prior Wars, 6 real exams', draw: drawPrior },
      { id: 'sources', cls: 'c12', title: 'Every real data source', sub: 'Merged across model cards, with licence and users', draw: drawSources },
    ] },
  { id: 'audits', label: 'Audits', kick: 'Forensic, read-only, two reviewers per model', title: 'Found in the open, fixed in the open', lede: 'Each released model was audited for silent bugs that pass the tests: normalisation, export parity, look-ahead, exam shortcuts, seeds, rung arithmetic, calibration. The reports, and what was done about each finding, are in the repository.', count: () => MODELS.filter(m => m.audit && mPass(m)).length + ' audited',
    cards: [
      { id: 'board', cls: 'c7', title: 'Severity board', sub: () => `${MODELS.filter(m => m.audit && mPass(m)).length} reports`, draw: drawBoard },
      { id: 'detail', cls: 'c5', title: 'Finding', sub: '', draw: drawDetail },
      { id: 'resolutions', cls: 'c6', title: 'What was done', sub: 'Findings by resolution', draw: drawResolutions },
      { id: 'effects', cls: 'c6', title: 'Effect on the published exams', sub: 'From the resolution pass, 27 Nov 2026', draw: drawEffects },
    ] },
  { id: 'live', label: 'Live', kick: 'Not a report about a model: the model', title: 'StormNet is running in this page', lede: 'The released network and its physics rivals, ported from Python and checked against it on load. Pick a real geomagnetic storm, scrub through it, then change the solar wind and watch the forecast move.', count: () => 'StormNet',
    cards: [
      { id: 'storm', cls: 'c8', title: 'Three-hour Dst forecasts through a real storm', sub: () => esc(STORM.storms[S.live.storm].label) + ' · ' + esc(STORM.storms[S.live.storm].source), draw: drawStorm },
      { id: 'side', cls: 'c4', title: 'At the selected hour', sub: '', draw: drawSide },
      { id: 'mae', cls: 'c6', title: 'Who forecast this storm best', sub: 'Mean absolute error, computed now', draw: drawMae },
      { id: 'parity', cls: 'c6', title: 'Same answers as Python', sub: 'Checked when this page opened', draw: drawParity },
    ] },
];

/* ============================================================ shell */
let RO = null, CARDS = [];
function cardEl(c) {
  const el = document.createElement('section');
  el.className = 'card ' + (c.cls || '');
  el.id = 'card-' + c.id;
  el.innerHTML = `<div class="card-h"><div><h3>${c.title}</h3>${c.sub !== undefined && c.sub !== '' ? '<p class="sub" data-sub></p>' : ''}</div><div class="acts">${c.acts ? c.acts() : ''}<button class="iconbtn" data-focus title="Focus on this visual (Esc to return)" aria-label="Focus on ${esc(c.title)}">⤢</button></div></div><div class="card-b"></div>`;
  c._el = el; c._b = $('.card-b', el); c._b._card = c; c._w = -1;
  $('[data-focus]', el).onclick = () => focusCard(el);
  return el;
}
function runCard(c) {
  if (!c._el || !c._el.isConnected) return;
  const sub = $('[data-sub]', c._el);
  if (sub) sub.textContent = typeof c.sub === 'function' ? c.sub() : c.sub;
  try { c.draw(c._b, c); } catch (e) {
    console.error(c.id, e);
    // forget anything the card cached, so the next draw rebuilds it from scratch instead of failing again
    c._b._svg = null; c._b._drawn = false; c._b._k = null; c._b._sc = null;
    c._b.innerHTML = `<div class="empty">This visual hit an error: ${esc(e.message)}</div>`;
  }
}
function runCardId(id) { const c = CARDS.find(x => x.id === id); if (c) runCard(c); }
function buildPage() {
  hideTip(); closeFocus();
  RO?.disconnect();
  const main = $('#main');
  main.innerHTML = '';
  let P = S.page.startsWith('model-') ? modelPage(S.page.slice(6)) : PAGES.find(p => p.id === S.page);
  if (!P) { S.page = 'overview'; P = PAGES[0]; }
  CARDS = P.cards;
  main.insertAdjacentHTML('beforeend', P.head ? P.head() : `<div class="page-head"><div><div class="kick">${P.kick}</div><h2>${P.title}</h2><p>${P.lede}</p></div></div>`);
  let host;
  if (P.columns) main.insertAdjacentHTML('beforeend', '<div class="mgrid"><div class="mcol" data-col="l"></div><div class="mcol" data-col="r"></div></div>');
  else { main.insertAdjacentHTML('beforeend', '<div class="grid" id="grid"></div>'); host = $('#grid'); }
  RO = new ResizeObserver(es => { for (const e of es) { const c = e.target._card; if (!c) continue; const w = Math.round(e.contentRect.width); if (w !== c._w) { c._w = w; runCard(c); } } });
  for (const c of CARDS) { const el = cardEl(c); (P.columns ? $(`[data-col="${c.col}"]`, main) : host).append(el); c.wire?.(el); RO.observe(c._b); }
  P.after?.();
  renderChrome();
  window.scrollTo({ top: 0 });
}
function setPage(id) {
  if (!id.startsWith('model-')) S.prevPage = id;
  S.page = id;
  buildPage();
  if (location.hash.slice(1) !== id) { try { location.hash = id; } catch (e) { /* the frame may not allow it */ } }
}
function openModel(id, rowKey = null, section = null) {
  S.focusRow = rowKey; S.focusSection = section;
  if (!S.page.startsWith('model-')) S.prevPage = S.page;
  setPage('model-' + id);
  S.focusSection = null;
}
function validPage(h) { return PAGES.some(p => p.id === h) || (h.startsWith('model-') && BYID.has(h.slice(6))) ? h : 'overview'; }
addEventListener('hashchange', () => { const id = validPage(location.hash.slice(1)); if (id !== S.page) { S.page = id; buildPage(); } });
function changed() { turnGear(); renderChrome(); if (!S.page.startsWith('model-')) CARDS.forEach(runCard); }
function renderChrome() { renderTabs(); renderPane(); renderStatus(); }
function renderTabs() {
  const t = $('#tabs');
  t.innerHTML = PAGES.map((p, i) => `<button role="tab" data-p="${p.id}" aria-selected="${S.page === p.id}" title="${i + 1}">${p.label}<span class="n">${p.count()}</span></button>`).join('') +
    (S.page.startsWith('model-') ? `<button role="tab" class="model-tab" aria-selected="true">◆ ${esc(BYID.get(S.page.slice(6)).name)}</button>` : '');
  $$('[data-p]', t).forEach(b => b.onclick = () => setPage(b.dataset.p));
}
function filterDesc() {
  const f = S.f, parts = [];
  const set = (k, lab, map) => { if (f[k].size) parts.push(`${lab}: ${[...f[k]].map(v => map ? map[v] || v : v).join(', ')}`); };
  set('domain', 'Domain'); set('platform', 'Simulator', PLAT); set('verdict', 'Verdict', VLABEL); set('audit', 'Audit', ACLASS); set('status', 'Status');
  if (f.model.size) parts.push('Model: ' + [...f.model].map(id => BYID.get(id).name).join(', '));
  if (f.dumb !== null) parts.push(f.dumb ? 'beats the dumb baseline' : 'behind the dumb baseline');
  if (f.classical !== null) parts.push(f.classical ? 'beats the classical method' : 'behind the classical method');
  return parts;
}
function renderPane() {
  const p = $('#pane'), f = S.f;
  const chip = (key, v, label, n, sw) => `<button class="chip" data-f="${key}" data-v="${esc(v)}" aria-pressed="${f[key] instanceof Set ? f[key].has(v) : f[key] === (v === 'true')}">${sw ? `<span class="sw" style="background:${sw}"></span>` : ''}${esc(label)}<span class="c">${n}</span></button>`;
  const mCount = (key, test) => MODELS.filter(m => mPass(m, key) && test(m)).length;
  const rCount = (key, test) => ROWS.filter(r => r.verdict !== 'info' && rPass(r, key) && test(r)).length;
  const act = filterDesc();
  p.innerHTML = `<button class="btn sm pane-close" data-close>Done</button><h4>Filters ${act.length ? '<button class="btn sm" data-clear>Clear</button>' : ''}</h4><div class="chips">${act.length ? act.map(a => `<span class="chip static">${esc(a)}</span>`).join('') : '<span class="muted" style="font-size:12px">None. Click any mark in a visual, or a chip below.</span>'}</div>
  <h4>Domain</h4><div class="chips">${DOMAINS.map(d => chip('domain', d, d, mCount('domain', m => m.domain === d))).join('')}</div>
  <h4>Simulator</h4><div class="chips">${PLATS.map(pl => chip('platform', pl, PLAT[pl], mCount('platform', m => m.platform === pl))).join('')}</div>
  <h4>Verdict</h4><div class="chips">${[...GRADED, 'info'].map(v => chip('verdict', v, VLABEL[v], ROWS.filter(r => rPass(r, 'verdict') && r.verdict === v).length, VC(v))).join('')}</div>
  <h4>Against the dumb baseline</h4><div class="chips">${chip('dumb', 'true', 'Beats it', rCount('dumb', r => r.beat.dumb === true))}${chip('dumb', 'false', 'Behind it', rCount('dumb', r => r.beat.dumb === false))}</div>
  <h4>Against the classical method</h4><div class="chips">${chip('classical', 'true', 'Beats it', rCount('classical', r => r.beat.classical === true))}${chip('classical', 'false', 'Behind it', rCount('classical', r => r.beat.classical === false))}</div>
  <h4>Audit finding</h4><div class="chips">${AORDER.map(c => chip('audit', c, ACLASS[c], mCount('audit', m => m.auditCls === c), `var(--${c})`)).join('')}</div>
  <h4>Status</h4><div class="chips">${['released', 'partial'].map(s => chip('status', s, s, mCount('status', m => m.status === s))).join('')}</div>
  <h4>Models</h4><div class="chips">${MODELS.map(m => chip('model', m.id, m.name, m.graded, m.accent)).join('')}</div>
  ${act.length ? '<button class="btn clear" data-clear>Clear every filter</button>' : ''}
  <p class="note" style="margin-top:14px">Chips here add to each other. In a visual, a click replaces the filter; Shift-click adds.</p>`;
}
$('#pane').addEventListener('click', ev => {
  if (ev.target.closest('[data-close]')) { $('#paneBtn').click(); return; }
  if (ev.target.closest('[data-clear]')) { resetFilters(); hideAnswer(); changed(); return; }
  const c = ev.target.closest('[data-f]'); if (!c) return;
  const k = c.dataset.f, v = c.dataset.v;
  if (k === 'dumb' || k === 'classical') setTri(k, v === 'true'); else toggle(k, v, true);
});
function renderStatus() {
  const st = stats(), tot = ROWS.filter(r => r.verdict !== 'info').length, act = filterDesc();
  $('#status').innerHTML = `<span><b>${st.ms.length}</b> of ${MODELS.length} models</span><span><b>${fmt(st.graded)}</b> of ${fmt(tot)} graded rows</span><span class="fdesc">${act.length ? 'Filtered: ' + esc(act.join(' · ')) : 'No filters'}</span>${act.length ? '<span class="fshort" style="display:none">Filtered</span>' : ''}<span class="r">Read from main @ ${esc(D.head.h)} (${esc(D.head.t.slice(0, 16).replace('T', ' '))}) · ${MODELS.length} model cards · ${MODELS.length} ratchet logs · ${D.auditTotal} audits · ${SER.length} commits</span>`;
}
let focusedEl = null;
function focusCard(el) {
  if (focusedEl === el) return closeFocus();
  closeFocus(); focusedEl = el; el.classList.add('focused'); $('#scrim').classList.add('on'); $('[data-focus]', el).textContent = '✕';
}
function closeFocus() { if (!focusedEl) return; focusedEl.classList.remove('focused'); $('[data-focus]', focusedEl).textContent = '⤢'; focusedEl = null; $('#scrim').classList.remove('on'); }
$('#scrim').onclick = closeFocus;
function pulse(id) { setTimeout(() => { const el = document.getElementById('card-' + id); if (!el) return; el.scrollIntoView({ behavior: 'smooth', block: MOBILE() ? 'start' : 'center' }); el.classList.remove('pulse'); void el.offsetWidth; el.classList.add('pulse'); }, 250); }

/* ============================================================ Ask */
const ALIAS = [
  [/\bhydrolog|\brivers?\b|\bstreams?\b|\bsnow|\breservoirs?\b|\bdams?\b|\bbasins?\b|\bstreamflow/, 'domain', 'Hydrology'],
  [/\bpower grids?\b|\bgrids?\b|\belectric|\bdemand\b/, 'domain', 'Power grids'], [/\bbuildings?\b|\bhvac\b|\bair handler/, 'domain', 'Buildings'],
  [/\bwildfires?\b|\bfires?\b/, 'domain', 'Wildfire'], [/\bastro|\bstars?\b|\bplanets?\b|\bexoplanet|\bblack holes?\b|\bgravitational|\bligo\b|\btess\b|\bkepler\b/, 'domain', 'Astrophysics'],
  [/\bvehicles?\b|\bcars?\b|\btrucks?\b|\bbus(es)?\b|\btransport/, 'domain', 'Transportation'], [/\bearthquakes?\b|\bquakes?\b|\bseism|\btsunamis?\b|\bgeophys/, 'domain', 'Geophysics'],
  [/\boceans?\b|\bswell\b|\bbuoys?\b/, 'domain', 'Oceanography'], [/\bspace weather\b|\bgeomagnetic|\bsolar wind\b|\bdst\b/, 'domain', 'Space weather'],
  [/\bbatter(y|ies)\b|\bwind (turbines?|farms?)\b|\bsolar (farms?|panels?|pv)\b|\benergy\b/, 'domain', 'Energy'], [/\bairfoils?\b|\baero/, 'domain', 'Aerodynamics'],
  [/\bhearts?\b|\becg\b|\bphysiolog/, 'domain', 'Physiology'], [/\bacoustic|\brooms?\b|\breverb/, 'domain', 'Acoustics'], [/\bbearings?\b|\bmachin/, 'domain', 'Machinery'],
  [/\broadsim\b/, 'platform', 'vehicle'], [/\bgridsim\b/, 'platform', 'grid'], [/\bbuildingsim\b/, 'platform', 'building'], [/\bwatersim\b/, 'platform', 'water'], [/\bfiresim\b/, 'platform', 'fire'], [/\bstandalone\b|\bown simulator\b/, 'platform', 'standalone'],
  [/\bcritical\b/, 'audit', 'critical'], [/\bclean\b|\bpassed\b/, 'audit', 'clean'], [/\bpartial\b/, 'status', 'partial'],
];
const EXAMPLES = ['where does it lose to a dumb baseline in hydrology?', 'wins in power grids', 'which models have critical audit findings?', 'ties on the building platform', 'are the error bars honest?', 'how often does the ratchet say no?', 'quakenet v2', 'run stormnet live'];
function ask(q) {
  const raw = q.trim(); if (!raw) return showExamples();
  let s = ' ' + raw.toLowerCase() + ' ';
  resetFilters();
  const read = []; let page = null, pul = null;
  const named = [...MODELS].sort((a, b) => b.name.length - a.name.length).filter(m => {
    const re = new RegExp('\\b' + m.name.toLowerCase().replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/\s+/g, '\\s*') + '\\b');
    if (re.test(s)) { s = s.replace(re, ' '); return true; } return false;
  });
  const lose = /\b(lose|loses|losing|lost|loss|losses|behind|worse|fail|fails|failed|wins? against it)\b/.test(s);
  const win = /\b(win|wins|won|winning|beat|beats|beating|ahead|better)\b/.test(s);
  const dumb = /\b(dumb|naive|median|climatology|persistence|trivial|simple baseline)\b/.test(s);
  const classical = /\b(classical|textbook|physics baseline|physics alone)\b/.test(s);
  if (dumb) { S.f.dumb = !(lose || /\bwhere does (a|the) (dumb|naive|median)/.test(s)) && win ? true : false; read.push(S.f.dumb ? 'beats the dumb baseline' : 'behind the dumb baseline'); }
  else if (classical) { S.f.classical = !lose && win; read.push(S.f.classical ? 'beats the classical method' : 'behind the classical method'); }
  else {
    if (win && !lose) { S.f.verdict.add('win'); read.push('Verdict: win'); }
    if (lose && !win) { S.f.verdict.add('loss'); read.push('Verdict: loss'); }
    if (/\b(tie|ties|tied|draw)\b/.test(s)) { S.f.verdict.add('tie'); read.push('Verdict: tie'); }
    if (/\bmixed\b/.test(s)) { S.f.verdict.add('mixed'); read.push('Verdict: mixed'); }
  }
  for (const [re, k, v] of ALIAS) if (re.test(s) && !S.f[k].has(v)) { S.f[k].add(v); read.push(`${{ domain: 'Domain', platform: 'Simulator', audit: 'Audit', status: 'Status' }[k]}: ${k === 'platform' ? PLAT[v] : k === 'audit' ? ACLASS[v] : v}`); }
  if (/\b(audit|audits|audited|defects?|bugs?|findings?|fixe?d?)\b/.test(s)) page = 'audits';
  else if (/\b(ratchet|rungs?|reject|rejected|rejects|ladders?|refus|says no)\b/.test(s)) page = 'ratchet';
  else if (/\b(coverage|error bars?|calibrat|overconfident|uncertaint|honest bands?|bands?)\b/.test(s)) { page = 'overview'; pul = 'coverage'; }
  else if (/\b(live|run|running|storm|stormnet)\b/.test(s)) page = 'live';
  else if (/\b(lineage|sankey|training|real data|labels?|sources?|licen[cs]e)\b/.test(s)) page = 'lineage';
  else if (/\b(table|rows?|list|which rows|csv)\b/.test(s) || S.f.verdict.size || S.f.dumb !== null || S.f.classical !== null) page = 'exams';
  if (named.length === 1 && !read.length && !page) { hideAnswer(); openModel(named[0].id); return; }
  if (named.length) { named.forEach(m => S.f.model.add(m.id)); read.push('Model: ' + named.map(m => m.name).join(', ')); }
  if (named.length === 1 && page === 'live' && named[0].id === 'stormnet') { resetFilters(); read.length = 0; read.push('Model: StormNet'); }
  if (!read.length && !page) { answer(`I could not match “${esc(raw)}” to anything in the collection. Ask runs in this page and only knows its words: domains, verdicts, baselines, audits, the ratchet, model names.`, true); return; }
  changed();
  setPage(page || 'overview');
  if (pul) pulse(pul);
  if (page === 'exams') pulse('wall');
  const st = stats(), g = st.rs.filter(r => r.verdict !== 'info');
  const top = d3.rollups(g, v => v.length, r => r.m.name).sort((a, b) => b[1] - a[1]).slice(0, 3);
  let msg = `<b>${fmt(g.length)}</b> graded rows from <b>${st.ms.length}</b> models: ${st.v.win} wins, ${st.v.loss} losses, ${st.v.tie + st.v.mixed} ties or mixed.`;
  if (top.length && st.ms.length > 1) msg += ` Most from ${top.map(([n, c]) => `${esc(n)} (${c})`).join(', ')}.`;
  let lead = '';
  if (page === 'ratchet') lead = `<b>${fmt(st.rej)}</b> of <b>${fmt(st.rungs)}</b> rung decisions were rejections (${st.rungs ? pct(st.rej / st.rungs) : '0%'}). `;
  if (page === 'audits') { const ms = st.ms.filter(m => m.audit), rs = ms.flatMap(m => m.audit.resolutions); lead = `<b>${ms.filter(m => m.auditCls === 'critical').length}</b> of ${ms.length} audited models had a critical finding; ${rs.filter(r => r.action === 'fixed').length} of their ${rs.length} findings are fixed. `; }
  if (pul === 'coverage') { const c = COVS.filter(x => mPass(x.m)); lead = `<b>${c.filter(x => covBand(x.actual) === 'near').length}</b> of ${c.length} error-bar outputs land within 10 points of their nominal 68%; ${c.filter(x => covBand(x.actual) === 'narrow').length} are too narrow. `; }
  if (page === 'live') lead = 'StormNet runs in this page: pick a storm and scrub through it. ';
  msg = lead + (lead ? `<span class="muted">${msg}</span>` : msg);
  answer(msg + `<div class="chips">${read.map(r => `<span class="chip static">${esc(r)}</span>`).join('')}${page ? `<span class="chip static">Page: ${esc((PAGES.find(p => p.id === page) || {}).label || page)}</span>` : ''}</div>`);
}
function answer(html, plain) { $('#answerText').innerHTML = html; $('#answer').classList.add('on'); setHdr(); if (plain) { /* keep page */ } }
function hideAnswer() { $('#answer').classList.remove('on'); setHdr(); }
function showExamples() {
  answer(`Try one of these, or your own words. Ask runs in this page: it maps your words to the filters and pages, with no model behind it.<div class="chips">${EXAMPLES.map(e => `<button class="chip" data-ex="${esc(e)}">${esc(e)}</button>`).join('')}</div>`);
}
$('#answer').addEventListener('click', ev => { const e = ev.target.closest('[data-ex]'); if (e) { $('#ask').value = e.dataset.ex; ask(e.dataset.ex); } });
$('#ask').addEventListener('keydown', ev => { if (ev.key === 'Enter') { ask(ev.target.value); if (MOBILE()) ev.target.blur(); } if (ev.key === 'Escape') { ev.target.blur(); hideAnswer(); } });
$('#ask').addEventListener('focus', ev => { if (!ev.target.value.trim()) showExamples(); });
$('#answerClose').onclick = () => { hideAnswer(); $('#ask').value = ''; resetFilters(); changed(); };

/* ============================================================ Tour */
const TOUR = [
  { k: 'The idea', h: 'Every dot is one real exam', go() { resetFilters(); S.swarmGroup = 'none'; setPage('overview'); pulse('swarm'); }, p: () => `${SROWS.length} comparisons on real data that chose nothing. Each dot puts one model's score beside the best baseline it was graded against. Right of the line the model did better; left, a baseline did. ${TOUCH ? 'Tap a dot for its numbers; tap it again to open the model.' : 'Hover any dot, click to open its model.'}` },
  { k: 'Honest losses', h: 'Where a dumb baseline wins', go() { resetFilters(); S.f.dumb = false; setPage('exams'); pulse('wall'); }, p: () => { const n = ROWS.filter(r => r.beat.dumb === false && r.verdict !== 'info').length, t = ROWS.filter(r => r.beat.dumb !== undefined && r.verdict !== 'info').length; return `${n} of the ${t} graded rows that list a dumb baseline are behind it: a median, a climatology, yesterday's value. They stay on the record. The lit squares are those rows.`; } },
  { k: 'A hard domain', h: 'Rivers and snow', go() { resetFilters(); S.f.domain.add('Hydrology'); S.swarmGroup = 'domain'; setPage('overview'); pulse('swarm'); }, p: () => { const g = ROWS.filter(r => r.m.domain === 'Hydrology' && r.verdict !== 'info'); return `Hydrology: ${g.filter(r => r.verdict === 'win').length} wins, ${g.filter(r => r.verdict === 'loss').length} losses and ${g.filter(r => r.verdict === 'tie').length} ties across ${new Set(g.map(r => r.m.id)).size} models. A tuned conceptual model is a hard baseline to beat on real rivers, and the collection says so.`; } },
  { k: 'The ratchet', h: 'Most rungs are refused', go() { resetFilters(); setPage('ratchet'); pulse('anatomy'); }, p: () => { const r = MODELS.flatMap(m => m.allRungs), j = r.filter(x => x.decision === 'rejected'); return `${fmt(j.length)} of ${fmt(r.length)} rung decisions were rejections. About ${j.filter(x => x.rc === 'gate').length} were stopped by the real dev gate after looking better on the simulator.`; } },
  { k: 'Error bars', h: 'Are the bands honest?', go() { resetFilters(); setPage('overview'); pulse('coverage'); }, p: () => { const c = COVS; return `${c.filter(x => covBand(x.actual) === 'near').length} of ${c.length} outputs put their 68% band around the truth within 10 points of 68%. ${c.filter(x => covBand(x.actual) === 'narrow').length} are too narrow on the real exam. Those numbers are next rungs.`; } },
  { k: 'Real data', h: 'Zero training labels', go() { resetFilters(); setPage('lineage'); pulse('sankey'); }, p: () => `No model learned from a real labelled example. Follow each band: real data is noise under the sims, a calibration source, a gate, an error-bar width and an exam. The training-labels box has nothing flowing into it.` },
  { k: 'Audits', h: 'Found, then fixed in the open', go() { resetFilters(); setPage('audits'); pulse('board'); }, p: () => { const a = MODELS.filter(m => m.audit), r = a.flatMap(m => m.audit.resolutions); return `${a.length} models audited by two reviewers each. ${a.filter(m => m.auditCls === 'critical').length} had a critical finding. Of ${r.length} findings, ${r.filter(x => x.action === 'fixed').length} are fixed and the rest are warned, documented or queued for a re-score.`; } },
  { k: 'Live', h: 'This dashboard runs the model', go() { resetFilters(); S.live.storm = 0; S.live.k = null; setPage('live'); pulse('storm'); }, p: () => `StormNet's network is inside this page. It just forecast every hour of the May 2024 storm (Dst −406 nT) and matched Python on all ${PAR.n} reference cases. Drag across the chart, then push the solar wind south.` },
];
let tourI = -1;
function tourGo(i) {
  hideAnswer();
  tourI = Math.max(0, Math.min(TOUR.length - 1, i));
  const t = TOUR[tourI];
  t.go(); changed();
  $('#tourK').textContent = `${tourI + 1} of ${TOUR.length} · ${t.k}`; $('#tourH').textContent = t.h; $('#tourP').textContent = t.p();
  $('#tourDots').innerHTML = TOUR.map((_, j) => `<i class="${j <= tourI ? 'on' : ''}"></i>`).join('');
  $('#tourPrev').disabled = tourI === 0; $('#tourNext').textContent = tourI === TOUR.length - 1 ? 'Done' : 'Next';
  $('#tour').classList.add('on');
}
function tourEnd() { $('#tour').classList.remove('on'); tourI = -1; }
$('#tourBtn').onclick = () => tourGo(0);
$('#tourNext').onclick = () => tourI === TOUR.length - 1 ? tourEnd() : tourGo(tourI + 1);
$('#tourPrev').onclick = () => tourGo(tourI - 1);
$('#tourEnd').onclick = tourEnd;

/* ============================================================ Go-to palette */
let palItems = [], palSel = 0;
function palAll() {
  return [
    ...PAGES.map((p, i) => ({ t: p.label, k: 'page ' + (i + 1), run: () => setPage(p.id) })),
    ...MODELS.map(m => ({ t: m.name, k: m.domain, sw: m.accent, run: () => openModel(m.id) })),
    ...TOUR.map((s, i) => ({ t: 'Tour: ' + s.h, k: 'step ' + (i + 1), run: () => tourGo(i) })),
    { t: 'Clear every filter', k: 'action', run: () => { resetFilters(); changed(); } },
    { t: 'Copy the filtered exam rows as CSV', k: 'action', run: () => copyText(csvRows(), `${tableRows().length} rows as CSV`) },
    { t: 'Switch light or dark', k: 'action', run: () => $('#themeBtn').click() },
  ];
}
function palRender() {
  const q = $('#palIn').value.toLowerCase().trim();
  const score = t => { if (!q) return 1; t = t.toLowerCase(); if (t.includes(q)) return 3 - t.indexOf(q) / 100; let i = 0; for (const ch of t) if (ch === q[i]) i++; return i === q.length ? 1 : 0; };
  palItems = palAll().map(x => ({ ...x, s: score(x.t + ' ' + x.k) })).filter(x => x.s > 0).sort((a, b) => b.s - a.s).slice(0, 40);
  palSel = Math.min(palSel, Math.max(0, palItems.length - 1));
  $('#palList').innerHTML = palItems.map((x, i) => `<li role="option" data-i="${i}" aria-selected="${i === palSel}">${x.sw ? `<span style="width:8px;height:8px;border-radius:50%;background:${x.sw}"></span>` : ''}${esc(x.t)}<span class="k mono muted">${esc(x.k)}</span></li>`).join('') || '<li class="muted">Nothing matches.</li>';
}
function openPalette() { $('#palette').classList.add('on'); $('#scrim').classList.add('on'); $('#palIn').value = ''; palSel = 0; palRender(); $('#palIn').focus(); }
function closePalette() { $('#palette').classList.remove('on'); if (!focusedEl) $('#scrim').classList.remove('on'); }
function palRun(i) { const x = palItems[i]; if (!x) return; closePalette(); x.run(); }
$('#palIn').addEventListener('input', () => { palSel = 0; palRender(); });
$('#palIn').addEventListener('keydown', ev => {
  if (ev.key === 'ArrowDown') { palSel = Math.min(palItems.length - 1, palSel + 1); palRender(); ev.preventDefault(); }
  else if (ev.key === 'ArrowUp') { palSel = Math.max(0, palSel - 1); palRender(); ev.preventDefault(); }
  else if (ev.key === 'Enter') palRun(palSel);
  else if (ev.key === 'Escape') closePalette();
});
$('#palList').addEventListener('click', ev => { const li = ev.target.closest('[data-i]'); if (li) palRun(+li.dataset.i); });
$('#palBtn').onclick = openPalette;
$('#scrim').addEventListener('click', closePalette);

/* ============================================================ keys, theme, pane, init */
addEventListener('keydown', ev => {
  const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || '');
  if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === 'k') { ev.preventDefault(); $('#palette').classList.contains('on') ? closePalette() : openPalette(); return; }
  if (ev.key === 'Escape') { closePalette(); closeFocus(); if (tourI >= 0) tourEnd(); return; }
  if (typing) return;
  if (ev.key === '/') { ev.preventDefault(); $('#ask').focus(); return; }
  if (tourI >= 0 && ev.key === 'ArrowRight') { $('#tourNext').click(); return; }
  if (tourI >= 0 && ev.key === 'ArrowLeft') { tourGo(tourI - 1); return; }
  const n = +ev.key; if (n >= 1 && n <= PAGES.length && !ev.metaKey && !ev.ctrlKey && !ev.altKey) setPage(PAGES[n - 1].id);
});
$('#themeBtn').onclick = () => {
  const r = document.documentElement, cur = r.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  r.dataset.theme = cur === 'dark' ? 'light' : 'dark';
  try { localStorage.setItem('mlr-theme', r.dataset.theme); } catch (e) { /* the same key as the rest of the site */ }
};
$('#paneBtn').onclick = () => {
  const off = $('#shell').classList.toggle('pane-off'); $('#paneBtn').setAttribute('aria-pressed', !off);
  if (innerWidth <= 980) $('#scrim').classList.toggle('on', !off);
};
$('#scrim').addEventListener('click', () => { if (innerWidth <= 980 && !$('#shell').classList.contains('pane-off')) { $('#shell').classList.add('pane-off'); $('#paneBtn').setAttribute('aria-pressed', 'false'); } });
if (TOUCH) document.addEventListener('pointerdown', ev => { if (!ev.target.closest('svg, .wall, #tip')) { tapKey = null; hideTip(); } }, true);
if (MOBILE()) $('#ask').placeholder = 'Ask the collection a question';
function setHdr() { document.documentElement.style.setProperty('--hdr', $('#hdr').offsetHeight + 'px'); }
new ResizeObserver(setHdr).observe($('#hdr'));
if (/Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent)) $('#kbdK').textContent = '⌘K';
if (innerWidth < 980) { $('#shell').classList.add('pane-off'); $('#paneBtn').setAttribute('aria-pressed', 'false'); }
$('#prov').textContent = `${MODELS.length} models · read from main @ ${D.head.h}, ${D.head.t.slice(0, 10)}`;
PAR = STORM.parity();
S.page = validPage(location.hash.slice(1));
if (!S.page.startsWith('model-')) S.prevPage = S.page;
buildPage();

// A handle for tests and the console (the app is a module, so nothing else is global).
window.__obs = { S, setPage, resetFilters, MODELS };
