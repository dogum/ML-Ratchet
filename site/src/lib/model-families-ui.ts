// Browser side of the Model families page (site/src/pages/model-families.astro). The page renders the inventory,
// catalogue, lenses, backlog, budget chart and references at build time; this module wires them up and draws the
// interactive parts: the twin-test triage, the problem-shape picker, the extrapolation lab, the opportunity-map
// detail, the TreeSHAP illustration and the rung drafter. Model facts arrive as JSON in #mf-data.
import { GROUPS, FAMILIES, CELLS, SHAP_CASES, SHAP_FEATURES, type Cell, type Family } from './model-families';

export interface PageModel { id: string; name: string; acc: string; gate: string; scored?: string; anchored: boolean }

const q = <T extends Element = HTMLElement>(s: string, r: ParentNode = document): T => r.querySelector<T>(s) as T;
const qa = <T extends Element = HTMLElement>(s: string, r: ParentNode = document): T[] => Array.from(r.querySelectorAll<T>(s));
const esc = (s: unknown) => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const FAM: Record<string, Family> = Object.fromEntries(FAMILIES.map((f) => [f.id, f]));
const fcol = (id: string) => `var(${GROUPS[FAM[id].g].v})`;
const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
const go = (el: Element | null) => el?.scrollIntoView({ behavior: reduced() ? 'auto' : 'smooth', block: 'start' });
const store = {
  get: (k: string): string | null => { try { return localStorage.getItem('mlr-mf:' + k); } catch { return null; } },
  set: (k: string, v: string) => { try { localStorage.setItem('mlr-mf:' + k, v); } catch { /* storage unavailable */ } },
};
const press = (group: HTMLElement[], on: HTMLElement) => group.forEach((b) => b.setAttribute('aria-pressed', String(b === on)));

export function init() {
  const data = JSON.parse(q('#mf-data').textContent || '{}') as { models: PageModel[] };
  const MOD: Record<string, PageModel> = Object.fromEntries(data.models.map((m) => [m.id, m]));
  const api = { openFamily: (_id: string) => {}, selectCell: (_m: string, _f: string) => {}, draft: (_m: string, _f: string) => {} };

  /* ---- contents bar: highlight the section in view. The bar scrolls sideways on its own; the page is never moved,
     because scrollIntoView on a link inside a sticky bar also scrolls the page toward the bar's unstuck position. ---- */
  const bar = q('.mf-toc-in');
  const links = Object.fromEntries(qa<HTMLAnchorElement>('a[data-id]', bar).map((a) => [a.dataset.id!, a]));
  const io = new IntersectionObserver((ents) => ents.forEach((e) => {
    if (!e.isIntersecting) return;
    const a = links[e.target.id]; if (!a) return;
    Object.values(links).forEach((x) => x.classList.toggle('on', x === a));
    const l = a.offsetLeft, r = l + a.offsetWidth;
    if (l < bar.scrollLeft || r > bar.scrollLeft + bar.clientWidth) bar.scrollLeft = Math.max(0, l - 24);
  }), { rootMargin: '-40% 0px -55% 0px' });
  Object.keys(links).forEach((id) => { const s = document.getElementById(id); if (s) io.observe(s); });

  /* ---- 01 inventory ---- */
  const rows = qa('#mf-tm tbody tr[data-m]');
  const showModel = (id: string) => {
    if (!q(`[data-mpanel="${id}"]`)) return;
    rows.forEach((tr) => { const on = tr.dataset.m === id; tr.classList.toggle('on', on); q('button', tr)?.setAttribute('aria-pressed', String(on)); });
    qa('[data-mpanel]').forEach((p) => { p.hidden = p.dataset.mpanel !== id; });
    store.set('model', id);
  };
  rows.forEach((tr) => tr.addEventListener('click', () => showModel(tr.dataset.m!)));
  qa('[data-mpanel] [data-cell]').forEach((b) => b.addEventListener('click', () => { const [m, f] = b.dataset.cell!.split(':'); api.selectCell(m, f); go(q('#map')); }));
  showModel(store.get('model') || 'energynet');

  /* ---- 02 triage ---- */
  type TNode = { q?: string; hint?: string; opts?: [string, string][]; v?: string; color?: string; title?: string; body?: string; ev?: string[]; fams?: [string, string][] };
  const N: Record<string, TNode> = {
    start: { q: 'Has the twin test been run on this model?', hint: 'Re-simulate each real dev unit with the full simulator (same event, same station, same car), then read both the twin and the real unit with the current network.', opts: [['Yes', 'twin'], ['Not yet', 'run']] },
    twin: { q: 'Does the network read the simulated twins correctly?', hint: 'Twins are simulated, so the network has seen their kind of data. Misreading them means the learner falls short even in its own world.', opts: [['Yes, the twins read well', 'twinok'], ['No, it misreads its own twins', 'twinbad']] },
    twinok: { q: 'And the real dev units?', opts: [['Much worse than their twins', 'simgap'], ['About as well as the twins', 'ok']] },
    twinbad: { q: 'Did a bigger simulated bank lift the frozen-set score?', hint: 'Simulations cost compute, not labels. Three times the bank on the same physics is the cheaper test.', opts: [['Yes, more data helped', 'data'], ['No, or not tried', 'rep']] },
    rep: { q: 'Does the input representation throw away structure the physics has?', hint: 'For example: stations read one at a time, trips or windows pooled by a fixed rule, a variable-length record cropped to a fixed length, a long record whose parts must be linked.', opts: [['Yes', 'struct'], ['No', 'learner']] },
    run: { v: 'First step', title: 'Run the twin test first', body: 'It is the cheapest experiment in the collection, and its answer decides which kind of rung comes next. The roadmap lists a reusable twin-test harness for mlratchet; until it exists, each model runs its own.', ev: ['<b>QuakeNet:</b> twins showed the shallow-depth problem lives in the surface waves after S.', '<b>BearingNet:</b> twins showed the textbook fault-level prior was wrong; fixing it took dev balanced accuracy from 0.31 to 0.68.'], fams: [] },
    simgap: { v: 'Verdict', color: 'var(--real)', title: 'The gap is in the simulator', body: 'A new learner trained on the same simulations learns the same wrong world. The next rung is a simulator rung: plot real dev units beside their twins, find the missing physics, build it in. Learner changes still help here as detectors and patches, not as fixes.', ev: ['<b>BearingNet rung 1:</b> 100% of twins read right, 22% of real dev files.', '<b>BatteryPilot:</b> dev twins capture 91–94% of perfect hindsight, real 2019 only 80%. The simulated price spikes are too predictable.', '<b>TripDetective:</b> a dev hybrid’s twin stops its engine 0.29 of the time, the real car 0.09.'], fams: [['ensemble', 'Two families as a disagreement detector'], ['symbolic', 'A formula for the gap, to write into the simulator'], ['gp', 'A learned discrepancy term (dial stop 4 on Mental models)']] },
    ok: { v: 'Verdict', color: 'var(--win)', title: 'No large learner or simulator gap', body: 'When twins and real units read alike and the exam still disappoints, the next places to look are the mix of cases, label noise, coverage and the baseline. The dumb baseline built from the context inputs may simply be strong.', ev: ['<b>GridPulse:</b> a 7.0 s × demand rule fitted on four dev events beats the network on GB inertia at 60 s, 15.1% against 16.4%.', '<b>TripDetective:</b> matching the fleet’s mass mix took the dev mass bias from +106 kg to −9 kg, with no change to the learner.'], fams: [['conformal', 'Coverage per stratum'], ['experts', 'A gate where strata behave differently']] },
    data: { v: 'Verdict', color: 'var(--sim)', title: 'Data-limited: grow the bank first', body: 'Grow the bank on the same physics and judge again. A faster simulator, possibly through a surrogate, is one way to grow it.', ev: ['<b>EnergyNet v2-r13:</b> three times the simulated data, +0.019 of skill, kept.', '<b>EnergyNet v2-r11:</b> wider networks (0.73 M weights each) with twice the data, −0.019, rejected.'], fams: [['operator', 'A surrogate to make the simulator faster']] },
    struct: { v: 'Verdict', color: 'var(--f-struct)', title: 'A representation gap: match the learner to the structure', body: 'Sets for interchangeable items, graphs for linked parts, sequence models for variable length or for linking distant parts. Same bank, same frozen set, same gate.', ev: ['<b>QuakeNet:</b> multi-station reading is on its own next list.', '<b>PlantDetective:</b> windows are pooled by summed log-probabilities, a fixed rule a set learner could replace.', '<b>StarQuake:</b> reads 27 days to 4 years with one network and a length-weighted blend.'], fams: [['sets', 'Sets'], ['gnn', 'Graphs'], ['ssm', 'State-space'], ['attn', 'Attention']] },
    learner: { v: 'Verdict', color: 'var(--f-tab)', title: 'A learner gap: try another family on the same bank', body: 'The learner misreads data like its training data, and the representation looks right. A different built-in assumption is the next test, and the ratchet decides.', ev: ['<b>EnergyNet:</b> reads RoadSim’s own dev twins 4.5% low on the FTP and 0.8% low on the highway; the remaining generator difference has not been found.'], fams: [['gbm', 'Trees for tables'], ['gp', 'GPs for small banks'], ['flow', 'Flows for answers with several modes'], ['anchor', 'A physics anchor, if there is none yet']] },
  };
  let path: { from: string; to: string; label: string }[] = [];
  const renderTriage = () => {
    const cur = path.length ? path[path.length - 1].to : 'start';
    q('#mf-tr-trail').innerHTML = path.length ? path.map((p) => `<li><span><b>${esc(N[p.from].q)}</b>${esc(p.label)}</span></li>`).join('') : '<li><span><b>Start</b>The first question is on the right.</span></li>';
    const n = N[cur], stage = q('#mf-tr-stage');
    if (n.q) {
      stage.innerHTML = `<p class="kicker">Question ${path.length + 1}</p><p class="tq">${esc(n.q)}</p>${n.hint ? `<p class="hint">${esc(n.hint)}</p>` : ''}<div class="opts">${n.opts!.map(([l, to]) => `<button type="button" class="opt" data-to="${to}">${esc(l)}</button>`).join('')}</div>`;
      qa<HTMLButtonElement>('.opt', stage).forEach((b) => b.addEventListener('click', () => { path.push({ from: cur, to: b.dataset.to!, label: b.textContent || '' }); renderTriage(); }));
    } else {
      stage.innerHTML = `<div class="verdict-box"><p class="vk" style="color:${n.color || 'var(--ink-3)'}">${esc(n.v)}</p><h3 class="display">${esc(n.title)}</h3><p class="hint">${esc(n.body)}</p><p class="kicker">Evidence from the collection</p><ul class="ev">${n.ev!.map((e) => `<li>${e}</li>`).join('')}</ul>${n.fams!.length ? `<p class="kicker">Families to look at</p><div class="togs">${n.fams!.map(([id, l]) => `<button type="button" class="tog" data-fam="${id}"><span class="sw" style="background:${fcol(id)}"></span>${esc(l)}</button>`).join('')}</div>` : ''}</div>`;
      qa('[data-fam]', stage).forEach((b) => b.addEventListener('click', () => api.openFamily(b.dataset.fam!)));
    }
  };
  q('#mf-tr-reset').addEventListener('click', () => { path = []; renderTriage(); });
  renderTriage();

  /* ---- 03 problem-shape picker ---- */
  const IN: Record<string, string> = { sig: 'A fixed-length signal', var: 'A record of varying length', grid: 'A 2-D grid or image', tab: 'A table of features', set: 'A set of items (trips, windows)', graph: 'A network of linked parts', field: 'A spatial field from a PDE', ctrl: 'Decisions over time' };
  const OUT: Record<string, string> = { point: 'A value with an error bar', post: 'A full posterior', cls: 'A class with probabilities', traj: 'A forecast trajectory', law: 'A readable law', act: 'An action' };
  const CON: Record<string, string> = { small: 'Small bank (under ~5k cases)', ood: 'Real cases may fall outside the bank', web: 'Must port to the browser easily', stream: 'Readings arrive live', explain: 'Must be explainable', strata: 'Strata behave differently', cover: 'Coverage must hold' };
  type Aff = { in: Record<string, number>; out: Record<string, number>; c: Record<string, [number, string]> };
  const A: Record<string, Aff> = {
    cnn1d: { in: { sig: 3, var: 1 }, out: { point: 1, cls: 1 }, c: { web: [1, 'already runs in site/src/lib/nn'], small: [-1, 'wants 10⁴ or more simulations'] } },
    cnn2d: { in: { grid: 3, field: 2, sig: 1 }, out: { point: 1, cls: 1 }, c: { web: [1, 'already runs in site/src/lib/nn'], small: [-1, 'wants 10⁴ or more simulations'], stream: [-1, 're-reads the whole grid'] } },
    attn: { in: { var: 3, sig: 2, set: 2, grid: 1, graph: 1, ctrl: 1 }, out: { point: 1, cls: 1, traj: 2 }, c: { small: [-2, 'needs large banks'], web: [-1, 'more code to port'], stream: [-1, 're-reads the whole record'], explain: [-1, 'hard to read'] } },
    ssm: { in: { var: 3, sig: 2, ctrl: 2 }, out: { point: 1, traj: 2, act: 1 }, c: { stream: [2, 'updates as each reading arrives'], small: [-1, 'needs more data than a CNN'] } },
    mlp: { in: { tab: 3, ctrl: 2, sig: 1 }, out: { point: 1, cls: 1, act: 1 }, c: { web: [1, 'already runs in site/src/lib/nn'], ood: [-1, 'continues linearly outside the bank'], explain: [-1, 'hard to read'] } },
    gbm: { in: { tab: 3, sig: 1, ctrl: 1 }, out: { point: 1, cls: 2, law: 1 }, c: { small: [1, 'trains well on small tables'], ood: [-2, 'goes flat outside the bank'], web: [1, 'arrays of thresholds, easy to port'], explain: [1, 'exact TreeSHAP attributions'], strata: [1, 'splits on strata naturally'] } },
    gp: { in: { tab: 2, var: 1, field: 1 }, out: { point: 2, traj: 1, law: 1, post: 1 }, c: { small: [2, 'strong on a few thousand cases'], ood: [-1, 'falls back to the mean outside the bank, with wider bars'], web: [-1, 'ships its training points'], explain: [1, 'kernels and length scales can be read'], cover: [1, 'error bars are native'] } },
    tabfm: { in: { tab: 2 }, out: { point: 1, cls: 1 }, c: { small: [2, 'built for small tables'], web: [-3, 'needs PyTorch and large weights'], explain: [-1, 'hard to read'] } },
    sets: { in: { set: 3, var: 1, graph: 1 }, out: { point: 1, cls: 1 }, c: { web: [1, 'pooling is a sum or a softmax'] } },
    gnn: { in: { graph: 3, set: 1, field: 1 }, out: { point: 1, cls: 1 }, c: {} },
    operator: { in: { field: 3, grid: 1 }, out: { traj: 2 }, c: { small: [-1, 'needs many solver runs'], web: [-1, 'needs an FFT port'] } },
    flow: { in: { sig: 1, var: 1, grid: 1, tab: 1, set: 1, graph: 1, field: 1 }, out: { post: 3 }, c: { small: [-1, 'needs many simulations'], cover: [1, 'returns the whole posterior'] } },
    diffusion: { in: { grid: 1, field: 1, sig: 1 }, out: { post: 2 }, c: { small: [-2, 'needs large banks'], web: [-2, 'many network passes per sample'], explain: [-1, 'hard to read'] } },
    symbolic: { in: { tab: 1 }, out: { law: 3, point: 1 }, c: { small: [1, 'needs few cases if the form is simple'], ood: [1, 'extrapolates if the form is right'], web: [2, 'a formula'], explain: [3, 'the output is a formula'] } },
    diffsim: { in: { ctrl: 3 }, out: { act: 3 }, c: { ood: [1, 'the physics executor still applies'] } },
    assim: { in: { var: 2, ctrl: 1, field: 1 }, out: { traj: 2, point: 1 }, c: { stream: [3, 'built for live correction'], ood: [1, 'carries the physics into new regimes'], explain: [1, 'its states are physical'], small: [1, 'needs a model, not a bank'] } },
  };
  const form = q<HTMLFormElement>('#mf-pk-form');
  const radio = (name: string, map: Record<string, string>, def: string, legend: string) => `<fieldset><legend class="kicker">${legend}</legend><div class="seg">${Object.entries(map).map(([k, l]) => `<label><input type="radio" name="${name}" value="${k}" id="mf-pk-${name}-${k}"${k === def ? ' checked' : ''}><span>${l}</span></label>`).join('')}</div></fieldset>`;
  form.innerHTML = radio('in', IN, 'tab', 'The input is') + radio('out', OUT, 'point', 'The answer should be') + `<fieldset><legend class="kicker">Constraints</legend><div class="seg">${Object.entries(CON).map(([k, l]) => `<label><input type="checkbox" name="con" value="${k}" id="mf-pk-con-${k}"${k === 'ood' || k === 'web' ? ' checked' : ''}><span>${l}</span></label>`).join('')}</div></fieldset>`;
  const PRE: [string, string, string, string[]][] = [
    ['Like QuakeNet', 'sig', 'point', ['cover', 'web']],
    ['Like DemandCast', 'tab', 'traj', ['ood', 'cover', 'web']],
    ['Like PlantDetective', 'set', 'point', ['explain', 'strata', 'web']],
    ['Like StarQuake', 'var', 'post', ['strata', 'cover', 'web']],
    ['GridPulse, live', 'var', 'traj', ['stream', 'small', 'web']],
    ['Like WakeNet', 'graph', 'point', ['ood', 'web']],
  ];
  const pre = q('#mf-pk-presets');
  pre.insertAdjacentHTML('beforeend', PRE.map((p, i) => `<button type="button" class="tog" data-p="${i}" aria-pressed="false">${p[0]}</button>`).join(''));
  const preBtns = qa('[data-p]', pre);
  const runPicker = () => {
    const i = q<HTMLInputElement>('input[name="in"]:checked', form).value, o = q<HTMLInputElement>('input[name="out"]:checked', form).value;
    const cs = qa<HTMLInputElement>('input[name="con"]:checked', form).map((c) => c.value);
    const scored = Object.entries(A).map(([id, a]) => {
      const why: [number, string][] = [];
      const si = (a.in[i] || 0) * 2; if (si) why.push([si, `built for ${IN[i].toLowerCase()}`]);
      const so = a.out[o] || 0; if (so) why.push([so, `gives ${OUT[o].toLowerCase()}`]);
      cs.forEach((c) => { const e = a.c[c]; if (e && e[0]) why.push([e[0], e[1]]); });
      return { id, score: why.reduce((s, w) => s + w[0], 0), why, fit: si };
    }).filter((x) => x.fit > 0 || x.score >= 3).sort((p, r) => r.score - p.score).slice(0, 6);
    const top = Math.max(1, scored[0]?.score || 1);
    q('#mf-pk-rank').innerHTML = scored.map((x, k) => {
      const f = FAM[x.id];
      return `<li><span class="pos num">${String(k + 1).padStart(2, '0')}</span><div><h4><button type="button" class="linkish" data-fam="${x.id}">${esc(f.name)}</button><span class="st st-${f.st}">${f.st === 'y' ? 'in use here' : f.st === 'p' ? 'partly here' : 'not used yet'}</span></h4><ul class="why">${x.why.map(([v, t]) => `<li><span class="${v > 0 ? 'p' : 'm'}">${v > 0 ? '+' : '−'}${Math.abs(v)}</span>${esc(t)}</li>`).join('')}</ul></div><div><span class="num score">${x.score}</span><div class="bar"><i style="width:${Math.max(4, (x.score / top) * 100)}%;background:${fcol(x.id)}"></i></div></div></li>`;
    }).join('');
    qa('#mf-pk-rank [data-fam]').forEach((b) => b.addEventListener('click', () => api.openFamily(b.dataset.fam!)));
    const add: [string, string][] = [];
    if (['point', 'traj', 'cls'].includes(o) || cs.includes('ood')) add.push(['anchor', '<b>Physics anchor.</b> Where a classical or physics estimate exists, predict a correction to it. Every learner above then falls back to physics outside the bank. Ten models here already do this.']);
    if (cs.includes('cover') || cs.includes('strata') || o === 'point') add.push(['conformal', `<b>Conformal widening${cs.includes('strata') ? ', per stratum (Mondrian)' : ''}.</b> One quantile per output from dev residuals: a cousin of <code>mlratchet.calibrate</code>, and coarse on a small dev set.`]);
    if (cs.includes('ood')) add.push(['ensemble', '<b>Two different families side by side.</b> Where they disagree on a real input, the bank probably never covered it. No labels needed (section 05).']);
    if (cs.includes('strata')) add.push(['experts', '<b>A gate across strata,</b> fixed in advance or trained on simulations, never chosen on dev.']);
    q('#mf-pk-addons').innerHTML = add.length ? '<p class="kicker">Add on top</p>' + add.map(([id, t]) => `<p class="addon" style="border-left-color:${fcol(id)}">${t}</p>`).join('') : '';
  };
  preBtns.forEach((b) => b.addEventListener('click', () => {
    const [, i, o, cs] = PRE[+b.dataset.p!];
    q<HTMLInputElement>(`#mf-pk-in-${i}`).checked = true; q<HTMLInputElement>(`#mf-pk-out-${o}`).checked = true;
    qa<HTMLInputElement>('input[name="con"]', form).forEach((c) => { c.checked = cs.includes(c.value); });
    press(preBtns, b); runPicker();
  }));
  form.addEventListener('change', () => { preBtns.forEach((c) => c.setAttribute('aria-pressed', 'false')); runPicker(); });
  runPicker();

  /* ---- 04 catalogue ---- */
  let cg = 'all', cst = 'all';
  const gBtns = qa('#mf-cat-groups [data-g]'), sBtns = qa('#mf-cat-status [data-s]');
  const applyCat = () => qa('#mf-cat .fam').forEach((c) => { c.hidden = !((cg === 'all' || c.dataset.g === cg) && (cst === 'all' || c.dataset.s === cst)); });
  gBtns.forEach((b) => b.addEventListener('click', () => { cg = b.dataset.g!; press(gBtns, b); applyCat(); }));
  sBtns.forEach((b) => b.addEventListener('click', () => { cst = b.dataset.s!; press(sBtns, b); applyCat(); }));
  api.openFamily = (id: string) => {
    cg = 'all'; cst = 'all'; press(gBtns, gBtns[0]); press(sBtns, sBtns[0]); applyCat();
    const card = document.getElementById('fam-' + id); if (!card) return;
    const d = q<HTMLDetailsElement>('details', card); if (d) d.open = true;
    go(card);
  };

  /* ---- 05 extrapolation lab ---- */
  lab();

  /* ---- 06 opportunity map ---- */
  const cellOf = (m: string, f: string): Cell | undefined => CELLS.find((c) => c.m === m && c.f === f);
  const detail = q('#mf-om-detail');
  const cellBtns = qa<HTMLButtonElement>('#mf-om [data-c]');
  const colBtns = qa<HTMLButtonElement>('#mf-om [data-col]');
  let col: string | null = null;
  const markCol = () => {
    colBtns.forEach((b) => b.classList.toggle('on', b.dataset.col === col));
    qa('#mf-om td[data-f]').forEach((td) => td.classList.toggle('col-on', td.dataset.f === col));
  };
  const select = (m: string, f: string) => {
    const c = cellOf(m, f); if (!c) return;
    col = null; markCol();
    cellBtns.forEach((b) => { const on = b.dataset.c === `${m}:${f}`; b.classList.toggle('on', on); b.setAttribute('aria-pressed', String(on)); });
    const M = MOD[m], F = FAM[f];
    const fit: [string, string] = c.inr ? ['Partly in already', 'var(--real)'] : c.s === 3 ? ['Strong fit', 'var(--win)'] : c.s === 2 ? ['Possible fit', 'var(--ink-2)'] : ['Long shot', 'var(--ink-3)'];
    detail.innerHTML = `<p class="fit" style="color:${fit[1]}">${fit[0]}</p><h3 class="display"><span style="color:${fcol(f)}">${esc(F.name)}</span><br>for ${esc(M.name)}</h3><p>${esc(c.why)}</p><div class="gapbox"><b>The gap it targets.</b> ${esc(c.gap)}</div><p class="judge"><b>How it would be judged.</b> ${esc(c.judge)}</p><div class="row"><button type="button" class="btn" data-act="draft">Draft this rung</button><button type="button" class="btn ghost" data-act="fam">About the family</button></div>`;
    q('[data-act="draft"]', detail).addEventListener('click', () => { api.draft(m, f); go(q('#propose')); });
    q('[data-act="fam"]', detail).addEventListener('click', () => api.openFamily(f));
    store.set('cell', `${m}:${f}`);
  };
  const famSummary = (f: string) => {
    const cs = CELLS.filter((c) => c.f === f).sort((a, b) => b.s - a.s), F = FAM[f];
    detail.innerHTML = `<p class="fit" style="color:${fcol(f)}">${esc(GROUPS[F.g].name)} · ${cs.length} ideas</p><h3 class="display">${esc(F.name)}</h3><p class="muted">${esc(F.one)}</p><div class="list">${cs.map((c) => `<button type="button" class="tog" data-c2="${c.m}:${c.f}"><span class="sw sq" style="background:${MOD[c.m].acc}"></span>${esc(MOD[c.m].name)} · ${c.inr ? 'partly in' : c.s === 3 ? 'strong' : c.s === 2 ? 'possible' : 'long shot'}</button>`).join('')}</div><div class="row"><button type="button" class="btn ghost" data-act="fam">About the family</button></div>`;
    qa('[data-c2]', detail).forEach((b) => b.addEventListener('click', () => { const [m, ff] = b.dataset.c2!.split(':'); select(m, ff); }));
    q('[data-act="fam"]', detail).addEventListener('click', () => api.openFamily(f));
  };
  cellBtns.forEach((b) => b.addEventListener('click', () => { const [m, f] = b.dataset.c!.split(':'); select(m, f); }));
  colBtns.forEach((b) => b.addEventListener('click', () => {
    col = col === b.dataset.col ? null : b.dataset.col!; markCol();
    if (col) famSummary(col); else { const [m, f] = (store.get('cell') || 'energynet:gbm').split(':'); select(m, f); }
  }));
  const omBtns = qa('#mf-om-filter [data-v]');
  omBtns.forEach((b) => b.addEventListener('click', () => {
    press(omBtns, b); const strong = b.dataset.v === 'strong';
    cellBtns.forEach((c) => { c.hidden = strong && c.dataset.s !== '3' && c.dataset.inr !== '1'; });
  }));
  api.selectCell = select;
  const saved = (store.get('cell') || '').split(':');
  if (cellOf(saved[0], saved[1])) select(saved[0], saved[1]); else select('energynet', 'gbm');

  /* ---- 07 TreeSHAP real-vs-twin diagnostic ---- */
  let sk = 0;
  const shapBtns = qa('#mf-shap-cases [data-i]');
  const drawShap = () => {
    const c = SHAP_CASES[sk], F = SHAP_FEATURES, W = 760, lw = 165, rw = 74, top = 26, rh = 32, H = top + F.length * rh + 30;
    const allV = [...c.twin, ...c.real, 0];
    const loRaw = Math.min(...allV), hiRaw = Math.max(...allV);
    const hasNeg = loRaw < -0.05;
    const span = Math.max(2, hiRaw - loRaw);
    const step = span <= 6 ? 1 : span <= 15 ? 2 : span <= 35 ? 5 : 10;
    const vmin = hasNeg ? Math.floor(loRaw / step) * step - step * 0.25 : 0;
    const vmax = Math.ceil(hiRaw / step) * step + step * 0.25;
    const xs = (v: number) => lw + ((v - vmin) / (vmax - vmin)) * (W - lw - rw);
    const x0 = xs(0);
    const D = F.map((_, i) => c.real[i] - c.twin[i]);
    const worst = D.reduce((b, d, i) => (Math.abs(d) > Math.abs(D[b]) ? i : b), 0);
    const sigThr = Math.max(0.4, 0.25 * Math.abs(D[worst]));
    let s = `<text x="${lw}" y="12" class="svg-t"><tspan style="fill:var(--sim)">■</tspan> simulated twin   <tspan style="fill:var(--real)">■</tspan> real dev trips</text><text x="${W}" y="12" text-anchor="end" class="svg-t">real − twin</text>`;
    const t0 = Math.ceil(vmin / step) * step;
    for (let v = t0; v <= vmax + 1e-9; v += step) {
      const xv = xs(v);
      const isZero = Math.abs(v) < 1e-6;
      s += `<line x1="${xv}" x2="${xv}" y1="${top - 4}" y2="${top + F.length * rh}" style="stroke:${isZero ? 'var(--ink-3)' : 'var(--rule-2)'};stroke-width:${isZero ? 1.2 : 1}"/>`;
      s += `<text x="${xv}" y="${top + F.length * rh + 12}" text-anchor="middle" class="svg-t">${v > 0 ? '+' : ''}${v}</text>`;
    }
    F.forEach((f, i) => {
      const y = top + i * rh;
      if (i === worst && Math.abs(D[i]) >= sigThr) s += `<rect x="0" y="${y - 3}" width="${W}" height="${rh}" style="fill:var(--paper-2)"/>`;
      s += `<text x="${lw - 10}" y="${y + 15}" text-anchor="end" class="svg-t2">${f}</text>`;
      const xT = xs(c.twin[i]), xR = xs(c.real[i]);
      s += `<rect x="${Math.min(x0, xT)}" y="${y + 2}" width="${Math.max(1.5, Math.abs(xT - x0))}" height="10" style="fill:var(--sim)"/>`;
      s += `<rect x="${Math.min(x0, xR)}" y="${y + 14}" width="${Math.max(1.5, Math.abs(xR - x0))}" height="10" style="fill:var(--real)"/>`;
      s += `<text x="${W}" y="${y + 17}" text-anchor="end" class="svg-t" style="fill:${Math.abs(D[i]) >= sigThr ? 'var(--signal)' : 'var(--ink-3)'}">${D[i] >= 0 ? '+' : '−'}${Math.abs(D[i]).toFixed(1)}%</text>`;
    });
    s += `<text x="${lw}" y="${H - 3}" class="svg-t">exact TreeSHAP contribution to log(E / E_phys) (% energy correction, real VED dev vs RoadSim)</text>`;
    const svg = q<SVGSVGElement>('#mf-shap-svg'); svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.innerHTML = s;
    q('#mf-shap-read').innerHTML = `<b>Measured read.</b> ${esc(c.read)}`;
  };
  shapBtns.forEach((b) => b.addEventListener('click', () => { sk = +b.dataset.i!; press(shapBtns, b); drawShap(); }));
  drawShap();

  /* ---- 09 lenses ---- */
  const lensBtns = qa<HTMLButtonElement>('#mf-lens-list [data-k]'), lensPanels = qa('[data-lens]');
  let li = +(store.get('lens') || 0); if (!(li >= 0 && li < lensBtns.length)) li = 0;
  const showLens = (k: number, focus = false) => {
    li = k; lensBtns.forEach((b, j) => { b.setAttribute('aria-selected', String(j === k)); b.tabIndex = j === k ? 0 : -1; });
    lensPanels.forEach((p, j) => { p.hidden = j !== k; });
    if (focus) lensBtns[k].focus();
    store.set('lens', String(k));
  };
  lensBtns.forEach((b, j) => b.addEventListener('click', () => showLens(j)));
  q('#mf-lens-list').addEventListener('keydown', (e: KeyboardEvent) => {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    e.preventDefault(); showLens((li + (e.key === 'ArrowDown' ? 1 : lensBtns.length - 1)) % lensBtns.length, true);
  });
  showLens(li);

  /* ---- 10 backlog ---- */
  const blBtns = qa('#mf-bl-filter [data-f]');
  blBtns.forEach((b) => b.addEventListener('click', () => {
    press(blBtns, b); const f = b.dataset.f!;
    qa('#mf-bl [data-fams]').forEach((c) => { c.hidden = f !== 'all' && !c.dataset.fams!.split(' ').includes(f); });
  }));

  /* ---- 11 rung drafter ---- */
  const ms = q<HTMLSelectElement>('#mf-prop-model'), fs = q<HTMLSelectElement>('#mf-prop-fam'), ta = q<HTMLTextAreaElement>('#mf-prop-change');
  const defaultChange = (m: string, f: string) => cellOf(m, f)?.change ?? `${FAM[f].name} on the existing simulated bank, with the incumbent's inputs and target`;
  let touched = false;
  const renderProp = () => {
    const m = MOD[ms.value], f = FAM[fs.value], change = ta.value.trim() || defaultChange(m.id, f.id);
    const none = /^None/.test(m.gate);
    const gateLine = none ? `no real dev gate (${m.gate.replace(/^None\s*[:(—-]?\s*/, '').replace(/\)$/, '')}), so the frozen set decides alone` : m.gate;
    const why = `proposal, not run. Keep if the frozen-set skill beats the incumbent by more than 0.01 AND ${none ? 'nothing else: ' + gateLine : gateLine.charAt(0).toLowerCase() + gateLine.slice(1)}`;
    const yq = (s: string) => s.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
    q('#mf-prop-file').textContent = `models/${m.id}/ratchet.yaml`;
    q('#mf-prop-yaml').textContent = `# Proposal only. Append under the current ladder once it has been judged.\n      - { id: try-${f.id}, change: "${yq(change)}", decision: pending, why: "${yq(why)}" }`;
    const items = [
      'Same simulated bank, read from <code>work/</code> through <code>training/_paths.py</code>. Nothing new is committed: no data, banks or checkpoints.',
      'Same target as the incumbent' + (m.anchored ? ', which is physics-anchored; keep that.' : '. Anchoring it on a classical estimate is worth trying first.'),
      'Hyperparameters and early stopping chosen on a simulated split only. Dev never chooses anything.',
      `Frozen set${m.scored ? ` <code>${esc(m.scored)}</code>` : ''}: more than +0.01 of skill over the incumbent.`,
      `Dev gate: ${esc(gateLine)}.`,
      'On the exam: beside the incumbent, the classical method and a dumb baseline built from the context inputs.',
      `Inference in numpy/scipy with no <code>mlratchet</code> import, matching the training library to about 1e-6${f.np < 2 ? '. This family is hard to run in numpy, so whether it could be released at all comes first.' : '.'}`,
      `If released: a port in <code>site/src/instruments/${m.id}/net.ts</code> that passes <code>npm test</code> parity.`,
      'The rung logged, kept or rejected, with its numbers and the reason.',
    ];
    q('#mf-prop-check').innerHTML = items.map((t) => `<li><span>${t}</span></li>`).join('');
  };
  const resetProp = () => { ta.value = defaultChange(ms.value, fs.value); touched = false; renderProp(); };
  ms.addEventListener('change', () => (touched ? renderProp() : resetProp()));
  fs.addEventListener('change', () => (touched ? renderProp() : resetProp()));
  ta.addEventListener('input', () => { touched = true; renderProp(); });
  api.draft = (m: string, f: string) => { ms.value = m; fs.value = f; resetProp(); };
  q('#mf-prop-copy').addEventListener('click', () => {
    const pre = q('#mf-prop-yaml'), ok = () => { const c = q('#mf-prop-copied'); c.hidden = false; setTimeout(() => { c.hidden = true; }, 1600); };
    const fallback = () => { const r = document.createRange(); r.selectNodeContents(pre); const s = getSelection(); s?.removeAllRanges(); s?.addRange(r); };
    try { navigator.clipboard.writeText(pre.textContent || '').then(ok, fallback); } catch { fallback(); }
  });
  api.draft('energynet', 'gbm');
}

/* ---- the extrapolation lab: three learners trained live on a toy simulated bank ---- */
function lab() {
  const phys = (x: number) => 1 + 0.5 * x + 0.06 * x * x;
  const simf = (x: number) => phys(x) + 0.7 * Math.sin(1.4 * x);
  const realf = (x: number, gap: number) => simf(x) + gap * 0.12 * x;
  const rng = (seed: number) => () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const gauss = (r: () => number) => { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); };
  const mean = (a: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; };
  type K = 'gbm' | 'mlp' | 'gp';
  const L: Record<K, { name: string; c: string }> = { gbm: { name: 'Boosted trees', c: 'var(--f-tab)' }, mlp: { name: 'ReLU network', c: 'var(--f-struct)' }, gp: { name: 'Gaussian process', c: 'var(--f-dist)' } };
  const KEYS = Object.keys(L) as K[];
  const S = { lo: 2.5, hi: 6.5, noise: 0.25, gap: 0.3, anchor: false, seed: 11, show: { gbm: true, mlp: true, gp: true } as Record<K, boolean> };
  q('#mf-lab-learners').innerHTML = KEYS.map((k) => `<label><input type="checkbox" id="mf-lab-show-${k}" checked><span class="swatch" style="background:${L[k].c}"></span>${L[k].name}</label>`).join('');

  function fitGBM(x: number[], t: number[]) {
    const n = x.length, idx = x.map((_, i) => i).sort((a, b) => x[a] - x[b]);
    const xs = idx.map((i) => x[i]), ts = idx.map((i) => t[i]);
    const f0 = mean(ts), F = new Float64Array(n).fill(f0), lr = 0.1, minLeaf = 5, stumps: [number, number, number][] = [];
    const P = new Float64Array(n + 1);
    for (let r = 0; r < 250; r++) {
      for (let i = 0; i < n; i++) P[i + 1] = P[i] + (ts[i] - F[i]);
      const tot = P[n]; let best = -1, bk = -1;
      for (let k = minLeaf; k <= n - minLeaf; k++) { if (xs[k - 1] === xs[k]) continue; const sl = P[k], sr = tot - sl, g = (sl * sl) / k + (sr * sr) / (n - k); if (g > best) { best = g; bk = k; } }
      if (bk < 0) break;
      const vl = P[bk] / bk, vr = (tot - P[bk]) / (n - bk);
      for (let i = 0; i < n; i++) F[i] += lr * (i < bk ? vl : vr);
      stumps.push([(xs[bk - 1] + xs[bk]) / 2, vl * lr, vr * lr]);
    }
    return (v: number) => { let s = f0; for (const [thr, a, b] of stumps) s += v < thr ? a : b; return s; };
  }
  function fitMLP(x: number[], t: number[], seed: number, epochs: number) {
    const n = x.length, H = 32, r = rng(seed);
    const mx = mean(x), sx = Math.sqrt(mean(x.map((v) => (v - mx) ** 2))) || 1;
    const mt = mean(t), st = Math.sqrt(mean(t.map((v) => (v - mt) ** 2))) || 1;
    const z = x.map((v) => (v - mx) / sx), y = t.map((v) => (v - mt) / st);
    const w1 = new Float64Array(H), b1 = new Float64Array(H), w2 = new Float64Array(H); let b2 = 0;
    for (let j = 0; j < H; j++) { w1[j] = gauss(r) * 1.5; b1[j] = -w1[j] * (r() * 4 - 2); w2[j] = gauss(r) / Math.sqrt(H); }
    const P = [w1, b1, w2], m = P.map(() => new Float64Array(H)), v = P.map(() => new Float64Array(H)); let mb = 0, vb = 0;
    const G = [new Float64Array(H), new Float64Array(H), new Float64Array(H)], hp = new Float64Array(H);
    const lr = 0.01, B1 = 0.9, B2 = 0.999, eps = 1e-8;
    for (let e = 1; e <= epochs; e++) {
      G.forEach((g) => g.fill(0)); let gb2 = 0;
      for (let i = 0; i < n; i++) {
        let out = b2; const zi = z[i];
        for (let j = 0; j < H; j++) { const a = w1[j] * zi + b1[j]; hp[j] = a; if (a > 0) out += w2[j] * a; }
        const d = out - y[i]; gb2 += d;
        for (let j = 0; j < H; j++) if (hp[j] > 0) { G[2][j] += d * hp[j]; const dh = d * w2[j]; G[0][j] += dh * zi; G[1][j] += dh; }
      }
      const c1 = 1 - Math.pow(B1, e), c2 = 1 - Math.pow(B2, e);
      for (let p = 0; p < 3; p++) for (let j = 0; j < H; j++) { const gr = G[p][j] / n + 1e-4 * P[p][j]; m[p][j] = B1 * m[p][j] + (1 - B1) * gr; v[p][j] = B2 * v[p][j] + (1 - B2) * gr * gr; P[p][j] -= (lr * (m[p][j] / c1)) / (Math.sqrt(v[p][j] / c2) + eps); }
      const gb = gb2 / n; mb = B1 * mb + (1 - B1) * gb; vb = B2 * vb + (1 - B2) * gb * gb; b2 -= (lr * (mb / c1)) / (Math.sqrt(vb / c2) + eps);
    }
    return (q0: number) => { const zq = (q0 - mx) / sx; let o = b2; for (let j = 0; j < H; j++) { const a = w1[j] * zq + b1[j]; if (a > 0) o += w2[j] * a; } return o * st + mt; };
  }
  function fitGP(x: number[], t: number[], noise: number) {
    const n = x.length, m = mean(t), sf2 = Math.max(0.05, mean(t.map((v) => (v - m) ** 2))), l2 = 2 * 0.8 * 0.8, sn2 = noise * noise + 1e-4;
    const Lc = new Float64Array(n * n), K = (a: number, b: number) => sf2 * Math.exp(-((a - b) ** 2) / l2);
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let s = K(x[i], x[j]) + (i === j ? sn2 : 0);
      for (let k = 0; k < j; k++) s -= Lc[i * n + k] * Lc[j * n + k];
      Lc[i * n + j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / Lc[j * n + j];
    }
    const fwd = (b: ArrayLike<number>) => { const y = new Float64Array(n); for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= Lc[i * n + k] * y[k]; y[i] = s / Lc[i * n + i]; } return y; };
    const yv = fwd(t.map((v) => v - m)), alpha = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) { let s = yv[i]; for (let k = i + 1; k < n; k++) s -= Lc[k * n + i] * alpha[k]; alpha[i] = s / Lc[i * n + i]; }
    return (v: number): [number, number] => {
      const ks = x.map((xi) => K(v, xi)); let mu = m; for (let i = 0; i < n; i++) mu += ks[i] * alpha[i];
      const vv = fwd(ks); let s = 0; for (let i = 0; i < n; i++) s += vv[i] * vv[i];
      return [mu, Math.sqrt(Math.max(0, sf2 - s))];
    };
  }

  const W = 760, Hh = 360, ml = 44, mr = 14, mt = 30, mb = 34;
  const xs = (x: number) => ml + (x / 10) * (W - ml - mr);
  let pending = 0;
  function compute(full: boolean) {
    const r = rng(S.seed), re = rng(S.seed * 7 + 3);
    const X: number[] = [], Y: number[] = [], EX: number[] = [], EY: number[] = [];
    for (let i = 0; i < 150; i++) { const x = S.lo + (S.hi - S.lo) * r(); X.push(x); Y.push(simf(x) + S.noise * gauss(r)); }
    for (let i = 0; i < 90; i++) { const x = (10 * (i + re())) / 90; EX.push(x); EY.push(realf(x, S.gap) + 0.5 * S.noise * gauss(re)); }
    const base = (x: number) => (S.anchor ? phys(x) : 0);
    const T = X.map((x, i) => Y[i] - base(x));
    const fg = fitGBM(X, T), fm = fitMLP(X, T, S.seed + 5, full ? 1800 : 700), fp = fitGP(X, T, S.noise);
    const pred: Record<K | 'phys', (x: number) => number> = { gbm: (x) => fg(x) + base(x), mlp: (x) => fm(x) + base(x), gp: (x) => fp(x)[0] + base(x), phys };
    const grid = Array.from({ length: 201 }, (_, i) => i * 0.05);
    draw(X, Y, EX, EY, grid, pred, fp, base);
    table(EX, EY, pred);
  }
  function draw(X: number[], Y: number[], EX: number[], EY: number[], grid: number[], pred: Record<string, (x: number) => number>, fp: (x: number) => [number, number], base: (x: number) => number) {
    const vals = grid.map((x) => realf(x, S.gap)).concat(grid.map(simf), Y, EY);
    const y0 = Math.floor(Math.min(...vals) - 0.8), y1 = Math.ceil(Math.max(...vals) + 0.8);
    const ys = (y: number) => mt + (1 - (y - y0) / (y1 - y0)) * (Hh - mt - mb);
    const line = (f: (x: number) => number) => grid.map((x, i) => `${i ? 'L' : 'M'}${xs(x).toFixed(1)} ${ys(f(x)).toFixed(1)}`).join('');
    const step = y1 - y0 > 12 ? 4 : 2;
    let s = `<defs><clipPath id="mf-lab-clip"><rect x="${ml}" y="${mt}" width="${W - ml - mr}" height="${Hh - mt - mb}"/></clipPath></defs>`;
    s += `<rect x="${xs(S.lo)}" y="${mt}" width="${xs(S.hi) - xs(S.lo)}" height="${Hh - mt - mb}" style="fill:var(--sim);fill-opacity:.07"/>`;
    s += `<text x="${(xs(S.lo) + xs(S.hi)) / 2}" y="${mt + 12}" text-anchor="middle" class="svg-t" style="fill:var(--sim)">simulated bank</text>`;
    for (let v = Math.ceil(y0 / step) * step; v <= y1; v += step) s += `<line x1="${ml}" x2="${W - mr}" y1="${ys(v)}" y2="${ys(v)}" style="stroke:var(--rule-2)"/><text x="${ml - 8}" y="${ys(v) + 3}" text-anchor="end" class="svg-t">${v}</text>`;
    for (let v = 0; v <= 10; v += 2) s += `<text x="${xs(v)}" y="${Hh - mb + 16}" text-anchor="middle" class="svg-t">${v}</text>`;
    s += `<text x="${W - mr}" y="${Hh - 4}" text-anchor="end" class="svg-t">input: a physical cause, arbitrary units</text><g clip-path="url(#mf-lab-clip)">`;
    if (S.show.gp) {
      const band = grid.map((x) => { const [mu, sd] = fp(x); return [x, mu + base(x) + sd, mu + base(x) - sd]; });
      s += `<path d="${band.map(([x, a], i) => `${i ? 'L' : 'M'}${xs(x).toFixed(1)} ${ys(a).toFixed(1)}`).join('')}${band.slice().reverse().map(([x, , b]) => `L${xs(x).toFixed(1)} ${ys(b).toFixed(1)}`).join('')}Z" style="fill:var(--f-dist);fill-opacity:.12"/>`;
    }
    s += `<path d="${line(simf)}" style="fill:none;stroke:var(--sim);stroke-opacity:.45;stroke-width:1.2"/><path d="${line((x) => realf(x, S.gap))}" style="fill:none;stroke:var(--ink);stroke-width:1.2;stroke-dasharray:4 4"/>`;
    if (S.anchor) s += `<path d="${line(phys)}" style="fill:none;stroke:var(--ink-3);stroke-width:1.2;stroke-dasharray:1 4"/>`;
    X.forEach((x, i) => { s += `<circle cx="${xs(x).toFixed(1)}" cy="${ys(Y[i]).toFixed(1)}" r="2.2" style="fill:var(--sim);fill-opacity:.55"/>`; });
    EX.forEach((x, i) => { s += `<circle cx="${xs(x).toFixed(1)}" cy="${ys(EY[i]).toFixed(1)}" r="2.6" style="fill:none;stroke:var(--real);stroke-width:1.3"/>`; });
    KEYS.forEach((k) => { if (S.show[k]) s += `<path d="${line(pred[k])}" style="fill:none;stroke:${L[k].c};stroke-width:2.4"/>`; });
    s += `</g><rect x="${ml}" y="${mt}" width="${W - ml - mr}" height="${Hh - mt - mb}" style="fill:none;stroke:var(--rule)"/>`;
    const leg: [string, string, string][] = [['dot', 'var(--sim)', 'simulated case'], ['ring', 'var(--real)', 'real exam case'], ['dash', 'var(--ink)', 'reality'], ['line', 'var(--sim)', 'simulator']];
    if (S.anchor) leg.push(['dot2', 'var(--ink-3)', 'physics estimate']);
    let lx = ml; const ly = 14;
    leg.forEach(([t, c, l]) => {
      s += t === 'dot' ? `<circle cx="${lx + 4}" cy="${ly - 3}" r="3" style="fill:${c}"/>` : t === 'ring' ? `<circle cx="${lx + 4}" cy="${ly - 3}" r="3" style="fill:none;stroke:${c};stroke-width:1.3"/>` : `<line x1="${lx}" x2="${lx + 14}" y1="${ly - 3}" y2="${ly - 3}" style="stroke:${c};stroke-width:1.4;${t === 'dash' ? 'stroke-dasharray:4 3' : t === 'dot2' ? 'stroke-dasharray:1 3' : 'stroke-opacity:.5'}"/>`;
      s += `<text x="${lx + 18}" y="${ly}" class="svg-t">${l}</text>`; lx += 34 + l.length * 7;
    });
    q('#mf-lab-svg').innerHTML = s;
    const D = grid.map((x) => Math.abs(pred.gbm(x) - pred.mlp(x)));
    const dmax = Math.max(1, ...D), dh = 92, dt = 18, db = 10, dy = (d: number) => dt + (1 - d / dmax) * (dh - dt - db);
    const inside = D.filter((_, i) => grid[i] >= S.lo && grid[i] <= S.hi), outside = D.filter((_, i) => grid[i] < S.lo || grid[i] > S.hi);
    let d = `<rect x="${xs(S.lo)}" y="${dt}" width="${xs(S.hi) - xs(S.lo)}" height="${dh - dt - db}" style="fill:var(--sim);fill-opacity:.07"/>`;
    d += `<path d="M${xs(0)} ${dy(0)}${grid.map((x, i) => `L${xs(x).toFixed(1)} ${dy(D[i]).toFixed(1)}`).join('')}L${xs(10)} ${dy(0)}Z" style="fill:var(--signal);fill-opacity:.18;stroke:var(--signal);stroke-width:1.4"/>`;
    d += `<line x1="${ml}" x2="${W - mr}" y1="${dy(0)}" y2="${dy(0)}" style="stroke:var(--rule)"/><text x="${ml}" y="10" class="svg-t" style="fill:var(--ink-2)">|trees − network|: disagreement, no labels needed</text>`;
    d += `<text x="${W - mr}" y="10" text-anchor="end" class="svg-t">mean inside the bank ${mean(inside).toFixed(2)} · outside ${outside.length ? mean(outside).toFixed(2) : '—'}</text>`;
    q('#mf-lab-dis').innerHTML = d;
  }
  function table(EX: number[], EY: number[], pred: Record<string, (x: number) => number>) {
    const inn = (x: number) => x >= S.lo && x <= S.hi;
    const rmse = (f: (x: number) => number, sel: (x: number) => boolean) => { const e = EX.map((x, i) => [x, EY[i]] as const).filter(([x]) => sel(x)); return e.length ? Math.sqrt(mean(e.map(([x, y]) => (f(x) - y) ** 2))) : null; };
    const rows = ([['phys', 'Physics estimate alone', 'var(--ink-3)']] as [string, string, string][]).concat(KEYS.filter((k) => S.show[k]).map((k) => [k, L[k].name, L[k].c]));
    const R = rows.map(([k, n, c]) => ({ k, n, c, a: rmse(pred[k], inn), b: rmse(pred[k], (x) => !inn(x)) }));
    const learners = R.filter((x) => x.k !== 'phys');
    const bestA = Math.min(...learners.map((x) => x.a ?? Infinity)), bestB = Math.min(...learners.map((x) => x.b ?? Infinity));
    const f = (v: number | null, best: number) => (v == null ? '—' : `<span${Math.abs(v - best) < 1e-9 ? ' class="best"' : ''}>${v.toFixed(2)}</span>`);
    q('#mf-lab-table').innerHTML = `<thead><tr><th>Learner${S.anchor ? ' (anchored on physics)' : ''}</th><th class="num">Exam RMSE inside the bank</th><th class="num">Outside the bank</th><th class="num">Outside ÷ inside</th></tr></thead><tbody>${R.map((x) => `<tr><td><span class="swatch" style="background:${x.c}"></span>${x.n}</td><td class="num">${f(x.a, bestA)}</td><td class="num">${f(x.b, bestB)}</td><td class="num">${x.a && x.b ? `${(x.b / x.a).toFixed(1)}×` : '—'}</td></tr>`).join('')}</tbody>`;
  }
  const ctl = q('#mf-lab-ctl');
  const inp = (id: string) => q<HTMLInputElement>(id);
  const sync = () => {
    inp('#mf-lab-lo').value = String(S.lo); inp('#mf-lab-hi').value = String(S.hi); inp('#mf-lab-noise').value = String(S.noise); inp('#mf-lab-gap').value = String(S.gap); inp('#mf-lab-anchor').checked = S.anchor;
    q('#mf-o-lo').textContent = S.lo.toFixed(1); q('#mf-o-hi').textContent = S.hi.toFixed(1); q('#mf-o-noise').textContent = S.noise.toFixed(2); q('#mf-o-gap').textContent = S.gap.toFixed(2);
    KEYS.forEach((k) => { inp(`#mf-lab-show-${k}`).checked = S.show[k]; });
  };
  const read = (e?: Event) => {
    S.lo = +inp('#mf-lab-lo').value; S.hi = +inp('#mf-lab-hi').value;
    if (S.hi - S.lo < 1) { if ((e?.target as HTMLElement | undefined)?.id === 'mf-lab-lo') S.hi = Math.min(10, S.lo + 1); else S.lo = Math.max(0, S.hi - 1); }
    S.noise = +inp('#mf-lab-noise').value; S.gap = +inp('#mf-lab-gap').value; S.anchor = inp('#mf-lab-anchor').checked;
    KEYS.forEach((k) => { S.show[k] = inp(`#mf-lab-show-${k}`).checked; });
    sync();
  };
  const schedule = (full: boolean) => { cancelAnimationFrame(pending); pending = requestAnimationFrame(() => compute(full)); };
  const PRE: [string, Partial<typeof S>][] = [
    ['Narrow bank', { lo: 3.5, hi: 6, gap: 0.3, anchor: false }],
    ['Narrow bank + anchor', { lo: 3.5, hi: 6, gap: 0.3, anchor: true }],
    ['Wide bank', { lo: 0.5, hi: 9.5, gap: 0.3, anchor: false }],
    ['Large sim-to-real gap', { lo: 1, hi: 9, gap: 1, anchor: true }],
  ];
  const pre = q('#mf-lab-presets');
  pre.insertAdjacentHTML('beforeend', PRE.map((p, i) => `<button type="button" class="tog" data-p="${i}" aria-pressed="false">${p[0]}</button>`).join(''));
  const preBtns = qa('[data-p]', pre);
  ctl.addEventListener('input', (e) => { read(e); schedule(false); preBtns.forEach((c) => c.setAttribute('aria-pressed', 'false')); });
  ctl.addEventListener('change', (e) => { read(e); schedule(true); });
  q('#mf-lab-seed').addEventListener('click', () => { S.seed = (S.seed * 31 + 7) % 100003; schedule(true); });
  preBtns.forEach((b) => b.addEventListener('click', () => { Object.assign(S, PRE[+b.dataset.p!][1]); sync(); press(preBtns, b); schedule(true); }));
  sync(); compute(true);
}
