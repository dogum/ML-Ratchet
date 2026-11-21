// FirePilot live instrument: three real exam fires at their first afternoon detection, as the planner saw them (the
// forecast FireSim run), and the value network scoring containment-line plans in the browser. Change the crew budget or
// draw a plan (build order, centre, width, offsets) and the line, its build times and the network's predicted share of
// the no-action growth stopped update live; "Re-plan" runs the planner's search. Outcomes are graded in simulation:
// the exam's truth-world scores (FireSim, hidden spread error) are shown for the exam plans at the exam budgets only.
import type { InstrumentContext } from '../types.ts';
import { fetchPack } from '../../lib/nn/pack.ts';
import { shell, picker, slider, button, scheduler, gauges, canvas, label, rng, hex, h, SCOPE, type Canvas2D } from '../kit.ts';
import { BASELINES, Context, HORIZON, fire, planBrowser, predict, valueNet, wrap180, type Fire, type FiresMeta, type NetMeta, type Plan, type PlanEval } from './net.ts';

const BUDGETS = [100, 150, 200, 300, 450, 650, 1000, 1500, 2000];
const NAMES: Record<string, string> = {
  firepilot: 'FirePilot (exam plan)', anchor_flank: 'Anchor and flank', direct_head: 'Direct attack at the head',
  indirect_1km: 'Indirect 1 km', indirect_2km: 'Indirect 2 km', mpc_forecast: 'MPC, forecast search', pf_bound: 'Perfect-foresight bound',
  none: 'No action', browser: 'Browser re-plan', custom: 'Your plan',
};
const AMBER = '#f2b84b', SIM = [111, 184, 214] as const;   // the forecast run in the page's 'sim' blue
const STYLE = `
.fp-inst .fp-info { font: 11px/1.5 var(--mono, ui-monospace, monospace); color: var(--scr-2ink); margin-top: 6px; }
.fp-inst .fp-info b { color: var(--scr-ink); font-weight: 600; }
.fp-inst .fp-tbl { width: 100%; border-collapse: collapse; margin: 6px 0 2px; font: 10.5px/1.45 var(--mono, ui-monospace, monospace); }
.fp-inst .fp-tbl th { font-weight: 400; text-align: right; color: var(--scr-2ink); padding: 1px 0 1px 6px; font-size: 9.5px; }
.fp-inst .fp-tbl td { text-align: right; color: var(--scr-ink); padding: 1px 0 1px 6px; }
.fp-inst .fp-tbl td:first-child, .fp-inst .fp-tbl th:first-child { text-align: left; padding-left: 0; color: var(--scr-2ink); }
.fp-inst .fp-tbl tr.me td { color: var(--fp-c); font-weight: 600; }
.fp-inst .fp-note { font-size: 10px; color: var(--scr-3ink, var(--scr-2ink)); margin-top: 4px; }
.fp-inst .fp-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 8px; }`;

const km = (m: number) => (m >= 1000 ? `${(m / 1000).toFixed(m < 3000 ? 2 : 1)} km` : `${Math.round(m)} m`);
const same = (a: Plan, b: Plan) => a.mode === b.mode && (['phi_c', 'omega', 'd_h', 'd_f'] as const).every((k) => Math.abs(a[k] - b[k]) < 1e-6);

export async function mount(ctx: InstrumentContext) {
  const [WP, FP] = await Promise.all([fetchPack<NetMeta>(`${ctx.base}weights`), fetchPack<FiresMeta>(`${ctx.base}fires`)]);
  const net = valueNet(WP);
  const fires: Fire[] = FP.meta.fires.map((_, i) => fire(FP, i));
  if (!document.getElementById('fp-inst-style')) document.head.append(h('style', { id: 'fp-inst-style' }, STYLE));
  const ui = shell(ctx.root, {
    title: 'FirePilot',
    subtitle: 'a real exam fire at first detection, the planner’s forecast, and the network ranking line plans · graded in simulation',
    params: `${net.params.toLocaleString('en-US')} weights · matches Python to 1e-15`,
  });
  ctx.root.classList.add('fp-inst');
  ctx.root.style.setProperty('--fp-c', ctx.accent);

  const st = { fi: 1, bi: 3, plan: { ...fires[1].meta.budgets['300'].firepilot_plan } as Plan, which: 'firepilot', seed: 1 };
  const ctxs = new Map<string, Context>();
  const getCtx = () => {
    const key = `${st.fi}|${BUDGETS[st.bi]}`;
    if (!ctxs.has(key)) ctxs.set(key, new Context(fires[st.fi], BUDGETS[st.bi]));
    return ctxs.get(key)!;
  };
  let view: { c: Context; pe: PlanEval } | null = null;

  picker(ui.picker, fires.map((f, i) => ({
    label: `${f.meta.name}`,
    sub: `${f.meta.period === 'recent' ? '2022-24' : '2017-21'} · detected ${Math.round(f.meta.a0_ha)} ha · ${f.nx} × ${f.ny} cells of 90 m`,
    value: i,
  })), (i) => { st.fi = i; examPlan(); }, st.fi);

  const plot = canvas(ui.screen, ctx.compact ? 1.5 : 1.25, (c) => draw(c));

  // ---- controls: the budget, the plan's five numbers, and three ways to set them
  const examB = () => (BUDGETS[st.bi] === 300 ? '300' : BUDGETS[st.bi] === 1000 ? '1000' : null);
  const sB = slider(ui.controls, { label: 'Crew budget', min: 0, max: BUDGETS.length - 1, step: 1, value: st.bi,
    fmt: (v) => `${BUDGETS[v]} m/h · ${(BUDGETS[v] * 12 / 1000).toFixed(1)} km per shift${BUDGETS[v] === 300 || BUDGETS[v] === 1000 ? ' (exam)' : ''}`,
    hint: 'metres of line per working hour, two crews, 08:00-20:00 local, from 2 h after detection',
    onInput: (v) => { st.bi = v; if (st.which === 'firepilot' || st.which === 'browser') { const b = examB(); if (b) setPlan(fires[st.fi].meta.budgets[b].firepilot_plan, 'firepilot'); else st.which = 'custom'; } go(); } });
  const row = h('div', { class: 'fp-row' });
  ui.controls.append(row);
  const bMode = button(row, '', () => { st.plan.mode = 1 - st.plan.mode; st.which = 'custom'; sync(); go(); });
  button(row, 'FirePilot’s exam plan', () => examPlan());
  button(row, 'Anchor and flank', () => { setPlan(BASELINES.anchor_flank, 'anchor_flank'); go(); });
  const bRe = button(row, 'Re-plan in the browser', () => { st.seed++; replan(); });
  const sPhi = slider(ui.controls, { label: 'Line centre', min: -180, max: 180, step: 1, value: 0,
    fmt: (v) => `${v > 0 ? '+' : ''}${v}° from the forecast head`, onInput: (v) => edit('phi_c', v) });
  const sOm = slider(ui.controls, { label: 'Half-width', min: 5, max: 180, step: 1, value: 90,
    fmt: (v) => (v >= 180 ? '180° (a full ring)' : `${v}° each side`), onInput: (v) => edit('omega', v) });
  const lg = (v: number) => Math.exp(Math.log(90) + v * (Math.log(8000) - Math.log(90)));
  const ilg = (d: number) => (Math.log(d) - Math.log(90)) / (Math.log(8000) - Math.log(90));
  const sDh = slider(ui.controls, { label: 'Offset at the centre', min: 0, max: 1, step: 0.005, value: 0,
    fmt: (v) => `${km(lg(v))} from the fire edge`, onInput: (v) => edit('d_h', lg(v)) });
  const sDf = slider(ui.controls, { label: 'Offset at the far side', min: 0, max: 1, step: 0.005, value: 0,
    fmt: (v) => `${km(lg(v))}`, hint: 'the offset blends from the centre value to this one around the fire', onInput: (v) => edit('d_f', lg(v)) });

  function edit(k: 'phi_c' | 'omega' | 'd_h' | 'd_f', v: number) { st.plan[k] = v; st.which = 'custom'; bMode.textContent = modeLabel(); go(); }
  const modeLabel = () => `Build order: ${st.plan.mode === 0 ? 'centre-out' : 'ends-in (from anchors)'}`;
  function sync() {
    const p = st.plan;
    bMode.textContent = modeLabel();
    sPhi.set(Math.round(wrap180(p.phi_c))); sOm.set(Math.round(p.omega)); sDh.set(ilg(p.d_h)); sDf.set(ilg(p.d_f));
  }
  function setPlan(p: Plan, which: string) { st.plan = { ...p }; st.which = which; sync(); }
  function examPlan() {
    if (!examB()) { st.bi = 3; sB.set(3); }
    setPlan(fires[st.fi].meta.budgets[examB()!].firepilot_plan, 'firepilot');
    go();
  }
  function replan() {
    ui.setBusy(true);
    bRe.disabled = true;
    requestAnimationFrame(() => setTimeout(() => {
      const t0 = performance.now();
      const { best, n } = planBrowser(getCtx(), net, rng(1000 * st.fi + st.seed));
      setPlan(best.p, 'browser');
      bRe.disabled = false;
      run(`browser search: ${n} plans scored in ${Math.round(performance.now() - t0)} ms (the same search as Python’s, with different random numbers, so picks can differ)`);
    }, 0));
  }

  // ---- readout
  const gBox = h('div', {}), info = h('div', { class: 'fp-info' });
  ui.readout.append(gBox, info);

  function truthKey(): string | null {
    const b = examB();
    if (!b) return null;
    if (same(st.plan, fires[st.fi].meta.budgets[b].firepilot_plan)) return 'firepilot';
    for (const [k, p] of Object.entries(BASELINES)) if (same(st.plan, p) && k in fires[st.fi].meta.budgets[b].truth_saved) return k;
    return null;
  }

  function run(note?: string) {
    ui.setBusy(true);
    const t0 = performance.now();
    const c = getCtx();
    const { x, pe } = c.features(st.plan);
    const { mean, sigma } = predict(net, x);
    const ms = performance.now() - t0;
    view = { c, pe };
    plot.redraw();
    const b = examB(), tk = truthKey(), F = fires[st.fi].meta;
    const truth = b && tk ? F.budgets[b].truth_saved[tk] : null;
    const L = pe.L, nl = L.idx.length;
    let built = 0, before = 0, overrun = 0;
    for (let k = 0; k < nl; k++) {
      if (Number.isFinite(L.tb[k])) built++;
      if (pe.won[k]) { before++; if (!pe.hold[k]) overrun++; }
    }
    gauges(gBox, [
      { label: 'Predicted share of the no-action growth stopped', value: mean, lo: Math.max(0, mean - sigma), hi: Math.min(1, mean + sigma),
        truth, truthLabel: `exam, FireSim truth world (${NAMES[tk ?? ''] ?? ''})`, range: [0, 1] },
      { label: 'Connectivity proxy on the forecast run (a feature, no learning)', value: pe.s_proxy, range: [0, 1] },
      { label: 'Line built within 72 h', unit: 'km', value: pe.km, range: [0, Math.max(10, Math.ceil(pe.km / 10) * 10)], digits: 1 },
    ]);
    const rows = b ? ['none', 'direct_head', 'indirect_1km', 'anchor_flank', 'mpc_forecast', 'firepilot', 'pf_bound'] : [];
    info.innerHTML = `<div><b>${nl}</b> line cells (8-connected) · <b>${built}</b> finished by 72 h · <b>${before}</b> finished before the forecast fire reaches them${overrun ? ` · ${overrun} overrun (flames > 2.4 m)` : ''}</div>
      <div>Forecast without action: ${Math.round(F.forecast_none_ha).toLocaleString('en-US')} ha by 72 h from ${Math.round(F.a0_ha)} ha detected · σ ${sigma.toFixed(2)}${st.which === 'custom' ? '' : ` · plan: ${NAMES[st.which] ?? st.which}`}</div>`
      + (b ? `<table class="fp-tbl"><tr><th>Exam at ${b} m/h, simulated</th><th>this fire</th><th>48 fires</th></tr>${rows.map((k) =>
        `<tr class="${k === tk ? 'me' : ''}"><td>${NAMES[k]}</td><td>${F.budgets[b].truth_saved[k].toFixed(2)}</td><td>${k === 'none' ? '0' : FP.meta.exam_means[b][k].toFixed(3)}</td></tr>`).join('')}</table>
        <div class="fp-note">Share of the no-action 72 h growth stopped, scored once in FireSim’s truth world (a hidden spread-rate error the planner never sees). These need FireSim, so they are stored, not recomputed; the network’s number above is live.</div>`
        : `<div class="fp-note">${BUDGETS[st.bi]} m/h was not an exam budget: no simulated outcome to compare with.</div>`);
    ui.stamp(ms, note ?? `${F.name} (${F.id}) · ${BUDGETS[st.bi]} m/h`);
    ui.setBusy(false);
  }
  const go = scheduler(() => run(), 40);

  // ---- the map: fuel cost, the forecast's arrival time, the detected fire, the line and its build state
  let img: { key: string; cv: HTMLCanvasElement } | null = null;
  function base(c: Context): HTMLCanvasElement {
    const key = `${st.fi}`;
    if (img && img.key === key) return img.cv;
    const { nx, ny } = c, f = c.fire, im = new ImageData(nx, ny);
    for (let i = 0; i < nx * ny; i++) {
      const k = i * 4, cost = f.cost[i], a = f.arr[i];
      let r = 20, g = 23, b = 20;
      if (cost === 0) { r = 34; g = 36; b = 40; } else if (cost < 0.9) { r = 38; g = 40; b = 30; } else if (cost < 1.5) { r = 26; g = 38; b = 28; } else { r = 42; g = 38; b = 28; }
      if (f.burned0[i]) { [r, g, b] = [235, 230, 216]; } else if (a === a && a <= HORIZON) {
        const t = 1 - a / HORIZON, al = 0.2 + 0.5 * t;
        r = r + (SIM[0] - r) * al; g = g + (SIM[1] - g) * al; b = b + (SIM[2] - b) * al;
      }
      im.data[k] = r; im.data[k + 1] = g; im.data[k + 2] = b; im.data[k + 3] = 255;
    }
    const cv = document.createElement('canvas');
    cv.width = nx; cv.height = ny;
    cv.getContext('2d')!.putImageData(im, 0, 0);
    img = { key, cv };
    return cv;
  }

  function draw(cc: Canvas2D) {
    const { ctx: g, w, h: hh } = cc;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, w, hh);
    if (!view) return;
    const { c, pe } = view, { nx, ny } = c, L = pe.L;
    // view box: the forecast burn and the line, padded, so small fires are not lost in a 15 km window
    let r0 = ny, r1 = -1, c0 = nx, c1 = -1;
    const grow = (i: number) => { const r = Math.floor(i / nx), q = i % nx; if (r < r0) r0 = r; if (r > r1) r1 = r; if (q < c0) c0 = q; if (q > c1) c1 = q; };
    for (let i = 0; i < nx * ny; i++) if (c.fburn[i] || c.fire.burned0[i]) grow(i);
    for (const i of L.idx) grow(i);
    const pad = Math.max(6, Math.round(0.2 * Math.max(r1 - r0, c1 - c0)));
    r0 = Math.max(0, r0 - pad); r1 = Math.min(ny - 1, r1 + pad); c0 = Math.max(0, c0 - pad); c1 = Math.min(nx - 1, c1 + pad);
    const padT = 18, padB = 30, vw = c1 - c0 + 1, vh = r1 - r0 + 1;
    const s = Math.min((w - 16) / vw, (hh - padT - padB) / vh);
    const x0 = (w - s * vw) / 2, y0 = padT + (hh - padT - padB - s * vh) / 2;
    g.imageSmoothingEnabled = false;
    g.drawImage(base(c), c0, r0, vw, vh, x0, y0, s * vw, s * vh);
    const X = (q: number) => x0 + (q - c0) * s, Y = (r: number) => y0 + (r - r0) * s;
    // line cells: finished before the forecast fire arrives (accent), finished late (amber), not finished in 72 h (grey)
    const [ar, ag, ab] = hex(ctx.accent);
    for (let k = 0; k < L.idx.length; k++) {
      const i = L.idx[k], r = Math.floor(i / nx), q = i % nx;
      g.fillStyle = !Number.isFinite(L.tb[k]) ? 'rgba(235,230,216,0.4)' : pe.won[k] ? (pe.hold[k] ? `rgb(${ar},${ag},${ab})` : '#ffffff') : AMBER;
      g.fillRect(X(q), Y(r), Math.max(s, 1), Math.max(s, 1));
    }
    // forecast head direction from the burned centroid
    let sr = 0, sc = 0, nb = 0;
    for (let i = 0; i < nx * ny; i++) if (c.fire.burned0[i]) { sr += Math.floor(i / nx); sc += i % nx; nb++; }
    const cx = X(sc / nb + 0.5), cy = Y(sr / nb + 0.5), az = (c.fire.head * Math.PI) / 180, R = Math.min(vw, vh) * s * 0.3;
    const ex = cx + R * Math.sin(az), ey = cy - R * Math.cos(az);
    g.strokeStyle = SCOPE.ink; g.lineWidth = 1.5; g.setLineDash([4, 3]);
    g.beginPath(); g.moveTo(cx, cy); g.lineTo(ex, ey); g.stroke(); g.setLineDash([]);
    label(cc, 'forecast head', ex + 4 * Math.sin(az), ey - 6, SCOPE.ink, Math.sin(az) >= 0 ? 'left' : 'right', 9);
    // scale bar and key
    const spanKm = (vw * c.res) / 1000, barKm = spanKm > 12 ? 5 : spanKm > 4 ? 2 : spanKm > 2 ? 1 : 0.5, bar = (barKm * 1000 / c.res) * s;
    g.strokeStyle = SCOPE.text; g.lineWidth = 1; g.beginPath(); g.moveTo(x0, hh - 8.5); g.lineTo(x0 + bar, hh - 8.5); g.stroke();
    label(cc, `${barKm} km · N up`, x0 + bar + 6, hh - 9, SCOPE.text, 'left', 9);
    label(cc, 'detected', x0, 8, SCOPE.ink, 'left', 9);
    label(cc, 'forecast burn by 72 h, no line (brighter = sooner)', x0 + 58, 8, `rgb(${SIM.join(',')})`, 'left', 9);
    const key: [string, string][] = [['line in time', ctx.accent], ['late', AMBER], ['unfinished', 'rgba(235,230,216,0.6)']];
    let kx = w - 8;
    for (const [t, col] of key.reverse()) { const tw = t.length * 6 + 14; kx -= tw; g.fillStyle = col; g.fillRect(kx, hh - 13, 8, 8); label(cc, t, kx + 11, hh - 9, SCOPE.text, 'left', 9); }
  }

  sync();
  run();
}
