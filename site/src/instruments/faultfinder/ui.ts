// FaultFinder live instrument: twelve real exam days (Iowa ERS, LBNL FLEXLAB, ORNL's rooftop unit, Nesbitt Hall),
// never used in training or for any choice. The browser recomputes the 35 feature rows and the network from the raw
// trend log on every change. Controls change the log the way a fault or a bad sensor would: a constant offset on one
// temperature sensor, the outdoor-air damper stuck (the mixed-air temperature rewritten from the mixing-box balance
// while the logged command stays as it was), extra sensor noise, or the mixed-air sensor removed. The dataset's own
// label is the truth; the rule baseline's verdict on the real day is shown next to it.
import type { InstrumentContext } from '../types.ts';
import { fetchPack } from '../../lib/nn/pack.ts';
import { read, perturb, R, NT, type FMeta, type FDay, type Perturb, type Reading } from './net.ts';
import { shell, picker, slider, button, scheduler, gauges, probs, canvas, gridLines, label, h, fmt, SCOPE, type Canvas2D } from '../kit.ts';

const STYLE = `
.ff-seg { display: flex; flex-direction: column; gap: 6px; }
.ff-seg > span { font-size: 12.5px; color: var(--scr-2ink); }
.ff-seg > div { display: flex; gap: 4px; }
.ff-seg button { flex: 1 1 auto; min-width: 0; padding: 5px 4px; border-radius: 3px; border: 1px solid var(--scr-rule); background: transparent; color: var(--scr-2ink); font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.04em; cursor: pointer; }
.ff-seg button[aria-pressed='true'] { border-color: var(--c); color: var(--scr); background: var(--c); }
.ff-seg button:disabled { opacity: 0.35; cursor: default; }
.ff-btns { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 8px; }
.ff-btns .inst-btn[aria-pressed='false'] { color: var(--scr-3ink); border-color: var(--scr-rule); text-decoration: line-through; }
.ff-kv { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 3px 10px; font-size: 12px; color: var(--scr-2ink); }
.ff-kv b { font-family: var(--mono); font-variation-settings: 'wdth' 87.5; font-weight: 400; font-size: 11.5px; color: var(--scr-ink); text-align: right; white-space: nowrap; }
.ff-kv b.hot { color: #e07a5f; }
.ff-kv b.ok { color: #7fc29b; }
.ff-note { font-size: 12px; line-height: 1.5; color: var(--scr-3ink); margin: 0; }
.ff-note b { font-weight: 500; color: var(--scr-2ink); }
`;

const SENSORS: [string, string][] = [['T_oa', 'OA'], ['T_ra', 'RA'], ['T_ma', 'MA'], ['T_sa', 'SA'], ['T_zone', 'Zone']];
const TCOL: Record<string, string> = { T_oa: '#9cbd6f', T_ra: '#b48bd2', T_ma: '#d8ac44', T_sa: '#5ba8df', T_sa_sp: '#5ba8df', T_zone: '#9aa8ad' };
const ACOL: Record<string, string> = { damper_oa: '#9cbd6f', valve_cc: '#5ba8df', valve_hc: '#ec7c52', fan_speed: '#ebe6d8' };
const ANAME: Record<string, string> = { damper_oa: 'OA damper', valve_cc: 'cooling', valve_hc: 'heating', fan_speed: 'fan' };
const SHORT: Record<string, string> = {
  'Iowa ERS (lab)': 'Iowa ERS', 'LBNL FLEXLAB (lab)': 'FLEXLAB', 'ORNL RTU (lab)': 'ORNL RTU', 'Nesbitt Hall (real building)': 'Nesbitt Hall',
};
const pretty = (s: string) => s.replace(/_/g, ' ').replace(/\bT (oa|sa|ma|ra|zone|chw)\b/g, (_, k) => `T_${k}`);

export async function mount(ctx: InstrumentContext) {
  const [W, X] = await Promise.all([fetchPack<FMeta>(`${ctx.base}weights`), fetchPack<FDay[]>(`${ctx.base}examples`)]);
  const days = X.meta;
  if (!document.getElementById('ff-inst-style')) document.head.append(h('style', { id: 'ff-inst-style' }, STYLE));
  const ui = shell(ctx.root, {
    title: 'FaultFinder',
    subtitle: 'is this air handler faulty today? one real exam day of trend log, never used in training',
    params: `${Math.round(W.meta.n_params / 1000)}k parameters · matches Python to 1e-5`,
  });
  const st = { i: 0, sensor: 'T_sa', offset: 0, damper: -5, noise: 0, ma: true };
  const base: Record<number, number> = {};
  let view: { r: Reading; x: Float64Array; x0: Float64Array } | null = null;

  picker(ui.picker, days.map((d, i) => ({
    label: `${SHORT[d.source]} · ${d.unit}`,
    sub: `${d.date} · ${d.fault === 'none' ? 'no fault' : pretty(d.fault)}`,
    value: i,
  })), (i) => { st.i = i; sync(); go(); }, 0);

  const narrow = () => ui.screen.getBoundingClientRect().width < 520;
  const plot = canvas(ui.screen, ctx.compact ? 1.9 : narrow() ? 1.1 : 1.6, (c) => draw(c));

  // ---- controls: what the trend log is given
  const seg = h('div', { class: 'ff-seg', role: 'group', 'aria-label': 'Sensor to offset' }, h('span', {}, 'Offset this temperature sensor'));
  const row = h('div', {});
  const sbtns = SENSORS.map(([c, n]) => {
    const b = h('button', { type: 'button', 'aria-pressed': String(c === st.sensor) }, n);
    b.addEventListener('click', () => { st.sensor = c; sbtns.forEach((x, k) => x.setAttribute('aria-pressed', String(SENSORS[k][0] === c))); go(); });
    row.append(b);
    return b;
  });
  seg.append(row);
  ui.controls.append(seg);
  const sOff = slider(ui.controls, {
    label: 'Sensor offset', min: -5, max: 5, step: 0.5, value: 0,
    fmt: (v) => `${v > 0 ? '+' : ''}${v.toFixed(1)} K`,
    hint: 'a bias like a mis-calibrated or badly placed sensor',
    onInput: (v) => { st.offset = v; go(); },
  });
  const sDamp = ctx.compact ? null : slider(ui.controls, {
    label: 'Outdoor-air damper stuck at', min: -5, max: 100, step: 5, value: -5,
    fmt: (v) => (v < 0 ? 'off (as logged)' : `${v} % outdoor air`),
    hint: 'mixed air rewritten from the mixing balance; the logged command stays',
    onInput: (v) => { st.damper = v; go(); },
  });
  const sNoise = ctx.compact ? null : slider(ui.controls, {
    label: 'Extra sensor noise', min: 0, max: 2, step: 0.1, value: 0,
    fmt: (v) => `${v.toFixed(1)} K`,
    hint: 'a fixed random pattern per day, on the five measured temperatures',
    onInput: (v) => { st.noise = v; go(); },
  });
  const btns = h('div', { class: 'ff-btns' });
  ui.controls.append(btns);
  const maBtn = ctx.compact ? null : button(btns, 'MA sensor: on', () => { st.ma = !st.ma; sync(); go(); });
  if (maBtn) maBtn.title = 'Remove the mixed-air temperature: the outdoor-air fraction and the mixing-box checks go with it';
  button(btns, 'Reset', () => {
    Object.assign(st, { offset: 0, damper: -5, noise: 0, ma: true });
    sOff.set(0); sDamp?.set(-5); sNoise?.set(0); sync(); go();
  });

  function sync() {
    const d = days[st.i];
    const canMix = d.has[R.T_oa] && d.has[R.T_ra];
    if (sDamp) {
      sDamp.input.disabled = !canMix;
      (sDamp.input.closest('.inst-slider') as HTMLElement).style.opacity = canMix ? '' : '0.35';
    }
    SENSORS.forEach(([c], k) => { sbtns[k].disabled = !d.has[R[c]]; });
    if (!d.has[R[st.sensor]]) {
      const first = SENSORS.find(([c]) => d.has[R[c]]);
      if (first) { st.sensor = first[0]; sbtns.forEach((x, k) => x.setAttribute('aria-pressed', String(SENSORS[k][0] === st.sensor))); }
    }
    if (maBtn) {
      maBtn.disabled = !d.has[R.T_ma];
      maBtn.textContent = `MA sensor: ${st.ma && d.has[R.T_ma] ? 'on' : 'off'}`;
      maBtn.setAttribute('aria-pressed', String(st.ma && d.has[R.T_ma]));
    }
  }
  sync();

  const gaugeBox = h('div', {}), kvBox = h('div', { class: 'ff-kv' }), probBox = h('div', {}), note = h('p', { class: 'ff-note' });
  ui.readout.append(gaugeBox, kvBox, probBox, note);

  const control = (): Perturb => {
    const d = days[st.i];
    return {
      sensor: st.sensor, offset: st.offset,
      damper: st.damper >= 0 && d.has[R.T_oa] && d.has[R.T_ra] ? st.damper / 100 : null,
      noise: st.noise, drop_ma: !st.ma,
    };
  };

  async function run() {
    ui.setBusy(true);
    await new Promise((r) => requestAnimationFrame(r));
    const i = st.i, x0 = Float64Array.from(X.t[`x${i}`].data), nz = X.t[`n${i}`].data;
    const t1 = performance.now();
    const x = perturb(x0, nz, control());
    const r = read(W, x);
    const ms = performance.now() - t1;
    base[i] ??= read(W, x0).p;
    view = { r, x, x0 };
    plot.redraw();
    render(r);
    ui.stamp(ms, `${days[i].id} · 35 features × 96 bins recomputed`);
    ui.setBusy(false);
  }
  const go = scheduler(run, 30);

  function render(r: Reading) {
    const d = days[st.i], edited = r.p !== base[st.i];
    gaugeBox.innerHTML = '';
    gauges(gaugeBox, [{
      label: 'P(fault) today', value: r.p, range: [0, 1], digits: 2,
      truth: edited ? base[st.i] : null, truthLabel: 'the real day, unedited',
    }]);
    const truth = d.fault === 'none' ? 'no fault' : `${pretty(d.fault)}${d.severity ? ` (${d.severity})` : ''}`;
    const ffFlag = r.p > 0.5;
    const right = (flag: boolean) => (flag === (d.fault !== 'none') ? 'ok' : 'hot');
    kvBox.innerHTML = [
      ['Dataset label', `<b>${truth}</b>`],
      ['FaultFinder at P = 0.5', `<b class="${edited ? '' : right(ffFlag)}">${ffFlag ? 'fault' : 'no fault'}</b>`],
      [`${d.rule.name} (real day)`, `<b class="${right(d.rule.flag)}">${d.rule.flag ? 'fault' : 'no fault'}</b>`],
    ].map(([k, v]) => `<span>${k}</span>${v}`).join('');
    probBox.innerHTML = '';
    const top = r.diag.slice(0, ctx.compact ? 4 : 6);
    if (d.diag_class && !top.some((t) => t.cls === d.diag_class)) {
      const t = r.diag.find((q) => q.cls === d.diag_class);
      if (t) top.push(t);
    }
    probs(probBox, 'If it is faulty, which fault (calibrated, T = 4.3)', top.map((t) => ({ label: t.label, p: t.p, truth: t.cls === d.diag_class })));
    const why = d.fault === 'none' ? ''
      : d.simulable ? ` FaultFinder can name this fault (class <b>${d.diag_class?.replace(/_/g, ' ')}</b>).`
        : ' BuildingSim cannot simulate this fault type, so it is graded for detection only.';
    note.innerHTML = `<b>${d.facility}, ${d.unit}, ${d.date}.</b>${why} On the exam, a third of healthy air-handler days read above 0.5 and probabilities sit near 0.5 (one temperature from 20 dev days squashes them); the ranking is what carries the skill (AUC 0.78 on 200 AHU days, rules 0.69). The rule verdict is for the real day and is not re-run on your edits. Data: ${d.credit}.`;
  }

  function draw(c: Canvas2D) {
    const { ctx: g, w, h: H } = c;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, w, H);
    if (!view) return;
    const { x, x0 } = view;
    const padL = 34, padR = 10, top = 16, gap = 30, bot = 18;
    const iw = w - padL - padR, ih = H - top - gap - bot;
    const h1 = Math.round(ih * 0.64), h2 = ih - h1, y2 = top + h1 + gap;
    const X = (i: number) => padL + ((i + 0.5) / NT) * iw;
    // temperatures: shared range over the unedited and edited day
    let lo = Infinity, hi = -Infinity;
    for (const c0 of ['T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_sa_sp', 'T_zone']) for (const arr of [x, x0]) for (let i = 0; i < NT; i++) {
      const v = arr[R[c0] * NT + i]; if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    }
    if (!Number.isFinite(lo)) { lo = 10; hi = 30; }
    lo = Math.floor(lo / 5) * 5; hi = Math.ceil(hi / 5) * 5; if (hi - lo < 10) hi = lo + 10;
    const Y = (v: number) => top + h1 - ((v - lo) / (hi - lo)) * h1;
    gridLines(c, padL, top, iw, h1, 8, (hi - lo) / 5);
    for (let v = lo; v <= hi; v += 5) label(c, `${v}`, padL - 4, Y(v), SCOPE.faint, 'right', 9);
    label(c, narrow() ? 'air temperatures, °C' : 'air temperatures, °C (15-min means; faint = the real day where you changed it)', padL + 4, top - 7, SCOPE.text, 'left', 9.5);
    const line = (arr: ArrayLike<number>, rw: number, col: string, lw: number, dash: number[] = [], alpha = 1, yf = Y) => {
      g.strokeStyle = col; g.lineWidth = lw; g.setLineDash(dash); g.globalAlpha = alpha; g.beginPath();
      let pen = false;
      for (let i = 0; i < NT; i++) {
        const v = arr[rw * NT + i];
        if (!Number.isFinite(v)) { pen = false; continue; }
        if (pen) g.lineTo(X(i), yf(v)); else { g.moveTo(X(i), yf(v)); pen = true; }
      }
      g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
    };
    const changed = (rw: number) => { for (let i = 0; i < NT; i++) { const a = x[rw * NT + i], b = x0[rw * NT + i]; if (a !== b && !(Number.isNaN(a) && Number.isNaN(b))) return true; } return false; };
    let lx = padL + 4;
    for (const c0 of ['T_oa', 'T_ra', 'T_ma', 'T_sa', 'T_sa_sp', 'T_zone']) {
      const rw = R[c0];
      if (changed(rw)) line(x0, rw, TCOL[c0], 1, [], 0.3);
      line(x, rw, changed(rw) ? ctx.accent : TCOL[c0], changed(rw) ? 2 : 1.3, c0 === 'T_sa_sp' ? [4, 3] : []);
      if (!narrow() || c0 !== 'T_sa_sp') {
        const name = c0 === 'T_sa_sp' ? 'SA setpoint' : c0.slice(2).toUpperCase().replace('ZONE', 'zone');
        const has = days[st.i].has[rw] && !(c0 === 'T_ma' && !st.ma);
        label(c, has ? name : `${name} –`, lx, top + h1 + 10, has ? TCOL[c0] : SCOPE.faint, 'left', 9);
        lx += name.length * 6.2 + 14;
      }
    }
    // actuators, 0..1
    gridLines(c, padL, y2, iw, h2, 8, 2);
    const Ya = (v: number) => y2 + h2 - Math.max(-0.05, Math.min(1.05, v)) * h2;
    const atitle = narrow() ? 'commands' : 'commands as logged, 0-1';
    label(c, atitle, padL + 4, y2 - 7, SCOPE.text, 'left', 9.5);
    const aEnd = padL + 4 + g.measureText(atitle).width + 10;
    label(c, '1', padL - 4, Ya(1), SCOPE.faint, 'right', 9); label(c, '0', padL - 4, Ya(0), SCOPE.faint, 'right', 9);
    let ax = padL + iw;
    for (const c0 of ['fan_speed', 'valve_hc', 'valve_cc', 'damper_oa']) {
      const rw = R[c0];
      if (!days[st.i].has[rw]) continue;
      line(x, rw, ACOL[c0], 1.2, [], 0.9, Ya);
      const lw = ANAME[c0].length * 6.2;
      if (ax - lw > aEnd) label(c, ANAME[c0], ax, y2 - 7, ACOL[c0], 'right', 9);
      ax -= lw + 12;
    }
    // occupied hours as a faint band
    const occ = R.occ * NT, fan = R.fan_on * NT;
    g.fillStyle = 'rgba(235,230,216,0.05)';
    for (let i = 0; i < NT; i++) {
      const o = Number.isFinite(x[occ + i]) ? x[occ + i] : x[fan + i];
      if (o > 0.5) g.fillRect(X(i) - iw / NT / 2, top, iw / NT + 0.5, h1);
    }
    for (let hr = 0; hr <= 24; hr += 6) label(c, `${String(hr).padStart(2, '0')}:00`, padL + (hr / 24) * iw, y2 + h2 + 10, SCOPE.faint, hr === 0 ? 'left' : hr === 24 ? 'right' : 'center', 9);
    label(c, `P(fault) ${fmt(view.r.p, 2)}`, padL + iw - 4, top + 8, ctx.accent, 'right', 10);
  }

  await run();
}
