// SunDoctor live instrument: a real fortnight from one of the two exam farms, the real network running in the browser,
// and faults to add to the input — a smaller inverter, a dead string, a snowstorm — or the irradiance sensor to take away.
import type { InstrumentContext } from '../types.ts';
import { fetchPack } from '../../lib/nn/pack.ts';
import {
  channels, diagnose, inject, chain, shares, fortnight, expectedPower, CAUSES, D, T,
  type SMeta, type SWindow, type SData, type Diagnosis, type Cause, type Faults,
} from './net.ts';
import { shell, picker, slider, button, scheduler, gauges, probs, canvas, label, h, hex, SCOPE, type Canvas2D } from '../kit.ts';

const COL: Record<Cause, string> = {
  dead: '#ff6f61', soiling: '#b08d57', stuck: '#a58cf0', shading: '#4f8fe0', snow: '#e4edf7', clipping: '#4fc7a4', outage: '#7d838c',
};
const NAME: Record<Cause, string> = {
  dead: 'dead strings', soiling: 'soiling', stuck: 'stuck trackers', shading: 'shading', snow: 'snow', clipping: 'clipping', outage: 'outage',
};
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const dlab = (s: string) => `${+s.slice(8, 10)} ${MON[+s.slice(5, 7) - 1]}`;
const pct = (v: number, d = 0) => `${(v * 100).toFixed(d)}%`;

type Known = (Partial<Record<Cause, number>> & { snowRaw?: number; snowYours?: boolean }) | null;
interface View { p: Float32Array; d: Diagnosis; known: Known[]; expd: Float32Array; norm: Float32Array; f: Faults }

export async function mount(ctx: InstrumentContext) {
  const [W, X] = await Promise.all([fetchPack<SMeta>(`${ctx.base}weights`), fetchPack<SWindow[]>(`${ctx.base}examples`)]);
  const all = X.meta.map((m, i) => ({ m, s: fortnight(X, i) }));
  const list = ctx.compact ? all.filter((_, i) => [0, 2, 4, 5].includes(i)) : all;
  const ui = shell(ctx.root, {
    title: 'SunDoctor',
    subtitle: 'two weeks of a real solar farm, diagnosed day by day — data it never trained on',
    params: '591 k parameters · matches Python to 1e-5',
  });
  const st = { k: 0, home: false, ceil: 1, kill: 0, snow: 0 };
  const cur = () => list[st.k];
  let view: View | null = null;

  picker(ui.picker, list.map((e, i) => ({ label: e.m.name, sub: `${e.m.label} · ${dlab(e.m.dates[0])} – ${dlab(e.m.dates[D - 1])}`, value: i })),
    (i) => { st.k = i; relabel(); go(); }, 0);

  // ---- screen: the fortnight as rows of days, and the diagnosis beside each day
  const wide = (ctx.root.getBoundingClientRect().width || 800) >= 640;
  const plot = canvas(ui.screen, ctx.compact || !wide ? 1.05 : 2.05, (c) => draw(c));
  const legend = h('div', { style: 'display:flex;flex-wrap:wrap;gap:4px 12px;padding:7px 10px 8px;border-top:1px solid var(--scr-rule);font-family:var(--mono);font-variation-settings:"wdth" 87.5;font-size:10px;color:var(--scr-3ink)' });
  ui.screen.append(legend);

  // ---- controls
  const sw = (on: boolean, text: string) => h('button', { type: 'button', class: 'inst-pick', 'aria-pressed': String(on), style: 'flex:1 1 0;align-items:center;padding:6px 8px' }, text);
  const bSensor = sw(true, 'with sensor'), bHome = sw(false, 'power only');
  const setMode = (home: boolean) => { st.home = home; bSensor.setAttribute('aria-pressed', String(!home)); bHome.setAttribute('aria-pressed', String(home)); go(); };
  bSensor.addEventListener('click', () => setMode(false));
  bHome.addEventListener('click', () => setMode(true));
  const mode = h('div', { class: 'inst-slider', role: 'group', 'aria-label': 'What SunDoctor is given' },
    h('span', { class: 'inst-sl-top' }, h('span', {}, 'SunDoctor is given')),
    h('div', { style: 'display:flex;gap:6px' }, bSensor, bHome),
    h('small', {}, 'power only = a home system with no irradiance sensor'));

  const kw = (v: number) => { const m = cur().m, x = v * m.ceiling * m.dc_kw; return x >= 100 ? `${Math.round(x)} kW` : `${x.toFixed(1)} kW`; };
  const sCeil = slider(ui.controls, {
    label: 'Inverter ceiling', min: 0.45, max: 1, step: 0.01, value: 1,
    fmt: (v) => (v >= 0.995 ? `${kw(1)}, as built` : kw(v)),
    hint: 'a smaller inverter clips the bright hours', onInput: (v) => { st.ceil = v; go(); },
  });
  const sKill = slider(ui.controls, {
    label: 'A string dies', min: 0, max: D, step: 1, value: 0,
    fmt: (v) => (v === 0 ? 'never' : `on ${dlab(cur().m.dates[v - 1])}`),
    hint: 'one of six strings switches off', onInput: (v) => { st.kill = v; go(); },
  });
  const sSnow = ctx.compact ? null : slider(ui.controls, {
    label: 'Snow buries the panels', min: 0, max: D, step: 1, value: 0,
    fmt: (v) => (v === 0 ? 'never' : `on ${dlab(cur().m.dates[v - 1])}`),
    hint: 'nothing all day; it slides off at noon next day', onInput: (v) => { st.snow = v; go(); },
  });
  ui.controls.prepend(mode);
  button(ui.controls, 'Remove my faults', () => {
    Object.assign(st, { ceil: 1, kill: 0, snow: 0 });
    sCeil.set(1); sKill.set(0); sSnow?.set(0); go();
  });

  function relabel() {
    const m = cur().m;
    const row = sKill.input.closest('label')!;
    row.querySelector('.inst-sl-top > span')!.textContent = `A ${m.unit} dies`;
    row.querySelector('small')!.textContent = m.unit === 'string' ? 'one of six strings switches off' : 'one of the module’s three DC combiners trips';
    sCeil.set(st.ceil); sKill.set(st.kill); sSnow?.set(st.snow);
  }

  // ---- readout
  const gBox = h('div', { style: 'display:flex;flex-direction:column;gap:14px' });
  const pBox = h('div');
  const nBox = h('div', { style: 'display:flex;flex-direction:column;gap:8px;font-size:12.5px;line-height:1.45;color:var(--scr-2ink)' });
  ui.readout.append(gBox, pBox, nBox);

  function faults(): Faults {
    const m = cur().m;
    return { ceil: st.ceil < 0.995 ? st.ceil * m.ceiling : null, kill: st.kill ? st.kill - 1 : null, snow: st.snow ? st.snow - 1 : null };
  }

  /** What is known each day: the farm's answer key, plus the faults you added (as shares of possible energy). */
  function known(e: { m: SWindow; s: SData }, f: Faults): Known[] {
    const { m, s } = e;
    const pre = inject(s, m.n, m.ceiling, { ...f, snow: null });
    const out: Known[] = [];
    for (let d = 0; d < D; d++) {
      let dead = m.key.dead?.[d] ?? null, snow = m.key.snow?.[d] ?? null;
      const stuck = m.key.stuck?.[d] ?? null, outage = m.key.outage?.[d] ?? null;
      let clip: number | null = null;
      if (f.kill != null && d >= f.kill) dead = (dead ?? 0) + 1;
      const keySnow = snow;
      if (f.snow != null && d === f.snow) snow = 1;
      if (f.snow != null && d === f.snow + 1) {
        let a = 0, b = 0;
        for (let t = 0; t < T; t++) { const v = pre[d * T + t]; if (Number.isFinite(v)) { a += v; if (t < 48) b += v; } }
        snow = Math.max(snow ?? 0, a > 0 ? b / a : 0);
      }
      if (f.ceil != null) {
        let a = 0, b = 0;
        const mult = f.kill != null && d >= f.kill ? (m.n - 1) / m.n : 1;
        for (let t = 0; t < T; t++) { const u = s.pu[d * T + t] * mult; if (Number.isFinite(u)) { a += u; b += Math.min(u, f.ceil); } }
        clip = a > 0 ? 1 - b / a : 0;
      }
      if ([dead, snow, stuck, outage, clip].every((v) => v == null)) { out.push(null); continue; }
      const ch = chain({ frac: 1 - (dead ?? 0) / m.n, soil: 0, stuck: stuck ?? 0, shade: 0, snow: snow ?? 0, clip: clip ?? 0, outage: outage ?? 0 });
      const k: NonNullable<Known> = {};
      if (dead != null) k.dead = ch.dead;
      if (stuck != null) k.stuck = ch.stuck;
      if (snow != null) { k.snow = ch.snow; k.snowRaw = snow; k.snowYours = snow !== keySnow; }
      if (clip != null) k.clipping = ch.clipping;
      if (outage != null) k.outage = ch.outage;
      out.push(k);
    }
    return out;
  }

  async function run() {
    ui.setBusy(true);
    await new Promise((r) => requestAnimationFrame(r));
    const e = cur(), m = e.m, f = faults();
    const ref = st.home ? m.ref.home : m.ref.poa;
    const p = inject(e.s, m.n, m.ceiling, f);
    const t0 = performance.now();
    const { d } = diagnose(W, channels(p, e.s, ref, st.home), m.mount);
    const ms = performance.now() - t0;
    // what the array should have made, each step: from the sensor's sunlight, or (power only) from clear-sky sunlight
    const expd = new Float32Array(D * T);
    for (let i = 0; i < D * T; i++) {
      const ta = Number.isFinite(e.s.ta[i]) ? e.s.ta[i] : 15, g = e.s.g[i];
      expd[i] = st.home ? expectedPower(e.s.gc[i], ta, e.s.ca[i]) : Number.isFinite(g) ? expectedPower(Math.max(g, 0), ta, e.s.ca[i]) : NaN;
    }
    // the ratio map's "usual": per time of day, the recorded fortnight's 70th percentile (85th against clear sky)
    const nk = `${st.k}:${st.home}`;
    if (!norms.has(nk)) {
      const nm = new Float32Array(T).fill(NaN);
      for (let t = 0; t < T; t++) {
        const r: number[] = [];
        for (let i = t; i < D * T; i += T) if (expd[i] >= 0.04 && Number.isFinite(e.s.p[i])) r.push(e.s.p[i] / expd[i]);
        if (r.length >= 3) { r.sort((a, b) => a - b); nm[t] = r[Math.min(r.length - 1, Math.floor((st.home ? 0.85 : 0.7) * (r.length - 1) + 0.5))]; }
      }
      norms.set(nk, nm);
    }
    view = { p, d, known: known(e, f), expd, norm: norms.get(nk)!, f };
    plot.redraw();
    readout(e.m, view);
    ui.stamp(ms, `${m.system}, ${dlab(m.dates[0])} – ${dlab(m.dates[D - 1])} ${m.dates[0].slice(0, 4)} · ${st.home ? 'power only' : 'with irradiance sensor'}`);
    ui.setBusy(false);
  }
  const go = scheduler(run, 40);
  const norms = new Map<string, Float32Array>();

  // ---- drawing
  function daylight(s: SData): [number, number] {
    let a = T, b = 0;
    for (let d = 0; d < D; d++) for (let t = 0; t < T; t++) if (s.gc[d * T + t] > 0.004) { a = Math.min(a, t); b = Math.max(b, t); }
    return [Math.max(0, a - 1), Math.min(T - 1, b + 1)];
  }

  function image(c: Canvas2D, x0: number, y0: number, w: number, hh: number, c0: number, c1: number, color: (i: number) => [number, number, number]) {
    const cols = c1 - c0 + 1;
    const off = document.createElement('canvas');
    off.width = cols; off.height = D;
    const g = off.getContext('2d')!, img = g.createImageData(cols, D);
    for (let d = 0; d < D; d++) for (let q = 0; q < cols; q++) {
      const [r, gg, b] = color(d * T + c0 + q), k = (d * cols + q) * 4;
      img.data[k] = r; img.data[k + 1] = gg; img.data[k + 2] = b; img.data[k + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    c.ctx.imageSmoothingEnabled = false;
    c.ctx.drawImage(off, x0, y0, w, hh);
    c.ctx.imageSmoothingEnabled = true;
  }

  const BG: [number, number, number] = [13, 15, 12], NIGHT: [number, number, number] = [22, 25, 21];
  const mix = (a: number[], b: number[], t: number) => [0, 1, 2].map((k) => a[k] + (b[k] - a[k]) * t) as [number, number, number];

  function draw(c: Canvas2D) {
    const { ctx: g, w, h: H } = c;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, w, H);
    if (!view) return;
    const e = cur(), m = e.m, v = view, d = v.d;
    const narrow = w < 560;
    const top = 30, bot = 22, left = 44, gap = narrow ? 10 : 12, right = 8;
    const rowH = (H - top - bot) / D;
    const avail = w - left - right;
    const wA = avail * (narrow ? 0.36 : 0.27), wB = narrow ? 0 : avail * 0.27;
    const xA = left, xB = xA + wA + gap, xC = narrow ? xB : xB + wB + gap, wC = w - right - xC;
    const [c0, c1] = daylight(e.s);
    const acc = hex(ctx.accent);

    // panel titles
    const title = (t: string, x: number, sub?: string) => { label(c, t, x, 9, SCOPE.ink, 'left', 9.5); if (sub) label(c, sub, x, 21, SCOPE.faint, 'left', 8.5); };
    title('POWER', xA, 'per kW of panels');
    if (!narrow) title('POWER ÷ EXPECTED', xB, st.home ? 'from clear sky (no sensor)' : 'from the sensor’s sunlight');
    title('DIAGNOSIS', xC, 'share of possible energy');

    // power heatmap (dark -> accent -> white)
    const pmax = m.ceiling;
    image(c, xA, top, wA, rowH * D, c0, c1, (i) => {
      const p = v.p[i];
      if (!Number.isFinite(p)) return NIGHT;
      const t = Math.max(0, Math.min(1, p / pmax)), q = t ** 0.8, hi = Math.max(0, q - 0.85) * 4;
      return [13 + (acc[0] - 13) * q + (255 - acc[0]) * hi, 15 + (acc[1] - 15) * q + (255 - acc[1]) * hi, 12 + (acc[2] - 12) * q + (255 - acc[2]) * hi];
    });
    // power / expected (neutral at 1, blue for a shortfall, accent for more than expected)
    if (!narrow) {
      const NEU = [46, 50, 42], BLUE = [64, 128, 222], PALE = [214, 230, 255];
      image(c, xB, top, wB, rowH * D, c0, c1, (i) => {
        const x = v.expd[i], p = v.p[i];
        if (!Number.isFinite(x) || !Number.isFinite(p) || x < 0.04) return x >= 0.04 ? NIGHT : BG;
        const r = p / x / (v.norm[i % T] || 1);
        if (r >= 1) return mix(NEU, acc, Math.min(1, (r - 1) / 0.3));
        const s = Math.min(1, 1 - r);
        return s < 0.35 ? mix(NEU, BLUE, s / 0.35) : mix(BLUE, PALE, (s - 0.35) / 0.65);
      });
    }

    // rows: dates, change marks, bars
    const tot = (o: Partial<Record<Cause, number>>) => CAUSES.reduce((a, k) => a + (o[k] ?? 0), 0);
    for (let k = 0; k <= 4; k++) {
      const x = Math.round(xC + (wC * k) / 4) + 0.5;
      g.strokeStyle = SCOPE.grid; g.beginPath(); g.moveTo(x, top); g.lineTo(x, top + rowH * D); g.stroke();
    }
    for (let i = 0; i < D; i++) {
      const y = top + i * rowH, flag = i > 0 && d.step_p[i] > 0.5;
      label(c, dlab(m.dates[i]), left - 6, y + rowH / 2, flag ? ctx.accent : SCOPE.text, 'right', rowH < 15 ? 8.5 : 9.5);
      if (flag) {
        g.strokeStyle = ctx.accent; g.lineWidth = 1.5; g.setLineDash([4, 3]);
        g.beginPath(); g.moveTo(xA - 2, Math.round(y) + 0.5); g.lineTo(xC - gap / 2, Math.round(y) + 0.5); g.stroke();
        g.setLineDash([]); g.lineWidth = 1;
      }
      // SunDoctor's bar
      const s = shares(d, i);
      let x = xC;
      const by = y + rowH * 0.1, bh = rowH * 0.5;
      for (const k of CAUSES) {
        const wk = s[k] * wC;
        if (wk > 0.25) { g.fillStyle = COL[k]; g.fillRect(x, by, wk, bh); }
        x += wk;
      }
      const lost = tot(s);
      if (lost >= 0.005) label(c, pct(lost), Math.min(x + 4, xC + wC - 24), by + bh / 2, SCOPE.text, 'left', 8.5);
      // what is known
      const kn = v.known[i];
      if (kn) {
        let xk = xC;
        const ky = y + rowH * 0.66, kh = Math.max(2, rowH * 0.2);
        g.globalAlpha = 0.85;
        for (const k of CAUSES) {
          const wk = (kn[k] ?? 0) * wC;
          if (wk > 0.25) { g.fillStyle = COL[k]; g.fillRect(xk, ky, wk, kh); }
          xk += wk;
        }
        g.globalAlpha = 1;
        g.fillStyle = SCOPE.faint; g.fillRect(xC, ky + kh / 2 - 0.5, Math.max(0, xk === xC ? 3 : 0), 1);
      }
    }
    // a dead string you added: mark its first day on the heatmaps
    if (v.f.kill != null) {
      const y = Math.round(top + v.f.kill * rowH) + 0.5;
      g.strokeStyle = SCOPE.ink; g.globalAlpha = 0.8; g.beginPath(); g.moveTo(xA, y); g.lineTo(xA + 6, y); g.moveTo((narrow ? xA + wA : xB + wB) - 6, y); g.lineTo(narrow ? xA + wA : xB + wB, y); g.stroke(); g.globalAlpha = 1;
    }
    // axes
    const yb = top + rowH * D + 11;
    const hx = (x0: number, ww: number, hr: number) => x0 + ((hr * 4 - 0.5 - c0 + 0.5) / (c1 - c0 + 1)) * ww;
    for (const hr of [6, 9, 12, 15, 18]) {
      const xa = hx(xA, wA, hr);
      if (xa < xA + 9 || xa > xA + wA - 9) continue;
      label(c, `${hr}h`, xa, yb, SCOPE.faint, 'center', 8.5);
      if (!narrow) label(c, `${hr}h`, hx(xB, wB, hr), yb, SCOPE.faint, 'center', 8.5);
    }
    for (let k = 0; k <= 4; k++) label(c, k === 0 ? '0' : `${k * 25}%`, xC + (wC * k) / 4, yb, SCOPE.faint, k === 0 ? 'left' : k === 4 ? 'right' : 'center', 8.5);
  }

  function readout(m: SWindow, v: View) {
    const d = v.d, f = v.f, last = D - 1, n = m.n;
    // dead strings / combiners on the last day, with the answer key
    let kTrue: number | null = m.key.dead?.[last] ?? m.key.dead?.[last - 1] ?? null;
    if (f.kill != null) kTrue = (kTrue ?? 0) + 1;
    const k = n * (1 - d.frac[last]), ks = n * d.frac_sig[last];
    // inverter ceiling (a fortnight-wide output)
    const dc = m.dc_kw, ceilTrue = f.ceil != null ? f.ceil : m.ceiling;
    // worst snow day
    let iS = 0;
    d.snow.forEach((x, i) => { if (x > d.snow[iS]) iS = i; });
    const kS = v.known[iS], snowTrue = kS?.snowRaw ?? null;
    gauges(gBox, [
      { label: `${m.unit}s out, last day`, unit: `of ${n}`, value: k, lo: Math.max(0, k - ks), hi: k + ks, truth: kTrue,
        truthLabel: f.kill != null ? (m.key.dead ? 'sibling key + yours' : 'yours') : 'sibling inverters', range: [0, n === 6 ? 3 : 2], digits: 1 },
      { label: 'Inverter ceiling', unit: 'kW', value: d.ac_limit * dc, lo: (d.ac_limit - d.ac_limit_sig) * dc, hi: (d.ac_limit + d.ac_limit_sig) * dc,
        truth: ceilTrue * dc, truthLabel: f.ceil != null ? 'your inverter' : m.ceiling_note, range: [0.3 * dc, dc], digits: dc > 100 ? 0 : 1 },
      { label: `Snow, worst day (${dlab(m.dates[iS])})`, unit: '%', value: d.snow[iS] * 100, lo: Math.max(0, d.snow[iS] - d.snow_sig[iS]) * 100,
        hi: Math.min(1, d.snow[iS] + d.snow_sig[iS]) * 100, truth: snowTrue != null ? snowTrue * 100 : null,
        truthLabel: kS?.snowYours ? 'yours' : 'plant key', range: [0, 100], digits: 0 },
    ]);
    // the fortnight's energy, weighted by what each day could have made
    let wsum = 0;
    const acc: Record<Cause, number> = { dead: 0, soiling: 0, stuck: 0, shading: 0, snow: 0, clipping: 0, outage: 0 };
    for (let i = 0; i < D; i++) {
      let wd = 0;
      for (let t = 0; t < T; t++) { const x = v.expd[i * T + t]; if (Number.isFinite(x)) wd += x; }
      const s = shares(d, i);
      for (const c of CAUSES) acc[c] += s[c] * wd;
      wsum += wd;
    }
    const lost = CAUSES.reduce((a, c) => a + acc[c], 0) / (wsum || 1);
    probs(pBox, 'Where the fortnight’s energy went', [
      { label: 'delivered', p: Math.max(0, 1 - lost) },
      ...CAUSES.filter((c) => m.mount === 'tracker' || c !== 'stuck').map((c) => ({ label: m.unit === 'combiner' && c === 'dead' ? 'dead combiners' : NAME[c], p: acc[c] / (wsum || 1) })),
    ]);
    // events: what SunDoctor flags, what the key says, what you did
    const flags = d.step_p.map((p, i) => (i > 0 && p > 0.5 ? `${dlab(m.dates[i])} (${Math.round(p * 100)}%)` : '')).filter(Boolean);
    const keyEv: string[] = [];
    if (m.key.dead) {
      let prev: number | null = null;
      m.key.dead.forEach((x, i) => { if (x != null) { if (prev != null && x !== prev) keyEv.push(`${x > prev ? 'string out' : 'repaired'} ${dlab(m.dates[i])}`); prev = x; } });
      if (!keyEv.length) keyEv.push(m.key.dead.some((x) => x != null) ? `${m.key.dead.find((x) => x != null)} strings out, no change` : 'no key for these days');
    }
    if (m.key.snow) {
      const days = (a: (number | null)[] | undefined, th: number) => (a ?? []).map((x, i) => (x != null && x > th ? dlab(m.dates[i]) : '')).filter(Boolean);
      const sn = days(m.key.snow, 0.2), stk = days(m.key.stuck, 0.1), out = days(m.key.outage, 0.5);
      if (sn.length) keyEv.push(`snow ${sn.join(', ')}`);
      if (stk.length) keyEv.push(`trackers stuck ${stk.join(', ')}`);
      if (out.length) keyEv.push(`outage ${out.join(', ')}`);
    }
    const you: string[] = [];
    if (f.ceil != null) you.push(`inverter cut to ${Math.round(f.ceil * dc * (dc > 100 ? 1 : 10)) / (dc > 100 ? 1 : 10)} kW`);
    if (f.kill != null) you.push(`a ${m.unit} dies ${dlab(m.dates[f.kill])}`);
    if (f.snow != null) you.push(`snow ${dlab(m.dates[f.snow])}`);
    const note: string[] = [];
    if (st.home) note.push('Power only: with no sunlight measurement a cloudy day and a fault look alike, so the ratio map is now against clear sky and the error bars widen.');
    if (f.snow != null) note.push('A panel buried all day makes nothing, exactly like a tripped inverter, so SunDoctor may call that day an outage. Real snow also comes with cold; this fortnight’s temperatures are unchanged.');
    if (f.kill != null && f.kill > 0 && d.step_p[f.kill] <= 0.5) note.push('The failure day was not flagged; the count may still step up.');
    if (!note.length) note.push(`${m.desc[0].toUpperCase()}${m.desc.slice(1)}. ${m.mount === 'tracker' ? 'Single-axis trackers, 592 kW of thin-film panels.' : 'Fixed rack, 37 kW of panels on a 30 kW inverter.'}`);
    const row = (t: string, body: string) => `<div><div class="ig-l" style="margin-bottom:2px">${t}</div><div style="font-family:var(--mono);font-variation-settings:'wdth' 87.5;font-size:11px;color:var(--scr-ink)">${body}</div></div>`;
    nBox.innerHTML = row('SunDoctor flags a change', flags.join(', ') || 'none')
      + row('Answer key', keyEv.join(' · ') || 'none for these days')
      + (you.length ? row('Your faults', you.join(' · ')) : '')
      + `<p style="margin:0">${note.join(' ')}</p>`;
    // legend under the screen
    const used = CAUSES.filter((c) => m.mount === 'tracker' || c !== 'stuck');
    legend.innerHTML = used.map((c) => `<span style="display:inline-flex;align-items:center;gap:5px"><i style="width:9px;height:9px;border-radius:2px;background:${COL[c]}"></i>${m.unit === 'combiner' && c === 'dead' ? 'dead combiners' : NAME[c]}</span>`).join('')
      + `<span style="flex-basis:100%;height:0"></span><span>upper bar SunDoctor · lower bar answer key + your faults</span><span><span style="color:${ctx.accent}">- - -</span> SunDoctor flags a change</span><span>ratio: 1 = the fortnight’s usual, <span style="color:#6aa0ea">blue</span> = short of it</span><span>solar time</span>`;
  }

  relabel();
  await run();
}
