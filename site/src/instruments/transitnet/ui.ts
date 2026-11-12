// TransitNet live instrument: three real TESS stars folded on their candidate's period, the real network running in
// the browser, and four things to change in the light curve itself — noise, the dip's depth (third light, or a bigger
// body), a secondary eclipse, unequal odd/even transits — to watch a planet turn into a binary, or a binary hide.
import type { InstrumentContext } from '../types.ts';
import { fetchPack } from '../../lib/nn/pack.ts';
import { readStar, points, NL, NS, NG, NO, type TMeta, type TExample, type Views, type Info, type Reading, type ClassId } from './net.ts';
import { dip, ldFor, LD } from './transit.ts';
import { shell, picker, slider, button, scheduler, probs, gauges, canvas, gridLines, label, rng, h, SCOPE, type Canvas2D, type Gauge } from '../kit.ts';

const LABEL: Record<ClassId, string> = { planet: 'Planet', eb: 'Eclipsing binary', blend: 'Background binary', nothing: 'Nothing real' };
const TAG: Record<string, string> = { toi270c: 'mini-Neptune · M dwarf', pimenc: 'super-Earth · bright Sun-like star', toi617: 'false positive · M dwarf' };
const EVEN = '#e8b04a';

const STYLE = `
.tn-note { display: flex; flex-direction: column; gap: 4px; padding-top: 10px; border-top: 1px solid var(--scr-rule);
  font-family: var(--mono); font-variation-settings: 'wdth' 87.5; font-size: 10.5px; line-height: 1.45; color: var(--scr-3ink); }
.tn-note b { font-weight: 400; color: var(--scr-ink); }
.tn-note .tn-warn { color: #ff8a7a; }
.tn-note .tn-ok { color: color-mix(in srgb, #4cc38a 85%, white); }
`;

const ppm = (v: number) => {
  const a = Math.abs(v);
  if (a >= 1e4) return `${(v / 1e4).toFixed(a >= 1e5 ? 0 : 1)}%`;
  return `${Math.round(v).toLocaleString('en-US').replace('-', '−')}`;
};
/** a depth or noise level: ppm below 1%, percent above */
const level = (v: number) => (Math.abs(v) >= 1e4 ? ppm(v) : `${ppm(v)} ppm`);
const niceTicks = (lo: number, hi: number, n = 4) => {
  const span = hi - lo;
  const st = [1, 2, 2.5, 5, 10].map((m) => m * 10 ** Math.floor(Math.log10(span / n))).find((s) => span / s <= n) || span / n;
  const o: number[] = [];
  for (let v = Math.ceil(lo / st) * st; v <= hi + 1e-12; v += st) o.push(+v.toPrecision(10));
  return o;
};

export async function mount(ctx: InstrumentContext) {
  const [W, X] = await Promise.all([fetchPack<TMeta>(`${ctx.base}weights`), fetchPack<TExample[]>(`${ctx.base}examples`)]);
  const ex = X.meta;
  if (!document.getElementById('tn-style')) document.head.append(h('style', { id: 'tn-style' }, STYLE));
  const ui = shell(ctx.root, {
    title: 'TransitNet',
    subtitle: 'a star’s light, folded on its candidate’s period — real TESS data, never used in training',
    params: '322 k parameters · matches Python to 1e-5',
  });
  const st = { i: 0, noise: 1, depth: 0, sec: 0, oe: 0, seed: 3 };
  const base = new Map<number, { sigma: number; ld: [number, number] }>(); // per star: its own point noise, drawing LD
  let view: { V: Views; info: Info; r: Reading; e: TExample; ld: [number, number] } | null = null;

  picker(ui.picker, ex.map((e, i) => ({ label: e.name, sub: TAG[e.id] ?? e.kind, value: i })), (i) => { st.i = i; go(); }, st.i);

  const small = ctx.root.getBoundingClientRect().width < 560;
  const plot = canvas(ui.screen, small ? 0.92 : ctx.compact ? 1.45 : 1.3, (c) => draw(c));
  const sig0 = () => base.get(st.i)?.sigma ?? 0;
  const ctl = {
    noise: slider(ui.controls, {
      label: 'Add noise', min: 1, max: 20, step: 0.5, value: 1,
      fmt: (v) => (v === 1 ? 'as observed' : `×${v} · ${level(sig0() * v * 1e6)}`),
      hint: 'white noise, in multiples of the star’s own scatter per 2-minute point',
      onInput: (v) => { st.noise = v; go(); },
    }),
    depth: slider(ui.controls, {
      label: 'Depth of the dip', min: -1.5, max: 1.5, step: 0.05, value: 0,
      fmt: (v) => (Math.abs(v) < 1e-9 ? 'as observed' : v < 0 ? `×1/${(10 ** -v).toPrecision(2)} diluted` : `×${(10 ** v).toPrecision(2)} deeper`),
      hint: 'left: an unresolved star shares the pixels and dilutes it, as in a background blend · right: a bigger body in front',
      onInput: (v) => { st.depth = v; go(); },
    }),
    sec: slider(ui.controls, {
      label: 'Secondary eclipse', min: 0, max: 1, step: 0.05, value: 0,
      fmt: (v) => (v === 0 ? 'none' : `${Math.round(v * 100)}% of the primary`),
      hint: 'copies the star’s own dip to half an orbit later: a companion that shines',
      onInput: (v) => { st.sec = v; go(); },
    }),
    oe: ctx.compact ? null : slider(ui.controls, {
      label: 'Odd vs even transits', min: 0, max: 1, step: 0.05, value: 0,
      fmt: (v) => (v === 0 ? 'equal' : `odd +${Math.round(v * 100)}%, even −${Math.round(v * 100)}%`),
      hint: 'a binary caught at half its period alternates deep and shallow eclipses',
      onInput: (v) => { st.oe = v; go(); },
    }),
  };
  if (!ctx.compact) button(ui.controls, 'New noise draw', () => { st.seed++; go(); });
  button(ui.controls, 'Reset', () => {
    Object.assign(st, { noise: 1, depth: 0, sec: 0, oe: 0 });
    ctl.noise.set(1); ctl.depth.set(0); ctl.sec.set(0); ctl.oe?.set(0);
    go();
  });

  async function run() {
    ui.setBusy(true);
    await new Promise((r) => requestAnimationFrame(r));
    const e = ex[st.i], pts = points(X, st.i);
    if (!base.has(st.i)) {
      const r0 = readStar(W, e, pts, {});
      const pub = e.published;
      const ld = pub.k && pub.b != null && pub.aR ? ldFor(e.P, pub.k, pub.b, pub.aR, r0.info.depth)
        : r0.r.p.planet >= 0.5 ? ldFor(e.P, r0.r.k, r0.r.b, r0.r.aR, r0.info.depth) : LD;
      base.set(st.i, { sigma: r0.info.noise_pt, ld });
      ctl.noise.set(st.noise); // refresh the ppm in the label for this star
    }
    const b = base.get(st.i)!;
    const g = rng(st.seed * 977 + st.i * 31 + 1);
    const t0 = performance.now();
    const { V, info, r } = readStar(W, e, pts, {
      sec: st.sec, oe: st.oe, depth: 10 ** st.depth,
      noise: b.sigma * Math.sqrt(Math.max(0, st.noise * st.noise - 1)), photon: b.sigma, gauss: g.n,
    });
    const ms = performance.now() - t0;
    view = { V, info, r, e, ld: b.ld };
    plot.redraw();
    readout(e, r, info);
    ui.stamp(ms, `${e.name}: ${e.n_points.toLocaleString('en-US')} two-minute points from ${e.n_sectors} TESS sectors, folded on P = ${e.P.toFixed(4)} d`);
    ui.setBusy(false);
  }
  const go = scheduler(run, 30);

  function readout(e: TExample, r: Reading, info: Info) {
    probs(ui.readout, 'What the dip is', (['planet', 'eb', 'blend', 'nothing'] as ClassId[]).map((c) => ({ label: LABEL[c], p: r.p[c], truth: e.truth === c })));
    const pub = e.published, rad = e.star.have > 0 && e.star.rad ? e.star.rad : null;
    const isPl = e.kind === 'planet';
    const gs: Gauge[] = [
      { label: 'Radius ratio Rp/R*', value: r.k, lo: r.k_lo, hi: r.k_hi, truth: pub.k ?? null, truthLabel: 'published', range: [0.004, 0.5], digits: r.k < 0.1 ? 4 : 3, log: true },
    ];
    if (rad) gs.push({ label: 'Planet radius', unit: 'R⊕', value: r.k * rad * 109.1, lo: r.k_lo * rad * 109.1, hi: r.k_hi * rad * 109.1, truth: pub.k ? pub.k * rad * 109.1 : null, truthLabel: 'published', range: [0.5, 40], digits: r.k * rad * 109.1 < 10 ? 2 : 1, log: true });
    gs.push(
      { label: 'Impact parameter b', value: r.b, lo: r.b_lo, hi: r.b_hi, truth: pub.b ?? null, truthLabel: 'published', range: [0, 1.4], digits: 2 },
      { label: 'Orbit size a/R*', value: r.aR, lo: r.aR_lo, hi: r.aR_hi, truth: pub.aR ?? null, truthLabel: 'published', range: [2, 80], digits: 1, log: true },
    );
    gauges(ui.readout, gs);
    ui.readout.querySelectorAll('.tn-note').forEach((n) => n.remove());
    const snr = info.depth / info.noise_bin;
    const note = h('div', { class: 'tn-note' });
    note.innerHTML = [
      `model-free: depth <b>${level(info.depth * 1e6)}</b> → √depth <b>${Math.sqrt(Math.max(info.depth, 0)).toFixed(4)}</b>; P/(π·duration) <b>${r.aR0.toFixed(1)}</b>. The network reads corrections to these.`,
      `depth ÷ noise per bin <b>${Number.isFinite(snr) ? snr.toFixed(1) : '—'}</b> · ${e.ntr} transits folded`,
      isPl ? '' : `<span class="tn-warn">published: ${pub.disposition ?? 'false positive'} — ${pub.comments ?? ''}</span>`,
      !rad ? `<span>no planet radius: the TIC lacks this star’s log g, so the network runs without the star’s size</span>` : '',
    ].filter(Boolean).map((s) => `<span>${s}</span>`).join('');
    ui.readout.append(note);
  }

  function draw(c: Canvas2D) {
    const { ctx: g, w, h: H } = c;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, w, H);
    if (!view) return;
    const { V, info, r, e, ld } = view;
    const sc = info.scale;
    const narrow = w < 480;
    const padL = narrow ? 44 : 52, padR = 10, top = 22, gap = narrow ? 40 : 46;
    const topH = Math.round((H - top - gap - 40) * 0.6), botY = top + topH + gap, botH = H - botY - 24;
    const lo = -1.42, hi = 0.48;
    const Yt = (v: number) => top + (1 - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * topH;
    const iw = w - padL - padR;

    // ---- top: the folded transit (local view) -------------------------------------------------
    gridLines(c, padL, top, iw, topH, 10, 0);
    const Xl = (x: number) => padL + ((x + 2.5) / 5) * iw;
    // y ticks in ppm (the network sees the dip divided by its depth; the depth itself enters as a number)
    for (const t of niceTicks(lo * sc * 1e6, hi * sc * 1e6, narrow ? 3 : 4)) {
      const y = Yt(t / 1e6 / sc);
      g.strokeStyle = t === 0 ? 'rgba(235,230,216,0.22)' : SCOPE.grid;
      g.beginPath(); g.moveTo(padL, Math.round(y) + 0.5); g.lineTo(padL + iw, Math.round(y) + 0.5); g.stroke();
      label(c, ppm(t), padL - 6, y, SCOPE.faint, 'right', 9);
    }
    label(c, 'ppm', padL - 6, top - 10, SCOPE.faint, 'right', 9);
    // model-free depth
    const dy = Yt(-info.depth / sc);
    g.strokeStyle = 'rgba(235,230,216,0.4)'; g.setLineDash([2, 4]);
    g.beginPath(); g.moveTo(padL, dy + 0.5); g.lineTo(padL + iw, dy + 0.5); g.stroke(); g.setLineDash([]);
    label(c, `depth ${level(info.depth * 1e6)}`, padL + iw - 4, dy + 9, SCOPE.text, 'right', 9);
    // the data the network sees
    dots(c, V.local, NL, 0, 1, Xl, -2.5, 2.5, Yt, SCOPE.ink, narrow ? 1.6 : 2);
    // model curves: published (dashed ink), network (accent)
    const curve = (k: number, b: number, aR: number, color: string, width: number, dash: number[], alpha: number) => {
      g.strokeStyle = color; g.lineWidth = width; g.setLineDash(dash); g.globalAlpha = alpha;
      g.beginPath();
      for (let i = 0; i <= 240; i++) {
        const x = -2.5 + (5 * i) / 240, v = dip(x * e.dur, e.P, k, Math.max(0, b), aR, ld) / sc;
        i ? g.lineTo(Xl(x), Yt(v)) : g.moveTo(Xl(x), Yt(v));
      }
      g.stroke(); g.setLineDash([]); g.globalAlpha = 1; g.lineWidth = 1;
    };
    const pub = e.published;
    if (pub.k && pub.b != null && pub.aR) curve(pub.k, pub.b, pub.aR, SCOPE.ink, 1.3, [5, 4], 0.7);
    const pl = r.p.planet >= 0.5;
    curve(r.k, r.b, r.aR, ctx.accent, 2.2, [], pl ? 1 : 0.45);
    // titles and legend
    label(c, `Folded transit · ${e.ntr} transits`, padL, top - 10, SCOPE.text, 'left', 10);
    let lx = padL + iw;
    const leg = (text: string, color: string, dash: boolean) => {
      g.font = '9px "Martian Mono Variable", ui-monospace, monospace';
      const tw = g.measureText(text).width;
      lx -= tw;
      label(c, text, lx, top - 10, SCOPE.text, 'left', 9);
      g.strokeStyle = color; g.lineWidth = dash ? 1.3 : 2.2; g.setLineDash(dash ? [4, 3] : []);
      g.beginPath(); g.moveTo(lx - 22, top - 10); g.lineTo(lx - 5, top - 10); g.stroke(); g.setLineDash([]); g.lineWidth = 1;
      lx -= 34;
    };
    if (!narrow) {
      if (pub.k) leg('published', SCOPE.ink, true);
      leg(pl ? 'TransitNet’s planet' : 'if it were a planet', ctx.accent, false);
    }
    // x ticks in hours
    const hrs = e.dur * 24;
    for (const x of narrow ? [-2, 0, 2] : [-2, -1, 0, 1, 2]) label(c, x === 0 ? 'mid-transit' : `${x > 0 ? '+' : '−'}${Math.abs(x * hrs).toFixed(1)} h`, Xl(x), top + topH + 11, SCOPE.faint, 'center', 9);

    // ---- bottom: the clues (same scale: 1 = the transit's depth) ----------------------------------
    const pg = narrow ? 8 : 14, pw = (w - padL - padR - 2 * pg) / 3;
    const Yb = (v: number) => botY + (1 - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * botH;
    const panels: { title: string; draw: (x0: number) => void }[] = [
      { title: narrow ? 'Orbit' : 'Whole orbit', draw: (x0) => {
        const X = (p: number) => x0 + ((p + 0.5) / 1) * pw;
        dots(c, V.glob, NG, 0, 1, X, -0.5, 0.5, Yb, SCOPE.ink, 1.3);
        foot(c, ['−½', '0', '+½'], [X(-0.47), X(0), X(0.47)], botY + botH + 10);
      } },
      { title: narrow ? '½ later' : 'Half an orbit later', draw: (x0) => {
        const X = (x: number) => x0 + ((x + 2.5) / 5) * pw;
        g.fillStyle = 'rgba(235,230,216,0.05)'; g.fillRect(X(-0.5), botY, X(0.5) - X(-0.5), botH);
        dots(c, V.secondary, NS, 0, 1, X, -2.5, 2.5, Yb, SCOPE.ink, 1.7);
        foot(c, ['', 'phase ½', ''], [X(-2), X(0), X(2)], botY + botH + 10);
      } },
      { title: narrow ? 'Odd·even' : 'Odd · even transits', draw: (x0) => {
        const X = (x: number) => x0 + ((x + 1.5) / 3) * pw;
        dots(c, V.oddeven, NO, 0, 2, X, -1.5, 1.5, Yb, SCOPE.ink, 1.7);
        dots(c, V.oddeven, NO, 1, 3, X, -1.5, 1.5, Yb, EVEN, 1.7);
        label(c, 'odd', x0 + pw - 30, botY + botH + 10, SCOPE.ink, 'right', 9);
        label(c, 'even', x0 + pw - 2, botY + botH + 10, EVEN, 'right', 9);
      } },
    ];
    panels.forEach((p, j) => {
      const x0 = padL + j * (pw + pg);
      g.fillStyle = 'rgba(255,255,255,0.025)'; g.fillRect(x0, botY, pw, botH);
      g.strokeStyle = SCOPE.grid; g.strokeRect(x0 + 0.5, botY + 0.5, pw - 1, botH - 1);
      for (const v of [0, -1]) {
        g.strokeStyle = v === 0 ? 'rgba(235,230,216,0.18)' : 'rgba(235,230,216,0.3)'; g.setLineDash(v === 0 ? [] : [2, 4]);
        g.beginPath(); g.moveTo(x0, Math.round(Yb(v)) + 0.5); g.lineTo(x0 + pw, Math.round(Yb(v)) + 0.5); g.stroke(); g.setLineDash([]);
      }
      label(c, p.title, x0, botY - 10, SCOPE.text, 'left', narrow ? 9 : 10);
      p.draw(x0);
    });
    label(c, '0', padL - 6, Yb(0), SCOPE.faint, 'right', 9);
    if (narrow) label(c, 'depth', padL + 3, Yb(-1) - 7, SCOPE.faint, 'left', 8);
    else label(c, '−depth', padL - 6, Yb(-1), SCOPE.faint, 'right', 9);
  }

  function dots(c: Canvas2D, t: { data: Float32Array }, n: number, row: number, mrow: number, X: (x: number) => number, a: number, b: number, Y: (v: number) => number, color: string, s: number) {
    const g = c.ctx;
    g.fillStyle = color;
    for (let i = 0; i < n; i++) {
      if (!(t.data[mrow * n + i] > 0)) continue;
      const x = a + ((b - a) * (i + 0.5)) / n;
      g.fillRect(X(x) - s / 2, Y(t.data[row * n + i]) - s / 2, s, s);
    }
  }
  function foot(c: Canvas2D, txt: string[], xs: number[], y: number) {
    txt.forEach((t, i) => t && label(c, t, xs[i], y, SCOPE.faint, 'center', 9));
  }

  await run();
}
