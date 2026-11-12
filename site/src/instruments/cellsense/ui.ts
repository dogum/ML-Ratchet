// CellSense live instrument: one real cell (Severson et al. 2019, b3c33) read at seven points through its life by the
// real networks, in the browser. Things to change: where the discharge stops (a partial discharge goes to the third
// network), voltage noise, and whether the network gets to compare with the cell's curve from when it was new.
import type { InstrumentContext } from '../types.ts';
import { fetchPack } from '../../lib/nn/pack.ts';
import { diagnose, L, type CMeta, type CExamples, type Diagnosis } from './net.ts';
import { shell, picker, slider, button, scheduler, gauges, canvas, gridLines, label, rng, fmt, h, SCOPE, type Canvas2D } from '../kit.ts';

const STYLE = `
.cs-block { display: flex; flex-direction: column; gap: 7px; }
.cs-row { display: grid; grid-template-columns: 74px minmax(0, 1fr) 92px; gap: 10px; align-items: center; }
.cs-row .n { font-size: 12.5px; color: var(--scr-2ink); }
.cs-row .v { font-family: var(--mono); font-variation-settings: 'wdth' 87.5; font-size: 12px; text-align: right; color: var(--scr-ink); white-space: nowrap; }
.cs-row .v small { font-size: 9.5px; color: var(--scr-3ink); margin-left: 3px; }
.cs-trk { position: relative; height: 12px; border-bottom: 1px solid var(--scr-rule); }
.cs-trk .z { position: absolute; top: 0; bottom: -1px; width: 0; border-left: 1px dashed var(--scr-rule); }
.cs-trk .b { position: absolute; top: 2px; height: 8px; border-radius: 2px; background: color-mix(in srgb, var(--c) 32%, transparent); transition: all 0.35s var(--ease); }
.cs-trk .d { position: absolute; top: 1px; width: 10px; height: 10px; margin-left: -5px; border-radius: 50%; background: var(--c); box-shadow: 0 0 10px var(--c); transition: left 0.35s var(--ease); }
.cs-axis { display: flex; justify-content: space-between; margin-left: 84px; margin-right: 102px; font-family: var(--mono); font-size: 9px; color: var(--scr-3ink); }
.cs-head { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: 8px; }
.cs-tag { font-family: var(--mono); font-variation-settings: 'wdth' 87.5; font-size: 9.5px; letter-spacing: 0.06em; text-transform: uppercase; padding: 2px 6px; border-radius: 3px; border: 1px solid var(--scr-rule); color: var(--scr-2ink); }
.cs-tag.part { color: var(--scr); background: var(--c); border-color: var(--c); }
.cs-off { font-size: 12px; line-height: 1.45; color: var(--scr-3ink); border: 1px dashed var(--scr-rule); border-radius: 4px; padding: 9px 10px; }
.cs-off b { font-weight: 400; color: var(--scr-2ink); }
.cs-left { font-family: var(--mono); font-variation-settings: 'wdth' 87.5; font-size: 10px; color: var(--scr-3ink); margin-top: 5px; }
.cs-left b { font-weight: 400; color: var(--scr-2ink); }
.cs-note { font-size: 12px; line-height: 1.5; color: var(--scr-3ink); margin: 0; }
.cs-note b { font-weight: 500; color: var(--scr-2ink); }
.cs-btns { display: flex; flex-direction: column; align-items: flex-start; justify-content: flex-end; gap: 8px; }
.cs-btns .inst-btn { align-self: stretch; }
@media (max-width: 760px) { .cs-row { grid-template-columns: 64px minmax(0, 1fr) 86px; } .cs-axis { margin-left: 74px; margin-right: 96px; } }
`;

const FULL = 1.1; // the stop slider's top: never stop, discharge all the way to 2.0 V

export async function mount(ctx: InstrumentContext) {
  const [W, X] = await Promise.all([fetchPack<CMeta>(`${ctx.base}weights`), fetchPack<CExamples>(`${ctx.base}examples`)]);
  const E = X.meta, VG = X.t.vgrid.data;
  const row = (t: { data: Float32Array }, i: number) => t.data.slice(i * L, (i + 1) * L);
  const Q0 = row(X.t.Q, 0), T0 = row(X.t.T, 0);
  const life = E.cycle_life;

  if (!document.getElementById('cs-inst-style')) document.head.append(h('style', { id: 'cs-inst-style' }, STYLE));
  const ui = shell(ctx.root, {
    title: 'CellSense',
    subtitle: `one real cell (${E.cell}), read at seven points through its life — never used in training`,
    params: '1.93 M parameters in three networks · matches Python to 2e-5',
  });
  const st = { i: 4, stop: FULL, noise: 0, ref: true, seed: 3 };
  let view: { q: Float32Array; kept: number; r: Diagnosis | null } = { q: row(X.t.Q, st.i), kept: L, r: null };

  const items = E.cycles.slice(1).map((c, k) => ({ label: `cycle ${c}`, sub: `measured ${fmt(E.measured_q_total[k + 1]!, 3)} Ah`, value: k + 1 }));
  picker(ui.picker, items, (i) => { st.i = i; run(); }, st.i - 1);

  const narrow = () => ui.screen.getBoundingClientRect().width < 520;
  const plot = canvas(ui.screen, ctx.compact ? 1.9 : narrow() ? 1.15 : 1.6, (c) => draw(c));
  slider(ui.controls, {
    label: 'Stop the discharge at', min: 0.3, max: FULL, step: 0.01, value: FULL,
    fmt: (v) => (v >= FULL ? '2.0 V' : `${v.toFixed(2)} Ah`),
    hint: 'the rest of the curve is thrown away; a curve that stops short goes to the partial-discharge network',
    onInput: (v) => { st.stop = v; go(); },
  });
  slider(ui.controls, {
    label: 'Voltage noise', min: 0, max: 10, step: 0.25, value: 0,
    fmt: (v) => (v === 0 ? 'none' : `σ ${v.toFixed(1)} mV`),
    hint: 'Gaussian error on each voltage reading, before the curve is put on the grid',
    onInput: (v) => { st.noise = v; go(); },
  });
  if (!ctx.compact) {
    const btns = h('div', { class: 'cs-btns' });
    ui.controls.append(btns);
    const refBtn = button(btns, 'Reference curve: on', () => {
      st.ref = !st.ref;
      refBtn.textContent = `Reference curve: ${st.ref ? 'on' : 'off'}`;
      refBtn.setAttribute('aria-pressed', String(st.ref));
      go();
    });
    refBtn.setAttribute('aria-pressed', 'true');
    refBtn.title = 'Give the networks the same cell’s discharge from when it was new (cycle 7), or read the curve on its own';
    button(btns, 'New noise draw', () => { st.seed++; if (st.noise > 0) go(); });
  }

  const capBox = h('div', {}), modeBox = h('div', { class: 'cs-block' }), lifeBox = h('div', {}), note = h('p', { class: 'cs-note' });
  ui.readout.append(capBox, modeBox, lifeBox, note);

  /** The discharge as the networks get it: voltage noise (re-gridded), then the cut. */
  function input(): { Q: Float32Array; T: Float32Array; kept: number } {
    const q0 = row(X.t.Q, st.i), T = row(X.t.T, st.i);
    let Q = q0;
    if (st.noise > 0) {
      // each grid level V reads the clean curve at V + noise: on the flat LFP plateau a few mV is a lot of charge
      const r = rng(st.seed * 977 + st.i), dv = VG[0] - VG[1];
      Q = new Float32Array(L);
      for (let k = 0; k < L; k++) {
        const f = Math.max(0, Math.min(L - 1, k - (st.noise / 1000) * r.n() / dv)), a = Math.floor(f), b = Math.min(L - 1, a + 1);
        Q[k] = q0[a] + (q0[b] - q0[a]) * (f - a);
      }
    }
    let kept = L;
    if (st.stop < FULL) for (let k = 0; k < L; k++) if (Q[k] > st.stop) { Q[k] = NaN; T[k] = NaN; if (kept === L) kept = k; }
    return { Q, T, kept };
  }

  async function run() {
    ui.setBusy(true);
    await new Promise((r) => requestAnimationFrame(r));
    const i = st.i, cyc = E.cycles[i];
    const { Q, T, kept } = input();
    const t0 = performance.now();
    const r = diagnose(W, Q, T, st.ref ? Q0 : null, st.ref ? T0 : null, cyc, E.charge_time);
    const ms = performance.now() - t0;
    view = { q: st.noise > 0 ? noisyFull() : row(X.t.Q, i), kept, r };
    plot.redraw();
    render(r, i);
    const nets = r.partial ? 'partial-discharge network' : 'health + lifetime networks';
    ui.stamp(ms, `cell ${E.cell} at cycle ${cyc} · ${nets}`);
    ui.setBusy(false);
  }
  /** the whole noisy curve (for drawing the thrown-away part too) */
  function noisyFull() { const s = st.stop; st.stop = FULL; const { Q } = input(); st.stop = s; return Q; }
  const go = scheduler(run, 30);

  function render(r: Diagnosis, i: number) {
    const mq = E.measured_q_total[i]!, mir = E.measured_ir_mohm[i]!;
    gauges(capBox, [
      { label: 'Total capacity', unit: 'Ah', value: r.q_total, lo: r.q_total - r.q_total_sig, hi: r.q_total + r.q_total_sig, truth: mq, truthLabel: 'measured (with hold)', range: [0.8, 1.15], digits: 3 },
      { label: 'Pulse resistance', unit: 'mΩ', value: r.ir, lo: r.ir - r.ir_sig, hi: r.ir + r.ir_sig, truth: mir, truthLabel: 'measured', range: [10, 24], digits: 1 },
    ]);
    // degradation modes: no instrument measures these on a real cell, so no truth tick
    const pos = (v: number) => `${Math.max(0, Math.min(1, (v * 100 + 2) / 37)) * 100}%`;
    const modes: [string, string, string][] = [['Lithium', 'lli', 'lithium inventory'], ['Graphite', 'lam_ne', 'negative electrode'], ['LFP', 'lam_pe', 'positive electrode']];
    modeBox.innerHTML = `<div class="cs-head"><span class="ig-l">Lost since new</span><span class="cs-tag${r.partial ? ' part' : ''}">${r.partial ? 'partial-discharge net' : 'health net'}</span></div>`
      + modes.map(([n, k, t]) => {
        const v = r[k], s = r[`${k}_sig`];
        return `<div class="cs-row" title="${t}"><span class="n">${n}</span><span class="cs-trk"><span class="z" style="left:${pos(0)}"></span><span class="b" style="left:${pos(v - s)};right:calc(100% - ${pos(v + s)})"></span><span class="d" style="left:${pos(v)}"></span></span><span class="v">${(100 * v).toFixed(1)}%<small>±${(100 * s).toFixed(1)}</small></span></div>`;
      }).join('')
      + `<div class="cs-axis"><span>0%</span><span>10%</span><span>20%</span><span>30%</span></div>`;
    // the forecast
    const left = life - E.cycles[i];
    if (r.partial) {
      lifeBox.innerHTML = `<div class="ig-l" style="margin-bottom:6px">Cycle life</div><div class="cs-off"><b>No forecast from a partial discharge.</b> The lifetime network reads full discharges to 2.0 V; only the third network learned curves cut short. Let the discharge run to the end to see the forecast. This cell’s life ended at cycle <b>${life}</b>.</div>`;
    } else {
      lifeBox.innerHTML = '';
      gauges(lifeBox, [{ label: 'Cycle life', unit: 'cycles', value: r.life, lo: r.life_lo, hi: r.life_hi, truth: life, truthLabel: 'real end of life', range: [300, 4000], digits: 0, log: true }]);
      lifeBox.append(h('div', { class: 'cs-left' }, 'until 0.88 Ah · cycles left ', h('b', {}, `${Math.round(r.left)} ± ${Math.round(r.left_sig)}`), ` · really ${left}`));
    }
    // what the reading is telling you right now
    const bits: string[] = [];
    if (r.partial) {
      bits.push(`<b>Partial discharge, stopped at ${st.stop.toFixed(2)} Ah.</b> The third network reads what is left${st.stop < 0.75 ? '; below about 0.75 Ah it does no better than assuming no fade' : ' and reads well down to about 0.75 Ah'}. Lithium and graphite loss show near the end of the curve, which is gone, so the network sees little of them.`);
    } else if (st.stop < FULL) {
      bits.push(`<b>The cell reached 2.0 V before ${st.stop.toFixed(2)} Ah,</b> so this is still a full discharge and the health network reads it.`);
    } else bits.push(`<b>Full 4C discharge to 2.0 V.</b> The health network reads capacity, resistance and degradation; the lifetime network adds the cycle number and the ${E.charge_time.toFixed(0)}-minute charge.`);
    if (st.noise > 0) bits.push(`With ${st.noise.toFixed(1)} mV of noise, capacity holds; resistance${r.partial ? '' : ' and the life forecast'}, read from the curve’s fine shape, move${r.partial ? 's' : ''} with each noise draw.`);
    if (!st.ref) bits.push('Without the new-cell curve the networks read this discharge on its own, and the degradation-mode error bars widen.');
    note.innerHTML = bits.join(' ');
  }

  function draw(c: Canvas2D) {
    const { ctx: g, w, h: hh } = c;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, w, hh);
    const padL = 36, padR = 12, top = 16, bot = 40;
    const iw = w - padL - padR, ih = hh - top - bot;
    const q0 = 0, q1 = 1.15, v0 = 1.7, v1 = 3.6;
    const X_ = (q: number) => padL + ((q - q0) / (q1 - q0)) * iw;
    const Y_ = (v: number) => top + (1 - (v - v0) / (v1 - v0)) * ih;
    gridLines(c, padL, top, iw, ih, 23, 19);
    // axes ticks
    for (let q = 0; q <= 1.1001; q += 0.2) label(c, q.toFixed(1), X_(q), hh - bot + 11, SCOPE.faint, 'center', 9);
    label(c, 'charge out (Ah) →', padL + iw, hh - bot + 24, SCOPE.faint, 'right', 9);
    for (const v of [2.0, 2.5, 3.0, 3.5]) label(c, v.toFixed(1), padL - 6, Y_(v), SCOPE.faint, 'right', 9);
    label(c, 'V', padL - 6, top - 6, SCOPE.faint, 'right', 9);

    const line = (q: ArrayLike<number>, i0: number, i1: number, color: string, width: number, alpha = 1, dash: number[] = []) => {
      g.strokeStyle = color; g.lineWidth = width; g.globalAlpha = alpha; g.setLineDash(dash);
      g.beginPath();
      let s = false;
      for (let k = i0; k < i1; k++) { const v = q[k]; if (!Number.isFinite(v)) continue; const x = X_(v), y = Y_(VG[k]); if (!s) { g.moveTo(x, y); s = true; } else g.lineTo(x, y); }
      g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
    };
    // 2.0 V floor
    g.strokeStyle = SCOPE.faint; g.setLineDash([2, 4]); g.beginPath(); g.moveTo(padL, Y_(2.0) + 0.5); g.lineTo(padL + iw, Y_(2.0) + 0.5); g.stroke(); g.setLineDash([]);
    // the new cell
    line(Q0, 0, L, SCOPE.ink, 1.2, st.ref ? 0.45 : 0.22, st.ref ? [] : [4, 4]);
    // the discharge being read, and what was thrown away
    const { q, kept, r } = view;
    if (st.noise > 0) {
      // with noise, draw what the network gets: one reading per grid voltage
      for (let k = 0; k < L; k++) {
        if (!Number.isFinite(q[k])) continue;
        g.fillStyle = ctx.accent; g.globalAlpha = k < kept ? 0.95 : 0.25;
        g.beginPath(); g.arc(X_(q[k]), Y_(VG[k]), k < kept ? 1.7 : 1.2, 0, 7); g.fill();
      }
      g.globalAlpha = 1;
    } else {
      if (kept < L) line(q, Math.max(0, kept - 1), L, ctx.accent, 1.2, 0.3, [3, 3]);
      line(q, 0, kept, ctx.accent, 2.2);
    }
    const qEnd = q[L - 1];
    // the cut
    if (kept < L) {
      const x = X_(st.stop);
      g.fillStyle = 'rgba(0,0,0,0.5)';
      g.fillRect(x, top, padL + iw - x, ih);
      g.strokeStyle = ctx.accent; g.lineWidth = 1; g.beginPath(); g.moveTo(x + 0.5, top); g.lineTo(x + 0.5, top + ih); g.stroke();
      label(c, `stopped at ${st.stop.toFixed(2)} Ah`, x - 6, top + 10, ctx.accent, 'right', 9.5);
      label(c, 'thrown away', x + 6, top + 10, SCOPE.faint, 'left', 9);
    }
    // legend
    const lx0 = padL + 14, ly0 = Y_(2.75);
    const sw = (y: number, color: string, alpha: number, width: number, text: string, dash: number[] = []) => {
      g.strokeStyle = color; g.globalAlpha = alpha; g.lineWidth = width; g.setLineDash(dash);
      g.beginPath(); g.moveTo(lx0, y); g.lineTo(lx0 + 18, y); g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
      label(c, text, lx0 + 26, y, SCOPE.text, 'left', 9.5);
    };
    const sm = w < 520;
    sw(ly0, ctx.accent, 1, 2.2, sm ? `cycle ${E.cycles[st.i]}, being read` : `cycle ${E.cycles[st.i]}: the 4C discharge being read`);
    sw(ly0 + 16, SCOPE.ink, st.ref ? 0.45 : 0.22, 1.2, `cycle ${E.cycles[0]}: ${sm ? 'new' : 'the same cell when new'}${st.ref ? '' : ' (not given)'}`, st.ref ? [] : [4, 4]);
    // total capacity: the reading (band) and the measurement (tick), on a strip under the 2.0 V floor
    const sy = Y_(v0);
    if (r) {
      const mq = E.measured_q_total[st.i]!;
      const ya = Y_(2.0) + 6;
      g.globalAlpha = 0.32; g.fillStyle = ctx.accent;
      g.fillRect(X_(r.q_total - r.q_total_sig), ya, X_(r.q_total + r.q_total_sig) - X_(r.q_total - r.q_total_sig), sy - ya - 4);
      g.globalAlpha = 1;
      g.fillStyle = ctx.accent; g.beginPath(); g.arc(X_(r.q_total), (ya + sy - 4) / 2, 3.5, 0, 7); g.fill();
      g.strokeStyle = SCOPE.ink; g.lineWidth = 2; g.beginPath(); g.moveTo(X_(mq), ya - 3); g.lineTo(X_(mq), sy - 1); g.stroke();
      // the hold: charge that comes out at 2.0 V after the 4C part ends
      if (!r.partial && Number.isFinite(qEnd) && mq > qEnd) {
        g.strokeStyle = SCOPE.ink; g.globalAlpha = 0.5; g.lineWidth = 1; g.setLineDash([1, 2]);
        g.beginPath(); g.moveTo(X_(qEnd), Y_(2.0) + 0.5); g.lineTo(X_(mq), Y_(2.0) + 0.5); g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
      }
      // key, left to right; short words if the full key would run into the band
      const my = (ya + sy - 4) / 2, stopX = X_(Math.min(r.q_total - r.q_total_sig, mq)) - 10;
      g.font = '9px "Martian Mono Variable", ui-monospace, monospace';
      const tw = (t: string) => g.measureText(t).width;
      let words = ['total capacity', 'reading ±1σ', 'measured'];
      if (padL + 8 + tw(words.join('')) + 64 > stopX) words = ['capacity', '±1σ', 'measured'];
      let x = padL + 8;
      const word = (t: string) => { label(c, t, x, my, SCOPE.text, 'left', 9); x += tw(t) + 8; };
      word(words[0]);
      g.globalAlpha = 0.5; g.fillStyle = ctx.accent; g.fillRect(x, my - 4, 10, 8); g.globalAlpha = 1; x += 15;
      word(words[1]);
      g.fillStyle = SCOPE.ink; g.fillRect(x, my - 5, 2, 10); x += 7;
      word(words[2]);
    }
  }

  await run();
}
