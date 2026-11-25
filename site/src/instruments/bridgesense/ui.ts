// BridgeSense live instrument: 8 real zero-shot exam cases (Concrete Box-Girder + Hell Bridge Test Arena,
// CC BY 4.0), the real 179k-param Dual-Axis Transformer running live in the browser, with honest physical
// controls (injected stiffness change, cold-snap temperature shift, added OMA frequency noise).
import type { InstrumentContext } from '../types.ts';
import { fetchPack } from '../../lib/nn/pack.ts';
import { predict, N_STEPS, MAX_MODES, type BSMeta, type BSExample, type BSReading } from './net.ts';
import { shell, picker, slider, button, scheduler, gauges, probs, canvas, gridLines, label, rng, SCOPE, type Canvas2D } from '../kit.ts';

const SHORT: Record<string, string> = {
  boxgirder_b00: 'Concrete freeze #1',
  boxgirder_b03: 'Concrete freeze #2',
  boxgirder_b10: 'Concrete load +680 kg',
  boxgirder_b14: 'Concrete load +2160 kg',
  MVS_P2_UDS_SM_Y_02: 'HBTA undamaged',
  MVS_P1_DS4_SM_Y_01: 'HBTA local DS4',
  MVS_P2_DS6_NM_Z_01: 'HBTA global DS6 (Z)',
  MVS_P2_DS6_SM_Y_01: 'HBTA global DS6 (Y)',
};

export async function mount(ctx: InstrumentContext) {
  const [W, X] = await Promise.all([
    fetchPack<BSMeta>(`${ctx.base}weights`),
    fetchPack<BSExample[]>(`${ctx.base}examples`),
  ]);
  const ex = X.meta;
  const ui = shell(ctx.root, {
    title: 'BridgeSense',
    subtitle: 'tracked modal frequencies & temperature — zero-shot real exam bridges (CC BY 4.0)',
    params: '179k parameters · Dual-Axis Transformer · matches Python to <1e-6',
  });

  const st = { i: 1, dkInject: 0, dT: 0, noisePct: 0, dropPct: 0, seed: 11 };
  let view: {
    T_ref: Float32Array; T_q: Float32Array;
    df_ref: Float32Array; df_q: Float32Array;
    mask_ref: Float32Array; mask_q: Float32Array;
    r: BSReading | null;
  } = {
    T_ref: X.t.e1_T_ref.data, T_q: X.t.e1_T_q.data,
    df_ref: X.t.e1_df_ref.data, df_q: X.t.e1_df_q.data,
    mask_ref: X.t.e1_mask_ref.data, mask_q: X.t.e1_mask_q.data,
    r: null,
  };

  picker(
    ui.picker,
    ex.map((e, i) => ({
      label: SHORT[e.id] ?? e.id,
      sub: `${e.bridge === 'boxgirder' ? 'Concrete · 2 modes' : 'Steel truss · 7 modes'} · Tmin ${e.T_min.toFixed(1)} °C`,
      value: i,
    })),
    (i) => { st.i = i; run(); },
    st.i,
  );

  const plot = canvas(ui.screen, ctx.compact ? 2.3 : 1.95, (c) => draw(c));
  slider(ui.controls, {
    label: 'Inject stiffness shift ΔK/K',
    min: -10, max: 10, step: 0.5, value: 0,
    fmt: (v) => (v === 0 ? 'as recorded' : `${v > 0 ? '+' : ''}${v.toFixed(1)} %`),
    hint: 'scales query frequencies by √(1 + ΔK/K)',
    onInput: (v) => { st.dkInject = v; go(); },
  });
  slider(ui.controls, {
    label: 'Query temperature shift',
    min: -10, max: 10, step: 0.5, value: 0,
    fmt: (v) => (v === 0 ? 'as recorded' : `${v > 0 ? '+' : ''}${v.toFixed(1)} °C`),
    hint: 'shifts query temperature into or out of freezing',
    onInput: (v) => { st.dT = v; go(); },
  });
  if (!ctx.compact) {
    slider(ui.controls, {
      label: 'Add OMA frequency noise',
      min: 0, max: 1.5, step: 0.05, value: 0,
      fmt: (v) => (v === 0 ? 'none' : `σ ${v.toFixed(2)} %`),
      hint: 'Gaussian jitter on query mode frequencies',
      onInput: (v) => { st.noisePct = v; go(); },
    });
    slider(ui.controls, {
      label: 'Drop query mode readings',
      min: 0, max: 70, step: 10, value: 0,
      fmt: (v) => (v === 0 ? 'none' : `${v.toFixed(0)} %`),
      hint: 'masks random query mode observations (tests masked attention)',
      onInput: (v) => { st.dropPct = v; go(); },
    });
    button(ui.controls, 'New noise / dropout draw', () => { st.seed++; go(); });
  }

  function makeInputs() {
    const i = st.i;
    const T_ref = X.t[`e${i}_T_ref`].data;
    const T_q = new Float32Array(X.t[`e${i}_T_q`].data);
    const df_ref = X.t[`e${i}_df_ref`].data;
    const df_q = new Float32Array(X.t[`e${i}_df_q`].data);
    const mask_ref = X.t[`e${i}_mask_ref`].data;
    const mask_q = new Float32Array(X.t[`e${i}_mask_q`].data);

    if (st.dT !== 0) for (let k = 0; k < T_q.length; k++) T_q[k] += st.dT;
    const scaleF = Math.sqrt(Math.max(0.5, 1 + st.dkInject / 100));
    const gen = rng(st.seed * 131 + i);
    const nSig = st.noisePct / 100;
    const dropP = st.dropPct / 100;
    for (let k = 0; k < df_q.length; k++) {
      if (mask_q[k] > 0.5) {
        if (dropP > 0 && gen.u() < dropP) {
          mask_q[k] = 0;
          df_q[k] = 0;
          continue;
        }
        let v = (1 + df_q[k]) * scaleF - 1;
        if (nSig > 0) v += nSig * gen.n();
        df_q[k] = v;
      } else {
        df_q[k] = 0;
      }
    }
    return { T_ref, T_q, df_ref, df_q, mask_ref, mask_q };
  }

  async function run() {
    ui.setBusy(true);
    await new Promise((r) => requestAnimationFrame(r));
    const e = ex[st.i];
    const inp = makeInputs();
    const t0 = performance.now();
    const r = predict(W, inp.T_ref, inp.T_q, inp.df_ref, inp.df_q, inp.mask_ref, inp.mask_q);
    const ms = performance.now() - t0;
    view = { ...inp, r };
    plot.redraw();

    const trueShifted = e.true_dk_pct !== null ? e.true_dk_pct + st.dkInject : null;
    const isChgTruth = st.dkInject !== 0 ? true : e.is_changed === 1;
    ui.readout.innerHTML = '';
    probs(ui.readout, 'Structural state', [
      { label: 'Structural change', p: r.p_change, truth: isChgTruth },
      { label: 'Thermal / weather only', p: 1 - r.p_change, truth: !isChgTruth },
    ]);
    gauges(ui.readout, [
      {
        label: 'Stiffness shift ΔK/K', unit: '%', value: r.delta_k_pct,
        lo: r.delta_k_pct - r.delta_k_sigma, hi: r.delta_k_pct + r.delta_k_sigma,
        truth: trueShifted, truthLabel: st.dkInject === 0 ? 'exam truth' : 'truth + injected',
        range: [-10, 10], digits: 2,
      },
      {
        label: 'Thermal freq shift', unit: '%', value: r.thermal_shift_pct,
        lo: r.thermal_shift_pct - r.thermal_shift_sigma, hi: r.thermal_shift_pct + r.thermal_shift_sigma,
        truth: null, range: [-5, 5], digits: 2,
      },
      {
        label: 'Linear OLS anchor ΔK/K', unit: '%', value: r.anchor_dk_pct,
        lo: r.anchor_dk_pct, hi: r.anchor_dk_pct,
        truth: trueShifted, truthLabel: 'classical OLS', range: [-10, 10], digits: 2,
      },
    ]);
    ui.stamp(ms, `${e.bridge_label} · ${e.truth_label} (${e.licence})`);
    ui.setBusy(false);
  }
  const go = scheduler(run, 30);

  function draw(c: Canvas2D) {
    const { ctx: g, w, h } = c;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, w, h);
    const padL = 38, padR = 14, top = 16, midGap = 18, bot = 22;
    const iw = w - padL - padR;
    const hTop = Math.floor((h - top - midGap - bot) * 0.56);
    const hBot = h - top - midGap - bot - hTop;
    const yBot = top + hTop + midGap;

    gridLines(c, padL, top, iw, hTop, 8, 4);
    gridLines(c, padL, yBot, iw, hBot, 8, 3);

    // Divider between baseline (left half) and query (right half)
    const xMid = padL + iw * 0.5;
    g.strokeStyle = SCOPE.faint;
    g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(xMid, top); g.lineTo(xMid, top + hTop); g.moveTo(xMid, yBot); g.lineTo(xMid, yBot + hBot); g.stroke();
    g.setLineDash([]);

    label(c, 'BASELINE (healthy ref)', padL + 4, top + 10, SCOPE.text, 'left', 9);
    label(c, 'QUERY WINDOW', xMid + 6, top + 10, ctx.accent, 'left', 9);
    label(c, 'Δf/f (%)', padL - 4, top + hTop / 2, SCOPE.text, 'right', 9);
    label(c, 'Air T (°C)', padL - 4, yBot + hBot / 2, SCOPE.text, 'right', 9);

    // Top panel: mean fractional frequency shift (%) across active modes
    const meanSeries = (df: Float32Array, msk: Float32Array) => {
      const out = new Float32Array(N_STEPS);
      for (let t = 0; t < N_STEPS; t++) {
        let s = 0, cnt = 0;
        for (let m = 0; m < MAX_MODES; m++) if (msk[t * MAX_MODES + m] > 0.5) { s += df[t * MAX_MODES + m] * 100; cnt++; }
        out[t] = cnt > 0 ? s / cnt : 0;
      }
      return out;
    };
    const fRef = meanSeries(view.df_ref, view.mask_ref);
    const fQ = meanSeries(view.df_q, view.mask_q);
    let maxF = 1.5;
    for (let t = 0; t < N_STEPS; t++) maxF = Math.max(maxF, Math.abs(fRef[t]), Math.abs(fQ[t]));
    maxF = Math.ceil(maxF * 1.15);

    const drawHalfLine = (vals: Float32Array, x0: number, spanW: number, y0: number, spanH: number, lo: number, hi: number, col: string) => {
      g.strokeStyle = col;
      g.lineWidth = 1.7;
      g.beginPath();
      for (let t = 0; t < N_STEPS; t++) {
        const x = x0 + (t / (N_STEPS - 1)) * spanW;
        const y = y0 + spanH * (1 - (vals[t] - lo) / (hi - lo));
        if (t === 0) g.moveTo(x, y); else g.lineTo(x, y);
      }
      g.stroke();
    };

    // Zero line for Δf/f
    const yZeroF = top + hTop * 0.5;
    g.strokeStyle = SCOPE.faint; g.lineWidth = 1;
    g.beginPath(); g.moveTo(padL, yZeroF); g.lineTo(padL + iw, yZeroF); g.stroke();
    label(c, `+${maxF}%`, padL - 4, top + 8, SCOPE.faint, 'right', 8);
    label(c, `-${maxF}%`, padL - 4, top + hTop - 2, SCOPE.faint, 'right', 8);

    drawHalfLine(fRef, padL, iw * 0.48, top, hTop, -maxF, maxF, SCOPE.ink);
    drawHalfLine(fQ, xMid + iw * 0.02, iw * 0.48, top, hTop, -maxF, maxF, ctx.accent);

    // Bottom panel: air temperature (°C) with 0 °C freezing threshold
    const tRef = new Float32Array(N_STEPS), tQ = new Float32Array(N_STEPS);
    let tLo = -10, tHi = 25;
    for (let t = 0; t < N_STEPS; t++) {
      tRef[t] = view.T_ref[t * 3]; tQ[t] = view.T_q[t * 3];
      tLo = Math.min(tLo, tRef[t] - 2, tQ[t] - 2);
      tHi = Math.max(tHi, tRef[t] + 2, tQ[t] + 2);
    }
    const yZeroT = yBot + hBot * (1 - (0 - tLo) / (tHi - tLo));
    if (yZeroT >= yBot && yZeroT <= yBot + hBot) {
      g.strokeStyle = '#38bdf8'; g.globalAlpha = 0.45; g.setLineDash([4, 3]);
      g.beginPath(); g.moveTo(padL, yZeroT); g.lineTo(padL + iw, yZeroT); g.stroke();
      g.setLineDash([]); g.globalAlpha = 1;
      label(c, '0 °C freeze', padL + iw - 4, yZeroT - 4, '#38bdf8', 'right', 8);
    }
    drawHalfLine(tRef, padL, iw * 0.48, yBot, hBot, tLo, tHi, SCOPE.ink);
    drawHalfLine(tQ, xMid + iw * 0.02, iw * 0.48, yBot, hBot, tLo, tHi, '#38bdf8');
  }

  await run();
}
