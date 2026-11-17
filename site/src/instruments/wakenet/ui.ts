// WakeNet live browser instrument: six real exam snapshots across Kelmarsh (6 x MM92) and the unseen
// 14-turbine Penmanshiel farm (14 x MM82). Renders the live 2-D aerodynamic wake plume map and
// per-turbine wake losses, and runs the 246k-parameter EdgeWakeGNN in the browser on every change.
import type { InstrumentContext } from '../types.ts';
import { fetchPack } from '../../lib/nn/pack.ts';
import { read, wakePlumeGrid, type WMeta, type WExampleMeta, type WakeReading } from './net.ts';
import { shell, picker, slider, button, scheduler, gauges, canvas, label, hex, h, SCOPE, type Canvas2D } from '../kit.ts';

const SHORT: Record<string, string> = {
  '01_kelmarsh_sw_wakes': 'Kelmarsh · SW wakes',
  '02_kelmarsh_nw_3row': 'Kelmarsh · NW 3-row',
  '03_kelmarsh_yaw_offset': 'Kelmarsh · yaw ≥10°',
  '04_kelmarsh_parked_unit': 'Kelmarsh · 1 parked',
  '05_penmanshiel_sw_14wt': 'Penmanshiel · 14 WT SW',
  '06_penmanshiel_nw_yaw': 'Penmanshiel · 14 WT ridge',
};

export async function mount(ctx: InstrumentContext) {
  const [W, X] = await Promise.all([
    fetchPack<WMeta>(`${ctx.base}weights`),
    fetchPack<WExampleMeta[]>(`${ctx.base}examples`),
  ]);
  const ex = X.meta;
  const ui = shell(ctx.root, {
    title: 'WakeNet',
    subtitle: 'wind-farm wakes read as a directed graph — real Kelmarsh & zero-shot Penmanshiel exam snapshots',
    params: '246k parameters · matches Python to 1e-6',
  });

  const st = {
    i: 0,
    dWd: 0,
    steer: 0,
    swd: ex[0].sigma_wd_deg,
    toggled: new Set<number>(),
    selected: -1,
  };

  let view: { r: WakeReading | null; yaw: Float64Array; online: boolean[]; wd: number } = {
    r: null, yaw: new Float64Array(0), online: [], wd: 0,
  };
  let hitTargets: { idx: number; x: number; y: number }[] = [];

  const swdSliderCtl = { set: (_v: number) => {} };
  const wdSliderCtl = { set: (_v: number) => {} };
  const steerSliderCtl = { set: (_v: number) => {} };

  picker(
    ui.picker,
    ex.map((e, i) => ({
      label: SHORT[e.id] ?? e.id,
      sub: `${e.codes.length}× ${e.turbine_model} · ${e.u_inf_ms.toFixed(1)} m/s @ ${Math.round(e.wind_dir_deg)}°`,
      value: i,
    })),
    (i) => {
      st.i = i;
      st.dWd = 0;
      st.steer = 0;
      st.swd = ex[i].sigma_wd_deg;
      st.toggled.clear();
      st.selected = -1;
      wdSliderCtl.set(0);
      steerSliderCtl.set(0);
      swdSliderCtl.set(st.swd);
      go();
    },
    st.i,
  );

  const plot = canvas(ui.screen, ctx.compact ? 1.65 : 1.42, (c) => draw(c));

  // Click any turbine on the map to select it; Shift-click (or double-click) parks/unparks it
  const cvEl = ui.screen.querySelector('canvas');
  if (cvEl) {
    cvEl.style.cursor = 'pointer';
    cvEl.addEventListener('click', (ev) => {
      const rect = cvEl.getBoundingClientRect();
      const px = ev.clientX - rect.left;
      const py = ev.clientY - rect.top;
      let best = -1, bestD = 22;
      for (const t of hitTargets) {
        const d = Math.hypot(px - t.x, py - t.y);
        if (d < bestD) { bestD = d; best = t.idx; }
      }
      if (best >= 0) {
        if (ev.shiftKey || st.selected === best) {
          if (st.toggled.has(best)) st.toggled.delete(best);
          else st.toggled.add(best);
          go();
        } else {
          st.selected = best;
          if (view.r) updateReadout(view.r);
          plot.redraw();
        }
      }
    });
  }

  const s1 = slider(ui.controls, {
    label: 'Rotate wind direction', min: -30, max: 30, step: 1, value: 0,
    fmt: (v) => (v === 0 ? `recorded (${Math.round(ex[st.i].wind_dir_deg)}°)` : `${v > 0 ? '+' : ''}${v}° → ${Math.round((ex[st.i].wind_dir_deg + v + 360) % 360)}°`),
    hint: 'sweeps upstream wakes across downstream rotors',
    onInput: (v) => { st.dWd = v; go(); },
  });
  wdSliderCtl.set = s1.set;

  const s2 = slider(ui.controls, {
    label: 'Front-row yaw steering', min: -24, max: 24, step: 1, value: 0,
    fmt: (v) => (v === 0 ? 'as recorded' : `${v > 0 ? '+' : ''}${v}°`),
    hint: 'deflects front-row wakes laterally at a cos^1.88 power cost',
    onInput: (v) => { st.steer = v; go(); },
  });
  steerSliderCtl.set = s2.set;

  if (!ctx.compact) {
    const s3 = slider(ui.controls, {
      label: 'Direction meandering σ_θ', min: 1, max: 16, step: 0.5, value: st.swd,
      fmt: (v) => `${v.toFixed(1)}°`,
      hint: '10-min wind-vane std dev · click a rotor twice on the map to park/unpark',
      onInput: (v) => { st.swd = v; go(); },
    });
    swdSliderCtl.set = s3.set;

    const btns = h('div', { style: 'display:flex;gap:8px;flex-wrap:wrap;grid-column:1/-1' });
    ui.controls.append(btns);
    button(btns, 'Park / unpark selected rotor', () => {
      const idx = st.selected >= 0 ? st.selected : 0;
      if (st.toggled.has(idx)) st.toggled.delete(idx);
      else st.toggled.add(idx);
      go();
    });
    button(btns, 'Reset to recorded SCADA', () => {
      st.dWd = 0; st.steer = 0; st.swd = ex[st.i].sigma_wd_deg; st.toggled.clear();
      wdSliderCtl.set(0); steerSliderCtl.set(0); swdSliderCtl.set(st.swd);
      go();
    });
  }

  function isModified(): boolean {
    return st.dWd !== 0 || st.steer !== 0 || Math.abs(st.swd - ex[st.i].sigma_wd_deg) > 0.05 || st.toggled.size > 0;
  }

  async function run() {
    ui.setBusy(true);
    await new Promise((r) => requestAnimationFrame(r));
    const i = st.i;
    const m = ex[i];
    const east = X.t[`east_${i}`].data;
    const north = X.t[`north_${i}`].data;
    const elev = X.t[`elev_${i}`].data;
    const hubH = X.t[`hub_h_${i}`].data;
    const baseYaw = X.t[`yaw_${i}`].data;
    const baseOn = Array.from(X.t[`online_${i}`].data, (v) => v > 0.5);
    const baseFree = Array.from(X.t[`free_${i}`].data, (v) => v > 0.5);

    const online = baseOn.map((on, k) => (st.toggled.has(k) ? !on : on));
    if (!online.some(Boolean)) online[0] = true;

    const wd = (m.wind_dir_deg + st.dWd + 360) % 360;
    const yaw = Float64Array.from(baseYaw, (y, k) => {
      if (!online[k]) return 0;
      return baseFree[k] ? Math.max(-30, Math.min(30, y + st.steer)) : y;
    });

    const t0 = performance.now();
    const r = read(W, east, north, elev, hubH, m.turbine_model, m.u_inf_ms, wd, m.ti_inf, st.swd, yaw, online, m.air_density, m.p_front_kw);
    const ms = performance.now() - t0;

    // Pick deepest waked turbine by default
    if (st.selected < 0 || st.selected >= east.length) {
      let deepest = 0, minEta = 99;
      for (let k = 0; k < east.length; k++) {
        if (online[k] && !r.is_free[k] && r.eta[k] < minEta) { minEta = r.eta[k]; deepest = k; }
      }
      st.selected = deepest;
    }

    view = { r, yaw, online, wd };
    plot.redraw();
    updateReadout(r);

    const farmNote = m.farm === 'penmanshiel'
      ? 'zero-shot unseen 14-turbine farm (steep 48 m ridge: farm wake loss loses to dev-mean 4.55% vs 4.75%, beats FLORIS 5.16%)'
      : `${m.stratum} · ${m.timestamp_utc} UTC`;
    ui.stamp(ms, farmNote);
    ui.setBusy(false);
  }

  const go = scheduler(run, 30);

  function updateReadout(r: WakeReading) {
    const i = st.i;
    const m = ex[i];
    const pTrue = X.t[`ptrue_${i}`].data;
    const baseOn = Array.from(X.t[`online_${i}`].data, (v) => v > 0.5);
    const mod = isModified();

    let sumTrueOn = 0, cntTrueOn = 0;
    for (let k = 0; k < pTrue.length; k++) {
      if (baseOn[k]) { sumTrueOn += pTrue[k] / m.p_front_kw; cntTrueOn++; }
    }
    const farmWakeTruePct = 100 * (1 - sumTrueOn / Math.max(cntTrueOn, 1));
    const farmPredPct = 100 * r.farm_wake_loss;
    const farmSigPct = 100 * r.farm_wake_sig;

    const sel = Math.max(0, Math.min(m.codes.length - 1, st.selected));
    const selCode = m.codes[sel];
    const selLossPct = 100 * r.wake_loss[sel];
    const selSigPct = 100 * r.eta_sig[sel];
    const selTrueLossPct = 100 * (1 - pTrue[sel] / m.p_front_kw);

    ui.readout.innerHTML = '';
    gauges(ui.readout, [
      {
        label: 'Farm wake loss',
        unit: '%',
        value: farmPredPct,
        lo: farmPredPct - farmSigPct,
        hi: farmPredPct + farmSigPct,
        truth: mod ? null : farmWakeTruePct,
        truthLabel: 'SCADA',
        range: [-10, 35],
        digits: 1,
      },
      {
        label: 'Total farm output',
        unit: 'MW',
        value: r.farm_power_kw / 1000,
        lo: (r.farm_power_kw - r.farm_free_kw * r.farm_wake_sig) / 1000,
        hi: (r.farm_power_kw + r.farm_free_kw * r.farm_wake_sig) / 1000,
        truth: mod ? null : Array.from(pTrue).reduce((a, b) => a + b, 0) / 1000,
        truthLabel: 'SCADA',
        range: [0, (m.codes.length * 2.05)],
        digits: 2,
      },
      {
        label: `${selCode} wake loss (${r.online[sel] ? (r.is_free[sel] ? 'front row' : 'waked') : 'parked'})`,
        unit: '%',
        value: r.online[sel] ? selLossPct : 0,
        lo: r.online[sel] ? selLossPct - selSigPct : undefined,
        hi: r.online[sel] ? selLossPct + selSigPct : undefined,
        truth: !mod && baseOn[sel] ? selTrueLossPct : null,
        truthLabel: 'SCADA',
        range: [-25, 75],
        digits: 1,
      },
      {
        label: `${selCode} active power`,
        unit: 'kW',
        value: r.power_kw[sel],
        lo: r.online[sel] ? Math.max(0, r.power_kw[sel] - r.power_sig_kw[sel]) : undefined,
        hi: r.online[sel] ? r.power_kw[sel] + r.power_sig_kw[sel] : undefined,
        truth: !mod ? pTrue[sel] : null,
        truthLabel: 'SCADA',
        range: [0, 2100],
        digits: 0,
      },
    ]);

    const hint = h('div', { style: 'font-family:var(--mono);font-size:10px;color:var(--scr-3ink);line-height:1.45;margin-top:2px' });
    hint.innerHTML = mod
      ? `counterfactual control active · recorded SCADA tick hidden · click any turbine to inspect`
      : `anchor (FLORIS): <b>${(100 * (1 - r.eta_anchor[sel])).toFixed(1)}%</b> on ${selCode} · click any rotor on map to inspect (double-click to park)`;
    ui.readout.append(hint);
  }

  function draw(c: Canvas2D) {
    const { ctx: g, w, h: H } = c;
    g.fillStyle = SCOPE.bg;
    g.fillRect(0, 0, w, H);
    const r = view.r;
    if (!r) return;

    const i = st.i;
    const m = ex[i];
    const east = X.t[`east_${i}`].data;
    const north = X.t[`north_${i}`].data;
    const pTrue = X.t[`ptrue_${i}`].data;
    const turb = W.meta.turbines[m.turbine_model];
    const D = turb.rotor_diameter;
    const [ar, ag, ab] = hex(ctx.accent);

    // Top 66%: 2-D plan-view wake plume field; Bottom 34%: per-turbine wake-loss bar strip
    const mapH = Math.round(H * 0.66);
    const barTop = mapH + 14;
    const barH = H - barTop - 22;
    const pad = 16;
    const mapW = w - 2 * pad;
    const mapInnerH = mapH - 22;

    // Bounding box in metres with aspect-preserving margin so wakes stay undistorted
    let emin = Infinity, emax = -Infinity, nmin = Infinity, nmax = -Infinity;
    for (let k = 0; k < east.length; k++) {
      if (east[k] < emin) emin = east[k];
      if (east[k] > emax) emax = east[k];
      if (north[k] < nmin) nmin = north[k];
      if (north[k] > nmax) nmax = north[k];
    }
    const margin = 3.2 * D;
    const cxM = 0.5 * (emin + emax), cyM = 0.5 * (nmin + nmax);
    const spanE0 = (emax - emin) + 2 * margin;
    const spanN0 = (nmax - nmin) + 2 * margin;
    const mapAspect = mapW / Math.max(mapInnerH, 60);
    let spanE = spanE0, spanN = spanN0;
    if (spanE / spanN > mapAspect) spanN = spanE / mapAspect;
    else spanE = spanN * mapAspect;

    const eLo = cxM - 0.5 * spanE, eHi = cxM + 0.5 * spanE;
    const nLo = cyM - 0.5 * spanN, nHi = cyM + 0.5 * spanN;

    // Render 2-D wake velocity deficit field
    const cols = 132, rows = Math.max(48, Math.round(cols / mapAspect));
    const field = wakePlumeGrid(east, north, turb, view.wd, m.ti_inf, view.yaw, view.online, r.ct_anchor, r.ti_anchor, eLo, eHi, nLo, nHi, cols, rows);
    const img = g.createImageData(cols, rows);
    for (let rr = 0; rr < rows; rr++) {
      for (let qq = 0; qq < cols; qq++) {
        const def = field[rr * cols + qq];
        const u = Math.min(1, def / 0.48);
        const gam = Math.pow(u, 0.85);
        const px = ((rows - 1 - rr) * cols + qq) * 4;
        img.data[px + 0] = 13 + (ar - 13) * gam * 0.85 + (245 - ar) * Math.max(0, gam - 0.78) * 2.2;
        img.data[px + 1] = 15 + (ag - 15) * gam * 0.85 + (240 - ag) * Math.max(0, gam - 0.78) * 2.2;
        img.data[px + 2] = 12 + (ab - 12) * gam * 0.85 + (225 - ab) * Math.max(0, gam - 0.78) * 2.2;
        img.data[px + 3] = 255;
      }
    }
    const off = document.createElement('canvas');
    off.width = cols; off.height = rows;
    off.getContext('2d')!.putImageData(img, 0, 0);
    g.imageSmoothingEnabled = true;
    g.drawImage(off, pad, 16, mapW, mapInnerH);

    g.strokeStyle = SCOPE.grid;
    g.strokeRect(pad + 0.5, 16.5, mapW - 1, mapInnerH - 1);

    const sx = (e: number) => pad + ((e - eLo) / spanE) * mapW;
    const sy = (n: number) => 16 + (1 - (n - nLo) / spanN) * mapInnerH;
    const pxPerM = mapW / spanE;

    // Incoming wind arrow in top-left corner of map
    const wx = pad + 34, wy = 46, arLen = 18;
    const th = (view.wd * Math.PI) / 180;
    const dxW = -Math.sin(th), dyW = Math.cos(th); // screen y is inverted North!
    g.strokeStyle = SCOPE.ink;
    g.lineWidth = 1.8;
    g.beginPath();
    g.moveTo(wx - dxW * arLen, wy - dyW * arLen);
    g.lineTo(wx + dxW * arLen, wy + dyW * arLen);
    g.stroke();
    // Arrowhead
    const ang = Math.atan2(dyW, dxW);
    g.fillStyle = SCOPE.ink;
    g.beginPath();
    g.moveTo(wx + dxW * arLen, wy + dyW * arLen);
    g.lineTo(wx + dxW * arLen - 7 * Math.cos(ang - 0.45), wy + dyW * arLen - 7 * Math.sin(ang - 0.45));
    g.lineTo(wx + dxW * arLen - 7 * Math.cos(ang + 0.45), wy + dyW * arLen - 7 * Math.sin(ang + 0.45));
    g.closePath();
    g.fill();
    label(c, `${m.u_inf_ms.toFixed(1)} m/s · ${Math.round(view.wd)}°`, wx + 24, wy - 4, SCOPE.ink, 'left', 9.5);
    label(c, `TI ${(100 * m.ti_inf).toFixed(0)}% · 5D = ${Math.round(5 * D)} m`, wx + 24, wy + 9, SCOPE.faint, 'left', 8.5);

    // 5D metric scale bar in bottom-right of map
    const barPx = 5 * D * pxPerM;
    const sbx = pad + mapW - 12 - barPx, sby = 16 + mapInnerH - 10;
    g.strokeStyle = SCOPE.text;
    g.lineWidth = 1.5;
    g.beginPath();
    g.moveTo(sbx, sby); g.lineTo(sbx + barPx, sby);
    g.moveTo(sbx, sby - 3); g.lineTo(sbx, sby + 3);
    g.moveTo(sbx + barPx, sby - 3); g.lineTo(sbx + barPx, sby + 3);
    g.stroke();
    label(c, `5D (${Math.round(5 * D)} m)`, sbx + barPx / 2, sby - 8, SCOPE.text, 'center', 8.5);

    // Draw turbines as oriented rotor disks + hubs
    hitTargets = [];
    const rHalfPx = Math.max(6, Math.min(14, 0.5 * D * pxPerM * 1.4));
    for (let k = 0; k < east.length; k++) {
      const xk = sx(east[k]), yk = sy(north[k]);
      hitTargets.push({ idx: k, x: xk, y: yk });
      const nacDeg = view.wd - view.yaw[k]; // compass bearing nacelle faces into
      const nr = (nacDeg * Math.PI) / 180;
      // Rotor plane is perpendicular to nacelle bearing (sin(nr), -cos(nr)) in screen coords -> tangent (cos(nr), sin(nr))
      const rx = Math.cos(nr) * rHalfPx, ry = Math.sin(nr) * rHalfPx;

      if (k === st.selected) {
        g.strokeStyle = '#ebe6d8';
        g.lineWidth = 1.2;
        g.setLineDash([2, 2]);
        g.beginPath();
        g.arc(xk, yk, rHalfPx + 5, 0, Math.PI * 2);
        g.stroke();
        g.setLineDash([]);
      }

      if (!view.online[k]) {
        g.strokeStyle = SCOPE.faint;
        g.lineWidth = 1.5;
        g.beginPath();
        g.arc(xk, yk, 4.5, 0, Math.PI * 2);
        g.stroke();
      } else {
        g.strokeStyle = r.is_free[k] ? '#ebe6d8' : ctx.accent;
        g.lineWidth = 2.6;
        g.beginPath();
        g.moveTo(xk - rx, yk - ry);
        g.lineTo(xk + rx, yk + ry);
        g.stroke();

        g.fillStyle = r.is_free[k] ? '#ebe6d8' : ctx.accent;
        g.beginPath();
        g.arc(xk, yk, 2.8, 0, Math.PI * 2);
        g.fill();
      }
      label(c, m.codes[k], xk, yk - rHalfPx - 6, k === st.selected ? SCOPE.ink : SCOPE.text, 'center', 8.5);
    }

    label(c, `${m.farm.toUpperCase()} (${m.codes.length}× Senvion ${m.turbine_model}, D=${D}m) — white rotors: unwaked front row · accent: waked`, pad, 8, SCOPE.faint, 'left', 9);

    // Bottom strip: per-turbine power ratio eta_i (0 to 1.25) with 68% error bars & SCADA truth ticks
    const nWt = m.codes.length;
    const mod = isModified();
    const yEta = (eta: number) => barTop + (1 - Math.max(0, Math.min(1.25, eta)) / 1.25) * barH;

    // 100% freestream reference dashed line
    const y100 = yEta(1.0);
    g.strokeStyle = 'rgba(235,230,216,0.22)';
    g.lineWidth = 1;
    g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(pad + 26, y100); g.lineTo(w - pad, y100); g.stroke();
    g.setLineDash([]);
    label(c, '100%', pad + 22, y100, SCOPE.faint, 'right', 8.5);
    label(c, '50%', pad + 22, yEta(0.5), SCOPE.faint, 'right', 8.5);
    label(c, 'P/P₀', pad + 22, barTop + 4, SCOPE.text, 'right', 8.5);

    const slotW = (w - 2 * pad - 30) / nWt;
    for (let k = 0; k < nWt; k++) {
      const cx = pad + 30 + (k + 0.5) * slotW;
      const bw = Math.min(22, slotW * 0.56);
      const yBase = yEta(0);

      if (view.online[k]) {
        const yVal = yEta(r.eta[k]);
        g.fillStyle = r.is_free[k] ? 'rgba(235,230,216,0.32)' : `rgba(${ar},${ag},${ab},0.55)`;
        if (k === st.selected) g.fillStyle = ctx.accent;
        g.fillRect(cx - bw / 2, yVal, bw, Math.max(1, yBase - yVal));

        // 68% error bar whisker
        const yLo = yEta(r.eta[k] - r.eta_sig[k]);
        const yHi = yEta(r.eta[k] + r.eta_sig[k]);
        g.strokeStyle = 'rgba(235,230,216,0.65)';
        g.lineWidth = 1.2;
        g.beginPath();
        g.moveTo(cx, yHi); g.lineTo(cx, yLo);
        g.stroke();

        // SCADA truth horizontal tick (when unperturbed)
        if (!mod) {
          const etaT = pTrue[k] / m.p_front_kw;
          const yT = yEta(etaT);
          g.strokeStyle = '#ebe6d8';
          g.lineWidth = 2;
          g.beginPath();
          g.moveTo(cx - bw / 2 - 2, yT);
          g.lineTo(cx + bw / 2 + 2, yT);
          g.stroke();
        }
      }
      label(c, m.codes[k], cx, H - 9, k === st.selected ? SCOPE.ink : SCOPE.faint, 'center', nWt > 10 ? 8 : 9);
    }
  }

  await run();
}
