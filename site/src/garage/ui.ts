// The garage: pick a vehicle and a real route; RoadSim's run of it (precomputed in Python, physics v3.1) is shown with its
// energy ledger, and the three networks run in the browser on every change: EnergyNet forecasts the trip before it is
// driven, TripDetective reads the run's OBD log back, HybridPilot's network is re-run over every decision second of its
// own drive. Truth (RoadSim's run and the vehicle spec) sits next to every reading.
import { fetchPack, type Pack } from '../lib/nn/pack.ts';
import type { EMeta, Prediction } from '../instruments/energynet/net.ts';
import type { TMeta } from '../instruments/tripdetective/net.ts';
import { loadNet, type Net, type WMeta } from '../instruments/hybridpilot/net.ts';
import { h, fmt, canvas, gauges, probs, label, scheduler, SCOPE, type Canvas2D } from '../instruments/kit.ts';
import { cellTrip, enRun, enVehicle, hpInputs, hpRun, tag, tdRaw, tdRead, unbits, type Cell, type GMeta, type HPTrace, type TDResult, type VehInfo, type VMeta } from './data.ts';
import { COMMON, NOTES } from './notes.ts';

// ---------------------------------------------------------------------------------------------- look
const C = {
  en: '#7cb518', td: '#c78d5f', hp: '#8794ea', sim: '#e8a33d', ink: SCOPE.ink,
  wheel: '#e8a33d', fuel: '#ff7f50', batt: '#45c3cd', hvac: '#b79be0', soc: '#45c3cd', on: '#f2b84b', elev: '#8fbf6a',
};
const MODEL_ACCENT: Record<string, string> = { EnergyNet: '#7cb518', TripDetective: '#9c6644', HybridPilot: '#3949ab', RoadSim: '#c98a1e' };
// the energy ledger's categories (RoadSim ledger keys, kJ)
const LED: { k: string; name: string; color: string }[] = [
  { k: 'aero', name: 'Air drag', color: '#5ec8d8' },
  { k: 'rolling', name: 'Rolling', color: '#c9c3b2' },
  { k: 'grade', name: 'Climbing (net)', color: '#8fbf6a' },
  { k: 'brake', name: 'Friction brakes', color: '#e07a5f' },
  { k: 'eng_loss', name: 'Engine losses', color: '#f2b84b' },
  { k: 'elec_loss', name: 'Motor + battery losses', color: '#4f8fe0' },
  { k: 'trans_loss', name: 'Transmission', color: '#a6895c' },
  { k: 'hvac', name: 'Heating / A/C', color: '#b79be0' },
  { k: 'other_aux', name: 'Other accessories', color: '#8a84a3' },
  { k: 'kinetic', name: 'Speed at the end', color: '#5d6058' },
];
const EN_COMP: [string, string][] = [['Air drag', '#5ec8d8'], ['Rolling', '#c9c3b2'], ['Friction brakes', '#e07a5f'], ['Accessories, HVAC', '#b79be0'], ['Powertrain losses', '#f2b84b']];
const TD_LED: [string, string][] = [['Air drag', '#5ec8d8'], ['Rolling', '#c9c3b2'], ['Climbing (net)', '#8fbf6a'], ['Friction brakes', '#e07a5f'], ['Powertrain', '#f2b84b'], ['Accessories', '#b79be0']];
const ANAME: Record<string, string> = {
  conv: 'conventional', hev_powersplit: 'power-split hybrid', hev_parallel_p2: 'parallel (P2) hybrid', hev_series: 'series hybrid',
  phev_powersplit: 'power-split plug-in', phev_series: 'series plug-in', bev: 'battery electric',
};
const SNAME: Record<string, string> = {
  conv_rule: 'shift rule', aecms: 'adaptive ECMS', phev_cdcs: 'deplete, then sustain', bev_regen_series: 'regeneration first',
  powersplit_rule: 'engine-on threshold rule', parallel_rule: 'electric-assist rule',
};
const RULE: Record<string, string> = { hev_powersplit: 'powersplit_rule', hev_parallel_p2: 'parallel_rule', phev_series: 'phev_cdcs', phev_powersplit: 'phev_cdcs' };

const n0 = (v: number) => Math.round(v).toLocaleString('en-US').replace('-', '−');
const sgn = (v: number, d = 1) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(d)}`;
const mmss = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const engineOf = (a: string) => a !== 'bev';
const electric = (a: string) => a !== 'conv';

/** Energy (MJ) in the unit people use for this powertrain */
function unitOf(arch: string) {
  if (arch === 'bev') return { u: 'kWh', f: 1 / 3.6, d: 2, per: 'kWh/100 km' };
  if (arch.startsWith('phev')) return { u: 'MJ', f: 1, d: 1, per: 'MJ/100 km' };
  return { u: 'L petrol', f: 1 / 0.042 / 745, d: 2, per: 'L/100 km' };
}

// ---------------------------------------------------------------------------------------------- plotting
interface Series { y: ArrayLike<number>; color: string; kind?: 'line' | 'area' | 'bits'; width?: number; alpha?: number; idx?: ArrayLike<number>; dash?: number[] }
interface Strip { label: string; unit?: string; lo: number; hi: number; series: Series[]; h: number; zero?: boolean; fmt?: (v: number) => string }
interface Frame { x0: number; w: number; n: number }
const PAD = { l: 8, r: 46, t: 6, b: 18 };

function drawStrips(c: Canvas2D, n: number, strips: Strip[], cursor: number | null): Frame {
  const { ctx } = c;
  ctx.clearRect(0, 0, c.w, c.h);
  const x0 = PAD.l, w = c.w - PAD.l - PAD.r, H = c.h - PAD.t - PAD.b, gap = 7;
  const tot = strips.reduce((s, q) => s + q.h, 0);
  const X = (i: number) => x0 + (i / Math.max(1, n - 1)) * w;
  let y = PAD.t;
  for (const s of strips) {
    const sh = ((H - gap * (strips.length - 1)) * s.h) / tot;
    const Y = (v: number) => y + sh - ((Math.max(s.lo, Math.min(s.hi, v)) - s.lo) / (s.hi - s.lo || 1)) * sh;
    ctx.strokeStyle = SCOPE.grid; ctx.lineWidth = 1; ctx.beginPath();
    for (const f of [0, 0.5, 1]) { const yy = Math.round(y + sh * f) + 0.5; ctx.moveTo(x0, yy); ctx.lineTo(x0 + w, yy); }
    ctx.stroke();
    if (s.zero && s.lo < 0 && s.hi > 0) { ctx.strokeStyle = SCOPE.faint; ctx.beginPath(); ctx.moveTo(x0, Math.round(Y(0)) + 0.5); ctx.lineTo(x0 + w, Math.round(Y(0)) + 0.5); ctx.stroke(); }
    for (const se of s.series) {
      ctx.globalAlpha = se.alpha ?? 1;
      if (se.kind === 'bits') {
        ctx.fillStyle = se.color;
        let i = 0;
        while (i < n) {
          if (!se.y[i]) { i++; continue; }
          let j = i; while (j < n && se.y[j]) j++;
          ctx.fillRect(X(i), y + 2, Math.max(1, X(j) - X(i)), sh - 4);
          i = j;
        }
      } else if (se.idx) { // sparse: values at the given seconds, broken at gaps
        ctx.strokeStyle = se.color; ctx.lineWidth = se.width ?? 1.3; ctx.setLineDash(se.dash ?? []); ctx.beginPath();
        let prev = -9;
        for (let q = 0; q < se.y.length; q++) {
          const k = se.idx[q], px = X(k), py = Y(se.y[q]);
          if (k - prev > 3) ctx.moveTo(px, py); else ctx.lineTo(px, py);
          prev = k;
        }
        ctx.stroke(); ctx.setLineDash([]);
      } else {
        const cols = Math.max(1, Math.floor(w)), N = se.y.length;
        const mean: number[] = [], lo: number[] = [], hi: number[] = [];
        for (let px = 0; px < cols; px++) {
          const a = Math.floor((px / cols) * N), b = Math.max(a + 1, Math.floor(((px + 1) / cols) * N));
          let s0 = 0, m = 0, mn = Infinity, mx = -Infinity;
          for (let i = a; i < b && i < N; i++) { const v = se.y[i]; if (!Number.isFinite(v)) continue; s0 += v; m++; if (v < mn) mn = v; if (v > mx) mx = v; }
          mean.push(m ? s0 / m : NaN); lo.push(mn); hi.push(mx);
        }
        if (se.kind === 'area') {
          ctx.fillStyle = se.color; ctx.globalAlpha = (se.alpha ?? 1) * 0.22; ctx.beginPath();
          const z = Y(Math.max(s.lo, 0));
          ctx.moveTo(x0, z);
          mean.forEach((v, px) => ctx.lineTo(x0 + px + 0.5, Number.isFinite(v) ? Y(hi[px]) : z));
          ctx.lineTo(x0 + cols, z); ctx.closePath(); ctx.fill();
          ctx.globalAlpha = se.alpha ?? 1;
        }
        ctx.strokeStyle = se.color; ctx.lineWidth = se.width ?? 1.2; ctx.setLineDash(se.dash ?? []); ctx.beginPath();
        let on = false;
        mean.forEach((v, px) => { if (!Number.isFinite(v)) { on = false; return; } const yy = Y(v); if (on) ctx.lineTo(x0 + px + 0.5, yy); else ctx.moveTo(x0 + px + 0.5, yy); on = true; });
        ctx.stroke(); ctx.setLineDash([]);
        if (N > cols * 1.5) { // min/max whiskers where several samples share a pixel
          ctx.globalAlpha = (se.alpha ?? 1) * 0.45; ctx.lineWidth = 1; ctx.beginPath();
          for (let px = 0; px < cols; px++) if (Number.isFinite(lo[px]) && hi[px] - lo[px] > 0) { ctx.moveTo(x0 + px + 0.5, Y(hi[px])); ctx.lineTo(x0 + px + 0.5, Y(lo[px])); }
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;
    }
    ctx.font = '9.5px "Martian Mono Variable", ui-monospace, monospace';
    ctx.fillStyle = 'rgba(13,15,12,0.78)';
    ctx.fillRect(x0 + 1, y + 1, ctx.measureText(s.label).width + 7, 14);
    label(c, s.label, x0 + 4, y + 8, SCOPE.text, 'left', 9.5);
    const f = s.fmt ?? ((v: number) => fmt(v, Math.abs(s.hi - s.lo) < 5 ? 1 : 0));
    if (s.series[0]?.kind !== 'bits') {
      label(c, f(s.hi), x0 + w + 6, y + 5, SCOPE.faint, 'left', 9);
      label(c, f(s.lo), x0 + w + 6, y + sh - 4, SCOPE.faint, 'left', 9);
    }
    if (s.unit) label(c, s.unit, x0 + w + 6, y + sh / 2, SCOPE.faint, 'left', 9);
    y += sh + gap;
  }
  // time axis
  const dur = n / 60, step = [1, 2, 5, 10, 15, 30].find((q) => dur / q <= 7) ?? 60;
  for (let m = 0; m <= dur; m += step) label(c, m === 0 ? '0' : `${m} min`, X(m * 60), c.h - 7, SCOPE.faint, m === 0 ? 'left' : 'center', 9);
  if (cursor != null) {
    const cx = Math.round(X(cursor)) + 0.5;
    ctx.strokeStyle = SCOPE.ink; ctx.globalAlpha = 0.55; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx, PAD.t); ctx.lineTo(cx, c.h - PAD.b); ctx.stroke(); ctx.globalAlpha = 1;
  }
  return { x0, w, n };
}

function hookCursor(el: HTMLElement, frame: () => Frame | null, set: (i: number | null) => void) {
  const at = (ev: PointerEvent) => {
    const f = frame(); if (!f) return;
    const r = el.getBoundingClientRect(), x = ev.clientX - r.left;
    set(Math.max(0, Math.min(f.n - 1, Math.round(((x - f.x0) / f.w) * (f.n - 1)))));
  };
  el.addEventListener('pointermove', at);
  el.addEventListener('pointerdown', at);
  el.addEventListener('pointerleave', (ev) => { if (ev.pointerType === 'mouse') set(null); });
}

function mix(a: string, b: string, t: number) {
  const p = (s: string) => [1, 3, 5].map((i) => parseInt(s.slice(i, i + 2), 16));
  const A = p(a), B = p(b);
  return `rgb(${A.map((x, i) => Math.round(x + (B[i] - x) * t)).join(',')})`;
}

// ---------------------------------------------------------------------------------------------- small DOM pieces
function seg(parent: HTMLElement, title: string, items: { label: string; title?: string }[], on: (i: number) => void, hint?: HTMLElement) {
  const row = h('div', { role: 'group', 'aria-label': title });
  const btns = items.map((it, i) => {
    const b = h('button', { type: 'button', 'aria-pressed': String(i === 0) }, it.label);
    if (it.title) b.title = it.title;
    b.addEventListener('click', () => { if (b.disabled) return; set(i); on(i); });
    row.append(b);
    return b;
  });
  const set = (i: number) => btns.forEach((b, j) => b.setAttribute('aria-pressed', String(i === j)));
  parent.append(h('div', { class: 'gar-seg' }, h('span', {}, title), row, ...(hint ? [hint] : [])));
  return { set, btns };
}

function panel(cls: string, accent: string, no: string, title: string, sub: string, tagText: string, live: boolean) {
  const root = h('section', { class: `inst ready gar-p ${cls}`, style: `--c:${accent}; --accent:${accent}` });
  const head = h('div', { class: 'inst-head' },
    ...(live ? [h('span', { class: 'inst-live' }, h('i'), 'live')] : []),
    h('span', { class: 'gar-no' }, no), h('span', { class: 'inst-title' }, title), h('span', { class: 'inst-sub' }, sub),
    h('span', { class: `gar-tag${live ? ' live' : ''}` }, tagText));
  root.append(head);
  return root;
}

function pairs(el: HTMLElement, rows: { name: string; color: string; m: number; sd?: number; truth: number; unit: string; d?: number }[]) {
  el.innerHTML = '';
  const all = rows.flatMap((r) => [r.m - (r.sd ?? 0), r.m + (r.sd ?? 0), r.truth]);
  const lo = Math.min(0, ...all), hi = Math.max(...all) * 1.04 || 1;
  const P = (v: number) => `${((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * 100}%`;
  for (const r of rows) {
    const z = P(0), a = r.m >= 0 ? z : P(r.m), wv = Math.abs(((r.m - lo) / (hi - lo)) - ((0 - lo) / (hi - lo))) * 100;
    const row = h('div', { class: 'gar-pair', style: `--k:${r.color}` });
    row.innerHTML = `<span>${r.name}</span><span class="t"><span class="m" style="left:${a};width:${wv}%"></span>`
      + (r.sd ? `<span class="sd" style="left:${P(r.m - r.sd)};width:${(2 * r.sd / (hi - lo)) * 100}%"></span>` : '')
      + `<span class="tr" style="left:${P(r.truth)}"></span></span>`
      + `<span class="v">${fmt(r.m, r.d ?? 0)} <small>/ ${fmt(r.truth, r.d ?? 0)} ${r.unit}</small></span>`;
    el.append(row);
  }
}

// ---------------------------------------------------------------------------------------------- mount
export async function mountGarage(root: HTMLElement, opts: { base: string; inst: string }) {
  const msg = root.querySelector<HTMLElement>('.msg');
  const say = (s: string) => { if (msg) msg.textContent = s; };
  say('loading the routes and the three networks…');
  const [R, WE, WT, HW] = await Promise.all([
    fetchPack<GMeta>(`${opts.base}routes`), fetchPack<EMeta>(`${opts.inst}energynet/weights`),
    fetchPack<TMeta>(`${opts.inst}tripdetective/weights`), fetchPack<WMeta>(`${opts.inst}hybridpilot/weights`),
  ]);
  const G = R.meta, hpNet: Net = loadNet(HW);
  const packs = new Map<string, Promise<Pack<VMeta>>>();
  let WB: Promise<Pack<TMeta>> | null = null;
  const vpack = (id: string) => { if (!packs.has(id)) packs.set(id, fetchPack<VMeta>(`${opts.base}v_${id}`)); return packs.get(id)!; };

  const st = { vi: 1, ri: 0, ti: 1, li: 0, pool: false, cursor: null as number | null };
  root.innerHTML = '';
  root.className = 'gar';

  // ---- console
  const con = h('section', { class: 'inst ready gar-console', style: `--c:${C.sim}; --accent:${C.sim}` });
  con.append(h('div', { class: 'inst-head' }, h('span', { class: 'inst-live' }, h('i'), 'garage'),
    h('span', { class: 'inst-title' }, 'Pick a vehicle and a real route'),
    h('span', { class: 'inst-sub' }, `RoadSim ${G.physics.replace('roadsim-physics-', 'physics ')} · ${G.vehicles.reduce((s, v) => s + v.routes.length, 0) * G.temps.length * 2} precomputed runs · three networks live`)));
  const vrow = h('div', { class: 'inst-picker gar-pick-veh', role: 'group', 'aria-label': 'Vehicle' });
  const rrow = h('div', { class: 'inst-picker gar-pick-route', role: 'group', 'aria-label': 'Route' });
  con.append(h('div', { class: 'gar-row' }, h('span', { class: 'ig-l' }, 'Vehicle'), vrow), h('div', { class: 'gar-row' }, h('span', { class: 'ig-l' }, 'Route'), rrow));
  const vbtn = G.vehicles.map((v, i) => {
    const b = h('button', { type: 'button', class: 'inst-pick' }, h('span', {}, v.name), h('small', {}, v.type));
    b.addEventListener('click', () => pickVehicle(i));
    vrow.append(b);
    return b;
  });
  const rbtn = G.routes.map((r, i) => {
    const lab = r.kind === 'ztbus' ? `Zurich, line ${r.line}` : `Ann Arbor, ${r.km.toFixed(1)} km`;
    const sub = r.kind === 'ztbus' ? `${r.km.toFixed(1)} km · ${Math.round(r.minutes)} min · ${Math.round(r.passengers ?? 0)} riders` : `${Math.round(r.minutes)} min · up to ${Math.round(r.vmax_kmh)} km/h`;
    const b = h('button', { type: 'button', class: 'inst-pick' }, h('span', {}, lab), h('small', {}, sub));
    b.addEventListener('click', () => { if (b.disabled) return; st.ri = i; st.cursor = null; sync(); go(); });
    rrow.append(b);
    return b;
  });
  const ctl = h('div', { class: 'gar-ctl' });
  con.append(ctl);
  const tHint = h('small', {}, '');
  const sT = seg(ctl, 'Air temperature (route mean)', G.temps.map((T) => ({ label: `${T > 0 ? '' : T < 0 ? '−' : ''}${Math.abs(T)} °C`, title: 'every second of the airport record shifted; air density follows' })), (i) => { st.ti = i; go(); }, tHint);
  const sL = seg(ctl, 'Load', [{ label: 'driver only' }, { label: '+400 kg' }], (i) => { st.li = i; go(); });
  const sP = seg(ctl, 'TripDetective reads', [{ label: 'this trip' }, { label: 'all its trips' }], (i) => { st.pool = i === 1; go(); });
  const sum = h('p', { class: 'gar-sum' });
  const status = h('div', { class: 'inst-status' }, 'loading…');
  con.append(sum, status);
  root.append(con);

  // ---- panels
  const grid = h('div', { class: 'gar-grid' });
  root.append(grid);

  // 1. route
  const pR = panel('', C.ink, '01', 'The route', 'real speed, grade and weather, 1 Hz', 'real data', false);
  const mapBox = h('div', { class: 'gar-scr', 'aria-label': 'GPS track of the route' });
  const profBox = h('div', { class: 'gar-scr', 'aria-label': 'Speed and elevation along the route' });
  const rKv = h('div', { class: 'gar-kv' });
  pR.append(h('div', { class: 'gar-body' }, h('div', { class: 'gar-col' }, mapBox), h('div', { class: 'gar-col' }, rKv)), profBox);
  grid.append(pR);

  // 2. EnergyNet
  const pE = panel('', C.en, '02', 'EnergyNet', 'the forecast, before the trip', 'live', true);
  const eBig = h('div', { class: 'gar-col' });
  const eG = h('div', {});
  const eComp = h('div', { class: 'gar-pairs' });
  const eTold = h('p', { class: 'gar-note' });
  pE.append(h('div', { class: 'gar-col' }, eBig, eG, h('span', { class: 'ig-l' }, 'Where it expects the energy to go (MJ)'), eComp,
    h('div', { class: 'gar-key' }, h('span', {}, h('i', { class: 'm' }), 'EnergyNet'), h('span', {}, h('i', { class: 'tr' }), 'RoadSim’s ledger')), eTold));
  const eStat = h('div', { class: 'inst-status' });
  pE.append(eStat);
  grid.append(pE);

  // 3. RoadSim
  const pS = panel('wide', C.sim, '03', 'RoadSim', 'the run: physics v3.1, second by second', 'precomputed', false);
  const sRead = h('div', { class: 'gar-read' }, 'hover or touch the traces');
  const simBox = h('div', { class: 'gar-scr', 'aria-label': 'Speed, power, charge and engine state over the trip' });
  const sKv = h('div', { class: 'gar-kv' });
  const sLed = h('div', { class: 'gar-led' });
  pS.append(h('div', { class: 'gar-body' }, h('div', { class: 'gar-col' }, sRead, simBox), h('div', { class: 'gar-col' }, sKv)),
    h('div', { class: 'gar-col', style: 'margin-top:14px' }, h('span', { class: 'ig-l' }, 'Where the energy went'), sLed));
  grid.append(pS);

  // 4. TripDetective
  const pT = panel('', C.td, '04', 'TripDetective', 'reading the run’s OBD log back', 'live', true);
  const tCh = h('div', { class: 'gar-chips' });
  const tProbs = h('div', {}), tG = h('div', { class: 'gar-col' }), tLed = h('div', { class: 'gar-pairs' }), tNote = h('p', { class: 'gar-note' });
  const tLedBox = h('div', { class: 'gar-col' }, h('span', { class: 'ig-l' }, 'This trip’s ledger (kJ/km): reading ± 1σ against RoadSim'), tLed,
    h('div', { class: 'gar-key' }, h('span', {}, h('i', { class: 'm' }), 'TripDetective'), h('span', {}, h('i', { class: 'tr' }), 'RoadSim')));
  pT.append(h('div', { class: 'gar-col' }, h('span', { class: 'ig-l' }, 'What the logger recorded'), tCh, tProbs, tG, tLedBox, tNote));
  const tStat = h('div', { class: 'inst-status' });
  pT.append(tStat);
  grid.append(pT);

  // 5. HybridPilot
  const pH = panel('', C.hp, '05', 'HybridPilot', 'driving the same trip', 'live network', true);
  const hBody = h('div', { class: 'gar-col' });
  const hOff = h('div', { class: 'gar-off' });
  const hTab = h('div', { class: 'gar-scroll' });
  const hRead = h('div', { class: 'gar-read' });
  const hpBox = h('div', { class: 'gar-scr', 'aria-label': 'Engine on and charge for each controller, and the network’s equivalence factor' });
  const hNote = h('p', { class: 'gar-note' });
  hBody.append(hTab, hRead, hpBox, hNote);
  pH.append(hBody, hOff);
  const hStat = h('div', { class: 'inst-status' });
  pH.append(hStat);
  grid.append(pH);

  // notes
  const notes = h('section', { class: 'frame gar-notes' });
  root.append(notes);

  // ---- state for drawing
  let cur: {
    v: VehInfo; P: Pack<VMeta>; cell: Cell; tg: string; trip: ReturnType<typeof cellTrip>; en: Prediction; td: TDResult; tdTrip: number;
    hp: HPTrace | null; ms: { en: number; td: number; hp: number };
  } | null = null;
  // ---- tensors of the current cell
  const T = (k: string) => cur?.P.t[`${cur.tg}_${k}`]?.data;
  const route = (k: string) => R.t[`r${st.ri}_${k}`].data;
  let mapPts: { x: number; y: number }[] = [];
  let fSim: Frame | null = null, fProf: Frame | null = null, fHp: Frame | null = null;
  let raf = 0;
  const setCursor = (i: number | null) => { if (st.cursor === i) return; st.cursor = i; cancelAnimationFrame(raf); raf = requestAnimationFrame(redrawTime); };

  const narrow = root.getBoundingClientRect().width < 640;   // phones: taller plots
  const mapCv = canvas(mapBox, narrow ? 1.15 : 1.35, (c) => drawMap(c));
  const profCv = canvas(profBox, narrow ? 2.3 : 4.2, (c) => { fProf = drawProf(c); });
  const simCv = canvas(simBox, narrow ? 0.95 : 2.0, (c) => { fSim = drawSim(c); });
  const hpCv = canvas(hpBox, narrow ? 0.9 : 1.55, (c) => { fHp = drawHp(c); });
  hookCursor(profBox, () => fProf, setCursor);
  hookCursor(simBox, () => fSim, setCursor);
  hookCursor(hpBox, () => fHp, setCursor);
  const mapAt = (ev: PointerEvent) => {
    if (!mapPts.length) return;
    const r = mapBox.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    let best = -1, bd = 900;
    mapPts.forEach((p, i) => { const d = (p.x - x) ** 2 + (p.y - y) ** 2; if (d < bd) { bd = d; best = i; } });
    setCursor(best >= 0 ? best : null);
  };
  mapBox.addEventListener('pointermove', mapAt);
  mapBox.addEventListener('pointerdown', mapAt);
  mapBox.addEventListener('pointerleave', (ev) => { if (ev.pointerType === 'mouse') setCursor(null); });

  function redrawTime() { mapCv.redraw(); profCv.redraw(); simCv.redraw(); hpCv.redraw(); readouts(); }


  function drawMap(c: Canvas2D) {
    const { ctx } = c;
    ctx.clearRect(0, 0, c.w, c.h);
    const xy = route('xy'), v = route('v'), n = v.length;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < n; i++) { const x = xy[2 * i], y = xy[2 * i + 1]; x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    const pad = 22, s = Math.min((c.w - 2 * pad) / Math.max(x1 - x0, 1), (c.h - 2 * pad) / Math.max(y1 - y0, 1));
    const ox = (c.w - (x1 - x0) * s) / 2, oy = (c.h - (y1 - y0) * s) / 2;
    const P = (i: number) => ({ x: ox + (xy[2 * i] - x0) * s, y: c.h - oy - (xy[2 * i + 1] - y0) * s });
    // a 1 km (or 500 m) grid in the background
    const g = (x1 - x0) > 3000 || (y1 - y0) > 3000 ? 1000 : 500;
    ctx.strokeStyle = SCOPE.grid; ctx.lineWidth = 1; ctx.beginPath();
    for (let gx = Math.ceil(x0 / g) * g; gx <= x1; gx += g) { const px = Math.round(ox + (gx - x0) * s) + 0.5; ctx.moveTo(px, 0); ctx.lineTo(px, c.h); }
    for (let gy = Math.ceil(y0 / g) * g; gy <= y1; gy += g) { const py = Math.round(c.h - oy - (gy - y0) * s) + 0.5; ctx.moveTo(0, py); ctx.lineTo(c.w, py); }
    ctx.stroke();
    mapPts = Array.from({ length: n }, (_, i) => P(i));
    let vmax = 0; for (let i = 0; i < n; i++) vmax = Math.max(vmax, v[i]);
    ctx.lineWidth = 2.4; ctx.lineCap = 'round';
    for (let i = 1; i < n; i++) {
      ctx.strokeStyle = mix('#3a3d36', C.sim, Math.min(1, v[i] / (vmax || 1)) ** 0.8);
      ctx.beginPath(); ctx.moveTo(mapPts[i - 1].x, mapPts[i - 1].y); ctx.lineTo(mapPts[i].x, mapPts[i].y); ctx.stroke();
    }
    const a = mapPts[0], b = mapPts[n - 1];
    ctx.strokeStyle = SCOPE.ink; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(a.x, a.y, 5, 0, 7); ctx.stroke();
    ctx.fillStyle = SCOPE.ink; ctx.fillRect(b.x - 4, b.y - 4, 8, 8);
    label(c, 'start', a.x + 8, a.y - 8, SCOPE.text, 'left', 9);
    // scale bar and north
    const bar = g * s;
    ctx.strokeStyle = SCOPE.text; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(12, c.h - 12); ctx.lineTo(12 + bar, c.h - 12); ctx.moveTo(12, c.h - 15); ctx.lineTo(12, c.h - 9); ctx.moveTo(12 + bar, c.h - 15); ctx.lineTo(12 + bar, c.h - 9); ctx.stroke();
    label(c, g >= 1000 ? '1 km' : '500 m', 16 + bar, c.h - 12, SCOPE.text, 'left', 9);
    label(c, 'N ↑', c.w - 12, 14, SCOPE.text, 'right', 10);
    label(c, 'colour: speed', c.w - 12, c.h - 12, SCOPE.faint, 'right', 9);
    if (st.cursor != null && st.cursor < n) {
      const p = mapPts[st.cursor];
      ctx.fillStyle = C.ink; ctx.shadowColor = C.sim; ctx.shadowBlur = 12; ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, 7); ctx.fill(); ctx.shadowBlur = 0;
    }
  }

  function drawProf(c: Canvas2D): Frame {
    const v = route('v'), el = route('elev'), n = v.length;
    const kmh = Float32Array.from(v, (x) => x * 3.6);
    let emin = Infinity, emax = -Infinity; for (let i = 0; i < n; i++) { emin = Math.min(emin, el[i]); emax = Math.max(emax, el[i]); }
    const mid = (emin + emax) / 2, half = Math.max(10, (emax - emin) / 2 + 2);
    return drawStrips(c, n, [
      { label: 'speed', unit: 'km/h', lo: 0, hi: Math.max(40, Math.ceil(Math.max(...kmh) / 20) * 20), h: 1.4, series: [{ y: kmh, color: C.ink, kind: 'area', alpha: 0.85 }] },
      { label: 'elevation', unit: 'm', lo: mid - half, hi: mid + half, h: 1, series: [{ y: el, color: C.elev }] },
    ], st.cursor);
  }

  function drawSim(c: Canvas2D): Frame | null {
    if (!cur) return null;
    const a = cur.v.arch, n = cur.cell.n;
    const kmh = Float32Array.from(route('v'), (x) => x * 3.6);
    const pw = T('pw')!, pf = T('pf'), pb = T('pb'), hv = T('hv')!, soc = T('soc'), on = T('on');
    let pmax = 10, pmin = 0;
    for (const s of [pw, pf, pb, hv]) if (s) for (let i = 0; i < n; i++) { pmax = Math.max(pmax, s[i]); pmin = Math.min(pmin, s[i]); }
    const strips: Strip[] = [
      { label: 'speed', unit: 'km/h', lo: 0, hi: Math.max(40, Math.ceil(Math.max(...kmh) / 20) * 20), h: 0.9, series: [{ y: kmh, color: C.ink, kind: 'area', alpha: 0.8 }] },
      {
        label: 'power', unit: 'kW', lo: Math.floor(pmin / 10) * 10, hi: Math.ceil(pmax / 10) * 10, h: 1.6, zero: true, series: [
          ...(pf ? [{ y: pf, color: C.fuel, alpha: 0.75 }] : []), ...(pb ? [{ y: pb, color: C.batt }] : []),
          { y: pw, color: C.wheel, width: 1.1 }, { y: hv, color: C.hvac, width: 1.4 },
        ],
      },
    ];
    if (soc) strips.push({ label: 'charge', unit: '%', lo: 0, hi: 100, h: 0.7, series: [{ y: Float32Array.from(soc, (x) => x / 2.5), color: C.soc, width: 1.5 }], fmt: (v) => `${Math.round(v)}` });
    if (on && engineOf(a)) strips.push({ label: 'engine on', lo: 0, hi: 1, h: 0.28, series: [{ y: unbits(on, n), color: C.on, kind: 'bits', alpha: 0.8 }] });
    return drawStrips(c, n, strips, st.cursor);
  }

  function drawHp(c: Canvas2D): Frame | null {
    if (!cur || !cur.hp || !cur.cell.hp) return null;
    const n = cur.cell.n;
    const onS = ['hp', 'ae', 'dp'].map((k) => unbits(T(`on_${k}`)!, n));
    const socS = ['hp', 'ae', 'dp'].map((k) => Float32Array.from(T(`soc_${k}`)!, (x) => x / 2.5));
    let lo = 100, hi = 0;
    for (const s of socS) for (let i = 0; i < n; i++) { lo = Math.min(lo, s[i]); hi = Math.max(hi, s[i]); }
    const pad = Math.max(2, (hi - lo) * 0.15);
    const hp = cur.hp;
    let smax = 0; for (let i = 0; i < hp.s.length; i++) smax = Math.max(smax, hp.s[i], hp.sNet[i]);
    return drawStrips(c, n, [
      { label: 'engine · HybridPilot', lo: 0, hi: 1, h: 0.3, series: [{ y: onS[0], color: C.hp, kind: 'bits' }] },
      { label: 'engine · A-ECMS', lo: 0, hi: 1, h: 0.3, series: [{ y: onS[1], color: '#9a958a', kind: 'bits' }] },
      { label: 'engine · DP optimum', lo: 0, hi: 1, h: 0.3, series: [{ y: onS[2], color: C.ink, kind: 'bits', alpha: 0.75 }] },
      { label: 'charge', unit: '%', lo: Math.max(0, Math.floor(lo - pad)), hi: Math.min(100, Math.ceil(hi + pad)), h: 1, series: [
        { y: socS[2], color: C.ink, width: 1, dash: [3, 3] }, { y: socS[1], color: '#9a958a', width: 1.2 }, { y: socS[0], color: C.hp, width: 1.6 }] },
      { label: 'network: equivalence factor s', lo: 0, hi: Math.ceil(smax + 0.3), h: 1, series: [
        { y: hp.sNet, idx: hp.k, color: C.hp, width: 1, dash: [2, 2], alpha: 0.7 }, { y: hp.s, idx: hp.k, color: C.hp, width: 1.5 }] },
      { label: 'network: P(engine on)', lo: 0, hi: 1, h: 0.55, series: [{ y: hp.pon, idx: hp.k, color: C.on, width: 1.2 }], fmt: (v) => v.toFixed(0) },
    ], st.cursor);
  }

  function readouts() {
    if (!cur) return;
    const i = st.cursor, a = cur.v.arch;
    if (i == null) { sRead.innerHTML = '<span>hover or touch the traces · the map follows</span>'; hRead.innerHTML = '<span>hover or touch the traces</span>'; return; }
    const g = (k: string) => T(k)?.[i];
    const kmh = route('v')[i] * 3.6;
    const parts = [`<span>t <b>${mmss(i)}</b></span>`, `<span><b>${kmh.toFixed(0)}</b> km/h</span>`,
      `<span><i style="background:${C.wheel}"></i>wheels <b>${fmt(g('pw') ?? NaN, 1)}</b> kW</span>`];
    if (g('pf') != null) parts.push(`<span><i style="background:${C.fuel}"></i>fuel <b>${fmt(g('pf')!, 1)}</b> kW</span>`);
    if (g('pb') != null) parts.push(`<span><i style="background:${C.batt}"></i>battery <b>${fmt(g('pb')!, 1)}</b> kW</span>`);
    parts.push(`<span><i style="background:${C.hvac}"></i>HVAC <b>${fmt(g('hv') ?? NaN, 1)}</b> kW</span>`);
    if (g('soc') != null) parts.push(`<span>charge <b>${(g('soc')! / 2.5).toFixed(1)}</b> %</span>`);
    if (T('on') && engineOf(a)) parts.push(`<span>engine <b>${unbits(T('on')!, cur.cell.n)[i] ? 'on' : 'off'}</b></span>`);
    sRead.innerHTML = parts.join('');
    if (cur.hp) {
      const k = cur.hp.k; let j = -1;
      for (let q = 0; q < k.length; q++) if (k[q] <= i) j = q; else break;
      const on = (t: string) => unbits(T(`on_${t}`)!, cur!.cell.n)[i] ? 'on' : 'off';
      hRead.innerHTML = `<span>t <b>${mmss(i)}</b></span><span>engine: HybridPilot <b>${on('hp')}</b>, A-ECMS <b>${on('ae')}</b>, DP <b>${on('dp')}</b></span>`
        + (j >= 0 && k[j] === i ? `<span>s <b>${cur.hp.s[j].toFixed(2)}</b> (network ${cur.hp.sNet[j].toFixed(2)})</span><span>P(on) <b>${cur.hp.pon[j].toFixed(2)}</b></span>` : '<span>no decision this second</span>');
    }
  }

  // ---- selection
  function sync() {
    const v = G.vehicles[st.vi];
    vbtn.forEach((b, i) => b.setAttribute('aria-pressed', String(i === st.vi)));
    rbtn.forEach((b, i) => { const ok = v.routes.includes(i); b.disabled = !ok; b.title = ok ? '' : 'the trolleybus drives its Zurich line only'; b.setAttribute('aria-pressed', String(i === st.ri)); });
    sT.set(st.ti); sL.set(st.li); sP.set(st.pool ? 1 : 0);
    const bus = v.cls === 'bus';
    sL.btns[1].textContent = bus ? '+60 riders' : '+400 kg';
    sL.btns[0].textContent = bus ? `logged riders (${Math.round(G.routes[st.ri].passengers ?? 0)})` : 'driver only';
    sP.btns[1].disabled = v.routes.length < 2;
    sP.btns[1].textContent = v.routes.length < 2 ? 'all its trips' : `all ${v.routes.length} trips`;
    if (v.routes.length < 2) { st.pool = false; sP.set(0); }
    tHint.textContent = `logged ${G.routes[st.ri].T_mean.toFixed(1)} °C here`;
  }
  function pickVehicle(i: number) {
    st.vi = i;
    const v = G.vehicles[i];
    if (!v.routes.includes(st.ri)) st.ri = v.routes[0];
    st.cursor = null;
    sync(); go();
  }

  // ---- compute (the networks) and render
  const go = scheduler(async () => {
    const v = G.vehicles[st.vi];
    status.innerHTML = '<span class="ok">●</span> loading this vehicle’s runs…';
    for (const p of [pE, pT, pH]) p.classList.add('busy');
    const P = await vpack(v.id);
    const bus = v.cls === 'bus';
    if (bus) WB ??= fetchPack<TMeta>(`${opts.base}tdbus`);
    const WTd = bus ? await WB! : WT;
    const tg = tag(st.ri, st.ti, st.li), cell = P.meta.cells[tg];
    const trip = cellTrip(R, st.ri, st.ti);
    let t0 = performance.now();
    const en = enRun(WE, R, v, st.ri, st.ti, st.li);
    const msEn = performance.now() - t0;
    t0 = performance.now();
    const pool = st.pool && !bus ? v.routes : [st.ri];
    const tdIn = pool.map((ri) => tdRaw(G, P, v, ri, st.ti, st.li, ri === st.ri ? trip : cellTrip(R, ri, st.ti)));
    const td = tdRead(WTd, tdIn, bus);
    const msTd = performance.now() - t0;
    t0 = performance.now();
    const hp = cell.hp ? hpRun(hpNet, G, cell, hpInputs(G, P, cell, tg)) : null;
    const msHp = performance.now() - t0;
    cur = { v, P, cell, tg, trip, en, td, tdTrip: pool.indexOf(st.ri), hp, ms: { en: msEn, td: msTd, hp: msHp } };
    for (const p of [pE, pT, pH]) p.classList.remove('busy');
    render();
  }, 30);

  function render() {
    if (!cur) return;
    const { v, cell, en, td, ms } = cur;
    const r = G.routes[st.ri], a = v.arch, U = unitOf(a), load = G.loads[v.cls][st.li];
    const T0 = G.temps[st.ti];
    sum.innerHTML = `<b>${v.name}</b> (${v.type}; ${v.source.split(' (')[0]}) on <b>${r.place}</b>, ${r.km.toFixed(1)} km, at <b>${T0} °C</b>`
      + `${load ? ` with <b>${v.cls === 'bus' ? '+60 riders' : '+400 kg'}</b>` : ''}. Likely real counterpart: ${v.likely}.`;
    status.innerHTML = `<span class="ok">●</span> networks ran in your browser: EnergyNet <b>${ms.en.toFixed(0)} ms</b> · TripDetective <b>${ms.td.toFixed(0)} ms</b>`
      + (cur.hp ? ` · HybridPilot <b>${ms.hp.toFixed(0)} ms</b> over ${cur.hp.k.length} decisions` : '') + ` · RoadSim run read from the grid`;

    // 1. route
    rKv.innerHTML = [
      ['Where', r.kind === 'ztbus' ? `Zurich, line ${r.line}` : 'Ann Arbor, MI'], ['Recorded', r.kind === 'ztbus' ? `${r.date}, ${r.season}` : r.date],
      ['Driven by', r.kind === 'ztbus' ? 'VBZ trolleybus' : `VED car ${(r.vehicle ?? '').replace('VED_', '')}`],
      ['Distance', `${r.km.toFixed(2)} km`], ['Duration', `${mmss(r.n)} min`], ['Mean speed', `${r.v_mean_kmh.toFixed(1)} km/h`],
      ['Top speed', `${r.vmax_kmh.toFixed(0)} km/h`], ['Above 70 km/h', `${Math.round(r.fast_share * 100)} %`], ['Stops', `${r.stops}`],
      ['Climbing', `${n0(r.climb_m)} m`], ['Air', `${T0} °C <small>logged ${r.T_mean.toFixed(1)}</small>`],
      ...(r.kind === 'ztbus' ? [['Riders', `${(r.passengers ?? 0).toFixed(1)} on average`], ['Real energy', `${(r.real_kWh ?? 0).toFixed(1)} kWh <small>logged by the bus</small>`]] : []),
    ].map(([k, x]) => `<span>${k}</span><b>${x}</b>`).join('');

    // 2. EnergyNet
    const truth = cell.E_MJ, err = en.energy_MJ / truth - 1, inside = truth >= en.energy_MJ * Math.exp(-en.sigma_log) && truth <= en.energy_MJ * Math.exp(en.sigma_log);
    const lo = en.energy_MJ * Math.exp(-en.sigma_log), hi = en.energy_MJ * Math.exp(en.sigma_log);
    eBig.innerHTML = `<span class="ig-l">Trip energy, forecast before the trip</span>
      <div class="gar-big"><span class="v">${(en.energy_MJ * U.f).toFixed(U.d)}</span><span class="u">${U.u}</span>
      <span class="e ${inside ? 'in' : 'out'}">${sgn(err * 100)} % vs RoadSim · ${inside ? 'inside' : 'outside'} its 68 % band</span></div>
      <div class="gar-kv"><span>RoadSim used</span><b>${(truth * U.f).toFixed(U.d)} ${U.u} <small>${(truth * U.f / cell.km * 100).toFixed(1)} ${U.per}</small></b>
      <span>Its physics prior</span><b>${(en.physics_estimate_MJ * U.f).toFixed(U.d)} ${U.u} <small>generic road-load estimate it corrects</small></b></div>`;
    gauges(eG, [{ label: 'forecast vs RoadSim', unit: U.u, value: en.energy_MJ * U.f, lo: lo * U.f, hi: hi * U.f, truth: truth * U.f, truthLabel: 'RoadSim', range: [0, Math.max(hi, truth) * U.f * 1.25], digits: U.d }]);
    const L = cell.ledger;
    const rsComp = [L.aero, L.rolling, L.brake, L.aux, L.eng_loss + L.mot_loss + L.trans_loss + L.batt_loss].map((x) => x / 1000);
    pairs(eComp, EN_COMP.map(([name, color], k) => ({ name, color, m: en.components_MJ[k], truth: rsComp[k], unit: 'MJ', d: 2 })));
    const ve = enVehicle(G, v, st.li);
    const told = [`${ve.class}, ${ANAME[ve.architecture]}`, ve.mass_kg ? `${n0(ve.mass_kg)} kg${load ? ' (load included)' : ''}` : 'no mass',
      ve.displacement_L ? `${ve.displacement_L.toFixed(1)} L engine` : '', ve.battery_kWh ? `${ve.battery_kWh} kWh battery` : '',
      ve.label_city ? `window label ${ve.label_city}/${ve.label_hwy} ${a === 'bev' ? 'kWh/100 mi' : 'mpg'}` : '', ve.model_year ? `model year ${Math.round(ve.model_year)}` : '',
      ve.soc_start != null ? `start charge ${Math.round(ve.soc_start * 100)} % of usable` : '', 'warm start'].filter(Boolean);
    eTold.innerHTML = `<b>Told:</b> ${told.join(' · ')}. <b>Not told:</b> the road load, the controller, anything about this run.`
      + (v.cls === 'bus' ? ' Its bus exam gave it only class and powertrain, so the riders you add never reach it.' : '');
    eStat.innerHTML = `<span class="ok">●</span> 5 networks · ${(WE.meta.n_params / 1e6).toFixed(2)} M parameters · ran in <b>${ms.en.toFixed(1)} ms</b>`;

    // 3. RoadSim
    const km = cell.km;
    const eng = engineOf(a);
    const kv: [string, string][] = [['Controller', `${SNAME[cell.strategy] ?? cell.strategy} <small>RoadSim default</small>`]];
    if (eng) kv.push(['Fuel', `${n0(cell.fuel_g)} g <small>${(cell.fuel_g / 745 / km * 100).toFixed(1)} L/100 km</small>`]);
    if (electric(a)) kv.push(['Battery, net', `${cell.batt_kWh.toFixed(2)} kWh <small>${(cell.batt_kWh / km * 100).toFixed(1)} kWh/100 km</small>`]);
    if (cell.soc_end != null && electric(a) && v.soc0 != null) kv.push(['Charge', `${Math.round(v.soc0 * 100)} → ${Math.round(cell.soc_end * 100)} %`]);
    kv.push(['Heating / A/C', `${(cell.hvac_kJ / 3600).toFixed(2)} kWh <small>${Math.round(cell.hvac_kJ / Math.max(1, L.fuel + L.batt_chem_out) * 100)} % of the energy in</small>`]);
    kv.push(['Cabin at the end', `${cell.T_cabin_end.toFixed(1)} °C`]);
    if (eng) kv.push(['Engine on', `${Math.round(cell.on_share * 100)} % of the time <small>${cell.starts} starts</small>`]);
    kv.push(['Total mass', `${n0(cell.mass_true)} kg`], ['Follows the route', `${cell.trace_rms.toFixed(2)} m/s rms`], ['Ledger closes', `${cell.ledger.closure_rel.toExponential(0)}`]);
    sKv.innerHTML = kv.map(([k, x]) => `<span>${k}</span><b>${x}</b>`).join('');
    const hvac = cell.hvac_kJ;
    const sinks: Record<string, number> = {
      aero: L.aero, rolling: L.rolling, grade: Math.max(0, L.grade), brake: L.brake, eng_loss: L.eng_loss, elec_loss: L.mot_loss + L.batt_loss,
      trans_loss: L.trans_loss, hvac, other_aux: Math.max(0, L.aux - hvac), kinetic: Math.max(0, L.kinetic_change),
    };
    const tot = Object.values(sinks).reduce((s, x) => s + x, 0) || 1;
    const shown = LED.filter((l) => sinks[l.k] > 0.002 * tot);
    sLed.innerHTML = `<div class="gar-stack">${shown.map((l) => `<span style="width:${(sinks[l.k] / tot) * 100}%;background:${l.color}" title="${l.name}"></span>`).join('')}</div>
      <div class="gar-lrows">${shown.map((l) => `<div class="gar-lrow"><i style="background:${l.color}"></i><span>${l.name}</span><b>${n0(sinks[l.k] / km)}</b><em>${Math.round((sinks[l.k] / tot) * 100)} %</em></div>`).join('')}</div>
      <p class="gar-note">kJ per km and share of everything spent. In: ${eng ? `fuel <b>${n0(L.fuel / km)}</b> kJ/km` : ''}${eng && electric(a) ? ', ' : ''}${electric(a) ? `battery <b>${n0(L.batt_chem_out / km)}</b> kJ/km` : ''}`
      + `${L.grade < 0 ? `, downhill <b>${n0(-L.grade / km)}</b> kJ/km` : ''}${L.kinetic_change < 0 ? ', speed lost' : ''}. Out and in agree to ${Math.max(cell.ledger.closure_rel, 1e-15).toExponential(0)} (RoadSim checks it on every run).</p>`;

    // 4. TripDetective
    const lg = v.logger;
    const chips = v.cls === 'bus' ? [['speed, grade, weather', true], ['total electric power', true]] as [string, boolean][]
      : ([['speed, grade, weather', true], [lg.fuel_est ? 'fuel rate (MAF estimate)' : 'fuel rate (OBD PID)', lg.has_fuel], ['engine rpm', lg.has_rpm], ['HV battery', lg.has_batt], ['A/C, heater power', lg.has_aux]] as [string, boolean][]);
    tCh.innerHTML = chips.map(([n, on]) => `<span class="${on ? '' : 'off'}">${n}</span>`).join('');
    const tr = v.truth, rd = td.reading;
    if (td.bus) {
      tProbs.innerHTML = '';
      gauges(tG, [{ label: 'Bus total mass', unit: 'kg', value: td.bus.mass, lo: td.bus.lo, hi: td.bus.hi, truth: cell.mass_true, truthLabel: 'RoadSim (19 t + riders)', range: [10000, 40000], digits: 0 }]);
      tLedBox.style.display = 'none';
      tNote.innerHTML = `The bus network reads total electric power as a car twelve times lighter (TripDetectiveBus). ${td.windows} windows of 1,024 s or less.`;
    } else {
      tLedBox.style.display = '';
      probs(tProbs, 'Powertrain', [...Object.keys(ANAME)].map((k) => ({ label: ANAME[k], p: rd.arch.find(([x]) => x === k)?.[1] ?? 0, truth: k === a })));
      const gs = [{ label: 'Total mass', unit: 'kg', value: rd.mass, lo: rd.massLo, hi: rd.massHi, truth: cell.mass_true, truthLabel: 'RoadSim', range: [900, 2700] as [number, number], digits: 0 },
        { label: 'Drag area CdA', unit: 'm²', value: rd.CdA, lo: rd.CdA * Math.exp(-rd.CdAsig), hi: rd.CdA * Math.exp(rd.CdAsig), truth: tr.CdA, truthLabel: 'spec', range: [0.3, 1.2] as [number, number], digits: 2 },
        { label: 'Rolling resistance', unit: '', value: rd.Crr, lo: rd.Crr * Math.exp(-rd.Crrsig), hi: rd.Crr * Math.exp(rd.Crrsig), truth: tr.Crr, truthLabel: 'spec', range: [0.004, 0.016] as [number, number], digits: 4 }];
      if (tr.disp) gs.push({ label: 'Engine size', unit: 'L', value: rd.disp, lo: rd.dispLo, hi: rd.dispHi, truth: tr.disp, truthLabel: 'spec', range: [0.6, 3.6], digits: 2 });
      gauges(tG, gs);
      const led = rd.ledger[Math.max(0, cur.tdTrip)];
      pairs(tLed, TD_LED.map(([name, color], k) => ({ name, color, m: led.mu[k], sd: led.sig[k], truth: cell.td_ledger[k], unit: '' })));
      const dm = rd.mass - cell.mass_true;
      tNote.innerHTML = `Mass ${dm >= 0 ? 'over' : 'under'} by <b>${n0(Math.abs(dm))} kg</b>${st.li ? ' with the extra 400 kg aboard' : ''}. `
        + `Read from ${td.windows} window${td.windows > 1 ? 's' : ''} of ${pool(st.pool, v)} (1,024 s at most each). `
        + `Its ledger was trained on simulated truth only and is graded here against the run that made the log.`;
    }
    tStat.innerHTML = `<span class="ok">●</span> ${td.bus ? 'bus network' : 'CNN + physics-regression head'} · ran in <b>${ms.td.toFixed(0)} ms</b>`;

    // 5. HybridPilot
    if (cell.hp && cur.hp) {
      hBody.style.display = ''; hOff.style.display = 'none';
      const H = cell.hp, U2 = (g: number) => (g / 745 / km * 100).toFixed(2);
      const rows: [string, string, string][] = [['dp', 'DP optimum <small>sees the whole trip</small>', C.ink], ['hybridpilot', 'HybridPilot <small>the network</small>', C.hp],
        ['aecms', 'Adaptive ECMS', '#9a958a'], ['rule', `${SNAME[RULE[a]] ?? 'rule'}`, '#5d6058']];
      hTab.innerHTML = `<table class="gar-race"><thead><tr><th>Controller</th><th>fuel g</th><th>L/100 km</th><th>battery kWh</th><th>% of DP</th><th>starts/km</th></tr></thead><tbody>`
        + rows.map(([k, name, col]) => {
          const x = H[k as keyof typeof H];
          const run = k === (a.startsWith('phev') ? 'rule' : 'aecms') ? ' <small>(the run above)</small>' : '';
          return `<tr class="${k === 'hybridpilot' ? 'me' : ''}"><td><i style="background:${col}"></i>${name}${run}</td><td>${n0(x.fuel_g)}</td><td>${U2(x.fuel_g)}</td><td>${x.E_kWh.toFixed(2)}</td>`
            + `<td>${(x.pct_dp * 100).toFixed(2)}${x.extrap ? '<small>*</small>' : ''}</td><td>${x.starts_km.toFixed(1)}</td></tr>`;
        }).join('') + '</tbody></table>';
      const gain = (H.hybridpilot.pct_dp - H.aecms.pct_dp) * 100;
      hNote.innerHTML = `<b>% of DP</b> is the exam’s grading: fuel against the DP optimum at the same final charge (DP re-solved to each controller’s own end of charge).`
        + ` Here HybridPilot is <b>${sgn(gain, 2)} points</b> against A-ECMS. ${Object.values(H).some((x) => x.extrap) ? '* outside the DP curve (extrapolated). ' : ''}`
        + `${Object.values(H).some((x) => x.pct_dp > 1) ? 'Above 100 % is possible: the DP reference is solved on a 121-point charge grid and interpolated between end charges. ' : ''}`
        + `The network trace is computed live: its own inputs at each of ${cur.hp.k.length} decision seconds, stored in float16 (s within 3e-3 of the run), dashed = before the SOC feedback.`;
      hStat.innerHTML = `<span class="ok">●</span> MLP 66 → 96 → 96 → 2 · ran in <b>${ms.hp.toFixed(1)} ms</b> over the trip`;
    } else {
      hBody.style.display = 'none'; hOff.style.display = '';
      hOff.innerHTML = `<div><b>No hybrid controller to race.</b><br>HybridPilot decides, every second, whether a hybrid’s engine runs and how hard. ${a === 'conv' ? 'This car has only an engine.' : 'This vehicle has no engine.'}<br>Pick the power-split, the P2 or the plug-in.</div>`;
      hStat.innerHTML = '<span class="ok">●</span> idle';
    }

    // notes
    const ns = NOTES[v.id] ?? [];
    notes.innerHTML = `<div><p class="kicker tick-label">Read this with it</p><h3>Where it is weak, for the ${v.short.toLowerCase()}</h3>
      <ul class="common">${COMMON.map((t) => `<li>${t}</li>`).join('')}</ul></div>
      <ul>${ns.map((nn) => `<li><span class="who" style="--k:${MODEL_ACCENT[nn.who]}"><i></i>${nn.who}</span> ${nn.text}</li>`).join('')}</ul>`;

    mapCv.redraw(); profCv.redraw(); simCv.redraw(); hpCv.redraw(); readouts();
  }
  const pool = (p: boolean, v: VehInfo) => (p ? `all ${v.routes.length} trips of this car (same air and load)` : 'this trip only');

  sync();
  go();
}
