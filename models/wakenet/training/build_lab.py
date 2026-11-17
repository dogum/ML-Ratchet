"""Build the self-contained interactive lab page models/wakenet/lab/index.html (< 1 MB).

Embeds the float16 GNN weights, the 6 stratified real exam snapshots, the 3-D Gaussian wake solver,
the live 3-layer EdgeWakeGNN in JavaScript, interactive wind-direction / yaw-steering / meandering /
turbine-parking controls, the 8-rung ratchet ladder, twin-test diagnostics, and the full 24,035-snapshot exam.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(MODEL))

INST = Path(ROOT) / "site" / "public" / "instruments" / "wakenet"
OUT = Path(MODEL) / "lab" / "index.html"


def main():
    w_man = json.loads((INST / "weights.json").read_text(encoding="utf-8"))
    w_b64 = base64.b64encode((INST / "weights.bin").read_bytes()).decode("ascii")
    e_man = json.loads((INST / "examples.json").read_text(encoding="utf-8"))
    e_b64 = base64.b64encode((INST / "examples.bin").read_bytes()).decode("ascii")
    exam = json.loads((Path(HERE) / "exam_results.json").read_text(encoding="utf-8"))
    twin = json.loads((Path(HERE) / "twin_report.json").read_text(encoding="utf-8"))

    payload = json.dumps({
        "wMan": w_man,
        "wB64": w_b64,
        "eMan": e_man,
        "eB64": e_b64,
        "exam": exam,
        "twin": twin,
    })

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>WakeNet v1.0 — Wind-farm wakes read as a directed graph</title>
<style>
  :root {{
    --paper: #f2eee4; --paper-2: #e9e3d6; --paper-3: #fbf9f4;
    --ink: #17150f; --ink-2: #48443a; --ink-3: #807a6b;
    --rule: #d3cbb9; --accent: #0284c7; --win: #1d7a4c; --loss: #c23a2b;
    --mono: ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace;
    --serif: 'Iowan Old Style', 'Palatino Linotype', Georgia, serif;
    --sans: 'Avenir Next', 'Segoe UI', system-ui, sans-serif;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --paper: #0c0d0b; --paper-2: #141613; --paper-3: #1a1c18;
      --ink: #ebe6d8; --ink-2: #b3ac9b; --ink-3: #7c7667;
      --rule: #2c2f29; --accent: #38bdf8; --win: #4cc38a; --loss: #ff6a59;
    }}
  }}
  :root[data-theme="dark"] {{
    --paper: #0c0d0b; --paper-2: #141613; --paper-3: #1a1c18;
    --ink: #ebe6d8; --ink-2: #b3ac9b; --ink-3: #7c7667;
    --rule: #2c2f29; --accent: #38bdf8; --win: #4cc38a; --loss: #ff6a59;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--paper); color: var(--ink); font: 16px/1.55 var(--sans); }}
  .wrap {{ max-width: 1120px; margin: 0 auto; padding: 28px 20px 72px; }}
  .top {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--rule); padding-bottom: 12px; margin-bottom: 28px; font: 11px var(--mono); text-transform: uppercase; letter-spacing: .08em; color: var(--ink-3); }}
  .top button {{ font: 11px var(--mono); background: transparent; color: var(--ink); border: 1px solid var(--rule); padding: 5px 10px; border-radius: 3px; cursor: pointer; }}
  h1 {{ font: 400 clamp(38px, 6vw, 72px)/0.95 var(--serif); margin: 8px 0 16px; letter-spacing: -0.02em; }}
  h2 {{ font: 400 clamp(26px, 3.6vw, 40px)/1.05 var(--serif); margin: 44px 0 12px; }}
  .lead {{ max-width: 68ch; font-size: 18px; color: var(--ink-2); }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin: 24px 0 32px; border-top: 1px solid var(--ink); border-bottom: 1px solid var(--rule); padding: 14px 0; }}
  .stats dt {{ font: 10px var(--mono); text-transform: uppercase; letter-spacing: .08em; color: var(--ink-3); }}
  .stats dd {{ margin: 4px 0 0; font: 26px var(--mono); }}
  .scope {{ background: #0d0f0c; color: #ebe6d8; border: 1px solid #000; border-radius: 6px; padding: 16px; box-shadow: 0 24px 48px -24px rgba(0,0,0,.55); }}
  .picks {{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 12px; }}
  .picks button {{ font: 11px var(--mono); background: transparent; color: rgba(235,230,216,.7); border: 1px solid rgba(235,230,216,.15); border-radius: 4px; padding: 6px 10px; cursor: pointer; text-align: left; }}
  .picks button.on {{ background: #0284c7; border-color: #0284c7; color: #fff; }}
  .sgrid {{ display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(250px, 1fr); gap: 18px; }}
  @media (max-width: 780px) {{ .sgrid {{ grid-template-columns: 1fr; }} }}
  canvas {{ width: 100%; display: block; border: 1px solid rgba(235,230,216,.12); border-radius: 4px; cursor: pointer; }}
  .ctls {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 12px; font-size: 12.5px; color: rgba(235,230,216,.75); }}
  .ctls label {{ display: flex; flex-direction: column; gap: 4px; }}
  .ctls input[type=range] {{ width: 100%; accent-color: #38bdf8; }}
  .readout {{ display: flex; flex-direction: column; gap: 12px; font-family: var(--mono); font-size: 12px; }}
  .card-g {{ padding: 10px 12px; background: rgba(255,255,255,.03); border: 1px solid rgba(235,230,216,.1); border-radius: 4px; }}
  .card-g .k {{ font-size: 10px; text-transform: uppercase; color: rgba(235,230,216,.48); letter-spacing: .06em; }}
  .card-g .v {{ font-size: 20px; margin-top: 2px; }}
  .card-g .sub {{ font-size: 10.5px; color: rgba(235,230,216,.55); margin-top: 3px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; margin: 12px 0 24px; }}
  th {{ text-align: left; font: 10.5px var(--mono); text-transform: uppercase; letter-spacing: .06em; color: var(--ink-3); border-bottom: 1px solid var(--ink); padding: 8px 10px 8px 0; }}
  td {{ border-bottom: 1px solid var(--rule); padding: 10px 10px 10px 0; vertical-align: top; }}
  .num {{ font-family: var(--mono); text-align: right; }}
  .win {{ color: var(--win); font-family: var(--mono); font-size: 11px; text-transform: uppercase; }}
  .loss {{ color: var(--loss); font-family: var(--mono); font-size: 11px; text-transform: uppercase; }}
  .stamp {{ margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(235,230,216,.1); font: 11px var(--mono); color: rgba(235,230,216,.5); }}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <span>ML Ratchet · WakeNet v1.0 · Energy · Directed Wake Graph</span>
    <button id="theme-btn" type="button">Toggle theme</button>
  </div>
  <h1>WakeNet: <em>who is stealing whose wind?</em></h1>
  <p class="lead">From a wind farm's coordinates, tower heights, rotor curves, nacelle yaw angles, and the wind measured on its unwaked front row, WakeNet reads every downstream turbine's wake power loss and 68% error bar. It trained on <b>0 real wake labels</b> — only procedural 5-to-16-turbine wind farms laid into real unwaked turbulence — and was graded once on 24,035 real 10-minute SCADA snapshots across two UK wind farms, including a 14-turbine farm it never saw.</p>
  <dl class="stats">
    <div><dt>Real training labels</dt><dd>0</dd></div>
    <div><dt>FLORIS 4.7 check</dt><dd>1.6e-15</dd></div>
    <div><dt>Rungs (kept / tried)</dt><dd>5 / 8</dd></div>
    <div><dt>Exam snapshots</dt><dd>24,035</dd></div>
    <div><dt>Exam record</dt><dd>7W · 3L</dd></div>
  </dl>

  <section class="scope">
    <div class="picks" id="picks"></div>
    <div class="sgrid">
      <div>
        <canvas id="cv" width="680" height="460"></canvas>
        <div class="ctls">
          <label><span>Rotate wind direction <b id="o-wd">0°</b></span><input id="s-wd" type="range" min="-30" max="30" step="1" value="0" /></label>
          <label><span>Front-row yaw steering <b id="o-st">0°</b></span><input id="s-st" type="range" min="-24" max="24" step="1" value="0" /></label>
          <label><span>Direction meandering σ_θ <b id="o-swd">7.4°</b></span><input id="s-swd" type="range" min="1" max="16" step="0.5" value="7.4" /></label>
        </div>
      </div>
      <div class="readout" id="readout"></div>
    </div>
    <div class="stamp" id="stamp">Loading float16 weights…</div>
  </section>

  <h2>The one-shot real exam (24,035 snapshots · 117,335 waked readings)</h2>
  <p>Run once after rung r6 was locked. Kelmarsh 2020–21 tests unseen years on the 6-turbine MM92 farm; Penmanshiel 2019–20 tests zero-shot transfer to an unseen 14-turbine MM82 farm on a 48 m Berwickshire coastal ridge.</p>
  <table>
    <thead><tr><th>Metric</th><th class="num">WakeNet</th><th class="num">FLORIS 3-D Gauss</th><th class="num">Jensen Park</th><th class="num">Dumb baseline</th><th>Verdict</th></tr></thead>
    <tbody>
      <tr><td>Kelmarsh (2020–21), all waked turbines MAE (n=39,008)</td><td class="num"><b>8.85%</b></td><td class="num">9.51%</td><td class="num">9.95%</td><td class="num">13.63% (unwaked)</td><td class="win">win</td></tr>
      <tr><td>Kelmarsh, direct-wake corridor MAE (n=11,977)</td><td class="num"><b>10.07%</b></td><td class="num">10.75%</td><td class="num">11.63%</td><td class="num">15.30% (dev mean)</td><td class="win">win</td></tr>
      <tr><td>Kelmarsh, multi-wake overlap (≥2 wakes) MAE (n=2,212)</td><td class="num"><b>8.86%</b></td><td class="num">9.90%</td><td class="num">10.12%</td><td class="num">14.46% (dev mean)</td><td class="win">win</td></tr>
      <tr><td>Kelmarsh, upstream yaw offset ≥8° MAE (n=5,401)</td><td class="num"><b>9.76%</b></td><td class="num">10.46%</td><td class="num">11.25%</td><td class="num">13.75% (unwaked)</td><td class="win">win</td></tr>
      <tr><td>Kelmarsh, partial-parked farm MAE (n=6,146)</td><td class="num"><b>9.18%</b></td><td class="num">9.94%</td><td class="num">9.81%</td><td class="num">13.62% (unwaked)</td><td class="win">win</td></tr>
      <tr><td>Kelmarsh, farm-wide wake loss MAE (n=15,059)</td><td class="num"><b>2.57%</b></td><td class="num">2.80%</td><td class="num">2.99%</td><td class="num">4.44% (dev mean)</td><td class="win">win</td></tr>
      <tr><td>Penmanshiel (14× MM82, zero-shot), all waked MAE (n=78,327)</td><td class="num"><b>15.01%</b></td><td class="num">16.63%</td><td class="num">17.90%</td><td class="num">15.88% (dev mean)</td><td class="win">win</td></tr>
      <tr><td>Penmanshiel (zero-shot), upstream yaw ≥8° MAE (n=14,813)</td><td class="num"><b>15.08%</b></td><td class="num">16.75%</td><td class="num">18.05%</td><td class="num">16.27% (unwaked)</td><td class="win">win</td></tr>
      <tr><td>Penmanshiel (zero-shot), multi-wake interior MAE (n=24,308)</td><td class="num">17.32%</td><td class="num">19.12%</td><td class="num">21.94%</td><td class="num"><b>16.43%</b> (dev mean)</td><td class="loss">loss</td></tr>
      <tr><td>Penmanshiel (zero-shot), farm-wide wake loss MAE (n=8,976)</td><td class="num">4.75%</td><td class="num">5.16%</td><td class="num">5.82%</td><td class="num"><b>4.55%</b> (dev mean)</td><td class="loss">loss</td></tr>
    </tbody>
  </table>
  <p><b>68% error-bar coverage:</b> Kelmarsh waked turbines <b>72.2%</b>, Kelmarsh farm wake loss <b>77.4%</b>; Penmanshiel waked turbines <b>50.7%</b>, Penmanshiel farm wake loss <b>61.5%</b>. On Penmanshiel's 48 m coastal ridge and adjacent forestry blocks, hilltop wind speedup accelerates ridge-top turbines (T12–T15) well above coastal-slope front-row units, adding terrain variance that gentle farmland hills do not teach.</p>

  <h2>The ratchet (8 rungs, re-scored on val3)</h2>
  <table>
    <thead><tr><th>Rung</th><th>Change</th><th class="num">Frozen val3</th><th class="num">Dev H1</th><th class="num">Dev H2</th><th class="num">Dev farm</th><th>Decision</th></tr></thead>
    <tbody>
      <tr><td>r1</td><td>2-D Jensen in Gaussian noise; node-only MLP (no graph)</td><td class="num">0.125</td><td class="num">14.25%</td><td class="num">14.68%</td><td class="num">4.27%</td><td>kept</td></tr>
      <tr><td>r2</td><td>Directed GNN over (dx/D, dy/D, dz/D, yaw) edges, unanchored</td><td class="num">0.374</td><td class="num">10.39%</td><td class="num">10.74%</td><td class="num">3.38%</td><td>kept</td></tr>
      <tr><td>r3</td><td>Physics-anchored EdgeWakeGNN (rejected on val1 by 0.005; kept on val2)</td><td class="num">0.426</td><td class="num">10.12%</td><td class="num">10.43%</td><td class="num">3.29%</td><td>kept</td></tr>
      <tr><td>r4</td><td>v2 physics: 3-D Bastankhah Gauss + hub-height shear + real 2017–18 backdrop</td><td class="num">0.517</td><td class="num">9.97%</td><td class="num">10.25%</td><td class="num">3.14%</td><td>kept</td></tr>
      <tr><td>r5</td><td>Naive meandering: smear wakes by 100% of raw nacelle vane StdDev (~7.7°)</td><td class="num">0.464</td><td class="num">11.09%</td><td class="num">11.37%</td><td class="num">3.30%</td><td>rejected</td></tr>
      <tr><td><b>r6</b></td><td><b>v3 physics (dev twins): meso-scale meandering (0.16–0.36× vane σ) + 3-D hill speedup</b></td><td class="num"><b>0.552</b></td><td class="num"><b>9.38%</b></td><td class="num"><b>9.56%</b></td><td class="num"><b>2.88%</b></td><td><b>released</b></td></tr>
      <tr><td>r7</td><td>Softmax attention in place of additive sigmoid-gated wake superposition</td><td class="num">0.553</td><td class="num">9.41%</td><td class="num">9.60%</td><td class="num">2.89%</td><td>rejected</td></tr>
      <tr><td>r8</td><td>Fine-tune on mixed v2 + v3 banks (48,000 simulated farms)</td><td class="num">0.543</td><td class="num">9.62%</td><td class="num">9.85%</td><td class="num">2.99%</td><td>rejected</td></tr>
    </tbody>
  </table>

  <h2>Limits &amp; credits</h2>
  <ul>
    <li><b>Complex coastal ridges &amp; forestry:</b> Penmanshiel's 48 m relief and forest edges speed hilltop rotors up past front-row power, causing losses vs constant baselines on multi-wake interior and farm-total rows and narrowing 68% bar coverage to 51%.</li>
    <li><b>Atmospheric stability is unobserved:</b> Nacelle 10-min SCADA carries no Obukhov heat-flux sensor, leaving convective vs nocturnal stable wake decay inside the residual bar.</li>
    <li><b>Data credit:</b> Charlie Plumley / Cubico Sustainable Investments — Kelmarsh (Zenodo 16807551) and Penmanshiel (Zenodo 16807304) Wind Farm Open Data, CC BY 4.0. Physics checked against NREL FLORIS 4.7.</li>
  </ul>
</div>
<script id="wn-data" type="application/json">{payload}</script>
<script>
const D = JSON.parse(document.getElementById('wn-data').textContent);
document.getElementById('theme-btn').onclick = () => {{
  const cur = document.documentElement.getAttribute('data-theme');
  document.documentElement.setAttribute('data-theme', cur === 'dark' ? 'light' : 'dark');
}};
function b64ToBuf(b64) {{
  const bin = atob(b64), u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return u.buffer;
}}
function f16ToF32(u16) {{
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {{
    const h = u16[i], s = (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x03ff;
    let bits = 0;
    if (e === 0) bits = m ? s | ((127 - 14) << 23) : s;
    else if (e === 31) bits = s | 0x7f800000 | (m << 13);
    else bits = s | ((e + 112) << 23) | (m << 13);
    const dv = new DataView(new ArrayBuffer(4)); dv.setUint32(0, bits); out[i] = dv.getFloat32(0);
  }}
  return out;
}}
function unpack(man, b64) {{
  const buf = b64ToBuf(b64), t = {{}};
  // mlratchet-pack/1: tensors is a {{name: {{dtype, shape, offset, bytes}}}} map (see site/src/lib/nn/pack.ts)
  for (const [name, e] of Object.entries(man.tensors)) {{
    const data = e.dtype === 'f16' ? f16ToF32(new Uint16Array(buf.slice(e.offset, e.offset + e.bytes)))
                                   : new Float32Array(buf.slice(e.offset, e.offset + e.bytes));
    t[name] = {{ data, shape: e.shape }};
  }}
  return {{ meta: man.meta, t }};
}}
const W = unpack(D.wMan, D.wB64), X = unpack(D.eMan, D.eB64);
const DEG = Math.PI / 180, clip = (v, a, b) => Math.min(b, Math.max(a, v));
const erf = (x) => {{ const a = Math.abs(x), t = 1/(1+0.3275911*a); return (x<0?-1:1)*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-a*a)); }};
const gelu = (x) => 0.5 * x * (1 + erf(x * 0.7071067811865476));
const sigm = (x) => 1 / (1 + Math.exp(-clip(x, -40, 40)));
function lin(x, w, b) {{
  const [O, I] = w.shape, out = new Float32Array(O), wd = w.data, bd = b.data;
  for (let o = 0; o < O; o++) {{ let s = bd[o], off = o * I; for (let i = 0; i < I; i++) s += wd[off + i] * x[i]; out[o] = s; }}
  return out;
}}
function interp1(x, xs, ys) {{
  if (x <= xs[0]) return ys[0]; if (x >= xs[xs.length-1]) return ys[ys.length-1];
  let k = 0; while (k < xs.length - 2 && xs[k+1] < x) k++;
  return ys[k] + ((x - xs[k]) / (xs[k+1] - xs[k])) * (ys[k+1] - ys[k]);
}}
function rotWind(east, north, wd) {{
  const n = east.length, ec = 0.5*(Math.min(...east)+Math.max(...east)), nc = 0.5*(Math.min(...north)+Math.max(...north));
  const s = Math.sin(wd*DEG), c = Math.cos(wd*DEG), x = new Float64Array(n), y = new Float64Array(n);
  for (let i = 0; i < n; i++) {{ const de = east[i]-ec, dn = north[i]-nc; x[i] = -de*s - dn*c + ec; y[i] = de*c - dn*s + nc; }}
  return {{ x, y }};
}}
function solveAnc(east, north, hubH, turb, uInf, wd, tiInf, yaw, on, rho) {{
  const n = east.length, Dm = turb.rotor_diameter, {{ x: xc, y: yc }} = rotWind(east, north, wd);
  const ord = Array.from({{length:n}}, (_,i)=>i).sort((a,b)=>xc[a]-xc[b]), inv = new Int32Array(n);
  ord.forEach((v,k)=>inv[v]=k);
  let hS=0,hC=0; for(let i=0;i<n;i++) if(on[i]){{hS+=hubH[i];hC++;}} const zRef=hC?hS/hC:turb.hub_height;
  const rhoS = Math.cbrt(rho/1.225), REL=[-0.25,0,0.25], yg=new Float64Array(n*9), zg=new Float64Array(n*9), u0=new Float64Array(n*9), wAcc=new Float64Array(n*9), tiG=new Float64Array(n*9).fill(tiInf), uFrS=new Float64Array(n);
  for(let s=0;s<n;s++){{ const id=ord[s]; let c3=0; for(let a=0;a<3;a++) for(let b=0;b<3;b++){{ const g=s*9+a*3+b; yg[g]=yc[id]+Dm*REL[a]; zg[g]=Math.max(hubH[id]+Dm*REL[b],1); u0[g]=uInf*Math.pow(zg[g]/zRef,0.14); c3+=u0[g]**3; }} uFrS[s]=Math.cbrt(c3/9); }}
  const uRS=new Float64Array(n), ctS=new Float64Array(n), pS=new Float64Array(n), pDefS=new Float64Array(n*n), pOvlS=new Float64Array(n*n), pDelS=new Float64Array(n*n), pSyS=new Float64Array(n*n);
  for(let j=0;j<n;j++){{
    const idJ=ord[j]; let s3=0; for(let q=0;q<9;q++) s3+=Math.max(u0[j*9+q]-Math.sqrt(wAcc[j*9+q]),0)**3;
    const uA=Math.cbrt(s3/9); uRS[j]=uA; if(!on[idJ]||uA<=0.1) continue;
    const yJ=yaw[idJ], cy=Math.max(Math.cos(yJ*DEG),0);
    pS[j]=interp1(uA*rhoS*Math.pow(cy,1.88/3), turb.wind_speed, turb.power_kw);
    const ctR=clip(interp1(uA, turb.wind_speed, turb.thrust_coefficient), 1e-4, 0.9999), ctJ=ctR*cy; ctS[j]=ctJ;
    const aiJ=cy>1e-6?(0.5/cy)*(1-Math.sqrt(Math.max(1-ctJ*cy,0))):0;
    const xJ=xc[idJ], yJc=yc[idJ], hJ=hubH[idJ], yo=-yJ, cyo=Math.max(Math.cos(yo*DEG),1e-6), ctC=clip(ctJ*cyo,0,0.9999), s1=Math.sqrt(Math.max(1-ctJ,1e-8)), s1c=Math.sqrt(Math.max(1-ctC,1e-8));
    const C0=1-s1, M0=Math.max(C0*(2-C0),1e-8), m0s=Math.sqrt(M0), E0=C0*C0-3*Math.exp(1/12)*C0+3*Math.exp(1/3);
    const sz0D=0.5*Dm*Math.sqrt(Math.max((ctC/Math.max(2*(1-s1c),1e-8))/(1+s1),1e-8)), sy0D=Math.max(sz0D*cyo,1e-6);
    const thC=((0.3*(yo*DEG))/cyo)*(1-s1c), sz0V=0.5*Dm*Math.sqrt(Math.max((ctJ/Math.max(2*(1-s1),1e-8))/(1+s1),1e-8)), sy0V=Math.max(sz0V*cyo,1e-6), sCore=0.501*Dm*Math.sqrt(ctJ/2);
    for(let i=j+1;i<n;i++){{
      const dx=xc[ord[i]]-xJ; if(dx<=1e-4) continue; let dfS=0,ov=0,cDl=0,cSy=0;
      for(let q=0;q<9;q++){{
        const ti=tiG[j*9+q], ky=0.38*ti+0.004; let dl=0;
        if(Math.abs(yJ)>=1e-7){{ const x0D=(Dm*cyo*(1+s1c))/(Math.SQRT2*Math.max(4*0.58*ti+2*0.077*(1-s1),1e-6)), d0=Math.tan(thC)*x0D;
          if(dx<=x0D) dl=(dx/Math.max(x0D,1e-6))*d0; else {{ const syF=ky*(dx-x0D)+sy0D, szF=ky*(dx-x0D)+sz0D, mr=Math.sqrt((syF*szF)/(sy0D*sz0D)); dl=d0+(thC*E0/5.2)*Math.sqrt((sy0D*sz0D)/(ky*ky*M0))*Math.log(Math.max((1.6+m0s)*(1.6*mr-m0s),1e-12)/Math.max((1.6-m0s)*(1.6*mr+m0s),1e-12)); }} }}
        const x0V=(Dm*cyo*(1+s1))/(Math.SQRT2*Math.max(4*0.58*ti+2*0.077*(1-s1),1e-6));
        const sy=dx<x0V?Math.max((1-dx/x0V)*sCore+(dx/x0V)*sy0V,1e-3):ky*(dx-x0V)+sy0V, sz=dx<x0V?Math.max((1-dx/x0V)*sCore+(dx/x0V)*sz0V,1e-3):ky*(dx-x0V)+sz0V;
        const gI=i*9+q, dy=yg[gI]-yJc-dl, dz=zg[gI]-hJ, r2=(dy*dy)/(2*sy*sy)+(dz*dz)/(2*sz*sz), cA=1-Math.sqrt(clip(1-(ctJ*cyo)/(8*sy*sz/(Dm*Dm)),0,1));
        const du=dx>0.1?u0[gI]*cA*Math.exp(-r2):0; wAcc[gI]+=du*du; dfS+=du/Math.max(u0[gI],0.1); if(du>0.05)ov++; if(q===4){{cDl=dl;cSy=sy;}}
      }}
      const aO=ov/9; pDefS[j*n+i]=dfS/9; pOvlS[j*n+i]=aO; pDelS[j*n+i]=cDl/Dm; pSyS[j*n+i]=cSy/Dm;
      if(aiJ>1e-6&&dx>0.1&&dx<=15*Dm){{ const wat=0.5*Math.pow(aiJ,0.8)*Math.pow(Math.max(tiInf,1e-4),0.1)*Math.pow(dx/Dm,-0.32);
        for(let q=0;q<9;q++){{ const gI=i*9+q; if(Math.abs(yJc-yg[gI])<2*Dm) tiG[gI]=Math.max(tiG[gI], Math.sqrt((aO*wat)**2+tiInf**2)); }} }}
    }}
  }}
  const uR=new Float64Array(n), uF=new Float64Array(n), ct=new Float64Array(n), p=new Float64Array(n), tiO=new Float64Array(n), pD=new Float64Array(n*n), pO=new Float64Array(n*n), pDl=new Float64Array(n*n), pSy=new Float64Array(n*n);
  for(let i=0;i<n;i++){{ const si=inv[i]; uR[i]=uRS[si]; uF[i]=uFrS[si]; ct[i]=ctS[si]; p[i]=pS[si]; let ts=0; for(let q=0;q<9;q++)ts+=tiG[si*9+q]; tiO[i]=ts/9;
    for(let k=0;k<n;k++){{ const sk=inv[k]; pD[i*n+k]=pDefS[si*n+sk]; pO[i*n+k]=pOvlS[si*n+sk]; pDl[i*n+k]=pDelS[si*n+sk]; pSy[i*n+k]=pSyS[si*n+sk]; }} }}
  return {{ x:xc, y:yc, uR, uF, ti:tiO, ct, p, pD, pO, pDl, pSy }};
}}
function runNet(idx, dWd, steer, swd, tog) {{
  const m = X.meta[idx], east = X.t['east_'+idx].data, north = X.t['north_'+idx].data, elev = X.t['elev_'+idx].data, hubH = X.t['hub_h_'+idx].data;
  const bYaw = X.t['yaw_'+idx].data, bOn = Array.from(X.t['online_'+idx].data, v=>v>0.5), bFr = Array.from(X.t['free_'+idx].data, v=>v>0.5);
  const on = bOn.map((v,k)=>tog.has(k)?!v:v); if(!on.some(Boolean)) on[0]=true;
  const wd = (m.wind_dir_deg + dWd + 360) % 360, n = east.length, turb = W.meta.turbines[m.turbine_model], Dm = turb.rotor_diameter;
  const yaw = Float64Array.from(bYaw, (y,k)=>!on[k]?0:(bFr[k]?clip(y+steer,-30,30):y));
  const {{x:xc, y:yc}} = rotWind(east, north, wd);
  const isFree = on.map((o,i)=> {{ if(!o) return false; for(let j=0;j<n;j++) if(j!==i&&on[j]&&(xc[i]-xc[j])>0.2*Dm&&(xc[i]-xc[j])<25*Dm&&Math.abs(yc[i]-yc[j])<1.6*Dm+0.14*(xc[i]-xc[j])) return false; return true; }});
  if(!isFree.some(Boolean)) isFree[0]=true;
  const a0 = solveAnc(east, north, hubH, turb, m.u_inf_ms, wd, m.ti_inf, yaw, on, m.air_density);
  const dT = clip(swd, 1.5, 18);
  const aP = solveAnc(east, north, hubH, turb, m.u_inf_ms, wd+dT, m.ti_inf, yaw.map(v=>v+0.35*dT), on, m.air_density);
  const aM = solveAnc(east, north, hubH, turb, m.u_inf_ms, wd-dT, m.ti_inf, yaw.map(v=>v-0.35*dT), on, m.air_density);
  const fNorm = (pa) => {{ let s=0,c=0; for(let i=0;i<n;i++) if(isFree[i]){{s+=pa[i];c++;}} const den=c?s/c:1; return Float32Array.from({{length:n}},(_,i)=>on[i]?(den<1?1:pa[i]/den):0); }};
  const eA = fNorm(a0.p), eP = fNorm(aP.p), eM = fNorm(aM.p);
  let xMin=Infinity,hS=0,zS=0,onC=0; for(let i=0;i<n;i++) if(on[i]){{ if(a0.x[i]<xMin)xMin=a0.x[i]; hS+=hubH[i]; zS+=elev[i]; onC++; }}
  const hMn=hS/onC, zMn=zS/onC, pFS=interp1(m.u_inf_ms*Math.cbrt(m.air_density/1.225), turb.wind_speed, turb.power_kw)/turb.rated_power_kw, ctFS=interp1(m.u_inf_ms, turb.wind_speed, turb.thrust_coefficient);
  const nx = new Float32Array(n*17), ex = new Float32Array(n*n*12), em = new Uint8Array(n*n);
  for(let i=0;i<n;i++){{ const b=i*17; nx[b]=on[i]?1:0; nx[b+1]=isFree[i]?1:0; nx[b+2]=(m.u_inf_ms-8.5)/3; nx[b+3]=pFS; nx[b+4]=ctFS; nx[b+5]=(m.ti_inf-0.09)/0.05; nx[b+6]=(swd-5)/4; nx[b+7]=clip(yaw[i]/25,-2.5,2.5); nx[b+8]=clip(1-Math.pow(Math.max(Math.cos(yaw[i]*DEG),0),1.88),0,1); nx[b+9]=(hubH[i]-hMn)/Dm; nx[b+10]=(elev[i]-zMn)/Dm; nx[b+11]=(a0.x[i]-xMin)/(10*Dm); nx[b+12]=eA[i]; nx[b+13]=on[i]?1-a0.uR[i]/Math.max(a0.uF[i],0.5):0; nx[b+14]=on[i]?(a0.ti[i]-m.ti_inf)/0.1:0; nx[b+15]=2*(eP[i]+eM[i]-2*eA[i]); nx[b+16]=2*(Math.abs(eP[i]-eA[i])+Math.abs(eM[i]-eA[i])); }}
  for(let j=0;j<n;j++) for(let i=0;i<n;i++) if(i!==j&&on[j]&&on[i]){{ const dxd=(a0.x[i]-a0.x[j])/Dm, dyd=(a0.y[i]-a0.y[j])/Dm; if(dxd>0.1&&dxd<30&&Math.abs(dyd)<4.5+0.25*dxd){{ const p=j*n+i, b=p*12; em[p]=1; ex[b]=1/(dxd+2); ex[b+1]=Math.log1p(dxd)/3; ex[b+2]=clip(dyd/3,-2.5,2.5); ex[b+3]=clip(Math.abs(dyd-a0.pDl[p])/(Math.max(a0.pSy[p],0.25)+0.35)/3,0,3); ex[b+4]=clip((hubH[i]-hubH[j])/Dm,-2,2); ex[b+5]=clip((elev[i]-elev[j])/Dm,-3,3); ex[b+6]=a0.pD[p]; ex[b+7]=a0.pO[p]; ex[b+8]=clip(a0.pDl[p]*4,-2.5,2.5); ex[b+9]=clip(yaw[j]/25,-2.5,2.5); ex[b+10]=a0.ct[j]; ex[b+11]=clip(swd*DEG*dxd,0,3); }} }}
  const H=96, h=new Float32Array(n*H), eE=new Float32Array(n*n*H), t=W.t;
  for(let i=0;i<n;i++) if(on[i]){{ const o=lin(nx.subarray(i*17,(i+1)*17),t['node_enc.weight'],t['node_enc.bias']); for(let k=0;k<H;k++)h[i*H+k]=gelu(o[k]); }}
  for(let p=0;p<n*n;p++) if(em[p]){{ const o=lin(ex.subarray(p*12,(p+1)*12),t['edge_enc.weight'],t['edge_enc.bias']); for(let k=0;k<H;k++)eE[p*H+k]=gelu(o[k]); }}
  const pv=new Float32Array(3*H), uv=new Float32Array(3*H);
  for(let l=0;l<3;l++){{ const pr='layers.'+l+'.', ag=new Float32Array(n*H);
    for(let j=0;j<n;j++) for(let i=0;i<n;i++){{ const p=j*n+i; if(!em[p])continue; pv.set(h.subarray(j*H,(j+1)*H),0); pv.set(h.subarray(i*H,(i+1)*H),H); pv.set(eE.subarray(p*H,(p+1)*H),2*H);
      const mh=lin(pv,t[pr+'msg_fc1.weight'],t[pr+'msg_fc1.bias']); for(let k=0;k<H;k++)mh[k]=gelu(mh[k]);
      const ms=lin(mh,t[pr+'msg_fc2.weight'],t[pr+'msg_fc2.bias']), gt=sigm(lin(mh,t[pr+'gate_fc.weight'],t[pr+'gate_fc.bias'])[0]);
      for(let k=0;k<H;k++) ag[i*H+k]+=gt*ms[k]; }}
    const hf=new Float32Array(H); for(let i=0;i<n;i++) if(on[i]) for(let k=0;k<H;k++) hf[k]+=h[i*H+k]/onC;
    const hn=new Float32Array(n*H); for(let i=0;i<n;i++) if(on[i]){{ uv.set(h.subarray(i*H,(i+1)*H),0); uv.set(ag.subarray(i*H,(i+1)*H),H); uv.set(hf,2*H);
      const u1=lin(uv,t[pr+'upd_fc1.weight'],t[pr+'upd_fc1.bias']); for(let k=0;k<H;k++)u1[k]=gelu(u1[k]);
      const dh=lin(u1,t[pr+'upd_fc2.weight'],t[pr+'upd_fc2.bias']); for(let k=0;k<H;k++)hn[i*H+k]=h[i*H+k]+dh[k]; }} h.set(hn);
  }}
  const eu=new Float32Array(n), ls=new Float32Array(n), ni=new Float32Array(H+17); let fS=0,fC=0;
  for(let i=0;i<n;i++){{ ni.set(h.subarray(i*H,(i+1)*H),0); ni.set(nx.subarray(i*17,(i+1)*17),H); const nh=lin(ni,t['node_head1.weight'],t['node_head1.bias']); for(let k=0;k<nh.length;k++)nh[k]=gelu(nh[k]);
    const no=lin(nh,t['node_head2.weight'],t['node_head2.bias']); ls[i]=clip(no[1],-5,-0.2); if(on[i]){{ eu[i]=eA[i]+0.5*Math.tanh(no[0]); if(isFree[i]){{fS+=eu[i];fC++;}} }} }}
  const fm=Math.max(fS/Math.max(fC,1),0.25), eta=Float32Array.from({{length:n}},(_,i)=>on[i]?clip(eu[i]/fm,0,1.35):0);
  let sE=0; for(let i=0;i<n;i++) if(on[i]) sE+=eta[i]; const fLoss = 1 - sE/onC;
  return {{ m, east, north, turb, wd, yaw, on, isFree, eta, sig: Float32Array.from(ls, v=>Math.exp(v)), eA, fLoss, ct: a0.ct, ti: a0.ti, pTrue: X.t['ptrue_'+idx].data }};
}}
let st = {{ i: 0, dWd: 0, steer: 0, swd: X.meta[0].sigma_wd_deg, tog: new Set(), sel: 0 }}, hits = [];
const picksEl = document.getElementById('picks');
X.meta.forEach((m, i) => {{
  const b = document.createElement('button');
  b.textContent = m.farm + ' · ' + m.id.slice(3).replace(/_/g,' ');
  b.onclick = () => {{ st.i=i; st.dWd=0; st.steer=0; st.swd=m.sigma_wd_deg; st.tog.clear(); document.getElementById('s-wd').value=0; document.getElementById('s-st').value=0; document.getElementById('s-swd').value=st.swd; render(); }};
  picksEl.append(b);
}});
document.getElementById('s-wd').oninput = (e) => {{ st.dWd = +e.target.value; render(); }};
document.getElementById('s-st').oninput = (e) => {{ st.steer = +e.target.value; render(); }};
document.getElementById('s-swd').oninput = (e) => {{ st.swd = +e.target.value; render(); }};
const cv = document.getElementById('cv'), g = cv.getContext('2d');
cv.onclick = (ev) => {{
  const r = cv.getBoundingClientRect(), px = (ev.clientX-r.left)*(cv.width/r.width), py = (ev.clientY-r.top)*(cv.height/r.height);
  for(const h of hits) if(Math.hypot(px-h.x, py-h.y)<18){{ if(st.sel===h.k){{ if(st.tog.has(h.k))st.tog.delete(h.k); else st.tog.add(h.k); }} st.sel=h.k; render(); break; }}
}};
function render() {{
  [...picksEl.children].forEach((b,k)=>b.className=k===st.i?'on':'');
  document.getElementById('o-wd').textContent = (st.dWd>=0?'+':'')+st.dWd+'°';
  document.getElementById('o-st').textContent = (st.steer>=0?'+':'')+st.steer+'°';
  document.getElementById('o-swd').textContent = st.swd.toFixed(1)+'°';
  const t0 = performance.now(), R = runNet(st.i, st.dWd, st.steer, st.swd, st.tog), ms = performance.now() - t0;
  const Wc = cv.width, Hc = cv.height, mapH = 300, Dm = R.turb.rotor_diameter;
  g.fillStyle = '#0d0f0c'; g.fillRect(0,0,Wc,Hc);
  const emin=Math.min(...R.east), emax=Math.max(...R.east), nmin=Math.min(...R.north), nmax=Math.max(...R.north), mg=3*Dm;
  let spE=(emax-emin)+2*mg, spN=(nmax-nmin)+2*mg; const asp=Wc/mapH; if(spE/spN>asp)spN=spE/asp; else spE=spN*asp;
  const eLo=0.5*(emin+emax)-0.5*spE, nLo=0.5*(nmin+nmax)-0.5*spN;
  const sx = e => ((e-eLo)/spE)*Wc, sy = n => (1-(n-nLo)/spN)*mapH;
  hits = [];
  for(let k=0;k<R.east.length;k++){{
    const x=sx(R.east[k]), y=sy(R.north[k]); hits.push({{k,x,y}});
    if(R.on[k]&&R.ct[k]>0.05){{
      const th=R.wd*DEG, dx=-Math.sin(th), dy=Math.cos(th), len=6*Dm*(Wc/spE);
      const grd=g.createLinearGradient(x,y,x+dx*len,y+dy*len); grd.addColorStop(0,'rgba(56,189,248,0.42)'); grd.addColorStop(1,'rgba(56,189,248,0)');
      g.strokeStyle=grd; g.lineWidth=Math.max(10, 1.3*Dm*(Wc/spE)); g.beginPath(); g.moveTo(x,y); g.lineTo(x+dx*len,y+dy*len); g.stroke();
    }}
  }}
  for(let k=0;k<R.east.length;k++){{
    const x=sx(R.east[k]), y=sy(R.north[k]);
    g.fillStyle = !R.on[k] ? '#555' : (R.isFree[k] ? '#ebe6d8' : '#38bdf8');
    g.beginPath(); g.arc(x, y, k===st.sel?7:5, 0, Math.PI*2); g.fill();
    g.fillStyle = '#ebe6d8'; g.font = '10px monospace'; g.fillText(R.m.codes[k], x-12, y-10);
  }}
  const nWt=R.m.codes.length, sw=(Wc-40)/nWt, bTop=320, bH=115, mod=st.dWd||st.steer||st.tog.size;
  for(let k=0;k<nWt;k++){{
    const cx=28+(k+0.5)*sw, hBar=Math.min(1.2,R.eta[k])/1.2*bH;
    g.fillStyle = R.isFree[k] ? 'rgba(235,230,216,0.35)' : 'rgba(56,189,248,0.65)';
    if(R.on[k]) g.fillRect(cx-8, bTop+bH-hBar, 16, hBar);
    if(!mod&&R.on[k]){{ const yt=bTop+bH-Math.min(1.2,R.pTrue[k]/R.m.p_front_kw)/1.2*bH; g.strokeStyle='#ebe6d8'; g.lineWidth=2; g.beginPath(); g.moveTo(cx-10,yt); g.lineTo(cx+10,yt); g.stroke(); }}
    g.fillStyle='rgba(235,230,216,0.6)'; g.font='9px monospace'; g.fillText(R.m.codes[k], cx-10, Hc-6);
  }}
  const sel = Math.min(st.sel, nWt-1), pF = R.m.p_front_kw;
  let tSum=0,tCnt=0; R.on.forEach((o,k)=>{{if(o){{tSum+=R.pTrue[k]/pF;tCnt++;}}}});
  document.getElementById('readout').innerHTML = `
    <div class="card-g"><div class="k">Farm wake loss</div><div class="v">${{(100*R.fLoss).toFixed(1)}}%</div><div class="sub">SCADA recorded: ${{(100*(1-tSum/tCnt)).toFixed(1)}}%</div></div>
    <div class="card-g"><div class="k">Selected · ${{R.m.codes[sel]}}</div><div class="v">${{(100*(1-R.eta[sel])).toFixed(1)}}% ± ${{(100*R.sig[sel]).toFixed(1)}}%</div><div class="sub">Power ${{(R.eta[sel]*pF).toFixed(0)}} kW · SCADA ${{R.pTrue[sel].toFixed(0)}} kW</div></div>
    <div class="card-g"><div class="k">Inflow (${{R.m.timestamp_utc}} UTC)</div><div class="v">${{R.m.u_inf_ms.toFixed(1)}} m/s @ ${{Math.round(R.wd)}}°</div><div class="sub">Click a rotor twice to park/unpark</div></div>`;
  document.getElementById('stamp').textContent = `● EdgeWakeGNN (245,767 params) ran in ${{ms.toFixed(1)}} ms in your browser · ${{R.m.stratum}}`;
}}
render();
</script>
</body>
</html>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e3:.1f} KB)")


if __name__ == "__main__":
    main()
