// Pure TypeScript port of WakeNet (models/wakenet/physics.py, pipeline.py, wakenet.py).
// Solves the 3-D Bastankhah-Porte-Agel Gaussian wake anchor on a 3x3 rotor grid, builds the
// directed wake graph, and runs the 3-layer EdgeWakeGNN in the browser. Zero DOM dependencies.
import { erf, linear } from '../../lib/nn/ops.ts';
import type { Pack } from '../../lib/nn/pack.ts';

export interface TurbineSpec {
  name: string;
  rotor_diameter: number;
  hub_height: number;
  rated_power_kw: number;
  wind_speed: number[];
  power_kw: number[];
  thrust_coefficient: number[];
  ref_air_density: number;
  cosine_loss_exponent_yaw: number;
}

export interface WMeta {
  model: string;
  version: string;
  released_rung: string;
  n_params: number;
  arch: { hidden: number; n_layers: number; head_hidden: number; anchored: boolean; use_edges: boolean; agg: string };
  sigma_factor: { waked_eta: number; free_eta: number; farm_wake: number };
  dev_baselines: { mean_waked_eta: number; mean_farm_wake_loss: number };
  turbines: Record<string, TurbineSpec>;
}

export interface WExampleMeta {
  id: string;
  stratum: string;
  farm: string;
  turbine_model: string;
  timestamp_utc: string;
  codes: string[];
  u_inf_ms: number;
  wind_dir_deg: number;
  ti_inf: number;
  sigma_wd_deg: number;
  air_density: number;
  p_front_kw: number;
}

export interface WakeReading {
  eta: Float32Array;
  eta_sig: Float32Array;
  wake_loss: Float32Array;
  power_kw: Float32Array;
  power_sig_kw: Float32Array;
  u_rotor_ms: Float32Array;
  eta_anchor: Float32Array;
  is_free: boolean[];
  online: boolean[];
  p_front_kw: number;
  farm_power_kw: number;
  farm_free_kw: number;
  farm_wake_loss: number;
  farm_wake_sig: number;
  x_m: Float64Array;
  y_m: Float64Array;
  ct_anchor: Float64Array;
  ti_anchor: Float64Array;
}

const DEG = Math.PI / 180;
const clip = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const geluScalar = (x: number) => 0.5 * x * (1 + erf(x * 0.7071067811865476));
const sigmoidScalar = (x: number) => {
  const z = clip(x, -40, 40);
  return 1 / (1 + Math.exp(-z));
};

function interp1(x: number, xs: number[], ys: number[], left = 0, right = 0): number {
  const n = xs.length;
  if (x < xs[0]) return left;
  if (x > xs[n - 1]) return right;
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  const dx = xs[hi] - xs[lo];
  if (dx <= 0) return ys[lo];
  return ys[lo] + ((x - xs[lo]) / dx) * (ys[hi] - ys[lo]);
}

export function rotateToWindFrame(east: ArrayLike<number>, north: ArrayLike<number>, windDirDeg: number): { x: Float64Array; y: Float64Array } {
  const n = east.length;
  let emin = Infinity, emax = -Infinity, nmin = Infinity, nmax = -Infinity;
  for (let i = 0; i < n; i++) {
    if (east[i] < emin) emin = east[i];
    if (east[i] > emax) emax = east[i];
    if (north[i] < nmin) nmin = north[i];
    if (north[i] > nmax) nmax = north[i];
  }
  const ec = 0.5 * (emin + emax), nc = 0.5 * (nmin + nmax);
  const th = windDirDeg * DEG;
  const s = Math.sin(th), c = Math.cos(th);
  const x = new Float64Array(n), y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const de = east[i] - ec, dn = north[i] - nc;
    x[i] = -de * s - dn * c + ec;
    y[i] = de * c - dn * s + nc;
  }
  return { x, y };
}

export function freestreamMask(east: ArrayLike<number>, north: ArrayLike<number>, D: number, windDirDeg: number, online: boolean[]): boolean[] {
  const n = east.length;
  const { x, y } = rotateToWindFrame(east, north, windDirDeg);
  const isFree = new Array<boolean>(n).fill(false);
  let anyFree = false;
  for (let i = 0; i < n; i++) {
    if (!online[i]) continue;
    let waked = false;
    for (let j = 0; j < n; j++) {
      if (j === i || !online[j]) continue;
      const dx = x[i] - x[j];
      const dy = Math.abs(y[i] - y[j]);
      if (dx > 0.2 * D && dx < 25.0 * D && dy < 1.6 * D + 0.14 * dx) {
        waked = true;
        break;
      }
    }
    if (!waked) {
      isFree[i] = true;
      anyFree = true;
    }
  }
  if (!anyFree) {
    let best = -1, bestX = Infinity;
    for (let i = 0; i < n; i++) {
      if (online[i] && x[i] < bestX) { bestX = x[i]; best = i; }
    }
    if (best >= 0) isFree[best] = true;
  }
  return isFree;
}

interface AnchorSolve {
  x_m: Float64Array;
  y_m: Float64Array;
  u_rotor_ms: Float64Array;
  u_free_ms: Float64Array;
  ti_eff: Float64Array;
  ct: Float64Array;
  power_kw: Float64Array;
  pair_deficit: Float64Array;     // [n * n]
  pair_overlap: Float64Array;     // [n * n]
  pair_deflection_d: Float64Array;// [n * n]
  pair_sigma_y_d: Float64Array;   // [n * n]
}

const ALPHA = 0.58, BETA = 0.077, KA = 0.38, KB = 0.004, DM = 1.0;
const CH_INIT = 0.1, CH_CONST = 0.5, CH_AI = 0.8, CH_DOWN = -0.32;
const SHEAR = 0.14;
const REL3 = [-0.25, 0.0, 0.25];

export function solveAnchor(
  east: ArrayLike<number>,
  north: ArrayLike<number>,
  hubH: ArrayLike<number>,
  turb: TurbineSpec,
  uInf: number,
  windDirDeg: number,
  tiInf: number,
  yawDeg: ArrayLike<number>,
  online: boolean[],
  airDensity: number,
): AnchorSolve {
  const n = east.length;
  const D = turb.rotor_diameter;
  const { x: xc, y: yc } = rotateToWindFrame(east, north, windDirDeg);

  // Upwind-to-downwind topological order
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => xc[a] - xc[b]);
  const invOrder = new Int32Array(n);
  for (let k = 0; k < n; k++) invOrder[order[k]] = k;

  let hSum = 0, hCnt = 0;
  for (let i = 0; i < n; i++) if (online[i]) { hSum += hubH[i]; hCnt++; }
  const zRef = hCnt > 0 ? hSum / hCnt : (hubH[0] || turb.hub_height);
  const rhoScale = Math.cbrt(airDensity / (turb.ref_air_density || 1.225));
  const pYaw = turb.cosine_loss_exponent_yaw || 1.88;

  // 3x3 rotor grids in sorted order: index (s * 9 + gy * 3 + gz)
  const yg = new Float64Array(n * 9);
  const zg = new Float64Array(n * 9);
  const uInit = new Float64Array(n * 9);
  const wakeAcc = new Float64Array(n * 9);
  const tiGrid = new Float64Array(n * 9).fill(tiInf);
  const uFreeRotorS = new Float64Array(n);

  for (let s = 0; s < n; s++) {
    const idx = order[s];
    const ys = yc[idx], hs = hubH[idx];
    let sumCube = 0;
    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < 3; b++) {
        const g = s * 9 + a * 3 + b;
        yg[g] = ys + D * REL3[a];
        const z = Math.max(hs + D * REL3[b], 1.0);
        zg[g] = z;
        const u0 = uInf * Math.pow(z / Math.max(zRef, 1.0), SHEAR);
        uInit[g] = u0;
        sumCube += u0 * u0 * u0;
      }
    }
    uFreeRotorS[s] = Math.cbrt(sumCube / 9);
  }

  const uRotorS = new Float64Array(n);
  const ctS = new Float64Array(n);
  const pKwS = new Float64Array(n);
  const pairDefS = new Float64Array(n * n);
  const pairOvlS = new Float64Array(n * n);
  const pairDelS = new Float64Array(n * n);
  const pairSyS = new Float64Array(n * n);

  for (let j = 0; j < n; j++) {
    const idxJ = order[j];
    let sumU3 = 0;
    for (let q = 0; q < 9; q++) {
      const g = j * 9 + q;
      const u = Math.max(uInit[g] - Math.sqrt(wakeAcc[g]), 0.0);
      sumU3 += u * u * u;
    }
    const uAvgJ = Math.cbrt(sumU3 / 9);
    uRotorS[j] = uAvgJ;

    if (!online[idxJ] || uAvgJ <= 0.1) continue;

    const yawJ = yawDeg[idxJ];
    const cosYaw = Math.max(Math.cos(yawJ * DEG), 0.0);
    const uEff = uAvgJ * rhoScale * Math.pow(cosYaw, pYaw / 3.0);
    pKwS[j] = interp1(uEff, turb.wind_speed, turb.power_kw, 0.0, 0.0);

    const ctRaw = clip(interp1(uAvgJ, turb.wind_speed, turb.thrust_coefficient, 0.0001, 0.0001), 0.0001, 0.9999);
    const ctJ = ctRaw * cosYaw;
    ctS[j] = ctJ;
    const aiJ = cosYaw > 1e-6 ? (0.5 / cosYaw) * (1.0 - Math.sqrt(Math.max(1.0 - ctJ * cosYaw, 0.0))) : 0.0;

    if (j === n - 1) continue;

    const xJ = xc[idxJ], yJ = yc[idxJ], hJ = hubH[idxJ];
    const yawOpp = -yawJ;
    const cosYo = Math.max(Math.cos(yawOpp * DEG), 1e-6);
    const ctCos = clip(ctJ * cosYo, 0.0, 0.9999);
    const sq1mCt = Math.sqrt(Math.max(1.0 - ctJ, 1e-8));
    const sq1mCtCos = Math.sqrt(Math.max(1.0 - ctCos, 1e-8));

    // Deflection constants
    const uRDef = ctCos / Math.max(2.0 * (1.0 - sq1mCtCos), 1e-8);
    const u0Rel = sq1mCt;
    const C0 = 1.0 - u0Rel;
    const M0 = Math.max(C0 * (2.0 - C0), 1e-8);
    const m0Sq = Math.sqrt(M0);
    const E0 = C0 * C0 - 3.0 * Math.exp(1.0 / 12.0) * C0 + 3.0 * Math.exp(1.0 / 3.0);
    const sz0Def = 0.5 * D * Math.sqrt(Math.max(uRDef / (1.0 + u0Rel), 1e-8));
    const sy0Def = Math.max(sz0Def * cosYo, 1e-6);
    const thetaC0 = DM * ((0.3 * (yawOpp * DEG)) / cosYo) * (1.0 - sq1mCtCos);
    const tanTheta = Math.tan(thetaC0);

    // Velocity deficit constants
    const uRVel = ctJ / Math.max(2.0 * (1.0 - sq1mCt), 1e-8);
    const sz0Vel = 0.5 * D * Math.sqrt(Math.max(uRVel / (1.0 + u0Rel), 1e-8));
    const sy0Vel = Math.max(sz0Vel * cosYo, 1e-6);
    const sCore = 0.501 * D * Math.sqrt(ctJ / 2.0);

    for (let i = j + 1; i < n; i++) {
      const dx = xc[order[i]] - xJ;
      if (dx <= 1e-4) continue;

      let duFracSum = 0;
      let ovlCnt = 0;
      let centerDefl = 0;
      let centerSy = 0;

      for (let q = 0; q < 9; q++) {
        const tiJq = tiGrid[j * 9 + q];
        const ky = KA * tiJq + KB;
        // 1. Gaussian deflection
        let defl = 0;
        if (Math.abs(yawJ) >= 1e-7) {
          const x0Def = (D * cosYo * (1.0 + sq1mCtCos)) / (Math.SQRT2 * Math.max(4.0 * ALPHA * tiJq + 2.0 * BETA * (1.0 - sq1mCt), 1e-6));
          const delta0 = tanTheta * x0Def;
          if (dx <= x0Def) {
            defl = (dx / Math.max(x0Def, 1e-6)) * delta0;
          } else {
            const syF = ky * (dx - x0Def) + sy0Def;
            const szF = ky * (dx - x0Def) + sz0Def;
            const midRatio = Math.sqrt((syF * szF) / Math.max(sy0Def * sz0Def, 1e-8));
            const lnNum = Math.max((1.6 + m0Sq) * (1.6 * midRatio - m0Sq), 1e-12);
            const lnDen = Math.max((1.6 - m0Sq) * (1.6 * midRatio + m0Sq), 1e-12);
            defl = delta0 + ((thetaC0 * E0) / 5.2) * Math.sqrt((sy0Def * sz0Def) / Math.max(ky * ky * M0, 1e-12)) * Math.log(lnNum / lnDen);
          }
        }

        // 2. Gaussian velocity deficit
        const x0Vel = (D * cosYo * (1.0 + sq1mCt)) / (Math.SQRT2 * Math.max(4.0 * ALPHA * tiJq + 2.0 * BETA * (1.0 - sq1mCt), 1e-6));
        let sy: number, sz: number;
        if (dx > 0.1 && dx < x0Vel) {
          const rUp = dx / Math.max(x0Vel, 1e-6);
          const rDn = 1.0 - rUp;
          sy = Math.max(rDn * sCore + rUp * sy0Vel, 1e-3);
          sz = Math.max(rDn * sCore + rUp * sz0Vel, 1e-3);
        } else if (dx >= x0Vel) {
          sy = Math.max(ky * (dx - x0Vel) + sy0Vel, 1e-3);
          sz = Math.max(ky * (dx - x0Vel) + sz0Vel, 1e-3);
        } else {
          sy = sy0Vel; sz = sz0Vel;
        }

        const gIdx = i * 9 + q;
        const dy = yg[gIdx] - yJ - defl;
        const dz = zg[gIdx] - hJ;
        const r2 = (dy * dy) / (2.0 * sy * sy) + (dz * dz) / (2.0 * sz * sz);
        const rad = clip(1.0 - (ctJ * cosYo) / ((8.0 * sy * sz) / (D * D)), 0.0, 1.0);
        const cAmp = 1.0 - Math.sqrt(rad);
        const du = dx > 0.1 ? uInit[gIdx] * cAmp * Math.exp(-r2) : 0.0;

        wakeAcc[gIdx] += du * du;
        duFracSum += du / Math.max(uInit[gIdx], 0.1);
        if (du > 0.05) ovlCnt++;
        if (q === 4) { centerDefl = defl; centerSy = sy; }
      }

      const areaOvl = ovlCnt / 9.0;
      pairDefS[j * n + i] = duFracSum / 9.0;
      pairOvlS[j * n + i] = areaOvl;
      pairDelS[j * n + i] = centerDefl / D;
      pairSyS[j * n + i] = centerSy / D;

      // 3. Crespo-Hernandez wake-added turbulence
      if (aiJ > 1e-6 && dx > 0.1 && dx <= 15.0 * D) {
        const wat = CH_CONST * Math.pow(aiJ, CH_AI) * Math.pow(Math.max(tiInf, 1e-4), CH_INIT) * Math.pow(dx / D, CH_DOWN);
        for (let q = 0; q < 9; q++) {
          const gIdx = i * 9 + q;
          if (Math.abs(yJ - yg[gIdx]) < 2.0 * D) {
            const tiAdd = areaOvl * wat;
            const cand = Math.sqrt(tiAdd * tiAdd + tiInf * tiInf);
            if (cand > tiGrid[gIdx]) tiGrid[gIdx] = cand;
          }
        }
      }
    }
  }

  // Un-permute back to original turbine index order
  const uRotor = new Float64Array(n);
  const uFree = new Float64Array(n);
  const ctOut = new Float64Array(n);
  const pOut = new Float64Array(n);
  const tiOut = new Float64Array(n);
  const pairDef = new Float64Array(n * n);
  const pairOvl = new Float64Array(n * n);
  const pairDel = new Float64Array(n * n);
  const pairSy = new Float64Array(n * n);

  for (let i = 0; i < n; i++) {
    const si = invOrder[i];
    uRotor[i] = uRotorS[si];
    uFree[i] = uFreeRotorS[si];
    ctOut[i] = ctS[si];
    pOut[i] = pKwS[si];
    let tSum = 0;
    for (let q = 0; q < 9; q++) tSum += tiGrid[si * 9 + q];
    tiOut[i] = tSum / 9.0;
    for (let k = 0; k < n; k++) {
      const sk = invOrder[k];
      pairDef[i * n + k] = pairDefS[si * n + sk];
      pairOvl[i * n + k] = pairOvlS[si * n + sk];
      pairDel[i * n + k] = pairDelS[si * n + sk];
      pairSy[i * n + k] = pairSyS[si * n + sk];
    }
  }

  return {
    x_m: xc, y_m: yc, u_rotor_ms: uRotor, u_free_ms: uFree, ti_eff: tiOut, ct: ctOut, power_kw: pOut,
    pair_deficit: pairDef, pair_overlap: pairOvl, pair_deflection_d: pairDel, pair_sigma_y_d: pairSy,
  };
}

export function read(
  W: Pack<WMeta>,
  east: ArrayLike<number>,
  north: ArrayLike<number>,
  elev: ArrayLike<number>,
  hubH: ArrayLike<number>,
  turbKey: string,
  uInf: number,
  windDirDeg: number,
  tiInf: number,
  sigmaWdDeg: number,
  yawDeg: ArrayLike<number>,
  online: boolean[],
  airDensity: number,
  pFrontKw: number | null = null,
): WakeReading {
  const n = east.length;
  const turb = W.meta.turbines[turbKey];
  const D = turb.rotor_diameter;
  const isFree = freestreamMask(east, north, D, windDirDeg, online);

  const anc0 = solveAnchor(east, north, hubH, turb, uInf, windDirDeg, tiInf, yawDeg, online, airDensity);
  const dTh = clip(sigmaWdDeg, 1.5, 18.0);
  const yawP = Float64Array.from({ length: n }, (_, i) => yawDeg[i] + 0.35 * dTh);
  const yawM = Float64Array.from({ length: n }, (_, i) => yawDeg[i] - 0.35 * dTh);
  const ancP = solveAnchor(east, north, hubH, turb, uInf, windDirDeg + dTh, tiInf, yawP, online, airDensity);
  const ancM = solveAnchor(east, north, hubH, turb, uInf, windDirDeg - dTh, tiInf, yawM, online, airDensity);

  const frontNorm = (pArr: Float64Array): Float32Array => {
    let sum = 0, cnt = 0;
    const useFree = isFree.some(Boolean);
    for (let i = 0; i < n; i++) {
      if (useFree ? isFree[i] : online[i]) { sum += pArr[i]; cnt++; }
    }
    const denom = cnt > 0 ? sum / cnt : 1.0;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      if (!online[i]) out[i] = 0;
      else out[i] = denom < 1.0 ? 1.0 : pArr[i] / denom;
    }
    return out;
  };

  const etaAnc = frontNorm(anc0.power_kw);
  const etaP = frontNorm(ancP.power_kw);
  const etaM = frontNorm(ancM.power_kw);

  let xMin = Infinity, hSum = 0, zSum = 0, onCnt = 0;
  for (let i = 0; i < n; i++) {
    if (online[i]) {
      if (anc0.x_m[i] < xMin) xMin = anc0.x_m[i];
      hSum += hubH[i];
      zSum += elev[i];
      onCnt++;
    }
  }
  if (onCnt === 0) {
    xMin = Math.min(...anc0.x_m);
    for (let i = 0; i < n; i++) { hSum += hubH[i]; zSum += elev[i]; }
    onCnt = n;
  }
  const hMean = hSum / onCnt, zMean = zSum / onCnt;
  const rhoScale = Math.cbrt(airDensity / (turb.ref_air_density || 1.225));
  const pFreeScalar = interp1(uInf * rhoScale, turb.wind_speed, turb.power_kw, 0, 0) / Math.max(turb.rated_power_kw, 1.0);
  const ctFreeScalar = interp1(uInf, turb.wind_speed, turb.thrust_coefficient, 0, 0);
  const pYawExp = turb.cosine_loss_exponent_yaw || 1.88;

  // Build node features [n, 17]
  const NF = 17, EF = 12;
  const nodeX = new Float32Array(n * NF);
  for (let i = 0; i < n; i++) {
    const b = i * NF;
    const cosY = Math.max(Math.cos(yawDeg[i] * DEG), 0.0);
    const cosLoss = 1.0 - Math.pow(cosY, pYawExp);
    const defAnc = online[i] ? 1.0 - anc0.u_rotor_ms[i] / Math.max(anc0.u_free_ms[i], 0.5) : 0.0;
    const tiAdd = online[i] ? (anc0.ti_eff[i] - tiInf) / 0.10 : 0.0;
    const dMeander = 0.5 * (etaP[i] + etaM[i]) - etaAnc[i];
    const sensWd = 0.5 * (Math.abs(etaP[i] - etaAnc[i]) + Math.abs(etaM[i] - etaAnc[i]));

    nodeX[b + 0] = online[i] ? 1 : 0;
    nodeX[b + 1] = isFree[i] ? 1 : 0;
    nodeX[b + 2] = (uInf - 8.5) / 3.0;
    nodeX[b + 3] = pFreeScalar;
    nodeX[b + 4] = ctFreeScalar;
    nodeX[b + 5] = (tiInf - 0.09) / 0.05;
    nodeX[b + 6] = (sigmaWdDeg - 5.0) / 4.0;
    nodeX[b + 7] = clip(yawDeg[i] / 25.0, -2.5, 2.5);
    nodeX[b + 8] = clip(cosLoss, 0.0, 1.0);
    nodeX[b + 9] = (hubH[i] - hMean) / D;
    nodeX[b + 10] = (elev[i] - zMean) / D;
    nodeX[b + 11] = (anc0.x_m[i] - xMin) / (10.0 * D);
    nodeX[b + 12] = etaAnc[i];
    nodeX[b + 13] = defAnc;
    nodeX[b + 14] = tiAdd;
    nodeX[b + 15] = dMeander * 4.0;
    nodeX[b + 16] = sensWd * 4.0;
  }

  // Build directed edge features [n * n, 12] for j -> i
  const edgeMask = new Uint8Array(n * n);
  const edgeX = new Float32Array(n * n * EF);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      if (i === j || !online[j] || !online[i]) continue;
      const dxD = (anc0.x_m[i] - anc0.x_m[j]) / D;
      const dyD = (anc0.y_m[i] - anc0.y_m[j]) / D;
      if (dxD > 0.10 && dxD < 30.0 && Math.abs(dyD) < 4.5 + 0.25 * Math.max(dxD, 0.0)) {
        const pIdx = j * n + i;
        edgeMask[pIdx] = 1;
        const eb = pIdx * EF;
        const pdef = anc0.pair_deficit[pIdx];
        const povl = anc0.pair_overlap[pIdx];
        const pdel = anc0.pair_deflection_d[pIdx];
        const psy = Math.max(anc0.pair_sigma_y_d[pIdx], 0.25);
        const absDyRel = Math.abs(dyD - pdel) / (psy + 0.35);
        const meanderPi = (sigmaWdDeg * DEG) * dxD;

        edgeX[eb + 0] = 1.0 / (dxD + 2.0);
        edgeX[eb + 1] = Math.log1p(dxD) / 3.0;
        edgeX[eb + 2] = clip(dyD / 3.0, -2.5, 2.5);
        edgeX[eb + 3] = clip(absDyRel / 3.0, 0.0, 3.0);
        edgeX[eb + 4] = clip((hubH[i] - hubH[j]) / D, -2.0, 2.0);
        edgeX[eb + 5] = clip((elev[i] - elev[j]) / D, -3.0, 3.0);
        edgeX[eb + 6] = pdef;
        edgeX[eb + 7] = povl;
        edgeX[eb + 8] = clip(pdel * 4.0, -2.5, 2.5);
        edgeX[eb + 9] = clip(yawDeg[j] / 25.0, -2.5, 2.5);
        edgeX[eb + 10] = anc0.ct[j];
        edgeX[eb + 11] = clip(meanderPi, 0.0, 3.0);
      }
    }
  }

  // Forward pass through EdgeWakeGNN
  const t = W.t;
  const H = W.meta.arch.hidden;
  const nLayers = W.meta.arch.n_layers;

  const h = new Float32Array(n * H);
  for (let i = 0; i < n; i++) {
    if (!online[i]) continue;
    const out = linear(nodeX.subarray(i * NF, (i + 1) * NF), t['node_enc.weight'], t['node_enc.bias']);
    for (let k = 0; k < H; k++) h[i * H + k] = geluScalar(out[k]);
  }

  const eEmb = new Float32Array(n * n * H);
  for (let pIdx = 0; pIdx < n * n; pIdx++) {
    if (!edgeMask[pIdx]) continue;
    const out = linear(edgeX.subarray(pIdx * EF, (pIdx + 1) * EF), t['edge_enc.weight'], t['edge_enc.bias']);
    for (let k = 0; k < H; k++) eEmb[pIdx * H + k] = geluScalar(out[k]);
  }

  const pairVec = new Float32Array(3 * H);
  const updVec = new Float32Array(3 * H);

  for (let l = 0; l < nLayers; l++) {
    const pref = `layers.${l}.`;
    const aggIn = new Float32Array(n * H);
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const pIdx = j * n + i;
        if (!edgeMask[pIdx]) continue;
        pairVec.set(h.subarray(j * H, (j + 1) * H), 0);
        pairVec.set(h.subarray(i * H, (i + 1) * H), H);
        pairVec.set(eEmb.subarray(pIdx * H, (pIdx + 1) * H), 2 * H);

        const mHid = linear(pairVec, t[pref + 'msg_fc1.weight'], t[pref + 'msg_fc1.bias']);
        for (let k = 0; k < H; k++) mHid[k] = geluScalar(mHid[k]);
        const msg = linear(mHid, t[pref + 'msg_fc2.weight'], t[pref + 'msg_fc2.bias']);
        const gLogit = linear(mHid, t[pref + 'gate_fc.weight'], t[pref + 'gate_fc.bias'])[0];
        const gate = sigmoidScalar(gLogit);
        const iBase = i * H;
        for (let k = 0; k < H; k++) aggIn[iBase + k] += gate * msg[k];
      }
    }

    const hFarm = new Float32Array(H);
    const denomOn = Math.max(onCnt, 1);
    for (let i = 0; i < n; i++) {
      if (!online[i]) continue;
      for (let k = 0; k < H; k++) hFarm[k] += h[i * H + k] / denomOn;
    }

    const hNext = new Float32Array(n * H);
    for (let i = 0; i < n; i++) {
      if (!online[i]) continue;
      updVec.set(h.subarray(i * H, (i + 1) * H), 0);
      updVec.set(aggIn.subarray(i * H, (i + 1) * H), H);
      updVec.set(hFarm, 2 * H);
      const u1 = linear(updVec, t[pref + 'upd_fc1.weight'], t[pref + 'upd_fc1.bias']);
      for (let k = 0; k < H; k++) u1[k] = geluScalar(u1[k]);
      const dh = linear(u1, t[pref + 'upd_fc2.weight'], t[pref + 'upd_fc2.bias']);
      for (let k = 0; k < H; k++) hNext[i * H + k] = h[i * H + k] + dh[k];
    }
    h.set(hNext);
  }

  // Node readout head
  const etaUnnorm = new Float32Array(n);
  const logSigEta = new Float32Array(n);
  const uDefPred = new Float32Array(n);
  const nhIn = new Float32Array(H + NF);

  let freeSum = 0, freeCnt = 0;
  for (let i = 0; i < n; i++) {
    nhIn.set(h.subarray(i * H, (i + 1) * H), 0);
    nhIn.set(nodeX.subarray(i * NF, (i + 1) * NF), H);
    const nh = linear(nhIn, t['node_head1.weight'], t['node_head1.bias']);
    for (let k = 0; k < nh.length; k++) nh[k] = geluScalar(nh[k]);
    const nout = linear(nh, t['node_head2.weight'], t['node_head2.bias']);
    logSigEta[i] = clip(nout[1], -5.0, -0.2);
    if (online[i]) {
      etaUnnorm[i] = etaAnc[i] + 0.5 * Math.tanh(nout[0]);
      uDefPred[i] = clip(nodeX[i * NF + 13] + 0.35 * Math.tanh(nout[2]), -0.15, 0.95);
      if (isFree[i]) { freeSum += etaUnnorm[i]; freeCnt++; }
    }
  }

  const fMean = Math.max(freeSum / Math.max(freeCnt, 1), 0.25);
  const etaPred = new Float32Array(n);
  let sumEtaOn = 0;
  for (let i = 0; i < n; i++) {
    if (online[i]) {
      etaPred[i] = clip(etaUnnorm[i] / fMean, 0.0, 1.35);
      sumEtaOn += etaPred[i];
    }
  }
  const denomOn = Math.max(onCnt, 1);
  const farmWakePred = 1.0 - sumEtaOn / denomOn;

  // Farm uncertainty head
  const hPoolMean = new Float32Array(H);
  const hWaked = new Float32Array(H);
  let wakedCnt = 0;
  for (let i = 0; i < n; i++) {
    if (online[i] && !isFree[i]) wakedCnt++;
  }
  const denomW = Math.max(wakedCnt, 1);
  for (let i = 0; i < n; i++) {
    if (!online[i]) continue;
    for (let k = 0; k < H; k++) {
      hPoolMean[k] += h[i * H + k] / denomOn;
      if (!isFree[i]) hWaked[k] += h[i * H + k] / denomW;
    }
  }
  const fhIn = new Float32Array(2 * H + 4);
  fhIn.set(hPoolMean, 0);
  fhIn.set(hWaked, H);
  fhIn[2 * H + 0] = farmWakePred;
  fhIn[2 * H + 1] = wakedCnt / denomOn;
  fhIn[2 * H + 2] = nodeX[2];
  fhIn[2 * H + 3] = nodeX[6];
  const fh = linear(fhIn, t['farm_head1.weight'], t['farm_head1.bias']);
  for (let k = 0; k < fh.length; k++) fh[k] = geluScalar(fh[k]);
  const logSigFarm = clip(linear(fh, t['farm_head2.weight'], t['farm_head2.bias'])[0], -5.5, -0.5);

  const sf = W.meta.sigma_factor;
  const etaSig = new Float32Array(n);
  const wakeLoss = new Float32Array(n);
  const powerKw = new Float32Array(n);
  const powerSigKw = new Float32Array(n);
  const uRotorMs = new Float32Array(n);

  const pRef = pFrontKw ?? interp1(uInf * rhoScale, turb.wind_speed, turb.power_kw, 0, 0);
  let farmP = 0;
  for (let i = 0; i < n; i++) {
    if (!online[i]) continue;
    const fScale = isFree[i] ? sf.free_eta : sf.waked_eta;
    etaSig[i] = Math.exp(logSigEta[i]) * fScale;
    wakeLoss[i] = 1.0 - etaPred[i];
    powerKw[i] = etaPred[i] * pRef;
    powerSigKw[i] = etaSig[i] * pRef;
    uRotorMs[i] = anc0.u_free_ms[i] * (1.0 - uDefPred[i]);
    farmP += powerKw[i];
  }

  return {
    eta: etaPred,
    eta_sig: etaSig,
    wake_loss: wakeLoss,
    power_kw: powerKw,
    power_sig_kw: powerSigKw,
    u_rotor_ms: uRotorMs,
    eta_anchor: etaAnc,
    is_free: isFree,
    online: [...online],
    p_front_kw: pRef,
    farm_power_kw: farmP,
    farm_free_kw: onCnt * pRef,
    farm_wake_loss: farmWakePred,
    farm_wake_sig: Math.exp(logSigFarm) * sf.farm_wake,
    x_m: anc0.x_m,
    y_m: anc0.y_m,
    ct_anchor: anc0.ct,
    ti_anchor: anc0.ti_eff,
  };
}

/** Fast 2-D hub-height velocity-deficit field for rendering live wake plumes on the scope canvas. */
export function wakePlumeGrid(
  east: ArrayLike<number>,
  north: ArrayLike<number>,
  turb: TurbineSpec,
  windDirDeg: number,
  tiInf: number,
  yawDeg: ArrayLike<number>,
  online: boolean[],
  ct: ArrayLike<number>,
  tiEff: ArrayLike<number>,
  eMin: number,
  eMax: number,
  nMin: number,
  nMax: number,
  cols: number,
  rows: number,
): Float32Array {
  const n = east.length;
  const D = turb.rotor_diameter;
  let eminT = Infinity, emaxT = -Infinity, nminT = Infinity, nmaxT = -Infinity;
  for (let i = 0; i < n; i++) {
    if (east[i] < eminT) eminT = east[i];
    if (east[i] > emaxT) emaxT = east[i];
    if (north[i] < nminT) nminT = north[i];
    if (north[i] > nmaxT) nmaxT = north[i];
  }
  const ec = 0.5 * (eminT + emaxT), nc = 0.5 * (nminT + nmaxT);
  const th = windDirDeg * DEG;
  const s = Math.sin(th), c = Math.cos(th);

  const tx = new Float64Array(n), ty = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const de = east[i] - ec, dn = north[i] - nc;
    tx[i] = -de * s - dn * c + ec;
    ty[i] = de * c - dn * s + nc;
  }

  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const nCoord = nMin + ((r + 0.5) / rows) * (nMax - nMin);
    const dn = nCoord - nc;
    for (let q = 0; q < cols; q++) {
      const eCoord = eMin + ((q + 0.5) / cols) * (eMax - eMin);
      const de = eCoord - ec;
      const gx = -de * s - dn * c + ec;
      const gy = de * c - dn * s + nc;

      let defSq = 0;
      for (let j = 0; j < n; j++) {
        if (!online[j] || ct[j] <= 0.01) continue;
        const dx = gx - tx[j];
        if (dx <= 0) continue;
        const ctJ = ct[j];
        const yawOpp = -yawDeg[j];
        const cosYo = Math.max(Math.cos(yawOpp * DEG), 0.2);
        const sq1mCt = Math.sqrt(Math.max(1.0 - ctJ, 1e-6));
        const tiJ = tiEff[j] || tiInf;
        const ky = KA * tiJ + KB;
        const x0 = (D * cosYo * (1.0 + sq1mCt)) / (Math.SQRT2 * Math.max(4.0 * ALPHA * tiJ + 2.0 * BETA * (1.0 - sq1mCt), 1e-4));
        const sy0 = 0.35 * D * cosYo;
        const sy = dx < x0 ? sy0 : sy0 + ky * (dx - x0);
        const thetaC0 = ((0.3 * (yawOpp * DEG)) / cosYo) * (1.0 - Math.sqrt(Math.max(1.0 - ctJ * cosYo, 0.0)));
        const defl = Math.tan(thetaC0) * Math.min(dx, x0) + (dx > x0 ? thetaC0 * 2.2 * D * Math.log1p((dx - x0) / (3.0 * D)) : 0);
        const dy = gy - ty[j] - defl;
        if (Math.abs(dy) > 3.5 * sy) continue;
        const rad = clip(1.0 - (ctJ * cosYo) / ((8.0 * sy * sy) / (D * D)), 0.0, 1.0);
        const cAmp = 1.0 - Math.sqrt(rad);
        const du = cAmp * Math.exp(-(dy * dy) / (2.0 * sy * sy));
        defSq += du * du;
      }
      out[r * cols + q] = Math.min(Math.sqrt(defSq), 0.75);
    }
  }
  return out;
}

export function parityVector(r: WakeReading): number[] {
  return [
    ...Array.from(r.eta_anchor),
    ...Array.from(r.eta),
    ...Array.from(r.eta_sig),
    ...Array.from(r.u_rotor_ms),
    r.farm_wake_loss,
    r.farm_wake_sig,
  ];
}
