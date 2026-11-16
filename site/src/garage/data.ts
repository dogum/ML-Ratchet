// The garage's data layer: decode the packs written by shared/vehicle/garage/export_garage.py and build each network's
// input for one cell (vehicle x route x air temperature x load) exactly as the exporter's reference does. The three
// networks are the instruments' own ports, imported, not copied. No DOM, so the parity test runs it in Node.
import { predict, perturb, FIELDS, type EMeta, type Prediction, type Trip, type Vehicle } from '../instruments/energynet/net.ts';
import { run as tdRunNet, reading as tdReading, NRAW, type Flags, type Raw, type Reading, type TMeta, type Heads } from '../instruments/tripdetective/net.ts';
import { forward, sOf, pOn, equivalence, type Net, type Policy, type Trip as HPTrip } from '../instruments/hybridpilot/net.ts';
import type { Pack } from '../lib/nn/pack.ts';

export interface RouteInfo {
  id: string; kind: 'ved' | 'ztbus'; n: number; km: number; minutes: number; T_mean: number; vmax_kmh: number; v_mean_kmh: number;
  fast_share: number; stops: number; climb_m: number; elev_range_m: number; date: string; dT: number[]; place: string;
  line?: string; passengers?: number; season?: string; real_kWh?: number; vehicle?: string;
}
export interface Truth { mass: number; CdA: number; Crr: number; disp: number | null; battery_kWh: number | null; P_kW: number | null; arch: string; curb: number }
export interface VehInfo {
  id: string; name: string; short: string; type: string; likely: string; source: string; arch: string; cls: string;
  en: Vehicle; routes: number[]; soc0: number | null; truth: Truth; logger: Record<string, boolean>;
}
export interface GMeta {
  physics: string; temps: number[]; loads: Record<string, number[]>; routes: RouteInfo[]; vehicles: VehInfo[];
  q: Record<string, number>; xcol: number[]; lhv_kJ_g: number; td_ledger: string[]; policy: Policy; note: string;
}
export interface HPRun { fuel_g: number; E_kWh: number; soc_end: number; pct_dp: number; extrap: boolean; starts_km: number; dist_km: number }
export interface Ledger {
  fuel: number; batt_chem_out: number; aero: number; rolling: number; grade: number; brake: number; eng_loss: number; mot_loss: number;
  trans_loss: number; batt_loss: number; aux: number; kinetic_change: number; residual: number; closure_rel: number;
}
export interface Cell {
  ri: number; ti: number; li: number; n: number; strategy: string; E_MJ: number; fuel_g: number; batt_kWh: number; km: number;
  soc_end: number | null; ledger: Ledger; hvac_kJ: number; trace_rms: number; T_cabin_end: number; starts: number; on_share: number;
  flags?: number[]; td_ledger: number[]; mass_true: number;
  hp?: Record<'dp' | 'hybridpilot' | 'aecms' | 'rule', HPRun>; hp_const?: number[]; hp_meta?: { ref: number; phev: boolean; soc_lo: number; soc_hi: number; prior_key?: string };
}
export interface VMeta { id: string; cells: Record<string, Cell> }

export const tag = (ri: number, ti: number, li: number) => `c${ri}_${ti}_${li}`;
const TD_FLAGS = ['has_fuel', 'fuel_est', 'has_rpm', 'has_batt', 'has_aux'];
const TD_ROW: [string, number][] = [['fuel', 5], ['rpm', 6], ['batt', 7], ['soc', 8], ['aux', 9]];

// ---------------------------------------------------------------------------------------------- packed formats
/** int16 values stored as little-endian byte pairs in a u8 tensor */
export function int16(b: ArrayLike<number>): Int16Array {
  const n = b.length >> 1, o = new Int16Array(n);
  for (let i = 0; i < n; i++) { const u = b[2 * i] + 256 * b[2 * i + 1]; o[i] = u >= 32768 ? u - 65536 : u; }
  return o;
}
/** numpy packbits(bitorder='little') -> 0/1 per second */
export function unbits(b: ArrayLike<number>, n: number): Uint8Array {
  const o = new Uint8Array(n);
  for (let i = 0; i < n; i++) o[i] = (b[i >> 3] >> (i & 7)) & 1;
  return o;
}

// ---------------------------------------------------------------------------------------------- the route and the controls
/** The real route as stored (float32), in EnergyNet's trip fields. */
export function routeTrip(R: Pack<GMeta>, ri: number): Trip {
  const t = {} as Record<string, Float32Array>;
  for (const k of FIELDS) t[k] = R.t[`r${ri}_${k}`].data;
  return t as unknown as Trip;
}
/** The trip every model sees for an air-temperature setting: EnergyNet's own perturb(dT) (air density at constant pressure). */
export function cellTrip(R: Pack<GMeta>, ri: number, ti: number): Trip {
  const dT = R.meta.routes[ri].dT[ti];
  return perturb(routeTrip(R, ri), {} as Vehicle, dT ? { dT } : {}).trip;
}
const ONE = Object.fromEntries(FIELDS.map((k) => [k, [0]])) as unknown as Trip;
/** EnergyNet's vehicle record for a load level: the record's mass scaled by EnergyNet's own mass control. */
export function enVehicle(G: GMeta, v: VehInfo, li: number): Vehicle {
  const load = G.loads[v.cls][li];
  const m = v.en.mass_kg;
  if (!load || !m) return { ...v.en };
  return perturb(ONE, v.en, { mass: (m + load) / m }).veh;
}

// ---------------------------------------------------------------------------------------------- EnergyNet
export function enRun(W: Pack<EMeta>, R: Pack<GMeta>, v: VehInfo, ri: number, ti: number, li: number): Prediction {
  return predict(W, cellTrip(R, ri, ti), enVehicle(R.meta, v, li));
}
/** what the exporter stores: energy, log sigma, physics estimate, value, lo68, hi68 (display unit), 5 components (MJ) */
export const enOutputs = (r: Prediction) => [r.energy_MJ, r.sigma_log, r.physics_estimate_MJ, r.value, r.lo68, r.hi68, ...r.components_MJ];

// ---------------------------------------------------------------------------------------------- TripDetective
/** TripDetective's raw log of one cell: the route channels of the trip as driven, the logger's channels decoded. */
export function tdRaw(G: GMeta, P: Pack<VMeta>, v: VehInfo, ri: number, ti: number, li: number, trip: Trip): { raw: Raw; flags: Flags } {
  const n = trip.v.length, tg = tag(ri, ti, li);
  const rows = Array.from({ length: NRAW }, () => new Float32Array(n));
  for (let i = 0; i < n; i++) {
    rows[0][i] = trip.v[i]; rows[1][i] = trip.grade[i] * 100; rows[2][i] = trip.T_amb[i];
    rows[3][i] = trip.wind_speed[i] * Math.cos((trip.wind_dir[i] - trip.heading[i]) * (Math.PI / 180));
    rows[4][i] = trip.rho_air[i]; rows[10][i] = 1;
  }
  if (v.cls === 'bus') { // TripDetectiveBus.raw(): power / 12 shown as a car's battery power, SOC held at 0.5
    const q = int16(P.t[`${tg}_q_pbus`].data), s = G.q.pbus;
    for (let i = 0; i < n; i++) { const pw = q[i] / s; rows[7][i] = ((pw / 1000 / 12) * 1000) / 1000; rows[3][i] = 0; rows[8][i] = 0.5; }
    return { raw: { rows, n }, flags: [0, 0, 0, 1, 0] };
  }
  for (const [k, r] of TD_ROW) {
    const t = P.t[`${tg}_q_${k}`];
    if (!t) continue;
    const q = int16(t.data), s = G.q[k];
    for (let i = 0; i < n; i++) rows[r][i] = q[i] / s;
  }
  return { raw: { rows, n }, flags: TD_FLAGS.map((f) => (v.logger[f] ? 1 : 0)) };
}

export interface TDResult { h: Heads; reading: Reading; bus?: { mass: number; lo: number; hi: number }; windows: number }
/** Read one car from one or several trips (every window pooled), as read_car(); the bus network as read_leg(). */
export function tdRead(W: Pack<TMeta>, trips: { raw: Raw; flags: Flags }[], bus = false): TDResult {
  const out = tdRunNet(W, trips);
  const reading = tdReading(W, out, trips.length);
  const r: TDResult = { h: out.h, reading, windows: out.tripOf.length };
  if (bus) {
    const sc = (W.meta as TMeta & { scale?: number }).scale ?? 12, k = (W.meta.sigk as Record<string, number>).mass_bus ?? 1;
    const m = 1800 * Math.exp(out.h.mu[0]) * sc, s = out.h.sig[0] * k;
    r.bus = { mass: m, lo: m * Math.exp(-s), hi: m * Math.exp(s) };
  }
  return r;
}

// ---------------------------------------------------------------------------------------------- HybridPilot
export interface HPInputs { k: Int32Array; X: Float64Array[]; soc: Float32Array; I: Float32Array }
/** HybridPilot's 66 inputs at every decision second of its own run: vehicle constants, the stored changing columns,
 *  zeros for the preview and navigation inputs (off). */
export function hpInputs(G: GMeta, P: Pack<VMeta>, c: Cell, tg: string): HPInputs {
  const xs = P.t[`${tg}_hp_x`], [m, F] = xs.shape, mask = unbits(P.t[`${tg}_hp_mask`].data, c.n);
  const k = new Int32Array(m);
  let j = 0;
  for (let i = 0; i < c.n; i++) if (mask[i]) k[j++] = i;
  const X: Float64Array[] = [];
  for (let r = 0; r < m; r++) {
    const x = new Float64Array(66);
    for (let q = 0; q < 12; q++) x[q] = Math.fround(c.hp_const![q]);
    for (let q = 0; q < F; q++) x[G.xcol[q]] = xs.data[r * F + q];
    X.push(x);
  }
  return { k, X, soc: P.t[`${tg}_hp_soc`].data, I: P.t[`${tg}_hp_I`].data };
}
export interface HPTrace { k: Int32Array; y0: Float64Array; y1: Float64Array; sNet: Float64Array; s: Float64Array; pon: Float64Array }
/** the network over the whole trip: raw outputs, s (equivalence factor) with the SOC feedback, P(engine on) */
export function hpRun(net: Net, G: GMeta, c: Cell, inp: HPInputs): HPTrace {
  const n = inp.X.length, o: HPTrace = { k: inp.k, y0: new Float64Array(n), y1: new Float64Array(n), sNet: new Float64Array(n), s: new Float64Array(n), pon: new Float64Array(n) };
  const trip = { ref: c.hp_meta!.ref, phev: c.hp_meta!.phev, prior_key: c.hp_meta!.prior_key } as HPTrip;
  for (let i = 0; i < n; i++) {
    const y = forward(net, inp.X[i]);
    o.y0[i] = y[0]; o.y1[i] = y[1]; o.sNet[i] = sOf(y); o.pon[i] = pOn(y);
    o.s[i] = equivalence(G.policy, trip, o.sNet[i], inp.soc[i], inp.I[i]);
  }
  return o;
}
