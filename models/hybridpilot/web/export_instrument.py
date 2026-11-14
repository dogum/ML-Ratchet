"""Export HybridPilot's live instrument for the website.

Three real VED exam routes (models/hybridpilot/examples/), each driven by RoadSim's simulated twin of the car that
drove it. Two parts:

  1. Trips (precomputed here, RoadSim is Python): every strategy on each route — rule-based, ECMS, A-ECMS, MPC,
     HybridPilot and the DP optimum — plus variants the page's sliders snap between (ambient temperature, cold start,
     vehicle mass, a navigation preview for HybridPilot). Graded as % of DP at matched final charge (hp_score.grade,
     the exam's own grading).
  2. Decisions (live in the browser): at recorded moments of HybridPilot's own run, the 66-feature vector, the
     candidate controls RoadSim scored that second (on a grid of wheel-power demand x coolant temperature, battery
     limits left out so the browser applies them for any SOC), the battery model and the decision rule's constants.
     The browser rebuilds the features for the SOC, demand, coolant temperature and preview you set, runs the MLP and
     picks the control. Also every traction second of the base run, so the page can run the network over the trip.

    python models/hybridpilot/web/export_instrument.py      # writes site/public/instruments/hybridpilot/
"""
import copy
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(MODEL))
sys.path.insert(0, MODEL)
sys.path.insert(0, os.path.join(MODEL, 'training'))
sys.path.insert(0, REPO)

import pipeline as PL  # noqa: E402  (puts RoadSim on the path)
import hybridpilot as HP  # noqa: E402
import hp_score as S  # noqa: E402  (DP curve, matched-SOC grading)
from roadsim import simulate  # noqa: E402
from roadsim.sim import get_vehicle  # noqa: E402

from mlratchet.webexport import instrument_dir, roundtrip, write_json, write_pack  # noqa: E402

OUT = instrument_dir('hybridpilot')
STRATS = ['rule', 'ecms', 'aecms', 'mpc', 'hybridpilot', 'dp']
TEMPS = [-10, 0, 10, 25, 35]
MASSES = [0.9, 1.2, 1.4]
PREVIEWS = [10, 30, 60]
N_MOMENTS = 12
D_FRACS = [0.03, 0.08, 0.15, 0.25, 0.4, 0.6]      # wheel-power demand levels, fraction of engine max power
T_LEVELS = [5.0, 20.0, 30.0, 45.0, 60.0, 75.0]    # coolant levels (C); the recorded value is added
SHORT = {'VED_125_1275': 'Camry-type hybrid', 'VED_428_465': 'Camry-type hybrid', 'VED_492_2826': 'Volt-type plug-in'}


# ------------------------------------------------------------------------------------------------ capture
def capture_strategy(**params):
    """HybridPilot exactly as pipeline.compare runs it, recording each traction decision and the candidate
    evaluation RoadSim made for it."""
    base = type(HP.make_roadsim_strategy())

    class Capture(base):
        def setup(self, veh, pt, trip, st):
            super().setup(veh, pt, trip, st)
            self.cap = []
            self.ev = pt.evaluate
            ev = pt.evaluate

            def wrap(T_w, w_w, ctrl, st_, env):
                if np.ndim(T_w) == 0:          # (the traction-limited fallback evaluates arrays; not a decision)
                    self._last = (float(T_w), float(w_w), np.array(ctrl, float), dict(st_), dict(env))
                return ev(T_w, w_w, ctrl, st_, env)
            pt.evaluate = wrap

        def select(self, C, ctrl, st, ctx):
            j = super().select(C, ctrl, st, ctx)
            cat = float(st['T_eng'] < 35.0 and self.t_cat < self.cat_warm_s)
            x = HP.features(self.vf, self.tc, self.vp, st, ctx, self.core.p['preview_s'], self.core.p['nav'], self.ref, cat)
            self.cap.append(dict(k=int(ctx['k']), st=dict(st), ctx=dict(ctx), C={a: np.array(b) for a, b in C.items()},
                                 ev=self._last, t_cat=float(self.t_cat), j=int(j), x=x, s=self.trace[-1][1], pon=self.trace[-1][2],
                                 forced=bool(self.forced_on(st, ctx))))
            return j

    return Capture(**params)


def heat_high(strat, st, k):
    """the part of RoadSim's need_heat that does not depend on the coolant: big cabin heat demand on an engine-heated cabin."""
    cab = strat.veh.cabin
    if cab.source != 'engine':
        return False
    Qd = cab.demand(st['T_cab'], float(strat.trip['T_amb'][k]), float(strat.trip['solar'][k]))
    return bool(Qd > 0.3 * cab.Q_heat_max)


def moment_grid(strat, c):
    """candidate controls for this second on a demand x coolant grid; battery power limits left out (ok_mech)."""
    T_w, w_w, ctrl, st, env = c['ev']
    u = c['ctx']['u']; r = strat.veh.r
    Pe = strat.vp['P_eng_max']
    P_rec = c['ctx']['P_dem_mech']
    assert abs(T_w - P_rec / u * r) < 1e-6 * max(1.0, abs(T_w)), (T_w, P_rec / u * r)
    D = sorted(set([round(f * Pe, 1) for f in D_FRACS] + [P_rec]))
    T = sorted(set(T_LEVELS + [float(st['T_eng'])]))
    env0 = dict(env, p_dis=np.inf, p_chg=np.inf)
    n = len(ctrl)
    PF = np.zeros((len(D), len(T), n)); PB = np.zeros_like(PF); OK = np.zeros(PF.shape, bool); ON = np.zeros(PF.shape, bool)
    for a, P in enumerate(D):
        for b, Te in enumerate(T):
            C = strat.ev(P / u * r, w_w, ctrl, dict(st, T_eng=Te), env0)
            PF[a, b] = C['P_fuel']; PB[a, b] = C['P_batt']; OK[a, b] = C['ok']; ON[a, b] = C['on'] & (C['P_fuel'] > 0)
    return dict(D=D, T=T, d_rec=D.index(P_rec), t_rec=T.index(float(st['T_eng'])), PF=PF, PB=PB, OK=OK, ON=ON)


# ------------------------------------------------------------------------------------------------ decision (reference)
def decide(core, strat, mo, g, soc, d, t, L, cand=None):
    """HybridPilot's decision at a recorded moment with SOC, demand level d, coolant level t and preview L changed:
    the model's own features() and HybridPilotCore.decide(); candidates as exported (float32)."""
    st = dict(mo['st'], soc=soc, T_eng=g['T'][t])
    b = strat.veh.battery
    p_dis, p_chg = b.limits(soc, st['T_bat'], st['v_rc'])
    need_heat = mo['heat_high'] and g['T'][t] < 55.0
    ctx = dict(mo['ctx'], p_dis=p_dis, p_chg=p_chg, P_dem_mech=g['D'][d], need_heat=need_heat)
    cat_pending = g['T'][t] < 35.0 and mo['t_cat'] < strat.cat_warm_s
    x = HP.features(strat.vf, strat.tc, strat.vp, st, ctx, L, False, strat.ref, float(cat_pending))
    PF, PB, OKm, ON = cand if cand is not None else (g['PF'][d, t], g['PB'][d, t], g['OK'][d, t], g['ON'][d, t])
    ok = OKm & (PB <= p_dis + 1e-6) & (PB >= -p_chg - 1e-6)
    I, _ = b.current_vec(PB, soc, st['T_bat'], st['v_rc'])
    chem = b.ocv(soc) * I
    forced = need_heat or (cat_pending and ctx['moving']) or soc < strat.soc_lo - 0.02
    j, s, pon = core.decide(x, soc, strat.ref, strat.phev, PF, chem, ON, ok, st['eng_on'], st['t_on'], st['t_off'],
                            strat.start_W, forced)
    return j, s, pon, x


# ------------------------------------------------------------------------------------------------ trips
def soc_start(ex, trip):
    if not ex['arch'].startswith('phev'):
        return 0.6
    s = trip['soc'][np.isfinite(trip['soc'])]
    s0 = float(s[0])
    return s0 / 100 if s0 > 1.5 else s0


def variants():
    V = [dict(tag='base')]
    V += [dict(tag=f'T{T}', T=T) for T in TEMPS]
    V += [dict(tag='cold', cold=True)]
    V += [dict(tag=f'm{m}', mass=m) for m in MASSES]
    return V


def apply(spec, trip, v):
    """as training/lab.py apply(): constant ambient temperature, or payload added for a mass factor."""
    spec = copy.deepcopy(spec); trip = copy.deepcopy(trip)
    if 'T' in v:
        trip['T_amb'] = np.full(len(trip['v']), float(v['T']))
        trip.pop('rho_air', None)
    if 'mass' in v:
        m0 = get_vehicle(spec).mass
        spec['payload_kg'] = float(spec.get('payload_kg', 0.0) or 0.0) + (v['mass'] - 1.0) * m0
        spec['id'] = spec['id'] + f"_m{v['mass']}"
    return spec, trip


def run_trip(spec, trip, soc0, v):
    """every strategy on one variant, graded against DP at matched final charge (as hp_score.run_case)."""
    spec, trip = apply(spec, trip, v)
    temps = None if v.get('cold') else dict(PL.WARM)
    veh = get_vehicle(spec)
    phev = veh.arch.startswith('phev')
    pts, outs, stage = S.dp_curve(spec, trip, soc0, temps, S.dp_targets(veh, soc0, phev), veh=veh)
    nat = soc0 if not (phev and soc0 > veh.battery.soc_lo + 0.08) else veh.battery.soc_lo + 0.05
    jn = int(np.argmin([abs(p['target'] - nat) for p in pts]))
    runs = {'dp': (outs[pts[jn]['target']][0], dict(pts[jn]))}
    pols = {'hybridpilot': {}}
    if v['tag'] == 'base':
        pols.update({f'hybridpilot_p{p}': dict(preview_s=p) for p in PREVIEWS})
    cap = None
    for nm in ['rule', 'ecms', 'aecms', 'mpc'] + list(pols):
        if nm in pols:
            stg = capture_strategy(**pols[nm]) if (nm == 'hybridpilot' and v['tag'] == 'base') else HP.make_roadsim_strategy(**pols[nm])
        else:
            stg = PL.RULE[veh.arch] if nm == 'rule' else nm
        o = simulate(spec, trip, strategy=stg, init_soc=soc0, init_temps=temps, record=True, options='roadsim-physics-v1')
        runs[nm] = (o, S.metrics(o, veh.arch, soc0))
        if nm == 'hybridpilot' and v['tag'] == 'base':
            cap = stg
    for nm, (o, m) in runs.items():
        if nm != 'dp':
            S.add_matched(dict(spec=spec, trip=trip, soc0=soc0, temps=temps), veh, stage, pts, m['soc_end'])
    for nm, (o, m) in runs.items():
        S.grade(m, pts, S.LHV.get(veh.fuel, 42.9e3) / 1e3, soc0)
    return runs, cap, veh


def main():
    t00 = time.time()
    npz = np.load(HP.WEIGHTS)
    W = {k: npz[k] for k in npz.files}
    keep = {'mu': 'f32', 'sd': 'f32'}             # feature scaling stays float32; the layers go float16
    write_pack(OUT / 'weights', W, dtype='f16', keep=keep,
               meta=dict(layers=[66, 96, 96, 2], act='tanh', params=int(sum(W[k].size for k in W if k[0] in 'Wb')),
                         out=['log(s + 0.05)', 'logit P(engine on)']))
    Wr = {k: (roundtrip({k: v}, keep.get(k, 'f16'))[k]).astype(np.float64) for k, v in W.items()}
    net_ref = HP.MLP(); net_ref.p = Wr
    core_ref = HP.HybridPilotCore(net=net_ref)
    core_f64 = HP.HybridPilotCore()

    index = json.load(open(os.path.join(MODEL, 'examples', 'index.json')))
    tens, trips_meta, cases = {}, [], []
    check = dict(recorded=0, f64_same=0, f16_same=0)
    for ti, ex in enumerate(index):
        tid = ex['trip']
        spec, trip = PL.load_example(tid)
        soc0 = soc_start(ex, trip)
        N = len(trip['v'])
        runs_meta, ON, SOC = [], [], []
        strat = None
        for v in variants():
            t0 = time.time()
            runs, cap, veh = run_trip(spec, trip, soc0, v)
            if cap is not None:
                strat, veh0 = cap, veh
            for nm, (o, m) in runs.items():
                runs_meta.append(dict(v=v['tag'], s=nm, fuel_g=float(o['summary']['fuel_g']), E_kWh=float(m['E_kJ']) / 3600,
                                      soc_end=float(o['soc'][-1]), pct_dp=float(m['pct_dp']), extrap=bool(m['extrap']),
                                      starts_km=float(m['starts_per_km']), L100=float(o['summary'].get('L_per_100km') or 0.0),
                                      dist_km=float(m['dist_km'])))
                ON.append((np.asarray(o['eng_on']) > 0.5).astype(np.uint8)[:N])
                SOC.append(np.asarray(o['soc'], float)[:N])
            print(f"{tid} {v['tag']:5s} " + ' '.join(f"{k}={runs[k][1]['pct_dp'] * 100:.2f}" for k in runs) + f"  ({time.time() - t0:.1f} s)", flush=True)
        tens[f't{ti}_on'] = np.stack(ON); tens[f't{ti}_soc'] = np.stack(SOC)
        tens[f't{ti}_v'] = np.asarray(trip['v'], float); tens[f't{ti}_grade'] = np.asarray(trip['grade'], float)

        # ---- decisions: the base HybridPilot run, every traction second, and N_MOMENTS moments in detail
        veh = veh0; b = veh.battery
        caps = [c for c in strat.cap if c['ctx']['moving'] and c['ctx']['u'] > 3.0 and c['ctx']['P_dem_mech'] > 500.0]
        tens[f't{ti}_xk'] = np.array([c['k'] for c in strat.cap], np.int32)
        tens[f't{ti}_xs'] = np.stack([c['x'] for c in strat.cap]).astype(np.float32)
        tens[f't{ti}_xsoc'] = np.array([c['st']['soc'] for c in strat.cap], np.float32)
        tens[f't{ti}_xsrec'] = np.array([c['s'] for c in strat.cap], np.float32)
        pick = np.unique(np.linspace(0.04 * len(caps), 0.96 * len(caps), N_MOMENTS).astype(int))
        moments = []
        PF, PB, OK, ONc = [], [], [], []
        for mi, ci in enumerate(pick):
            c = caps[ci]
            st, ctx = c['st'], c['ctx']
            mo = dict(st=st, ctx=ctx, t_cat=c['t_cat'], heat_high=heat_high(strat, st, c['k']))
            if st['T_eng'] < 55.0:
                assert mo['heat_high'] == bool(ctx['need_heat']), (tid, c['k'])
            g = moment_grid(strat, c)
            d, t = g['d_rec'], g['t_rec']
            # the grid at the recorded point reproduces RoadSim's own candidate evaluation
            assert np.allclose(g['PF'][d, t], c['C']['P_fuel']) and np.allclose(g['PB'][d, t], c['C']['P_batt'])
            ok_rec = g['OK'][d, t] & (g['PB'][d, t] <= ctx['p_dis'] + 1e-6) & (g['PB'][d, t] >= -ctx['p_chg'] - 1e-6)
            assert (ok_rec == c['C']['ok']).all()
            # and the decision rule, fed the model's own features, picks what RoadSim recorded
            j64, _, _, x64 = decide(core_f64, strat, mo, g, st['soc'], d, t, 0)
            assert np.allclose(x64, c['x']) and j64 == c['j'], (tid, c['k'], j64, c['j'])
            f32 = tuple(np.asarray(a, np.float32).astype(np.float64) if a.dtype != bool else a
                        for a in (g['PF'][d, t], g['PB'][d, t], g['OK'][d, t], g['ON'][d, t]))
            j16, _, _, _ = decide(core_ref, strat, mo, g, st['soc'], d, t, 0, cand=f32)
            check['recorded'] += 1; check['f64_same'] += int(j64 == c['j']); check['f16_same'] += int(j16 == c['j'])
            PF.append(g['PF']); PB.append(g['PB']); OK.append(g['OK']); ONc.append(g['ON'])
            k = c['k']
            moments.append(dict(
                k=k, v_kmh=float(trip['v'][k] * 3.6), u=float(ctx['u']), soc=float(st['soc']), T_eng=float(st['T_eng']),
                T_bat=float(st['T_bat']), v_rc=float(st['v_rc']), eng_on=bool(st['eng_on']), t_on=float(st['t_on']),
                t_off=float(st['t_off']), t_cat=float(c['t_cat']), moving=bool(ctx['moving']), heat_high=mo['heat_high'],
                P_aux=float(ctx['P_aux_el']), x0=[float(q) for q in c['x']],
                prev60=[float(q) for q in strat.tc.preview(min(k, strat.tc.N - 1), 60)],
                D=[float(q) for q in g['D']], T=[float(q) for q in g['T']], d_rec=d, t_rec=t,
                j_rec=c['j'], s_rec=float(c['s']), pon_rec=float(c['pon'])))
            # ---- parity cases (first four moments): as recorded and perturbed
            if mi % 3 == 0:
                lo, hi = b.soc_lo, b.soc_hi
                var = [('rec', st['soc'], d, t, 0), ('soc_low', lo + 0.02, d, t, 0), ('soc_high', hi - 0.03, d, t, 0),
                       ('preview30', st['soc'], d, t, 30), ('preview60_soc_mid', 0.5 * (lo + hi), d, t, 60),
                       ('demand_max', st['soc'], len(g['D']) - 1, t, 0), ('demand_min_cold', st['soc'], 0, 0, 0),
                       ('coolant40', st['soc'], d, g['T'].index(45.0), 10)]
                for name, soc, dd, tt, L in var:
                    cand = tuple(np.asarray(a, np.float32).astype(np.float64) if a.dtype != bool else a
                                 for a in (g['PF'][dd, tt], g['PB'][dd, tt], g['OK'][dd, tt], g['ON'][dd, tt]))
                    j, s, pon, x = decide(core_ref, strat, mo, g, soc, dd, tt, L, cand=cand)
                    y = net_ref(x)
                    cases.append(dict(example=ti, variant=name, moment=len(moments) - 1, soc=soc, d=dd, t=tt, L=L,
                                      outputs=[float(y[0]), float(y[1]), float(s), float(pon), float(j)]))
        tens[f't{ti}_pf'] = np.stack(PF); tens[f't{ti}_pb'] = np.stack(PB)
        tens[f't{ti}_ok'] = np.stack(OK).astype(np.uint8); tens[f't{ti}_onc'] = np.stack(ONc).astype(np.uint8)
        ctrl = strat.cap[0]['ev'][2]
        e = veh.engine
        trips_meta.append(dict(
            id=tid, vehicle=ex['vehicle'], short=SHORT.get(tid, ex['arch']), arch=veh.arch,
            likely=spec['meta'].get('likely_models', ''), N=N, dist_km=float(ex['distance_km']), T_route=float(np.mean(trip['T_amb'])),
            logged_fuel_g=float(np.nansum(trip['fuel_rate'])), soc0=soc0, mass_kg=float(veh.mass), Pe=float(e.P_max),
            E_kWh=float(b.E_nom / 3.6e6), soc_lo=b.soc_lo, soc_hi=b.soc_hi, ref=float(strat.ref), phev=bool(strat.phev),
            ctrl=[float(q) for q in ctrl], start_W=float(strat.start_W), cat_warm_s=float(strat.cat_warm_s),
            batt=dict(ocv=[float(q) for q in b.ocv.y], R0_25=b.R0_25, Ea=b.Ea, v_min=b.v_min, v_max=b.v_max,
                      P_dis_max=b.P_dis_max, P_chg_max=b.P_chg_max, E_nom=b.E_nom, mass=float(veh.mass)),
            runs=runs_meta, moments=moments))

    meta = dict(
        trips=trips_meta, strategies=STRATS, temps=TEMPS, masses=MASSES, previews=PREVIEWS,
        policy=dict(HP.DEFAULTS), n_feat=HP.N_FEAT, bins=HP.BINS,
        idx=dict(x_soc=12, soc_frac=13, soc_minus_ref=14, T_eng=18, need_heat=21, cat_pending=22, p_dis=23, p_chg=24,
                 P_dem=29, prev=37, E_avail=65),
        note='RoadSim twins of the VED cars on their real exam routes; simulated fuel, not logged fuel.')
    keepx = {}
    for name in tens:
        if name.endswith(('_on', '_ok', '_onc')):
            keepx[name] = 'u8'
        elif name.endswith('_xk'):
            keepx[name] = 'i32'
        elif name.endswith(('_pf', '_pb', '_xs', '_xsoc', '_xsrec')):
            keepx[name] = 'f32'
    write_pack(OUT / 'examples', tens, dtype='f16', keep=keepx, meta=meta)
    write_json(OUT / 'reference.json', {'cases': cases, 'tolerance': 2e-3})
    size = sum(os.path.getsize(OUT / f) for f in os.listdir(OUT))
    print(f"moments: {check['recorded']}, decision matches RoadSim's recorded choice: float64 weights {check['f64_same']}, "
          f"float16 weights {check['f16_same']}")
    print(f"wrote {OUT}: {len(index)} trips, {len(cases)} parity cases, {size / 1e6:.2f} MB, {time.time() - t00:.0f} s")


if __name__ == '__main__':
    main()
