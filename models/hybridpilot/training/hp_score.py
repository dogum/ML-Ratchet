"""Scoring: run every strategy on one (vehicle, trip, start state) and grade it against the DP optimum at matched
battery energy.

SOC-corrected fuel: the DP closed-loop policy is re-solved for several terminal SOC targets on the SAME trip; this
traces DP fuel as a function of net battery chemical energy used, F_DP(E). A strategy that used E_s kJ of battery
energy and burned F_s is graded  pct_DP = F_DP(E_s) / F_s  (both at the same net battery energy: "matched SOC").
Linear interpolation inside the DP points, linear extrapolation with the end slope outside (flagged).
The Phase 1 fixed-factor correction (F + E/(0.33*0.90) fuel-equivalent) is also reported for comparison.
"""
import _paths as _P  # portable paths (see _paths.py)
import sys, math, time
sys.path.insert(0, _P.VEHICLE + '/roadsim' + '')
import numpy as np
from roadsim import simulate
from roadsim.sim import get_vehicle
from roadsim.strategies import list_strategies
from roadsim.dp import DPStrategy
from roadsim.powertrain import ARCH_CLASSES
from hp_policy import HybridPilot, make_stage, dp_backward

LHV = {'gasoline': 42.9e3, 'diesel': 42.8e3}     # kJ/kg (only for the fixed-factor comparison)


class DPFixed(DPStrategy):
    """DP closed-loop policy from a precomputed solution."""
    name = 'dp'

    def setup(self, veh, pt, trip, st):
        from roadsim.strategies import HybridStrategy
        HybridStrategy.setup(self, veh, pt, trip, st)
        self.cat_warm_s = 0.0
        self.sol = self.p['sol']; self.k = 0


def baseline_names(arch):
    L = list_strategies(arch)
    return [s for s in L if s not in ('dp',)]


def metrics(out, spec_arch, soc0):
    S = out['summary']; L = out['ledger']
    on = out['eng_on'] > 0.5 if 'eng_on' in out else np.zeros(1, bool)
    d_km = max(S['distance_m'] / 1e3, 1e-3)
    starts = int(np.sum(on[1:] & ~on[:-1]))
    # short on-periods (< 5 s) and short off-periods (< 5 s) = chatter
    runs = []
    if len(on) > 1:
        idx = np.flatnonzero(np.diff(on.astype(int)) != 0) + 1
        seg = np.split(on, idx)
        runs = [(bool(x[0]), len(x)) for x in seg if len(x)]
    short_on = sum(1 for o, n in runs[1:-1] if o and n < 5)
    short_off = sum(1 for o, n in runs[1:-1] if (not o) and n < 5)
    soc = out['soc']
    return dict(fuel_g=S['fuel_g'], E_kJ=L['batt_chem_out'], soc_end=float(soc[-1]), soc_min=float(soc.min()), soc_max=float(soc.max()),
                dsoc=float(soc[-1] - soc0), starts=starts, starts_per_km=starts / d_km, short_on=short_on, short_off=short_off,
                chatter_per_km=(short_on + short_off) / d_km, on_frac=float(on.mean()), dist_km=d_km,
                trace_rms=S['trace_rms_mps'], closure=L['closure_rel'], wall=S['wall_s'])


def dp_curve(spec, trip, soc0, temps, targets, stage=None, veh=None):
    veh = veh or get_vehicle(spec)
    if stage is None:
        pt = ARCH_CLASSES[veh.arch](veh)
        from roadsim.sim import normalize_trip
        stage = make_stage(veh, pt, normalize_trip(trip))
    pts = []; outs = {}
    for x in targets:
        sol = dp_backward(veh, stage, soc_final=x)
        o = simulate(spec, trip, strategy=DPFixed(sol=sol), init_soc=soc0, init_temps=temps, record=True, options='roadsim-physics-v1')
        m = metrics(o, veh.arch, soc0); m['target'] = x
        pts.append(m); outs[x] = (o, sol)
    return pts, outs, stage


def interp_dp(pts, E):
    """F_DP at battery energy E (kJ) from the DP points (sorted by E); returns (F, extrapolated?)."""
    P = sorted(((p['E_kJ'], p['fuel_g']) for p in pts))
    Es = np.array([p[0] for p in P]); Fs = np.array([p[1] for p in P])
    # enforce monotone non-increasing F(E) (more battery energy used -> no more fuel), merge near-duplicates
    keep = [0]
    for i in range(1, len(Es)):
        if Es[i] - Es[keep[-1]] > 1.0:
            keep.append(i)
        elif Fs[i] < Fs[keep[-1]]:
            keep[-1] = i
    Es, Fs = Es[keep], np.minimum.accumulate(Fs[keep])
    if len(Es) == 1:
        return float(Fs[0]), True
    if E < Es[0]:
        sl = (Fs[1] - Fs[0]) / (Es[1] - Es[0]); return float(Fs[0] + sl * (E - Es[0])), True
    if E > Es[-1]:
        sl = (Fs[-1] - Fs[-2]) / (Es[-1] - Es[-2]); return float(max(Fs[-1] + sl * (E - Es[-1]), 0.0)), True
    return float(np.interp(E, Es, Fs)), False


def dp_targets(veh, soc0, phev):
    b = veh.battery
    cs = b.soc_lo + 0.05
    if phev and soc0 > cs + 0.03:
        return [round(cs, 4), round(soc0, 4)]          # full depletion and full sustain; matched points are added later
    lo, hi = b.soc_lo + 0.02, b.soc_hi - 0.02
    return [min(max(soc0 + d, lo), hi) for d in (-0.06, 0.0, 0.06)]


def add_matched(case, veh, stage, pts, soc_end, tol=0.01):
    """DP re-solved to end at a strategy's own end SOC (exact matched-SOC reference); appended to pts."""
    if any(abs(p['soc_end'] - soc_end) < tol for p in pts):
        return False
    b = veh.battery
    x = float(min(max(soc_end, b.soc_lo + 0.005), b.soc_hi - 0.005))
    sol = dp_backward(veh, stage, soc_final=x)
    o = simulate(case['spec'], case['trip'], strategy=DPFixed(sol=sol), init_soc=case['soc0'], init_temps=case['temps'], record=True, options='roadsim-physics-v1')
    m = metrics(o, veh.arch, case['soc0']); m['target'] = x; m['matched'] = True
    pts.append(m)
    return True


def f_cs(pts, soc0):
    """DP fuel when sustaining the start SOC (the trip's charge-sustaining optimum): the normaliser of pct_DP."""
    j = int(np.argmin([abs(p['target'] - soc0) + (0 if not p.get('matched') else 1.0) for p in pts]))
    return max(pts[j]['fuel_g'], 1.0)


def grade(res_m, pts, fuel_kJ_per_g, soc0=None):
    """pct_DP = 1 - (F_s - F_DP(E_s)) / F_DP,CS : excess fuel over the DP optimum at the same net battery energy, as a
    fraction of the trip's charge-sustaining DP fuel (equals F_DP/F_s to first order for HEVs; stays finite for PHEV
    trips with almost no fuel)."""
    F, ex = interp_dp(pts, res_m['E_kJ'])
    res_m['F_dp_matched'] = F; res_m['extrap'] = ex
    Fcs = f_cs(pts, soc0 if soc0 is not None else pts[0]['target'])
    res_m['F_cs'] = Fcs
    res_m['pct_dp'] = 1.0 - (res_m['fuel_g'] - F) / Fcs
    res_m['pct_dp_ratio'] = (F + 1.0) / (res_m['fuel_g'] + 1.0)
    res_m['fuel_corr_fixed_g'] = res_m['fuel_g'] + res_m['E_kJ'] / (0.33 * 0.90) / fuel_kJ_per_g
    return res_m


def run_case(case, policies=None, baselines=True, extra_targets=None, keep_stage=True):
    """case: dict(spec, trip, soc0, temps, ...). policies: {name: HybridPilot params}. Returns per-strategy metrics with
    pct_DP at matched battery energy (DP re-solved at each strategy's end SOC)."""
    spec, trip, soc0, temps = case['spec'], case['trip'], case['soc0'], case['temps']
    veh = get_vehicle(spec)
    phev = veh.arch.startswith('phev')
    t0 = time.time()
    tg = dp_targets(veh, soc0, phev) + list(extra_targets or [])
    pts, outs, stage = dp_curve(spec, trip, soc0, temps, tg, veh=veh)
    res = {}
    nat = soc0 if not (phev and soc0 > veh.battery.soc_lo + 0.08) else veh.battery.soc_lo + 0.05
    jn = int(np.argmin([abs(p['target'] - nat) for p in pts]))
    res['dp'] = dict(pts[jn])
    names = baseline_names(veh.arch) if baselines else []
    for nm in names:
        o = simulate(spec, trip, strategy=nm, init_soc=soc0, init_temps=temps, record=True, options='roadsim-physics-v1')
        res[nm] = metrics(o, veh.arch, soc0)
    for nm, prm in (policies or {}).items():
        st = HybridPilot(**prm)
        o = simulate(spec, trip, strategy=st, init_soc=soc0, init_temps=temps, record=True, options='roadsim-physics-v1')
        res[nm] = metrics(o, veh.arch, soc0)
    for nm, m in res.items():
        if nm != 'dp':
            add_matched(case, veh, stage, pts, m['soc_end'])
    fuel_kJ_per_g = LHV.get(veh.fuel, 42.9e3) / 1e3
    for nm, m in res.items():
        grade(m, pts, fuel_kJ_per_g, soc0)
    res['_dp_points'] = [dict(target=p['target'], E_kJ=p['E_kJ'], fuel_g=p['fuel_g'], soc_end=p['soc_end'], starts_per_km=p['starts_per_km'],
                              chatter_per_km=p['chatter_per_km'], matched=bool(p.get('matched'))) for p in pts]
    if keep_stage:
        g, PF, PB, OK, ON, Tb = stage
        res['_stage'] = (g, PF.astype(np.float32), PB.astype(np.float32), OK, ON, Tb)
    res['_meta'] = dict(arch=veh.arch, cls=veh.cls, hd=bool(veh.hd), phev=phev, soc0=soc0, warm=temps is not None,
                        T_amb=float(np.mean(trip['T_amb'])) if 'T_amb' in trip else 20.0, N=len(trip['v']),
                        dist_km=res['dp']['dist_km'], wall=time.time() - t0, soc_lo=veh.battery.soc_lo, soc_hi=veh.battery.soc_hi,
                        name=trip.get('_name'), kind=trip.get('_kind'), vehicle=spec.get('id'))
    return res
