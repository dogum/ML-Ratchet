"""ASHRAE Standard 140 (BESTEST) envelope cases with the 5R1C zone, against the published ranges of the 1995
reference programs (ESP-r, BLAST, DOE-2.1D, SERIRES/SUNCODE, S3PAS, TRNSYS, TASE).

Weather: the Denver-Stapleton TMY (WBAN 23062) from climate.onebuilding.org, the same TMY that BESTEST's
DRYCOLD.TMY was made from; run in local standard time (UTC-7). Building: 8 x 6 x 2.7 m, 12 m2 of south glazing
(620/920: 6 m2 east + 6 m2 west), 0.5 ACH, 200 W constant sensible gains, ideal heating below 20 C and cooling
above 27 C, floor exposed to outdoor air (U 0.039), solar absorptance 0.6. 5R1C mass parameters are derived from
the BESTEST layer tables (ISO 13790 12.3: C_m = sum kappa_j A_j over layers inside the insulation,
A_m = C_m^2 / sum A_j kappa_j^2).

Reference ranges: ASHRAE 140-2001 informative results tables (min-max of the reference programs). They were
transcribed from secondary literature, not the standard itself, and should be confirmed against it.
"""
import os as _os
BLDG_CODE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))  # shared/building
BLDG_ROOT = _os.environ.get('ML_RATCHET_BUILDING') or BLDG_CODE   # data/, buildingsim/refdata/, checks/out/ live here
import numpy as np
import pandas as pd

from _common import write
from buildingsim import read_epw
from buildingsim.simulate import Simulator

EPW = f'{BLDG_ROOT}/buildingsim/refdata/weather/bestest/USA_CO_Denver-Stapleton.Intl.AP.724690_TMY.epw'

# (annual heating MWh, annual cooling MWh, peak heating kW, peak cooling kW) min/max
RANGES = {
    '600': dict(heat=(4.296, 5.709), cool=(6.137, 7.964), peak_heat=(3.437, 4.354), peak_cool=(5.965, 6.827)),
    '610': dict(heat=(4.355, 5.786), cool=(3.915, 5.778)),
    '620': dict(heat=(4.613, 5.944), cool=(3.417, 5.004)),
    '640': dict(heat=(2.751, 3.803), cool=(5.952, 7.811)),
    '650': dict(heat=(0.0, 0.0), cool=(4.816, 6.545)),
    '900': dict(heat=(1.170, 2.041), cool=(2.132, 3.415), peak_heat=(2.850, 3.797), peak_cool=(2.888, 3.932)),
    '910': dict(heat=(1.512, 2.282), cool=(0.821, 1.917)),
    '920': dict(heat=(3.261, 4.300), cool=(1.840, 3.578)),
    '940': dict(heat=(0.793, 1.411), cool=(2.079, 3.338)),
    '950': dict(heat=(0.0, 0.0), cool=(0.387, 0.921)),
    '600FF': dict(Tmax=(64.9, 69.5), Tmin=(-18.8, -15.6), Tmean=(24.2, 27.4)),
    '900FF': dict(Tmax=(41.8, 46.4), Tmin=(-6.4, -1.6), Tmean=(24.5, 27.5)),
    '650FF': dict(Tmax=(63.2, 68.2), Tmin=(-23.0, -21.0), Tmean=(18.0, 19.6)),
    '950FF': dict(Tmax=(35.5, 38.5), Tmin=(-20.2, -17.8), Tmean=(14.0, 15.3)),
}

# 5R1C mass from the BESTEST layers (interior side of the insulation)
A_WALL, A_ROOF, A_FLOOR = 63.6, 48.0, 48.0
K_LIGHT = dict(wall=0.012 * 950 * 840, roof=0.010 * 950 * 840, floor=0.025 * 650 * 1200)
K_HEAVY = dict(wall=0.100 * 1400 * 1000, roof=0.010 * 950 * 840, floor=0.080 * 1400 * 1000)


def mass_params(heavy):
    K = K_HEAVY if heavy else K_LIGHT
    Cm = A_WALL * K['wall'] + A_ROOF * K['roof'] + A_FLOOR * K['floor']
    Am = Cm ** 2 / (A_WALL * K['wall'] ** 2 + A_ROOF * K['roof'] ** 2 + A_FLOOR * K['floor'] ** 2)
    return Cm, Am


def spec_for(case):
    heavy = case.startswith('9')
    Cm, Am = mass_params(heavy)
    base = case.replace('FF', '')
    win = {'S': 12.0, 'N': 0.0, 'E': 0.0, 'W': 0.0}
    if base in ('620', '920'):
        win = {'S': 0.0, 'N': 0.0, 'E': 6.0, 'W': 6.0}
    g = dict(aspect_ratio=8 / 6, floor_to_floor_m=2.7, ceiling_m=2.7, orientation_deg=0.0, foundation='adiabatic',
             floor_U=0.039, shading_factor=1.0, roof_absorptance=0.6, wall_absorptance=0.6, attic=False,
             window_area_by_facade=win, C_m_J_K=Cm, A_m_m2=Am, A_t_m2=171.6)
    if base in ('610', '910'):
        g.update(window_overhang=0.5, overhang_gap=0.25)
    return dict(id='bestest' + case, primary_use='other', floor_area_m2=48.0, n_floors=1, year_built=1995,
                wall_U=0.512 if heavy else 0.514, roof_U=0.318, window_U=3.0, SHGC=0.789, window_wwr=0.0,
                infiltration_ach=0.5, infiltration_model='constant', thermal_mass='light', geometry=g,
                internal_gains=dict(people=0.0, lighting=0.0, plug=0.0, process=200.0 / 48.0, cooking=0.0,
                                    exterior_lighting=0.0),
                setpoints=dict(heat=20.0, cool=27.0, setback_heat=None, setback_cool=None),
                ventilation=dict(L_s_person=0.0, L_s_m2=0.0, economizer=False, mech_vent=False),
                system='none' if case.endswith('FF') else 'ideal',
                dhw=dict(fuel='electric', L_per_day=0.0, UA_W_K=0.0), moisture_capacitance=1.0)


def run_case(case, df, meta, dt='1h'):
    spec = spec_for(case)
    sim = Simulator(spec, (df, meta), '2019-01-01 07:00', '2020-01-01 07:00', dt=dt, warmup_days=14)
    hour = sim.wx['cal']['hour']
    base = case.replace('FF', '')
    n = sim.n
    h = np.full(n, 20.0)
    c = np.full(n, 27.0)
    if base in ('640', '940'):
        h = np.where((hour >= 23) | (hour < 7), 10.0, 20.0)
    if base in ('650', '950'):
        day = (hour >= 7) & (hour < 18)
        h = np.full(n, -100.0)
        c = np.where(day, 27.0, 100.0)
        rho = np.asarray(sim.rho)
        sim.m_v = np.where(~day, rho * 1703.16 / 3600.0, 0.0).tolist()
        sim.T_v = sim.wx['T_out'].tolist()
        sim.W_v = sim.wx['W_out'].tolist()
    sim.hsp, sim.csp = h.tolist(), c.tolist()
    out = sim.run()
    ts = out['ts']
    dth = sim.dt / 3600
    q = ts['Q_hvac_sens_kW'].values
    res = dict(heat=float(q[q > 0].sum() * dth / 1000), cool=float(-q[q < 0].sum() * dth / 1000),
               peak_heat=float(max(q.max(), 0)), peak_cool=float(max(-q.min(), 0)),
               Tmax=float(ts['T_zone'].max()), Tmin=float(ts['T_zone'].min()), Tmean=float(ts['T_zone'].mean()),
               ledger_closure=out['ledger']['closure_rel'])
    return res


def main():
    df, meta = read_epw(EPW)
    meta = dict(meta, tz='Etc/GMT+7')         # local standard time, no DST (as BESTEST)
    rows = {}
    n_in = n_tot = 0
    for case in RANGES:
        r = run_case(case, df, meta)
        row = {}
        for k, (lo, hi) in RANGES[case].items():
            v = r[k]
            inside = (lo - 1e-9) <= v <= (hi + 1e-9)
            mid = 0.5 * (lo + hi)
            row[k] = dict(sim=round(v, 3), lo=lo, hi=hi, inside=inside,
                          dev_from_mid_pct=round(100 * (v - mid) / mid, 1) if mid else None)
            if not (lo == hi == 0):
                n_in += inside
                n_tot += 1
        row['ledger_closure'] = r['ledger_closure']
        rows[case] = row
        print(case, {k: (v['sim'], v['lo'], v['hi'], 'IN' if v['inside'] else 'out') for k, v in row.items()
                     if isinstance(v, dict)})
    # sensitivities (less weather-dependent): differences between cases vs differences of range mid-points
    def mid(c, k):
        lo, hi = RANGES[c][k]
        return 0.5 * (lo + hi)
    sens = {}
    for a, b, k in (('900', '600', 'heat'), ('900', '600', 'cool'), ('640', '600', 'heat'), ('620', '600', 'cool'),
                    ('610', '600', 'cool'), ('920', '900', 'cool'), ('910', '900', 'cool'), ('940', '900', 'heat')):
        sens[f'{a}-{b} {k}'] = dict(sim=round(rows[a][k]['sim'] - rows[b][k]['sim'], 3),
                                    ref_mid=round(mid(a, k) - mid(b, k), 3))
    write('bestest.json', dict(
        what='ASHRAE 140 BESTEST envelope cases (5R1C zone, 1 h step) vs the 1995 reference-program ranges',
        weather='Denver-Stapleton TMY (WBAN 23062), climate.onebuilding.org; local standard time',
        mass_params={'light': mass_params(False), 'heavy': mass_params(True)},
        cases=rows, inside=n_in, total=n_tot, sensitivities=sens,
        caveats=['ranges transcribed from secondary literature (ASHRAE 140-2001 informative tables); confirm',
                 '5R1C has one mass node and a fixed 50/50 convective/radiative split of gains (BESTEST: 40/60)',
                 'window angular transmittance from pvlib physical IAM (2 panes), not the BESTEST glazing tables',
                 'case 610/910 overhang modelled as an infinite horizontal projection (no side effects)']))


if __name__ == '__main__':
    main()
