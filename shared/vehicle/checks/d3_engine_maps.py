"""Measured fuel maps of older Toyota hybrid engines from D3 dev 10 Hz data (RoadSim v3.1 pass, item A).

No map of a pre-TNGA Toyota hybrid engine is reachable from a script (CALIBRATION.md H22). D3 logs engine speed and
fuel flow on its dev Toyota-system cars, so the brake power can be rebuilt from the dyno and the battery:

    P_net  = P_batt - P_dcdc / 0.92                       (HV battery power left for the drive; + = discharge)
    P_eng  = (P_wheel - 0.90 x P_net) / 0.93               while the pack discharges
    P_eng  = (P_wheel - P_net / 0.90) / 0.93               while it charges
    P_wheel = dyno tractive force x roll speed

Assumptions (stated, not fitted): 0.93 engine-to-wheel efficiency of the power-split at cruise (mechanical path ~0.97,
electrical path ~0.85, about 30 % of the power through it), 0.90 battery-to-wheel / engine-to-battery efficiency,
0.92 for the 12 V DC/DC converter (logged on the Prime, 0 where not logged). A 3-point error in either efficiency moves
the brake power, and so the measured efficiency, by about 3 %.

Windows: 5-s moving averages of the 1 Hz contract arrays; engine running (> 900 rpm), engine-speed spread < 6 % of
its mean in the window, wheel power > 0, brake power > 3 kW, coolant > 70 C, std ambient tests; every third window.
For each window the twin's own engine (the D3 twin spec: the TNGA hybrid paper map resized to the car's rated power)
gives the map's fuel power at the same speed and torque. Ratio = measured fuel power / map fuel power (warm map, no
era scaling). Binned by brake mean effective pressure (BMEP = 4 pi T / displacement, 1 bar bins); median per bin.

Fit set and check set: the 2017 Prius Prime's non-standard tests (steady states, pedal maps, passing, WOT, idle and
other schedules) build the table; its UDDS / HWFET / US06 tests are the check (the twin check, check_d3.py, runs only
those). The 2010 Prius (5 tests, modal carbon-balance fuel, legacy era) is shown for comparison only; its fuel signal
lags and scatters, so it does not set the table.

Table (clipping and smoothing): bins with >= 20 windows; ratios clipped to [0.85, 1.6]; made non-increasing in BMEP
(pool-adjacent-violators: the excess of a real engine over the map falls with load); above the last bin the last
value holds. Writes roadsim/roadsim/data/engine_era_d3.json (the table, the bins, the check, provenance).

usage: python d3_engine_maps.py [--write]
"""
import os, sys, math, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VEH = os.path.dirname(HERE)
sys.path.insert(0, VEH); sys.path.insert(0, os.path.join(VEH, 'roadsim'))
from realdata.d3 import d3_index, load_d3, d3_specs
from roadsim.components import Engine
from roadsim.constants import FUELS

OUT = os.path.join(VEH, 'roadsim', 'roadsim', 'data', 'engine_era_d3.json')
ETA_SYS, ETA_E, ETA_DCDC = 0.93, 0.90, 0.92
LHV = FUELS['gasoline']['lhv']                    # J/kg
STD_CYC = ('UDDS', 'HWFET', 'US06')
CARS = [('D3_2017_Toyota_Prius_Prime', '2017 Toyota Prius Prime'), ('D3_2010_Toyota_Prius', '2010 Toyota Prius')]
BINS = np.arange(1.0, 13.01, 1.0)                # bar


def mav(x, n=5):
    return np.convolve(np.nan_to_num(np.asarray(x, float)), np.ones(n) / n, mode='same')


def _sig(a, k):
    x = a.get(k)
    return None if x is None or np.all(np.isnan(x)) else np.asarray(x, float)


def windows(vid, veh):
    """matched windows of one D3 dev car: (test_id, cycle, rpm, P_eng W, T Nm, bmep bar, fuel W measured, fuel W map)."""
    S = d3_specs()[vid]
    e = Engine(S['engine'])
    disp = float(S['engine']['displacement_L'])
    d = d3_index(vehicles=veh)
    d = d[d.T_class == 'std']
    rows = []
    for tid in d.test_id:
        m, a = load_d3(tid)
        F, rpm, fr = _sig(a, 'F_tractive'), _sig(a, 'engine_rpm'), _sig(a, 'fuel_rate')
        if F is None or rpm is None or fr is None:
            continue
        v = np.asarray(a['v'], float)
        Pb = _sig(a, 'batt_power')
        if Pb is None:
            Pb = np.asarray(a['batt_current'], float) * np.asarray(a['batt_voltage'], float)
        Pd = _sig(a, 'aux_power'); Pd = np.zeros_like(v) if Pd is None else np.nan_to_num(Pd)
        Tc = _sig(a, 'T_coolant'); Tc = np.full_like(v, 85.0) if Tc is None else Tc
        Pw = mav(F * v); Pnet = mav(Pb) - mav(Pd) / ETA_DCDC
        rm = mav(rpm); fm = mav(fr)
        rsd = np.sqrt(np.maximum(mav(rpm ** 2) - rm ** 2, 0.0))
        Pe = np.where(Pnet > 0, (Pw - ETA_E * Pnet) / ETA_SYS, (Pw - Pnet / ETA_E) / ETA_SYS)
        ok = ((rm > 900) & (rpm > 900) & (rsd < 0.06 * rm) & (Pw > 0) & (Pe > 3000) & (Tc > 70) & (fm > 0.1)
              & np.isfinite(Pe) & np.isfinite(fm))
        ok[:10] = False; ok[-10:] = False
        for i in np.where(ok)[0][::3]:
            w = rm[i] * math.pi / 30; T = Pe[i] / w
            rows.append((tid, str(m['row']['cycle']), float(rm[i]), float(Pe[i]), float(T), float(4 * math.pi * T / (disp * 1e-3) / 1e5),
                         float(fm[i] * LHV / 1e3), float(e.fuel_W(w, T))))
    return rows


def binned(rows, min_n=1):
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        r = [x[6] / x[7] for x in rows if lo <= x[5] < hi]
        if len(r) >= min_n:
            out.append(dict(bmep_lo=float(lo), bmep_hi=float(hi), n=len(r), ratio=round(float(np.median(r)), 3),
                            rpm=round(float(np.median([x[2] for x in rows if lo <= x[5] < hi]))),
                            P_kW=round(float(np.median([x[3] for x in rows if lo <= x[5] < hi])) / 1e3, 1)))
    return out


def pava_dec(y, w):
    """non-increasing least-squares fit (pool adjacent violators)."""
    blocks = [[float(v), float(u), 1] for v, u in zip(y, w)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            a, b = blocks[i], blocks[i + 1]
            u = a[1] + b[1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / u, u, a[2] + b[2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for v, u, k in blocks:
        out += [v] * k
    return out


def table_of(bins, min_n=20):
    b = [x for x in bins if x['n'] >= min_n]
    x = [0.5 * (q['bmep_lo'] + q['bmep_hi']) for q in b]
    y = np.clip([q['ratio'] for q in b], 0.85, 1.6)
    y = pava_dec(y, [q['n'] for q in b])
    return [[round(xx, 2), round(float(yy), 3)] for xx, yy in zip(x, y)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    res = {}
    for vid, veh in CARS:
        rows = windows(vid, veh)
        fit = [r for r in rows if r[1] not in STD_CYC]
        chk = [r for r in rows if r[1] in STD_CYC]
        res[vid] = dict(n_windows=len(rows), fit=binned(fit), check=binned(chk), all=binned(rows))
        print(vid, len(rows), 'windows (fit', len(fit), 'check', len(chk), ')')
        for q in res[vid]['all']:
            f = next((z for z in res[vid]['fit'] if z['bmep_lo'] == q['bmep_lo']), None)
            c = next((z for z in res[vid]['check'] if z['bmep_lo'] == q['bmep_lo']), None)
            print(f"  BMEP {q['bmep_lo']:4.0f}-{q['bmep_hi']:<4.0f} bar  rpm {q['rpm']:5d}  P {q['P_kW']:5.1f} kW   all {q['ratio']:.3f} (n{q['n']:4d})"
                  f"   fit {f['ratio'] if f else float('nan'):.3f} (n{f['n'] if f else 0:4d})   check {c['ratio'] if c else float('nan'):.3f} (n{c['n'] if c else 0:4d})")
    tab = table_of(res['D3_2017_Toyota_Prius_Prime']['fit'])
    print('table (bmep bar, fuel ratio vs the TNGA paper map):', tab)
    out = dict(
        about='Measured fuel ratio of an older Toyota hybrid engine (2ZR-FXE, 2017 Prius Prime, D3 dev) to the TNGA '
              'hybrid paper map at the same engine speed and torque, by BMEP; used by the v3.1 option engine_era '
              '(roadsim/eras.py, mode "table"). Built by checks/d3_engine_maps.py; method and assumptions in its docstring.',
        source='Argonne D3, dev vehicles only. ' + 'This data is from the Downloadable Dynamometer Database and was generated at the Advanced Mobility '
               'Technology Laboratory (AMTL) at Argonne National Laboratory under the funding and guidance of the U.S. Department of Energy (DOE).',
        assumptions=dict(eta_sys=ETA_SYS, eta_batt_path=ETA_E, eta_dcdc=ETA_DCDC, window_s=5, min_rpm=900, rpm_spread=0.06,
                         min_P_kW=3.0, min_coolant_C=70.0, bins_bar=1.0, min_n=20, clip=[0.85, 1.6], monotone='non-increasing'),
        table_2zr_estec=tab, cars=res)
    if a.write:
        json.dump(out, open(OUT, 'w', encoding='utf-8', newline='\n'), indent=1)
        print('wrote', OUT)


if __name__ == '__main__':
    main()
