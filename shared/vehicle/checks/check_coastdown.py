"""Check: coast-down. For every EPA validation vehicle (47; >= 20 required) the spec is built from its EPA test weight
and target road-load A/B/C (CdA from C, Crr from A, speed coefficient from B, at the reference air density).
roadsim's road-load model then *drives a virtual coast-down*: 80 -> 15 mph in neutral, both directions, steady wind of
random speed (0-3 m/s) and direction, 10 Hz speed with 0.05 mph noise. A, B, C are recovered with an SAE J2263-style
fit and compared with the EPA targets: road-load force at 20/40/60 mph and the coefficients.

Second part (independent reference): the same coast-down at 5 C ambient (rho from the ideal-gas law), corrected back to
20 C with SAE J2263's standard corrections (aero x rho_std/rho, rolling x [1 + 0.0086 (T - 20)]); this tests roadsim's
ISO 28580-based tyre temperature model and air-density model against the SAE correction.

Tolerance ('the test's spread'): the coefficient of variation of the 50 mph road-load force among repeat EPA target
entries of the same make/model/test weight in the test-car lists (MY2022-24).
"""
import os, sys, json, copy
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'roadsim'))
from roadsim.sim import get_vehicle
from roadsim.coastdown import coast, fit_abc, to_lbf
from roadsim.constants import MPH, LBF

V = json.load(open(os.path.join(HERE, 'epa_vehicles.json')))


def force_lbf(abc, mph):
    A, B, C = abc
    return A + B * mph + C * mph ** 2


def epa_spread():
    d = pd.read_csv(os.path.join(HERE, '..', 'roadsim', 'refdata', 'epa_candidates.csv'))
    d = d.dropna(subset=['Target Coef A (lbf)', 'Target Coef B (lbf/mph)', 'Target Coef C (lbf/mph**2)'])
    d = d.drop_duplicates(subset=['Represented Test Veh Make', 'Represented Test Veh Model', 'Equivalent Test Weight (lbs.)',
                                  'Target Coef A (lbf)', 'Target Coef B (lbf/mph)', 'Target Coef C (lbf/mph**2)'])
    d['F50'] = d['Target Coef A (lbf)'] + 50 * d['Target Coef B (lbf/mph)'] + 2500 * d['Target Coef C (lbf/mph**2)']
    g = d.groupby(['Represented Test Veh Make', 'Represented Test Veh Model', 'Equivalent Test Weight (lbs.)'])['F50']
    cv = (g.std() / g.mean())[g.count() >= 2].dropna()
    return float(np.median(cv)) * 100, float(np.percentile(cv, 75)) * 100, int(len(cv))


def run_one(spec, rng, T=20.0, rho=1.1767):
    veh = get_vehicle(dict(spec, tire=dict(spec.get('tire', {}), temp_effects=True, K_t=0.008)))
    runs = []
    for k in range(3):   # 3 pairs of opposite-direction runs
        w = rng.uniform(0, 3.0); dr = rng.uniform(0, 360)
        runs.append(coast(veh, wind=w, wind_dir_rel=dr, rho=rho, T_amb=T))
        runs.append(coast(veh, wind=w, wind_dir_rel=(dr + 180) % 360, rho=rho, T_amb=T))
    return veh, fit_abc(veh, runs, rng=rng)


def main():
    rng = np.random.default_rng(42)
    cv_med, cv_p75, n_groups = epa_spread()
    rows = []
    for v in V:
        spec = copy.deepcopy(v['spec'])
        veh, (A, B, C) = run_one(spec, rng)
        fit = to_lbf(A, B, C)
        # 5 C test corrected to standard conditions (SAE J2263)
        T5 = 5.0; rho5 = 98210.0 / (287.05 * (T5 + 273.15)) * (101325 / 98210.0)
        veh5, (A5, B5, C5) = run_one(spec, rng, T=T5, rho=rho5)
        # J2263: split measured force into aero (C v^2) and mechanical (A + B v) parts, correct each
        kroll = 1 + 0.0086 * (T5 - 20.0)
        A5c, B5c, C5c = A5 * kroll, B5 * kroll, C5 * (1.1767 / rho5)
        fit5 = to_lbf(A5c, B5c, C5c)
        tgt = v['abc']
        row = dict(name=v['name'], architecture=v['spec']['architecture'], etw_lb=v['etw_lb'], target_abc_lbf=tgt,
                   fitted_abc_lbf=[round(x, 4) for x in fit], fitted_5C_corrected_abc_lbf=[round(x, 4) for x in fit5])
        for mph in (20, 40, 60):
            row[f'err_F{mph}_pct'] = round(100 * (force_lbf(fit, mph) / force_lbf(tgt, mph) - 1), 3)
            row[f'err5C_F{mph}_pct'] = round(100 * (force_lbf(fit5, mph) / force_lbf(tgt, mph) - 1), 3)
        rows.append(row)
    e = np.array([[r['err_F20_pct'], r['err_F40_pct'], r['err_F60_pct']] for r in rows])
    e5 = np.array([[r['err5C_F20_pct'], r['err5C_F40_pct'], r['err5C_F60_pct']] for r in rows])
    res = dict(check='coast-down vs EPA target road-load coefficients', n_vehicles=len(rows), method=__doc__.strip(),
               epa_repeat_target_cv_F50_pct=dict(median=round(cv_med, 2), p75=round(cv_p75, 2), n_model_groups=n_groups),
               standard_conditions=dict(abs_err_pct_median=[round(float(x), 3) for x in np.median(np.abs(e), 0)],
                                        abs_err_pct_max=[round(float(x), 3) for x in np.max(np.abs(e), 0)],
                                        speeds_mph=[20, 40, 60]),
               at_5C_J2263_corrected=dict(err_pct_median=[round(float(x), 3) for x in np.median(e5, 0)],
                                          abs_err_pct_max=[round(float(x), 3) for x in np.max(np.abs(e5), 0)], speeds_mph=[20, 40, 60]),
               frac_within_epa_spread=round(float(np.mean(np.max(np.abs(e), 1) <= cv_med)), 3),
               frac_within_epa_spread_5C=round(float(np.mean(np.max(np.abs(e5), 1) <= cv_med)), 3),
               vehicles=rows)
    json.dump(res, open(os.path.join(HERE, 'coastdown.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ('vehicles', 'method')}, indent=1))


if __name__ == '__main__':
    main()
