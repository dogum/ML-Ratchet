"""Reference implementations: GR4J vs the real airGR package (CRAN 1.7.9) and HBV vs TUWmodel (CRAN 1.1-1), both
compiled from the CRAN sources in an isolated R (watersim/.refenv, conda-forge r-base 4.4 + gfortran).
Run ref/airgr_run.R and ref/tuw_run.R first (see run_all.sh)."""
import numpy as np
import pandas as pd

import os

from _common import write, _DATA
import watersim as ws
from watersim import metrics as M

REF = os.path.join(_DATA, 'checks', 'ref') + '/'   # reference CSVs: data root (git-ignored)
res = {'what': 'WaterSim GR4J vs airGR and HBV vs TUWmodel (independent Fortran codes run in R)'}

# ------------------------------------------------------------------ GR4J vs airGR
X = {'X1': 257.238, 'X2': 1.012, 'X3': 88.235, 'X4': 2.208}
inp = pd.read_csv(REF + 'airgr_L0123001_inputs.csv', parse_dates=['date']).set_index('date')
f = pd.DataFrame({'P': inp['P'], 'T_mean': 10.0, 'PET': inp['E']}, index=inp.index)
spec = {'id': 'L0123001', 'area_km2': 360.0, 'lat': 47.0, 'elev_mean_m': 300}
ref = pd.read_csv(REF + 'airgr_gr4j_explicit.csv', parse_dates=['date']).set_index('date')
out = ws.simulate(spec, f, '1989-01-01', '1999-12-31', model='gr4j', params=X, snow='none', routing='none',
                  pet='forcing', init={'model': {'S': 0.3 * X['X1'], 'R': 0.5 * X['X3']}})
ts = out['ts']
d = {}
ref['Exch2'] = 2 * ref['Exch']   # airGR reports the potential semi-exchange F; WaterSim the actual total (both branches)
for ours, theirs in (('Q', 'Qsim'), ('soil', 'Prod'), ('groundwater', 'Rout'), ('ET_soil', 'AE'), ('exchange', 'Exch2')):
    diff = ts[ours].values - ref[theirs].values
    d[ours + '_vs_' + theirs] = {'max_abs_diff_mm': float(np.max(np.abs(diff))),
                                 'rel_to_mean': float(np.max(np.abs(diff)) / max(np.mean(np.abs(ref[theirs])), 1e-12))}
help_ref = pd.read_csv(REF + 'airgr_gr4j_help_example.csv', parse_dates=['date']).set_index('date')
q9099 = ts['Q'].loc['1990-01-01':'1999-12-31']
nse_ours = M.nse(inp['Qobs'].loc['1990-01-01':'1999-12-31'].values, q9099.values)
res['gr4j_vs_airGR'] = {
    'case': 'airGR help example (?RunModel_GR4J): basin L0123001, X1=257.238, X2=1.012, X3=88.235, X4=2.208; '
            'explicit initial stores S=0.3 X1, R=0.5 X3 on 1989-01-01, no warm-up, run to 1999-12-31',
    'n_days': len(ts), 'diffs': d,
    'published_help_NSE_1990_1999': 0.7988, 'airGR_NSE_this_build': float(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ref', 'airgr_meta.txt')).read().split('help_nse=')[1].split()[0]),
    'watersim_NSE_1990_1999': nse_ours,
    'max_abs_Q_diff_vs_help_example_mm': float(np.max(np.abs(q9099.values - help_ref['Qsim'].values))),
    'ledger_ok': out['ledger']['ok'],
}
res['gr4j_vs_airGR']['pass'] = bool(d['Q_vs_Qsim']['max_abs_diff_mm'] < 1e-6 and abs(nse_ours - 0.7988) < 5e-5)

# ------------------------------------------------------------------ HBV vs TUWmodel
t = pd.read_csv(REF + 'tuw_vils_matched.csv')
t.index = pd.date_range('1976-01-01', periods=len(t), freq='D')
# TUWmodel sets PET to zero below -0.1 C; give WaterSim the same PET series and no snow-cover PET reduction
E = np.where(t['T'] < -0.1, 0.0, t['E'])
f2 = pd.DataFrame({'P': t['P'], 'T_mean': t['T'], 'PET': E}, index=t.index)
tuw = {'SCF': 1.02, 'DDF': 1.70, 'Tr': 2.0, 'Ts': 0.0, 'Tm': 1.0, 'LPrat': 0.934, 'FC': 121, 'BETA': 2.52, 'k0': 0.473,
       'k1': 9.06, 'k2': 142, 'lsuz': 50.1, 'cperc': 2.38}
p = {'TT': 1.0, 'TTI': 2.0, 'CFMAX': 1.70, 'SFCF': 1.02, 'CFR': 0.0, 'CWH': 0.0, 'TLAPSE': -6.5, 'PGRAD': 0.0,
     'FC': 121.0, 'LP': 0.934, 'BETA': 2.52, 'PERC': 2.38, 'UZL': 50.1,
     'K0': np.exp(-1 / 0.473) / 0.473, 'K1': np.exp(-1 / 9.06) / 9.06, 'K2': np.exp(-1 / 142.0) / 142.0, 'MAXBAS': 1.0}
# K = exp(-1/k)/k is the share of the store TUWmodel releases per day (its respfunc: q0 = (SUZ-LSUZ)/k0 exp(-1/k0) etc.)
spec2 = {'id': 'vils', 'area_km2': 198.0, 'lat': 47.5, 'elev_mean_m': 1300}
o2 = ws.simulate(spec2, f2, model='hbv', params=p, snow='degree_day', routing='none', pet='forcing',
                 pet_snow_reduction=False, init={'model': {'SM': 50.0, 'SUZ': 2.5, 'SLZ': 2.5}})
ts2 = o2['ts']
cmp = {}
for ours, theirs in (('SWE', 'swe'), ('soil', 'moist'), ('ET_soil', 'eta'), ('rain', 'rain'), ('melt', 'melt')):
    a = ts2[ours].values
    b = t[theirs].values
    if theirs == 'rain':
        b = t['rain'].values
    cmp[ours + '_vs_' + theirs] = {'max_abs_diff_mm': float(np.max(np.abs(a - b))), 'mean_ref': float(np.mean(b))}
cmp['snowfall_vs_snow_x_SCF'] = {'max_abs_diff_mm': float(np.max(np.abs(ts2['snowfall'].values - 1.02 * t['snow'].values)))}
q_ours = ts2['Q'].values
o3 = ws.simulate(spec2, f2, model='hbv', params={**p, 'MAXBAS': 10.0}, snow='degree_day', routing='none', pet='forcing',
                 pet_snow_reduction=False, init={'model': {'SM': 50.0, 'SUZ': 2.5, 'SLZ': 2.5}}, labels=False)
res['hbv_vs_TUWmodel'] = {
    'case': 'TUWmodel help example, lumped Vils (198 km2, Austria) 1976-2008, parameters of ?TUWmodel with the melt '
            'threshold Tm set to (Tr+Ts)/2 = 1 C (HBV-light has one threshold), routing off (bmax=1) or fixed (bmax=10)',
    'tuw_params': tuw, 'watersim_params': p, 'n_days': len(t),
    'snow_and_soil_routines': cmp,
    'note_response': 'The response routines differ by design: TUWmodel releases (S/k) exp(-1/k) per day and lets '
                     'part of the percolation reach the lower-box outflow on the same day, in a different order; '
                     'HBV-light (WaterSim) uses an explicit daily step (PERC, then Q0, Q1, Q2). WaterSim is given '
                     'K = exp(-1/k)/k (the same daily release share). The triangular weights also differ (TUW: '
                     'discrete half-day midpoints). So total runoff is compared by NSE/KGE/volume, not to round-off.',
    'runoff_no_routing': {'NSE_ours_vs_tuw': M.nse(t['q'].values, q_ours), 'KGE': M.kge(t['q'].values, q_ours),
                          'volume_ratio': float(q_ours.sum() / t['q'].sum())},
    'runoff_maxbas10': {'NSE_ours_vs_tuw': M.nse(t['q_route10'].values, o3['ts']['Q'].values),
                        'KGE': M.kge(t['q_route10'].values, o3['ts']['Q'].values),
                        'volume_ratio': float(o3['ts']['Q'].sum() / t['q_route10'].sum())},
    'ledger_ok': o2['ledger']['ok'],
}
res['hbv_vs_TUWmodel']['pass'] = bool(max(v['max_abs_diff_mm'] for v in cmp.values()) < 1e-3 and
                                      res['hbv_vs_TUWmodel']['runoff_no_routing']['NSE_ours_vs_tuw'] > 0.98)
res['hbv_light_published_example'] = ('not reproduced: HBV-light (Seibert & Vis 2012) is a Windows GUI program with no '
                                      'published numeric test series we could download; TUWmodel (an independent HBV '
                                      'code with published example data) stands in')
write('reference_models.json', res)
print({k: v.get('pass') for k, v in res.items() if isinstance(v, dict)})
print(res['gr4j_vs_airGR']['diffs'], res['gr4j_vs_airGR']['watersim_NSE_1990_1999'])
print(res['hbv_vs_TUWmodel']['snow_and_soil_routines'], res['hbv_vs_TUWmodel']['runoff_no_routing'], res['hbv_vs_TUWmodel']['runoff_maxbas10'])
