"""Assemble proc/baselines/summary.json (exam baseline table + checks + literature context) and checks.json."""
import json, os, sys, glob
import numpy as np, pandas as pd
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/water (code)
_DATA = os.environ.get('ML_RATCHET_WATER') or _CODE                                      # data root
sys.path.insert(0, _CODE)
import waterdata as wd
from waterdata.models import gr4j_core
from waterdata.metrics import nse

ROOT = os.path.join(_DATA, 'data').replace(os.sep, '/')
B = f'{ROOT}/proc/baselines'


def main():
    out = {'what': 'WaterData exam baselines (Phase 1). Exam units only unless stated. Medians over units.',
           'flow_camels_us': json.load(open(f'{B}/flow_summary.json')),
           'flow_notes': {
               'obs': 'val: Caravan/CAMELS Q; recent: USGS NWIS daily values; both mm/day over spec area_km2',
               'gr4j': 'waterdata.models.gr4j_snow (GR4J + 1-layer degree-day snow), 6 params, DE seed 20261118, NSE objective, '
                       'calibrated per basin on 1999-10-01..2008-09-30 with Caravan ERA5-Land forcing (NOT agent A\'s watersim)',
               'nn_dev / nn_k12': 'ungauged: output average of the 5 attribute-nearest donors (12 standardised attributes)',
               'nwm': 'NOAA NWM v3.0 retrospective daily mean over the local standard day; recent scored WY2016-2022 only. '
                      'NOAA calibrated NWM on many CAMELS/GAGES-II gauges, so it is an operational reference, not an ungauged baseline',
               'persist1': '1-day-lead persistence; peak errors are trivially ~0 because annual maxima are shifted copies',
               'clim': 'day-of-year mean of the basin\'s own calibration-period Q, 31-day smoothing'},
           'literature_context_not_reproduced': {
               'gauged': 'Kratzert et al. (2019, HESS 23, 5089): EA-LSTM on 531 CAMELS basins, Daymet/Maurer/NLDAS forcing, '
                         'same calibration/validation years: median NSE about 0.74 (single) to 0.76 (ensemble).',
               'ungauged': 'Kratzert et al. (2019, WRR 55, 11344): LSTM in 12-fold ungauged cross-validation, median NSE about 0.69.',
               'caveat': 'Different forcing (CAMELS Daymet etc. vs Caravan ERA5-Land) and basin set; context only, never a baseline row.'},
           'detective_camels_us': json.load(open(f'{B}/detective_summary.json')),
           'flood_peaks_camels_us': json.load(open(f'{B}/peak_summary.json')),
           'snow_snotel': json.load(open(f'{B}/snow_summary.json')),
           'snow_notes': 'calibration WY2001-2010, scored WY2011-2025; dd = degree-day with the station\'s own P and T',
           'reservoir_resopsus': json.load(open(f'{B}/reservoir_summary.json'))}
    rs = pd.read_parquet(f'{B}/reservoir_scores.parquet')
    meta = wd.reservoir_meta().set_index('res_id')
    rs['mb_corr'] = rs.res_id.map(meta.mb_corr)
    clean = rs[(rs.split == 'exam') & (rs.mb_corr >= 0.9)]
    out['reservoir_resopsus_mass_balance_ok'] = {p: {'n': int(len(g)), 'J_median': float(g.J.median()),
                                                     'shortage_ratio_median': float(g.shortage_ratio.median()),
                                                     'flood_days_per_yr_median': float(g.flood_days_per_yr.median())}
                                                 for p, g in clean.groupby('policy')}
    rr = pd.read_parquet(f'{B}/reservoir_recent_scores.parquet')
    out['reservoir_cdec_recent_wy2021_2025'] = {p: {'n': int(len(g)), 'J_median': float(g.J.median())}
                                                for p, g in rr[rr.split == 'exam'].groupby('policy')}
    # checks
    o = pd.read_csv(os.path.join(_DATA, 'checks', 'ref', 'airgr_gr4j_explicit.csv'))
    q = gr4j_core(o.P.values.astype(float), o.E.values.astype(float), 257.238, 1.012, 88.235, 2.208, 0.3, 0.5)
    m = pd.to_datetime(o.date) >= '1990-01-01'
    checks = {'gr4j_vs_airGR': {'ref': 'airGR 1.7.9 RunModel_GR4J help example (L0123001), output written by agent A in water/checks/ref',
                                'max_abs_diff_mm': float(np.abs(q - o.Qsim.values).max()),
                                'nse_1990_1999': float(nse(o.Qobs[m], q[m])), 'published': 0.7988},
              'caravan_area_convention': 'implied area / NWIS area median 0.9995 (671 gauges); 36 gauges differ > 2% (see derive_area.py)',
              'caravan_vs_nwis_q': 'NSE(Caravan Q, NWIS Q) WY1990-2014 after the area fix: median 0.999997, min 0.942 (camels_12020000; NWIS revisions), 4 gauges < 0.99; per gauge in proc/baselines/check_caravan_vs_nwis.csv',
              'nwm_local_vs_utc_day': 'val-period median NSE 0.549 (local standard day) vs 0.533 (UTC day); local better at 60% of gauges'}
    json.dump(checks, open(f'{B}/checks.json', 'w'), indent=1)
    out['checks'] = checks
    json.dump(out, open(f'{B}/summary.json', 'w'), indent=1)
    print(json.dumps(out['reservoir_resopsus_mass_balance_ok'], indent=0))
    print(json.dumps(out['reservoir_cdec_recent_wy2021_2025'], indent=0))
    print(checks['gr4j_vs_airGR'])


if __name__ == '__main__':
    main()
