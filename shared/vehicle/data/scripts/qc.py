"""Quality checks for VED grade/elevation and weather join -> proc/checks/qc_summary.json"""
import os as _os
# the shared vehicle platform: shared/vehicle in the repo, or $ML_RATCHET_VEHICLE
_VEH = _os.environ.get('ML_RATCHET_VEHICLE') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, _VEH)
import realdata as rd
from realdata.geo import utm_forward
from realdata.weather import Weather
ROOT = _VEH + '/data'
idx = rd.ved_index()
ok = idx[idx.q_route_ok]
out = {}

# ---- grade repeatability across trips/vehicles on the same road cell and direction ----
samp = ok.sample(min(6000, len(ok)), random_state=1)
recs = []
for r in samp.itertuples():
    with np.load(f'{ROOT}/proc/ved/{r.path}') as z:
        v, g, lat, lon, h = z['v'], z['grade'], z['lat'], z['lon'], z['heading']
    m = (v > 5) & np.isfinite(g) & np.isfinite(h)
    if m.sum() < 10: continue
    x, y = utm_forward(lat[m], lon[m])
    recs.append(pd.DataFrame(dict(cx=(x // 20).astype(int), cy=(y // 20).astype(int), hb=((h[m] + 22.5) // 45 % 8).astype(int),
                                  g=g[m], veh=r.vehicle_id, trip=r.trip_id)))
G = pd.concat(recs)
# one value per trip per cell-direction (mean), then spread across trips
T = G.groupby(['cx', 'cy', 'hb', 'trip']).agg(g=('g', 'mean'), veh=('veh', 'first')).reset_index()
C = T.groupby(['cx', 'cy', 'hb']).agg(n=('g', 'size'), nv=('veh', 'nunique'), gm=('g', 'mean'), gs=('g', 'std')).reset_index()
C5 = C[(C.n >= 5) & (C.nv >= 3)]
out['grade_repeatability'] = {
    'cells_with_5plus_trips_3plus_vehicles': int(len(C5)),
    'median_between_trip_std_of_grade': float(C5.gs.median()), 'p90_between_trip_std': float(C5.gs.quantile(0.9)),
    'spread_of_cell_mean_grade_std': float(C5.gm.std()), 'abs_grade_p50_p90_p99': [float(np.nanpercentile(np.abs(G.g), q)) for q in (50, 90, 99)],
    'explained_fraction': float(1 - (C5.gs ** 2).mean() / (C5.gm.var() + (C5.gs ** 2).mean()))}
# opposite directions: g(h) + g(h+180) should be ~0
Cm = C[(C.n >= 3)].set_index(['cx', 'cy', 'hb']).gm
pairs = []
for (cx, cy, hb), gm in Cm.items():
    if hb < 4 and (cx, cy, hb + 4) in Cm.index:
        pairs.append((gm, Cm[(cx, cy, hb + 4)]))
P = np.array(pairs)
out['grade_direction_antisymmetry'] = {'n_cell_pairs': int(len(P)), 'median_abs_sum': float(np.median(np.abs(P.sum(1)))),
                                        'median_abs_grade': float(np.median(np.abs(P[:, 0]))),
                                        'corr_forward_vs_minus_backward': float(np.corrcoef(P[:, 0], -P[:, 1])[0, 1])}
out['elevation_cleaning'] = {'trips_with_any_closing_fill_gt_2m': float((ok.elev_closing_frac > 0).mean()),
                             'median_fraction_of_track_filled': float(ok.elev_closing_frac.median()),
                             'p95_fraction_filled': float(ok.elev_closing_frac.quantile(0.95)),
                             'outside_dem_trip_frac': float((idx.elev_nodem_frac > 0).mean()),
                             'elev_gain_m_per_km_median': float((ok.elev_gain_m / (ok.distance_m / 1000)).median())}
out['gps_vs_odometer_length_ratio'] = {q: float(ok.gps_len_over_odo.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}

# ---- weather ----
W = Weather()
a = W.st['ARB'].set_index('t'); b = W.st['YIP'].set_index('t')
j = a[['tmpc', 'ws', 'p_stn']].join(b[['tmpc', 'ws', 'p_stn']], rsuffix='_yip', how='inner').dropna()
out['weather_station_agreement_ARB_vs_YIP'] = {
    'n_hours': int(len(j)), 'T_diff_median_abs_C': float((j.tmpc - j.tmpc_yip).abs().median()), 'T_diff_p95_abs_C': float((j.tmpc - j.tmpc_yip).abs().quantile(.95)),
    'wind_diff_median_abs_mps': float((j.ws - j.ws_yip).abs().median()), 'p_stn_diff_median_hpa': float((j.p_stn - (j.p_stn_yip - 0.12 * (251 - 215) / 1.0)).abs().median())}
out['weather_coverage'] = {'trip_T_source_counts': idx.wx_station_T.value_counts().to_dict(),
                           'T_amb_range_C': [float(idx.T_amb_mean.min()), float(idx.T_amb_mean.max())],
                           'T_amb_trip_mean_percentiles_5_50_95': [float(idx.T_amb_mean.quantile(q)) for q in (.05, .5, .95)],
                           'wet_trip_frac': float((idx.wet_frac > 0.5).mean()), 'snow_trip_frac': float((idx.snow_frac > 0.5).mean())}
# OAT vs ASOS: use the second half of each trip (sensor warm-soak decays), trips >= 10 min
s = idx[idx.oat_mean.notna() & (idx.duration_s >= 600)].sample(3000, random_state=2)
d = []
for r in s.itertuples():
    with np.load(f'{ROOT}/proc/ved/{r.path}') as z:
        n = len(z['t']); o = z['oat'][n // 2:]; T = z['T_amb'][n // 2:]
    if np.isfinite(o).any(): d.append((r.vehicle_id, float(np.nanmean(o)), float(np.nanmean(T))))
D = pd.DataFrame(d, columns=['veh', 'oat', 'asos'])
D['e'] = D.oat - D.asos
out['oat_vs_asos'] = {'n_trips': int(len(D)), 'n_vehicles': int(D.veh.nunique()), 'median_diff_C': float(D.e.median()),
                      'MAD_C': float((D.e - D.e.median()).abs().median()), 'corr': float(np.corrcoef(D.oat, D.asos)[0, 1]),
                      'slope_oat_on_asos': float(np.polyfit(D.asos, D.oat, 1)[0]),
                      'MAD_after_per_vehicle_bias_C': float((D.e - D.groupby('veh').e.transform('median')).abs().median()),
                      'vehicles_with_abs_bias_gt_5C': int((D.groupby('veh').e.median().abs() > 5).sum())}
json.dump(out, open(f'{ROOT}/proc/checks/qc_summary.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
