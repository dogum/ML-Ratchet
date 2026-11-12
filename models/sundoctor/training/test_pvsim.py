"""Check SunDoctor's vectorized physics against pvlib, block by block."""
import _paths as _P  # portable paths (see _paths.py)
import sys, numpy as np, pandas as pd, pvlib, json
sys.path.insert(0, _P.SRC)
import pvsim as S

res = {}
lat, lon = 38.996, -122.134
times = pd.date_range('2019-01-01', '2019-12-31 23:45', freq='15min', tz='UTC')
sp = pvlib.solarposition.get_solarposition(times, lat, lon, altitude=10, method='nrel_numpy')
doy = times.dayofyear.values; ut = times.hour.values + times.minute.values / 60
tsol = ut + lon / 15 + S.eot_minutes(doy) / 60
cz, el, az = S.sun(lat, doy, tsol)
up = sp['apparent_elevation'].values > 5
res['sun_elevation_maxerr_deg'] = float(np.max(np.abs(el[up] - sp['elevation'].values[up])))
daz = (az[up] - sp['azimuth'].values[up] + 180) % 360 - 180
res['sun_azimuth_maxerr_deg'] = float(np.max(np.abs(daz)))
# clear sky (same Linke turbidity, sea level)
tl = 3.0
cs = pvlib.clearsky.ineichen(sp['apparent_zenith'], pvlib.atmosphere.get_absolute_airmass(pvlib.atmosphere.get_relative_airmass(sp['apparent_zenith'])),
                             linke_turbidity=tl, altitude=0, dni_extra=pvlib.irradiance.get_extra_radiation(times, method='spencer'))
g, dn, dh = S.ineichen(np.cos(np.radians(sp['apparent_zenith'].values)), doy, tl)
res['clearsky_ghi_rel_err_noon'] = float(np.median(np.abs(g[up] - cs['ghi'].values[up]) / np.maximum(cs['ghi'].values[up], 50)))
res['clearsky_dni_rel_err'] = float(np.median(np.abs(dn[up] - cs['dni'].values[up]) / np.maximum(cs['dni'].values[up], 50)))
# Erbs
e = pvlib.irradiance.erbs(cs['ghi'] * 0.6, sp['zenith'], times)
dni2, dhi2 = S.erbs(cs['ghi'].values * 0.6, np.cos(np.radians(sp['zenith'].values)), doy)
res['erbs_dhi_rel_err'] = float(np.median(np.abs(dhi2[up] - e['dhi'].values[up]) / np.maximum(e['dhi'].values[up], 20)))
# Hay-Davies transposition, 25 deg tilt facing south
tilt, azm = 25.0, 180.0
tot = pvlib.irradiance.get_total_irradiance(tilt, azm, sp['zenith'], sp['azimuth'], cs['dni'], cs['ghi'], cs['dhi'],
                                            dni_extra=pvlib.irradiance.get_extra_radiation(times, method='spencer'), model='haydavies', albedo=0.2)
ca = S.cos_aoi(tilt, azm, np.cos(np.radians(sp['zenith'].values)), sp['azimuth'].values)
b, sk, gn = S.haydavies(tilt, ca, np.cos(np.radians(sp['zenith'].values)), cs['dni'].values, cs['dhi'].values, cs['ghi'].values, doy, 0.2)
res['poa_rel_err'] = float(np.median(np.abs((b + sk + gn)[up] - tot['poa_global'].values[up]) / np.maximum(tot['poa_global'].values[up], 50)))
# Faiman and PVWatts
poa = np.linspace(0, 1100, 50); ta = np.full(50, 25.0); ws = np.full(50, 2.0)
res['faiman_maxerr_C'] = float(np.max(np.abs(S.faiman(poa, ta, ws) - pvlib.temperature.faiman(poa, ta, ws))))
tc = S.faiman(poa, ta, ws)
res['pvwatts_dc_maxerr'] = float(np.max(np.abs(S.pvwatts_dc(poa, tc, 1.0, -0.004) - pvlib.pvsystem.pvwatts_dc(poa, tc, 1.0, -0.004))))
pdc = np.linspace(0.001, 1.2, 50)
res['pvwatts_inv_maxerr'] = float(np.max(np.abs(S.pvwatts_inv(pdc, 1.0, 0.96, 0.96) - np.asarray(pvlib.inverter.pvwatts(pdc, 1.0, 0.96)))))
iam = pvlib.iam.ashrae(np.degrees(np.arccos(np.clip(ca[up], 0.05, 1))), b=0.05)
res['iam_maxerr'] = float(np.max(np.abs(S.iam_ashrae(ca[up], 0.05) - np.clip(iam, 0, 1))))
print(json.dumps({k: round(v, 5) for k, v in res.items()}, indent=1))
json.dump(res, open(_P.WORK + '/physics_check.json', 'w'), indent=1)
