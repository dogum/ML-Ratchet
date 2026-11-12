"""Single-axis tracker geometry (horizontal axis), vectorized, with backtracking; checked against pvlib.tracking.singleaxis."""
import _paths as _P  # portable paths (see _paths.py)
import numpy as np
D2R = np.pi / 180

def rotation(cz, az, axis_az=180.0, max_angle=55.0, gcr=0.4, backtrack=True):
    """ideal rotation angle (deg; + toward the west for a north-south axis) from cos(zenith) and sun azimuth"""
    sz = np.sqrt(np.clip(1 - cz ** 2, 0, 1))
    x = sz * np.sin((az - axis_az) * D2R)                  # sun component across the axis, horizontal
    th = np.arctan2(x, np.maximum(cz, 1e-6))
    if backtrack:
        temp = np.abs(np.cos(th)) / gcr
        corr = np.where(temp < 1, -np.sign(th) * np.arccos(np.clip(temp, -1, 1)), 0.0)
        th = th + corr
    th = np.clip(th / D2R, -max_angle, max_angle)
    return np.where(cz > 0, th, 0.0)

def true_angle(cz, az, axis_az=180.0):
    """projected solar zenith angle across the axis (deg): where a perfect, unlimited tracker would point"""
    sz = np.sqrt(np.clip(1 - cz ** 2, 0, 1)); return np.arctan2(sz * np.sin((az - axis_az) * D2R), np.maximum(cz, 1e-6)) / D2R

def shaded_fraction(theta, theta_s, gcr):
    """row-to-row shaded fraction of a tracker row at rotation theta, sun at projected zenith theta_s (flat ground;
    Anderson & Jensen 2024, as pvlib.shading.shaded_fraction1d)"""
    f = 1 - (1 / gcr) * np.cos(theta_s * D2R) / np.maximum(np.abs(np.cos((theta - theta_s) * D2R)), 1e-6)
    return np.clip(f, 0, 1)

def surface(theta, axis_az=180.0):
    """rotation angle -> (surface tilt, surface azimuth)"""
    tilt = np.abs(theta); saz = np.where(theta >= 0, axis_az + 90.0, axis_az - 90.0) % 360
    return tilt, saz

def cos_aoi(theta, cz, az, axis_az=180.0):
    sz = np.sqrt(np.clip(1 - cz ** 2, 0, 1)); t = theta * D2R
    # surface normal for a horizontal axis: tilted by theta about the axis
    return np.cos(t) * cz + np.sin(t) * sz * np.sin((az - axis_az) * D2R)

if __name__ == '__main__':
    import pandas as pd, pvlib, json, sys
    sys.path.insert(0, _P.SRC); import pvsim as S
    times = pd.date_range('2019-01-01', '2019-12-31 23:45', freq='15min', tz='UTC')
    sp = pvlib.solarposition.get_solarposition(times, 40.386, -104.551)
    up = sp['apparent_elevation'].values > 3
    ref = pvlib.tracking.singleaxis(sp['apparent_zenith'], sp['azimuth'], axis_tilt=0, axis_azimuth=180, max_angle=55, backtrack=True, gcr=0.4)
    th = rotation(np.cos(np.radians(sp['apparent_zenith'].values)), sp['azimuth'].values, 180, 55, 0.4)
    ca = cos_aoi(th, np.cos(np.radians(sp['apparent_zenith'].values)), sp['azimuth'].values)
    ok = up & np.isfinite(ref['tracker_theta'].values)
    th_tt = rotation(np.cos(np.radians(sp['apparent_zenith'].values)), sp['azimuth'].values, 180, 55, 0.4, backtrack=False)
    ts = true_angle(np.cos(np.radians(sp['apparent_zenith'].values)), sp['azimuth'].values)
    sf = shaded_fraction(th_tt, ts, 0.4)
    sf_ref = pvlib.shading.shaded_fraction1d(sp['apparent_zenith'], sp['azimuth'], 180, th_tt, collector_width=2.0, pitch=5.0)
    res_sf = float(np.max(np.abs(sf[ok] - np.asarray(sf_ref)[ok])))
    res = {'tracker_shaded_fraction_maxerr': res_sf, 'tracker_theta_maxerr_deg': float(np.max(np.abs(th[ok] - ref['tracker_theta'].values[ok]))),
           'tracker_aoi_maxerr_deg': float(np.max(np.abs(np.degrees(np.arccos(np.clip(ca[ok], -1, 1))) - ref['aoi'].values[ok])))}
    print(json.dumps(res)); d = json.load(open(_P.WORK + '/physics_check.json')); d.update(res); json.dump(d, open(_P.WORK + '/physics_check.json', 'w'), indent=1)
