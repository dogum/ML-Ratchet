"""Hourly ASOS weather (Iowa Environmental Mesonet) interpolated to 1 Hz trip times.
Stations: KARB (Ann Arbor, primary), KYIP (Willow Run), KDTW (Detroit Metro) as fall-backs.
Times are seconds since 2017-11-01 00:00 UTC."""
import os as _os
# the shared vehicle platform: shared/vehicle in the repo, or $ML_RATCHET_VEHICLE
_VEH = _os.environ.get('ML_RATCHET_VEHICLE') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import numpy as np, pandas as pd

DATA = _VEH + '/data'
STATIONS = {'ARB': dict(lat=42.224, lon=-83.7397, elev=251.0),
            'YIP': dict(lat=42.2379, lon=-83.5304, elev=215.0),
            'DTW': dict(lat=42.23, lon=-83.33, elev=195.0)}
EPOCH = pd.Timestamp('2017-11-01 00:00')
RD, RV = 287.058, 461.495


def _station_pressure_hpa(alti_inhg, elev_m):
    a = alti_inhg * 33.8639
    return (a ** 0.190263 - 8.417286e-5 * elev_m) ** (1 / 0.190263)


def _vapour_pressure_hpa(td_c):
    return 6.112 * np.exp(17.67 * td_c / (td_c + 243.5))


class Weather:
    def __init__(self):
        self.st = {}
        for s in STATIONS:
            w = pd.read_csv(f'{DATA}/raw/weather/asos_{s}.csv', na_values=['M'], low_memory=False)
            w['t'] = (pd.to_datetime(w.valid) - EPOCH).dt.total_seconds()
            w = w.sort_values('t').drop_duplicates('t', keep='last')
            w['p_stn'] = _station_pressure_hpa(w.alti, STATIONS[s]['elev'])
            ws = w.sknt * 0.514444
            rad = np.radians(w.drct)
            # wind vector pointing TO; 'drct' is direction the wind comes FROM
            w['u'] = -ws * np.sin(rad); w['v'] = -ws * np.cos(rad)
            w.loc[ws == 0, ['u', 'v']] = 0.0
            w['ws'] = ws
            wx = w.wxcodes.fillna('').astype(str)
            p01 = pd.to_numeric(w.p01i.replace('T', '0.001'), errors='coerce').fillna(0)
            snow = wx.str.contains('SN|PL|GS|FZ')
            wet = wx.str.contains('RA|DZ|TS|UP') | (p01 > 0)
            w['surface'] = np.where(snow, 4, np.where(wet, 2, 0))
            self.st[s] = w

    def _series(self, var, t, max_gap=2 * 3600):
        """Interpolate var at times t from ARB, falling back to YIP then DTW where ARB has no obs within max_gap."""
        out = np.full(len(t), np.nan); src = np.full(len(t), '', dtype=object)
        for s in ['ARB', 'YIP', 'DTW']:
            w = self.st[s]; ok = w[var].notna().values
            tt = w.t.values[ok]; vv = w[var].values[ok].astype(float)
            if len(tt) < 2: continue
            i = np.clip(np.searchsorted(tt, t), 1, len(tt) - 1)
            gap = np.minimum(np.abs(t - tt[i - 1]), np.abs(tt[i] - t))
            val = np.interp(t, tt, vv)
            take = np.isnan(out) & (gap <= max_gap)
            out[take] = val[take]; src[take] = s
        return out, src

    def at(self, t_utc_s, elev_m):
        """Return dict of 1 Hz arrays: T_amb, wind_speed, wind_dir, rho_air, p_hpa, surface, and QC info."""
        t = np.asarray(t_utc_s, float)
        T, srcT = self._series('tmpc', t)
        Td, _ = self._series('dwpc', t)
        u, _ = self._series('u', t); v, _ = self._series('v', t)
        ws, _ = self._series('ws', t)
        ps, srcP = self._series('p_stn', t)
        wd = (np.degrees(np.arctan2(-u, -v)) + 360) % 360  # direction the wind comes FROM
        wd = np.where(ws < 0.1, 0.0, wd)
        # pressure from station elevation to vehicle elevation (hypsometric, using station by source)
        st_el = np.array([STATIONS[s]['elev'] if s else 251.0 for s in srcP])
        z = np.where(np.isfinite(elev_m), elev_m, 251.0)
        Tk = T + 273.15
        p = ps * np.exp(-9.80665 * (z - st_el) / (RD * Tk))
        e = np.where(np.isfinite(Td), _vapour_pressure_hpa(np.minimum(Td, T)), 0.0)
        rho = (p - e) * 100 / (RD * Tk) + e * 100 / (RV * Tk)
        # surface: weather code of the latest ARB report within the past 75 min (wet persists briefly)
        w = self.st['ARB']; tt = w.t.values
        j = np.clip(np.searchsorted(tt, t, side='right') - 1, 0, len(tt) - 1)
        surf = np.where(t - tt[j] <= 4500, w.surface.values[j], 0)
        # ARB reports since previous hour: treat wet if either of the last two reports was wet/snow
        jm = np.clip(j - 1, 0, len(tt) - 1)
        surf = np.maximum(surf, np.where((t - tt[jm] <= 4500), w.surface.values[jm], 0))
        return dict(T_amb=T.astype(np.float32), wind_speed=ws.astype(np.float32), wind_dir=wd.astype(np.float32),
                    rho_air=rho.astype(np.float32), p_hpa=p.astype(np.float32), surface=surf.astype(np.int8),
                    src_T=srcT, src_P=srcP)
