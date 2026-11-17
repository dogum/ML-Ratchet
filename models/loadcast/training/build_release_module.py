"""Write release/loadcast.py: the feature code of lc_common copied verbatim (inspect.getsource) + numpy inference.
Run after choosing the rung; the weights are written by export.py."""
import inspect, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lc_common as C, lcnet as N

FUNCS = ['country_of_tz', '_easter', '_nth_dow', '_last_dow', '_observed', 'holiday_dates', 'is_holiday',
         '_rolling_mean_past', '_ewm', 'weather_features', 'calendar_features', 'annual_climate', 'meters_of',
         'hvac_class_from_meters', 'static_features', 'towt_X', 'towt_fit', 'towt_predict', 'cp_fit',
         'history_encode', 'history_hourly', 'g14']
HEAD = '''"""LoadCast v1.0: hourly building energy per meter from weather, calendar and metadata, with 90%% error bars.
numpy / pandas / scipy only. Trained only on BuildingSim simulations (see MODEL_CARD.md).

    import loadcast as lc
    m = lc.LoadCast()                                   # loads loadcast_weights.npz next to this file
    fc = m.forecast(weather, tz='America/New_York', lat=40.2, use='education', area_m2=8000, meter='electricity',
                    meters_present=['electricity', 'chilledwater', 'steam'], start='2017-01-01', end='2018-01-01',
                    history=None)                       # history: hourly kWh Series (UTC) -> warm mode
    fc[['kW', 'lo90', 'hi90']]

weather: hourly UTC-indexed DataFrame with T_out (degC), dewpoint (degC), GHI (W/m2), wind_speed (m/s). It must cover
the history window (warm mode) and the forecast window, plus 7 days before each (rolling features). The annual
climate summary comes from the forecast window's local calendar year(s) present in the frame: give at least a year.
"""
import functools, os
import numpy as np, pandas as pd

METERS = %r
IREF = %r
USES = %r
HVAC_CLASSES = %r
SCHED_CLASSES = %r
COUNTRY = %r
KNOTS = np.array(%r)
STATIC_NAMES = %r
WX_NAMES = %r
CAL_NAMES = %r
HIST_NAMES = %r
QMIN = 0.1
Z90 = 1.6449

''' % (C.METERS, C.IREF, C.USES, C.HVAC_CLASSES, C.SCHED_CLASSES, C.COUNTRY, C.KNOTS.tolist(), C.STATIC_NAMES,
       C.WX_NAMES, C.CAL_NAMES, C.HIST_NAMES)

TAIL = '''

def np_forward(W, x, base):
    h = (x - W['mu_x']) / W['sd_x']
    for i in sorted(int(k.split('.')[1]) for k in W if k.startswith('body.') and k.endswith('.weight')):
        h = h @ W[f'body.{i}.weight'].T + W[f'body.{i}.bias']
        h = h / (1.0 + np.exp(-np.clip(h, -60, 60)))      # SiLU
    o = h @ W['head.weight'].T + W['head.bias']
    return base + o[:, 0], np.clip(o[:, 1], -7, 4)


class LoadCast:
    def __init__(self, weights=None):
        weights = weights or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'loadcast_weights.npz')
        z = np.load(weights, allow_pickle=False)
        self.W = {}
        for mode in ('cold', 'warm'):
            self.W[mode] = {k[len(mode) + 1:]: z[k].astype(np.float64) for k in z.files if k.startswith(mode + '/')}
        self.widen = {str(k): float(v) for k, v in zip(z['widen_keys'], z['widen_vals'])}

    @staticmethod
    def _frame(weather, tz):
        w = weather.sort_index()
        loc = w.index.tz_convert(tz); naive = loc.tz_localize(None); days = naive.normalize()
        T = w['T_out'].interpolate(limit_direction='both').values
        dew = np.minimum(w['dewpoint'].interpolate(limit_direction='both').values, T)
        ghi = w['GHI'].fillna(0).clip(lower=0).values
        wind = w['wind_speed'].interpolate(limit_direction='both').fillna(3).values
        dcode = ((days - days[0]) / pd.Timedelta(days=1)).astype(int).values
        hol = is_holiday(days, country_of_tz(tz))
        wx = weather_features(T, dew, ghi, wind, dcode)
        cal = calendar_features(loc.hour.values, loc.dayofweek.values, hol, loc.month.values, loc.day.values,
                                loc.dayofyear.values)
        Tday = pd.Series(T).groupby(dcode).transform('mean').values
        return dict(idx=w.index, T=T, Tday=Tday, T24=wx[:, 1] * 10, wx=wx, cal=cal, dcode=dcode, hol=hol,
                    hour=loc.hour.values, tow=loc.dayofweek.values * 24 + loc.hour.values, year=loc.year.values)

    def forecast(self, weather, tz, lat, use, area_m2, meter, meters_present=None, start=None, end=None,
                 history=None, year_built=None, hvac='unknown', sched='unknown', calibrated=True, frame=None):
        """Returns a DataFrame (UTC index over [start, end)) with kW, sigma, lo90, hi90 and mode ('cold'/'warm')."""
        f = frame or self._frame(weather, tz)
        idx = f['idx']
        t0 = pd.Timestamp(start) if start is not None else idx[0]
        t1 = pd.Timestamp(end) if end is not None else idx[-1] + pd.Timedelta(hours=1)
        if t0.tzinfo is None: t0 = t0.tz_localize(tz).tz_convert('UTC')
        if t1.tzinfo is None: t1 = t1.tz_localize(tz).tz_convert('UTC')
        tsel = np.flatnonzero((idx >= t0) & (idx < t1))
        yrs = np.unique(f['year'][tsel])
        clim = annual_climate(f['T'][np.isin(f['year'], yrs)])
        meters_present = list(meters_present or [meter])
        if hvac == 'unknown': hvac = hvac_class_from_meters(meters_present)
        st = static_features(use, area_m2, year_built, meter, meters_present, hvac, sched, clim, lat)
        unit = area_m2 * IREF[meter] / 1000.0           # kW per model unit
        X = [np.broadcast_to(st, (len(tsel), len(st))), f['wx'][tsel], f['cal'][tsel]]
        if history is None:
            mode = 'cold'; base = np.zeros(len(tsel)); Q = 1.0
        else:
            mode = 'warm'
            h = history.reindex(idx).values.astype(float) / unit
            hi = np.flatnonzero((idx >= history.index.min()) & (idx <= history.index.max()))
            H = history_encode(h[hi], f['T'][hi], f['tow'][hi], f['dcode'][hi], f['hol'][hi], f['hour'][hi], 1.0)
            hh = history_hourly(H, f['T'][tsel], f['tow'][tsel], f['Tday'][tsel], f['T24'][tsel])
            X.append(hh); Q = max(H['S'], QMIN); base = hh[:, 0] * H['S'] / Q
        X = np.concatenate(X, 1)
        mu, ls = np_forward(self.W[mode], X, base)
        weeks = None if history is None else len(history.dropna()) / 168.0
        variant = 'full' if mode == 'cold' or weeks > 20 else ('w8' if weeks > 4 else 'w2')
        k = self.widen.get(f'{mode}_{variant}_{meter}', 1.0) if calibrated else 1.0
        kW = np.clip(mu, 0, None) * Q * unit; sd = np.exp(ls) * Q * unit * k
        out = pd.DataFrame(dict(kW=kW, sigma=sd, lo90=np.maximum(kW - Z90 * sd, 0), hi90=kW + Z90 * sd), index=idx[tsel])
        out.attrs['mode'] = mode; out.attrs['widen'] = k
        return out
'''


def main():
    src = HEAD + '\n\n'.join(inspect.getsource(getattr(C, f)) for f in FUNCS) + TAIL
    open(f'{C.ROOT}/release/loadcast.py', 'w').write(src)
    print('wrote release/loadcast.py', len(src))


if __name__ == '__main__':
    main()
