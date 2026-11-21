"""Dead fuel moisture checks.

1. Isotherm: Nelson (1984) form fitted to Simard (1968) EMC; error over T 0-40 C, RH 10-95 %.
2. Timelag: step response of the 1-, 10-, 100-h sticks (constant 20 C, RH 20 -> 80 %, no sun): e-folding time
   vs the nominal class timelag (the definition of the classes).
3. Diurnal plausibility on a generated dry summer week: 1-h daily range, 10-h damping, lag of the 1-h minimum
   behind the RH minimum; comparison with the FFMC-implied moisture (Van Wagner 1987) at noon.
4. Rain response: 10 mm in 2 h raises the 10-h stick toward fibre saturation and it dries back over ~1-2 days.
5. Real 10-h fuel sticks on dev sites (B's data) are compared in check_moisture_real.py when B's data exist.
"""
import numpy as np
import pandas as pd

from _common import write
import firesim as fs
from firesim import moisture as M

out = {}
T = np.linspace(0, 40, 21)
H = np.linspace(10, 95, 35)
TT, HH = np.meshgrid(T, H)
nel = np.vectorize(lambda t, h: 100 * M.emc_nelson(t, h / 100, M.ISO_A, M.ISO_B))(TT, HH)
sim = M.simard_emc(TT, HH)
d = nel - sim
out['isotherm'] = dict(a=M.ISO_A, b=M.ISO_B, rmse_pct_points=float(np.sqrt(np.mean(d ** 2))),
                       max_abs_pct_points=float(np.max(np.abs(d))),
                       at_21C=dict(rh=[10, 20, 30, 50, 70, 90], nelson=[float(100 * M.emc_nelson(21, h / 100, M.ISO_A, M.ISO_B)) for h in (10, 20, 30, 50, 70, 90)],
                                   simard=[float(M.simard_emc(21, h)) for h in (10, 20, 30, 50, 70, 90)]))
print('isotherm', out['isotherm']['rmse_pct_points'], out['isotherm']['max_abs_pct_points'])

# 2. timelag step response
idx = pd.date_range('2021-07-01', periods=24 * 60, freq='h', tz='UTC')
w = pd.DataFrame(dict(T=20.0, RH=20.0, wind_speed=2.0, wind_dir=0.0, precip=0.0, srad=0.0), index=idx)
w.iloc[24 * 20:, w.columns.get_loc('RH')] = 80.0
mo = M.dead_moisture(w, classes=('1h', '10h', '100h'))
tl = {}
for c, nom in (('1h', 1), ('10h', 10), ('100h', 100)):
    s = mo['m' + c].to_numpy()
    k0 = 24 * 20 - 1
    m0 = s[k0]
    mf = 100 * M.emc_nelson(20.0, 0.80, M.ISO_A, M.ISO_B)
    frac = (s[k0:] - m0) / (mf - m0)
    j = np.argmax(frac >= 1 - np.exp(-1))
    # sub-hour interpolation
    if j > 0:
        tau = (j - 1) + (1 - np.exp(-1) - frac[j - 1]) / (frac[j] - frac[j - 1])
    else:
        tau = np.nan
    tl[c] = dict(nominal_h=nom, efold_h=float(tau), rel_err=float(tau / nom - 1))
out['timelag'] = tl
print('timelag', tl)

# 3. diurnal week
wg = fs.generate_weather('2021-07-10', hours=24 * 10, climate='west_dry_summer', lat=44.0, lon=-116.0, rng=3)
mo = M.dead_moisture(wg)
lst = (mo.index + pd.Timedelta(hours=-116 / 15)).hour
day = (mo.index + pd.Timedelta(hours=-116 / 15)).normalize()
g = mo.groupby(day)
rng1 = (g['m1h'].max() - g['m1h'].min()).iloc[2:-1]
rng10 = (g['m10h'].max() - g['m10h'].min()).iloc[2:-1]
rng100 = (g['m100h'].max() - g['m100h'].min()).iloc[2:-1]
lag = []
for dd, gg in mo.groupby(day):
    if len(gg) < 24:
        continue
    wl = wg.loc[gg.index]
    lag.append(((gg['m1h'].idxmin() - wl['RH'].idxmin()) / pd.Timedelta(hours=1)))
dly = fs.fwi.daily_from_hourly(wg, lon=-116.0)
f = fs.fwi.fwi_series(dly, ffmc0=88.0)
mf_ffmc = 147.2 * (101 - f['FFMC']) / (59.5 + f['FFMC'])
noon = mo[lst == 12]
noon.index = (noon.index + pd.Timedelta(hours=-116 / 15)).tz_localize(None).normalize()
j = noon.join(mf_ffmc.rename('m_ffmc'), how='inner').iloc[2:]
out['diurnal'] = dict(m1h_daily_range_mean=float(rng1.mean()), m10h_daily_range_mean=float(rng10.mean()),
                      m100h_daily_range_mean=float(rng100.mean()),
                      m1h_min_lag_after_rh_min_h_median=float(np.median(lag)),
                      m1h_noon_mean=float(j['m1h'].mean()), ffmc_moisture_noon_mean=float(j['m_ffmc'].mean()),
                      corr_m1h_vs_ffmc_moisture=float(np.corrcoef(j['m1h'], j['m_ffmc'])[0, 1]),
                      note_ffmc='FFMC describes litter with a ~16-h timelag; its moisture scale is not the 1-h stick',
                      expected='1-h range several %, 10-h range smaller, 100-h smallest; 1-h minimum at or shortly '
                               'after the RH minimum; noon 1-h moisture similar to FFMC-implied moisture')
print('diurnal', out['diurnal'])
# 4. rain pulse
idx = pd.date_range('2021-07-01', periods=24 * 8, freq='h', tz='UTC')
w = pd.DataFrame(dict(T=22.0, RH=35.0, wind_speed=2.0, wind_dir=0.0, precip=0.0, srad=0.0), index=idx)
w.iloc[48:50, w.columns.get_loc('precip')] = 5.0
w.iloc[48:50, w.columns.get_loc('RH')] = 95.0
mo = M.dead_moisture(w)
pk = mo['m10h'].iloc[48:60].max()
base = mo['m10h'].iloc[47]
back = mo['m10h'].iloc[50:]
half = (back <= base + 0.5 * (pk - base)).to_numpy()
t_half = float(np.argmax(half)) if half.any() else None
out['rain'] = dict(m10h_before=float(base), m10h_peak=float(pk), m1h_peak=float(mo['m1h'].iloc[48:60].max()),
                   m100h_change=float(mo['m100h'].iloc[48:60].max() - mo['m100h'].iloc[47]),
                   m10h_hours_to_half_recovery=t_half)
print('rain', out['rain'])
write('moisture.json', dict(what='Dead fuel moisture model checks (synthetic); real 10-h sticks in moisture_real.json',
                            params=M.DEFAULT_NELSON, **out))
