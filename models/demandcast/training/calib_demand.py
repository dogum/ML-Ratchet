"""Fit the building-stock model to real hourly demand of one BA over a set of years (robust nonlinear least squares).

  python3 calib_demand.py dev     -> data/calib/demand_dev.json   (2019, 2021: the physics priors used at exam time)
  python3 calib_demand.py train   -> data/calib/demand_train.json + data/noise/demand_resid.npz
                                     (2018, 2020, 2022: real residual series used as noise in the simulator)
"""
import sys, os, json, time
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from common import *
import buildings as B

FREE = [  # name, lo, hi, transform ('log' or None)
    ('P0', 1e3, 2e5, 'log'), ('trend', -0.08, 0.12, None),
    ('sh_res', 0.15, 0.6, None), ('sh_com', 0.15, 0.6, None),
    ('res_m', 0.0, 0.8, None), ('res_e', 0.0, 1.0, None), ('res_mt', 5.0, 10.0, None), ('res_et', 16.0, 22.0, None),
    ('com_open', 5.0, 10.0, None), ('com_close', 15.0, 22.0, None), ('com_wkend', 0.1, 1.0, None),
    ('ind_wkend', 0.7, 1.05, None), ('wh', -0.05, 0.15, None), ('light', 0.0, 0.2, None),
    ('ua', 0.01, 1.0, 'log'), ('tau1', 1.0, 12.0, None), ('tau2', 12.0, 240.0, None), ('f1', 0.05, 0.95, None),
    ('g_win', 0.0, 8.0, None), ('mu_h', 12.0, 24.0, None), ('sd_h', 0.8, 8.0, None),
    ('mu_c', 17.0, 30.0, None), ('sd_c', 0.8, 8.0, None),
    ('e_res', 0.0, 1.0, None), ('e_hp', 0.0, 1.0, None), ('btm', 0.0, 0.4, None),
]


def pack(p):
    return np.array([np.log(p[n]) if tr == 'log' else p[n] for n, lo, hi, tr in FREE])


def unpack(x, base):
    p = dict(base)
    for v, (n, lo, hi, tr) in zip(x, FREE):
        p[n] = float(np.exp(v) if tr == 'log' else v)
    return p


def bounds():
    lo = [np.log(l) if tr == 'log' else l for n, l, h, tr in FREE]
    hi = [np.log(h) if tr == 'log' else h for n, l, h, tr in FREE]
    return np.array(lo), np.array(hi)


def inputs(ba):
    W = load_bundle(ba)
    wx = dict(T=W['T'].astype(float), q=W['q'].astype(float), ghi=W['ghi'].astype(float), btm_cf=W['btm_cf'].astype(float))
    cal = B.calendar(HOURS, TZ[ba])
    d = gd.eia930(ba, columns=['d', 'df', 'q_ok']).reindex(HOURS)
    y = np.where(d.q_ok.fillna(False).values.astype(bool), d.d.values, np.nan).astype(float)
    return wx, cal, y, d.df.values.astype(float)


def fit(ba, years, verbose=True):
    wx, cal, y, _ = inputs(ba)
    yr = HOURS.year.values
    m = np.isin(yr, years) & np.isfinite(y)
    # exclude Winter Storm Uri load shed (ERCO, 14-20 Feb 2021): demand there was curtailed, not weather-driven
    m &= ~((HOURS >= pd.Timestamp('2021-02-14', tz='UTC')) & (HOURS < pd.Timestamp('2021-02-21', tz='UTC')) & (ba == 'ERCO'))
    ym = np.nanmean(y[m])
    p0 = dict(B.PARAMS); p0['P0'] = 0.8 * ym; p0['ua'] = 0.15
    tref = float(np.mean(years)) + 0.5
    lo, hi = bounds()
    if len(years) == 1:      # a single year cannot separate growth from season: no trend
        it = [n for n, *_ in FREE].index('trend'); p0['trend'] = 0.0; lo[it], hi[it] = -1e-4, 1e-4
    x0 = np.clip(pack(p0), lo + 1e-6, hi - 1e-6)

    def resid(x):
        out = B.demand(wx, cal, unpack(x, p0), t_ref=tref)['total']
        return (out[m] - y[m]) / ym

    t = time.time()
    r = least_squares(resid, x0, bounds=(lo, hi), loss='soft_l1', f_scale=0.03, x_scale='jac', max_nfev=400)
    p = unpack(r.x, p0); p['t_ref'] = tref
    comp = B.demand(wx, cal, p, t_ref=tref)
    pred = comp['total']
    mape = float(np.mean(np.abs(pred[m] - y[m]) / y[m]) * 100)
    if verbose:
        print(f'{ba} {years}: MAPE {mape:.2f}%  nfev {r.nfev}  {time.time() - t:.0f}s  '
              + ' '.join(f'{n}={p[n]:.3g}' for n, *_ in FREE), flush=True)
    return p, mape, pred, y


def main(which):
    years = {'dev': DEV_YEARS, 'train': TRAIN_YEARS, 'dev2019': (2019,), 'dev2021': (2021,)}[which]
    os.makedirs(f'{DATA}/calib', exist_ok=True); os.makedirs(f'{DATA}/noise', exist_ok=True)
    out, res = {}, {}
    for ba in BAS:
        p, mape, pred, y = fit(ba, list(years))
        out[ba] = dict(params=p, mape_in_sample=mape, years=list(years))
        if which == 'train':
            r = y / pred - 1
            r[~np.isin(HOURS.year.values, years)] = np.nan
            res[ba] = r.astype(np.float32)
        jdump(out, f'{DATA}/calib/demand_{which}.json')
    if which == 'train':
        np.savez_compressed(f'{DATA}/noise/demand_resid.npz', **res)


if __name__ == '__main__':
    main(sys.argv[1])
