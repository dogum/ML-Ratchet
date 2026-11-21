"""Fit the HRRR-daily -> gridMET operator on DEV fires (same place, same local day). Output out/calib_wx.json.
Used to make simulated weather features look like the gridMET features the network sees on real fires."""
from common import *  # noqa
import pandas as pd
import wxfeat as WX

fd_ = fd()
fs_ = fs()
from firesim import realdata as RD  # noqa


def collect():
    rows = []
    for fid in fd_.fires('dev').Event_ID:
        W = fd_.weather(fid)
        if 'grid' not in W or 'daily' not in W:
            continue
        r = fd_._row(fid)
        lon, lat = fd_._lonlat(r)
        w = RD.weather_series(W, float(r.cx), float(r.cy))
        w = RD.fill_precip_from_daily(w, W['daily'], lon)
        gam = RD.grid_convergence_deg(float(r.cx), float(r.cy))
        w['wind_dir'] = (w['wind_dir'] - gam) % 360.0          # back to true north for this comparison
        m = fs_.moisture.dead_moisture(w, classes=('100h',))
        hd = WX.hrrr_daily(w, lon, m['m100h'].to_numpy())
        hd = hd.iloc[3:]                                        # 72 h moisture spin-up
        gm = W['daily'].copy()
        gm.index = pd.DatetimeIndex(gm.index).normalize()
        for shift in (-1, 0, 1):
            g2 = gm.copy()
            g2.index = g2.index + pd.Timedelta(days=shift)
            j = hd.join(g2, how='inner')
            j['fire_id'] = fid
            j['shift'] = shift
            j['precip_from_daily'] = bool(w['precip_from_daily'].any())
            rows.append(j)
    return pd.concat(rows)


def fit(x, y, log=False):
    k = np.isfinite(x) & np.isfinite(y)
    x, y = x[k], y[k]
    if log:
        x, y = np.log(np.maximum(x, 0.3)), np.log(np.maximum(y, 0.3))
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    return dict(a=float(a), b=float(b), sd=float(np.std(res)), r=float(np.corrcoef(x, y)[0, 1]), n=int(k.sum()))


def main():
    J = collect()
    out = dict(by_shift={})
    for s in (-1, 0, 1):
        j = J[J['shift'] == s]
        out['by_shift'][s] = dict(vs=fit(j.ws_mean.values, j.vs.values, True)['r'], rmin=fit(j.rhmin.values, j.rmin.values)['r'],
                                  tmmx=fit(j.tmax.values, j.tmmx.values)['r'])
    best = max(out['by_shift'], key=lambda s: sum(out['by_shift'][s].values()))
    j = J[J['shift'] == best]
    out['shift_days'] = int(best)
    out['vs'] = fit(j.ws_mean.values, j.vs.values, True)
    out['rmin'] = fit(j.rhmin.values, j.rmin.values)
    out['tmmx'] = fit(j.tmax.values, j.tmmx.values)
    out['vpd'] = fit(j.vpd_h.values, j.vpd.values)
    out['fm100'] = fit(j.m100.values, j.fm100.values)
    d = (j.th.values - j.dir_from.values + 540) % 360 - 180
    w = j.ws_mean.values > 2.0
    out['th'] = dict(offset_deg=float(np.median(d[w])), sd_deg=float(1.4826 * np.median(np.abs(d[w] - np.median(d[w])))),
                     n=int(w.sum()), frac_within_45=float(np.mean(np.abs(d[w]) < 45)))
    jj = j[~j.precip_from_daily]
    tot_h, tot_g = float(np.nansum(jj.p)), float(np.nansum(jj.pr))
    out['pr'] = dict(b=tot_g / max(tot_h, 1e-6), hrrr_total=tot_h, gridmet_total=tot_g, n=int(len(jj)))
    out['n_days'] = int(len(j))
    out['n_fires'] = int(j.fire_id.nunique())
    out['note'] = 'dev fires only; HRRR series at the fire centre vs gridMET at the centroid; local solar day'
    jdump(out, f'{OUT}/calib_wx.json')
    print(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()
