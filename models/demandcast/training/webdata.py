"""Data for the web page: a fixed random draw of exam days (stratified: one day per BA per quarter of each exam year,
plus each BA's hottest and coldest exam day), the network weights (float32 base64), and per-day inputs so the page can
re-run the physics and the network in JS when the sliders move.
python3 webdata.py <ckpt>  -> web/data.json, web/weights.json"""
import sys, json, base64, numpy as np, pandas as pd, torch
from common import *
import real, buildings as B, evaluate as E, features as F

MW = np.round(np.arange(0.5, 1.51, 0.1), 2)      # wind-speed multipliers tabulated
MS = np.round(np.arange(0.3, 1.51, 0.1), 2)      # irradiance multipliers tabulated


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, np.float16).tobytes()).decode()     # float16 halves the page


def r(a, k=4):
    return [None if not np.isfinite(v) else round(float(v), k) for v in np.ravel(a)]


def states(g, iss):
    """Filter states at the last hour before issue (full-series filters, as in Region.finalize)."""
    p = g.pdesc
    T = g.wx['T']; ghi = g.wx['ghi'] / 1000.0
    Tsa = T + p['a_sol'] * ghi
    return dict(T30=float(B.rc_filter(T[:iss], 240.0)[-1]), L1=float(B.rc_filter(Tsa[:iss], p['tau1'])[-1]),
                L2=float(B.rc_filter(Tsa[:iss], p['tau2'])[-1]), G=float(B.rc_filter(ghi[:iss], 2.0)[-1]))


def day_record(g, i0, sc):
    iss, ie = i0 - 14, i0 + 24
    s = g.sample(i0)
    p = g.pdesc
    sl = slice(iss, ie)
    cal = {k: v[sl] for k, v in g.cal.items()}
    res, com, ind, occ = B.occupancy(cal, p)
    growth = 1 + p['trend'] * (cal['year'] - p.get('t_ref', 2020.0))
    ph_total_mean = float(np.nanmean(g.pact['total'][i0 - 182:iss]))
    rec = dict(ba=g.ba, day=str((HOURS[i0] + pd.Timedelta(hours=LST[g.ba])).date()),
               st=states(g, iss), T=r(g.wx['T'][sl], 2), q=r(g.wx['q'][sl], 3), ghi=r(g.wx['ghi'][sl], 1),
               clr=r(g.wx['ghi_clr'][sl], 1), btm=r(g.wx['btm_cf'][sl], 4), res=r(res), com=r(com), ind=r(ind), occ=r(occ),
               gr=r(growth, 5), sd=ph_total_mean, xd=r(s['xd'][:329], 5),
               y=r(g.yd[i0:ie], 0), df=r(g.df[i0:ie], 0), pers=r(g.yd[i0 - 24:i0], 0), prior=r(s['pd_total'], 0))
    n_pts = g.W['pws50'].shape[0]
    if 'xw' in s:
        cw = s['capw']
        rec['w'] = dict(cap=round(float(cw), 1), x=r(s['xw'][:164], 5), y=r(g.yw[i0:ie] / cw),
                        tab=[r(g.wind(i0, ie, np.full((n_pts, 24), m)) / cw) for m in MW],
                        ws=r(g.wind_ws(i0, ie, np.ones((n_pts, 24))), 3))
    if 'xs' in s:
        cs = s['caps']
        rec['s'] = dict(cap=round(float(cs), 1), x=r(s['xs'][:156], 5), y=r(g.ys[i0:ie] / cs), night=r(s['night'], 0),
                        tab=[r(g.solar(i0, ie, np.full((n_pts, 24), m)) / cs) for m in MS])
    return rec


def main(ckpt):
    nets = E.load(ckpt)
    ex = json.load(open(f'{DATA}/exam/{ckpt}.json'))
    sc = ex['sigma_scale_dev']
    rng = np.random.default_rng(2024)
    days = []
    for ba in BAS:
        real._CACHE.clear()
        g = real.real_region(ba)
        picks = []
        for y in EXAM_YEARS:
            dd = days_of_year(ba, y)
            for qtr in range(4):
                c = [d for d in dd if (d.month - 1) // 3 == qtr]
                picks.append(c[rng.integers(len(c))])
        # hottest / coldest exam day by daily max / min population-weighted temperature
        allx = [d for y in EXAM_YEARS for d in days_of_year(ba, y)]
        tmax = [g.wx['T'][day_start_index(ba, d):day_start_index(ba, d) + 24].max() for d in allx]
        tmin = [g.wx['T'][day_start_index(ba, d):day_start_index(ba, d) + 24].min() for d in allx]
        hot, cold = allx[int(np.argmax(tmax))], allx[int(np.argmin(tmin))]
        for d, tag in [(d, 'random') for d in sorted(picks)] + [(hot, 'hottest'), (cold, 'coldest')]:
            rec = day_record(g, day_start_index(ba, d), sc); rec['tag'] = tag
            days.append(rec)
        print(ba, len(days), flush=True)
    params = {ba: json.load(open(f'{DATA}/calib/demand_dev.json'))[ba]['params'] for ba in BAS}
    json.dump(dict(days=days, params=params, MW=list(MW), MS=list(MS), sigma_scale=sc, lst=LST), open(f'{ROOT}/web/data.json', 'w'),
              separators=(',', ':'))
    W = {}
    for k, net in nets.items():
        sd = {n: v.numpy() for n, v in net.state_dict().items()}
        W[k] = {n: dict(shape=list(v.shape), b64=b64(v)) for n, v in sd.items()}
    json.dump(W, open(f'{ROOT}/web/weights.json', 'w'), separators=(',', ':'))


if __name__ == '__main__':
    main(sys.argv[1])
