"""Twin test (dev only): simulate each grid's dev-year twin with the full simulator and check whether the network reads
twin and real alike. Twin = the grid's real weather of that dev year (no perturbation), true building params = that
year's own fit, descriptor = the other dev year's fit (exactly what the real dev bank uses), real residual noise,
the real wind/solar fleet of that year with the simulator's weather-error process and reporting scale = real k.
python3 twin.py <ckpt> -> checks/twin_<ckpt>.json"""
import sys, json, numpy as np, pandas as pd
from common import *
import sim, region as RG, renew as R, buildings as B, real, evaluate as E
from real import desc_vec


def twin(ba, year, rng):
    a = max(0, hidx(pd.Timestamp(f'{year}-01-01', tz='UTC')) - 960); b = hidx(pd.Timestamp(f'{year + 1}-01-01', tz='UTC')) + 48
    sl = slice(a, b); n = b - a
    W0 = sim.bundle(ba)
    g = RG.Region(); g.ba = ba; g.off = a; g.year = year
    g.cal = RG.calendar_arrays(HOURS[sl], TZ[ba], LST[ba])
    g.wx = {k: W0[k][sl].astype(float) for k in ('T', 'q', 'RH', 'ghi', 'ghi_clr', 'btm_cf')}
    g.W = {k: W0[k][:, sl] for k in ('pT', 'pRH', 'pws10', 'pws50', 'pps', 'pcf_track', 'pcf_fixed')}
    own = json.load(open(f'{DATA}/calib/demand_dev{year}.json'))[ba]['params']
    other = json.load(open(f'{DATA}/calib/demand_dev{2021 if year == 2019 else 2019}.json'))[ba]['params']
    g.pdesc = other; g.desc = desc_vec(other)
    truth = B.demand({k: g.wx[k] for k in ('T', 'q', 'ghi', 'btm_cf')}, g.cal, own, t_ref=own['t_ref'])
    res = sim.RCAT[ba]; idx = (np.arange(n) + 24 * rng.integers(0, len(res) // 24) + a) % len(res)
    r = np.nan_to_num(res[idx]); g.yd = truth['total'] * (1 + r)
    (wg, wcap), (sg, scap) = sim.groups(ba)
    g.wg, g.sg, g.wcap, g.scap = wg, sg, wcap[:, sl], scap[:, sl]
    rk = json.load(open(f'{DATA}/calib/renew_dev{year}.json'))[ba]
    n_pts = g.W['pws50'].shape[0]
    sd = 0.18; common = sim.ar1(rng, n, 0.95, sd * 0.7); local = sim.ar1(rng, n, 0.9, sd * 0.7, (n_pts,))
    vm = np.exp(common[None] + local)
    sub = {k: g.W[k] for k in ('pws10', 'pws50', 'pT', 'pps', 'pRH')}
    g.yw = rk['wind']['k'] * R.wind_mw(sub, wg, g.wcap, vmult=vm)
    cm = np.exp(sim.ar1(rng, n, 0.9, 0.15)[None] + sim.ar1(rng, n, 0.9, 0.2, (n_pts,)))
    subs = {k: g.W[k] for k in ('pcf_track', 'pcf_fixed')}
    g.has_solar = rk['solar'] is not None
    g.ys = (rk['solar']['k'] if g.has_solar else 1) * R.solar_mw(subs, sg, g.scap, cfmult=cm)
    rk_other = json.load(open(f'{DATA}/calib/renew_dev{2021 if year == 2019 else 2019}.json'))[ba]
    g.wk = rk_other['wind']['k']; g.sk = rk_other['solar']['k'] if g.has_solar else 1.0
    g.finalize()
    return g


def main(ck):
    nets = E.load(ck); rng = np.random.default_rng(11)
    Bd = dict(np.load(f'{DATA}/banks/real_dev.npz')); pr = E.predict(nets, Bd)
    out = {}
    for ba in BAS:
        for year in DEV_YEARS:
            g = twin(ba, year, rng)
            rows = []
            for d in days_of_year(ba, year)[::3]:
                i0 = day_start_index(ba, d) - g.off
                if i0 - 182 - 400 < 0: continue
                rows.append(g.sample(i0))
            Bt = {k: np.stack([r[k] for r in rows]) for k in ('xd', 'yd', 'xw', 'yw', 'pw', 'xs', 'ys', 'ps', 'night') if all(k in r for r in rows)}
            if 'xs' not in Bt: Bt['xs'] = np.zeros((len(rows), 156), np.float32); Bt['ys'] = np.full((len(rows), 24), np.nan); Bt['ps'] = np.zeros((len(rows), 24)); Bt['night'] = np.ones((len(rows), 24))
            pt = E.predict(nets, Bt)
            rr = (Bd['meta_ba'] == ba) & np.array([d.startswith(str(year)) for d in Bd['meta_day']])
            rec = {}
            for k in ('d', 'w', 's'):
                et = E.errors(Bt, pt, k); er = E.errors(Bd, pr, k, rows=rr)
                rec[k] = dict(twin=et['err'], real=er['err'], twin_cov68=et['cov68'], real_cov68=er['cov68'])
            out[f'{ba}_{year}'] = rec
            print(ba, year, ' '.join(f"{k}: twin {rec[k]['twin']:.2f} real {rec[k]['real']:.2f}" for k in 'dws'), flush=True)
    jdump(out, f'{ROOT}/checks/twin_{ck}.json')


if __name__ == '__main__':
    main(sys.argv[1])
