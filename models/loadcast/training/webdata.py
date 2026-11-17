"""Data for the web page (web/data_<run>.json): a fixed random draw of EXAM building-meters (stratified by meter, seed
20261117, never chosen by result), their site weather and calendar for local 2017 (+30 days lead-in), the 2016 history
encodings, the 2017 real meter, a G14 change-point baseline fitted on 2016 (BuildingData's cp_best), and Python
reference predictions for the JS == Python check. Plus exam/ratchet/calibration summaries.

    python3 webdata.py <run>
"""
import sys, os, json, base64
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import BLDG  # noqa: E402
sys.path.insert(0, os.path.join(BLDG, 'data', 'scripts'))   # BuildingData scripts (was the sandbox workspace)
import lc_common as C, dataset as D, evaluate as E, lcnet as N
from baselines import cp_best

PICK = {'electricity': 6, 'chilledwater': 4, 'steam': 3, 'hotwater': 2, 'gas': 1}
LEAD = 720


def b64(a, dt):
    return base64.b64encode(np.ascontiguousarray(a).astype(dt).tobytes()).decode()


def pick_units():
    q = C.bd.bdg2_quality(split='exam')
    rng = np.random.default_rng(20261117)
    out = []
    for m, n in PICK.items():
        ids = sorted(q[(q.meter == m) & q.q_ok].building_id)
        out += [(b, m) for b in rng.choice(ids, n, replace=False)]
    return out


def main(run):
    nets = {m: N.load(f'{C.ROOT}/ckpt/{run}_{m}.pt')[0] for m in ('cold', 'warm')}
    W = json.load(open(f'{C.ROOT}/data/widen_{run}.json'))
    picks = pick_units()
    U = []
    for m in PICK:
        bs = [b for b, mm in picks if mm == m]
        U += [u for u in D.real_units('exam', meters=[m]) if u['bid'] in bs]
    sites = {}; units = []
    for u in U:
        D.prepare(u, ('full', 'w8', 'w2'))
        f = D.site_feats(u['site']); c = C.site_calendar(u['site']); t = u['tidx']
        if u['site'] not in sites:
            w, wm = C.site_weather(u['site'])
            lo = t[0] - LEAD; sl = slice(lo, t[-1] + 1)
            T = w['T_out'].interpolate(limit_direction='both').values[sl]
            dew = w['dewpoint'].interpolate(limit_direction='both').values[sl]
            ghi = w['GHI'].fillna(0).clip(lower=0).values[sl]
            wind = w['wind_speed'].interpolate(limit_direction='both').fillna(3).values[sl]
            dcode = f['dcode'][sl] - f['dcode'][lo]
            cal = (c['hour'][sl] | (c['dow'][sl] << 5) | (c['holiday'][sl].astype(int) << 8) | (c['month'][sl] << 9)
                   | (c["mday"][sl] << 13) | (c["doy"][sl] << 18))
            sites[u['site']] = dict(lo=int(lo), n=int(t[-1] + 1 - lo), lead=int(t[0] - lo), tz=wm['tz'], lat=wm['lat'],
                                    country=C.country_of_tz(wm['tz']), start_utc=str(w.index[lo]),
                                    T=b64(np.round(T * 100), '<i2'), dew=b64(np.round(dew * 100), '<i2'),
                                    ghi=b64(np.round(ghi), '<i2'), wind=b64(np.round(wind * 100), '<i2'),
                                    cal=b64(cal, '<i4'), day=b64(dcode, '<i2'))
        s = sites[u['site']]
        unit = u['area'] * C.IREF[u['meter']] / 1000.0
        y = u['y'][t].astype(float) * unit
        sc = np.nanmax(y) / 30000 if np.nanmax(y) > 0 else 1.0
        yq = np.where(np.isfinite(y), np.round(y / sc), -32768)
        # G14 change-point baseline (BuildingData's cp_best) on local 2016 daily means + 2016 time-of-week shape
        h16 = np.flatnonzero(f['year'] == 2016)
        yy = u['y'][h16].astype(float) * unit
        dd = pd.DataFrame(dict(y=yy, T=f['T'][h16], d=f['dcode'][h16]))
        g = dd.groupby('d'); cnt = g.y.count(); yd = g.y.mean()[cnt >= 20]; Td = g['T'].mean()[yd.index]
        cp = cp_best(Td.values, yd.values)
        hw = pd.Series(yy).groupby(f['tow'][h16]).mean(); hw = (hw / hw.mean()).reindex(range(168)).fillna(1.0).values
        ref = {}
        for mode, v in (('cold', 'full'), ('warm', 'full'), ('warm', 'w8'), ('warm', 'w2')):
            mu, sd, yt = E.unit_predict(nets[mode], u, mode, v)
            k = W.get(f'{mode}_{v}_{u["meter"]}', 1.0)
            cv, nm, _ = C.g14(yt * unit, mu * unit)
            ref[f'{mode}_{v}'] = dict(mu=(mu[:336] * unit).round(5).tolist(), sd=(sd[:336] * unit).round(5).tolist(),
                                      cv=cv, nmbe=nm, cov=E.cover(yt, mu, sd, k), k=k)
        units.append(dict(id=u['id'], bid=u['bid'], site=u['site'], use=u['use'], area=u['area'], yb=u['yb'],
                          meter=u['meter'], meters=u['meters'], hvac=u['hvac'], cz=u['cz'], unit=unit,
                          static=u['static'].tolist(), H={v: dict(S=h['S'], coef=h['coef'].tolist(), cp=h['cp'].tolist(),
                                                                   prof=h['prof'].tolist(), scal=h['scal'].tolist())
                                                          for v, h in u['H'].items()},
                          y=b64(yq, '<i2'), yscale=sc, cp=dict(kind=cp['kind'], params={k: float(v) for k, v in cp['params'].items()},
                                                               shape=hw.round(4).tolist()), ref=ref))
        print(u['id'], u['site'], {k: round(v['cv'], 1) for k, v in ref.items()}, flush=True)
    os.makedirs(f'{C.ROOT}/web', exist_ok=True)
    C.jdump(dict(run=run, sites=sites, units=units, widen=W), f'{C.ROOT}/web/data_{run}.json')
    print('sites', len(sites), 'size', os.path.getsize(f'{C.ROOT}/web/data_{run}.json'))


if __name__ == '__main__':
    main(sys.argv[1])
