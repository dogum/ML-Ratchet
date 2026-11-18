"""Dev twin test of what the network adds (17 Nov 2026; dev only, released weights).

cold: anchor = the dev-calibrated BuildingSim twin built from metadata only (rescore_seen.py twinsim, variant 'med');
warm: anchor = LoadCast's ridge TOWT fitted on the 2016 history (the warm network's own base).
For every dev building-meter, local 2017: correction = network - anchor, needed = truth - anchor, both divided by the
meter's 2017 mean. Reports per meter type the hourly correlation (pooled), the correlation of the annual means
(level) across buildings, the median within-building hourly correlation (shape), and where `needed` has structure
(hour of week, month, temperature, holidays) next to what the network supplies there.

    python twin_anchor.py         -> WORK/exam/twin_anchor_r3.json
"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lc_common as C, dataset as D, evaluate as E
import rescore_seen as R


def grp(m):
    return m if m in ('electricity', 'chilledwater') else 'heating (steam, hot water, gas)'


def main():
    nets = R.np_nets()
    z = np.load(f'{R.OUT}/twins.npz')
    dev = D.real_units('dev')
    recs = []; per = []
    for u in dev:
        D.prepare(u, ('full',))
        f = D.site_feats(u['site']); t = u['tidx']; y = u['y'][t].astype(float)
        ybar = np.nanmean(y)
        k = f"med|{u['bid']}|{u['meter']}"
        tw = z[k][t] / (u['area'] * C.IREF[u['meter']] / 1000.0) if k in z.files else None
        mu_c, _, _ = E.unit_predict(nets['cold'], u, 'cold')
        mu_w, _, _ = E.unit_predict(nets['warm'], u, 'warm')
        h = u['H']['full']; towt = C.towt_predict(h['coef'], f['T'][t], f['tow'][t]) * h['S']
        c = C.site_calendar(u['site'])
        for mode, mu, a in (('cold', mu_c, tw), ('warm', mu_w, towt)):
            if a is None: continue
            ok = np.isfinite(y) & np.isfinite(a)
            cor = (mu - a) / ybar; need = (y - a) / ybar
            per.append(dict(meter=u['meter'], mode=mode, id=u['id'], corr_mean=float(np.mean(cor[ok])), need_mean=float(np.mean(need[ok])),
                            r_within=float(np.corrcoef(cor[ok] - cor[ok].mean(), need[ok] - need[ok].mean())[0, 1]) if np.std(cor[ok]) > 0 else np.nan,
                            cv_anchor=C.g14(y, a)[0], cv_net=C.g14(y, mu)[0]))
            tow = f['tow'][t][ok]
            recs.append(pd.DataFrame(dict(meter=u['meter'], mode=mode, id=u['id'], cor=cor[ok], need=need[ok],
                                          dcor=cor[ok] - cor[ok].mean(), dneed=need[ok] - need[ok].mean(),
                                          wk=np.where(tow // 24 >= 5, 'weekend', np.where((tow % 24 >= 8) & (tow % 24 < 18), 'weekday 8-18', 'weekday night')),
                                          month=c['month'][t][ok], Tbin=pd.cut(f['T'][t][ok], [-50, 0, 10, 18, 24, 50]).astype(str),
                                          hol=f['hol'][t][ok].astype(bool))))
    H = pd.concat(recs, ignore_index=True); P = pd.DataFrame(per)
    out = dict(note='dev only, local 2017, released weights; values are fractions of the meter mean', by=[])
    for (g, mode), d in H.groupby([H.meter.map(grp), 'mode']):
        p = P[(P.meter.map(grp) == g) & (P['mode'] == mode)]
        r = dict(group=g, mode=mode, n_meters=int(p.id.nunique()), n_hours=len(d),
                 r_hourly=float(np.corrcoef(d.cor, d.need)[0, 1]),
                 r_level=float(np.corrcoef(p.corr_mean, p.need_mean)[0, 1]) if len(p) > 2 else None,
                 r_within_median=float(p.r_within.median()), r_within_dev=float(np.corrcoef(d.dcor, d.dneed)[0, 1]),
                 median_cv_anchor=float(p.cv_anchor.median()), median_cv_net=float(p.cv_net.median()),
                 net_beats_anchor=float((p.cv_net < p.cv_anchor).mean()),
                 level_need_median=float(p.need_mean.median()), level_corr_median=float(p.corr_mean.median()))
        for col in ('wk', 'month', 'Tbin', 'hol'):
            s = d.groupby(col)[['dneed', 'dcor']].mean()
            r[f'by_{col}'] = {str(k): [round(float(v.dneed), 3), round(float(v.dcor), 3)] for k, v in s.iterrows()}
        out['by'].append(r)
        print(f"{g:32s} {mode}: n {r['n_meters']:3d}  r hourly {r['r_hourly']:+.2f}  r level {r['r_level'] if r['r_level'] is None else round(r['r_level'], 2)}"
              f"  r within (median) {r['r_within_median']:+.2f}  CV anchor {r['median_cv_anchor']:.1f} -> net {r['median_cv_net']:.1f}"
              f"  need level {r['level_need_median']:+.2f} net level {r['level_corr_median']:+.2f}", flush=True)
        for col in ('wk', 'Tbin'):
            print('     ', col, r[f'by_{col}'])
    C.jdump(out, f'{R.OUT}/twin_anchor_r3.json')
    P.to_csv(f'{R.OUT}/twin_anchor_units_r3.csv', index=False)


if __name__ == '__main__':
    main()
