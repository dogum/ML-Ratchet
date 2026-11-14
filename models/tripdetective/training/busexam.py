"""Bus exam (run once, after the bus release is chosen): ZTBus exam legs (1,115 exam missions, by-month split).
Truth per leg: 19,000 kg + 70 kg x the leg's mean passenger count. Baselines fitted on the dev gate legs only:
  dumb      the dev median truth for every leg (a constant);
  classical least-squares road-load fit of the logged power (baselines.lsq_fit; battery-to-wheel 0.85), scaled by
            the median ratio truth / fit on dev.
Reports mean / median absolute error, bias, the slope of estimated mass on passenger mass (1 = follows every
passenger, 0 = ignores them), and the correlation with passengers; by season.
usage: python busexam.py <bus_ckpt> [tag]"""
from common import *
import pandas as pd
import evaluate as E, busbank, busladder as BL, baselines as B


def lsq_mass(rb, legs):
    o = rb['offs']; out = []
    for t in legs:
        js = np.where(rb['trip'] == t)[0]
        rows = [B.lsq_rows(rb['raw'][o[j]:o[j + 1]].T.astype(np.float32), rb['flags'][j]) for j in js]
        f = B.lsq_fit(rows)
        out.append(f['mass'] * busbank.BUS_SCALE if f else np.nan)
    return np.array(out)


def run(ckpt, tag='bus_final'):
    net = E.load_net(f'{ROOT}/ckpt/{ckpt}')
    if not os.path.exists(f'{ROOT}/data/bus_exam.npz'): busbank.build_real('exam')
    dev = busbank.load_real('dev'); ex = busbank.load_real('exam')
    # dev: baseline constants and the widen-only sigma factor
    dl, dm, ds = BL.real_predict(net, dev); dmeta = {x['trip_id']: x for x in dev['meta']}
    dtruth = np.array([dmeta[t]['mass_kg'] for t in dl])
    med = float(np.nanmedian(dtruth))
    dlsq = lsq_mass(dev, dl); ok = np.isfinite(dlsq) & np.isfinite(dtruth)
    scale = float(np.median(dlsq[ok] / dtruth[ok]))
    z = np.log(dm / dtruth) / ds; z = np.abs(z[np.isfinite(z)])
    sigk = float(max(1.0, np.percentile(z, 68.3)))
    # exam
    legs, m, s = BL.real_predict(net, ex); meta = {x['trip_id']: x for x in ex['meta']}
    M = [meta[t] for t in legs]
    lsq = lsq_mass(ex, legs) / scale
    rep = dict(ckpt=ckpt, n_legs=len(legs), n_missions=len({x['mission'] for x in M}), sigk=sigk,
               baseline_fit=dict(dev_median_kg=med, lsq_scale=scale, n_dev_legs=len(dl)),
               detective=BL.leg_metrics(m, M, s, sigk), classical=BL.leg_metrics(lsq, M), dumb=BL.leg_metrics(np.full(len(legs), med), M))
    d = pd.DataFrame(M); d['m_hat'] = m; d['m_lsq'] = lsq
    rep['by_season'] = {k: dict(n=len(g), detective=BL.leg_metrics(g.m_hat.values, g.to_dict('records')),
                                classical=BL.leg_metrics(g.m_lsq.values, g.to_dict('records'))) for k, g in d.groupby('season')}
    d.to_csv(f'{ROOT}/data/bus_exam_{tag}.csv', index=False)
    jsave(rep, f'{ROOT}/exam_{tag}.json')
    print(json.dumps(rep, indent=1, default=str)[:4000])
    return rep


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'bus_final')
