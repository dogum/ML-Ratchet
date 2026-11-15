"""Bus exam (run once, after the bus release is chosen): ZTBus exam legs (1,115 exam missions, by-month split).
Truth per leg: 19,000 kg + 70 kg x the leg's mean passenger count. Baselines fitted on the dev gate legs only:
  dumb      the dev median truth for every leg (a constant);
  classical least-squares road-load fit of the logged power (baselines.lsq_fit; battery-to-wheel 0.85), scaled by
            the median ratio truth / fit on dev.
Reports mean / median absolute error, bias, the slope of estimated mass on passenger mass (1 = follows every
passenger, 0 = ignores them), and the correlation with passengers; by season.
usage: python busexam.py <bus_ckpt> [tag]   |   python busexam.py --cal [ckpt=pb1] [tag=bus_v21]  (v2.1)"""
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




# ---------------- v2.1 (15 Nov 2026): the day-relative calibration (buscal.py), scored once ----------------
def run_cal(ckpt='pb1', tag='bus_v21'):
    """The v2.1 bus reading on the exam legs: buscal's map (coefficients fitted on dev gate legs only, work/buscal_<ckpt>.json)
    applied with the mission context of every exam leg of the mission. Same baselines as run() (fitted on dev only);
    the v2.0 reading (raw, and with the dev-fitted level) on the same legs for comparison."""
    import buscal
    cal = jload(f'{ROOT}/buscal_{ckpt}.json')
    net = E.load_net(f'{ROOT}/ckpt/{ckpt}.pt')
    dev = busbank.load_real('dev'); ex_all = busbank.load_real('exam', all_legs=True)
    dl, dm, ds = BL.real_predict(net, dev); dmeta = {x['trip_id']: x for x in dev['meta']}
    dtruth = np.array([dmeta[t]['mass_kg'] for t in dl]); med = float(np.nanmedian(dtruth))
    dlsq = lsq_mass(dev, dl); ok = np.isfinite(dlsq) & np.isfinite(dtruth); scale = float(np.median(dlsq[ok] / dtruth[ok]))
    level = float(np.median(dtruth[np.isfinite(dtruth)] / dm[np.isfinite(dtruth)]))
    ctx = buscal.leg_table(net, ex_all)
    d = ctx[ctx.label_ok & np.isfinite(ctx.truth)].reset_index(drop=True)
    pred = buscal.apply(d, cal['rung'], np.array(cal['coef']))
    M = [dict(pax=p, mass_kg=t) for p, t in zip(d.pax, d.truth)]
    lsq = {t: m for t, m in zip(*[list(x) for x in (d.trip_id, lsq_mass(ex_all, list(d.trip_id)) / scale)])}
    e = pred - d.truth.values
    rep = dict(ckpt=ckpt, rung=cal['rung'], coef=cal['coef'], n_legs=len(d), n_missions=int(d.mission.nunique()),
               legs_alone_in_mission=int((ctx[ctx.label_ok].groupby('mission').size() == 1).sum()),
               v21=BL.leg_metrics(pred, M), v21_in1s=float(np.mean(np.abs(e) < cal['sigma_kg'])), v21_in2s=float(np.mean(np.abs(e) < 2 * cal['sigma_kg'])),
               v20_raw=BL.leg_metrics(d.m.values, M), v20_level=BL.leg_metrics(d.m.values * level, M), level=level,
               classical=BL.leg_metrics(np.array([lsq[t] for t in d.trip_id]), M), dumb=BL.leg_metrics(np.full(len(d), med), M),
               baseline_fit=dict(dev_median_kg=med, lsq_scale=scale))
    d['pred'] = pred
    rep['by_season'] = {k: dict(n=len(g), v21=BL.leg_metrics(g.pred.values, [dict(pax=p, mass_kg=t) for p, t in zip(g.pax, g.truth)]),
                                dumb_mae=float(np.mean(np.abs(med - g.truth)))) for k, g in d.groupby('season')}
    rep['by_bus'] = {k: dict(n=len(g), v21_mae=float(np.mean(np.abs(g.pred - g.truth))), dumb_mae=float(np.mean(np.abs(med - g.truth))))
                     for k, g in d.groupby('bus')}
    d.to_csv(f'{ROOT}/data/bus_exam_{tag}.csv', index=False)
    jsave(rep, f'{ROOT}/exam_{tag}.json')
    print(json.dumps(rep, indent=1, default=str)[:5000])
    return rep


if __name__ == '__main__':
    if sys.argv[1] == "--cal": run_cal(*sys.argv[2:])
    else: run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "bus_final")
