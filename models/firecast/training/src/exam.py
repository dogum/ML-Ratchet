"""THE EXAM (run once, at the end, with the frozen final rung). Exam data are loaded only here (for_exam_scoring=True).

  python3 exam.py run <rung>        # FireCast on every exam growth fire-day (resumable per fire) -> out/exam/firecast.jsonl
  python3 exam.py summary <rung>    # tables vs B's baselines (same fire-days), by period; replay subset table
Modes: F (fire + gridMET weather; every exam fire) and, on the 48 replay-subset fires that have LANDFIRE packages,
also L (+ landscape). Calibration (T, b, threshold) fixed on dev beforehand (out/cal_<rung>_<mode>.json).
"""
import time
from common import *  # noqa
import pandas as pd
import torch
import evaluate as EV
import realdays as RDY

EXD = f'{OUT}/exam'
os.makedirs(EXD, exist_ok=True)
RB = np.linspace(0, 1, 21)


def replay_ids():
    return set(json.load(open(f'{PROC}/baselines/replay_subset.json'))['fires'])


def run(rung):
    fd_ = fd()
    net = EV.load_net(rung)
    cal = {m: json.load(open(shipped(f'cal_{rung}_{m}.json'))) for m in ('F', 'L') if os.path.exists(shipped(f'cal_{rung}_{m}.json'))}
    days = pd.read_parquet(f'{PROC}/baselines/fire_days_exam.parquet')
    ids = sorted(days[days.new_obs_km2 > 0].fire_id.unique())
    ids = list(np.random.default_rng(20261118).permutation(ids))     # fixed order: a partial run is a random subset
    rep = replay_ids()
    fn = f'{EXD}/firecast_{rung}.jsonl'
    done = {json.loads(l)['fire_id'] for l in open(fn)} if os.path.exists(fn) else set()
    t0 = time.time()
    with open(fn, 'a') as fo:
        for n, fid in enumerate(ids):
            if fid in done:
                continue
            rows, rel = [], {}
            try:
                for d in RDY.fire_days(fid, for_exam_scoring=True, with_land=fid in rep):
                    for mode in (('F', 'L') if d['land'] is not None else ('F',)):
                        X, s = EV.real_inputs(d, land=(mode == 'L'))
                        lg = EV.logits_full(net, X, s)
                        c = cal.get(mode, cal['F'])
                        p = EV.sig(lg / c['T'] + c['b'])
                        m = EV.score_real(d, p, c['thr'])
                        m05 = EV.score_real(d, p, 0.5)
                        m.update(f1_at_0p5=m05['f1'], dice_total_at_0p5=m05['dice_total'],
                                 log10_area_ratio_at_0p5=m05['log10_area_ratio'], new_pred_km2_at_0p5=m05['new_pred_km2'])
                        m.update(fire_id=fid, k=d['k'], mode=mode, area_km2=d['area_km2'], first=bool(d['first']))
                        rows.append(m)
                        # reliability counts (scored candidate cells)
                        win = fd_.metrics.score_window(d['now'], RES, WIN_KM) & ~d['now']
                        y = (d['nxt'] & ~d['now'])[win]
                        pv = p[win]
                        b = np.minimum((pv * 20).astype(int), 19)
                        r = rel.setdefault(mode, dict(n=np.zeros(20), p=np.zeros(20), y=np.zeros(20)))
                        r['n'] += np.bincount(b, minlength=20)
                        r['p'] += np.bincount(b, weights=pv, minlength=20)
                        r['y'] += np.bincount(b, weights=y.astype(float), minlength=20)
            except Exception as ex:
                print('FAIL', fid, repr(ex), flush=True)
                rows = [dict(fire_id=fid, error=repr(ex))]
            fo.write(json.dumps(dict(fire_id=fid, rows=rows, rel={k: {kk: vv.tolist() for kk, vv in v.items()}
                                                                    for k, v in rel.items()}), default=float) + '\n')
            fo.flush()
            if n % 50 == 0:
                print(n, len(ids), fid, len(rows), f'{time.time() - t0:.0f}s', flush=True)
    print('DONE', flush=True)


def boot_ci(df, col, n=1000, seed=0):
    """95 % CI of the mean over fire-days, resampling fires."""
    g = df.groupby('fire_id')[col]
    s, c = g.sum().values, g.count().values
    rng = np.random.default_rng(seed)
    k = len(s)
    v = []
    for _ in range(n):
        i = rng.integers(k, size=k)
        v.append(s[i].sum() / max(c[i].sum(), 1))
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


def table(df, label):
    d = df[df.new_obs_km2 > 0]
    out = dict(method=label, n_fires=int(d.fire_id.nunique()), n_days=int(len(d)),
               ap_mean=float(d.ap.mean()), ap_ci=boot_ci(d, 'ap'), ap_median=float(d.ap.median()),
               f1_mean=float(d.f1.mean()), f1_ci=boot_ci(d.assign(f1=d.f1.fillna(0)), 'f1'),
               dice_total_median=float(d.dice_total.median()), log10_area_ratio_median=float(d.log10_area_ratio.median()),
               ap_mean_of_fires=float(d.groupby('fire_id').ap.mean().mean()))
    return out


def summary(rung):
    fd_ = fd()
    fc = [json.loads(l) for l in open(f'{EXD}/firecast_{rung}.jsonl')]
    R = pd.DataFrame([r for x in fc for r in x['rows'] if 'error' not in r])
    errs = [x['fire_id'] for x in fc if x['rows'] and 'error' in x['rows'][0]]
    B = []
    for l in open(f'{PROC}/baselines/spread_exam.jsonl'):
        x = json.loads(l)
        B += [r for r in x['rows'] if r['method'] in ('persist', 'clim', 'grow_prev', 'grow_prev_k', 'fl_prev')]
    B = pd.DataFrame(B)
    fires = fd_.fires('exam').set_index('Event_ID')
    per = fires.exam_period.to_dict()
    reg = fires.region.to_dict()
    R['period'] = R.fire_id.map(per)
    B['period'] = B.fire_id.map(per)
    R['region'] = R.fire_id.map(reg)
    F = R[R['mode'] == 'F']
    key = set(zip(F.fire_id, F.k))
    Bk = B[[(f, k) in key for f, k in zip(B.fire_id, B.k)]]
    res = dict(rung=rung, n_fail_fires=len(errs), fail_fires=errs[:20])
    for period in ('all', 'early', 'main', 'recent'):
        fsub = F if period == 'all' else F[F.period == period]
        bsub = Bk if period == 'all' else Bk[Bk.period == period]
        T = [table(fsub, 'FireCast (F: fire + gridMET)')]
        T.append(table(fsub.assign(f1=fsub.f1_at_0p5, dice_total=fsub.dice_total_at_0p5,
                                   log10_area_ratio=fsub.log10_area_ratio_at_0p5), 'FireCast, threshold 0.5'))
        for m in ('persist', 'clim', 'grow_prev', 'grow_prev_k', 'fl_prev'):
            T.append(table(bsub[bsub.method == m], m))
        res[period] = T
    # by region (F, AP mean) vs fl_prev
    res['by_region'] = {}
    for rg, sub in F.groupby('region'):
        bb = Bk[(Bk.method == 'fl_prev') & Bk.fire_id.map(reg).eq(rg)]
        res['by_region'][rg] = dict(n_days=int((sub.new_obs_km2 > 0).sum()), firecast_ap=float(sub[sub.new_obs_km2 > 0].ap.mean()),
                                    fl_prev_ap=float(bb[bb.new_obs_km2 > 0].ap.mean()),
                                    firecast_f1=float(sub[sub.new_obs_km2 > 0].f1.mean()),
                                    fl_prev_f1=float(bb[bb.new_obs_km2 > 0].f1.mean()))
    # by size bin
    res['by_size'] = {}
    for lo, hi in ((0, 5), (5, 20), (20, 100), (100, 500), (500, 1e9)):
        sub = F[(F.area_km2 >= lo) & (F.area_km2 < hi) & (F.new_obs_km2 > 0)]
        keys = set(zip(sub.fire_id, sub.k))
        bb = Bk[(Bk.method == 'fl_prev') & (Bk.new_obs_km2 > 0)]
        bb = bb[[(f, k) in keys for f, k in zip(bb.fire_id, bb.k)]]
        res['by_size'][f'{lo}-{hi}'] = dict(n=int(len(sub)), firecast_ap=float(sub.ap.mean()), fl_prev_ap=float(bb.ap.mean()),
                                            firecast_f1=float(sub.f1.mean()), fl_prev_f1=float(bb.f1.mean()))
    # win rate per fire-day vs fl_prev
    j = F[F.new_obs_km2 > 0].merge(Bk[(Bk.method == 'fl_prev')][['fire_id', 'k', 'ap', 'f1']], on=['fire_id', 'k'],
                                   suffixes=('', '_fl'))
    res['win_rate_ap_vs_fl_prev'] = float((j.ap > j.ap_fl).mean())
    res['win_rate_f1_vs_fl_prev'] = float((j.f1.fillna(0) > j.f1_fl.fillna(0)).mean())
    # reliability (pooled cells)
    rel = {}
    for x in fc:
        for mode, r in x.get('rel', {}).items():
            a = rel.setdefault(mode, dict(n=np.zeros(20), p=np.zeros(20), y=np.zeros(20)))
            for kk in ('n', 'p', 'y'):
                a[kk] += np.array(r[kk])
    res['reliability'] = {m: [dict(bin=f'{i / 20:.2f}-{(i + 1) / 20:.2f}', n=int(a['n'][i]),
                                   p_mean=float(a['p'][i] / a['n'][i]) if a['n'][i] else None,
                                   obs=float(a['y'][i] / a['n'][i]) if a['n'][i] else None) for i in range(20)]
                          for m, a in rel.items()}
    # replay subset: F and L vs baselines (+ physics replay if present)
    rep = replay_ids()
    sub = R[R.fire_id.isin(rep)]
    keyr = set(zip(sub.fire_id, sub.k))
    T = [table(sub[sub['mode'] == 'F'], 'FireCast F'), table(sub[sub['mode'] == 'L'], 'FireCast L (+LANDFIRE)')]
    Br = B[[(f, k) in keyr for f, k in zip(B.fire_id, B.k)]]
    for m in ('persist', 'clim', 'grow_prev', 'fl_prev'):
        T.append(table(Br[Br.method == m], m))
    fnr = f'{EXD}/replay_calibrated.jsonl'
    if os.path.exists(fnr):
        P = pd.DataFrame([r for l in open(fnr) for r in json.loads(l)['rows'] if 'ap' in r])
        for lab, col in (('physics replay, calibrated (deterministic)', 'det'), ('physics replay, calibrated (ensemble)', 'ens')):
            q = P[P.kind == col]
            if len(q):
                T.append(table(q, lab))
        # FireCast restricted to the days the replay covered
        kk = set(zip(P.fire_id, P.k))
        s2 = sub[[(f, k) in kk for f, k in zip(sub.fire_id, sub.k)]]
        T.append(table(s2[s2['mode'] == 'F'], 'FireCast F (replay days)'))
        T.append(table(s2[s2['mode'] == 'L'], 'FireCast L (replay days)'))
        b2 = Br[[(f, k) in kk for f, k in zip(Br.fire_id, Br.k)]]
        T.append(table(b2[b2.method == 'fl_prev'], 'fl_prev (replay days)'))
    res['replay_subset'] = T
    jdump(res, f'{EXD}/summary_{rung}.json')
    for period in ('all', 'early', 'main', 'recent'):
        print(period)
        for t in res[period]:
            print(f"  {t['method'][:34]:34s} n={t['n_days']:6d} AP {t['ap_mean']:.3f} {t['ap_ci']} F1 {t['f1_mean']:.3f} "
                  f"Dice {t['dice_total_median']:.3f} logA {t['log10_area_ratio_median']:+.2f}")
    print('replay subset')
    for t in res['replay_subset']:
        print(f"  {t['method'][:44]:44s} n={t['n_days']:5d} AP {t['ap_mean']:.3f} F1 {t['f1_mean']:.3f} "
              f"Dice {t['dice_total_median']:.3f} logA {t['log10_area_ratio_median']:+.2f}")




def _gamma(fid):
    from firesim import realdata as RD
    r = fd()._row(fid)
    return float(RD.grid_convergence_deg(float(r.cx), float(r.cy)))


def examples(rung, n_web=12, n_demo=3, win=240):
    """Fixed seeded draw of exam growth fire-days within size strata (no cherry-picking) for the page and demo.
    Saves raw inputs + FireCast/fl_prev maps on a <= win x win window around the burned area."""
    import pickle
    import features as FT
    from scipy import ndimage
    days = pd.read_parquet(f'{PROC}/baselines/fire_days_exam.parquet')
    g = days[days.new_obs_km2 > 0].copy()
    fires = fd().fires('exam').set_index('Event_ID')
    g['period'] = g.fire_id.map(fires.exam_period)
    g['name'] = g.fire_id.map(fires.Incid_Name)
    g['stratum'] = pd.cut(g.area_km2, [0, 5, 50, 1e9], labels=['small', 'medium', 'large'])
    rng = np.random.default_rng(20261119)
    pick = []
    for st in ('small', 'medium', 'large'):
        sub = g[g.stratum == st].sort_values(['fire_id', 'k'])
        idx = rng.choice(len(sub), size=n_web // 3, replace=False)
        pick += [dict(sub.iloc[i]) for i in idx]
    net = EV.load_net(rung)
    cal = json.load(open(shipped(f'cal_{rung}_F.json')))
    P = json.load(open(f'{PROC}/baselines/spread_params.json'))
    out = []
    for p in pick:
        d = [x for x in RDY.fire_days(p['fire_id'], for_exam_scoring=True, with_land=False) if x['k'] == p['k']][0]
        X, s = EV.real_inputs(d)
        lg = EV.logits_full(net, X, s)
        prob = EV.sig(lg / cal['T'] + cal['b'])
        m = EV.score_real(d, prob, cal['thr'])
        # fl_prev baseline map (B's definition)
        now = d['now']
        dist = np.maximum(ndimage.distance_transform_edt(~now, sampling=RES) - RES / 2, 0)
        sb = int(np.digitize(d['area_km2'], P['size_bins_km2']) - 1)
        dgp = max(d['d_prev_m'], 0.0) if np.isfinite(d['d_prev_m']) else P['clim_d_m'][sb]
        dfl = np.maximum(ndimage.distance_transform_edt(~d['fl'], sampling=RES) - RES / 2, 0) if (d['fl'] is not None and d['fl'].any()) else dist
        flp = 1 / (1 + (dfl / max(dgp, 1.0)) ** 4) if dgp > 0 else np.zeros(now.shape)
        mb = EV.score_real(d, flp)
        # window
        H, W = now.shape
        rr, cc = np.nonzero(now)
        hh, ww = min(win, H // 16 * 16), min(win, W // 16 * 16)
        r0 = int(np.clip(rr.mean() - hh / 2, 0, H - hh))
        c0 = int(np.clip(cc.mean() - ww / 2, 0, W - ww))
        F = FT.full_fields(now, d['prev'] if d['has_prev'] else None, d['fl'], d['pix_pm'], d['pix_am'], None)
        sl = (slice(r0, r0 + hh), slice(c0, c0 + ww))
        rec = dict(meta=dict(fire_id=d['fire_id'], name=str(p['name']).title(), k=int(d['k']), t=d['t'], period=str(p['period']),
                             stratum=str(p['stratum']), area_km2=float(d['area_km2']), new_obs_km2=float(d['new_obs_km2']),
                             firecast=dict(ap=m['ap'], f1=m['f1'], new_pred_km2=m['new_pred_km2']),
                             fl_prev=dict(ap=mb['ap'], f1=mb['f1'], new_pred_km2=mb['new_pred_km2']),
                             window=(r0, c0, hh, ww), crop_shape=(H, W), gamma=_gamma(d['fire_id'])),
                   wx=s[:16], sc=s[16:], fields={k: v[sl] for k, v in F.items()},
                   now=now[sl], nxt=d['nxt'][sl], prev=None if d['prev'] is None else d['prev'][sl],
                   fl=None if d['fl'] is None else d['fl'][sl], prob=prob[sl].astype(np.float32),
                   flprev=flp[sl].astype(np.float32), raw=d if len(out) < 99 else None)
        out.append(rec)
        print(rec['meta'], flush=True)
    pickle.dump(out, open(f'{EXD}/examples_{rung}.pkl', 'wb'))


if __name__ == '__main__':
    torch.set_num_threads(1)
    {'run': run, 'summary': summary, 'examples': examples}[sys.argv[1]](sys.argv[2])
