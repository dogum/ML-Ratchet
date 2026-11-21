"""Dev-fire replays: the classical physics benchmark for next-day spread (DEV fires only; exam fires never touched).

For each dev fire (fire/firedata: LANDFIRE 90 m landscape, hourly HRRR, FEDS PM-to-PM daily perimeters) and each
fire-day i -> i+1 (12-36 h apart, burned area > 0 at i):
  * start from the observed burned mask at the PM overpass of day i, spread to the PM overpass of day i+1;
  * weather: HRRR hourly series at the cell nearest the burned centroid (wind rotated to grid north);
  * crop: burned bbox + 20 km; crown fire on; no spotting; no suppression;
  * runs: uncalibrated deterministic + 12-member ensemble (wind speed x LN(0.25), direction + N(0, 20 deg), ROS x
    LN(0.5)); a sweep of global ROS factors {0.05 ... 2} (deterministic) for calibration;
  * metrics: firedata.metrics.nextday (B's exam definitions) on the fire's full 90 m window grid; AP of the
    deterministic run uses the score exp(-arrival / dt).
Phases (resumable, one json per fire in checks/out/replay/):
  python3 replay_dev.py sweep       # uncalibrated + factor sweep
  python3 replay_dev.py calibrate   # per-fuel-group factors (dev, in-sample) + leave-one-fire-out estimate
  python3 replay_dev.py calibrated  # calibrated deterministic + ensemble
  python3 replay_dev.py report      # -> checks/replay_dev.json
"""
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

from _common import write, OUT, _clean
import firesim as fs
from firesim import realdata as RD
from firesim import replay as RP

fd = RD._fd()
OUTD = os.path.join(OUT, 'replay')
CALD = os.path.join(OUT, 'replay_cal')
os.makedirs(OUTD, exist_ok=True)
os.makedirs(CALD, exist_ok=True)
FACTORS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0]
MARGIN_M = 20000.0
CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'replay_calibration.json')


def dev_ids():
    f = fd.fires('dev')
    assert (f.split == 'dev').all()
    return list(f.sort_values('feds_area_km2').Event_ID)      # small fires first


def fire_days(fid):
    """Yield fire-day inputs for a dev fire."""
    assert not fd.guard.is_exam(fid)
    L = fd.landscape(fid)
    W = fd.weather(fid)
    if 'grid' not in W:
        raise FileNotFoundError('no HRRR package yet')
    ls_full = RD.landscape_from_firedata(L)
    prog = fd.progression(fid)
    B = prog['burned']
    tt = pd.DatetimeIndex(prog['t_utc']).tz_localize('UTC')
    wt0 = pd.Timestamp(np.asarray(W['grid']['time'])[0]).tz_localize('UTC')
    wt1 = pd.Timestamp(np.asarray(W['grid']['time'])[-1]).tz_localize('UTC')
    gt = ls_full.meta['transform']
    for i in range(len(B) - 1):
        now, nxt = B[i], B[i + 1] | B[i]
        dt = (tt[i + 1] - tt[i]) / pd.Timedelta(hours=1)
        info = dict(fire_id=fid, i=i, t0=str(tt[i]), t1=str(tt[i + 1]), dt_h=float(dt), n_now=int(now.sum()),
                    n_new_obs=int((nxt & ~now).sum()))
        if now.sum() == 0 or not (12 <= dt <= 36) or info['n_new_obs'] == 0:
            info['skip'] = 'empty start' if now.sum() == 0 else (f'gap {dt:.0f} h' if not (12 <= dt <= 36) else
                                                                 'no growth (not scored: growth days only)')
            yield info, None
            continue
        start, end = tt[i].floor('h'), tt[i + 1]
        if start < wt0 or end > wt1 + pd.Timedelta(hours=1):
            info['skip'] = 'weather period'
            yield info, None
            continue
        r0, r1, c0, c1 = RP.crop_box(now, ls_full.res, MARGIN_M)
        ls = ls_full.subset(r0, r1, c0, c1)
        rr, cc = np.nonzero(now)
        x = gt[0] + (cc.mean() + 0.5) * gt[1]
        y = gt[3] + (rr.mean() + 0.5) * gt[5]
        w = RD.weather_series(W, x, y)
        w = RD.fill_precip_from_daily(w, W.get('daily'), w.attrs['lonlat'][0])
        yield info, dict(ls=ls, box=(r0, r1, c0, c1), now=now, nxt=nxt, start=start, end=end, w=w,
                         shape=now.shape, res=ls_full.res)


def embed(a, box, shape, fill=0.0):
    out = np.full(shape, fill, np.float32)
    r0, r1, c0, c1 = box
    out[r0:r1, c0:c1] = a
    return out


def score_day(d, pred_c, score_c=None, prob_c=None):
    cell_km2 = d['res'] ** 2 / 1e6
    pred = embed(pred_c.astype(np.float32), d['box'], d['shape']) > 0.5
    m = fd.metrics.nextday(d['now'], d['nxt'], pred.astype(float), cell_km2)
    if score_c is not None:
        s = embed(score_c, d['box'], d['shape'])
        cand = ~d['now']
        m['ap'] = fd.metrics.average_precision(d['nxt'][cand], s[cand])
    if prob_c is not None:
        p = embed(prob_c, d['box'], d['shape'])
        mp = fd.metrics.nextday(d['now'], d['nxt'], p, cell_km2)
        m = dict(det=m, ens=mp)
    return m


def run_sweep(fid):
    fn = os.path.join(OUTD, fid + '.json')
    if os.path.exists(fn):
        return
    rows = []
    t0 = time.time()
    for info, d in fire_days(fid):
        if d is None:
            rows.append(info)
            continue
        grp, share = RP.ring_fuel_group(d['ls'], d['now'][d['box'][0]:d['box'][1], d['box'][2]:d['box'][3]])
        now_c = d['now'][d['box'][0]:d['box'][1], d['box'][2]:d['box'][3]]
        res = RP.replay_day(d['ls'], d['w'], now_c, d['start'], d['end'], ensemble=True, rng=info['i'],
                            keep_prepared=True)
        m = score_day(d, res['pred'], res['score'], res['prob'])
        info['persist_dice_total'] = fd.metrics.dice(d['now'], d['nxt'])
        info.update(ring_group=grp, ring_share=share, wind_mean_ms=res['labels']['wind_mean_ms'],
                    uncal=m, wall_s=res['wall_s'],
                    crown_frac_new=float((res['fire_type'][res['pred'] & ~now_c] >= 2).mean()) if (res['pred'] & ~now_c).any() else 0.0)
        pr = res['prepared']
        sw = {}
        for f in FACTORS:
            if f == 1.0:
                sw[str(f)] = {k: m['det'][k] for k in ('f1', 'dice_new', 'dice_total', 'log10_area_ratio', 'new_pred_km2', 'ap')}
                continue
            o = fs.simulate(d['ls'], d['w'], {'mask': now_c}, d['start'], d['end'], spotting=False,
                            prepared=RP._perturbed(pr, 1.0, 0.0, f), return_perimeters=False)
            a = o['arrival_time']
            pred = np.isfinite(a) & (a <= res['dt_h'])
            sc = np.where(np.isfinite(a), np.exp(-np.nan_to_num(a, nan=1e9) / res['dt_h']), 0.0)
            mm = score_day(d, pred, sc)
            sw[str(f)] = {k: mm[k] for k in ('f1', 'dice_new', 'dice_total', 'log10_area_ratio', 'new_pred_km2', 'ap')}
        info['sweep'] = sw
        rows.append(info)
        print(fid, info['i'], grp, 'n_now', info['n_now'], 'new obs', info['n_new_obs'], 'f1 det %.3f ens %.3f ap %.3f' % (
            m['det']['f1'], m['ens']['f1'], m['ens']['ap'] if m['ens']['ap'] == m['ens']['ap'] else -1),
            '%.1fs' % (time.time() - t0), flush=True)
    with open(fn + '.tmp', 'w') as f:
        json.dump(_clean(dict(fire_id=fid, rows=rows, wall_s=time.time() - t0)), f)
    os.replace(fn + '.tmp', fn)


def load_rows(d=OUTD):
    rows = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith('.json'):
            rows += json.load(open(os.path.join(d, fn)))['rows']
    return [r for r in rows if 'skip' not in r]


def fit_factors(rows, min_days=5):
    """Per ring fuel group: the sweep factor with the highest mean F1 on growth days (dev). Groups with fewer than
    min_days growth days get the all-groups factor."""
    g = [r for r in rows if r['n_new_obs'] > 0]

    def best(sub):
        if not sub:
            return 1.0, None
        m = {f: np.nanmean([r['sweep'][str(f)]['f1'] for r in sub]) for f in FACTORS}
        fb = max(m, key=lambda k: (m[k] if m[k] == m[k] else -1))
        return fb, m
    f_all, m_all = best(g)
    out = dict(default=f_all, factors={}, n_days={}, mean_f1_by_factor={'all': m_all})
    for grp in sorted(set(r['ring_group'] for r in g)):
        sub = [r for r in g if r['ring_group'] == grp]
        out['n_days'][grp] = len(sub)
        if len(sub) >= min_days:
            fb, m = best(sub)
            out['factors'][grp] = fb
            out['mean_f1_by_factor'][grp] = m
    # first observed day of a fire (i == 0: the fire is young, little suppression yet): one pooled factor
    first = [r for r in g if r['i'] == 0]
    if len(first) >= min_days:
        out['factor_first_day'], out['mean_f1_by_factor']['first_day'] = best(first)
        out['n_days']['first_day'] = len(first)
    # also the area-matching factor (median log10 area ratio closest to 0), for reference
    am = {f: abs(np.nanmedian([r['sweep'][str(f)]['log10_area_ratio'] for r in g])) for f in FACTORS}
    out['area_matching_factor_all'] = min(am, key=am.get)
    return out


def factor_for(cal, group, i):
    if i == 0 and 'factor_first_day' in cal:
        return cal['factor_first_day']
    return cal['factors'].get(group, cal['default'])


def calibrate():
    rows = load_rows()
    fires = sorted(set(r['fire_id'] for r in rows))
    cal = fit_factors(rows)
    # leave-one-fire-out: factors fitted without the fire, applied through the sweep (global factor of its group)
    lofo = []
    for fid in fires:
        cf = fit_factors([r for r in rows if r['fire_id'] != fid])
        for r in rows:
            if r['fire_id'] == fid and r['n_new_obs'] > 0:
                f = factor_for(cf, r['ring_group'], r['i'])
                lofo.append(dict(fire_id=fid, i=r['i'], factor=f, **r['sweep'][str(f)]))
    lo = pd.DataFrame(lofo)
    cal['lofo'] = dict(n_fire_days=len(lo), f1_mean=float(lo.f1.mean()), f1_median=float(lo.f1.median()),
                       dice_total_median=float(lo.dice_total.median()), ap_mean=float(lo.ap.mean()),
                       log10_area_ratio_median=float(lo.log10_area_ratio.median()))
    cal['fitted_on'] = dict(n_fires=len(fires), fires=fires, n_growth_days=int(sum(r['n_new_obs'] > 0 for r in rows)),
                            split='dev only', objective='mean F1 of new burned cells (firedata.metrics.nextday)')
    cal['ensemble'] = RP.ENSEMBLE
    cal['how_to_use'] = ("params = firesim.replay.calibrated_params(json.load(open('checks/replay_calibration.json'))); "
                         "firesim.replay.replay_day(ls, w, burned_now, start, end, params=params, ensemble=True)")
    with open(CALIB_FILE, 'w') as f:
        json.dump(_clean(cal), f, indent=1)
    print(json.dumps(_clean({k: v for k, v in cal.items() if k != 'mean_f1_by_factor'}), indent=1))


def run_calibrated(fid):
    fn = os.path.join(CALD, fid + '.json')
    if os.path.exists(fn):
        return
    cal = json.load(open(CALIB_FILE))
    params = RP.calibrated_params(cal)
    rows = []
    t0 = time.time()
    for info, d in fire_days(fid):
        if d is None:
            continue
        now_c = d['now'][d['box'][0]:d['box'][1], d['box'][2]:d['box'][3]]
        p = params
        if info['i'] == 0 and 'factor_first_day' in cal:
            p = RP.calibrated_params(dict(factors={}, default=cal['factor_first_day']))
        res = RP.replay_day(d['ls'], d['w'], now_c, d['start'], d['end'], params=p, ensemble=True,
                            rng=1000 + info['i'])
        info['cal'] = score_day(d, res['pred'], res['score'], res['prob'])
        info['wall_s'] = res['wall_s']
        rows.append(info)
        print(fid, info['i'], 'cal f1 det %.3f ens %.3f' % (info['cal']['det']['f1'], info['cal']['ens']['f1']), flush=True)
    with open(fn + '.tmp', 'w') as f:
        json.dump(_clean(dict(fire_id=fid, rows=rows, wall_s=time.time() - t0)), f)
    os.replace(fn + '.tmp', fn)


def summarize(rows, key):
    out = {}
    for k in ('det', 'ens'):
        rr = [dict(fire_id=r['fire_id'], **r[key][k]) for r in rows if key in r]
        out[k] = fd.metrics.summarize(rr)
    return out


def held_out(rows, cal):
    """Fires not used to fit the calibration: calibrated deterministic score via the sweep (global factor of the
    fire-day's ring group), vs uncalibrated, on the same fire-days."""
    if not cal:
        return None
    fitted = set(cal['fitted_on'].get('fires', []))
    out = {}
    for fid in sorted(set(r['fire_id'] for r in rows) - fitted):
        rr = [r for r in rows if r['fire_id'] == fid and r['n_new_obs'] > 0]
        cs = []
        for r in rr:
            sw = dict(r['sweep'][str(factor_for(cal, r['ring_group'], r['i']))])
            sw['area_err_km2'] = sw['new_pred_km2'] - r['uncal']['det']['new_obs_km2']
            cs.append(dict(fire_id=fid, **sw))
        us = [dict(fire_id=fid, **r['uncal']['det']) for r in rr]
        out[fid] = dict(n_growth_days=len(rr), calibrated_det_via_sweep=fd.metrics.summarize(cs, growth_only=False),
                        uncalibrated_det=fd.metrics.summarize(us, growth_only=False),
                        uncalibrated_ens=fd.metrics.summarize([dict(fire_id=fid, **r['uncal']['ens']) for r in rr],
                                                              growth_only=False))
    return out


def report():
    rows = load_rows()
    crow = {(r['fire_id'], r['i']): r for r in load_rows(CALD)}
    for r in rows:
        c = crow.get((r['fire_id'], r['i']))
        if c:
            r['cal'] = c['cal']
    cal = json.load(open(CALIB_FILE)) if os.path.exists(CALIB_FILE) else None
    res = dict(
        what='Dev-fire next-day replays (FEDS PM-to-PM) with FireSim: uncalibrated and dev-calibrated (per-fuel-group '
             'ROS factors). Metrics from firedata.metrics (B\'s exam definitions); growth days only.',
        n_fires=len(set(r['fire_id'] for r in rows)), n_fire_days=len(rows),
        n_growth_days=int(sum(r['n_new_obs'] > 0 for r in rows)),
        uncalibrated=summarize(rows, 'uncal'),
        calibrated_in_sample=summarize([r for r in rows if 'cal' in r], 'cal') if crow else None,
        calibration=cal and {k: cal.get(k) for k in ('default', 'factors', 'factor_first_day', 'n_days',
                                                        'area_matching_factor_all', 'lofo', 'fitted_on')},
        persistence=dict(f1=0.0, dice_total_median=float(np.median([r['persist_dice_total'] for r in rows if r['n_new_obs'] > 0])),
                         note='no growth predicted: F1 = 0, AP undefined (constant score)'),
        by_group={g: dict(n=int(sum(1 for r in rows if r['ring_group'] == g and r['n_new_obs'] > 0)),
                          uncal_f1_mean=float(np.nanmean([r['uncal']['det']['f1'] for r in rows if r['ring_group'] == g and r['n_new_obs'] > 0] or [np.nan])),
                          cal_f1_mean=float(np.nanmean([r['cal']['det']['f1'] for r in rows if r['ring_group'] == g and r['n_new_obs'] > 0 and 'cal' in r] or [np.nan])),
                          uncal_log10_area_ratio_median=float(np.nanmedian([r['uncal']['det']['log10_area_ratio'] for r in rows if r['ring_group'] == g and r['n_new_obs'] > 0] or [np.nan])))
                  for g in sorted(set(r['ring_group'] for r in rows))},
        held_out_fires=held_out(rows, cal),
        wall_s_median=float(np.median([r['wall_s'] for r in rows])),
        crown_frac_new_mean=float(np.mean([r.get('crown_frac_new', 0) for r in rows])),
    )
    write('replay_dev.json', res)
    print(json.dumps(_clean({k: v for k, v in res.items() if k != 'by_group'}), indent=1)[:4000])


if __name__ == '__main__':
    phase = sys.argv[1]
    if phase in ('sweep', 'calibrated'):
        ids = dev_ids() if len(sys.argv) < 3 else sys.argv[2:]
        for fid in ids:
            try:
                (run_sweep if phase == 'sweep' else run_calibrated)(fid)
            except FileNotFoundError as e:
                print(fid, 'not ready:', e, flush=True)
            except Exception:
                print(fid, 'FAILED', flush=True)
                traceback.print_exc()
    elif phase == 'calibrate':
        calibrate()
    elif phase == 'report':
        report()
