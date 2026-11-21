"""EXAM baseline: the dev-calibrated physics replay (firesim.replay + checks/replay_calibration.json, exactly as stored)
on the 48-fire exam replay subset, on the same fire-days and score windows as FireCast (B's definition: PM -> PM 24 h,
growth days, cells within 10 km). Run once, after FireCast is frozen.
Memory guard (shared 1.2 GB/process limit): fire-days whose crop exceeds MAX_CELLS (500k cells = 4,050 km2) are
skipped and logged (skip='crop > MAX_CELLS'); FireCast is compared on exactly the replayed days.
Usage: python3 exam_replay.py [n_members=12]   -> out/exam/replay_calibrated.jsonl (resumable per fire)"""
MAX_CELLS = 500_000
import time
from common import *  # noqa
import pandas as pd
import realdays as RDY
import evaluate as EV


def main(n_members=12):
    import firesim as fs_
    from firesim import realdata as RD
    from firesim import replay as RP
    fd_ = fd()
    cal = json.load(open(f'{FIRE}/checks/replay_calibration.json'))
    ids = json.load(open(f'{PROC}/baselines/replay_subset.json'))['fires']
    fn = f'{OUT}/exam/replay_calibrated.jsonl'
    done = {json.loads(l)['fire_id'] for l in open(fn)} if os.path.exists(fn) else set()
    t0 = time.time()
    with open(fn, 'a') as fo:
        for fid in ids:
            if fid in done:
                continue
            rows = []
            try:
                L = fd_.landscape(fid, for_exam_scoring=True)
                W = fd_.weather(fid, for_exam_scoring=True)
                r = fd_._row(fid)
                lon, lat = fd_._lonlat(r)
                ls_full = RD.landscape_from_firedata(L)
                for d in RDY.fire_days(fid, for_exam_scoring=True, with_land=False):
                    y0, y1, x0, x1 = d['box']
                    if (y1 - y0) * (x1 - x0) > MAX_CELLS:
                        rows.append(dict(fire_id=fid, k=d['k'], skip='crop > MAX_CELLS', cells=(y1 - y0) * (x1 - x0)))
                        continue
                    ls = ls_full.subset(y0, y1, x0, x1)
                    gt = ls.meta['transform']
                    rr, cc = np.nonzero(d['now'])
                    w = RD.weather_series(W, gt[0] + (cc.mean() + 0.5) * gt[1], gt[3] + (rr.mean() + 0.5) * gt[5])
                    w = RD.fill_precip_from_daily(w, W.get('daily'), lon)
                    start = (pd.Timestamp(d['t']) + pd.Timedelta(hours=1.5 - lon / 15.0)).tz_localize('UTC')
                    end = start + pd.Timedelta(hours=24)
                    if w.index[0] > start - pd.Timedelta(hours=1) or w.index[-1] < end - pd.Timedelta(hours=1):
                        rows.append(dict(fire_id=fid, k=d['k'], skip='weather period'))
                        continue
                    prm = RP.calibrated_params(cal, fire_day_index=d['k'])
                    res = RP.replay_day(ls, w, d['now'], start.floor('h'), end, params=prm,
                                        ensemble=dict(n=n_members) if n_members else None, rng=d['k'])
                    m = EV.score_real(d, res['pred'].astype(float))
                    m['ap'] = EV.score_real(d, res['score'])['ap']
                    m.update(fire_id=fid, k=d['k'], kind='det', wall_s=res['wall_s'])
                    rows.append(m)
                    if res['prob'] is not None:
                        me = EV.score_real(d, res['prob'])
                        me.update(fire_id=fid, k=d['k'], kind='ens')
                        rows.append(me)
            except Exception as ex:
                rows.append(dict(fire_id=fid, error=repr(ex)))
            fo.write(json.dumps(dict(fire_id=fid, rows=rows), default=float) + '\n')
            fo.flush()
            print(fid, len(rows), f'{time.time() - t0:.0f}s', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
