"""FireCast demo: three real exam fire-days (a fixed seeded draw: one small, one medium, one large fire).
Runs the numpy model from raw inputs and scores it exactly like the exam. Usage: python demo.py"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import firecast as fc  # noqa: E402


def load_days(fn):
    with np.load(fn) as f:
        z = {k: f[k] for k in f.files}
    meta = json.loads(str(z['meta']))
    days = []
    for j, m in enumerate(meta):
        p = f'd{j}_'
        shape = tuple(z[p + 'shape'])
        n = int(np.prod(shape))
        bits = lambda k: (np.unpackbits(z[p + k])[:n].reshape(shape).astype(bool) if p + k in z else None)
        pix = lambda k: dict(zip(('row', 'col', 'ps', 'pt', 'frp'), z[p + k]))
        days.append(dict(now=bits('now'), nxt=bits('nxt'), prev=bits('prev'), fl=bits('fl'), pix_pm=pix('pix_pm'),
                         pix_am=pix('pix_am'), wx=z[p + 'wx'], has_prev=m['has_prev'], first=m['first'], meta=m))
    return days


def main():
    net = fc.FireCastNP(os.path.join(HERE, 'firecast_weights.npz'))
    print(f'FireCast v1.0  (calibration T={net.cal_T:.3f} b={net.cal_b:+.3f}, threshold {net.thr:.2f})\n')
    for d in load_days(os.path.join(HERE, 'examples', 'demo_days.npz')):
        m = d['meta']
        t0 = time.time()
        prob = fc.predict(net, d, land=False)
        win = fc.score_window(d['now'])
        s = fc.nextday(d['now'], d['nxt'], prob, fc.CELL_KM2, thr=net.thr, window=win)
        print(f"{m['name']} ({m['fire_id']}), {m['date']}, fire-day {m['k']}, {m['stratum']}, exam period {m['period']}")
        print(f"  burned at issue {m['area_km2']:.1f} km2; observed new burn {s['new_obs_km2']:.2f} km2; "
              f"FireCast predicts {s['new_pred_km2']:.2f} km2 at the threshold")
        print(f"  FireCast AP {s['ap']:.3f}  F1 {s['f1']:.3f}   (exam run: AP {m['exam_ap']:.3f} F1 {m['exam_f1']:.3f})")
        print(f"  baseline 'yesterday's growth from the fire line': AP {m['fl_prev_ap']:.3f} F1 {m['fl_prev_f1']:.3f}")
        print(f"  grid {d['now'].shape}, {time.time() - t0:.1f} s\n")


if __name__ == '__main__':
    main()
