"""release/demo_days.npz: three real exam fire-days (examples #0, #4, #8 of the seeded stratified draw: one small,
one medium, one large), raw inputs exactly as the exam script built them, plus the exam-run scores for comparison."""
import pickle
from common import *  # noqa


def main(rung):
    ex = pickle.load(open(f'{OUT}/exam/examples_{rung}.pkl', 'rb'))
    Z, meta = {}, []
    for j, i in enumerate((0, 4, 8)):
        d = ex[i]['raw']
        p = f'd{j}_'
        for k in ('now', 'nxt', 'prev', 'fl'):
            if d[k] is not None:
                Z[p + k] = np.packbits(np.asarray(d[k], bool))
        Z[p + 'shape'] = np.array(d['now'].shape)
        for k in ('pix_pm', 'pix_am'):
            t = d[k]
            Z[p + k] = np.stack([np.asarray(t[c], float) for c in ('row', 'col', 'ps', 'pt', 'frp')]) if len(t['row']) else np.zeros((5, 0))
        Z[p + 'wx'] = np.asarray(d['wx'], np.float32)
        m = ex[i]['meta']
        meta.append(dict(fire_id=d['fire_id'], name=m['name'], date=d['t'][:10], k=int(d['k']), period=m['period'],
                         stratum=m['stratum'], has_prev=bool(d['has_prev']), first=bool(d['first']),
                         exam_ap=m['firecast']['ap'], exam_f1=m['firecast']['f1'], fl_prev_ap=m['fl_prev']['ap'],
                         fl_prev_f1=m['fl_prev']['f1'], area_km2=float(d['area_km2']), new_obs_km2=float(d['new_obs_km2'])))
    Z['meta'] = np.array(json.dumps(meta))
    np.savez_compressed(f'{ROOT}/release/demo_days.npz', **Z)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
