"""Calibrate the FEDS emulator (VIIRS pixel footprints -> FEDS-like burned mask) on DEV fires only.
Apply the emulator to the REAL dev VIIRS pixels and compare the emulated area with the real FEDS perimeter area per
half-day step. Output: out/calib_feds.json."""
from common import *
import firedata as fd

import importlib
G = importlib.import_module('firedata.grid')
import feats

RES = feats.RES


def real_steps(fid, for_exam=False):
    pr = fd.perimeters(fid, for_exam_scoring=for_exam)
    fp = fd.firepix(fid, for_exam_scoring=for_exam)
    x, y = to5070(fp.Lon.values.astype(float), fp.Lat.values.astype(float))
    fp = fp.assign(x=x, y=y)
    steps = []
    for t in pr.t:
        d = fp[fp.t == t]
        steps.append(dict(x=d.x.values, y=d.y.values, ds=d.DS.values.astype(float), dt=d.DT.values.astype(float),
                          frp=d.FRP.values.astype(float), daynight=d.DNFlag.values))
    return pr, steps


if __name__ == '__main__':
    dev = fd.fires('dev')
    recs = []
    cache = {}
    for fid in dev.Event_ID:
        pr, steps = real_steps(fid)
        g = G.grid_from_window(*fd.grid(fid)['bounds'], RES)
        masks = np.stack([G.burn([geom], g) for geom in pr.geometry])
        masks = np.maximum.accumulate(masks, axis=0)
        cache[fid] = (pr, steps, g, masks)
    grid = [(c, s) for c in (0, 1, 2, 3, 4) for s in (0.5, 0.75, 1.0)]
    res = {}
    for c, s in grid:
        lr = []
        for fid, (pr, steps, g, masks) in cache.items():
            x0, y0, x1, y1 = g['bounds']
            em = feats.feds_emulate(g['shape'], x0, y1, RES, steps, dict(close_cells=c, shrink=s))
            a_real = masks.reshape(len(masks), -1).sum(1)
            a_em = em.reshape(len(em), -1).sum(1)
            ok = a_real > 0
            lr += list(np.log10((a_em[ok] + 1) / (a_real[ok] + 1)))
        lr = np.array(lr)
        res[f'{c}_{s}'] = dict(close_cells=c, shrink=s, n=len(lr), median_log10=float(np.median(lr)),
                               mad_log10=float(np.median(np.abs(lr))), p10=float(np.percentile(lr, 10)),
                               p90=float(np.percentile(lr, 90)))
        print(c, s, res[f'{c}_{s}'], flush=True)
    best = min(res.values(), key=lambda r: r['mad_log10'])
    jdump(dict(what='FEDS emulator (pixel footprints -> closed, hole-filled mask) applied to REAL dev VIIRS pixels vs '
                    'real FEDS perimeters, 180 m grid, all half-day steps with area > 0; dev only',
               grid=res, best=best), f'{ROOT}/out/calib_feds.json')
    print('best', best)
