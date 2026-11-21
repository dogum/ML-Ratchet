"""Fit the FEDS emulator (b, a, d) on DEV fires: real VIIRS pixels -> emulated perimeters / fire lines vs the real
FEDS products. Output out/calib_feds.json. Dev only (exam fires are never loaded)."""
import itertools
import time
from common import *  # noqa
import pandas as pd
from pyproj import Transformer
import feds_emu as E

fd_ = fd()
TR = Transformer.from_crs(4326, 5070, always_xy=True)
FL_CACHE = {}


def firelines(year):
    if year not in FL_CACHE:
        FL_CACHE.clear()
        FL_CACHE[year] = pd.read_parquet(f'{PROC}/feds/firelines/{year}.parquet')
    return FL_CACHE[year]


def fire_inputs(fid):
    import shapely
    assert not fd_.guard.is_exam(fid)
    r = fd_._row(fid)
    pr = fd_.perimeters(fid)
    g = fd_.grid(fid)
    pix = fd_.firepix(fid)
    x, y = TR.transform(pix.Lon.values, pix.Lat.values)
    a, _, c0, _, e, f0 = g['transform']
    pix['col'] = (x - c0) / a - 0.5
    pix['row'] = (y - f0) / e - 0.5
    steps = list(pr.t)
    pts = [E.points_raster(pix.row[pix.t == t], pix.col[pix.t == t], g['shape']) for t in steps]
    real = np.maximum.accumulate(np.stack([fd_._grid.burn([gm], g) for gm in pr.geometry]), axis=0)
    fl = firelines(int(r.year))
    fl = fl[fl.Event_ID == fid]
    lines = []
    for t in steps:
        ln = fl[fl.t == t]
        lines.append(fd_._grid.burn(list(shapely.from_wkb(ln.wkb.values)), g, all_touched=True) if len(ln)
                     else np.zeros(g['shape'], bool))
    # crop to the final mask + 3 km
    rr, cc = np.nonzero(real[-1] | np.any(np.stack(pts), 0))
    p = 40
    sl = (slice(max(rr.min() - p, 0), rr.max() + p + 1), slice(max(cc.min() - p, 0), cc.max() + p + 1))
    return dict(pts=[q[sl] for q in pts], real=real[:, sl[0], sl[1]], lines=[q[sl] for q in lines],
                pm=(pr.half == 'PM').values, fid=fid)


def score(F, prm):
    masks, lines = E.sequence(F['pts'], F['pts'][0].shape, RES, prm)
    dice = [fd_.metrics.dice(m, r) for m, r in zip(masks, F['real'])]
    ratio = [(m.sum() + 1) / (r.sum() + 1) for m, r in zip(masks, F['real'])]
    # fire line agreement with 1-cell tolerance (both directions)
    lf = []
    for l_e, l_r in zip(lines, F['lines']):
        if not l_r.any() and not l_e.any():
            continue
        de = ndimage.binary_dilation(l_e, iterations=1)
        dr = ndimage.binary_dilation(l_r, iterations=1)
        prec = (l_e & dr).sum() / max(l_e.sum(), 1)
        rec = (l_r & de).sum() / max(l_r.sum(), 1)
        lf.append(2 * prec * rec / max(prec + rec, 1e-9))
    return float(np.nanmean(dice)), float(np.median(np.log10(ratio))), float(np.mean(lf)) if lf else np.nan


def main():
    from scipy import ndimage as _nd  # noqa
    global ndimage
    ndimage = _nd
    ids = list(fd_.fires('dev').Event_ID)
    t0 = time.time()
    F = [fire_inputs(f) for f in ids]
    print('loaded', len(F), f'{time.time() - t0:.0f}s', flush=True)
    grid_b = [150.0, 190.0, 250.0]
    grid_a = [0.0, 300.0, 600.0, 1000.0, 1500.0]
    res = []
    for b, a in itertools.product(grid_b, grid_a):
        s = [score(f, dict(b_m=b, a_m=a, d_m=750.0)) for f in F]
        row = dict(b_m=b, a_m=a, dice_mean=float(np.mean([x[0] for x in s])),
                   log_ratio_median=float(np.median([x[1] for x in s])))
        res.append(row)
        print(row, flush=True)
    best = max(res, key=lambda r: r['dice_mean'])
    res_d = []
    for d in [375.0, 500.0, 750.0, 1000.0, 1500.0]:
        s = [score(f, dict(b_m=best['b_m'], a_m=best['a_m'], d_m=d)) for f in F]
        row = dict(d_m=d, line_f1_mean=float(np.nanmean([x[2] for x in s])))
        res_d.append(row)
        print(row, flush=True)
    bd = max(res_d, key=lambda r: r['line_f1_mean'])
    out = dict(best=dict(b_m=best['b_m'], a_m=best['a_m'], d_m=bd['d_m']), grid_perimeter=res, grid_line=res_d,
               n_fires=len(F), note='dev fires only; real VIIRS S-NPP pixels -> emulated FEDS vs real FEDS (all 12-h steps)')
    # per-fire dice at best
    out['per_fire'] = {f['fid']: score(f, out['best']) for f in F}
    jdump(out, f'{OUT}/calib_feds.json')
    print('BEST', out['best'], flush=True)


if __name__ == '__main__':
    main()
