"""Real fire-days (FEDS-MTBS + VIIRS + gridMET [+ LANDFIRE]) in FireCast's raw record format.
Fire-day definition = firedata baselines (B): issue PM perimeter k -> verify PM perimeter k+1 exactly 24 h later,
crop = burned bbox at issue + 10 km + 2 cells, scored cells = firedata.metrics.score_window (10 km).
Exam fires need for_exam_scoring=True (only the once-only exam script passes it)."""
from common import *  # noqa
import pandas as pd
import shapely
from pyproj import Transformer
import features as FT

TR = Transformer.from_crs(4326, 5070, always_xy=True)
_FL = {}
_FD = {}
PAD = int(np.ceil(WIN_KM * 1000 / RES)) + 2


def firelines(year):
    if year not in _FL:
        _FL.clear()
        _FL[year] = pd.read_parquet(f'{PROC}/feds/firelines/{year}.parquet')
    return _FL[year]


def fire_days_table(split):
    if split not in _FD:
        _FD[split] = pd.read_parquet(f'{PROC}/baselines/fire_days_{split}.parquet')
    return _FD[split]


def gridmet_row(row, day):
    return {k: row[f'{k}_{day}'] for k in ('vs', 'th', 'rmin', 'tmmx', 'vpd', 'pr', 'fm100')}


def fire_days(fid, for_exam_scoring=False, with_land=True, growth_only=True):
    """Yield dict per fire-day: raw inputs on the crop + truth + meta."""
    fd_ = fd()
    from firesim import realdata as RD
    split = 'exam' if fd_.guard.is_exam(fid) else 'dev'
    if split == 'exam' and not for_exam_scoring:
        raise fd_.ExamLeakError(fid)
    r = fd_._row(fid)
    pr = fd_.progression(fid, res=RES, half='PM', for_exam_scoring=for_exam_scoring)
    t = pd.to_datetime(pr['t'])
    B = pr['burned']
    if len(B) < 2:
        return
    A = B.reshape(len(B), -1).sum(1) * CELL_KM2
    g = pr['grid']
    a_, _, c0_, _, e_, f0_ = g['transform']
    pix = fd_.firepix(fid, for_exam_scoring=for_exam_scoring)
    if len(pix):
        x, y = TR.transform(pix.Lon.values, pix.Lat.values)
        pix = pix.assign(col=(x - c0_) / a_ - 0.5, row=(y - f0_) / e_ - 0.5)
    fl = firelines(int(r.year))
    fl = fl[fl.Event_ID == fid]
    land = None
    if with_land:
        try:
            L = fd_.landscape(fid, for_exam_scoring=for_exam_scoring)
            land = FT.land_from_landscape(L['layers'])
            assert land['fg'].shape == B.shape[1:]
        except FileNotFoundError:
            land = None
    gam = RD.grid_convergence_deg(float(r.cx), float(r.cy))
    fdt = fire_days_table(split)
    fdt = fdt[fdt.fire_id == fid].set_index('k')
    first = None
    for k in range(len(B)):
        if B[k].any():
            first = k
            break
    for k in range(len(B) - 1):
        if (t[k + 1] - t[k]) != pd.Timedelta(days=1):
            continue
        now = B[k]
        if not now.any():
            continue
        nxt = B[k + 1] | now
        if growth_only and not (nxt & ~now).any():
            continue
        ys, xs = np.where(now)
        y0, y1 = max(ys.min() - PAD, 0), min(ys.max() + PAD + 1, now.shape[0])
        x0, x1 = max(xs.min() - PAD, 0), min(xs.max() + PAD + 1, now.shape[1])
        sl = (slice(y0, y1), slice(x0, x1))
        has_prev = k >= 1 and (t[k] - t[k - 1]) == pd.Timedelta(days=1)
        prev = B[k - 1][sl] if has_prev else None
        line = fl[fl.t == t[k]]
        flm = None
        if len(line):
            flm = fd_._grid.burn(list(shapely.from_wkb(line.wkb.values)), g, all_touched=True)[sl]
        def dets(tt):
            if not len(pix):
                return dict(row=[], col=[], ps=[], pt=[], frp=[])
            q = pix[pix.t == tt]
            return dict(row=(q.row.values - y0), col=(q.col.values - x0), ps=q.DS.values, pt=q.DT.values,
                        frp=q.FRP.values)
        dpm = dets(t[k])
        dam = dets(t[k] - pd.Timedelta(hours=12))
        if k in fdt.index:
            row = fdt.loc[k]
            from wxfeat import feature_vector
            wx = feature_vector(gridmet_row(row, 'd0'), gridmet_row(row, 'd1'), gam)
        else:
            wx = None
        area = A[k]
        d_prev = FT.req_m(A[k]) - FT.req_m(A[k - 1]) if has_prev else np.nan
        yield dict(fire_id=fid, k=k, t=str(t[k]), now=now[sl], nxt=nxt[sl], prev=prev, fl=flm, pix_pm=dpm, pix_am=dam,
                   land=None if land is None else {kk: v[sl] for kk, v in land.items()}, wx=wx,
                   area_km2=area, d_prev_m=d_prev, has_prev=has_prev, first=(k == first),
                   new_obs_km2=float((nxt & ~now).sum() * CELL_KM2), box=(y0, y1, x0, x1), split=split)
