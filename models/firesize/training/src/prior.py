"""Ignition prior and the sample tables (all seeded, guard-checked).

Training ignitions ("train"): the ignition prior is calibrated on DEV ignitions (FPA-FOD dev, attributes only, never
their sizes): a window centre = a random dev ignition location + N(0, 20 km) jitter; within the window, M ignition
points within +-2 km of the centre; each takes the attributes (year, day of year +- 7 d, discovery hour, cause,
agency class, region) of a random dev ignition within 150 km. Every window passes fd.guard.check_landscape and every
ignition's weather period passes fd.guard.check_weather (else redrawn / skipped).

Dev twins ("twin"): dev ignitions of the 'calib' half (all escapes + a seeded draw of non-escapes, with weights), in a
window centred on their own location, with their own date. Used to calibrate the suppression model. Skipped when the
guard refuses the window or the weather.
-> data/windows.parquet (one row per window), data/ignitions.parquet (one row per ignition)
"""
import sys
from common import *
import firedata as fd
import importlib; fgrid = importlib.import_module("firedata.grid")

N_WIN = int(sys.argv[1]) if len(sys.argv) > 1 else 2400
M_IGN = 8
JIT_M = 20000.0
HALF = WIN * RES / 2
WX_BEFORE, WX_AFTER = 24, 5          # days of weather needed before / after discovery


def wx_ok(x, y, date):
    t0 = pd.Timestamp(date) - pd.Timedelta(days=WX_BEFORE)
    t1 = pd.Timestamp(date) + pd.Timedelta(days=WX_AFTER)
    try:
        fd.guard.check_weather((x - 1000, y - 1000, x + 1000, y + 1000), t0, t1)
        return True
    except fd.guard.ExamLeakError:
        return False


def land_ok(cx, cy):
    try:
        fd.guard.check_landscape((cx - HALF, cy - HALF, cx + HALF, cy + HALF))
        return True
    except fd.guard.ExamLeakError:
        return False


def main():
    rs = np.random.default_rng(SEED)
    dev = fod_records('dev')
    dev['half'] = [dev_half(i) for i in dev.FOD_ID]
    dev['date'] = pd.to_datetime(dev.DISCOVERY_DATE).dt.normalize()
    from scipy.spatial import cKDTree
    tree = cKDTree(dev[['x', 'y']].values)
    hours = dev.hour.dropna().values
    W, I = [], []
    ext = (-2362425, 258915, 2263815, 3177435)
    wid = 0
    tries = 0
    while wid < N_WIN and tries < N_WIN * 6:
        tries += 1
        r = dev.iloc[rs.integers(len(dev))]
        cx, cy = snap(r.x + rs.normal(0, JIT_M)), snap(r.y + rs.normal(0, JIT_M))
        if not (ext[0] + HALF < cx < ext[2] - HALF and ext[1] + HALF < cy < ext[3] - HALF):
            continue
        if not land_ok(cx, cy):
            continue
        near = tree.query_ball_point([cx, cy], 150000.0)
        if len(near) < 3:
            continue
        n_ok = 0
        for k in range(M_IGN):
            for _ in range(6):
                q = dev.iloc[near[rs.integers(len(near))]]
                date = q.date + pd.Timedelta(days=int(rs.integers(-7, 8)))
                if date.year != q.date.year:
                    continue
                ix, iy = snap(cx + rs.uniform(-2000, 2000)), snap(cy + rs.uniform(-2000, 2000))
                if not wx_ok(ix, iy, date):
                    continue
                hr = q.hour
                I.append(dict(kind='train', win=wid, x=ix, y=iy, date=date, year=date.year, doy=date.dayofyear,
                              hour=hr, hour_sim=hr if np.isfinite(hr) else float(rs.choice(hours)),
                              cause=q.cause, fed=int(q.fed), region=q.region, owner=q.OWNER_DESCR, src_fod=int(q.FOD_ID),
                              weight=1.0, FOD_ID=-1, size_ha=np.nan, escape=-1))
                n_ok += 1
                break
        if n_ok:
            W.append(dict(win=wid, kind='train', cx=cx, cy=cy, frozen=int(h01(wid, 'frozen') < 0.12), blank=0))
            wid += 1
        print('win', wid, tries, flush=True) if wid % 100 == 0 and n_ok else None
    # dev twins (calib half)
    cal = dev[dev.half == 'calib']
    esc = cal[cal.escape == 1]
    non = cal[cal.escape == 0]
    n_non = 1100
    pick = non.sample(n=min(n_non, len(non)), random_state=SEED)
    w_non = len(non) / len(pick)
    nskip = dict(land=0, wx=0)
    for part, wgt in ((esc, 1.0), (pick, w_non)):
        for _, q in part.iterrows():
            cx, cy = snap(q.x), snap(q.y)
            blank = 0
            if not land_ok(cx, cy):
                g = fgrid.grid_from_window(cx - 100 * RES, cy - 100 * RES, cx + 101 * RES, cy + 101 * RES)
                m = fd.guard.exam_mask(g)
                if m[95:106, 95:106].any() or m.mean() > 0.5:      # ignition within ~500 m of an exam footprint
                    nskip['land'] += 1
                    continue
                blank = 1
            if not wx_ok(cx, cy, q.date):
                nskip['wx'] += 1
                continue
            hr = q.hour
            W.append(dict(win=wid, kind='twin', cx=cx, cy=cy, frozen=0, blank=blank))
            I.append(dict(kind='twin', win=wid, x=cx, y=cy, date=q.date, year=q.date.year, doy=q.date.dayofyear,
                          hour=hr, hour_sim=hr if np.isfinite(hr) else float(rs.choice(hours)), cause=q.cause,
                          fed=int(q.fed), region=q.region, owner=q.OWNER_DESCR, src_fod=int(q.FOD_ID), weight=wgt,
                          FOD_ID=int(q.FOD_ID), size_ha=q.size_ha, escape=int(q.escape)))
            wid += 1
    W = pd.DataFrame(W)
    I = pd.DataFrame(I)
    I['ign'] = np.arange(len(I))
    W.to_parquet(f'{DATA}/windows.parquet')
    I.to_parquet(f'{DATA}/ignitions.parquet')
    print('windows', W.kind.value_counts().to_dict(), 'ignitions', I.kind.value_counts().to_dict(), 'twin skips', nskip,
          'tries', tries, 'non-escape weight', w_non, flush=True)


if __name__ == '__main__':
    main()
