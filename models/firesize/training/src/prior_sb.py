"""Slash-blowdown training ignitions (27 Nov 2026, audit follow-up; seeded, guard-checked).

Why: none of the 8,500 simulated training ignitions landed on slash-blowdown fuel (LANDFIRE FBFM40 201-204) at the
250 m feature pixel, so pt_SB had sd 1e-6 and the 91 exam ignitions on SB fuel entered the network at z = 1e6.

What: N_WIN extra windows, each centred on a seeded random SB cell of the 250 m national fuel grid (uniform over the
7,184 SB cells). In each window up to M_IGN ignition points are placed on SB cells within +-2 km of the centre
(seeded, with replacement), snapped to the 90 m grid, and kept only if the 250 m feature pixel of the snapped point
is SB. Each takes the attributes of a random dev ignition within 150 km (year, day of year +- 7 d, discovery hour,
cause, agency class, region), exactly as prior.py does: dev attributes only, never a size. Every window passes
fd.guard.check_landscape and every ignition's weather period fd.guard.check_weather.

N_HOLD of the windows (seeded hash) are marked sbhold = 1: never trained on, scored separately as a small simulated
SB check. All SB windows have frozen = 0, so the ratchet's frozen set is unchanged.

Rows are appended to data/ignitions.parquet and data/windows.parquet with kind 'trainsb' (ign and win ids continue
after the existing ones; the existing rows are not touched; copies of both files are kept in data/pre_sb/ first).
usage: prior_sb.py [n_win]"""
import os, sys, shutil
from common import *
import firedata as fd
import features as F
from prior import wx_ok, land_ok, HALF

N_WIN = int(sys.argv[1]) if len(sys.argv) > 1 else 60
N_HOLD_FRAC = 0.2
M_IGN = 8


def main():
    rs = np.random.default_rng([SEED, 201])
    I0 = pd.read_parquet(f'{DATA}/ignitions.parquet')
    W0 = pd.read_parquet(f'{DATA}/windows.parquet')
    if (I0.kind == 'trainsb').any():
        raise SystemExit('trainsb rows already present')
    os.makedirs(f'{DATA}/pre_sb', exist_ok=True)
    for f in ('ignitions.parquet', 'windows.parquet'):
        if not os.path.exists(f'{DATA}/pre_sb/{f}'):
            shutil.copy2(f'{DATA}/{f}', f'{DATA}/pre_sb/{f}')
    a = np.load(f'{DATA}/nat/fuel.npz')['a']
    sb = (a >= 201) & (a <= 204)
    r, c = np.nonzero(sb)
    cx_all = F.X0 + (c + 0.5) * F.R
    cy_all = F.Y1 - (r + 0.5) * F.R
    print('SB cells', len(r), flush=True)
    dev = fod_records('dev')
    dev['date'] = pd.to_datetime(dev.DISCOVERY_DATE).dt.normalize()
    from scipy.spatial import cKDTree
    tree = cKDTree(dev[['x', 'y']].values)
    sbtree = cKDTree(np.c_[cx_all, cy_all])
    hours = dev.hour.dropna().values
    ext = (-2362425, 258915, 2263815, 3177435)
    wid = int(max(W0.win.max(), I0.win.max())) + 1
    ign = int(I0.ign.max()) + 1
    W, I = [], []
    tries = 0
    nwin = 0
    skip = dict(edge=0, land=0, near=0, none=0)
    while nwin < N_WIN and tries < N_WIN * 20:
        tries += 1
        k = rs.integers(len(r))
        cx, cy = snap(cx_all[k]), snap(cy_all[k])
        if not (ext[0] + HALF < cx < ext[2] - HALF and ext[1] + HALF < cy < ext[3] - HALF):
            skip['edge'] += 1
            continue
        if not land_ok(cx, cy):
            skip['land'] += 1
            continue
        near = tree.query_ball_point([cx, cy], 150000.0)
        if len(near) < 3:
            skip['near'] += 1
            continue
        cand = [j for j in sbtree.query_ball_point([cx, cy], 2000.0 * np.sqrt(2))
                if abs(cx_all[j] - cx) <= 2000 and abs(cy_all[j] - cy) <= 2000]
        rows = []
        for m in range(M_IGN):
            for _ in range(6):
                j = cand[rs.integers(len(cand))]
                ix = snap(cx_all[j] + rs.uniform(-125, 125) - 45)       # features read the 250 m pixel at x + 45
                iy = snap(cy_all[j] + rs.uniform(-125, 125) - 45)
                pr = int(np.floor((F.Y1 - (iy + 45)) / F.R)); pc = int(np.floor((ix + 45 - F.X0) / F.R))
                if not sb[pr, pc]:                                          # the same pixel rule as land_features
                    continue
                q = dev.iloc[near[rs.integers(len(near))]]
                date = q.date + pd.Timedelta(days=int(rs.integers(-7, 8)))
                if date.year != q.date.year:
                    continue
                if not wx_ok(ix, iy, date):
                    continue
                hr = q.hour
                rows.append(dict(kind='trainsb', win=wid, x=ix, y=iy, date=date, year=date.year, doy=date.dayofyear,
                                 hour=hr, hour_sim=hr if np.isfinite(hr) else float(rs.choice(hours)),
                                 cause=q.cause, fed=int(q.fed), region=q.region, owner=q.OWNER_DESCR,
                                 src_fod=int(q.FOD_ID), weight=1.0, FOD_ID=-1, size_ha=np.nan, escape=-1))
                break
        if not rows:
            skip['none'] += 1
            continue
        for rw in rows:
            rw['ign'] = ign
            ign += 1
        I += rows
        W.append(dict(win=wid, kind='trainsb', cx=cx, cy=cy, frozen=0, blank=0,
                      sbhold=int(h01(wid, 'sbhold') < N_HOLD_FRAC)))
        wid += 1
        nwin += 1
    W = pd.DataFrame(W)
    I = pd.DataFrame(I)
    W0 = W0.assign(sbhold=0)
    Wn = pd.concat([W0, W], ignore_index=True)
    In = pd.concat([I0, I[I0.columns]], ignore_index=True)
    In.to_parquet(f'{DATA}/ignitions.parquet')
    Wn.to_parquet(f'{DATA}/windows.parquet')
    print('trainsb windows', len(W), 'held out', int(W.sbhold.sum()), 'ignitions', len(I),
          'held-out ignitions', int(I.win.isin(W.win[W.sbhold == 1]).sum()), 'tries', tries, skip,
          'regions', I.region.value_counts().to_dict(), flush=True)


if __name__ == '__main__':
    main()
