"""Classical next-day spread baselines on FEDS-MTBS daily (PM-to-PM) progression, 90 m EPSG:5070 fire grids.

Usage:  python3 -I baselines_spread.py dev      # dev pass: per-fire-day rows + fit dev parameters
        python3 -I baselines_spread.py exam     # exam pass (uses dev parameters only); resumable per fire
Outputs: proc/baselines/spread_<split>.jsonl (one line per fire, all fire-days), spread_params.json (dev fits),
         spread_summary.json (tables).

Fire-day k: issue = PM perimeter k (S-NPP day overpass, ~13:30 local), verify = PM perimeter k+1 exactly 24 h later.
Distances: from the burned-area (or fire-line) cell edge to the cell centre (EDT - 45 m).
Scored cells: the fire grid within 10 km of the burned area at issue (firedata.metrics.score_window).
Methods (all are distance buffers; score s = 1 / (1 + (dist / d)^4), so s = 0.5 at the buffer distance d):
  persist      no growth (s = 0)
  grow_prev    d = yesterday's equivalent-radius growth  r_eq(k) - r_eq(k-1), r_eq = sqrt(A / pi); first day: clim
  grow_prev_k  d = k_dev * yesterday's growth (k_dev chosen on dev to maximise mean Dice on new area)
  clim         d = dev median next-day radial growth for the current size bin (dumb "guess the median" baseline)
  fl_prev      grow_prev, but distance measured from the FEDS active fire line at issue (fallback: perimeter)
"""
import os, sys, json, time
import numpy as np, pandas as pd, shapely
from scipy import ndimage
_CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # shared/fire (code)
_DATA = os.environ.get('ML_RATCHET_FIRE') or _CODE                                      # data root
sys.path.insert(0, _CODE)
import firedata as fd
import importlib
from firedata import metrics as M
G = importlib.import_module('firedata.grid')

P = f'{_DATA}/data/proc'
OUT = f'{P}/baselines'; os.makedirs(OUT, exist_ok=True)
RES = 90.0; CELL_KM2 = RES * RES / 1e6
WIN_KM = 10.0
SIZE_BINS_KM2 = [0, 5, 20, 100, 500, 1e9]
METHODS = ['persist', 'grow_prev', 'grow_prev_k', 'clim', 'fl_prev']


def req(a_km2): return np.sqrt(np.maximum(a_km2, 0) * 1e6 / np.pi)


def score(dist, d):
    d = max(float(d), 1.0)
    return 1.0 / (1.0 + (dist / d) ** 4)


_fl_cache = {}


def firelines(year):
    if year not in _fl_cache:
        _fl_cache.clear()
        _fl_cache[year] = pd.read_parquet(f'{P}/feds/firelines/{year}.parquet')
    return _fl_cache[year]


def fire_days(fid, exam):
    """Yield per-fire-day inputs (cropped arrays) for consecutive PM pairs."""
    r = fd._row(fid)
    pr = fd.progression(fid, res=RES, half='PM', for_exam_scoring=exam)
    t = pd.to_datetime(pr['t']); B = pr['burned']; A = B.reshape(len(B), -1).sum(1) * CELL_KM2
    if len(B) < 2: return
    fl = firelines(int(r.year)); fl = fl[fl.Event_ID == fid]
    g = pr['grid']
    pad = int(np.ceil(WIN_KM * 1000 / RES)) + 2
    for k in range(len(B) - 1):
        if (t[k + 1] - t[k]) != pd.Timedelta(days=1): continue
        now, nxt = B[k], B[k + 1]
        if not now.any(): continue
        ys, xs = np.where(now)
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, now.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, now.shape[1])
        sl = (slice(y0, y1), slice(x0, x1))
        prev_area = A[k - 1] if k >= 1 and (t[k] - t[k - 1]) == pd.Timedelta(days=1) else np.nan
        line = fl[fl.t == t[k]]
        flm = None
        if len(line):
            gm = shapely.from_wkb(line.wkb.values)
            flm = G.burn(list(gm), g, all_touched=True)[sl]
            if not flm.any(): flm = None
        yield dict(k=k, t=str(t[k]), now=now[sl], nxt=nxt[sl] | now[sl], A=A[k], A_prev=prev_area, A_next=A[k + 1],
                   fl=flm, outside=float((nxt & ~now).sum()))


def run_fire(fid, exam, params):
    rows = []
    for fdy in fire_days(fid, exam):
        now, nxt = fdy['now'], fdy['nxt']
        dist = ndimage.distance_transform_edt(~now, sampling=RES)
        win = M.score_window(now, RES, WIN_KM, dist=dist)
        dist = np.maximum(dist - RES / 2, 0.0)  # distance from the burned-area edge to the cell centre
        d_prev = req(fdy['A']) - req(fdy['A_prev']) if np.isfinite(fdy['A_prev']) else np.nan
        sb = int(np.digitize(fdy['A'], SIZE_BINS_KM2) - 1)
        d_clim = params['clim_d_m'][sb] if params else 0.0
        d_gp = max(d_prev, 0.0) if np.isfinite(d_prev) else d_clim
        base = dict(fire_id=fid, k=fdy['k'], t=fdy['t'], area_km2=fdy['A'], d_prev_m=d_prev,
                    d_next_m=req(fdy['A_next']) - req(fdy['A']), size_bin=sb, has_fireline=fdy['fl'] is not None,
                    new_outside_window=float((nxt & ~now & ~win).sum()))
        S = {'persist': np.zeros(now.shape), 'grow_prev': score(dist, d_gp) if d_gp > 0 else np.zeros(now.shape)}
        if params:
            dk = params['k_dev'] * d_gp
            S['grow_prev_k'] = score(dist, dk) if dk > 0 else np.zeros(now.shape)
            S['clim'] = score(dist, d_clim) if d_clim > 0 else np.zeros(now.shape)
        if fdy['fl'] is not None:
            dfl = np.maximum(ndimage.distance_transform_edt(~fdy['fl'], sampling=RES) - RES / 2, 0.0)
        else:
            dfl = dist
        S['fl_prev'] = score(dfl, d_gp) if d_gp > 0 else np.zeros(now.shape)
        for m, s in S.items():
            rows.append(dict(base, method=m, **M.nextday(now, nxt, s, CELL_KM2, window=win)))
        # dev: also store a coarse grid of Dice-new vs buffer factor for fitting k_dev
        if not params and np.isfinite(d_prev) and d_prev > 0:
            y = (nxt & ~now)[win]; dd = dist[win]
            rows.append(dict(base, method='_kgrid', kdice=[M.dice(dd <= kk * d_prev, y) if (y.any() or (dd <= kk * d_prev).any())
                                                       else np.nan for kk in KGRID]))
    return rows


KGRID = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]


def fit_params(dev_rows):
    d = pd.DataFrame([r for r in dev_rows if r['method'] == 'grow_prev'])
    clim = []
    for b in range(len(SIZE_BINS_KM2) - 1):
        x = d.d_next_m[d.size_bin == b].clip(lower=0)
        clim.append(float(np.median(x)) if len(x) >= 5 else float(np.median(d.d_next_m.clip(lower=0))))
    kg = pd.DataFrame([r for r in dev_rows if r['method'] == '_kgrid'])
    kd = np.nanmean(np.array(kg.kdice.tolist(), float), axis=0)
    k_dev = KGRID[int(np.nanargmax(kd))]
    return dict(clim_d_m=clim, size_bins_km2=SIZE_BINS_KM2, k_dev=k_dev, kgrid=KGRID, kgrid_mean_dice_new=kd.tolist(),
                n_dev_fire_days=int(len(d)), fitted_on='dev fire-days only')


def main(split):
    exam = split == 'exam'
    pfile = f'{OUT}/spread_params.json'
    params = json.load(open(pfile)) if os.path.exists(pfile) else None
    if exam and not params: raise SystemExit('run dev first')
    ids = list(fd.fires(split).Event_ID)
    if exam:  # fixed order (shuffled by seed) so a partial run is an unbiased subset
        ids = list(np.random.default_rng(20261118).permutation(sorted(ids)))
    out = f'{OUT}/spread_{split}.jsonl'
    if split == 'dev' and params is None and os.path.exists(out): os.remove(out)
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)['fire_id'] for l in open(out)}
    t0 = time.time()
    with open(out, 'a') as fo:
        for i, fid in enumerate(ids):
            if fid in done: continue
            try:
                rows = run_fire(fid, exam, params)
            except Exception as e:
                print('FAIL', fid, repr(e), flush=True); continue
            fo.write(json.dumps(dict(fire_id=fid, rows=rows), default=float) + '\n'); fo.flush()
            if i % 50 == 0: print(i, len(ids), fid, len(rows), f'{time.time() - t0:.0f}s', flush=True)
    if split == 'dev' and params is None:
        allrows = [r for l in open(out) for r in json.loads(l)['rows']]
        params = fit_params(allrows)
        json.dump(params, open(pfile, 'w'), indent=1)
        print('params', params)
        os.remove(out)
        main('dev')  # re-run dev with fitted params so dev has every method
        return
    print('DONE', split, flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
