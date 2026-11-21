"""Build FireDetective records for REAL fires (dev now; exam only from exam.py with for_exam=True).

A record = model inputs (seq + static features, wind visible and hidden) + truths with ground truth:
  hs      MTBS high-severity fraction of the window's burned area (LANDFIRE Annual Disturbance codes 11-14 MTBS,
          111-114 MTBS gap-filled; LFPS Landfire_Disturbance/LF20xx_DistYY_CONUS)
  wind    mean gridMET daily 10 m wind speed (vs, m/s) over the window's local days
  lfmc    mean Globe-LFMC (%) within 25 km and +-10 days of the window's middle day
plus a sample of burnable cells (for the classical Van Wagner / Rothermel diagnostics).
"""
from common import *
import time, importlib, functools
import requests, rasterio
import firedata as fd
import firesim as fs
import feats

G = importlib.import_module('firedata.grid')
RES = feats.RES
LFPS = 'https://lfps.usgs.gov/arcgis/rest/services'
SESSION = requests.Session()
MTBS_CODES = np.array([11, 12, 13, 14, 111, 112, 113, 114])
HS_CODES = np.array([14, 114])
MODHS_CODES = np.array([13, 14, 113, 114])


def dist_service(year):
    yy = year % 100
    if year <= 2010: return None
    if year <= 2012: rel = 2012
    elif year <= 2014: rel = 2014
    elif year <= 2016: rel = 2016
    elif year <= 2020: rel = 2020
    elif year <= 2022: rel = 2022
    elif year == 2023: rel = 2023
    else: return None
    return f'Landfire_Disturbance/LF{rel}_Dist{yy:02d}_CONUS'


def lf_service(layer, fire_year):
    if layer in ('Elev', 'SlpD', 'Asp'):
        return f'Landfire_Topo/LF2020_{layer}_CONUS'
    ver = 2016 if fire_year <= 2022 else (2022 if fire_year == 2023 else 2023)
    return f'Landfire_LF{ver}/LF{ver}_{layer}_CONUS'


def lfps_export(service, g, retries=6):
    x0, y0, x1, y1 = g['bounds']
    ny, nx = g['shape']
    url = f'{LFPS}/{service}/ImageServer/exportImage'
    p = dict(bbox=f'{x0},{y0},{x1},{y1}', bboxSR=5070, imageSR=5070, size=f'{nx},{ny}', format='tiff',
             interpolation='RSP_NearestNeighbor', f='image')
    err = ''
    for i in range(retries):
        try:
            r = SESSION.get(url, params=p, timeout=(30, 300))
            if r.status_code == 200 and r.content[:2] in (b'II', b'MM'):
                with rasterio.MemoryFile(r.content) as mf, mf.open() as s:
                    a = s.read(1)
                assert a.shape == (ny, nx), (a.shape, ny, nx)
                return a
            err = f'{r.status_code} {r.text[:200]}'
        except Exception as e:
            err = repr(e)
        time.sleep(5 * (i + 1))
    raise RuntimeError(f'LFPS export failed {service}: {err}')


def landscape180(fid, year, g180, for_exam=False, use_package=True):
    """LANDFIRE layers on the 180 m grid. Dev/replay packages (90 m) are coarsened 2x; otherwise fetched from LFPS
    at 180 m (nearest neighbour)."""
    fn = f'{fd.P}/landscapes/{fid}.npz'
    if use_package and os.path.exists(fn):
        L = fd.landscape(fid, for_exam_scoring=for_exam)['layers']
        L = dict(L)
        L['fuel'] = np.where(L['fuel'] <= 0, 99, L['fuel'])
        C = feats.coarsen_layers(L, 2)
        ny, nx = g180['shape']
        out = {}
        for k, v in C.items():
            a = np.zeros((ny, nx), v.dtype) + (99 if k == 'fuel' else 0)
            a[:min(ny, v.shape[0]), :min(nx, v.shape[1])] = v[:ny, :nx]
            out[k] = a
        return out, 'package'
    cache = f'{REAL}/lf/{fid}.npz'
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}, 'lfps_cache'
    out = {}
    for lay, key, scale in (('FBFM40', 'fuel', 1), ('CC', 'cc', 1), ('CH', 'ch', 10), ('CBH', 'cbh', 10),
                            ('CBD', 'cbd', 100), ('SlpD', 'slope', 1), ('Elev', 'elev', 1)):
        a = lfps_export(lf_service(lay, year), g180).astype(np.float32)
        if key == 'fuel':
            out[key] = np.where(a <= 0, 99, a).astype(np.int64)
        else:
            out[key] = np.where(a < 0, 0, a / scale).astype(np.float32)
    os.makedirs(f'{REAL}/lf', exist_ok=True)
    np.savez_compressed(cache, **out)
    return out, 'lfps'


def severity(fid, year, g90, region90, for_exam=False):
    cache = f'{REAL}/dist/{fid}.npz'
    if os.path.exists(cache):
        a = np.load(cache)['dist']
    else:
        svc = dist_service(year)
        if svc is None:
            return None
        a = lfps_export(svc, g90).astype(np.int16)
        os.makedirs(f'{REAL}/dist', exist_ok=True)
        np.savez_compressed(cache, dist=a)
    v = a[region90]
    mapped = np.isin(v, MTBS_CODES)
    n = int(mapped.sum())
    if n < 20:
        return dict(hs=np.nan, mh=np.nan, cover=n / max(len(v), 1), n=n)
    return dict(hs=float(np.isin(v[mapped], HS_CODES).mean()), mh=float(np.isin(v[mapped], MODHS_CODES).mean()),
                cover=n / max(len(v), 1), n=n)


@functools.lru_cache(None)
def _lfmc():
    l = pd.read_parquet(f'{fd.P}/lfmc/lfmc_us.parquet')
    l = l[l.date >= '2016-06-01']
    x, y = to5070(l['Longitude (WGS84, EPSG:4326)'].values.astype(float), l['Latitude (WGS84, EPSG:4326)'].values.astype(float))
    from scipy.spatial import cKDTree
    return l.assign(x=x, y=y).reset_index(drop=True), cKDTree(np.c_[x, y])


def lfmc_truth(cx, cy, mid_day, R=25e3, D=10):
    l, T = _lfmc()
    idx = T.query_ball_point([cx, cy], R)
    if not idx:
        return None
    s = l.iloc[idx]
    dd = np.abs((s.date.values - np.datetime64(mid_day)) / np.timedelta64(1, 'D'))
    s = s[dd <= D]
    if not len(s):
        return None
    v = s['LFMC value (%)'].astype(float)
    ft = s['Species functional type']
    return dict(lfmc=float(v.median()), n=int(len(s)), n_sites=int(s['Site name'].nunique()),
                lfmc_shrub=float(v[ft == 'Shrub'].median()) if (ft == 'Shrub').any() else np.nan,
                lfmc_tree=float(v[ft == 'Tree'].median()) if (ft == 'Tree').any() else np.nan)


def wx_rows(daily, days):
    out = []
    for d in days:
        if d in daily.index:
            r = daily.loc[d]
            out.append({k: float(r[k]) for k in ('tmmx', 'rmin', 'rmax', 'vs', 'th', 'fm100', 'fm1000', 'erc', 'bi',
                                                  'pr', 'vpd')})
        else:
            out.append(None)
    # fill gaps with neighbours
    for i in range(len(out)):
        if out[i] is None:
            j = next((j for j in list(range(i - 1, -1, -1)) + list(range(i + 1, len(out))) if out[j] is not None), None)
            out[i] = out[j] if j is not None else dict(tmmx=25, rmin=20, rmax=60, vs=3, th=225, fm100=10, fm1000=12,
                                                         erc=50, bi=40, pr=0, vpd=2)
    return out


def step_day(t_label):
    """Local day whose weather drives the growth reported at a FEDS step: PM (12:00) -> same day; AM (00:00) ->
    previous day (growth since the previous afternoon)."""
    t = pd.Timestamp(t_label)
    return t.normalize() if t.hour == 12 else t.normalize() - pd.Timedelta(days=1)


def run_distances(masks, res=RES, q=95):
    """Per step: the q-th percentile distance (km) of newly burned cells from the previous perimeter (head run)."""
    from scipy import ndimage as ndi
    out = [0.0]
    for k in range(1, len(masks)):
        new = masks[k] & ~masks[k - 1]
        if new.sum() == 0 or masks[k - 1].sum() == 0:
            out.append(0.0); continue
        d = ndi.distance_transform_edt(~masks[k - 1])
        out.append(float(np.percentile(d[new], q)) * res / 1000.0)
    return out


def build_record(fid, for_exam=False, use_package=True, n_cells=400, rng=0):
    r = fd._row(fid)
    year = int(r.year)
    pr = fd.perimeters(fid, for_exam_scoring=for_exam)
    fp = fd.firepix(fid, for_exam_scoring=for_exam)
    g90 = fd.grid(fid)
    g180 = G.coarsen(g90, 2)
    x0, y0, x1, y1 = g180['bounds']
    # window: from the first step with area > 0, up to T_STEPS steps
    k0 = int(np.argmax(pr.farea_km2.values > 0)) if (pr.farea_km2.values > 0).any() else None
    if k0 is None:
        return None
    pw = pr.iloc[k0:k0 + feats.T_STEPS].reset_index(drop=True)
    masks = np.stack([G.burn([geom], g180) for geom in pw.geometry])
    masks = np.maximum.accumulate(masks, axis=0)
    if masks[-1].sum() == 0:
        return None
    gx, gy = to5070(fp.Lon.values.astype(float), fp.Lat.values.astype(float))
    dets, day = [], []
    for t, half in zip(pw.t, pw.half):
        d = fp[fp.t == t]
        xs, ys = gx[d.index], gy[d.index]
        dets.append(dict(col=(xs - x0) / RES - 0.5, row=(y1 - ys) / RES - 0.5, frp=d.FRP.values.astype(float)))
        day.append(half == 'PM')
    W = fd.weather(fid, for_exam_scoring=for_exam)
    daily = W['daily'].copy()
    daily.index = pd.DatetimeIndex(daily.index).normalize()
    days = [step_day(t) for t in pw.t]
    wx = wx_rows(daily, days)
    gamma = fs.realdata.grid_convergence_deg(float(r.cx), float(r.cy))
    seq = feats.seq_features(masks, dets, np.array(day), wx, gamma_deg=gamma)
    run_km = run_distances(masks)
    seq_h = feats.seq_features(masks, dets, np.array(day), wx, wind_hidden=True, gamma_deg=gamma)
    L, lsrc = landscape180(fid, year, g180, for_exam, use_package)
    am = feats.area_mask_from(masks)
    doy = pd.Timestamp(pw.t.iloc[0]).dayofyear
    st = feats.static_features(L, am, doy, len(pw))
    # truths
    last = masks[-1]
    fin = G.burn([fd.final_perimeter(fid, for_exam_scoring=for_exam)], g90)
    up = np.zeros(g90['shape'], bool)
    ny2, nx2 = last.shape
    up[:2 * ny2, :2 * nx2] = np.repeat(np.repeat(last, 2, 0), 2, 1)
    region90 = up & fin
    if region90.sum() < 20:
        region90 = fin
    try:
        sev = severity(fid, year, g90, region90, for_exam)
    except Exception as e:
        sev = dict(hs=np.nan, mh=np.nan, cover=0.0, n=0, err=repr(e))
    uniq = sorted(set(days))
    vs = [daily.loc[d, 'vs'] for d in uniq if d in daily.index]
    wind = float(np.mean(vs)) if vs else np.nan
    mid = uniq[len(uniq) // 2]
    lf = lfmc_truth(float(r.cx), float(r.cy), mid)
    # cell sample for the classical diagnostics
    rs = np.random.default_rng(rng)
    rr, cc = np.nonzero(last & (L['fuel'] < 91) | last & (L['fuel'] > 99))
    sel = rs.choice(len(rr), min(n_cells, len(rr)), replace=False) if len(rr) else np.zeros(0, int)
    cells = {k: np.asarray(L[k])[rr[sel], cc[sel]] for k in ('fuel', 'cc', 'ch', 'cbh', 'cbd', 'slope')}
    fgm = feats.fuel_group_masks(L['fuel'])
    sel_b = last
    forest_frac = float((fgm['TU'] | fgm['TL'] | fgm['SB'])[sel_b].mean()) if sel_b.any() else 0.0
    lon, lat = fs.realdata.lonlat_of(float(r.cx), float(r.cy))
    return dict(fid=fid, split=r.split, period=r.exam_period or 'dev', region=r.region, year=year,
                fuel_group=r.fuel_group, size_class=r.size_class, name=r.Incid_Name, lon=lon, lat=lat,
                n_steps=len(pw), t0=str(pw.t.iloc[0]), days=[str(d.date()) for d in days], landscape_source=lsrc,
                seq=seq, seq_hidden=seq_h, static=st, area_km2=[float(m.sum()) * (RES / 1000) ** 2 for m in masks],
                hs=sev.get('hs', np.nan) if sev else np.nan, mh=sev.get('mh', np.nan) if sev else np.nan,
                sev_cover=sev.get('cover', 0) if sev else 0, wind=wind, wind_days=vs,
                lfmc=lf['lfmc'] if lf else np.nan, lfmc_info=lf, forest_frac=forest_frac,
                cells=cells, wx=wx, gamma=gamma, postfire_landfire=bool(r.lf_postfire), run_km=run_km,
                day_flags=day, slope_mean=float(st[11] * 30))


def save_records(recs, fn):
    import pickle
    with open(fn + '.tmp', 'wb') as f:
        pickle.dump(recs, f)
    os.replace(fn + '.tmp', fn)


def load_records(fn):
    import pickle
    with open(fn, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    # DEV records only. Exam records are built inside exam.py.
    which = sys.argv[1] if len(sys.argv) > 1 else 'package'
    out = f'{REAL}/dev_records_{which}.pkl'
    recs = []
    for fid in fd.fires('dev').Event_ID:
        t = time.time()
        rec = build_record(fid, use_package=(which == 'package'))
        recs.append(rec)
        print(fid, round(time.time() - t, 1), None if rec is None else
              (rec['n_steps'], round(rec['hs'], 3) if rec['hs'] == rec['hs'] else None, round(rec['wind'], 2),
               rec['lfmc'], round(rec['forest_frac'], 2), rec['sev_cover']), flush=True)
    save_records(recs, out)
