"""simulate(): spread a fire over a landscape under hourly weather (the shared contract)."""
import time as _time

import numpy as np
import pandas as pd

from . import rothermel as R
from .fuels import FM, _build_arrays, lookup_rows
from .kernel import run_mtt
from .landscape import as_landscape, rasterize_lines
from .moisture import dead_moisture
from .weather import check_weather, U10_TO_U20
from . import stencil as _st

DEFAULTS = dict(
    physics='p2',               # 'p2' (22 Nov 2026: U1, U9, U11 fixed in the solver) or 'p1' (the solver every v1.0 model
                                # was built and scored with; reproduces it exactly). See shared/fire/STATUS.md.
    wind_limit='none',          # 'none' (Andrews et al. 2013 recommendation), 'rothermel' (0.9 IR), 'andrews2013'
    lw='behaveplus',            # 'behaveplus' (1 + 0.25 UE mi/h, Andrews 2018) or 'anderson1983' (U in mi/h)
    lw_max=8.0,
    crown_spread='rothermel1991',   # or 'cruz2005'
    crown_ros_mult=1.0,
    fmc=100.0,                  # foliar moisture %
    mherb=60.0, mwoody=90.0,    # live moisture % when the weather has no mherb/mwoody columns (S&B scenario L2)
    curing=None,                # None: from live herb moisture (Scott & Burgan); else fraction 0-1
    stencil=16,
    simplex=None,               # two-point updates (remove most grid-direction bias); None = on except method 'ca'
    waf=None,                   # None: per cell (Andrews 2012 sheltered/unsheltered); or a number / raster
    ros_adjust=None,            # {fuel group or fuel number: factor}; groups from fuel_group()
    ros_factor=None,            # raster multiplier (e.g. retardant, calibration maps)
    moisture_spinup_h=72,
    nelson=None,                # Nelson model parameter overrides
    # spotting
    ember_rate=2.0,             # expected firebrands per hectare of fully crowning area (scaled by CFB)
    spot_med_frac=0.35, spot_sigma=0.5, spot_dir_sd=0.2, spot_p_est=0.5, spot_delay_h=0.1,
    spot_surface_fli=0.0,       # > 0: surface cells above this FLI (kW/m) also loft firebrands (at 10% of the rate)
    shade=0.5, max_spots=200000, spot_seed=None,
)

GROUPS = {'GR': range(101, 110), 'GS': range(121, 125), 'SH': range(141, 150), 'TU': range(161, 166),
          'TL': range(181, 190), 'SB': range(201, 205), 'A_grass': range(1, 4), 'A_shrub': range(4, 8),
          'A_timber': range(8, 11), 'A_slash': range(11, 14)}


def fuel_group(num):
    for g, r in GROUPS.items():
        if int(num) in r:
            return g
    return 'NB'


def _rowcol_from_xy(ls, x, y):
    t = ls.meta.get('transform')
    if t is None:
        raise ValueError('landscape has no transform: give ignitions as rowcol')
    return ls.rowcol(x, y)


def parse_ignition(ign, ls, start):
    """Returns (cells (flat indices), times (h since start), mask_time or None). mask_time: for cells inside a mask or
    polygon ignition, the hour (since start) it ignites; +inf elsewhere."""
    ny, nx = ls.shape
    items = ign if isinstance(ign, list) else [ign]
    cells, times = [], []
    mask_t = np.full((ny, nx), np.inf)
    for it in items:
        if isinstance(it, np.ndarray) and it.dtype == bool:
            it = {'mask': it}
        if isinstance(it, tuple) and len(it) == 2:
            it = {'rowcol': [it]}
        t = it.get('time', None)
        if t is None:
            th = 0.0
        elif isinstance(t, (int, float, np.floating, np.integer)):
            th = float(t)
        else:
            tt = pd.Timestamp(t)
            if tt.tz is None:
                tt = tt.tz_localize('UTC')
            th = (tt - start) / pd.Timedelta(hours=1)
        rcs = []
        if 'rowcol' in it:
            rcs = [tuple(map(int, rc)) for rc in it['rowcol']]
        if 'points' in it:
            p = np.asarray(it['points'], float).reshape(-1, 2)
            r, c = _rowcol_from_xy(ls, p[:, 0], p[:, 1])
            rcs += list(zip(r.tolist(), c.tolist()))
        if 'polygon' in it:
            from matplotlib.path import Path
            p = np.asarray(it['polygon'], float)
            t6 = ls.meta['transform']
            if it.get('pixel', False):
                pc = p
            else:
                pc = np.column_stack([(p[:, 0] - t6[0]) / t6[1], (p[:, 1] - t6[3]) / t6[5]])
            yy, xx = np.mgrid[0:ny, 0:nx]
            inside = Path(pc).contains_points(np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5])).reshape(ny, nx)
            it = dict(it, mask=inside | it.get('mask', np.zeros((ny, nx), bool)))
        if 'mask' in it:
            m = np.asarray(it['mask'], bool)
            mask_t[m] = np.minimum(mask_t[m], th)
            from scipy.ndimage import binary_erosion
            edge = m & ~binary_erosion(m, structure=np.ones((3, 3)), border_value=0)
            r, c = np.nonzero(edge)
            rcs += list(zip(r.tolist(), c.tolist()))
        for r, c in rcs:
            if 0 <= r < ny and 0 <= c < nx:
                cells.append(r * nx + c)
                times.append(th)
    return np.array(cells, np.int64), np.array(times, float), (mask_t if np.isfinite(mask_t).any() else None)


def _zone_weather(weather, ls):
    if isinstance(weather, dict) and 'series' in weather:
        series = [check_weather(w) for w in weather['series']]
        zone = np.asarray(weather['zones'], np.int64)
        if zone.shape != ls.shape:
            raise ValueError('weather zones raster must match the landscape')
        for s in series[1:]:
            if not s.index.equals(series[0].index):
                raise ValueError('all zone weather series must share one hourly index')
        return series, zone
    w = check_weather(weather)
    if hasattr(weather, 'attrs'):
        w.attrs.update(weather.attrs)
    return [w], np.zeros(ls.shape, np.int64)


def prepare(landscape, weather, start, end, params, moisture):
    ls = as_landscape(landscape)
    P = dict(DEFAULTS)
    P.update(params or {})
    start = pd.Timestamp(start)
    if start.tz is None:
        start = start.tz_localize('UTC')
    end = pd.Timestamp(end)
    if end.tz is None:
        end = end.tz_localize('UTC')
    fm = FM if not P.get('custom_fuels') else _build_arrays(P['custom_fuels'])
    series, zone = _zone_weather(weather, ls)
    idx = series[0].index
    if idx[0] > start:
        raise ValueError(f'weather starts {idx[0]} after the simulation start {start}')
    if idx[-1] < end - pd.Timedelta(hours=1):
        raise ValueError(f'weather ends {idx[-1]} before the simulation end {end}')
    t0 = start.floor('h')
    steps = pd.date_range(t0, end.ceil('h'), freq='h')
    nk = len(steps)
    nz = len(series)
    m_all = []
    for w in series:
        if moisture == 'given':
            need = ['m1h', 'm10h', 'm100h']
            if any(c not in w for c in need):
                raise ValueError("moisture='given' needs m1h, m10h, m100h columns in the weather")
            md = w[need]
        else:
            sp = pd.Timedelta(hours=int(P['moisture_spinup_h']))
            ww = w[(w.index >= max(idx[0], t0 - sp)) & (w.index <= steps[-1])]
            md = dead_moisture(ww, method=moisture, params=P.get('nelson'))
        md = md.reindex(steps, method='nearest')
        ws = w.reindex(steps, method='nearest')
        mh = ws['mherb'].to_numpy(float) if 'mherb' in ws else np.full(nk, P['mherb'])
        mw = ws['mwoody'].to_numpy(float) if 'mwoody' in ws else np.full(nk, P['mwoody'])
        m_all.append(pd.DataFrame(dict(m1h=md['m1h'].to_numpy(), m10h=md['m10h'].to_numpy(),
                                       m100h=md['m100h'].to_numpy(), mherb=mh, mwoody=mw,
                                       T=ws['T'].to_numpy(), RH=ws['RH'].to_numpy(),
                                       wind_speed=ws['wind_speed'].to_numpy(), wind_dir=ws['wind_dir'].to_numpy(),
                                       precip=ws['precip'].to_numpy()), index=steps))
    nrow = len(fm['nums'])
    bundles = np.zeros((nk, nz, nrow, R.NBUNDLE))
    bundle10 = np.zeros((nk, nz, R.NBUNDLE))
    u20 = np.zeros((nk, nz))
    wto = np.zeros((nk, nz))
    u10k = np.zeros((nk, nz))
    m1 = np.zeros((nk, nz))
    tair = np.zeros((nk, nz))
    rows = np.arange(nrow, dtype=np.int64)
    r10 = int(FM['lut'][10])   # standard FM10 in the default table (custom fuels reorder fm; PR #7 review)
    cur = -1.0 if P['curing'] is None else float(P['curing'])
    for z, df in enumerate(m_all):
        M = df[['m1h', 'm10h', 'm100h', 'm1h', 'mherb', 'mwoody']].to_numpy(float) / 100.0
        for k in range(nk):
            R.bundles_for(rows, fm['w0'], fm['sav'], fm['heat'], fm['depth'], fm['mx'], fm['dynamic'], M[k], cur,
                          bundles[k, z])
            R.bundle_one(FM['w0'][r10], FM['sav'][r10], FM['heat'][r10], FM['depth'][r10], FM['mx'][r10],
                         FM['dynamic'][r10], M[k], cur, bundle10[k, z])
        u20[:, z] = df['wind_speed'].to_numpy() * U10_TO_U20 * R.MS_TO_FTMIN
        wto[:, z] = np.radians((df['wind_dir'].to_numpy() + 180.0) % 360.0)
        u10k[:, z] = df['wind_speed'].to_numpy() * 3.6
        m1[:, z] = df['m1h'].to_numpy()
        tair[:, z] = df['T'].to_numpy()
    # per-cell arrays
    rowsc = lookup_rows(ls['fuel'], fm)
    depth = np.where(rowsc >= 0, fm['depth'][np.maximum(rowsc, 0)], 0.0)
    if P['waf'] is None:
        waf, sheltered = R.waf(depth, ls['cc'], ls['ch'], ls['cbh'])
    else:
        waf = np.broadcast_to(np.asarray(P['waf'], float), ls.shape).copy()
        sheltered = np.zeros(ls.shape, bool)
    adj = np.ones(ls.shape)
    if P['ros_adjust']:
        nums = ls['fuel']
        for key, fac in P['ros_adjust'].items():
            if isinstance(key, str) and key in GROUPS:
                adj[np.isin(nums, list(GROUPS[key]))] *= fac
            elif key == 'all':
                adj *= fac
            else:
                adj[nums == int(key)] *= fac
    if P['ros_factor'] is not None:
        adj = adj * np.asarray(P['ros_factor'], float)
    return dict(ls=ls, P=P, fm=fm, start=start, end=end, t0=t0, steps=steps, zone=zone, moist=m_all,
                bundles=bundles, bundle10=bundle10, u20=u20, wto=wto, u10k=u10k, m1=m1, tair=tair, rowsc=rowsc,
                waf=waf, sheltered=sheltered, adj=adj)


def _blocking(ls, barriers, suppression, start):
    ny, nx = ls.shape
    t_block = np.full((ny, nx), np.inf)
    nonburn = np.zeros((ny, nx), bool)
    if barriers is not None:
        items = barriers if isinstance(barriers, list) else [barriers]
        for b in items:
            if isinstance(b, np.ndarray):
                nonburn |= b.astype(bool)
            elif isinstance(b, dict) and 'lines' in b:
                nonburn |= rasterize_lines(ls.shape, b['lines'], ls.res, None if b.get('pixel') else
                                           ls.meta.get('transform'), b.get('width_cells', 1))
            elif isinstance(b, dict) and 'mask' in b:
                nonburn |= np.asarray(b['mask'], bool)
    ros_factor = None
    ia = None
    if suppression:
        for ln in suppression.get('lines', []):
            m = ln['mask'] if 'mask' in ln else rasterize_lines(ls.shape, ln['lines'], ls.res, None if ln.get('pixel')
                                                               else ls.meta.get('transform'), ln.get('width_cells', 1))
            t = ln.get('time', 0.0)
            if not isinstance(t, (int, float)):
                t = (pd.Timestamp(t).tz_localize('UTC') if pd.Timestamp(t).tz is None else pd.Timestamp(t)) - start
                t = t / pd.Timedelta(hours=1)
            t_block = np.where(m, np.minimum(t_block, t), t_block)
        if 'ros_factor' in suppression:
            ros_factor = np.asarray(suppression['ros_factor'], float)
        if 'initial_attack' in suppression:
            ia = dict(start_h=1.0, line_rate_m_per_h=600.0, lw=2.0)
            ia.update(suppression['initial_attack'])
    return nonburn, t_block, ros_factor, ia


def simulate(landscape, weather, ignition, start, end, dt='1h', spread='rothermel', crown='van_wagner',
             spotting=False, method='mtt', moisture='nelson', barriers=None, suppression=None, params=None, rng=None,
             perimeter_times=None, return_perimeters=True, prepared=None):
    """Simulate fire spread. See README 'API'. Returns a dict with rasters, perimeters, area series, labels, ledger."""
    t_wall = _time.perf_counter()
    if spread != 'rothermel':
        raise ValueError("only spread='rothermel' is implemented")
    meth = {'mtt': 16, 'level_set': 32, 'mtt32': 32, 'ca': 8, 'mtt8': 8}
    if method == 'huygens':
        raise NotImplementedError("method='huygens' (vertex-based FARSITE wavelets) is not implemented; use 'mtt' "
                                  "or 'level_set' (arrival-time solvers that give the same elliptical shapes)")
    if method not in meth:
        raise ValueError(method)
    pr = prepared if prepared is not None else prepare(landscape, weather, start, end, params, moisture)
    ls, P = pr['ls'], pr['P']
    if params and 'stencil' in params:
        nst = int(params['stencil'])
    else:
        nst = meth[method]
    if P['physics'] not in ('p1', 'p2'):
        raise ValueError(f"params['physics'] must be 'p1' or 'p2', not {P['physics']!r}")
    nonburn, t_block, sup_rf, ia = _blocking(ls, barriers, suppression, pr['start'])
    adj = pr['adj'] * (sup_rf if sup_rf is not None else 1.0)
    rowsc = np.where(nonburn, -1, pr['rowsc'])
    ny, nx = ls.shape
    offset_h = (pr['start'] - pr['t0']) / pd.Timedelta(hours=1)     # simulation clock starts at `start`
    t_end = (pr['end'] - pr['start']) / pd.Timedelta(hours=1)
    cells, times, mask_t = parse_ignition(ignition, ls, pr['start'])
    stn = _st.build(nst)
    crown_mode = 0 if crown in (None, 'none', False) else (2 if P['crown_spread'] == 'cruz2005' else 1)
    wl = {'none': R.WL_NONE, 'rothermel': R.WL_ROTHERMEL, 'andrews2013': R.WL_ANDREWS}[P['wind_limit']]
    lwm = {'behaveplus': R.LW_BEHAVE, 'anderson1983': R.LW_ANDERSON}[P['lw']]
    rs = np.random.default_rng(rng)
    seed = int(P['spot_seed']) if P['spot_seed'] is not None else int(rs.integers(0, 2 ** 31 - 1))
    # shift the weather so that kernel time 0 == start (weather step k covers [k - offset, k + 1 - offset) h)
    kern_times = times + offset_h
    out = run_mtt(ny, nx, ls.res, rowsc.ravel().astype(np.int64), pr['zone'].ravel(), np.tan(np.radians(ls['slope'])).ravel(),
                  np.radians((ls['aspect'] + 180.0) % 360.0).ravel(), pr['waf'].ravel().astype(float),
                  ls['cbh'].ravel().astype(float), ls['cbd'].ravel().astype(float), ls['ch'].ravel().astype(float),
                  np.ascontiguousarray(adj, float).ravel(), (t_block + offset_h).ravel(),
                  pr['bundles'], pr['bundle10'], pr['u20'], pr['wto'], pr['u10k'], pr['m1'], pr['tair'],
                  float(P['fmc']), 1.0, crown_mode, wl, lwm, float(P['lw_max']), float(P['crown_ros_mult']),
                  stn['off_dy'], stn['off_dx'], stn['off_len'], stn['seg_start'], stn['seg_n'], stn['seg_dy'],
                  stn['seg_dx'], stn['seg_frac'], stn['corner_a'], stn['corner_b'],
                  stn['opp'], stn['nb_prev'], stn['nb_next'], bool(method != 'ca' if P['simplex'] is None else P['simplex']),
                  P['physics'] == 'p2', stn['corner_j'],
                  cells, kern_times, t_end + offset_h,
                  bool(spotting), seed, float(P['ember_rate']), float(P['spot_med_frac']), float(P['spot_sigma']),
                  float(P['spot_dir_sd']), float(P['spot_p_est']), float(P['spot_delay_h']),
                  float(P['spot_surface_fli']), float(P['shade']),
                  ia is not None, float((ia or {}).get('start_h', 0.0)) + offset_h,
                  float((ia or {}).get('line_rate_m_per_h', 0.0)), float((ia or {}).get('lw', 2.0)), int(P['max_spots']))
    (arr, adir, ftype, cfb, rosd, roshead, hpa, phiw, phis, lwr, i0, rc, cfl, src, sp, overflow, cont_t) = out
    if overflow:
        raise RuntimeError('heap overflow in the MTT solver (please report)')
    arr = arr.reshape(ny, nx) - offset_h
    arr[~np.isfinite(arr)] = np.nan
    if mask_t is not None:
        # Every cell of a mask or polygon ignition burns at that ignition's time (parse_ignition seeds only the edge,
        # so the kernel would otherwise give the interior later arrival times); ignitions after the end leave their
        # cells as the kernel set them. PR #7 review, 21 Nov 2026.
        m = mask_t <= t_end
        arr[m] = np.fmin(arr[m], mask_t[m])
    res = _postprocess(arr, ftype.reshape(ny, nx), cfb.reshape(ny, nx), rosd.reshape(ny, nx),
                       roshead.reshape(ny, nx), hpa.reshape(ny, nx), cfl.reshape(ny, nx), ls.res)
    rasters = dict(arrival_time=arr, **res, phi_w=phiw.reshape(ny, nx), phi_s=phis.reshape(ny, nx),
                   lw_ratio=lwr.reshape(ny, nx), crown_i0=i0.reshape(ny, nx), crown_ros_active=rc.reshape(ny, nx),
                   source=src.reshape(ny, nx), arrival_dir_deg=np.degrees(adir.reshape(ny, nx)) % 360.0)
    for k in ('phi_w', 'phi_s', 'lw_ratio', 'crown_ros_active', 'arrival_dir_deg'):
        rasters[k] = np.where(np.isnan(arr), np.nan, rasters[k]).astype(np.float32)
    rasters['crown_i0'] = np.where(np.isnan(arr) | ~np.isfinite(rasters['crown_i0']), np.nan,
                                   rasters['crown_i0']).astype(np.float32)
    rasters['waf'] = pr['waf'].astype(np.float32)
    # time series
    step = pd.Timedelta(dt)
    tgrid = pd.date_range(pr['start'], pr['end'], freq=step)
    hrs = (tgrid - pr['start']) / pd.Timedelta(hours=1)
    cell_ha = ls.res ** 2 / 1e4
    a_sorted = np.sort(arr[np.isfinite(arr)])
    area = np.searchsorted(a_sorted, hrs.to_numpy(), side='right') * cell_ha
    area_series = pd.Series(area, index=tgrid, name='burned_ha')
    perims = []
    if return_perimeters:
        ptimes = perimeter_times if perimeter_times is not None else list(hrs)
        for th in ptimes:
            if not isinstance(th, (int, float, np.floating)):
                th = (pd.Timestamp(th).tz_localize('UTC') if pd.Timestamp(th).tz is None else pd.Timestamp(th)) - pr['start']
                th = th / pd.Timedelta(hours=1)
            perims.append(dict(time=pr['start'] + pd.Timedelta(hours=float(th)), hours=float(th),
                               **perimeter(arr, th, ls)))
    spots = pd.DataFrame(sp, columns=['t_src_h', 'src', 'dst', 'dist_m', 't_ign_h', 'ignited'])
    if len(spots):
        spots['t_src_h'] -= offset_h
        spots['t_ign_h'] -= offset_h
        for c in ('src', 'dst'):
            v = spots[c].astype(np.int64).to_numpy()
            spots[c + '_row'] = v // nx
            spots[c + '_col'] = v % nx
        spots = spots.drop(columns=['src', 'dst'])
    moist = pr['moist'][0]
    labels = dict(moisture=moist[['m1h', 'm10h', 'm100h', 'mherb', 'mwoody']].copy(),
                  moisture_zones=[m[['m1h', 'm10h', 'm100h', 'mherb', 'mwoody']] for m in pr['moist']] if len(pr['moist']) > 1 else None,
                  weather=moist[['T', 'RH', 'wind_speed', 'wind_dir', 'precip']].copy(),
                  wind_20ft_ms=moist['wind_speed'] * U10_TO_U20,
                  curing=_curing_series(moist, P), params=P, moisture_method=moisture, method=method, stencil=nst,
                  crown=crown, spotting=bool(spotting), spot_events=spots, n_spot_ignitions=int(spots['ignited'].sum()) if len(spots) else 0,
                  ros_adjust_raster_mean=float(np.mean(adj)),
                  waf_mean_burned=float(np.nanmean(np.where(np.isfinite(arr), pr['waf'], np.nan))) if np.isfinite(arr).any() else np.nan,
                  phi_w_mean_burned=float(np.nanmean(rasters['phi_w'])) if np.isfinite(arr).any() else np.nan,
                  phi_s_mean_burned=float(np.nanmean(rasters['phi_s'])) if np.isfinite(arr).any() else np.nan,
                  contained_h=(cont_t - offset_h) if np.isfinite(cont_t) else None,
                  initial_attack=ia)
    ledger = _ledger(arr, rasters, ls, hrs.to_numpy(), perims)
    return dict(rasters=rasters, arrival_time=arr, perimeters=perims, area=area_series, labels=labels,
                ledger=ledger, meta=dict(ls.meta), fuel_raster=np.asarray(ls['fuel']), start=pr['start'], end=pr['end'],
                wall_s=_time.perf_counter() - t_wall)


def simulate_batch(landscape, weather, ignitions, start, end, params=None, moisture='nelson', rng=None,
                   keep=('arrival_time',), **kw):
    """Batch mode: many ignitions on one landscape and weather. Fuel moisture, fuel bundles and WAF are computed once
    (prepare), then the arrival-time kernel runs per ignition. Returns a list of dicts with the requested `keep`
    entries ('arrival_time', any raster name, 'area', 'labels', 'ledger', or 'all') plus 'wall_s'.
    For many landscapes, call it once per landscape (prepare dominates only for long weather series)."""
    pr = prepare(landscape, weather, start, end, params, moisture)
    rs = np.random.default_rng(rng)
    res = []
    kw.setdefault('return_perimeters', False)
    for ign in ignitions:
        o = simulate(landscape, weather, ign, start, end, params=params, moisture=moisture,
                     rng=int(rs.integers(0, 2 ** 31 - 1)), prepared=pr, **kw)
        if 'all' in keep:
            res.append(o)
            continue
        d = dict(wall_s=o['wall_s'])
        for k in keep:
            d[k] = o[k] if k in o else o['rasters'][k]
        res.append(d)
    return res


def _curing_series(m, P):
    if P['curing'] is not None:
        return pd.Series(float(P['curing']), index=m.index)
    return pd.Series(np.clip(1.333 - 1.11 * m['mherb'].to_numpy() / 100.0, 0, 1), index=m.index)


def _postprocess(arr, ftype, cfb, rosd, roshead, hpa, cfl, res):
    """Front-normal ROS from the arrival-time gradient (1/|grad T|), capped by the cell's directional range;
    fireline intensity (Byram) with the crown fuel contribution; flame length (Byram for surface, Thomas 1963 for
    crown fires)."""
    burned = np.isfinite(arr)
    tm = np.where(burned, arr * 60.0, np.nan)      # minutes
    gy = np.full(arr.shape, np.nan)
    gx = np.full(arr.shape, np.nan)
    p = np.pad(tm, 1, constant_values=np.nan)
    c = p[1:-1, 1:-1]
    for g, a, b in ((gx, p[1:-1, :-2], p[1:-1, 2:]), (gy, p[:-2, 1:-1], p[2:, 1:-1])):
        both = np.isfinite(a) & np.isfinite(b)
        g[both] = (b[both] - a[both]) / (2 * res)
        only_b = ~both & np.isfinite(b) & np.isfinite(c)
        g[only_b] = (b[only_b] - c[only_b]) / res
        only_a = ~both & np.isfinite(a) & np.isfinite(c)
        g[only_a] = (c[only_a] - a[only_a]) / res
    gm = np.hypot(np.nan_to_num(gx), np.nan_to_num(gy))
    with np.errstate(divide='ignore'):
        rn = np.where(gm > 0, 1.0 / gm, np.nan)
    ros = np.where(np.isfinite(rn), np.minimum(rn, np.maximum(roshead, rosd)), rosd)
    ros = np.where(burned, ros, np.nan)
    heat = hpa + cfb * cfl * 18000.0
    fli = heat * ros / 60.0
    fl = np.where(ftype >= 2, 0.0266 * np.maximum(fli, 0) ** (2 / 3), 0.0775 * np.maximum(fli, 0) ** 0.46)
    nanify = lambda a: np.where(burned, a, np.nan).astype(np.float32)
    return dict(ros=nanify(ros), ros_head=nanify(roshead), ros_dir=nanify(rosd), fireline_intensity=nanify(fli),
                flame_length=nanify(fl), fire_type=np.where(burned, ftype, 0).astype(np.int8),
                heat_per_area=nanify(heat), hpa_surface=nanify(hpa), cfb=nanify(cfb))


def perimeter(arr, th, ls):
    """Burned mask and boundary polygons (pixel row/col coordinates, and projected x/y if a transform exists)."""
    m = np.isfinite(arr) & (arr <= th)
    polys = []
    if m.any():
        try:
            from skimage.measure import find_contours
            cs = find_contours(np.pad(m.astype(float), 1), 0.5)
            cs = [c - 1 for c in cs]
        except ImportError:  # pragma: no cover
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig = plt.figure()
            cs = [v[:, ::-1] - 0.5 for v in plt.contour(np.pad(m.astype(float), 1), [0.5]).allsegs[0]]
            cs = [c - 1 + 0.5 for c in cs]
            plt.close(fig)
        t = ls.meta.get('transform')
        for c in cs:
            rc = c + 0.5     # find_contours returns cell-centre coordinates; +0.5 to pixel-edge frame
            d = dict(rowcol=c)
            if t is not None and len(t) == 6:
                d['xy'] = np.column_stack([t[0] + rc[:, 1] * t[1], t[3] + rc[:, 0] * t[5]])
            polys.append(d)
    return dict(area_ha=float(m.sum() * ls.res ** 2 / 1e4), mask_cells=int(m.sum()), polygons=polys)


def _ledger(arr, r, ls, hrs, perims):
    """Consistency ledger: (1) area growth vs integrated spread (sum over each hour's newly burned cells of front
    length x normal ROS, with the perimeter length from the contours); (2) energy released; (3) area by fire type."""
    burned = np.isfinite(arr)
    cell = ls.res ** 2
    led = dict(burned_cells=int(burned.sum()), burned_ha=float(burned.sum() * cell / 1e4))
    E_surf = float(np.nansum(np.where(burned, r['hpa_surface'], 0)) * cell / 1e6)      # GJ
    E_tot = float(np.nansum(np.where(burned, r['heat_per_area'], 0)) * cell / 1e6)
    led.update(energy_surface_GJ=E_surf, energy_total_GJ=E_tot, energy_crown_GJ=E_tot - E_surf)
    for k, name in ((1, 'surface'), (2, 'passive_crown'), (3, 'active_crown')):
        led[f'area_ha_{name}'] = float(((r['fire_type'] == k) & burned).sum() * cell / 1e4)
    # integrated spread (coarea check): A(t1) - A(t0) ~ (t1 - t0) * line integral of the front-normal ROS along the
    # mid-time isochrone T = (t0 + t1) / 2 (contoured on the arrival-time field, ROS sampled at the nearest cell)
    rows = []
    if len(hrs) >= 2 and burned.sum() > 50:
        try:
            from skimage.measure import find_contours
        except ImportError:  # pragma: no cover
            find_contours = None
        if find_contours is not None:
            big = float(np.nanmax(arr)) + 1e3
            fld = np.pad(np.where(burned, arr, big), 1, constant_values=big)
            ros = np.pad(np.where(burned, r['ros'], np.nan), 1, constant_values=np.nan)
            a_sorted = np.sort(arr[burned])
            for t0, t1 in zip(hrs[:-1], hrs[1:]):
                n0, n1 = np.searchsorted(a_sorted, [t0, t1], side='right')
                if n1 - n0 < 20 or n0 < 20:
                    continue
                tm = 0.5 * (t0 + t1)
                pred = 0.0
                for c in find_contours(fld, tm):
                    mid = 0.5 * (c[1:] + c[:-1])
                    seg = np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1])) * ls.res
                    r0_ = np.clip(np.floor(mid[:, 0]).astype(int), 0, fld.shape[0] - 2)
                    c0_ = np.clip(np.floor(mid[:, 1]).astype(int), 0, fld.shape[1] - 2)
                    # the 2 x 2 cells around the segment: ROS from the burned (inside) cells; the segment is an
                    # active front only if an outside cell burns later (not a barrier, unburnable or the edge)
                    v_sum = np.zeros(len(mid))
                    v_n = np.zeros(len(mid))
                    active = np.zeros(len(mid), bool)
                    for dr in (0, 1):
                        for dc in (0, 1):
                            f_ = fld[r0_ + dr, c0_ + dc]
                            ins = f_ <= tm
                            rv = ros[r0_ + dr, c0_ + dc]
                            ok = ins & np.isfinite(rv)
                            v_sum += np.where(ok, rv, 0.0)
                            v_n += ok
                            active |= (~ins) & (f_ < big)
                    v = np.where(active & (v_n > 0), v_sum / np.maximum(v_n, 1), 0.0)
                    pred += float(np.sum(seg * v) * 60.0 * (t1 - t0))
                rows.append((t1, (n1 - n0) * cell, pred))
        if rows:
            a = np.array(rows)
            led['growth_check'] = dict(hours=a[:, 0].tolist(), dA_raster_m2=a[:, 1].tolist(),
                                       dA_perimeter_x_ros_m2=a[:, 2].tolist(),
                                       ratio_total=float(a[:, 2].sum() / max(a[:, 1].sum(), 1e-9)),
                                       ratio_median=float(np.median(a[:, 2] / np.maximum(a[:, 1], 1e-9))))
    return led


def _poly_len(c):
    return float(np.sum(np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1]))))
