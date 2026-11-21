"""FireSize free-burn simulation of one ignition in a real 90 m LANDFIRE window under real gridMET weather.

Physics (all FireSim): Rothermel/Albini surface spread with Scott & Burgan fuels, Van Wagner crown fire, MTT arrival
times on the 90 m grid, Nelson dead fuel moisture from hourly weather disaggregated from gridMET daily
(fs.hourly_from_daily), live moisture from the GSI (Jolly 2005, NFDRS-style mapping). No spotting.

Two growth models are combined into one free-burn area curve A(t) (t = hours since ignition):
  * sub-cell: an analytic ellipse at the ignition cell (head + back ROS integrated hour by hour, area pi/4 L^2 / LW),
    because a 90 m cell is 0.81 ha and most FPA-FOD fires are smaller than that;
  * grid: the burned area from the MTT arrival times, once the analytic ellipse is bigger than 4 cells.
Suppression is applied afterwards (supp.py) as a Fried & Fried line-vs-perimeter race on A(t), so one simulation
serves any suppression model and the suppression model can be calibrated without re-simulating.
"""
import time
from common import *
import firesim as fs
import importlib
from firesim import kernel as K
SIM = importlib.import_module("firesim.simulate")
from firesim.weather import hourly_from_daily
from firesim.moisture import dead_moisture, gsi, live_moisture_from_gsi, daylength_h
from firesim.realdata import grid_convergence_deg

TG = np.r_[0.0, np.geomspace(0.02, 96.0, 160)]       # output times, h since ignition
T_END = 96.0
SW_CELLS = 4.0
_WX = {}


def load_window(win):
    if _LW.get('win') == win:
        ls = _LW['ls']
        return fs.Landscape(meta=dict(ls.meta), res_m=RES, **{k: v.copy() for k, v in ls.items()}), _LW['x0'], _LW['y1']
    ls, x0, y1 = _load_window(win)
    _LW.update(win=win, ls=ls, x0=x0, y1=y1)
    return load_window(win)


_LW = {}


def _load_window(win):
    z = np.load(f'{DATA}/win/{win}.npz')
    x0, y1 = float(z['x0']), float(z['y1'])
    ls = fs.Landscape(fuel=z['fuel'].astype(np.int64), cc=z['cc'].astype(float), ch=z['ch'] / 10.0,
                      cbh=z['cbh'] / 10.0, cbd=z['cbd'] / 100.0, elev=z['elev'].astype(float), res_m=RES,
                      meta=dict(crs='EPSG:5070', transform=(x0, RES, 0.0, y1, 0.0, -RES)))
    ls['fuel'] = np.where(ls['fuel'] == 0, 99, ls['fuel'])      # nodata -> non-burnable
    return ls, x0, y1


def wx_table():
    if 'w' not in _WX:
        z = np.load(f'{DATA}/wx_points.npz')
        _WX['w'] = z['wx']
        _WX['off'] = int(z['off'])
        _WX['vars'] = list(z['vars'])
    return _WX


def daily_frame(ign_idx, date):
    T = wx_table()
    a = T['w'][ign_idx]
    days = pd.date_range(pd.Timestamp(date) - pd.Timedelta(days=T['off']), periods=a.shape[1], freq='D')
    d = pd.DataFrame(a.T, index=days, columns=T['vars'])
    return d.interpolate(limit_direction='both')


_TR = {}


def tr_inv():
    if 'inv' not in _TR:
        from pyproj import Transformer
        _TR['inv'] = Transformer.from_crs(5070, 4326, always_xy=True)
        _TR['fwd'] = Transformer.from_crs(4326, 5070, always_xy=True)
    return _TR['inv'], _TR['fwd']


def convergence(x, y):
    """Same as firesim.realdata.grid_convergence_deg, with cached transformers."""
    inv, fwd = tr_inv()
    lon, lat = inv.transform(x, y)
    x2, y2 = fwd.transform(lon, lat + 0.01)
    return float(np.degrees(np.arctan2(x2 - x, y2 - y)))


def hourly_fast(day, lat, lon, t_peak_lst=15.0, wind_amp=0.45):
    """Vectorised copy of firesim.weather.hourly_from_daily (same formulas; checked equal in tests)."""
    n = len(day)
    tn = day.tmin.to_numpy(float); tx = day.tmax.to_numpy(float)
    rmin = day.rmin.to_numpy(float); rmax = day.rmax.to_numpy(float)
    ws = day.vs.to_numpy(float); wd = day.th.to_numpy(float); pr = day.pr.to_numpy(float)
    txp = np.r_[tx[0], tx[:-1]]; tnn = np.r_[tn[1:], tn[-1]]
    lst = np.arange(24) + 0.5
    TN, TX, TXP, TNN = (a[:, None] for a in (tn, tx, txp, tnn))
    L = lst[None, :]
    T = np.where(L < 6.0, TN + (TXP - TN) * np.exp(-(L + 24.0 - t_peak_lst) / 4.0),
                 np.where(L <= t_peak_lst, TN + (TX - TN) * np.sin(np.pi / 2 * (L - 6.0) / (t_peak_lst - 6.0)),
                          TNN + (TX - TNN) * np.exp(-(L - t_peak_lst) / 4.0)))
    frac = np.clip((T - TN) / np.maximum(TX - TN, 0.1), 0, 1)
    rh = rmax[:, None] + (rmin - rmax)[:, None] * frac
    u = np.maximum(ws[:, None] * (1.0 + wind_amp * np.cos(2 * np.pi * (L - 15.0) / 24.0)), 0.0)
    p = np.where((L >= 14) & (L < 20), pr[:, None] / 6.0, 0.0)
    off = lon / 15.0
    base = pd.DatetimeIndex(day.index).tz_localize(None).normalize()
    t = (base.values[:, None] + ((np.arange(24) - off) * 3600e9).astype('timedelta64[ns]')[None, :]).ravel()
    out = pd.DataFrame(dict(T=T.ravel(), RH=rh.ravel(), wind_speed=u.ravel(), wind_dir=np.repeat(wd, 24),
                            precip=p.ravel()), index=pd.DatetimeIndex(t).tz_localize('UTC').round('h'))
    out.index.name = 'time'
    out = out[~out.index.duplicated()]
    out.attrs['lat'] = lat
    out.attrs['lon'] = lon
    return out


def hourly_weather(daily, x, y, date, lat, lon):
    """Hourly contract weather [date-10 d, date+5 d) with Nelson dead moisture and GSI live moisture."""
    d0 = pd.Timestamp(date)
    g = gsi(daily.tmmn.values, daily.vpd.values * 1000.0, daylength_h(daily.index.dayofyear.values, lat))
    mh, mw = live_moisture_from_gsi(g)
    dd = daily.loc[d0 - pd.Timedelta(days=10): d0 + pd.Timedelta(days=5)]
    day = pd.DataFrame(dict(tmin=dd.tmmn, tmax=dd.tmmx, rmin=dd.rmin, rmax=dd.rmax, vs=dd.vs, th=dd.th, pr=dd.pr))
    w = hourly_fast(day, lat, lon)
    w['wind_dir'] = (w['wind_dir'] + convergence(x, y)) % 360.0
    m = dead_moisture(w, method='nelson')
    for c in m:
        w[c] = m[c].to_numpy()
    lm = pd.Series(mh, index=daily.index).reindex(w.index.tz_localize(None).normalize(), method='nearest')
    lw = pd.Series(mw, index=daily.index).reindex(w.index.tz_localize(None).normalize(), method='nearest')
    w['mherb'] = lm.to_numpy()
    w['mwoody'] = lw.to_numpy()
    return w, float(mh[np.searchsorted(daily.index, d0)]), float(mw[np.searchsorted(daily.index, d0)])


def point_growth(pr, cell, t_ign_k, tgrid):
    """Analytic ellipse at the ignition cell: returns area (ha) and L/W at tgrid (h since ignition), using FireSim's
    own cell_state (same Rothermel, WAF, crown and ros_adjust as the grid solver)."""
    ls, P = pr['ls'], pr['P']
    rowsc = pr['rowsc'].ravel().astype(np.int64)
    zone = pr['zone'].ravel()
    st = np.zeros(K.NS)
    tmp = np.zeros(64)
    res3 = np.zeros(3)
    slope_tan = np.tan(np.radians(ls['slope'])).ravel()
    upaz = np.radians((ls['aspect'] + 180.0) % 360.0).ravel()
    nk = pr['u20'].shape[0]
    hours = int(np.ceil(tgrid[-1])) + 1
    L = np.zeros(hours + 1)          # cumulative length (m) at whole hours since ignition
    lwsum = np.zeros(hours + 1)
    lw_now = 1.0
    waf = pr['waf'].ravel().astype(float); cbh = ls['cbh'].ravel().astype(float)
    cbd = ls['cbd'].ravel().astype(float); ch = ls['ch'].ravel().astype(float)
    adj = np.ascontiguousarray(pr['adj'], float).ravel()
    for h in range(hours):
        k = min(int(t_ign_k + h), nk - 1)
        K.cell_state(cell, k, rowsc, zone, slope_tan, upaz, waf, cbh, cbd, ch, adj, pr['bundles'], pr['bundle10'],
                     pr['u20'], pr['wto'], pr['u10k'], pr['m1'], float(P['fmc']), 1, 0, 0, float(P['lw_max']),
                     float(P['crown_ros_mult']), st, tmp)
        if st[0] > 0:
            K.dir_ros(st, st[1], 1, res3)
            rh = res3[0]
            e = st[2]
            rb = rh * (1 - e) / (1 + e)
            lw_now = max(st[13], 1.0)
        else:
            rh = rb = 0.0
        L[h + 1] = L[h] + (rh + rb) * 60.0
        lwsum[h + 1] = lw_now
    Lt = np.interp(tgrid, np.arange(hours + 1), L)
    lwt = np.interp(tgrid, np.arange(hours + 1), lwsum)
    lwt[0] = lwt[1] if len(lwt) > 1 else 1.0
    A = np.pi / 4.0 * Lt ** 2 / np.maximum(lwt, 1.0) / 1e4
    return A, lwt


PHYS = {'v0': dict(ag=None, reloc=0),            # LANDFIRE as is: agriculture (93) and urban (91) never burn
        'v1': dict(ag=101, reloc=2),             # agriculture burns as GR1; NB ignition cell -> nearest burnable <= 2 cells
        'v1b': dict(ag=102, reloc=2),
        'v1c': dict(ag=None, reloc=2)}


def run_one(ign, rs, ros_mult=None, tau0=None, physics='v0'):
    """Free-burn simulation of one ignition row. Returns dict with the growth curve and the sim draws."""
    t_wall = time.perf_counter()
    ph = PHYS[physics]
    ls, x0, y1 = load_window(ign.win)
    if ph['ag'] is not None:
        ls['fuel'] = np.where(ls['fuel'] == 93, ph['ag'], ls['fuel'])
    lon, lat = tr_inv()[0].transform(ign.x + 45, ign.y + 45)
    daily = daily_frame(int(ign.ign), ign.date)
    w, mh0, mw0 = hourly_weather(daily, ign.x, ign.y, ign.date, lat, lon)
    # ignition time (UTC) = local discovery time - tau0 (pre-discovery free burn, broad prior; reweighted later)
    if tau0 is None:
        tau0 = float(np.exp(rs.normal(np.log(1.0), 1.0)))
    tau0 = float(np.clip(tau0, 0.02, 24.0))
    if ros_mult is None:
        ros_mult = float(np.exp(rs.uniform(np.log(0.2), np.log(1.5))))
    disc_local = pd.Timestamp(ign.date) + pd.Timedelta(hours=float(ign.hour_sim))
    disc_utc = (disc_local - pd.Timedelta(hours=lon / 15.0)).tz_localize('UTC')
    t_ign = disc_utc - pd.Timedelta(hours=tau0)
    start = t_ign.floor('h')
    end = start + pd.Timedelta(hours=T_END + 2)
    r, c = int((y1 - ign.y - 1) // RES), int((ign.x + 1 - x0) // RES)
    r, c = min(max(r, 0), WIN - 1), min(max(c, 0), WIN - 1)
    fuel0 = int(ls['fuel'][r, c])
    if ph['reloc'] and not (100 < fuel0 < 210):
        k = ph['reloc']
        sub = ls['fuel'][max(r - k, 0):r + k + 1, max(c - k, 0):c + k + 1]
        ok = np.argwhere((sub > 100) & (sub < 210))
        if len(ok):
            dd = ((ok - [min(r, k), min(c, k)]) ** 2).sum(1)
            j = ok[np.argmin(dd + rs.uniform(0, 0.1, len(dd)))]
            r, c = int(max(r - k, 0) + j[0]), int(max(c - k, 0) + j[1])
    fuel_pt = fuel0
    fuel0 = int(ls['fuel'][r, c])
    params = dict(ros_adjust={'all': ros_mult}, moisture_spinup_h=0)
    w = w[(w.index >= start - pd.Timedelta(hours=1)) & (w.index <= end + pd.Timedelta(hours=1))]
    if not (100 < fuel0 < 210):         # non-burnable ignition cell: the fire cannot spread
        z = np.zeros(len(TG), np.float32)
        return dict(A=z, lw=z + 1, edge_t=np.inf, tau0=tau0, ros_mult=ros_mult, fuel0=fuel0, fuel_pt=fuel_pt,
                    mherb=mh0, mwoody=mw0, burnable=False, wall=time.perf_counter() - t_wall)
    pr = SIM.prepare(ls, w, start, end, params, 'given')
    cell = r * WIN + c
    off = (t_ign - start) / pd.Timedelta(hours=1)
    A_pt, lw = point_growth(pr, cell, off, TG)
    # grid run (skipped when the ignition cell cannot burn)
    A_grid = np.zeros_like(TG)
    edge_t = np.inf
    burnable = pr['rowsc'][r, c] >= 0
    if burnable:
        o = fs.simulate(ls, w, {'rowcol': [(r, c)], 'time': float(off)}, start, end, moisture='given', params=params,
                        prepared=pr, return_perimeters=False, crown='van_wagner', spotting=False)
        arr = o['arrival_time'] - off
        a = np.sort(arr[np.isfinite(arr)])
        A_grid = np.searchsorted(a, TG, side='right') * (RES * RES / 1e4)
        e = np.r_[arr[0], arr[-1], arr[:, 0], arr[:, -1]]
        e = e[np.isfinite(e)]
        edge_t = float(e.min()) if len(e) else np.inf
    cell_ha = RES * RES / 1e4
    sw = A_pt >= SW_CELLS * cell_ha
    if sw.any():
        k = int(np.argmax(sw))
        A = np.where(np.arange(len(TG)) < k, A_pt, np.maximum(A_grid, A_pt[k]))
    else:
        A = A_pt
    A = np.maximum.accumulate(A)
    return dict(A=A.astype(np.float32), A_pt=A_pt.astype(np.float32), A_grid=A_grid.astype(np.float32),
                lw=lw.astype(np.float32), edge_t=edge_t, tau0=tau0, ros_mult=ros_mult, fuel0=fuel0, fuel_pt=fuel_pt, mherb=mh0,
                mwoody=mw0, burnable=bool(burnable), wall=time.perf_counter() - t_wall)
