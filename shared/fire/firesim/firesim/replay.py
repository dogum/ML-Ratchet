"""Fire-day replays: the classical physics benchmark for next-day spread.

A fire-day is the period between two observed perimeters (FEDS PM-to-PM, ~24 h). The replay starts from the
observed burned mask at the issue time (its edge cells ignite at t = 0, the interior counts as burned), spreads under
the hourly weather (HRRR at the fire, or gridded) over the real LANDFIRE landscape, and predicts the burned area at
the verification time. No suppression, no spotting (deterministic), crown fire on.

* replay_day(...)       deterministic run + optional ensemble -> arrival time, probability map, score map.
* Ensemble members perturb, per member and constant over the day: the 10-m wind speed (x lognormal), the wind
  direction (+ normal), and the spread rate (x lognormal, standing for fuel moisture / fuel model / suppression
  errors). The probability map is the fraction of members whose fire reaches a cell by the verification time.
* score map (deterministic): exp(-arrival / dt) (0 where not reached within 3 dt) - a ranking for AP.
* Calibration (dev only): ros_adjust factors by fuel group, as practitioners do (e.g. FARSITE/FlamMap "ROS
  adjustment factors"), fitted on dev fire-days; see checks/replay_dev.py and checks/replay_calibration.json.
"""
import numpy as np
import pandas as pd

from .simulate import simulate, prepare, fuel_group, GROUPS

ENSEMBLE = dict(n=12, ws_sd=0.25, wd_sd_deg=20.0, ros_sd=0.5)


def crop_box(burned_now, res_m, margin_m=20000.0, shape=None):
    """Row/col box around the burned mask plus a margin (metres)."""
    r, c = np.nonzero(burned_now)
    m = int(np.ceil(margin_m / res_m))
    ny, nx = shape if shape is not None else burned_now.shape
    return max(r.min() - m, 0), min(r.max() + m + 1, ny), max(c.min() - m, 0), min(c.max() + m + 1, nx)


def _perturbed(pr, f_ws, d_wd_rad, f_ros):
    q = dict(pr)
    q['u20'] = pr['u20'] * f_ws
    q['u10k'] = pr['u10k'] * f_ws
    q['wto'] = pr['wto'] + d_wd_rad
    q['adj'] = pr['adj'] * f_ros
    return q


def replay_day(ls, weather, burned_now, start, end, params=None, crown='van_wagner', ensemble=None, rng=0,
               moisture='nelson', score_tau=None, keep_prepared=False):
    """Replay one fire-day. ls: Landscape (already cropped if wanted); burned_now: bool mask on ls; start/end: UTC.
    Returns dict(arrival (h; NaN not reached), pred (bool, reached by end), score (deterministic ranking),
    prob (ensemble fraction or None), dt_h, wall_s, members (list of per-member factors))."""
    import time
    t0 = time.perf_counter()
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    start = start.tz_localize('UTC') if start.tz is None else start
    end = end.tz_localize('UTC') if end.tz is None else end
    dt_h = (end - start) / pd.Timedelta(hours=1)
    sim_end = end
    pr = prepare(ls, weather, start, sim_end, params, moisture)
    ign = {'mask': np.asarray(burned_now, bool)}
    out = simulate(ls, weather, ign, start, sim_end, crown=crown, spotting=False, params=params, moisture=moisture,
                   prepared=pr, return_perimeters=False, rng=rng)
    arr = out['arrival_time']
    tau = score_tau or dt_h
    score = np.where(np.isfinite(arr), np.exp(-np.nan_to_num(arr, nan=1e9) / tau), 0.0)
    pred = np.isfinite(arr) & (arr <= dt_h)
    res = dict(arrival=arr, pred=pred, score=score, prob=None, dt_h=dt_h, members=[],
               fire_type=out['rasters']['fire_type'], labels=dict(waf_mean=out['labels']['waf_mean_burned'],
                                                                 wind_mean_ms=float(out['labels']['weather']['wind_speed'].mean())))
    if ensemble:
        E = dict(ENSEMBLE)
        E.update(ensemble if isinstance(ensemble, dict) else {})
        rs = np.random.default_rng(rng)
        cnt = np.zeros(arr.shape, np.float32)
        for k in range(int(E['n'])):
            f_ws = float(np.exp(E['ws_sd'] * rs.standard_normal()))
            d_wd = float(np.radians(E['wd_sd_deg'] * rs.standard_normal()))
            f_ros = float(np.exp(E['ros_sd'] * rs.standard_normal()))
            o = simulate(ls, weather, ign, start, sim_end, crown=crown, spotting=False, params=params,
                         moisture=moisture, prepared=_perturbed(pr, f_ws, d_wd, f_ros), return_perimeters=False,
                         rng=rng)
            a = o['arrival_time']
            cnt += (np.isfinite(a) & (a <= dt_h))
            res['members'].append(dict(f_ws=f_ws, d_wd_deg=float(np.degrees(d_wd)), f_ros=f_ros))
        res['prob'] = cnt / float(E['n'])
    if keep_prepared:
        res['prepared'] = pr
    res['wall_s'] = time.perf_counter() - t0
    return res


def ring_fuel_group(ls, burned_now, width_cells=10):
    """Dominant burnable fuel group in a ring just outside the burned area (what the fire is about to burn)."""
    from scipy.ndimage import binary_dilation
    ring = binary_dilation(burned_now, iterations=width_cells) & ~burned_now
    f = np.asarray(ls['fuel'])[ring]
    f = f[(f < 91) | (f > 99)]
    if not len(f):
        return 'NB', {}
    groups = np.array([fuel_group(v) for v in np.unique(f)])
    cnt = {}
    u, c = np.unique(f, return_counts=True)
    for g, n in zip(groups, c):
        cnt[g] = cnt.get(g, 0) + int(n)
    tot = sum(cnt.values())
    share = {g: n / tot for g, n in cnt.items()}
    return max(share, key=share.get), share


def calibrated_params(calib, base=None, fire_day_index=None):
    """params dict with ros_adjust by fuel group from a calibration dict {'factors': {group: f}, 'default': f,
    'factor_first_day': f (optional)}. fire_day_index 0 (the first observed day of a fire) uses factor_first_day for
    every group when the calibration has one."""
    if fire_day_index == 0 and 'factor_first_day' in calib:
        calib = dict(factors={}, default=calib['factor_first_day'])
    p = dict(base or {})
    adj = {g: float(v) for g, v in calib['factors'].items() if g in GROUPS}
    for g in GROUPS:
        adj.setdefault(g, float(calib.get('default', 1.0)))
    p['ros_adjust'] = adj
    return p
