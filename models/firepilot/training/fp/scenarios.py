"""Scenario builders.

* real_scenario(fid, ...): a REAL fire at its first afternoon (PM) FEDS detection: the observed burned mask, the
  LANDFIRE window cropped to the burned bbox + 15 km, the hourly HRRR series at the burned centroid (winds rotated to
  grid north; missing precipitation from gridMET daily), and the observed PM masks of the next days (for fidelity).
  Exam fires only with for_exam_scoring=True (final exam script only).
* synth_scenario(seed, ...): a training / validation scenario: a real non-exam LANDFIRE window (dev landscapes minus
  the gate fires, firedata.guard) or a parametric FireSim landscape; real HRRR weather (dev packages minus the gate
  fires, guard-checked) or FireSim's weather generator; the initial fire grown from a point ignition for 6-30 h so
  the detection mask has a realistic shape; a random crew budget and hidden truth/forecast errors.
"""
import numpy as np
import pandas as pd

from .env import Scenario, PERTURB, HORIZON_H, fs, RP
from firesim import realdata as RD
import firedata as fd

GATE_SEED = 20261119
CROP_M = 15000.0
BUDGETS_EVAL = (300.0, 1000.0)       # m of line per working hour (two crews): ~3.6 and ~12 km per 12 h shift
FIRST_ID = None


def dev_ids():
    return sorted(fd.fires('dev').Event_ID)


def gate_ids(n=12):
    """Fixed seeded draw of dev fires for the real dev gate (never used for training banks)."""
    ids = dev_ids()
    rs = np.random.default_rng(GATE_SEED)
    return sorted(rs.choice(ids, n, replace=False).tolist())


def train_ids():
    g = set(gate_ids())
    return [f for f in dev_ids() if f not in g]


def hidden_draw(rs):
    return dict(f_ros=float(np.exp(PERTURB['ros_sd'] * rs.standard_normal())),
                f_ws=float(np.exp(PERTURB['ws_sd'] * rs.standard_normal())),
                d_wd=float(PERTURB['wd_sd_deg'] * rs.standard_normal()))


def real_scenario(fid, rate_mph, hidden, for_exam_scoring=False, physics='p1', crop_m=CROP_M, day=None):
    L = fd.landscape(fid, for_exam_scoring=for_exam_scoring)
    W = fd.weather(fid, for_exam_scoring=for_exam_scoring)
    prog = fd.progression(fid, for_exam_scoring=for_exam_scoring)
    ls_full = RD.landscape_from_firedata(L)
    B = prog['burned']
    tt = pd.DatetimeIndex(prog['t_utc']).tz_localize('UTC')
    nz = np.flatnonzero(B.reshape(len(B), -1).sum(1) > 0)
    if not len(nz):
        raise ValueError('no burned PM perimeter')
    i = int(nz[0]) if day is None else int(day)
    now = B[i]
    start = tt[i].floor('h')
    r0, r1, c0, c1 = RP.crop_box(now, ls_full.res, crop_m)
    ls = ls_full.subset(r0, r1, c0, c1)
    gt = ls_full.meta['transform']
    rr, cc = np.nonzero(now)
    w = RD.weather_series(W, gt[0] + (cc.mean() + 0.5) * gt[1], gt[3] + (rr.mean() + 0.5) * gt[5])
    w = RD.fill_precip_from_daily(w, W.get('daily'), w.attrs['lonlat'][0])
    if w.index[-1] < start + pd.Timedelta(hours=HORIZON_H - 1):
        raise ValueError('weather ends before the horizon')
    # observed PM masks after detection at ~24 h steps (for fidelity)
    obs = []
    for j in range(i + 1, len(B)):
        h = (tt[j] - tt[i]) / pd.Timedelta(hours=1)
        if h > HORIZON_H + 6:
            break
        obs.append(dict(h=float(h), mask=B[j][r0:r1, c0:c1] | now[r0:r1, c0:c1]))
    meta = dict(fire_id=fid, kind='real', day_index=i, box=(r0, r1, c0, c1), obs=obs,
                lonlat=w.attrs['lonlat'], edge_window=bool(r0 == 0 or c0 == 0 or r1 == B.shape[1] or c1 == B.shape[2]))
    return Scenario(ls, w, now[r0:r1, c0:c1], start, w.attrs['lonlat'][0], rate_mph, hidden, physics=physics, meta=meta)


def _pm_start(w, lon, spin_h):
    """First hour >= spin_h into the series whose local solar time is ~13-14 h."""
    idx = w.index
    for k in range(int(spin_h), len(idx) - int(HORIZON_H) - 1):
        loc = (idx[k].hour + lon / 15.0) % 24.0
        if 13.0 <= loc < 14.0:
            return idx[k]
    return None


def synth_scenario(seed, physics='p1', rate_mph=None, p_real_ls=0.6, p_real_w=0.75, max_tries=30):
    """Seeded synthetic scenario (training / frozen validation). Returns a Scenario or raises."""
    rs = np.random.default_rng(seed)
    tids = train_ids()
    for attempt in range(max_tries):
        n = int(rs.integers(140, 361))
        shape = (n, int(np.clip(n * rs.uniform(0.75, 1.33), 120, 380)))
        try:
            if rs.random() < p_real_ls:
                ls = fs.generate_landscape(shape, 90.0, kind='real', rng=int(rs.integers(2 ** 31)), fire_ids=tids)
            else:
                region = rs.choice(['west_dry', 'mixed_conifer', 'chaparral', 'great_basin', 'southeast'])
                ls = fs.generate_landscape(shape, 90.0, kind='parametric', region=str(region),
                                           rng=int(rs.integers(2 ** 31)))
            spin = 72 + 36
            if rs.random() < p_real_w:
                w = RD.random_weather(hours=int(HORIZON_H) + 24, rng=int(rs.integers(2 ** 31)), fire_ids=tids,
                                      spinup_h=spin)
                lon = float(w.attrs['lonlat'][0])
                w = w.copy()
                w['precip'] = w['precip'].fillna(0.0)
                for c in ('T', 'RH', 'wind_speed', 'wind_dir'):
                    w[c] = w[c].interpolate(limit_direction='both')
            else:
                clim = str(rs.choice(['west_dry_summer', 'california_autumn', 'southwest_spring',
                                      'northern_rockies_summer', 'southeast_spring']))
                lon = float(rs.uniform(-122, -95))
                d0 = pd.Timestamp('2019-06-01', tz='UTC') + pd.Timedelta(days=int(rs.integers(0, 120)))
                w = fs.generate_weather(d0, hours=spin + int(HORIZON_H) + 24, climate=clim, lon=lon,
                                        rng=int(rs.integers(2 ** 31)))
            start = _pm_start(w, lon, spin)
            if start is None:
                continue
            # grow the detection-time fire from a point ignition in the central half of the window
            fuel = np.asarray(ls['fuel'])
            ok = ((fuel < 91) | (fuel > 99)) & (fuel > 0)
            ny, nx = fuel.shape
            cand = np.argwhere(ok[ny // 4:3 * ny // 4, nx // 4:3 * nx // 4]) + [ny // 4, nx // 4]
            if not len(cand):
                continue
            r, c = cand[int(rs.integers(len(cand)))]
            grow_h = float(rs.uniform(6, 30))
            t_ig = start - pd.Timedelta(hours=grow_h)
            from .env import calibrated_params
            P = calibrated_params(physics)
            P['moisture_spinup_h'] = 48
            o = fs.simulate(ls, w, {'rowcol': [(int(r), int(c))]}, t_ig.floor('h'), start, params=P, crown='van_wagner',
                            spotting=False, return_perimeters=False,
                            rng=0)
            a = o['arrival_time']
            b0 = np.isfinite(a)
            ha = b0.sum() * 0.81
            if not (15 <= ha <= 6000):
                continue
            rate = rate_mph if rate_mph is not None else float(np.exp(rs.uniform(np.log(150), np.log(2000))))
            hid = dict(f_ros=float(np.exp(0.5 * rs.standard_normal())), f_ws=float(np.exp(0.25 * rs.standard_normal())),
                       d_wd=float(20.0 * rs.standard_normal()))
            meta = dict(kind='synth', seed=int(seed), attempt=attempt, ls_gen=ls.meta.get('generator', 'parametric'),
                        ls_src=ls.meta.get('source_fire'), w_src=w.attrs.get('source_fire', w.attrs.get('climate')),
                        grow_h=grow_h, a0_ha=float(ha))
            return Scenario(ls, w, b0, start, lon, rate, hid, physics=physics, meta=meta)
        except (RuntimeError, ValueError) as e:  # no clean window / weather; redraw
            last = e
            continue
    raise RuntimeError(f'no scenario for seed {seed}')
