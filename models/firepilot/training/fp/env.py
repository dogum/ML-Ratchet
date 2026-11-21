"""FirePilot environment: containment-line planning on FireSim (Phase 1) without editing it.

A scenario is a fire at its first afternoon detection (a burned mask), a landscape window, 72 h of hourly weather,
and a crew budget (metres of line per working hour). The planner chooses ONE line plan at detection time. Crews
build the line in shifts; each line cell blocks spread from the moment it is built (FireSim suppression lines with
per-cell build times) unless the fire would reach it with a flame length above the holding limit (NWCG hauling-chart
rule: lines do not hold where flames exceed ~2.4 m / 8 ft). The score is the area burned at 72 h.

Two worlds per scenario:
  truth    : the scenario's weather (real HRRR analysis on real fires) + dev-calibrated ROS factors x a hidden
             ROS factor f_ros ~ LN(0, 0.5)  (the Phase-1 replay ensemble spread).
  forecast : what the planner sees: the same weather with a wind-speed error x LN(0, 0.25) and a direction error
             N(0, 20 deg) (constant over the horizon), and the dev-calibrated ROS factors without the hidden factor.
"""
import os
import sys
import json

import _paths  # noqa: E402  (training/_paths.py: shared/fire and shared/fire/firesim on sys.path, NUMBA_CACHE_DIR under WORK)

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import firesim as fs
from firesim import replay as RP
from firesim.simulate import GROUPS as _FS_GROUPS
from .core import *  # noqa: F401,F403
from .core import GROUPS

CAL_FILE = _paths.CALIBRATION
PERTURB = dict(ros_sd=0.5, ws_sd=0.25, wd_sd_deg=20.0)
assert all(list(GROUPS[k]) == list(_FS_GROUPS[k]) for k in GROUPS)


def calibrated_params(physics='p1'):
    """Model physics. p0 = uncalibrated FireSim; p1 = dev-calibrated per-fuel-group ROS factors as stored."""
    if physics == 'p0':
        return {}
    cal = json.load(open(CAL_FILE))
    return RP.calibrated_params(cal)


class Scenario:
    """One planning problem. ls: FireSim Landscape; w: hourly weather (truth); burned0: bool mask at detection;
    start: UTC Timestamp of detection; lon: degrees (for local solar time); rate_mph: crew budget (m of line per
    working hour, two crews); hidden: dict(f_ros, f_ws, d_wd) (truth ROS factor, forecast wind errors)."""

    def __init__(self, ls, w, burned0, start, lon, rate_mph, hidden, physics='p1', horizon_h=HORIZON_H, meta=None):
        self.ls, self.w, self.burned0 = ls, w, np.asarray(burned0, bool)
        self.start = pd.Timestamp(start)
        if self.start.tz is None:
            self.start = self.start.tz_localize('UTC')
        self.end = self.start + pd.Timedelta(hours=horizon_h)
        self.lon, self.rate = float(lon), float(rate_mph)
        self.hidden = dict(hidden)
        self.horizon = horizon_h
        self.meta = dict(meta or {})
        self.res = float(ls.meta['res_m'])
        self.P = calibrated_params(physics)
        self.physics = physics
        base = fs.prepare(ls, w, self.start, self.end, self.P, 'nelson')
        self.pr_true = RP._perturbed(base, 1.0, 0.0, self.hidden.get('f_ros', 1.0))
        self.pr_fcst = RP._perturbed(base, self.hidden.get('f_ws', 1.0), np.radians(self.hidden.get('d_wd', 0.0)), 1.0)
        self.cost = cost_raster(ls['fuel'])
        self.clock = work_clock(self.start, self.lon, horizon_h)
        self._none = None

    # -- simulation ------------------------------------------------------------------------------------------
    def run(self, world='truth', lines=None, full=False):
        """lines: (idx, tb) flat cell indices and build times (inf = never built). Returns arrival time (h)."""
        pr = self.pr_true if world == 'truth' else self.pr_fcst
        sup = None
        if lines is not None and len(lines[0]):
            idx, tb = lines
            ok = np.isfinite(tb)
            idx, tb = idx[ok], np.ceil(tb[ok] / DT_BLOCK) * DT_BLOCK
            items = []
            for tv in np.unique(tb):
                m = np.zeros(self.burned0.size, bool)
                m[idx[tb == tv]] = True
                items.append({'mask': m.reshape(self.burned0.shape), 'time': float(tv)})
            sup = {'lines': items} if items else None
        o = fs.simulate(self.ls, None, {'mask': self.burned0}, self.start, self.end, crown='van_wagner',
                        spotting=False, suppression=sup, prepared=pr, return_perimeters=False, rng=0)
        if full:
            return o
        return o['arrival_time']

    def area_ha(self, arr):
        return float((np.isfinite(arr) & (arr <= self.horizon)).sum() * self.res ** 2 / 1e4)

    def none(self):
        """No-action truth run (cached): arrival, flame length (for the holding rule), area."""
        if self._none is None:
            o = self.run('truth', full=True)
            arr = o['arrival_time']
            fl = np.nan_to_num(o['rasters']['flame_length'], nan=0.0)
            self._none = dict(arr=arr, fl=fl, area=self.area_ha(arr))
        return self._none

    def a0(self):
        return float(self.burned0.sum() * self.res ** 2 / 1e4)

    # -- plans ---------------------------------------------------------------------------------------------
    def lines_for(self, fr, plan, fl_world=None):
        """Plan -> (idx, tb_effective, km_built, n_cells, n_overrun). Cells where the fire would arrive with flame
        length > FL_HOLD_M (in `fl_world`, default the truth no-action run) are built but do not hold."""
        idx, tb, km = build_times(fr, plan, self.cost, self.rate, self.clock, self.horizon)
        fl = self.none()['fl'] if fl_world is None else fl_world
        over = fl.ravel()[idx] > FL_HOLD_M
        tb_eff = np.where(over, np.inf, tb)
        return idx, tb_eff, km, len(idx), int((over & np.isfinite(tb)).sum())

    def evaluate(self, fr, plan, return_arr=False):
        """Truth outcome of a plan: area (ha), saved fraction of the no-action growth, km built."""
        nn = self.none()
        idx, tb, km, n, nover = self.lines_for(fr, plan)
        arr = self.run('truth', (idx, tb))
        a = self.area_ha(arr)
        g0 = nn['area'] - self.a0()
        saved = (nn['area'] - a) / g0 if g0 > 0 else 0.0
        out = dict(area=a, saved=saved, km=km, n_line=n, n_overrun=nover, area_none=nn['area'])
        if return_arr:
            out['arr'] = arr
        return out


