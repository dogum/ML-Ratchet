"""FirePilot pipeline for your own fire (needs FireSim, the ML Ratchet Wildfire simulator, for the forecast run).

    from pipeline import plan_fire
    plan, info, ctx = plan_fire(landscape, weather, burned_mask, start_utc, lon, rate_m_per_h, firesim_path=...)

landscape: firesim.Landscape (90 m); weather: hourly DataFrame in the FireSim contract (your FORECAST);
burned_mask: bool raster of the detected fire on the landscape grid; start_utc: detection time; lon: degrees
(for local solar shift hours); rate_m_per_h: metres of line per working hour for two crews.
Physics: the dev-calibrated FireSim replay factors (shared/fire/checks/replay_calibration.json), crown fire on, no
spotting. FireSim is the ML Ratchet Wildfire platform (shared/fire/firesim); by default plan_fire imports it from this
repository, and firesim_path / calibration point it elsewhere. Only plan_fire needs it: without FireSim you can still
plan from stored arrays, firepilot.Context(burned0, arr_f, fl_f, cost, rate, clock, res), as demo.py does.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import firepilot as fp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIRE = os.path.join(HERE, '..', '..', 'shared', 'fire')    # the Wildfire platform code in ML Ratchet


def plan_fire(landscape, weather, burned_mask, start_utc, lon, rate_m_per_h, firesim_path=os.path.join(FIRE, 'firesim'),
              calibration=os.path.join(FIRE, 'checks', 'replay_calibration.json'), horizon_h=72.0, seed=0):
    sys.path.insert(0, firesim_path)
    import firesim as fs
    from firesim import replay as RP
    start = pd.Timestamp(start_utc)
    start = start.tz_localize('UTC') if start.tz is None else start
    # v1.0 was trained and graded in FireSim physics p1; keep it there (p2, 22 Nov 2026, is FireSim's default)
    P = dict(RP.calibrated_params(json.load(open(calibration))), physics='p1')
    o = fs.simulate(landscape, weather, {'mask': np.asarray(burned_mask, bool)}, start,
                    start + pd.Timedelta(hours=horizon_h), crown='van_wagner', spotting=False, params=P,
                    return_perimeters=False)
    fl = np.nan_to_num(o['rasters']['flame_length'], nan=0.0)
    clock = fp.work_clock(start, lon, horizon_h)
    ctx = fp.Context(np.asarray(burned_mask, bool), o['arrival_time'], fl, fp.cost_raster(np.asarray(landscape['fuel'])),
                     float(rate_m_per_h), clock, float(landscape.meta['res_m']), weather, horizon_h)
    net = fp.ValueNet(os.path.join(HERE, 'firepilot_weights.npz'))
    plan, info = fp.plan_value(ctx, net, np.random.default_rng(seed))
    idx, tb, km = fp.build_times(ctx.fr, plan, ctx.cost, ctx.rate, ctx.clock, horizon_h)
    info.update(line_cells=idx, build_time_h=tb, km_built=km)
    return plan, info, ctx
