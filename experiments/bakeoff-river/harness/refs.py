"""Reference rows (not arms): daily series of WaterData's baselines, scored by the harness with score.py exactly like
the arms. Reproduces WaterData's run_baselines_flow.py (checked on dev basins by tests/check_refs.py).

  udev         ungauged NN regionalisation, 5 dev donors' calibrated GR4J+snow on the target forcing   [ungauged]
  uk12         same, donors from the other 11 folds                                                    [ungauged]
  gr4j         GR4J+snow calibrated on the basin's own 1999-2008 flow                                  [gauged, simulation]
  nwm          NOAA NWM v3.0 retrospective (recent exam clipped to WY2016-2022, where NWM ends)        [operational]
  clim         day-of-year mean of own 1999-2008 Caravan flow, 31-day circular smoothing             [gauged, dumb]
  persistence  observed flow `lead` days earlier                                                      [gauged forecast]
FlowCast v1.0 (and its nowcast) is a registered reference ARM (arms_ref/flowcast_v1), not a row here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REF_ROWS = {
    'udev': {'label': 'NN regionalisation, dev donors (U-dev)', 'protocol': 'ungauged'},
    'uk12': {'label': 'NN regionalisation, 12-fold donors (U-k12)', 'protocol': 'ungauged'},
    'gr4j': {'label': 'GR4J + snow, calibrated per basin', 'protocol': 'gauged-calibration'},
    'nwm': {'label': 'NOAA NWM v3.0 retrospective', 'protocol': 'operational'},
    'clim': {'label': 'Climatology, own 1999-2008 day-of-year mean', 'protocol': 'gauged-calibration'},
    'persistence': {'label': 'Persistence (observed flow lead days earlier)', 'protocol': 'gauged'},
}
CAL = ('1999-10-01', '2008-09-30')
REC_NWM_END = '2022-09-30'
_CACHE = {}


def _sims(name):
    if name not in _CACHE:
        fn = {'udev': 'nn_dev', 'uk12': 'nn_k12', 'gr4j': 'gr4j_gauged'}[name]
        _CACHE[name] = pd.read_parquet(os.path.join(C.BASELINES, 'sims', f'{fn}.parquet'))
    return _CACHE[name]


def _nwm():
    if 'nwm' not in _CACHE:
        _CACHE['nwm'] = pd.read_parquet(C.NWM_DAILY)
    return _CACHE['nwm']


def clim_series(qcal, index):
    d = qcal.groupby(qcal.index.dayofyear).mean().reindex(range(1, 367)).interpolate(limit_direction='both')
    v = np.concatenate([d.values[-15:], d.values, d.values[:15]])
    sm = pd.Series(np.convolve(v, np.ones(31) / 31, mode='valid'), index=range(1, 367))
    return pd.Series(sm.reindex(index.dayofyear).values, index=index)


def reference(name, bid, period, obs_full=None, lead=1):
    """Daily reference series (mm/day) over the period's target days, or None if the reference has no data.
    obs_full: the scored observed series (needed for persistence; full record incl. the days before the window).
    The caller must hold the right to read this basin-period (dev, or exam under the guard)."""
    import waterdata as wd
    (s0, s1), src = C.PERIODS[period]
    idx = pd.date_range(s0, s1)
    if name in ('udev', 'uk12', 'gr4j'):
        S = _sims(name)
        return S[bid].reindex(idx).astype(float) if bid in S.columns else None
    if name == 'nwm':
        g = bid.split('_')[1]
        N = _nwm()
        if g not in N.columns:
            return None
        s = (N[g] * 86.4 / wd.spec(bid)['area_km2']).reindex(idx)
        if period == 'recent':
            s[s.index > pd.Timestamp(REC_NWM_END)] = np.nan
        return s.astype(float)
    if name == 'clim':
        qc = wd.streamflow(bid, 'caravan').Q_mm.loc[CAL[0]:CAL[1]]
        return clim_series(qc, idx)
    if name == 'persistence':
        if obs_full is None:
            obs_full = wd.streamflow(bid, src).Q_mm
        return obs_full.shift(lead, freq='D').reindex(idx).astype(float)
    raise ValueError(name)
