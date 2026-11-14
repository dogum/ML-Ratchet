"""DemandCast shared definitions: paths, BAs, time conventions, loaders."""
import os, sys, json
import numpy as np, pandas as pd

from _paths import WORK, GRID, SRC  # noqa: E402  (sys.path for gridsim and griddata)

ROOT = WORK                  # was the sandbox workspace; outputs (data/, ckpt/, checks/, web/, release/) go to work/
DATA = f'{ROOT}/data'
import griddata as gd  # noqa: E402

BAS = ['ERCO', 'CISO', 'PJM', 'MISO', 'SWPP', 'ISNE', 'NYIS', 'BPAT']
# local standard time offset (h) used for day blocks; civil time zone for occupancy (DST-aware)
LST = {'ERCO': -6, 'CISO': -8, 'PJM': -5, 'MISO': -6, 'SWPP': -6, 'ISNE': -5, 'NYIS': -5, 'BPAT': -8}
TZ = {'ERCO': 'America/Chicago', 'CISO': 'America/Los_Angeles', 'PJM': 'America/New_York', 'MISO': 'America/Chicago',
      'SWPP': 'America/Chicago', 'ISNE': 'America/New_York', 'NYIS': 'America/New_York', 'BPAT': 'America/Los_Angeles'}
T0 = pd.Timestamp('2018-01-01', tz='UTC')
T1 = pd.Timestamp('2026-01-01', tz='UTC')
HOURS = pd.date_range(T0, T1, freq='h', inclusive='left')
NH = len(HOURS)
TRAIN_YEARS = (2018, 2020, 2022)
DEV_YEARS = (2019, 2021)
EXAM_YEARS = (2023, 2024, 2025)
ISSUE_HOUR_LST = 10      # forecast issued 10:00 local standard time on D-1 (like a BA day-ahead forecast)
HIST_H = 168             # demand history used: the 168 h before issue time
GAP_H = 14               # hours from issue to the start of day D (10:00 -> 24:00)


def hidx(ts):
    """Index into HOURS of a UTC timestamp."""
    return int((pd.Timestamp(ts).tz_convert('UTC') - T0) / pd.Timedelta('1h'))


def day_start_index(ba, day):
    """HOURS index of 00:00 local standard time of `day` (date-like)."""
    t = pd.Timestamp(day).tz_localize(None).normalize() - pd.Timedelta(hours=LST[ba])
    return hidx(t.tz_localize('UTC'))


def days_of_year(ba, year):
    """Target days (LST) of a calendar year whose full input window fits in HOURS."""
    out = []
    for d in pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D'):
        i0 = day_start_index(ba, d)
        if i0 - GAP_H - HIST_H >= 0 and i0 + 24 <= NH:
            out.append(d)
    return out


def load_bundle(ba):
    return dict(np.load(f'{DATA}/wx/{ba}.npz', allow_pickle=False))


def jdump(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o))
    json.dump(obj, open(path, 'w'), indent=1, default=conv)
