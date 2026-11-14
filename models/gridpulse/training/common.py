"""Shared paths and helpers for GridPulse. (Paths go through _paths.py: outputs in WORK, the platform in GRID.)"""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
from _paths import WORK, GRID, LAB, SRC   # puts GRID/gridsim and GRID on sys.path
import numpy as np
import pandas as pd
ROOT = WORK
DATA = f'{ROOT}/data'; FIGS = f'{ROOT}/figs'; CKPT = f'{ROOT}/ckpt'
GB_DEV = ['GB_20190809T155233', 'GB_20190701T072734', 'GB_20190711T075329', 'GB_20191203T200030']


def real_event(eid, pre=60, post=120):
    """(t rel. to onset [s], f [Hz]) at 1 s for a GB event, onset = griddata t0_utc."""
    import griddata as gd
    meta, t, f = gd.event_trace(eid)
    off = (pd.Timestamp(meta['event_t0_utc']) - pd.Timestamp(meta['start_utc'])).total_seconds()
    tr = t - off
    m = (tr >= -pre) & (tr <= post)
    return tr[m], f[m]
