"""Law 3b config: GridSim's GB frequency-nadir law (depth of the dip after a trip)."""
import glob, os
import numpy as np, pandas as pd
from grid_real import events

from _paths import ROOT  # the work folder (training/_paths.py)
NAME = 'grid_nadir'
TITLE = 'GB frequency nadir'
VERSIONS = ['g1', 'g2']
VARS = ['L', 'E', 'D']
VAR_INFO = {'L': {'label': 'power lost', 'symbol': 'ΔP', 'unit': 'GW', 'range': [0.15, 2.0]},
            'E': {'label': 'system inertia', 'symbol': 'E', 'unit': 'GVA·s', 'range': [90, 420]},
            'D': {'label': 'demand', 'symbol': 'D', 'unit': 'GW', 'range': [16, 46]}}
TARGET = {'name': 'nadir depth', 'label': 'frequency drop to the nadir f0 - f_min', 'symbol': 'Δf', 'unit': 'Hz'}
SEARCH_SECONDS = 360


def bank_ready(ver):
    return (len(glob.glob(f'{ROOT}/banks/grid_nadir_{ver}/train_*.parquet')) >= 16
            and len(glob.glob(f'{ROOT}/banks/grid_nadir_{ver}/val_*.parquet')) >= 6)


def X_of(df):
    return np.c_[df['loss_MW'].values / 1e3, df['inertia_GVAs'].values, df['demand_MW'].values / 1e3]


def load_bank(ver, split):
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{ROOT}/banks/grid_nadir_{ver}/{split}_*.parquet'))],
                   ignore_index=True)
    df = df[df.detected]                       # GridData's catalogue rule: 10 s fall <= -0.15 Hz
    ok = np.isfinite(X_of(df)).all(1) & np.isfinite(df.df_nadir_Hz)
    return df[ok].reset_index(drop=True)


def target(df):
    return df['df_nadir_Hz'].values.astype(float)


def load_dev():
    ev = events('dev')
    return ev[np.isfinite(ev.loss_MW) & np.isfinite(ev.inertia_GVAs) & np.isfinite(ev.demand_MW)].reset_index(drop=True)


def dev_error(y, yh):
    # median, not mean: on 9 Aug 2019 (dev) later trips took the loss from 1,131 MW (the onset label) to 1,878 MW,
    # so that event's 1.24 Hz nadir is not a single-loss nadir and would dominate a mean over 4 events
    return float(np.median(np.abs(yh - y)))


def swing_ramp(L, E, k):
    """Textbook: swing equation with a primary response ramping to full over T: nadir depth = f0 L^2 T / (4 E R),
    with T/R lumped into one constant k (fitted on dev): df = k L^2 / E."""
    return k * np.asarray(L) ** 2 / np.asarray(E)
