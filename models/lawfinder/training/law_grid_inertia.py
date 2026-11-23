"""Law 3a config: GridSim's GB inertia law (system inertia from demand and IBR share)."""
import glob, os
import numpy as np, pandas as pd
from grid_real import events

from _paths import ROOT  # the work folder (training/_paths.py)
NAME = 'grid_inertia'
TITLE = 'GB system inertia'
VERSIONS = ['i1']
VARS = ['D', 's']
VAR_INFO = {'D': {'label': 'demand (transmission generation)', 'symbol': 'D', 'unit': 'GW', 'range': [15, 50]},
            's': {'label': 'inverter-based share (wind, solar, imports)', 'symbol': 's', 'unit': '-', 'range': [0.05, 0.8]}}
TARGET = {'name': 'inertia', 'label': 'system inertia E', 'symbol': 'E', 'unit': 'GVA·s'}
SEARCH_SECONDS = 240


def bank_ready(ver):
    return (len(glob.glob(f'{ROOT}/banks/grid_inertia_{ver}/train_*.parquet')) >= 12
            and len(glob.glob(f'{ROOT}/banks/grid_inertia_{ver}/val_*.parquet')) >= 4)


def X_of(df):
    return np.c_[df['demand_MW'].values / 1e3, df['ibr_share'].values]


def load_bank(ver, split):
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{ROOT}/banks/grid_inertia_{ver}/{split}_*.parquet'))],
                   ignore_index=True)
    ok = np.isfinite(X_of(df)).all(1) & (df.inertia_GVAs > 0)
    return df[ok].reset_index(drop=True)


def target(df):
    return df['inertia_GVAs'].values.astype(float)


def load_dev():
    ev = events('dev')
    return ev[np.isfinite(ev.inertia_GVAs) & np.isfinite(ev.demand_MW)].reset_index(drop=True)


def dev_error(y, yh):
    return float(np.mean(np.abs(yh / y - 1)))
