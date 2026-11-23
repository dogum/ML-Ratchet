"""Law 1 config: WaterSim's long-term water balance (runoff ratio)."""
import glob, os
import numpy as np, pandas as pd

from _paths import ROOT  # the work folder (training/_paths.py)
NAME = 'water'
TITLE = 'Long-term runoff ratio'
VERSIONS = ['w1', 'w2']
VARS = ['phi', 'fs', 'sync', 'S']          # identifiers used in expressions / JS
VAR_INFO = {
    'phi': {'label': 'aridity PET/P', 'symbol': 'φ', 'unit': '-', 'range': [0.2, 5.0]},
    'fs': {'label': 'snow fraction of P', 'symbol': 'f_s', 'unit': '-', 'range': [0.0, 0.7]},
    'sync': {'label': 'P-PET seasonal synchrony (corr. of monthly means)', 'symbol': 'δ', 'unit': '-', 'range': [-1, 1]},
    'S': {'label': 'soil storage index S_max / annual P', 'symbol': 'S', 'unit': '-', 'range': [0.02, 0.4]},
}
TARGET = {'name': 'rr', 'label': 'runoff ratio Q/P', 'symbol': 'Q/P'}
SEARCH_SECONDS = 420


def bank_ready(ver):
    return (len(glob.glob(f'{ROOT}/banks/water_{ver}/train_*.parquet')) >= 32
            and len(glob.glob(f'{ROOT}/banks/water_{ver}/val_*.parquet')) >= 12)


def X_of(df):
    return np.c_[df['aridity'].values, df['frac_snow'].values, df['sync'].values,
                 df['soil_proxy'].values / (365.0 * df['p_mean'].values)]


def load_bank(ver, split):
    fs = sorted(glob.glob(f'{ROOT}/banks/water_{ver}/{split}_*.parquet'))
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    ok = np.isfinite(df['rr']) & (df['rr'] >= 0) & (df['rr'] < 2.0) & np.isfinite(X_of(df)).all(1)
    return df[ok].reset_index(drop=True)


def target(df):
    return df['rr'].values.astype(float)


def load_dev():
    df = pd.read_parquet(f'{ROOT}/results/water_dev.parquet')
    return df[np.isfinite(df['rr'])].reset_index(drop=True)


def dev_error(y, yh):
    return float(np.mean(np.abs(yh - y)))


# --- the textbook rival: Budyko with Fu's curve, runoff ratio = 1 - E/P = (1 + phi^w)^(1/w) - phi
def fu_rr(phi, w):
    phi = np.asarray(phi, float)
    return (1 + phi ** w) ** (1 / w) - phi


def fit_fu(phi, rr):
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda w: np.mean((fu_rr(phi, w) - rr) ** 2), bounds=(1.05, 8.0), method='bounded')
    return float(r.x)


def seeds():
    # Fu's form planted in the first population (so the search can find it and refine it)
    t = ('-', ('^', ('+', ('c', 0), ('^', ('x', 0), ('c', 1))), ('inv', ('c', 2))), ('x', 0))
    return [(t, [1.0, 2.6, 2.6])]
