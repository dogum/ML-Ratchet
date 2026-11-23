"""Law 4 config: BuildingSim's energy signature for single-family homes (annual site energy per floor area)."""
import glob, os
import numpy as np, pandas as pd

from _paths import ROOT  # the work folder (training/_paths.py)
NAME = 'bldg'
TITLE = 'Home energy signature'
VERSIONS = ['b1', 'b2']
VARS = ['H', 'C', 'A', 'Y']
VAR_INFO = {'H': {'label': 'heating degree days, base 18 °C', 'symbol': 'HDD', 'unit': '1000 K·d', 'range': [0.2, 6.0]},
            'C': {'label': 'cooling degree days, base 18 °C', 'symbol': 'CDD', 'unit': '1000 K·d', 'range': [0.0, 2.5]},
            'A': {'label': 'floor area', 'symbol': 'A', 'unit': '100 m²', 'range': [0.5, 4.0]},
            'Y': {'label': 'year built', 'symbol': 'Y', 'unit': 'year', 'range': [1920, 2020]}}
TARGET = {'name': 'EUI', 'label': 'site energy per floor area', 'symbol': 'EUI', 'unit': 'kWh/m²/yr'}
SEARCH_SECONDS = 360
SEARCH_KW = {}


def bank_ready(ver):
    return (len(glob.glob(f'{ROOT}/banks/bldg_{ver}/train_*.parquet')) >= 10
            and len(glob.glob(f'{ROOT}/banks/bldg_{ver}/val_*.parquet')) >= 3)


def X_of(df):
    return np.c_[df['hdd'].values / 1e3, df['cdd'].values / 1e3, df['area_m2'].values / 100.0, df['year_built'].values]


def load_bank(ver, split):
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{ROOT}/banks/bldg_{ver}/{split}_*.parquet'))],
                   ignore_index=True)
    ok = np.isfinite(X_of(df)).all(1) & np.isfinite(df.eui) & (df.eui > 0)
    return df[ok].reset_index(drop=True)


def target(df):
    return df['eui'].values.astype(float)


def recs_rows(split):
    """RECS 2020 single-family homes (primary_use residential_single) of one BuildingData split."""
    import buildingdata as bd
    r = bd.recs(split)
    r = r[(r.primary_use == 'residential_single') & (r.floor_area_m2 > 20) & (r.site_kWh > 0)]
    return pd.DataFrame({'id': r['id'].values, 'eui': (r.site_kWh / r.floor_area_m2).values,
                         'hdd': r.hdd18C_year.values, 'cdd': r.cdd18C_year.values, 'area_m2': r.floor_area_m2.values,
                         'year_built': ((r.year_built_lo + r.year_built_hi) / 2).values, 'system': r.system.values,
                         'heating_fuel': r.heating_fuel.values, 'division': r.division.values,
                         'housing_type': r.housing_type.values}).reset_index(drop=True)


def load_dev():
    return recs_rows('dev')


def dev_error(y, yh):
    return float(np.median(np.abs(yh / y - 1)))      # MdAPE as a fraction


def signature(df, coef):
    """Textbook rival: the degree-day energy signature (ASHRAE inverse-model style, fixed 18 C balance point):
    EUI = a + b HDD + c CDD."""
    a, b, c = coef
    return a + b * df['hdd'].values / 1e3 + c * df['cdd'].values / 1e3


def fit_signature(df):
    X = np.c_[np.ones(len(df)), df['hdd'].values / 1e3, df['cdd'].values / 1e3]
    return [float(v) for v in np.linalg.lstsq(X, df['eui'].values, rcond=None)[0]]
