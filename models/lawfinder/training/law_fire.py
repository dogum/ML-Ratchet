"""Law 2 config: FireSim's daily growth law (new burned area in the next 24 h)."""
import glob, os
import numpy as np, pandas as pd

from _paths import ROOT  # the work folder (training/_paths.py)
NAME = 'fire'
TITLE = 'Daily fire growth'
VERSIONS = ['f1', 'f2']
VARS = ['P', 'W', 'M', 'R0']
VAR_INFO = {
    'P': {'label': 'current perimeter (equivalent circle, 2·sqrt(pi·A))', 'symbol': 'P', 'unit': 'km', 'range': [0.5, 60]},
    'W': {'label': 'daily mean 10 m wind', 'symbol': 'U', 'unit': 'm/s', 'range': [0.5, 10]},
    'M': {'label': '100-h dead fuel moisture', 'symbol': 'M', 'unit': '%', 'range': [2, 25]},
    'R0': {'label': 'fuel speed: Rothermel head ROS of the group at a standard day', 'symbol': 'R₀', 'unit': 'm/min',
           'range': [0.7, 13]},
}
TARGET = {'name': 'log10 new area', 'label': 'log10(new area km² + 1 cell)', 'symbol': 'log₁₀ G'}
CELL = 0.0081                        # one 90 m cell in km2 (FireData's +1 cell convention)
SEARCH_SECONDS = 420
REP = {'GR': 'GR2', 'GS': 'GS2', 'SH': 'SH5', 'TU': 'TU5', 'TL': 'TL6'}     # most common model per group on dev landscapes
STD = dict(m1=6, m10=7, m100=8, mherb=90, mwoody=120, wind_speed=4.0, wind_height='10m', slope=0)
_R0 = None


def r0_table():
    global _R0
    if _R0 is None:
        import firesim as fs
        _R0 = {g: float(fs.behave(m, **STD)['ros_mmin']) for g, m in REP.items()}
    return _R0


def perim(a_km2):
    return 2 * np.sqrt(np.pi * np.maximum(np.asarray(a_km2, float), CELL))


def X_of(df):
    r0 = r0_table()
    return np.c_[perim(df['area0_km2']), df['wind_ms'].values, df['fm100'].values,
                 df['fuel_group'].map(r0).values]


def bank_ready(ver):
    return (len(glob.glob(f'{ROOT}/banks/fire_{ver}/train_*.parquet')) >= 30
            and len(glob.glob(f'{ROOT}/banks/fire_{ver}/val_*.parquet')) >= 12)


def load_bank(ver, split):
    fs_ = sorted(glob.glob(f'{ROOT}/banks/fire_{ver}/{split}_*.parquet'))
    df = pd.concat([pd.read_parquet(f) for f in fs_], ignore_index=True)
    df = df[df.fuel_group.isin(list(REP))]
    ok = np.isfinite(X_of(df)).all(1) & np.isfinite(df['new_km2'])
    return df[ok].reset_index(drop=True)


def target(df):
    return np.log10(df['new_km2'].values + CELL)


def real_rows(split):
    """FireData fire-day rows (issue PM -> next PM) with next-day weather (gridMET d1) and the fire's fuel group."""
    import firedata as fd
    from _paths import FIRE
    fd_ = pd.read_parquet(f'{FIRE}/data/proc/baselines/fire_days_{split}.parquet')
    fires = pd.read_parquet(f'{FIRE}/data/proc/fires.parquet')[['Event_ID', 'fuel_group', 'region', 'split']]
    fd_ = fd_.merge(fires, left_on='fire_id', right_on='Event_ID', how='left')
    assert (fd_['split'] == split).all()
    df = pd.DataFrame({'fire_id': fd_.fire_id, 'k': fd_.k, 'area0_km2': fd_.area_km2, 'new_km2': fd_.new_obs_km2,
                       'wind_ms': fd_.vs_d1, 'fm100': fd_.fm100_d1, 'fuel_group': fd_.fuel_group,
                       'region': fd_.region, 'year': fd_.year, 'flinelen_km': fd_.flinelen_km})
    ok = df.fuel_group.isin(list(REP)) & np.isfinite(df.new_km2) & np.isfinite(df.wind_ms) & np.isfinite(df.fm100)
    return df[ok].reset_index(drop=True)


def load_dev():
    return real_rows('dev')


def dev_error(y, yh):
    return float(np.mean(np.abs(yh - y)))          # mean |log10 ratio|


# --- textbook rival: Rothermel head ROS (representative model of the group, the day's wind and 100-h moisture,
# all dead classes at that moisture) x perimeter x 24 h, times a constant k
def rothermel_growth(df, k=1.0):
    import firesim as fs
    out = np.empty(len(df))
    cache = {}
    for i, (g, w, m, p) in enumerate(zip(df.fuel_group, df.wind_ms, df.fm100, perim(df.area0_km2))):
        key = (g, round(float(w), 1), round(float(m), 1))
        if key not in cache:
            cache[key] = float(fs.behave(REP[g], m1=key[2], m10=key[2], m100=key[2], mherb=90, mwoody=120,
                                         wind_speed=key[1], wind_height='10m', slope=0)['ros_mmin'])
        out[i] = cache[key] * p * 1.44                     # m/min x km x 1440 min / 1000 -> km2
    return k * out


def fit_k(raw, y_log):
    """k minimising mean |log10 ratio| (a median in log space)."""
    return float(10 ** np.median(y_log - np.log10(raw + CELL)))
