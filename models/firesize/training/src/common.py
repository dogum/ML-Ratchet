"""FireSize shared settings and helpers (paths, dev split, records)."""
import os, sys, json, hashlib
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # training/ (for _paths)
from _paths import WORK, TRAIN, MODEL, FIRE, FIRE_DATA, SRC, latest   # noqa: E402,F401  (puts shared/fire, shared/fire/firesim on sys.path)
os.environ.setdefault('NUMBA_CACHE_DIR', f'{WORK}/.numba')
import numpy as np, pandas as pd

ROOT = WORK                         # every output (data/, models/, results/, web/, release/, ratchet.json); was the sandbox's fire/firesize
DATA = f'{ROOT}/data'
FP = f'{FIRE_DATA}/data/proc'       # the platform's processed data (was the sandbox's fire/data/proc)
SEED = 20261119
RES = 90.0
WIN = 201                      # simulation window, cells (18.1 km at 90 m)
ACRE_HA = 0.404686
ESC_HA = 100 * ACRE_HA         # escape threshold, 40.47 ha
LARGE_HA = 1000 * ACRE_HA
REGION = json.load(open(f'{FP}/splits.json'))['meta']['region_map']
REGIONS = ['CA', 'NW', 'NR', 'GB', 'SW', 'RM', 'SP', 'SE', 'NE_MW']
CAUSES = ['Human', 'Natural', 'Missing data/not specified/undetermined']
FED = {'FS', 'BIA', 'BLM', 'NPS', 'FWS', 'TRIBE', 'DOD', 'BOR', 'DOE', 'OTHER', 'IA'}
WX = ['erc', 'bi', 'fm100', 'fm1000', 'vs', 'th', 'rmin', 'rmax', 'tmmx', 'tmmn', 'pr', 'vpd']


def h01(s, salt=''):
    return int(hashlib.sha256(f'{SEED}{salt}{s}'.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def fod_records(split=None):
    """FPA-FOD 2012-2020 CONUS with the frozen FireSize split, region, cause, agency class, discovery hour, size."""
    fod = pd.read_parquet(f'{FP}/fpafod/fod.parquet')
    fod = fod[fod.FIRE_YEAR >= 2012]
    sp = pd.read_parquet(f'{FP}/firesize_split.parquet')[['FOD_ID', 'split', 'exam_period']]
    d = fod.merge(sp, on='FOD_ID')
    if split is not None:
        d = d[d.split == split]
    d = d.copy()
    d['region'] = d.STATE.astype(str).map(REGION).fillna('NE_MW')
    d['cause'] = d.NWCG_CAUSE_CLASSIFICATION.astype(str)
    d['fed'] = d.NWCG_REPORTING_AGENCY.astype(str).isin(FED).astype(int)
    hh = pd.to_numeric(d.DISCOVERY_TIME.astype(str).str[:2], errors='coerce')
    mm = pd.to_numeric(d.DISCOVERY_TIME.astype(str).str[2:4], errors='coerce')
    d['hour'] = hh + mm.fillna(0) / 60.0
    d.loc[(d.hour < 0) | (d.hour >= 24.5), 'hour'] = np.nan
    d['size_ha'] = d.FIRE_SIZE.astype(float) * ACRE_HA
    d['escape'] = (d.FIRE_SIZE >= 100).astype(int)
    d['large'] = (d.FIRE_SIZE >= 1000).astype(int)
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, 5070, always_xy=True)
    d['x'], d['y'] = tr.transform(d.LONGITUDE.values, d.LATITUDE.values)
    return d.reset_index(drop=True)


def dev_half(fod_id):
    """Seeded dev halves: 'calib' (suppression / prior calibration, twins) and 'gate' (ratchet dev gate)."""
    return 'calib' if h01(fod_id, 'half') < 0.5 else 'gate'


def snap(v):
    return 15.0 + RES * np.round((np.asarray(v) - 15.0) / RES)
