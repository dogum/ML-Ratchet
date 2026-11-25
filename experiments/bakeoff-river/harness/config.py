"""River Bake-off harness: paths, windows and constants (single source of truth for every harness script).

Importing this module sets the one-thread environment, points WaterData at the data root and puts WaterSim + WaterData
first on sys.path. As run in the cloud (22-23 Nov 2026) the data root was /home/claude/water and the code was a vendored
byte copy of the repo's shared/water (vendor/VENDOR.json). In this repository (imported 24 Nov 2026): the data root is
$ML_RATCHET_WATER, default shared/water; the code is shared/water itself (all 31 vendored files byte-identical on
import) unless a vendor/ copy is restored next to this file.
"""
import os
import sys

HARNESS = os.path.dirname(os.path.abspath(__file__))
BAKEOFF = os.path.dirname(HARNESS)                       # experiments/bakeoff-river (cloud: /home/claude/bakeoff/river)
REPO = os.path.dirname(os.path.dirname(BAKEOFF))         # the repository root (cloud: /home/claude/mlr-repo, read-only)
WATER_DATA_ROOT = os.path.abspath(os.environ.get('ML_RATCHET_WATER') or os.path.join(REPO, 'shared', 'water'))   # contains data/proc

for k, v in (('ML_RATCHET_WATER', WATER_DATA_ROOT), ('WATERSHED_ROOT', WATER_DATA_ROOT),
             ('OMP_NUM_THREADS', '1'), ('MKL_NUM_THREADS', '1'), ('OPENBLAS_NUM_THREADS', '1'),
             ('NUMBA_NUM_THREADS', '1'), ('NUMBA_CACHE_DIR', os.path.join(HARNESS, '.numba')),
             ('PYTHONDONTWRITEBYTECODE', '1')):
    os.environ.setdefault(k, v)
sys.dont_write_bytecode = True
VENDOR = os.path.join(HARNESS, 'vendor')
_CODE = ([VENDOR] if os.path.isdir(os.path.join(VENDOR, 'watersim')) else
         [os.path.join(REPO, 'shared', 'water', 'watersim'), os.path.join(REPO, 'shared', 'water')])
for _p in reversed(_CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BANK = os.path.join(HARNESS, 'bank')
RESULTS = os.path.join(HARNESS, 'results') if os.path.isdir(os.path.join(HARNESS, 'results')) else os.path.join(BAKEOFF, 'results')  # repo: beside harness/
REGISTRY = os.path.join(HARNESS, 'registry')
LOGS = os.path.join(HARNESS, 'logs')
SPLITS = os.path.join(WATER_DATA_ROOT, 'data', 'proc', 'splits.json')
SPLITS_SHA256 = '33f88ca0225180dcce979eceed26bf5157b92455e13bb7b78e135e9fd4d98c6b'
BASELINES = os.path.join(WATER_DATA_ROOT, 'data', 'proc', 'baselines')
NWM_DAILY = os.path.join(WATER_DATA_ROOT, 'data', 'proc', 'nwm', 'daily_local.parquet')
FLOWCAST_DIR = os.path.join(REPO, 'models', 'flowcast')

# ------------------------------------------------------------------ time windows (FlowCast / WaterData)
FREE = [('1951-01-01', '1988-09-30'), ('1999-10-01', '2015-09-30')]    # simulated-training forcing only
DEV = ('2000-10-01', '2014-09-30')        # dev gate, WY2001-2014; warm-up from 1999-10-01 (never exam)
MAIN = ('1989-10-01', '1999-09-30')       # main exam, WY1990-1999 (Caravan Q)
RECENT = ('2015-10-01', '2023-09-30')     # recent exam, WY2016-2023 (NWIS Q)
WARM = 365                                # warm-up days given before the first target day
PERIODS = {'dev': (DEV, 'caravan'), 'main': (MAIN, 'caravan'), 'recent': (RECENT, 'nwis')}
LEADS = (1, 3, 7)                         # gauged track leads (days)
MIN_OBS_DAYS = 365                        # a basin-period is scored when it has >= 365 observed days

# ------------------------------------------------------------------ inputs
DYN = ['P', 'PET', 'T_mean', 'T_min', 'T_max', 'SRAD', 'vp', 'wind']
ATTR_NAMES = ['area_km2', 'elev_mean_m', 'elev_range_m', 'slope_mkm', 'forest_frac', 'soil_porosity', 'ksat_cmh',
              'carbonate_frac', 'reservoir_storage_frac', 'irrigation_frac', 'urban_frac']
STATIC_NAMES = ['log_area', 'elev_mean_km', 'elev_range_km', 'log_slope_mkm', 'forest_frac', 'soil_porosity',
                'log_ksat_cmh', 'carbonate_frac', 'reservoir_frac', 'irrigation_frac', 'urban_frac',
                'log_p_mean', 'pet_mean', 'log_aridity', 'frac_snow', 'hp_freq', 'lp_freq', 't_mean_c']
QKEYS = ['q05', 'q25', 'q50', 'q75', 'q95']
QLEVELS = [0.05, 0.25, 0.50, 0.75, 0.95]
CLASSES = ['snow', 'arid', 'humid', 'regulated', 'large']

# ------------------------------------------------------------------ bank
PHYS = 'p1'                 # FlowCast's released physics (water-yield multiplier)
NDAYS = 1462                # stored days per run: 4 years + 1 day
N_TRAIN, N_VAL = 12000, 800
SEED_TRAIN, SEED_VAL = 1, 7  # FlowCast's released bank seeds (NOTEBOOK.md: simbank.py p1 val 800 7; train 12000 1)
SHARD = 400

SEED = 20261118             # bootstrap / showcase seed (WaterData's split seed)
NBOOT = 10000
