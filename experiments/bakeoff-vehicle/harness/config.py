"""Vehicle Bake-off harness: paths and constants (single source of truth for every harness script).

Importing this module sets the one-thread environment, points RoadData at the processed data and puts RoadSim, RealData
and EnergyNet (v2.1 code and weights) on sys.path. As run in the cloud (23-24 Nov 2026): processed data in
/home/claude/road/data and the code as vendored byte copies under vendor/ (hashes in vendor/VENDOR.json). In this
repository (imported 24 Nov 2026): the repo's own shared/vehicle (data under $ML_RATCHET_VEHICLE/data, default
shared/vehicle/data) and models/energynet, all 140 vendored files equal to the committed blobs on import, unless a
vendor/ copy is restored next to this file.
"""
import os
import sys

HARNESS = os.path.dirname(os.path.abspath(__file__))
BAKEOFF = os.path.dirname(HARNESS)                        # experiments/bakeoff-vehicle (cloud: /home/claude/bakeoff/vehicle)
REPO = os.path.dirname(os.path.dirname(BAKEOFF))          # the repository root (cloud: /home/claude/mlr-repo, read-only)
VENDOR = os.path.join(HARNESS, 'vendor')
_VENDORED = os.path.isdir(os.path.join(VENDOR, 'vehicle', 'roadsim'))
VEHICLE = (os.path.join(VENDOR, 'vehicle') if _VENDORED else   # roadsim/, realdata/, checks/hybrid_specs.py, data/
           os.path.abspath(os.environ.get('ML_RATCHET_VEHICLE') or os.path.join(REPO, 'shared', 'vehicle')))
ROAD_DATA = os.path.join(VEHICLE, 'data')                 # processed RoadData (cloud: /home/claude/road/data)
ENET = os.path.join(VENDOR, 'energynet') if _VENDORED else os.path.join(REPO, 'models', 'energynet')  # v2.1 code, weights
WORK = os.path.join(HARNESS, 'work')                      # EnergyNet training scripts' work dir (real records live here)

for k, v in (('ML_RATCHET_VEHICLE', VEHICLE), ('ML_RATCHET_WORK', WORK),
             ('OMP_NUM_THREADS', '1'), ('MKL_NUM_THREADS', '1'), ('OPENBLAS_NUM_THREADS', '1'),
             ('NUMBA_NUM_THREADS', '1'), ('NUMBA_CACHE_DIR', os.path.join(HARNESS, '.numba')),
             ('PYTHONDONTWRITEBYTECODE', '1')):
    os.environ.setdefault(k, v)
sys.dont_write_bytecode = True
for p in (os.path.join(ENET, 'training'), os.path.join(VEHICLE, 'roadsim'), VEHICLE):
    if p not in sys.path:
        sys.path.insert(0, p)

BANK = os.path.join(HARNESS, 'bank')
REAL = os.path.join(WORK, 'real')                         # real records (dev in real/dev, exam built by exam.py only); cloud: harness/real
RESULTS = os.path.join(HARNESS, 'results') if os.path.isdir(os.path.join(HARNESS, 'results')) else os.path.join(BAKEOFF, 'results')  # repo: beside harness/
# MOCK exam (tests/test_exam_mock.sh only): the DEV records stand in for the exam and every result goes to tmp/, so the
# exam-mode code path (lock, guard, workers, aggregation) can be tested end to end without any exam data.
MOCK = os.environ.get('BAKEOFF_MOCK_EXAM') == '1'
if MOCK:
    RESULTS = os.path.join(HARNESS, 'tmp', 'mock_results')
    os.makedirs(RESULTS, exist_ok=True)


def real_dir(split):
    return os.path.join(REAL, 'dev' if (MOCK and split == 'exam') else split)

REGISTRY = os.path.join(HARNESS, 'registry')
LOGS = os.path.join(HARNESS, 'logs')
SPLITS = os.path.join(ROAD_DATA, 'proc', 'splits.json')
SPLITS_SHA256 = 'd917df78b099167ceadd7c5cd384a9fd9c45f2e14ca69fa3927c98b9b3233ad9'
ZTB_FULL_SPLITS = os.path.join(REPO, 'shared', 'vehicle', 'data', 'proc', 'hd', 'ztbus_full_splits.json')  # repo, tracked
ENET_EXAM3 = os.path.join(REPO, 'models', 'energynet', 'training', 'exam3.json')     # EnergyNet v2.1's published exam
ENET_RATCHET2 = os.path.join(REPO, 'models', 'energynet', 'training', 'ratchet2.json')

# ------------------------------------------------------------------ physics and the bank (EnergyNet v2.1's recipe)
PHYSICS = 'roadsim-physics-v2'          # pinned, as models/energynet/training/physics.py
DONOR_MODEL = 'v2'
TARGET_VERSION = 'h'                    # EnergyNet v2.1's frozen-set version: e (OBD label model + plug-in patch)
                                        # on banks with the SC03 / cold-FTP EPA test conditions and no idle-stop
FLEET_TRAIN = ('train', 2400, 201)      # (name, vehicles, seed): EnergyNet's train2 fleet
FLEET_VAL = ('val', 400, 202)           # EnergyNet's val2 fleet
BANK_VAL = ('val', 10, 777)             # (name, trips per vehicle, bank seed): EnergyNet's val2 bank
BANK_TRAIN = ('train', 12, 101)         # one of EnergyNet's three train banks (seeds 101-103); see HARNESS.md
VAL_TIER_SEED = 777                     # EnergyNet's frozen tiers for the val set
TRACE_MAX_S = 7200                      # stored trace length cap (s); HD synthetic drives are <= 7200 s
TRACE_CH = ['v', 'grade', 'elev', 'T_amb', 'wind_head', 'wind_cross', 'rho_air', 'surface']

# ------------------------------------------------------------------ scoring
QKEYS = ['q05', 'q25', 'q50', 'q75', 'q95']
QLEVELS = [0.05, 0.25, 0.50, 0.75, 0.95]
SEED = 20261123                         # bootstrap / showcase seed
NBOOT = 10000
LHV = 0.042                             # MJ/g gasoline (E10), RoadData's label definition
CNG_LHV = 0.0471
MI = 1609.344
