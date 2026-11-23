"""Where LawFinder's training scripts read and write.

The scripts were first run in a cloud sandbox folder (lawfinder/ with src/, vendor/, banks/, results/, logs/, web/,
release/), next to the platform workspaces (water/, fire/, grid/, bldg/). In the repository:

    SRC       this folder (training/): the scripts
    WORK      models/lawfinder/work/ (git-ignored) or $ML_RATCHET_WORK/lawfinder — what the scripts called ROOT:
              banks/, results/, logs/, web/, release/, splits.json
    MODEL     models/lawfinder/ (lawfinder.py, lawfinder_laws.json, splits.json)
    Platforms the code always comes from the repository (shared/water, shared/fire, shared/grid, shared/building,
              shared/vehicle; the sandbox used read-only copies in vendor/). Data roots can point elsewhere:
              $ML_RATCHET_WATER, $ML_RATCHET_FIRE, $ML_RATCHET_GRID, $ML_RATCHET_BUILDING, $ML_RATCHET_VEHICLE.
    other_work(id)  another model's work folder (FlowCast's, FireCast's and GridPulse's exam outputs are read from
              there for the network rows of the exams).

On first use the shipped records are copied into WORK if WORK has none: training/results/* -> WORK/results/,
training/{readings,limits,next}.json, fire_ros.json, credits.html, template.html -> WORK/web/, splits.json -> WORK/.
So build_card.py, build_page.py and build_release.py run on the shipped results, and a re-run writes new copies under
WORK without touching the shipped ones. Nothing here writes inside the repository.

Run order per law: <law>_bank.py (or run_*_banks.sh) -> fit_law.py <law> -> exam_<law>.py (once) -> webdata.py,
build_page.py, build_release.py, build_card.py.
"""
import os
import shutil
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(SRC)
REPO = os.path.dirname(os.path.dirname(MODEL))
WORK = os.path.abspath(os.path.join(os.environ['ML_RATCHET_WORK'], 'lawfinder') if os.environ.get('ML_RATCHET_WORK')
                       else os.path.join(MODEL, 'work')).replace(os.sep, '/')
ROOT = WORK
SHARED = os.path.join(REPO, 'shared')


def _root(env, name):
    return os.path.abspath(os.environ.get(env) or os.path.join(SHARED, name)).replace(os.sep, '/')


WATER = _root('ML_RATCHET_WATER', 'water')
FIRE = _root('ML_RATCHET_FIRE', 'fire')
GRID = _root('ML_RATCHET_GRID', 'grid')
BUILDING = _root('ML_RATCHET_BUILDING', 'building')
VEHICLE = _root('ML_RATCHET_VEHICLE', 'vehicle')


def other_work(model_id):
    w = os.environ.get('ML_RATCHET_WORK')
    return os.path.abspath(os.path.join(w, model_id) if w else os.path.join(REPO, 'models', model_id, 'work')).replace(os.sep, '/')


for p in (os.path.join(SHARED, 'water'), os.path.join(SHARED, 'water', 'watersim'),
          os.path.join(SHARED, 'fire'), os.path.join(SHARED, 'fire', 'firesim'),
          os.path.join(SHARED, 'grid'), os.path.join(SHARED, 'grid', 'gridsim'),
          os.path.join(SHARED, 'building'), os.path.join(SHARED, 'building', 'buildingsim'),
          os.path.join(SHARED, 'vehicle'), os.path.join(SHARED, 'vehicle', 'roadsim')):
    if p not in sys.path:
        sys.path.append(p)
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def import_gpsim():
    """GridPulse's GB event sampler (models/gridpulse/training/gpsim.py), imported with GridPulse's own _paths.py
    (both modules are called _paths; the sandbox patched a vendored copy instead)."""
    import importlib
    gp = os.path.join(REPO, 'models', 'gridpulse', 'training')
    mine = sys.modules.pop('_paths', None)
    sys.path.insert(0, gp)
    try:
        return importlib.import_module('gpsim')
    finally:
        sys.path.remove(gp)
        if mine is not None:
            sys.modules['_paths'] = mine

for d in ('banks', 'results', 'logs', 'web', 'release', 'tmp'):
    os.makedirs(os.path.join(WORK, d), exist_ok=True)


def _seed():
    res = os.path.join(SRC, 'results')
    for f in os.listdir(res) if os.path.isdir(res) else []:
        dst = os.path.join(WORK, 'results', f)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(res, f), dst)
    for f in ('readings.json', 'limits.json', 'next.json', 'fire_ros.json', 'credits.html', 'template.html'):
        src, dst = os.path.join(SRC, f), os.path.join(WORK, 'web', f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
    if not os.path.exists(os.path.join(WORK, 'splits.json')):
        shutil.copy2(os.path.join(MODEL, 'splits.json'), os.path.join(WORK, 'splits.json'))


_seed()

if __name__ == '__main__':      # the shell wrappers ask for WORK
    print(WORK)
