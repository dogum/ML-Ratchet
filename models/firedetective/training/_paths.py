"""Where FireDetective's training scripts read and write.

The scripts were first run on 18-19 Nov 2026 in a cloud sandbox workspace (fire/firedetective/: code in src/, and
data/, models/, out/, logs/, release/, web/ beside it), next to the Wildfire Phase 1 tree in fire/ (firesim/,
firedata/, checks/). Nothing was re-run on import. In this repository:

    SRC        this folder (training/): the scripts as run, the page template (template.html) and its network port
               (fdnet.js), the shipped ratchet log (ratchet.json) and the shipped result JSONs (results/, the old
               out/ folder)
    MODEL      models/firedetective/ (firedetective.py, feats.py, pipeline.py, weights/, examples/)
    REPO       the repository root
    WORK       models/firedetective/work/ (git-ignored), or $ML_RATCHET_WORK/firedetective/: the old workspace root
               (common.ROOT). data/bank/ (simulated banks), data/real/ (dev and exam records, LANDFIRE exports),
               models/ (checkpoints, ensembles, calib_<rung>.json), out/, logs/, tmp/, release/, web/,
               ratchet.json, .numba/
    FIRE_CODE  the shared fire platform code, always from the repository: shared/fire/ (firesim/firesim,
               firedata, checks/ with replay_calibration.json). shared/fire and shared/fire/firesim go on sys.path
               so `import firesim` and `import firedata` work.
    FIRE       the fire data root: $ML_RATCHET_FIRE, default shared/fire (firedata reads the same variable). The
               real tables (FEDS-MTBS progression, VIIRS pixels, gridMET, Globe-LFMC) are not in the repository:
               rebuild them with `mlratchet data fire` before any script that touches real fires.

Re-running a script writes new copies under WORK; the shipped JSONs in SRC stay as they were. The scripts that read
a shipped result (build_page.py, make_text.py, make_page_text.py, calibrate.py, exam.py score) take the WORK copy
if a re-run wrote one, else the shipped copy (latest()). Inference (firedetective.py), demo.py and the instrument
export need none of this.
"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
MODEL = os.path.dirname(SRC)
REPO = os.path.dirname(os.path.dirname(MODEL))
WORK = os.path.abspath(os.path.join(os.environ['ML_RATCHET_WORK'], 'firedetective')
                       if os.environ.get('ML_RATCHET_WORK') else os.path.join(MODEL, 'work')).replace(os.sep, '/')
FIRE_CODE = os.path.join(REPO, 'shared', 'fire').replace(os.sep, '/')
FIRE = os.path.abspath(os.environ.get('ML_RATCHET_FIRE') or FIRE_CODE).replace(os.sep, '/')
CHECKS = f'{FIRE_CODE}/checks'

for p in (FIRE_CODE, f'{FIRE_CODE}/firesim', SRC):   # import firesim, firedata and the scripts here
    if p not in sys.path:
        sys.path.insert(1, p)

# shipped name in SRC -> name a re-run writes under WORK
SHIPPED = {'out/exam.json': 'results/exam.json', 'out/baselines_dev.json': 'results/baselines_dev.json',
           'out/calib_feds.json': 'results/calib_feds.json', 'out/twin_v1.json': 'results/twin_v1.json',
           'out/twin_waf.json': 'results/twin_waf.json', 'models/calib_R6.json': 'results/calib_R6.json',
           'ratchet.json': 'ratchet.json'}


def latest(rel):
    """A result written under WORK by a re-run if there is one, else the shipped copy in SRC (if shipped)."""
    w = f'{WORK}/{rel}'
    if os.path.exists(w) or rel not in SHIPPED:
        return w
    return f'{SRC}/{SHIPPED[rel]}'
