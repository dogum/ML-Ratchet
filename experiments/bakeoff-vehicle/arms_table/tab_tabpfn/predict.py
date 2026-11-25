"""TabPFN yardstick arm (a lens, not a release). Built with PriorLabs-TabPFN.

TabPFN v2 (regressor, open weights, Prior Labs License 1.1: weights/LICENSE.txt) used IN CONTEXT: no training and
no gradient step. The context (ctx.npz) is simulated train-bank trips only: ZERO real labels. Inputs: a scoped subset
of the table track's 151 EnergyNet features (featsel.py: the top features of the GBM arm's gain ranking on the
simulated bank + architecture / class codes). Target: z = log(E / E_phys) (EnergyNet's correction). Output: TabPFN's
predictive distribution of z -> quantiles 5 / 25 / 50 / 75 / 95 %, pred = the median; E = E_phys x exp(z).

Needs torch (system) and the TabPFN package copied in pylib/ (tabpfn 10.2.0 + 4 small dependencies): numpy_only false.
One thread, CPU, fit_mode 'fit_with_cache' (the context's keys / values are computed once at import, ~10-30 s).
"""
import os
import sys

os.environ.setdefault('TABPFN_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TABPFN_NO_BROWSER', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(HERE, 'pylib') not in sys.path:
    sys.path.insert(0, os.path.join(HERE, 'pylib'))
import importlib.util  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
_spec = importlib.util.spec_from_file_location('tab_tabpfn_featsel', os.path.join(HERE, 'featsel.py'))
featsel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(featsel)
CFG = json.load(open(os.path.join(HERE, 'tp_config.json')))
_C = np.load(os.path.join(HERE, 'ctx.npz'))
IDX = _C['feature_index'].astype(int).tolist()
QS = [0.05, 0.25, 0.5, 0.75, 0.95]
CHUNK = 512
_M = None


def _model():
    global _M
    if _M is None:
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        nf = _C['X'].shape[1]
        m = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2, model_path=os.path.join(HERE, 'weights', 'tabpfn-v2-regressor.ckpt'),
            n_estimators=int(CFG.get('n_estimators', 1)), random_state=int(CFG.get('random_state', 0)), device='cpu',
            ignore_pretraining_limits=True, memory_saving_mode=True, fit_mode='fit_with_cache',
            categorical_features_indices=[nf - 2, nf - 1])
        m.fit(_C['X'], _C['z'])
        _M = m
    return _M


def predict(batch):
    X = featsel.scoped(np.asarray(batch['X'], np.float32), IDX).astype(np.float32)
    lE = np.log(np.asarray(batch['E_phys'], np.float64))
    m = _model()
    Q = np.zeros((len(X), 5))
    for s in range(0, len(X), CHUNK):
        o = m.predict(X[s:s + CHUNK], output_type='quantiles', quantiles=QS)
        Q[s:s + CHUNK] = np.stack([np.asarray(q, np.float64) for q in o], 1)
    Q = np.sort(Q, 1)
    out = {'pred': np.exp(lE + Q[:, 2])}
    for j, k in enumerate(['q05', 'q25', 'q50', 'q75', 'q95']):
        out[k] = np.exp(lE + Q[:, j])
    return out
