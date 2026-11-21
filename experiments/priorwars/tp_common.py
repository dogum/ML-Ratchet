"""Shared TabPFN helpers for Prior Wars. One thread, CPU, TabPFN-v2 open weights only."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TABPFN_NO_BROWSER", "1")
import numpy as np
import torch
torch.set_num_threads(1)
# Memory (shared machine, after the 18:00 incident): memory_saving_mode + low_memory fit and predictions in batches of
# 256 test rows. These change memory use only, not predictions (test rows attend only to the context).
import pw_paths as _P
ROOT = _P.WORK          # was the sandbox experiment folder priorwars/; the harness writes ROOT/<model>/ now
WEIGHTS = _P.WEIGHTS
REG_CKPT = os.path.join(WEIGHTS, "TabPFN-v2-reg", "tabpfn-v2-regressor.ckpt")
CLF_CKPT = os.path.join(WEIGHTS, "TabPFN-v2-clf", "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt")


def regressor(n_estimators=4, seed=0, **kw):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, model_path=REG_CKPT,
                                                   n_estimators=n_estimators, random_state=seed,
                                                   device="cpu", ignore_pretraining_limits=True,
                                                   memory_saving_mode=True, fit_mode="low_memory", **kw)
    return m


def classifier(n_estimators=4, seed=0, **kw):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    m = TabPFNClassifier.create_default_for_version(ModelVersion.V2, model_path=CLF_CKPT,
                                                    n_estimators=n_estimators, random_state=seed,
                                                    device="cpu", ignore_pretraining_limits=True,
                                                   memory_saving_mode=True, fit_mode="low_memory", **kw)
    return m


def predict_batched(m, X, bs=512, **kw):
    out = []
    for i in range(0, len(X), bs):
        out.append(m.predict(X[i:i + bs], **kw))
    return np.concatenate(out)
