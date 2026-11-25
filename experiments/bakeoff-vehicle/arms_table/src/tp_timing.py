"""TabPFN timing / memory probe on the real table (sim rows only). usage: tp_timing.py NCTX NTEST NEST FITMODE [NFEAT]"""
import os
import resource
import sys
import time

os.environ.setdefault('TABPFN_DISABLE_TELEMETRY', '1')
import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, '/home/claude/bakeoff/vehicle/harness')
import bake  # noqa: E402

nctx, ntest, nest, mode = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
nfeat = int(sys.argv[5]) if len(sys.argv) > 5 else 151
B = bake.load_bank('train')
T = bake.draw_tiers(B, np.random.default_rng(5))
D = bake.sim_table(B, T)
ok = np.where(D['y'] > 0.02)[0]
rng = np.random.default_rng(0)
ci = rng.choice(ok, nctx, replace=False)
import featsel
IDX = featsel.ranked('/home/claude/bakeoff/vehicle/arms_table/tab_gbm/work/rungs/r0/train.json', nfeat - 2) if nfeat < 151 else None
sc = (lambda A: featsel.scoped(A, IDX)) if IDX else (lambda A: A)
X = sc(D['X'])
z = np.log(D['y'] / D['E_phys'])
from tabpfn import TabPFNRegressor  # noqa: E402
from tabpfn.constants import ModelVersion  # noqa: E402
m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, model_path='/home/claude/priorwars/weights/TabPFN-v2-reg/tabpfn-v2-regressor.ckpt',
                                               n_estimators=nest, random_state=0, device='cpu', ignore_pretraining_limits=True,
                                               memory_saving_mode=True, fit_mode=mode, categorical_features_indices=([nfeat - 2, nfeat - 1] if IDX else None))
t = time.process_time(); m.fit(X[ci], z[ci]); tf = time.process_time() - t
Bv = bake.load_bank('val'); Tv = bake.val_tiers(); Dv = bake.sim_table(Bv, Tv)
idx = bake.simval_rows()[:ntest]
Xt = sc(Dv['X'][idx]); zt = np.log(Dv['y'][idx] / Dv['E_phys'][idx])
t = time.process_time()
o = m.predict(Xt, output_type='quantiles', quantiles=[0.05, 0.25, 0.5, 0.75, 0.95])
tp = time.process_time() - t
med = np.asarray(o[2])
print(f'ctx {nctx} test {ntest} nest {nest} {mode} nfeat {nfeat}: fit {tf:.1f}s predict {tp:.1f}s = {tp / ntest * 1000:.1f} ms/row; '
      f'med|err| {np.median(np.abs(med - zt)):.4f}; maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB', flush=True)
