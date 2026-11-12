import _paths as _P  # portable paths (see _paths.py)
import sys, time, numpy as np
sys.path.insert(0, _P.SRC)
import gwphys as G
t = time.time(); ps = G.training_psds()
b, p = G.make_bank(8000, seed=11, psds=ps); np.save(_P.WORK + '/bank_train.npy', b); np.save(_P.WORK + '/par_train.npy', p)
print('train bank', b.shape, '%.0fs' % (time.time() - t), flush=True)
b, p = G.make_bank(1500, seed=22, psds=ps); np.save(_P.WORK + '/bank_val.npy', b); np.save(_P.WORK + '/par_val.npy', p)
np.save(_P.WORK + '/sky_train.npy', G.sky_table(200000, seed=3)); np.save(_P.WORK + '/sky_val.npy', G.sky_table(20000, seed=4))
print('done %.0fs' % (time.time() - t), flush=True)
