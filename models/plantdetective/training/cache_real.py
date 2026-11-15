"""Cache real CAMPD unit records (dev now; exam only at the final exam) as npz: op, P, HI, meas, t."""
from common import *
import numpy as np, pandas as pd
from realdata import units, record
split = sys.argv[1]
out = f'{DATA}/real_{split}'; os.makedirs(out, exist_ok=True)
for uid in units()[units().split == split].unit_id:
    f = f'{out}/{uid.replace("*", "s")}.npz'
    if os.path.exists(f): continue
    r = record(uid)
    if r.empty: print('empty', uid); continue
    np.savez_compressed(f, op=r.op.values.astype(np.float32), P=r.P.values.astype(np.float32),
                        HI=r.HI.values.astype(np.float32), meas=r.meas.values, t=epoch_s(r.index))
    print(uid, len(r), flush=True)
