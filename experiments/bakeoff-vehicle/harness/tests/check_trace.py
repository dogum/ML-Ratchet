"""Trace-track consistency: the physics estimate and the route vector recomputed from a trace-track batch (trace +
spec + rl) against the table track's E_phys and R, on sim val and on every dev set. -> results/check_trace.json"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
import numpy as np
import bake, features as F


def compare(bt, btab):
    e = bake.e_phys_from_trace(bt)
    r = np.abs(np.log(e / btab['E_phys']))
    R = np.stack([F.route_vec(bake.prep_trace(t)) for t in bt['traces']])
    dR = np.abs(R - btab['X'][:, :F.NR])
    return r, dR


out = {}
B = bake.load_bank('val'); T = bake.val_tiers(); idx = bake.simval_rows()[:400]
bt, _ = bake.sim_batch(B, T, idx, 'trace'); bb, _ = bake.sim_batch(B, T, idx, 'table')
r, dR = compare(bt, bb)
out['simval_400'] = dict(ephys_logdiff_median=float(np.median(r)), ephys_logdiff_p99=float(np.percentile(r, 99)),
                         ephys_logdiff_max=float(r.max()), R_absdiff_max=float(dR.max()), R_absdiff_p99=float(np.percentile(dR, 99)))
for name in bake.SETS:
    gt = bake._real_batches('dev', 'trace', sets=(name,), chunk=200); gb = bake._real_batches('dev', 'table', sets=(name,), chunk=200)
    _, _, bt = next(gt); _, _, bb = next(gb)
    r, dR = compare(bt, bb)
    out[f'dev_{name}_200'] = dict(ephys_logdiff_median=float(np.median(r)), ephys_logdiff_p99=float(np.percentile(r, 99)),
                                  ephys_logdiff_max=float(r.max()), R_absdiff_max=float(dR.max()), R_absdiff_p99=float(np.percentile(dR, 99)),
                                  trace_len_max=int(max(len(t) for t in bt['traces'])))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(C.RESULTS, 'check_trace.json'), 'w'), indent=1)
