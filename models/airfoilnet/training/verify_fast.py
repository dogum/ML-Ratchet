"""Check mfoil_fast (numba) against the unmodified mfoil.py on whole solves. Writes work/verify_fast.json."""
import sys, os, time, json, importlib.util
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location('mfoil_orig', os.path.join(HERE, 'mfoil.py'))
mo = importlib.util.module_from_spec(spec); spec.loader.exec_module(mo)
import mfoil_fast  # noqa: E402  (patches the 'mfoil' module)
mf = mfoil_fast.mf


def run(mod, naca, npanel, alphas, Re):
    M = mod.mfoil(naca=naca, npanel=npanel); M.param.verb = 0; M.param.doplot = False
    out = []; t0 = time.time()
    for k, a in enumerate(alphas):
        M.setoper(alpha=a, Re=Re)
        if k > 0: M.oper.initbl = False
        M.solve()
        out.append([M.post.cl, M.post.cd, M.post.cm, float(M.glob.conv)])
    return np.array(out), time.time() - t0, M


res = {}
for naca, Re, alphas in [('2412', 1e6, [2, 5]), ('0012', 2e5, [0, 4]), ('4415', 5e5, [3])]:
    a, ta, Ma = run(mo, naca, 120, alphas, Re)
    b, tb, Mb = run(mf, naca, 120, alphas, Re)
    dU = np.max(np.abs(Ma.glob.U - Mb.glob.U)) if Ma.glob.U.shape == Mb.glob.U.shape else None
    res[f'NACA{naca}_Re{Re:.0e}'] = dict(orig=a.tolist(), fast=b.tolist(), max_abs_diff=float(np.max(np.abs(a - b))),
                                       max_abs_dU=dU, t_orig=round(ta, 2), t_fast=round(tb, 2))
    print(naca, Re, 'maxdiff', np.max(np.abs(a - b)), 'dU', dU, 't', round(ta, 1), round(tb, 1), flush=True)
from _paths import WORK
json.dump(res, open(os.path.join(WORK, 'verify_fast.json'), 'w'), indent=1)
