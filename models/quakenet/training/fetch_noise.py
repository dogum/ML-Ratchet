import _paths as _P  # portable paths (see _paths.py)
import sys, time, concurrent.futures as cf
sys.path.insert(0, _P.SRC); import fetch as F
J = [j for j in F.jobs() if j[0] == 'no'][::-1]; print(len(J), 'noise jobs', flush=True); t = time.time(); res = {}
with cf.ThreadPoolExecutor(3) as ex:
    for i, r in enumerate(ex.map(F.get, J)):
        res[r.split(' ')[0]] = res.get(r.split(' ')[0], 0) + 1
        if i % 200 == 0: print(i, res, '%.0fs' % (time.time() - t), flush=True)
print('done', res, flush=True)
