"""Replay a bank chunk's random draws (routes, start states, run seeds) without simulating, to re-simulate chosen rows
with a changed vehicle (rung v2-r15). bank.py draws everything from one generator per chunk (default_rng([seed, chunk]))
and the simulator from a per-row seed, so the replay gives each row's exact trip and start state.
find_seed(): the seed of a bank whose seed was not logged (train2b, train2c), found by matching the first rows."""
import _paths as _P
import sys, json, glob
sys.path.insert(0, _P.SRC)
from common import *
import numpy as np
from roadsim.vehicle import fill_spec
import bank as BK, features as F


def replay_vehicle(spec, K, rng):
    """-> list of K draws (obs, tru, src, dyno, name, temps, cold, soc, disp, seed) consuming rng as bank.vehicle_examples."""
    full = fill_spec(spec); cls = spec['class']
    out = []
    for k in range(K):
        tr, src, dyno, name = BK.draw_route(rng, cls)
        obs, tru = BK.observed_and_true(tr, rng, dyno)
        temps, cold, soc, disp = BK.init_state(rng, spec, full, tru, src, dyno, name)
        seed = int(rng.integers(1 << 30))
        out.append(dict(obs=obs, tru=tru, src=src, dyno=dyno, name=name, temps=temps, cold=cold, soc=soc, disp=disp, seed=seed,
                        dur=float(len(obs['v'])), dist=float(F.prep(obs)['v'].sum())))
    return out


def replay_chunk(fleet, K, seed, ci, upto=None):
    """-> {vid: [draws]} for the chunk's vehicles (first `upto` vehicles if given)."""
    V = json.load(open(f'{EN}/data/fleet_{fleet}/chunk_{ci:03d}.json'))
    rng = np.random.default_rng([seed, ci])
    res = {}
    for vi, x in enumerate(V[:upto] if upto else V):
        res[ci * 100 + vi] = replay_vehicle(x['spec'], K, rng)
    return res


def find_seed(bank, fleet, K, cands):
    B = dict(np.load(f'{EN}/data/bank_{bank}/chunk_000.npz'))
    d0 = B['dur'][:K]; s0 = B['dist'][:K]
    for s in cands:
        r = replay_chunk(fleet, K, s, 0, upto=1)[0]
        if all(abs(a['dur'] - b) < 0.5 for a, b in zip(r, d0)) and all(abs(a['dist'] - b) < 1e-2 * max(b, 1) for a, b in zip(r, s0)):
            return s
    return None


if __name__ == '__main__':
    bank, fleet, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
    lo, hi = int(sys.argv[4]), int(sys.argv[5])
    print(bank, find_seed(bank, fleet, K, list(range(lo, hi))), flush=True)
