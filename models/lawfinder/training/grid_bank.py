"""Simulated GB frequency-event and inertia banks from GridSim (LawFinder, law 3).

nadir mode: GridPulse's GB state sampler (gpsim, copied unchanged from models/gridpulse/training) rescales a random
GridSim fleet to a GB operating point (demand, inertia and IBR share drawn jointly from NESO half-hours of the
TRAIN-INPUT years 2018/2020/2022, or wide ranges) and adds GB frequency services; GridSim's aggregated swing equation
then runs the trip for 68 s at 0.1 s. Measured like the real catalogue: 1 s means, kept only if the largest 10 s fall
is <= -0.15 Hz (GridData's detection rule); nadir = lowest 1 s value.
    g1 = gpsim prior p0 (generic GB response)      g2 = gpsim p1 (GridPulse's dev-calibrated GB response = its release)
inertia mode: GridSim's own unit commitment (generate_states, GB fleets): inertia from the synchronous units it commits
for a demand and an IBR share. Version i1 = GridSim as shipped.

usage: python3 grid_bank.py <nadir|inertia> <ver> <train|val> <n> [seed]   (resumable shards)
"""
import os, sys, math, time, json
import numpy as np, pandas as pd

from _paths import ROOT  # the work folder (training/_paths.py)
SHARD = 500
PRIOR = {'g1': 'p0', 'g2': 'p1'}


def measure(t, f, f0):
    """1 s means of the 0.1 s trace, the largest 10 s fall and the 1 s nadir."""
    n = int(len(t) // 10)
    f1 = np.asarray(f[:n * 10], float).reshape(n, 10).mean(1)
    fall10 = float(np.min(f1[10:] - f1[:-10])) if n > 10 else 0.0
    return f0 - float(np.min(f1)), fall10


def nadir_shard(ver, seed, n):
    from _paths import import_gpsim
    gpsim = import_gpsim()
    st, ev, lab = gpsim.sample(n, seed, 'GB', PRIOR[ver])
    out = gpsim.run(st, ev)
    rows = []
    for i, o in enumerate(out):
        dfn, fall10 = measure(o['t'], o['f'], o['f0'])
        L = lab.iloc[i]
        rows.append({'loss_MW': float(L.loss_MW), 'inertia_GVAs': float(L.inertia_MWs) / 1e3,
                     'demand_MW': float(L.demand_MW), 'ibr_share': float(L.ibr_share), 'df_nadir_Hz': dfn,
                     'df_nadir_true_Hz': float(o['f0'] - o['nadir_Hz']), 'fall10_Hz': fall10,
                     'detected': bool(fall10 <= -0.15), 'headroom_MW': float(L.headroom_MW),
                     'dc_MW': float(L.dc_MW), 'ffr_MW': float(L.ffr_MW), 'damping': float(L.D),
                     'loss_ramp': int(L.loss_ramp), 'shed_MW': float(np.max(o['shed_MW'])) if np.size(o['shed_MW']) else 0.0})
    return rows


def inertia_shard(ver, seed, n):
    import gridsim as gs
    st, ls, lab = gs.generate_states(n, seed=seed, systems=('GB',), aggregate=True)
    return [{'demand_MW': float(r.demand_MW), 'ibr_share': float(r.ibr_share), 'inertia_GVAs': float(r.inertia_MWs) / 1e3,
             'n_sync_units': int(r.n_sync_units), 'largest_unit_MW': float(r.largest_unit_MW)} for r in lab.itertuples()]


def build(mode, ver, split, n, seed=1):
    out = f'{ROOT}/banks/grid_{mode}_{ver}'
    os.makedirs(out, exist_ok=True)
    nsh = int(math.ceil(n / SHARD))
    for k in range(nsh):
        fn = f'{out}/{split}_{k:03d}.parquet'
        if os.path.exists(fn):
            continue
        t0 = time.time()
        s = int(np.random.SeedSequence([seed, 0 if split == 'train' else 1, k, 7]).generate_state(1)[0] % (2**31))
        rows = (nadir_shard if mode == 'nadir' else inertia_shard)(ver, s, SHARD)
        pd.DataFrame(rows).to_parquet(fn + '.tmp'); os.replace(fn + '.tmp', fn)
        print(f'{mode} {ver} {split} shard {k+1}/{nsh} {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 1)
