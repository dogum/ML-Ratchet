"""Run free-burn simulations for a set of ignitions (resumable chunks).
usage: simbank.py <kind: twin|train> <reps> <physics version> [max_ign]
-> data/bank/<ver>/<kind>_<chunk>.npz"""
import os, sys, time, glob
from common import *
import sim

KIND, REPS, VER = sys.argv[1], int(sys.argv[2]), sys.argv[3]
MAXI = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
CH = 100
OUT = f'{DATA}/bank/{VER}'
os.makedirs(OUT, exist_ok=True)


def main():
    I = pd.read_parquet(f'{DATA}/ignitions.parquet')
    I = I[I.kind == KIND].head(MAXI)
    jobs = [(i, k) for i in I.ign.values for k in range(REPS)]
    I = I.set_index('ign', drop=False)
    nch = (len(jobs) + CH - 1) // CH
    t0 = time.time(); done = 0
    for ch in range(nch):
        fn = f'{OUT}/{KIND}_{ch:04d}.npz'
        if os.path.exists(fn):
            continue
        rows = []
        for (i, k) in jobs[ch * CH:(ch + 1) * CH]:
            ign = I.loc[i]
            if not os.path.exists(f'{DATA}/win/{ign.win}.npz'):
                continue
            rs = np.random.default_rng([SEED, int(i), k])
            try:
                r = sim.run_one(ign, rs, physics=VER)
            except Exception as e:
                print('FAIL', i, k, repr(e)[:300], flush=True)
                continue
            r['ign'] = int(i); r['rep'] = k
            rows.append(r)
        if not rows:
            continue
        keys = [k for k in rows[0] if k not in ('A_pt', 'A_grid')]
        np.savez_compressed(fn + '.tmp.npz', **{k: np.array([r[k] for r in rows]) for k in keys})
        os.replace(fn + '.tmp.npz', fn)
        done += len(rows)
        print(ch, nch, done, f'{(time.time() - t0) / done:.3f}s/sim', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
