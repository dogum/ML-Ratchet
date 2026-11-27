"""Run free-burn simulations for a set of ignitions (resumable chunks).
usage: simbank.py <kind: twin|train|trainsb> <reps> <physics version> [max_ign] [bank tag]
-> data/bank/<bank tag, default the physics version>/<kind>_<chunk>.npz
27 Nov 2026: the bank tag lets the full v1p1 train bank be rebuilt next to data/bank/v1p1/, whose train chunks hold
only the frozen-set ignitions of the 22 Nov 2026 p2 re-score (a resumed run there would skip their other ignitions)."""
import os, sys, time, glob
from common import *
import sim

KIND, REPS, VER = sys.argv[1], int(sys.argv[2]), sys.argv[3]
MAXI = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
CH = 100
OUT = f'{DATA}/bank/{sys.argv[5] if len(sys.argv) > 5 else VER}'
os.makedirs(OUT, exist_ok=True)


def main():
    I = pd.read_parquet(f'{DATA}/ignitions.parquet')
    I = I[I.kind == KIND].head(MAXI)
    jobs = [(i, k) for i in I.ign.values for k in range(REPS)]
    I = I.set_index('ign', drop=False)
    nch = (len(jobs) + CH - 1) // CH
    t0 = time.time(); done = 0
    part, nparts = (int(v) for v in os.environ.get('SIMBANK_PART', '0/1').split('/'))   # parallel parts (27 Nov 2026)
    for ch in range(nch):
        fn = f'{OUT}/{KIND}_{ch:04d}.npz'
        if ch % nparts != part or os.path.exists(fn):
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
