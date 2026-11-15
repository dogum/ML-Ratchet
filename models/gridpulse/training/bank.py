"""Simulation bank: python3 bank.py <name> <prior_version> <n_total> <seed> [gb_frac]
Chunks of 200 states (alternating systems by gb_frac), resumable. Each chunk: dev (n, 681) float32 = simulated
frequency deviation [Hz] at 0.1 s from the loss (t = 0 .. 68 s), plus labels."""
from common import *
import gpsim, time

CH = 200


def build(name, version, n, seed, gb_frac=0.6):
    out = f'{DATA}/bank_{name}'; os.makedirs(out, exist_ok=True)
    nch = int(np.ceil(n / CH)); rng = np.random.default_rng(seed)
    systems = np.where(rng.random(nch) < gb_frac, 'GB', 'ERCO')
    for c in range(nch):
        fn = f'{out}/c{c:04d}.npz'
        if os.path.exists(fn):
            continue
        t = time.time()
        S, E, L = gpsim.sample(CH, seed * 100000 + c, systems[c], version)
        ev = gpsim.run(S, E)
        f0 = L.f0.values[:, None]
        dev = np.array([e['f'] for e in ev]) - f0
        tt = ev[0]['t']
        L['nadir_true'] = dev.min(1); L['t_nadir_true'] = tt[dev.argmin(1)]
        L['pointB_true'] = dev[:, (tt >= 20) & (tt <= 52)].mean(1)
        L['settle_true'] = dev[:, (tt >= 55) & (tt <= 65)].mean(1)
        L['ufls'] = [len(e['ufls']) for e in ev]; L['lost_total_MW'] = [sum(x[1] for x in e) for e in E]
        L['chunk'] = c
        np.savez_compressed(fn, dev=dev.astype(np.float32), labels=L.to_records(index=False))
        print(c, systems[c], f'{time.time() - t:.1f}s', flush=True)


def load(name):
    fs = sorted(f for f in os.listdir(f'{DATA}/bank_{name}') if f.endswith('.npz'))
    D, Ls = [], []
    for f in fs:
        z = np.load(f'{DATA}/bank_{name}/{f}', allow_pickle=True)
        D.append(z['dev']); Ls.append(pd.DataFrame(z['labels']))
    return np.concatenate(D), pd.concat(Ls, ignore_index=True)


if __name__ == '__main__':
    a = sys.argv
    build(a[1], a[2], int(a[3]), int(a[4]), float(a[5]) if len(a) > 5 else 0.6)
