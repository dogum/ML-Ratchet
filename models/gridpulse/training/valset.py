"""Frozen simulated validation set: python3 valset.py <name> <bank> <mode> [tau]
Fixed seed 99, held-out noise days (2022-01 days 21+), every event at N = 5, 10 and 60 s."""
from common import *
import bank as BK, dataset as DS


def build(name, bank, mode='mean', tau=0.0, seed=99):
    dev, L = BK.load(bank)
    keep = DS.selected(dev, L); dev, L = dev[keep], L[keep].reset_index(drop=True)
    noise = DS.noise_split('val'); rng = np.random.default_rng(seed)
    if mode == 'mix':
        modes = rng.choice(['sample', 'mean', 'lag'], size=len(L))
        parts = {}
        for m in ('sample', 'mean', 'lag'):
            s = modes == m
            parts[m] = DS.make(dev[s], L[s].reset_index(drop=True), rng, noise, 'mean' if m == 'lag' else m,
                               0.5 if m == 'lag' else 0.0, Nfix=60)
        order = np.concatenate([np.where(modes == m)[0] for m in ('sample', 'mean', 'lag')])
        inv = np.argsort(order)
        w = np.concatenate([parts[m][0]['w'] for m in ('sample', 'mean', 'lag')])[inv]
        y = np.concatenate([parts[m][2] for m in ('sample', 'mean', 'lag')])[inv]
    else:
        meas, _, y = DS.make(dev, L, rng, noise, mode, tau, Nfix=60); w = meas['w']
    np.savez_compressed(f'{DATA}/val_{name}.npz', w=w, y=y, system=L.system.values.astype(str), f0=L.f0.values,
                        mode=mode, tau=tau, bank=bank, inertia=L.inertia_MWs.values, loss=L.loss_MW.values,
                        demand=L.demand_MW.values)
    print(name, len(L), L.system.value_counts().to_dict())


if __name__ == '__main__':
    a = sys.argv
    build(a[1], a[2], a[3] if len(a) > 3 else 'mean', float(a[4]) if len(a) > 4 else 0.0)
