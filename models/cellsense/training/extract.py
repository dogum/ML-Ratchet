"""Pull what CellSense needs out of one Severson et al. (2019) batch file (MATLAB v7.3 / HDF5): per-cell summary (capacity,
internal resistance, temperatures, charge time) and, per cycle, the 4C discharge curve Q(V) and T(V) on the fixed
1000-point voltage grid (3.6 -> 2.0 V). Cycles 1-200 are kept every cycle (early prediction), later ones every 5th."""
import sys, h5py, numpy as np

def text(f, ref):
    return f[ref][()].tobytes()[::2].decode()

def main(path, tag, out):
    f = h5py.File(path, 'r'); b = f['batch']; n = b['summary'].shape[0]; cells = {}
    vgrid = np.array(f[b['Vdlin'][0, 0]][()]).ravel() if 'Vdlin' in b else None
    for i in range(n):
        s = f[b['summary'][i, 0]]; S = {k: np.array(s[k][()]).ravel() for k in s.keys()}
        c = f[b['cycles'][i, 0]]; nc = c['Qdlin'].shape[0]
        keep = [j for j in range(nc) if j < 200 or j % 5 == 0]
        Q = np.full((len(keep), 1000), np.nan, np.float32); T = np.full((len(keep), 1000), np.nan, np.float32)
        for a, j in enumerate(keep):
            q = np.array(f[c['Qdlin'][j, 0]][()]).ravel(); t = np.array(f[c['Tdlin'][j, 0]][()]).ravel()
            if q.size == 1000: Q[a] = q
            if t.size == 1000: T[a] = t
        cells[f'{tag}c{i}'] = dict(summary=S, cyc_idx=np.array(keep), Qdlin=Q, Tdlin=T, cycle_life=float(np.array(f[b['cycle_life'][i, 0]][()]).ravel()[0]),
                                    policy=text(f, b['policy_readable'][i, 0]), barcode=text(f, b['barcode'][i, 0]), channel=text(f, b['channel_id'][i, 0]))
        print(tag, i, nc, cells[f'{tag}c{i}']['cycle_life'], cells[f'{tag}c{i}']['policy'], flush=True)
    flat = {}
    for k, v in cells.items():
        for a, x in v['summary'].items(): flat[f'{k}/summary/{a}'] = x
        for a in ('cyc_idx', 'Qdlin', 'Tdlin'): flat[f'{k}/{a}'] = v[a]
        flat[f'{k}/meta'] = np.array([v['cycle_life'], 0.0]); flat[f'{k}/policy'] = np.array(v['policy']); flat[f'{k}/barcode'] = np.array(v['barcode']); flat[f'{k}/channel'] = np.array(v['channel'])
    if vgrid is not None: flat['Vdlin'] = vgrid
    np.savez_compressed(out, **flat)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
