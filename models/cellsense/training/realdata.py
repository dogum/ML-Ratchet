"""The 124 Severson et al. (2019) cells, assembled the way the authors' own loading notebook does it: five batch-1 cells
continued in batch 2 are joined back up, and cells the authors excluded are dropped (batch 1: never reached 80%;
batch 3: noisy channels). Cycle life is recomputed here as the first cycle whose discharge capacity (median-smoothed over
5 cycles) falls below 0.88 Ah, 80% of nominal, and checked against the published value."""
import _paths as _P  # portable paths (see _paths.py)
import numpy as np
D = _P.WORK + '/data/'
DROP = {'b1c8', 'b1c10', 'b1c12', 'b1c13', 'b1c22', 'b3c37', 'b3c2', 'b3c23', 'b3c32', 'b3c42', 'b3c43'}
CONT = {'b1c0': ('b2c7', 662), 'b1c1': ('b2c8', 981), 'b1c2': ('b2c9', 1060), 'b1c3': ('b2c15', 208), 'b1c4': ('b2c16', 482)}
SUM = ('IR', 'QCharge', 'QDischarge', 'Tavg', 'Tmax', 'Tmin', 'chargetime', 'cycle')

def _cell(z, k):
    return dict(summary={a: z[f'{k}/summary/{a}'].astype(np.float64) for a in SUM}, cyc_idx=z[f'{k}/cyc_idx'].astype(int), Qdlin=z[f'{k}/Qdlin'], Tdlin=z[f'{k}/Tdlin'],
                cycle_life_pub=float(z[f'{k}/meta'][0]), policy=str(z[f'{k}/policy']), barcode=str(z[f'{k}/barcode']), channel=str(z[f'{k}/channel']))

def load():
    Z = {b: np.load(D + f'{b}.npz') for b in ('b1', 'b2', 'b3')}; V = Z['b1']['Vdlin'].astype(np.float64); cells = {}
    for b, z in Z.items():
        for k in sorted({x.split('/')[0] for x in z.files if '/' in x}, key=lambda s: int(s.split('c')[1])):
            if k in DROP or k in {v[0] for v in CONT.values()}: continue
            c = _cell(z, k)
            if k in CONT:
                k2, add = CONT[k]; c2 = _cell(Z['b2'], k2); n1 = len(c['summary']['cycle'])
                c['summary'] = {a: np.concatenate([c['summary'][a], c2['summary'][a] + (n1 if a == 'cycle' else 0)]) for a in SUM}
                c['cyc_idx'] = np.concatenate([c['cyc_idx'], c2['cyc_idx'] + n1]); c['Qdlin'] = np.concatenate([c['Qdlin'], c2['Qdlin']]); c['Tdlin'] = np.concatenate([c['Tdlin'], c2['Tdlin']])
                c['cycle_life_pub'] += add
            c['batch'] = b; cells[k] = c
    for k, c in cells.items():
        qd = c['summary']['QDischarge']; qd = np.where((qd > 0.5) & (qd < 1.2), qd, np.nan)
        sm = np.array([np.nanmedian(qd[max(0, i - 2):i + 3]) for i in range(len(qd))]); below = np.where(sm < 0.88)[0]
        c['cycle_life'] = float(below[0] + 1) if len(below) else np.nan; c['qd_smooth'] = sm
        # first-crossing as published (cycle index + 1); curve rows: cycle number = cyc_idx + 1
    return cells, V

if __name__ == '__main__':
    cells, V = load(); print(len(cells), 'cells')
    d = np.array([[c['cycle_life'], c['cycle_life_pub']] for c in cells.values()]); ok = np.isfinite(d).all(1)
    print('recomputed vs published cycle life: max |diff|', np.nanmax(np.abs(d[ok, 0] - d[ok, 1])), 'median', np.median(np.abs(d[ok, 0] - d[ok, 1])), 'n missing pub', (~np.isfinite(d[:, 1])).sum(), 'missing ours', (~np.isfinite(d[:, 0])).sum())
    print('cycle life range', np.nanmin(d[:, 0]), np.nanmax(d[:, 0]))
    for b in ('b1', 'b2', 'b3'): print(b, sum(c['batch'] == b for c in cells.values()))
    k = 'b1c0'; c = cells[k]; print(k, 'summary cycle head', c['summary']['cycle'][:3], 'QD head', np.round(c['summary']['QDischarge'][:3], 4), 'IR head', c['summary']['IR'][:3], 'curves', c['Qdlin'].shape, 'life', c['cycle_life'])
    bad = [(k, c['cycle_life'], c['cycle_life_pub']) for k, c in cells.items() if not (np.isfinite(c['cycle_life']) and np.isfinite(c['cycle_life_pub']) and abs(c['cycle_life'] - c['cycle_life_pub']) <= 3)]
    print('cells differing by >3 cycles:', bad)
