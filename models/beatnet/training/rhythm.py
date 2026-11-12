"""Per-beat rhythm labels for the real banks from the reference rhythm annotations (aux_note '(AFIB', '(AFL', '(N', ...).
Each beat gets the rhythm in force at its position: 1 = atrial fibrillation, 2 = atrial flutter, 0 = anything else.
Writes data/<db>_rhythm.npy aligned with data/<db>_beats.npz. Used only for scoring the AF output, never for training."""
import _paths as _P  # portable paths (see _paths.py)
import sys, numpy as np, wfdb
sys.path.insert(0, _P.SRC); import common as C

def labels(db):
    z = np.load(C.D + f'{db}_beats.npz'); rec, pos = z['rec'], z['pos']; out = np.zeros(len(rec), np.int8)
    for r in np.unique(rec):
        name = f'{r:03d}' if db == 'mitdb' else str(r); a = wfdb.rdann(C.D + f'{db}/{name}', 'atr'); cur = 0; ch_pos, ch_val = [0], [0]
        for s, aux in zip(a.sample, a.aux_note):
            aux = aux.strip('\x00').strip()
            if aux.startswith('('): ch_pos.append(s); ch_val.append(1 if aux.startswith('(AFIB') else 2 if aux.startswith('(AFL') else 0)
        m = rec == r; k = np.searchsorted(ch_pos, pos[m], side='right') - 1; out[m] = np.array(ch_val)[k]
    np.save(C.D + f'{db}_rhythm.npy', out); return out

if __name__ == '__main__':
    for db in ('mitdb', 'svdb'):
        o = labels(db); print(db, np.bincount(o, minlength=3))
