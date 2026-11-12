"""Real beats through the same front end: MIT-BIH Arrhythmia (MLII; paced records 102, 104, 107, 217 excluded per AAMI)
and the MIT-BIH Supraventricular Arrhythmia Database (first channel, 128 Hz). Beat positions = reference annotations."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, wfdb
sys.path.insert(0, _P.SRC); import common as C, prep
D = _P.WORK + '/data/'

def one(db, rec):
    r = wfdb.rdrecord(D + f'{db}/{rec}'); a = wfdb.rdann(D + f'{db}/{rec}', 'atr')
    ch = r.sig_name.index('MLII') if 'MLII' in r.sig_name else 0; pos, cls, sym = C.beats(np.asarray(a.sample), np.asarray(a.symbol))
    f = prep.features(r.p_signal[:, ch], r.fs, pos); return f, cls, pos, r.fs, r.sig_name[ch]

def build(db, recs, out, split_of):
    R = {k: [] for k in ('w', 'tpl', 'rr', 'y', 'rec', 'pos')}; meta = []
    for rec in recs:
        f, cls, pos, fs, lead = one(db, rec); n = len(cls)
        for k in ('w', 'tpl', 'rr'): R[k].append(f[k])
        R['y'].append(cls); R['rec'].append(np.full(n, int(rec))); R['pos'].append(pos); meta.append(dict(rec=int(rec), fs=fs, lead=lead, n=n, split=split_of(int(rec)), counts=np.bincount(cls, minlength=5).tolist()))
        print(db, rec, lead, n, np.bincount(cls, minlength=5).tolist(), flush=True)
    np.savez(out, **{k: np.concatenate(v).astype(np.float16 if k in ('w', 'tpl') else (np.float32 if k == 'rr' else np.int32)) for k, v in R.items()})
    json.dump(meta, open(out.replace('.npz', '_meta.json'), 'w'), indent=0)

if __name__ == '__main__':
    dev = C.dev_records(); sp = lambda r: 'dev' if r in dev else ('ds1rest' if r in C.DS1 else 'ds2')
    build('mitdb', C.DS1 + C.DS2, D + 'mitdb_beats.npz', sp)
    if len(sys.argv) > 1 and sys.argv[1] == 'svdb':
        recs = [l.strip() for l in open(D + 'svdb/RECORDS') if l.strip()] if False else sorted({f.split('.')[0] for f in __import__('os').listdir(D + 'svdb') if f.endswith('.atr')})
        build('svdb', recs, D + 'svdb_beats.npz', lambda r: 'svdb')
