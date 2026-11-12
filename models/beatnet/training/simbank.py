"""Simulated training / validation banks: rhythm strips -> the same front end as real records (prep.features).
Each patient gives 2 strips; from each strip up to 8 beats are kept, rare classes preferred. Resumable in chunks."""
import _paths as _P  # portable paths (see _paths.py)
import os, sys, json, time, argparse, numpy as np
sys.path.insert(0, _P.SRC); import sim, strip as S, prep
B = _P.WORK + '/bank/'; os.makedirs(B, exist_ok=True)
CW = np.array([1.0, 6.0, 4.0, 20.0])        # keep-weights per class N S V F

def one_patient(seed, nstrips=2, nbeats=46, per=8):
    rng = np.random.default_rng(seed + 7919); p = sim.Patient(seed); out = []
    for k in range(nstrips):
        P = S.program(rng); V, A = S.events(p, P, rng, nbeats); sig, on, wd = S.synth(p, P, V, A, rng); x, ann = S.record(sig, on, wd, rng)
        lab = np.array([e['label'] for e in V]); cand = np.arange(4, len(V) - 4); w = CW[lab[cand]]; w = w / w.sum()
        keep = np.sort(rng.choice(cand, min(per, len(cand)), replace=False, p=w))
        f = prep.features(x[0], 360, ann, keep)
        out.append(dict(w=f['w'], tpl=f['tpl'], rr=f['rr'], y=lab[keep], qrs=wd[keep].astype(np.float32),
                        info=np.array([[p.bbb != 'none', ['sinus', 'af', 'flutter'].index(P['base']), seed]] * len(keep), np.int32)))
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--name', required=True); ap.add_argument('--seed0', type=int, required=True); ap.add_argument('--patients', type=int, required=True); ap.add_argument('--chunk', type=int, default=250); ap.add_argument('--phys', type=int, default=1)
    a = ap.parse_args(); t0 = time.time(); sim.set_version(a.phys)
    for c0 in range(0, a.patients, a.chunk):
        fn = f'{B}{a.name}_{c0 // a.chunk:03d}.npz'
        if os.path.exists(fn): continue
        R = [r for s in range(a.seed0 + c0, a.seed0 + min(c0 + a.chunk, a.patients)) for r in one_patient(s)]
        cat = lambda k: np.concatenate([r[k] for r in R]); np.savez(fn, w=cat('w').astype(np.float16), tpl=cat('tpl').astype(np.float16), rr=cat('rr'), y=cat('y'), qrs=cat('qrs'), info=cat('info'))
        print(json.dumps(dict(chunk=fn, n=len(cat('y')), classes=np.bincount(cat('y'), minlength=4).tolist(), t=round(time.time() - t0))), flush=True)
    print('done')
