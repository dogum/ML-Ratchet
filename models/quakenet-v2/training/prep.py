"""Turn banks (simulated) or real_nodes (real) into graph samples: which stations, the physics anchor (v1.2 combined),
and the truth. Simulated events give several samples each (all stations, 2 stations, and for training a random subset
in between); real events give the exam's station counts (first 2, 3, 5 and all stations in the frozen order)."""
import os, sys, glob, json, numpy as np
from _paths import H; sys.path.insert(0, H + 'src')
import sim2 as S2, locate2 as LC, tt as TT
MU = np.array([6.08668, 59.2584, 1.6155]); SD = None

def enc_norm():
    import torch; ck = torch.load(H + 'ckpt/q7_zrt.pt', weights_only=False); return np.array(ck['mu_y']), np.array(ck['sd_y'])

def load_bank(pattern):
    F = sorted(glob.glob(H + 'bank/' + pattern)); Z = [np.load(f) for f in F]; out = {}; ev_off = 0; evs = []
    for z in Z:
        for k in z.files:
            if k == 'events': continue
            v = z[k] + (ev_off if k == 'ev' else 0); out.setdefault(k, []).append(v)
        evs.append(z['events']); ev_off += len(z['events'])
    out = {k: np.concatenate(v) for k, v in out.items()}; out['events'] = np.concatenate(evs); return out

def nodes_from_bank(B):
    mu, sd = enc_norm(); R = S2.decode(B['out8'], mu, sd); n = dict(slat=S2.SLAT[B['sta']], slon=S2.SLON[B['sta']], ls=B['ls'], peak=B['peak'], emb=B['emb'], pz=B['pz'],
                                                                   tP=B['tP'], pickerr=B['pickerr'], res=B['res'], serr=B['serr'], tS=B['tS'], ev=B['ev'])
    n.update({'r_' + k: v for k, v in R.items()}); return n

def Rof(n, idx): return {k[2:]: n[k][idx] for k in n if k.startswith('r_')}

def sim_samples(n, events, seed, train=True):
    rng = np.random.default_rng(seed); by = {}
    for i, e in enumerate(n['ev']): by.setdefault(int(e), []).append(i)
    S = []
    for e, idx in sorted(by.items()):
        idx = np.array(idx); subs = [idx]
        if len(idx) > 2: subs.append(idx[:2])
        if train and len(idx) > 3: subs.append(rng.permutation(idx)[:rng.integers(3, len(idx))])
        for sub in subs:
            A = LC.combine_v12(n['slat'][sub], n['slon'][sub], Rof(n, sub)); la, lo, z, mw = events[e]
            S.append(dict(idx=sub, ev=e, A=A, truth=(la, lo, np.log10(z), mw, 0.0)))
    return S

def pack(S):
    K = np.array([len(s['idx']) for s in S]); I = -np.ones((len(S), 10), int)
    for j, s in enumerate(S): I[j, :len(s['idx'])] = s['idx']
    A = np.array([[s['A'][k] for k in ('lat', 'lon', 'logz', 'mw', 'logz_sig', 'mw_sig')] for s in S]); C = np.array([s['A']['cov'] for s in S])
    return dict(idx=I, K=K, ev=np.array([s['ev'] for s in S]), anchor=A, anchor_cov=C, truth=np.array([s['truth'] for s in S], float))

def real_samples():
    """dev: first 2 and all stations; exam: first 2 (all exam events with 2+), and first 3, 5 and all for the 400-event subset"""
    z = np.load(H + 'data/real_nodes.npz'); rows = json.load(open(H + 'data/real_nodes.json')); SP = json.load(open(H + 'splits.json')); mu, sd = enc_norm()
    R = S2.decode(z['out8'], mu, sd); n = dict(slat=np.array([r['slat'] for r in rows]), slon=np.array([r['slon'] for r in rows]), ls=z['ls'], peak=z['peak'], emb=z['emb'],
                                              pz=z['pz'] if 'pz' in z.files else np.zeros((len(rows), 240), np.float16),
                                              tpick=np.array([r['tpick'] for r in rows]), tspick=np.array([np.nan if r['tspick'] is None else r['tspick'] for r in rows]))
    n.update({'r_' + k: v for k, v in R.items()}); by = {}
    good = np.isfinite(z['out8']).all(1) & np.isfinite(z['emb'].astype(np.float32)).all(1)     # drop records the v1.2 network cannot read (3 exam records with empty windows)
    for i, r in enumerate(rows):
        if good[i]: by.setdefault(r['id'], []).append(i)
    out = {}
    for split in ('dev', 'exam'):
        S = []; meta = []
        for k in SP[split]:
            idx = np.array(sorted(by.get(k, []), key=lambda i: rows[i]['order']))
            if len(idx) < 2: continue
            e = SP['events'][k]; multi = split == 'dev' or k in SP['exam_multi']
            for tag, N in (('2', 2), ('3', 3), ('5', 5), ('all', len(idx))):
                if N > len(idx) or (tag != '2' and not multi) or (tag in ('3', '5') and split == 'dev'): continue
                if tag == 'all' and N == 2 and split == 'exam': continue
                sub = idx[:N]; A = LC.combine_v12(n['slat'][sub], n['slon'][sub], Rof(n, sub))
                S.append(dict(idx=sub, ev=len(meta), A=A, truth=(e['lat'], e['lon'], np.log10(max(e['depth'], 1.0)), e['mw'], 0.0))); meta.append(dict(id=k, tag=tag, K=int(N), depth=e['depth']))
        P = pack(S); P['tag'] = np.array([m['tag'] for m in meta]); P['event_id'] = np.array([m['id'] for m in meta])
        np.savez(H + f'data/{split}_samples.npz', **P, **{'n_' + k: v for k, v in n.items()}); out[split] = len(S); print(split, len(S), 'samples', {t: int((P['tag'] == t).sum()) for t in ('2', '3', '5', 'all')})
    return out

if __name__ == '__main__':
    what = sys.argv[1]
    if what in ('val1', 'tr1'):
        B = load_bank(what + '_*.npz'); n = nodes_from_bank(B); S = sim_samples(n, B['events'], 1 if what == 'val1' else 2, train=(what != 'val1')); P = pack(S)
        if what == 'val1': P['res_fixed'] = np.random.default_rng(77).normal(0, 1, len(n['ev']))     # the frozen set's residual draw (unit; scaled by the physics version's spread)
        np.savez(H + f'bank/{what}_samples.npz', **P, **{'n_' + k: v for k, v in n.items()}, events=B['events']); print(what, len(S), 'samples', len(n['ev']), 'nodes')
    if what == 'real': real_samples()
