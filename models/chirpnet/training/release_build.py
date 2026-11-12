import _paths as _P  # portable paths (see _paths.py)
import sys, os, json, shutil, numpy as np, torch
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.WORK + '/release')
from evaluate import load
from realeval import exam
from export import export

ckpt, version = sys.argv[1], sys.argv[2]
net = load(ckpt)
ex = exam(net, slides=range(3, 4))
bg_scores = []
# re-run background collection to keep the raw scores
import realeval, gwphys as G
from evaluate import run
D = realeval.prepare()
for name, d in D.items():
    h, l = d['H1'], d['L1']; tev = d['tev'] - d['t0']
    core = slice(int(2 * G.FS), int(30 * G.FS)); hc, lc = h[core], l[core]; n = len(hc); te = tev - 2
    for k_s in np.arange(3, 25.01, 0.25):
        sh = int(k_s * G.FS); ls_ = np.roll(lc, sh); tl = (te + k_s) % (n / G.FS)
        keep = [s for s in range(0, n - G.WIN, 256) if not (s / G.FS - 0.5 < te < s / G.FS + 2.5) and not (s / G.FS - 0.5 < tl < s / G.FS + 2.5)]
        bg_scores.append(run(net, realeval.windows(hc, ls_, keep))[:, 0])
bg = np.concatenate(bg_scores); live = len(bg) * 256 / G.FS
thr = float(np.ceil((bg.max() + 0.5) * 2) / 2)
# lineage: accepted rungs leading to this checkpoint
R = json.load(open(_P.WORK + '/ratchet.json'))
samples = 0
for name in sorted({r['ckpt'].replace(' +TTA', '') for r in R}):
    ck = torch.load(_P.WORK + '/ckpt/' + name, weights_only=False)
    if 'args' in ck: samples += ck['args']['steps'] * ck['args']['bs']
ck = torch.load(ckpt, weights_only=False)
if ck.get('arch') == 'cascade':     # finder -> chirpnet_weights.npz (carries threshold, background, veto); checker -> chirpnet_coinc_weights.npz
    carr, _ = export(ck['parts'][1], _P.WORK + '/release/chirpnet_coinc_weights.npz', 0.0, [0.0], 1.0, {'role': 'coincidence checker'})
    arr, meta = export(ck['parts'][0], _P.WORK + '/release/chirpnet_weights.npz', thr, bg, live,
                       {'checker': 'chirpnet_coinc_weights.npz', 'veto': float(ck['veto']), 'drop': 30.0})
    params = int(sum(v.size for v in arr.values()) + sum(v.size for v in carr.values()))
else:
    arr, meta = export(ckpt, _P.WORK + '/release/chirpnet_weights.npz', thr, bg, live)
    params = int(sum(v.size for k, v in arr.items()))
macs = None
meta['release'] = {'version': version, 'params': params, 'samples': int(samples), 'bank': 8000, 'checkpoint': os.path.basename(ckpt)}
z = dict(np.load(_P.WORK + '/release/chirpnet_weights.npz'))
z.pop('meta')
z['meta'] = json.dumps(meta)
np.savez_compressed(_P.WORK + '/release/chirpnet_weights.npz', **z)
json.dump({'threshold': thr, 'bg_n': int(len(bg)), 'bg_max': float(bg.max()), 'live_s': live, 'events': {k: {kk: v[kk] for kk in v if not kk.startswith('trace')} for k, v in ex['events'].items()}},
          open(_P.WORK + '/release_summary.json', 'w'), indent=1)
print('threshold', thr, 'background windows', len(bg), 'livetime h %.2f' % (live / 3600), 'max %.2f' % bg.max(), 'samples', samples, 'params', params)
