"""Network twin test on the 18 dev stars: the incumbent reads each real dev star and its 4 simulated twins (v2 physics,
whole light curve, the real star's own gaps). If the simulator is faithful, the real error should look like a draw
from the twins' errors. Writes twin_net.json."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np, torch
sys.path.insert(0, _P.SRC); import twin, simbank as SB, calib as CB, snet as SN
torch.set_num_threads(1)
ck = sys.argv[1]; net = SN.load_net(_P.WORK + '/ckpt/' + ck)
sp = json.load(open(twin.D + 'split.json')); pool = SB.load_pool('noise_train'); rows = []
for kic in sp['dev']:
    Y, real, tw = twin.twins(kic, pool, calib=CB.make('v2'))
    X = [real] + tw; P = SN.predict(net, np.stack([x[0] for x in X]), np.stack([x[1] for x in X]), np.stack([x[2] for x in X]))
    e1 = P['ln_numax'] - np.log(Y['numax']); e2 = P['ln_dnu'] - np.log(Y['dnu'])
    rows.append(dict(kic=kic, numax=Y['numax'], real=[float(e1[0]), float(e2[0])], twins=[e1[1:].tolist(), e2[1:].tolist()], p_det=P['p_det'].tolist()))
    print(kic, '%.1f' % Y['numax'], 'real err numax %+.3f dnu %+.3f | twins numax %s dnu %s' % (e1[0], e2[0], np.round(e1[1:], 3), np.round(e2[1:], 3)), flush=True)
r = np.array([x['real'] for x in rows]); t1 = np.array([x['twins'][0] for x in rows]); t2 = np.array([x['twins'][1] for x in rows])
S = dict(ckpt=ck, real_med_abs=[float(np.median(np.abs(r[:, 0]))), float(np.median(np.abs(r[:, 1])))], twin_med_abs=[float(np.median(np.abs(t1))), float(np.median(np.abs(t2)))],
         real_bias=[float(np.median(r[:, 0])), float(np.median(r[:, 1]))], twin_bias=[float(np.median(t1)), float(np.median(t2))], rows=rows)
json.dump(S, open(_P.WORK + '/twin_net.json', 'w'), indent=1); print({k: v for k, v in S.items() if k != 'rows'})
