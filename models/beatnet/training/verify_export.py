"""numpy release (BatchNorm folded) vs torch checkpoint on 2000 real DS2 beats: max |difference| of the raw outputs."""
import _paths as _P  # portable paths (see _paths.py)
import sys, json, numpy as np
sys.path.insert(0, _P.SRC); sys.path.insert(0, _P.WORK + '/release'); import bnet as BN, train as TR
from beatnet import BeatNet
ck, npz = sys.argv[1], sys.argv[2]; net = BN.load_net(ck); nb = BeatNet(npz)
db, idx = BN.split_idx('ds2'); Z = BN.real_bank(db); idx = np.random.default_rng(0).choice(idx, 2000, replace=False)
ot = BN.predict(net, Z, idx); x, r = TR.inputs(Z, idx, net.cin); on = np.concatenate([nb.forward(x[i:i + 250], r[i:i + 250]) for i in range(0, len(x), 250)])
d = float(np.abs(ot - on).max()); agree = float((ot[:, :4].argmax(1) == on[:, :4].argmax(1)).mean())
res = dict(ckpt=ck, n=len(idx), max_abs_diff=d, label_agreement=agree); print(res); json.dump(res, open(_P.WORK + '/verify_export.json', 'w'), indent=1)
