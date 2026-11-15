"""Dev probe (15 Nov 2026): per-car and per-trip VED-type calls of a checkpoint and of the rule on the 16 dev cars."""
from common import *
import evaluate as E, realbank, baselines as B, features as F
import realdata as rd
ck = sys.argv[1]
net = E.load_net(f'{ROOT}/ckpt/{ck}')
rb = realbank.load('dev'); R = E.real_predict(net, rb); vt = rd.ved_vehicles()
o = rb['offs']
tp = np.array([VEDTYPE[ARCHS[i]] for i in R['Pt']['arch'].argmax(1)])
for ci, c in enumerate(R['cars']):
    js = np.where(rb['vid'] == c)[0]
    raws = [rb['raw'][o[j]:o[j + 1]].T.astype(np.float32) for j in js]; fls = [rb['flags'][j] for j in js]
    rule = VEDTYPE[B.rule_arch(raws, fls)]
    pc = R['Pc']['arch'][ci]
    tt = [tp[k] for k, v in enumerate(R['trip_vid']) if v == c]
    tru = vt.loc[c, 'ved_type']
    # engine-off share while moving (rule's signal), per window
    off = [np.mean(r[6][(r[0] > 5) & (r[10] > .5)] < 0.1) for r, f in zip(raws, fls) if f[2] > .5 and ((r[0] > 5) & (r[10] > .5)).sum() >= 10]
    print(f"{c:8s} {tru:4s} {vt.loc[c,'displacement_L']} net={VEDTYPE[ARCHS[pc.argmax()]]:4s} pHEV={pc[1:4].sum():.2f} rule={rule:4s} "
          f"trips {np.mean(np.array(tt)==tru):.2f} ({len(tt)})  off-share med {np.median(off) if off else float('nan'):.3f}")
