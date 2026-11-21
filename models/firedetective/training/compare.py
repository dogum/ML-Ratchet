"""Sim-vs-real feature check (dev only): quantiles of each sequence/static feature in the bank vs the dev records,
and the per-step relation between detections and area growth. Output out/compare_<bank>.json and a figure."""
from common import *
import glob
import feats
from real import load_records

bank = sys.argv[1] if len(sys.argv) > 1 else 'val_v1'
S, St = [], []
for fn in sorted(glob.glob(f'{BANK}/{bank}/shard_*.npz')):
    z = np.load(fn)
    if len(z['lab']):
        S.append(z['seq'].astype(np.float32)); St.append(z['static'])
S = np.concatenate(S); St = np.concatenate(St)
recs = [r for r in load_records(f'{REAL}/dev_records_package.pkl') if r is not None]
R = np.stack([r['seq'] for r in recs]); Rt = np.stack([r['static'] for r in recs])
q = [10, 50, 90]
out = dict(bank=bank, n_sim=len(S), n_dev=len(R), seq={}, static={})
ms, mr = S[..., 0] > 0, R[..., 0] > 0
for j, nm in enumerate(feats.SEQ_NAMES):
    a, b = S[..., j][ms], R[..., j][mr]
    out['seq'][nm] = dict(sim=np.percentile(a, q).round(3).tolist(), dev=np.percentile(b, q).round(3).tolist())
for j, nm in enumerate(feats.STAT_NAMES):
    out['static'][nm] = dict(sim=np.percentile(St[:, j], q).round(3).tolist(), dev=np.percentile(Rt[:, j], q).round(3).tolist())
# growth steps: log detections vs dlogA
for tag, X, m in (('sim', S, ms), ('dev', R, mr)):
    g = m & (X[..., 3] > 0.01)
    out[f'{tag}_frac_growth_steps'] = float(g.sum() / m.sum())
    out[f'{tag}_corr_logN_dr'] = float(np.corrcoef(X[..., 5][g], X[..., 4][g])[0, 1])
    out[f'{tag}_frp_per_det_median_day'] = float(np.median(X[..., 7][m & (X[..., 1] > 0) & (X[..., 5] > 0)]))
    out[f'{tag}_frp_per_det_median_night'] = float(np.median(X[..., 7][m & (X[..., 1] == 0) & (X[..., 5] > 0)]))
    out[f'{tag}_steps_median'] = float(np.median(m.sum(1)))
jdump(out, f'{ROOT}/out/compare_{bank}.json')
for k, v in out['seq'].items():
    print(f'{k:12s} sim {v["sim"]}  dev {v["dev"]}')
for k, v in out['static'].items():
    print(f'{k:12s} sim {v["sim"]}  dev {v["dev"]}')
print({k: v for k, v in out.items() if k not in ('seq', 'static')})
